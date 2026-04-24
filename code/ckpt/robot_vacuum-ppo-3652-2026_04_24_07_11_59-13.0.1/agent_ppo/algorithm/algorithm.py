#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Standard PPO algorithm for Robot Vacuum.
清扫大作战 PPO 算法。

Loss composition / 损失组成：
  total_loss = vf_coef * value_loss + policy_loss - beta * entropy_loss + guide_coef * guide_loss
"""

import os
import time

import numpy as np
import torch

from agent_ppo.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.model = model
        self.optimizer = optimizer
        self.parameters = [p for pg in optimizer.param_groups for p in pg["params"]]
        self.device = device
        self.logger = logger
        self.monitor = monitor

        self.clip_param = Config.CLIP_PARAM
        self.vf_coef = Config.VF_COEF
        self.var_beta = Config.BETA_START
        self.label_size = Config.ACTION_NUM

        self.train_step = 0
        self.last_report_time = 0

        # ===== 规则引导损失参数 =====
        self.guide_coef_start = 1.0
        self.guide_coef_end = 0.05
        self.guide_decay_steps = 100000

    def _get_guide_coef(self):
        """规则引导损失系数退火。"""
        progress = min(1.0, self.train_step / max(1, self.guide_decay_steps))
        coef = self.guide_coef_start + progress * (self.guide_coef_end - self.guide_coef_start)
        return float(coef)

    def _as_tensor(self, x, dtype=torch.float32):
        """稳健地把 numpy/list/scalar/tensor 转成 torch.Tensor。"""
        if isinstance(x, torch.Tensor):
            return x.detach().clone().to(dtype=dtype)
        return torch.as_tensor(x, dtype=dtype)

    def _stack_1d_scalar_field(self, list_sample_data, field_name, dtype=torch.float32):
        """
        将每个样本中的标量/shape=(1,) 字段整理成 [B, 1]
        例如: reward, value, advantage, done, act, used_rule, rule_action
        """
        tensors = []
        for s in list_sample_data:
            v = getattr(s, field_name)
            t = self._as_tensor(v, dtype=dtype).reshape(-1)
            if t.numel() == 0:
                raise RuntimeError(f"{field_name} is empty")
            if t.numel() != 1:
                raise RuntimeError(f"{field_name} should have exactly 1 element, got shape {tuple(t.shape)}")
            tensors.append(t[:1])
        return torch.stack(tensors, dim=0)  # [B, 1]

    def _stack_vector_field(self, list_sample_data, field_name, vec_dim, dtype=torch.float32):
        """
        将每个样本中的向量字段整理成 [B, vec_dim]
        例如: prob, legal_action
        """
        tensors = []
        for s in list_sample_data:
            v = getattr(s, field_name)
            t = self._as_tensor(v, dtype=dtype).reshape(-1)
            if t.numel() != vec_dim:
                raise RuntimeError(
                    f"{field_name} shape error before stack: expect {vec_dim}, got {tuple(t.shape)}"
                )
            tensors.append(t)
        return torch.stack(tensors, dim=0)  # [B, vec_dim]

    def _stack_obs_field(self, list_sample_data, field_name, dtype=torch.float32):
        """obs 只要求 batch 维一致，特征维交给模型自己处理。"""
        tensors = []
        first_shape = None
        for s in list_sample_data:
            v = getattr(s, field_name)
            t = self._as_tensor(v, dtype=dtype)
            if first_shape is None:
                first_shape = tuple(t.shape)
            elif tuple(t.shape) != first_shape:
                raise RuntimeError(
                    f"{field_name} inconsistent shape in batch: first={first_shape}, now={tuple(t.shape)}"
                )
            tensors.append(t)
        return torch.stack(tensors, dim=0)

    def _fail_fast_batch_check(
        self,
        obs,
        legal_action,
        act,
        old_prob,
        old_value,
        reward_sum,
        advantage,
        rule_action,
        used_rule,
    ):
        """在进入 CUDA loss 计算前做一次静默检查。
        出错时直接抛异常，并保存坏样本，避免大量日志淹没关键信息。
        """
        problems = []

        # 1. act 必须在 [0, ACTION_NUM-1]
        act_cpu = act.detach().cpu().view(-1)
        invalid_act_mask = (act_cpu < 0) | (act_cpu >= self.label_size)
        if invalid_act_mask.any():
            bad_vals = act_cpu[invalid_act_mask].tolist()[:20]
            problems.append(f"invalid act values: {bad_vals}")

        # 2. used_rule == 1 时，rule_action 才必须合法
        rule_cpu = rule_action.detach().cpu().view(-1)
        used_rule_cpu = used_rule.detach().cpu().view(-1)
        invalid_rule_mask = (used_rule_cpu > 0.5) & ((rule_cpu < 0) | (rule_cpu >= self.label_size))
        if invalid_rule_mask.any():
            bad_vals = rule_cpu[invalid_rule_mask].tolist()[:20]
            problems.append(f"invalid rule_action values when used_rule=1: {bad_vals}")

        # 3. legal_action 不能全 0，shape 必须是 [B, ACTION_NUM]
        legal_cpu = legal_action.detach().cpu()
        if legal_cpu.dim() != 2 or legal_cpu.size(1) != self.label_size:
            problems.append(f"legal_action shape error: {tuple(legal_cpu.shape)}")
        else:
            zero_mask_rows = legal_cpu.sum(dim=1) <= 0
            if zero_mask_rows.any():
                bad_rows = torch.nonzero(zero_mask_rows).view(-1).tolist()[:20]
                problems.append(f"legal_action has all-zero rows at indices: {bad_rows}")

        # 4. old_prob 必须有限，且 shape 正确
        old_prob_cpu = old_prob.detach().cpu()
        if old_prob_cpu.dim() != 2 or old_prob_cpu.size(1) != self.label_size:
            problems.append(f"old_prob shape error: {tuple(old_prob_cpu.shape)}")
        if not torch.isfinite(old_prob_cpu).all():
            problems.append("old_prob contains NaN/Inf")

        # 5. 其他关键张量不能有 NaN/Inf
        for name, tensor in [
            ("obs", obs),
            ("old_value", old_value),
            ("reward_sum", reward_sum),
            ("advantage", advantage),
        ]:
            t = tensor.detach().cpu()
            if not torch.isfinite(t).all():
                problems.append(f"{name} contains NaN/Inf")

        if problems:
            dump_path = f"/tmp/ppo_bad_batch_step_{self.train_step}.pt"
            torch.save(
                {
                    "obs": obs.detach().cpu(),
                    "legal_action": legal_action.detach().cpu(),
                    "act": act.detach().cpu(),
                    "old_prob": old_prob.detach().cpu(),
                    "old_value": old_value.detach().cpu(),
                    "reward_sum": reward_sum.detach().cpu(),
                    "advantage": advantage.detach().cpu(),
                    "rule_action": rule_action.detach().cpu(),
                    "used_rule": used_rule.detach().cpu(),
                },
                dump_path,
            )
            raise RuntimeError(
                "Batch validation failed before CUDA loss computation: "
                + " | ".join(problems)
                + f" | dumped to {dump_path}"
            )

    def learn(self, list_sample_data):
        """Training entry: perform multiple PPO mini-batch updates on one rollout batch.

        训练入口：接收一批 SampleData，对同一批样本做多轮 mini-batch PPO 更新。
        """
        # ===== 先在 CPU 上拼 batch，先检查，再搬到 GPU =====
        obs = self._stack_obs_field(list_sample_data, "obs", dtype=torch.float32)
        legal_action = self._stack_vector_field(
            list_sample_data, "legal_action", self.label_size, dtype=torch.float32
        )

        act = self._stack_1d_scalar_field(list_sample_data, "act", dtype=torch.int64)
        old_prob = self._stack_vector_field(
            list_sample_data, "prob", self.label_size, dtype=torch.float32
        )
        old_value = self._stack_1d_scalar_field(list_sample_data, "value", dtype=torch.float32)
        reward_sum = self._stack_1d_scalar_field(list_sample_data, "reward_sum", dtype=torch.float32)
        advantage = self._stack_1d_scalar_field(list_sample_data, "advantage", dtype=torch.float32)
        reward = self._stack_1d_scalar_field(list_sample_data, "reward", dtype=torch.float32)

        # ===== 规则引导字段 =====
        rule_action = self._stack_1d_scalar_field(list_sample_data, "rule_action", dtype=torch.int64)
        used_rule = self._stack_1d_scalar_field(list_sample_data, "used_rule", dtype=torch.float32)

        # ===== fail-fast 检查：不刷屏，出错直接停 =====
        self._fail_fast_batch_check(
            obs=obs,
            legal_action=legal_action,
            act=act,
            old_prob=old_prob,
            old_value=old_value,
            reward_sum=reward_sum,
            advantage=advantage,
            rule_action=rule_action,
            used_rule=used_rule,
        )

        # ===== 检查通过后再搬到 GPU =====
        obs = obs.to(self.device)
        legal_action = legal_action.to(self.device)
        act = act.to(self.device)
        old_prob = old_prob.to(self.device)
        old_value = old_value.to(self.device)
        reward_sum = reward_sum.to(self.device)
        advantage = advantage.to(self.device)
        reward = reward.to(self.device)
        rule_action = rule_action.to(self.device)
        used_rule = used_rule.to(self.device)

        used_rule_ratio = float(torch.mean(used_rule.float()).item())
        guide_coef = self._get_guide_coef()

        advantage = self._normalize_advantage(advantage)
        batch_size = obs.shape[0]
        mini_batch_size = min(Config.MINI_BATCH_SIZE, batch_size)
        epochs_ran = 0
        early_stop = False

        loss_stats = []
        value_stats = []
        policy_stats = []
        entropy_stats = []
        guide_stats = []
        kl_stats = []
        clip_fraction_stats = []

        self.model.set_train_mode()
        for epoch in range(Config.PPO_EPOCHS):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, mini_batch_size):
                mb_idx = indices[start: start + mini_batch_size]

                rst_list = self.model(obs[mb_idx])
                logits, value_pred = rst_list[0], rst_list[1]

                total_loss, info = self._compute_loss(
                    logits=logits,
                    value_pred=value_pred,
                    legal_action=legal_action[mb_idx],
                    old_action=act[mb_idx],
                    old_prob=old_prob[mb_idx],
                    old_value=old_value[mb_idx],
                    reward_sum=reward_sum[mb_idx],
                    advantage=advantage[mb_idx],
                    rule_action=rule_action[mb_idx],
                    used_rule=used_rule[mb_idx],
                    guide_coef=guide_coef,
                )

                self.optimizer.zero_grad()
                total_loss.backward()

                if Config.USE_GRAD_CLIP:
                    torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)

                self.optimizer.step()
                self.train_step += 1

                loss_stats.append(total_loss.item())
                value_stats.append(info["value_loss"])
                policy_stats.append(info["policy_loss"])
                entropy_stats.append(info["entropy_loss"])
                guide_stats.append(info["guide_loss"])
                kl_stats.append(info["approx_kl"])
                clip_fraction_stats.append(info["clip_fraction"])

                if info["approx_kl"] > Config.TARGET_KL:
                    early_stop = True
                    break

            epochs_ran += 1
            if early_stop:
                break

        with torch.no_grad():
            value_after = self.model(obs)[1]

        explained_variance = self._explained_variance(
            reward_sum.squeeze(-1) if reward_sum.dim() > 1 else reward_sum,
            value_after.squeeze(-1) if value_after.dim() > 1 else value_after,
        )

        results = {
            "total_loss": float(np.mean(loss_stats)) if loss_stats else 0.0,
            "epochs_ran": epochs_ran,
            "approx_kl": float(np.mean(kl_stats)) if kl_stats else 0.0,
            "clip_fraction": float(np.mean(clip_fraction_stats)) if clip_fraction_stats else 0.0,
            "explained_variance": explained_variance,
        }

        # Periodic monitoring report
        now = time.time()
        if now - self.last_report_time >= 60:
            reward_mean = float(reward.mean().item())

            results["value_loss"] = round(float(np.mean(value_stats)) if value_stats else 0.0, 4)
            results["policy_loss"] = round(float(np.mean(policy_stats)) if policy_stats else 0.0, 4)
            results["entropy_loss"] = round(float(np.mean(entropy_stats)) if entropy_stats else 0.0, 4)
            results["guide_loss"] = round(float(np.mean(guide_stats)) if guide_stats else 0.0, 4)
            results["guide_coef"] = round(float(guide_coef), 4)
            results["used_rule_ratio"] = round(float(used_rule_ratio), 4)

            results["reward"] = round(reward_mean, 4)
            results["approx_kl"] = round(results["approx_kl"], 6)
            results["clip_fraction"] = round(results["clip_fraction"], 4)
            results["explained_variance"] = round(results["explained_variance"], 4)

            if self.logger:
                self.logger.info(
                    f"policy_loss: {results['policy_loss']}, "
                    f"value_loss: {results['value_loss']}, "
                    f"entropy_loss: {results['entropy_loss']}, "
                    f"guide_loss: {results['guide_loss']}, "
                    f"guide_coef: {results['guide_coef']}, "
                    f"used_rule_ratio: {results['used_rule_ratio']}, "
                    f"reward: {results['reward']}, "
                    f"approx_kl: {results['approx_kl']}, "
                    f"clip_fraction: {results['clip_fraction']}, "
                    f"explained_variance: {results['explained_variance']}"
                )

            if self.monitor:
                self.monitor.put_data({os.getpid(): results})

            self.last_report_time = now

        return results

    def _compute_loss(
        self,
        logits,
        value_pred,
        legal_action,
        old_action,
        old_prob,
        old_value,
        reward_sum,
        advantage,
        rule_action,
        used_rule,
        guide_coef,
    ):
        """Compute PPO loss + guide imitation loss.

        计算 PPO 损失 + 规则引导模仿损失。
        """
        # ===== Value loss =====
        tdret = reward_sum.squeeze(-1) if reward_sum.dim() > 1 else reward_sum
        vp = value_pred.squeeze(-1) if value_pred.dim() > 1 else value_pred
        ov = old_value.squeeze(-1) if old_value.dim() > 1 else old_value

        vp_clip = ov + (vp - ov).clamp(-self.clip_param, self.clip_param)
        value_loss = (
            0.5
            * torch.maximum(
                (tdret - vp) ** 2,
                (tdret - vp_clip) ** 2,
            ).mean()
        )

        # ===== Policy distribution =====
        prob_dist = self._masked_softmax(logits, legal_action)
        entropy_loss = (-(prob_dist * torch.log(prob_dist.clamp(1e-9, 1.0))).sum(1)).mean()

        # ===== PPO policy loss =====
        one_hot = torch.nn.functional.one_hot(old_action[:, 0].long(), self.label_size).float()
        new_prob = (one_hot * prob_dist).sum(1, keepdim=True)
        old_action_prob = (one_hot * old_prob).sum(1, keepdim=True)
        ratio = new_prob / old_action_prob.clamp(1e-9)

        adv = advantage.squeeze(-1) if advantage.dim() > 1 else advantage
        adv = adv.unsqueeze(-1)

        policy_loss = torch.maximum(
            -ratio * adv,
            -ratio.clamp(1 - self.clip_param, 1 + self.clip_param) * adv,
        ).mean()

        approx_kl = (torch.log(old_action_prob.clamp(1e-9)) - torch.log(new_prob.clamp(1e-9))).mean()
        clip_fraction = ((ratio - 1.0).abs() > self.clip_param).float().mean()

        # ===== Guide imitation loss =====
        guide_mask = used_rule[:, 0] > 0.5

        if guide_mask.any():
            guided_rule_action = rule_action[guide_mask, 0].long()
            guided_prob_dist = prob_dist[guide_mask]

            rule_one_hot = torch.nn.functional.one_hot(guided_rule_action, self.label_size).float()
            rule_prob = (rule_one_hot * guided_prob_dist).sum(1, keepdim=True)
            guide_loss = -torch.log(rule_prob.clamp(1e-9, 1.0)).mean()
        else:
            guide_loss = torch.zeros(1, device=logits.device, dtype=torch.float32).mean()

        # ===== Total loss =====
        total_loss = (
            self.vf_coef * value_loss
            + policy_loss
            - self.var_beta * entropy_loss
            + guide_coef * guide_loss
        )

        return total_loss, {
            "value_loss": value_loss.item(),
            "policy_loss": policy_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "guide_loss": guide_loss.item(),
            "approx_kl": approx_kl.item(),
            "clip_fraction": clip_fraction.item(),
        }

    def _normalize_advantage(self, advantage):
        """Normalize advantages over the full rollout batch."""
        adv = advantage.squeeze(-1) if advantage.dim() > 1 else advantage
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        return adv.unsqueeze(-1)

    def _explained_variance(self, y_true, y_pred):
        """Compute explained variance for critic diagnostics."""
        var_y = torch.var(y_true)
        if var_y.item() < 1e-8:
            return 0.0
        return float((1.0 - torch.var(y_true - y_pred) / var_y).item())

    def _masked_softmax(self, logits, legal_action):
        """Apply legal action mask to logits before computing softmax."""
        label_max, _ = torch.max(logits * legal_action, dim=1, keepdim=True)
        logits = logits - label_max
        logits = logits * legal_action
        logits = logits + 1e5 * (legal_action - 1)
        return torch.nn.functional.softmax(logits, dim=1)
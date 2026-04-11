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
  total_loss = vf_coef * value_loss + policy_loss - beta * entropy_loss
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

    def learn(self, list_sample_data):
        """Training entry: perform multiple PPO mini-batch updates on one rollout batch.

        训练入口：接收一批 SampleData，对同一批样本做多轮 mini-batch PPO 更新。
        """
        obs = torch.stack([s.obs for s in list_sample_data]).to(self.device)
        legal_action = torch.stack([s.legal_action for s in list_sample_data]).to(self.device)
        act = torch.stack([s.act for s in list_sample_data]).to(self.device).view(-1, 1)
        old_prob = torch.stack([s.prob for s in list_sample_data]).to(self.device)
        old_value = torch.stack([s.value for s in list_sample_data]).to(self.device)
        reward_sum = torch.stack([s.reward_sum for s in list_sample_data]).to(self.device)
        advantage = torch.stack([s.advantage for s in list_sample_data]).to(self.device)
        reward = torch.stack([s.reward for s in list_sample_data]).to(self.device)

        # 额外奖励分项（仅用于监控与可视化）
        cleaning_reward = float(np.mean([s.cleaning_reward for s in list_sample_data]))
        dirt_approach_reward = float(np.mean([s.dirt_approach_reward for s in list_sample_data]))
        obstacle_reward = float(np.mean([s.obstacle_reward for s in list_sample_data]))
        charger_reward = float(np.mean([s.charger_reward for s in list_sample_data]))
        step_penalty = float(np.mean([s.step_penalty for s in list_sample_data]))

        advantage = self._normalize_advantage(advantage)
        batch_size = obs.shape[0]
        mini_batch_size = min(Config.MINI_BATCH_SIZE, batch_size)
        epochs_ran = 0
        early_stop = False

        loss_stats = []
        value_stats = []
        policy_stats = []
        entropy_stats = []
        kl_stats = []
        clip_fraction_stats = []

        self.model.set_train_mode()
        for epoch in range(Config.PPO_EPOCHS):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, mini_batch_size):
                mb_idx = indices[start : start + mini_batch_size]

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
        # 定期上报监控
        now = time.time()
        if now - self.last_report_time >= 60:
            reward_mean = float(reward.mean().item())

            results["value_loss"] = round(float(np.mean(value_stats)) if value_stats else 0.0, 4)
            results["policy_loss"] = round(float(np.mean(policy_stats)) if policy_stats else 0.0, 4)
            results["entropy_loss"] = round(float(np.mean(entropy_stats)) if entropy_stats else 0.0, 4)
            results["reward"] = round(reward_mean, 4)
            results["approx_kl"] = round(results["approx_kl"], 6)
            results["clip_fraction"] = round(results["clip_fraction"], 4)
            results["explained_variance"] = round(results["explained_variance"], 4)

            # 新增奖励分项
            results["cleaning_reward"] = round(cleaning_reward, 4)
            results["dirt_approach_reward"] = round(dirt_approach_reward, 4)
            results["obstacle_reward"] = round(obstacle_reward, 4)
            results["charger_reward"] = round(charger_reward, 4)
            results["step_penalty"] = round(step_penalty, 4)

            if self.logger:
                self.logger.info(
                    f"policy_loss: {results['policy_loss']}, "
                    f"value_loss: {results['value_loss']}, "
                    f"entropy_loss: {results['entropy_loss']}, "
                    f"reward: {results['reward']}, "
                    f"approx_kl: {results['approx_kl']}, "
                    f"clip_fraction: {results['clip_fraction']}, "
                    f"explained_variance: {results['explained_variance']}, "
                    f"cleaning_reward: {results['cleaning_reward']}, "
                    f"dirt_approach_reward: {results['dirt_approach_reward']}, "
                    f"obstacle_reward: {results['obstacle_reward']}, "
                    f"charger_reward: {results['charger_reward']}, "
                    f"step_penalty: {results['step_penalty']}"
                )

            if self.monitor:
                self.monitor.put_data({os.getpid(): results})

            self.last_report_time = now

        return results

    def _compute_loss(self, logits, value_pred, legal_action, old_action, old_prob, old_value, reward_sum, advantage):
        """Compute standard PPO loss (policy + value + entropy).

        计算标准 PPO 三项损失。
        """
        # Value loss (clipped)
        # 价值损失（裁剪）
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

        # Policy loss (PPO clip)
        # 策略损失（PPO clip）
        prob_dist = self._masked_softmax(logits, legal_action)
        entropy_loss = (-(prob_dist * torch.log(prob_dist.clamp(1e-9, 1))).sum(1)).mean()

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

        # Total loss
        # 总损失
        total_loss = self.vf_coef * value_loss + policy_loss - self.var_beta * entropy_loss

        return total_loss, {
            "value_loss": value_loss.item(),
            "policy_loss": policy_loss.item(),
            "entropy_loss": entropy_loss.item(),
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
        """Apply legal action mask to logits before computing softmax.

        对 logits 应用合法动作掩码后计算 softmax。
        """
        label_max, _ = torch.max(logits * legal_action, dim=1, keepdim=True)
        logits = logits - label_max
        logits = logits * legal_action
        logits = logits + 1e5 * (legal_action - 1)
        return torch.nn.functional.softmax(logits, dim=1)
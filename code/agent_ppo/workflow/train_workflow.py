#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Training workflow for Robot Vacuum.
清扫大作战训练工作流。
"""

import os
import time
import numpy as np

from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import SampleData, sample_process
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    last_save_model_time = time.time()
    env = envs[0]
    agent = agents[0]

    # Read and validate user configuration
    usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_ppo/conf/train_env_conf.toml")
        return

    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        usr_conf=usr_conf,
        logger=logger,
        monitor=monitor,
    )

    while True:
        for g_data in episode_runner.run_episodes():
            agent.send_sample_data(g_data)
            g_data.clear()

            now = time.time()
            if now - last_save_model_time >= 1800:
                agent.save_model()
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agent, usr_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor

        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0

        # ===== 规则引导参数 =====
        self.guide_prob_start = 0.80
        self.guide_prob_end = 0.05
        self.guide_decay_episodes = 100000
        self.guide_prob = 0.70

        # 低电量阈值（最好和 preprocessor 里保持一致）
        self.low_battery_threshold = 0.30

        # ===== 奖励拆分监控统计 =====
        self.reward_metric_sums = {
            "cleaning_reward": 0.0,
            "clean_streak_reward": 0.0,
            "dirt_approach_reward": 0.0,
            "obstacle_reward": 0.0,
            "charger_reward": 0.0,
            "charger_approach_reward": 0.0,
            "leave_charger_penalty": 0.0,
            "charging_reward": 0.0,
            "battery_dead_penalty": 0.0,
            "no_move_penalty": 0.0,
            "step_penalty": 0.0,
            "low_battery_mode": 0.0,
            "used_rule_ratio": 0.0,
        }
        self.reward_metric_count = 0

    def _reset_reward_metrics(self):
        """重置奖励拆分统计。"""
        for k in self.reward_metric_sums:
            self.reward_metric_sums[k] = 0.0
        self.reward_metric_count = 0

    def _update_reward_metrics(self, reward_info, used_rule):
        """按步累计奖励拆分项，最终按均值上报到控制面板。"""
        if reward_info is None:
            reward_info = {}

        self.reward_metric_sums["cleaning_reward"] += float(reward_info.get("cleaning_reward", 0.0))
        self.reward_metric_sums["clean_streak_reward"] += float(reward_info.get("clean_streak_reward", 0.0))
        self.reward_metric_sums["dirt_approach_reward"] += float(reward_info.get("dirt_approach_reward", 0.0))
        self.reward_metric_sums["obstacle_reward"] += float(reward_info.get("obstacle_reward", 0.0))
        self.reward_metric_sums["charger_reward"] += float(reward_info.get("charger_reward", 0.0))
        self.reward_metric_sums["charger_approach_reward"] += float(reward_info.get("charger_approach_reward", 0.0))
        self.reward_metric_sums["leave_charger_penalty"] += float(reward_info.get("leave_charger_penalty", 0.0))
        self.reward_metric_sums["charging_reward"] += float(reward_info.get("charging_reward", 0.0))
        self.reward_metric_sums["battery_dead_penalty"] += float(reward_info.get("battery_dead_penalty", 0.0))
        self.reward_metric_sums["no_move_penalty"] += float(reward_info.get("no_move_penalty", 0.0))
        self.reward_metric_sums["step_penalty"] += float(reward_info.get("step_penalty", 0.0))
        self.reward_metric_sums["low_battery_mode"] += float(reward_info.get("low_battery_mode", 0.0))
        self.reward_metric_sums["used_rule_ratio"] += float(used_rule)
        self.reward_metric_count += 1

    def _report_reward_metrics(self):
        """将奖励拆分项按均值上报到控制面板。"""
        if not self.monitor:
            return

        now = time.time()
        if now - self.last_report_monitor_time < 60:
            return

        metric_count = max(1, self.reward_metric_count)

        monitor_dict = {
            "cleaning_reward": self.reward_metric_sums["cleaning_reward"] / metric_count,
            "clean_streak_reward": self.reward_metric_sums["clean_streak_reward"] / metric_count,
            "dirt_approach_reward": self.reward_metric_sums["dirt_approach_reward"] / metric_count,
            "obstacle_reward": self.reward_metric_sums["obstacle_reward"] / metric_count,
            "charger_reward": self.reward_metric_sums["charger_reward"] / metric_count,
            "charger_approach_reward": self.reward_metric_sums["charger_approach_reward"] / metric_count,
            "leave_charger_penalty": self.reward_metric_sums["leave_charger_penalty"] / metric_count,
            "charging_reward": self.reward_metric_sums["charging_reward"] / metric_count,
            "battery_dead_penalty": self.reward_metric_sums["battery_dead_penalty"] / metric_count,
            "no_move_penalty": self.reward_metric_sums["no_move_penalty"] / metric_count,
            "step_penalty": self.reward_metric_sums["step_penalty"] / metric_count,
            "low_battery_mode": self.reward_metric_sums["low_battery_mode"] / metric_count,
            "used_rule_ratio": self.reward_metric_sums["used_rule_ratio"] / metric_count,
        }

        self.monitor.put_data({os.getpid(): monitor_dict})
        self.last_report_monitor_time = now

    def _get_guide_prob(self):
        """根据训练局数计算当前规则接管概率。"""
        progress = min(1.0, self.episode_cnt / max(1, self.guide_decay_episodes))
        guide_prob = self.guide_prob_start + progress * (self.guide_prob_end - self.guide_prob_start)
        return float(guide_prob)

    def _should_use_rule(self, preprocessor):
        """判断当前是否启用规则动作接管。"""
        battery_ratio = float(preprocessor.battery) / max(float(preprocessor.battery_max), 1.0)
        if battery_ratio >= self.low_battery_threshold:
            return False

        if preprocessor.rule_action is None or preprocessor.rule_action < 0:
            return False

        guide_prob = self._get_guide_prob()
        return np.random.rand() < guide_prob

    def run_episodes(self):
        """Run a single episode and yield collected samples.

        单局流程（generator），完成一局后 yield 整局样本。
        """
        while True:
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None and self.logger:
                    self.logger.info(f"training_metrics: {training_metrics}")

            env_obs = self.env.reset(self.usr_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")

            obs_data, remain_info = self.agent.observation_process(env_obs)

            collector = []
            self.episode_cnt += 1
            done = False
            step = 0
            total_reward = 0.0
            self._reset_reward_metrics()

            if self.logger:
                self.logger.info(f"Episode {self.episode_cnt} start")

            while not done:
                act_data_list = self.agent.predict([obs_data])
                act_data = act_data_list[0]

                policy_act = self.agent.action_process(act_data)
                act = policy_act
                used_rule = 0
                rule_action = -1

                fm = self.agent.preprocessor
                if hasattr(fm, "rule_action"):
                    rule_action = int(fm.rule_action)

                if self._should_use_rule(fm):
                    act = rule_action
                    used_rule = 1
                    act_data.action = np.array([rule_action], dtype=np.int64)

                env_reward, env_obs = self.env.step(act)
                if handle_disaster_recovery(env_obs, self.logger):
                    break
                if env_obs is None:
                    if self.logger:
                        self.logger.error("env.step returned None env_obs, abort episode")
                    break
                if env_obs.get("observation") is None:
                    if self.logger:
                        self.logger.error("env_obs['observation'] is None, abort episode")
                    break

                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                frame_no = env_obs["frame_no"]
                step += 1
                done = terminated or truncated

                _obs_data, _ = self.agent.observation_process(env_obs)
                _obs_data.frame_no = frame_no

                reward_scalar = float(self.agent.last_reward)
                total_reward += reward_scalar

                reward_info = getattr(self.agent.preprocessor, "reward_info", {})
                self._update_reward_metrics(reward_info, used_rule)

                final_reward = 0.0
                if done:
                    fm = self.agent.preprocessor

                    if truncated:
                        cleaning_ratio = fm.dirt_cleaned / max(fm.total_dirt, 1)
                        final_reward = 5.0 + 5.0 * cleaning_ratio
                        result_str = "WIN"
                    else:
                        if getattr(fm, "collision_type", "none") == "charger":
                            final_reward = 2.0
                            result_str = "CHARGER_REACHED"
                        else:
                            final_reward = -2.0
                            result_str = "FAIL"

                    if self.logger:
                        self.logger.info(
                            f"[GUIDE] ep:{self.episode_cnt} "
                            f"guide_prob:{self._get_guide_prob():.3f} "
                            f"last_rule_action:{rule_action} used_rule:{used_rule} "
                            f"battery:{fm.battery}/{fm.battery_max} "
                            f"charger_dist:{getattr(fm, 'nearest_charger_dist', -1.0):.2f} "
                            f"result:{result_str}"
                        )

                reward_arr = np.array([reward_scalar], dtype=np.float32)
                value_arr = np.array(act_data.value, dtype=np.float32).reshape(-1)[: Config.VALUE_NUM]

                frame = SampleData(
                    obs=np.array(obs_data.feature, dtype=np.float32),
                    legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                    act=np.array(act_data.action, dtype=np.int64),

                    reward=reward_arr.astype(np.float32),
                    cleaning_reward=np.array([reward_info.get("cleaning_reward", 0.0)], dtype=np.float32),
                    dirt_approach_reward=np.array([reward_info.get("dirt_approach_reward", 0.0)], dtype=np.float32),
                    obstacle_reward=np.array([reward_info.get("obstacle_reward", 0.0)], dtype=np.float32),
                    charger_reward=np.array([reward_info.get("charger_reward", 0.0)], dtype=np.float32),
                    step_penalty=np.array([reward_info.get("step_penalty", 0.0)], dtype=np.float32),

                    clean_streak_reward=np.array([reward_info.get("clean_streak_reward", 0.0)], dtype=np.float32),
                    charger_approach_reward=np.array([reward_info.get("charger_approach_reward", 0.0)], dtype=np.float32),
                    leave_charger_penalty=np.array([reward_info.get("leave_charger_penalty", 0.0)], dtype=np.float32),
                    charging_reward=np.array([reward_info.get("charging_reward", 0.0)], dtype=np.float32),
                    battery_dead_penalty=np.array([reward_info.get("battery_dead_penalty", 0.0)], dtype=np.float32),
                    no_move_penalty=np.array([reward_info.get("no_move_penalty", 0.0)], dtype=np.float32),
                    low_battery_mode=np.array([reward_info.get("low_battery_mode", 0.0)], dtype=np.float32),

                    rule_action=np.array([rule_action], dtype=np.int64),
                    used_rule=np.array([used_rule], dtype=np.float32),

                    reward_sum=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                    done=np.array([float(done)], dtype=np.float32),
                    value=value_arr.astype(np.float32),
                    next_value=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                    advantage=np.zeros(Config.VALUE_NUM, dtype=np.float32),

                    prob=np.array(act_data.prob, dtype=np.float32),
                )
                collector.append(frame)

                if done:
                    collector[-1].reward = collector[-1].reward + np.array([final_reward], dtype=np.float32)

                    self._report_reward_metrics()

                    if collector:
                        collector = sample_process(collector)
                        yield collector
                    break

                obs_data = _obs_data
#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Data definition and GAE computation for Robot Vacuum.
清扫大作战数据类定义与 GAE 计算。
"""

import numpy as np
from common_python.utils.common_func import create_cls
from agent_ppo.conf.conf import Config


# ObsData: feature vector + legal action mask
# 观测数据：feature 为特征向量，legal_action 为合法动作掩码
ObsData = create_cls("ObsData", feature=None, legal_action=None)

# ActData: sampled action, greedy action, action probabilities, state value
# 动作数据：action 为采样动作，d_action 为贪心动作，prob 为动作概率，value 为状态价值
ActData = create_cls(
    "ActData",
    action=None,
    d_action=None,
    prob=None,
    value=None,
)

# SampleData: int values are treated as dimensions by the framework
# 训练样本数据：字段值为 int 时框架自动按维度处理
# SampleData: int values are treated as dimensions by the framework
# 训练样本数据：字段值为 int 时框架自动按维度处理
SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION,
    legal_action=Config.ACTION_NUM,
    act=1,

    reward=Config.VALUE_NUM,
    cleaning_reward=Config.VALUE_NUM,
    dirt_approach_reward=Config.VALUE_NUM,
    obstacle_reward=Config.VALUE_NUM,
    charger_reward=Config.VALUE_NUM,
    step_penalty=Config.VALUE_NUM,

    clean_streak_reward=Config.VALUE_NUM,
    charger_approach_reward=Config.VALUE_NUM,
    leave_charger_penalty=Config.VALUE_NUM,
    charging_reward=Config.VALUE_NUM,
    battery_dead_penalty=Config.VALUE_NUM,
    no_move_penalty=Config.VALUE_NUM,
    low_battery_mode=Config.VALUE_NUM,

    rule_action=1,
    used_rule=1,

    reward_sum=Config.VALUE_NUM,
    done=1,
    value=Config.VALUE_NUM,
    next_value=Config.VALUE_NUM,
    advantage=Config.VALUE_NUM,
    prob=Config.ACTION_NUM,
)


def sample_process(list_sample_data):
    """Fill next_value and compute GAE advantage.

    计算 GAE 并填充 next_value。
    """
    if not list_sample_data:
        return list_sample_data

    # 给每一步填 next_value，最后一步保持 0
    for i in range(len(list_sample_data) - 1):
        list_sample_data[i].next_value = list_sample_data[i + 1].value

    # 最后一帧 next_value 保持 0，表示 episode 结束
    list_sample_data[-1].next_value = np.zeros(Config.VALUE_NUM, dtype=np.float32)

    _calc_gae(list_sample_data)
    return list_sample_data


def _calc_gae(list_sample_data):
    """Compute advantage and cumulative return using GAE(λ).

    使用 GAE(λ) 计算优势函数与累积回报。
    """
    gae = np.zeros(Config.VALUE_NUM, dtype=np.float32)
    gamma = Config.GAMMA
    lamda = Config.LAMDA

    for sample in reversed(list_sample_data):
        reward = np.array(sample.reward, dtype=np.float32)
        value = np.array(sample.value, dtype=np.float32)
        next_value = np.array(sample.next_value, dtype=np.float32)
        done = float(np.array(sample.done).reshape(-1)[0])

        # done=1 时不再 bootstrap
        not_done = 1.0 - done
        delta = reward + gamma * next_value * not_done - value
        gae = delta + gamma * lamda * not_done * gae

        sample.advantage = gae.astype(np.float32)
        sample.reward_sum = (gae + value).astype(np.float32)
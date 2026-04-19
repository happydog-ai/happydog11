#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration for Robot Vacuum PPO agent.
清扫大作战 PPO 配置。
"""


class Config:

    # Feature dimensions:
    # 1) full local view 21x21
    # 2) coarse global memory map 4x16x16
    # 3) scalar features
    # 特征维度：
    # 1) 完整局部视野 21x21
    # 2) 粗粒度全局记忆地图 4x16x16
    # 3) 标量特征
    FEATURES = [
        21 * 21,
        4 * 16 * 16,
        12,
        8,
        3,
        3,
        3,
        3,
        3,
        1,
        8
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURES)
    DIM_OF_OBSERVATION = FEATURE_LEN

    # Action space: 8 directional moves
    # 动作空间：8个方向移动
    ACTION_NUM = 8

    # Single-head value
    # 单头价值
    VALUE_NUM = 1

    # PPO hyperparameters
    # PPO 超参数
    GAMMA = 0.99
    LAMDA = 0.95

    INIT_LEARNING_RATE_START = 0.0002
    BETA_START = 0.001
    CLIP_PARAM = 0.2
    VF_COEF = 0.5
    PPO_EPOCHS = 4
    MINI_BATCH_SIZE = 128
    TARGET_KL = 0.02

    LABEL_SIZE_LIST = [ACTION_NUM]
    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5

    # Model architecture
    # 模型结构超参数
    LOCAL_VIEW_CHANNELS = 1
    LOCAL_VIEW_SIZE = 21
    GLOBAL_MAP_CHANNELS = 4
    GLOBAL_MAP_SIZE = 16
    LOCAL_ENCODE_DIM = 128
    GLOBAL_MAP_ENCODE_DIM = 128
    SCALAR_ENCODE_DIM = 128
    ACTOR_HIDDEN_DIM = 256
    CRITIC_HIDDEN_DIM = 256

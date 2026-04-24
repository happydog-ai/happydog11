#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Dual-tower PPO network for Robot Vacuum.
清扫大作战双塔 PPO 策略网络。
"""

import torch
import torch.nn as nn

from agent_ppo.conf.conf import Config


def _make_fc(in_dim, out_dim, gain=1.41421):
    """Create a linear layer with orthogonal initialization.

    创建正交初始化的线性层。
    """
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class ResidualBlock(nn.Module):
    """Two-layer residual MLP block with LayerNorm."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = _make_fc(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = _make_fc(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.act(out)
        out = self.fc2(out)
        out = self.ln2(out)
        out = self.act(out + identity)
        return out


class MapEncoder(nn.Module):
    """Encode 2D map-like features with lightweight convolutions."""

    def __init__(self, in_channels, size, out_dim):
        super().__init__()
        self.in_channels = in_channels
        self.size = size
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            _make_fc(32 * 4 * 4, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, feature_map):
        batch_size = feature_map.shape[0]
        feature_map = feature_map.view(
            batch_size,
            self.in_channels,
            self.size,
            self.size,
        )
        return self.encoder(feature_map)


class ScalarFeatureEncoder(nn.Module):
    """Encode non-spatial scalar features with an MLP."""

    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            _make_fc(input_dim, Config.SCALAR_ENCODE_DIM),
            nn.LayerNorm(Config.SCALAR_ENCODE_DIM),
            nn.ReLU(),
            _make_fc(Config.SCALAR_ENCODE_DIM, Config.SCALAR_ENCODE_DIM),
            nn.ReLU(),
        )

    def forward(self, scalar_feature):
        return self.encoder(scalar_feature)


class PPOEncoder(nn.Module):
    """Independent encoder tower for actor or critic."""

    def __init__(self, hidden_dim):
        super().__init__()
        feature_dims = Config.FEATURE_SPLIT_SHAPE
        scalar_dim = sum(feature_dims[2:])

        self.local_encoder = MapEncoder(
            in_channels=Config.LOCAL_VIEW_CHANNELS,
            size=Config.LOCAL_VIEW_SIZE,
            out_dim=Config.LOCAL_ENCODE_DIM,
        )
        self.global_map_encoder = MapEncoder(
            in_channels=Config.GLOBAL_MAP_CHANNELS,
            size=Config.GLOBAL_MAP_SIZE,
            out_dim=Config.GLOBAL_MAP_ENCODE_DIM,
        )
        self.scalar_encoder = ScalarFeatureEncoder(scalar_dim)
        fusion_dim = (
            Config.LOCAL_ENCODE_DIM
            + Config.GLOBAL_MAP_ENCODE_DIM
            + Config.SCALAR_ENCODE_DIM
        )
        self.fusion = nn.Sequential(
            _make_fc(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
        )

    def forward(self, x):
        local_view, global_map, *scalar_parts = torch.split(x, Config.FEATURE_SPLIT_SHAPE, dim=1)
        scalar_feature = torch.cat(scalar_parts, dim=1)

        local_embedding = self.local_encoder(local_view)
        global_map_embedding = self.global_map_encoder(global_map)
        scalar_embedding = self.scalar_encoder(scalar_feature)
        fused = torch.cat(
            [local_embedding, global_map_embedding, scalar_embedding],
            dim=1,
        )
        return self.fusion(fused)


class PolicyHead(nn.Module):
    """Actor head for action logits."""

    def __init__(self, input_dim, act_num):
        super().__init__()
        self.head = nn.Sequential(
            _make_fc(input_dim, 128),
            nn.ReLU(),
            _make_fc(128, act_num, gain=0.01),
        )

    def forward(self, x):
        return self.head(x)


class ValueHead(nn.Module):
    """Critic head for scalar value prediction."""

    def __init__(self, input_dim):
        super().__init__()
        self.head = nn.Sequential(
            _make_fc(input_dim, 128),
            nn.ReLU(),
            _make_fc(128, 1, gain=1.0),
        )

    def forward(self, x):
        return self.head(x)


class Model(nn.Module):
    """Dual-tower PPO network for Robot Vacuum.

    清扫大作战 PPO 双塔网络：
    Actor 与 Critic 各自拥有独立编码器，避免价值学习与策略学习相互干扰。
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "robot_vacuum"
        self.device = device

        act_num = Config.ACTION_NUM

        self.actor_encoder = PPOEncoder(Config.ACTOR_HIDDEN_DIM)
        self.critic_encoder = PPOEncoder(Config.CRITIC_HIDDEN_DIM)
        self.actor_head = PolicyHead(Config.ACTOR_HIDDEN_DIM, act_num)
        self.critic_head = ValueHead(Config.CRITIC_HIDDEN_DIM)

    def forward(self, s, inference=False):
        """Forward pass.

        前向传播。
        """
        x = s.to(torch.float32)
        actor_feature = self.actor_encoder(x)
        critic_feature = self.critic_encoder(x)

        logits = self.actor_head(actor_feature)
        value = self.critic_head(critic_feature)
        return [logits, value]

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()

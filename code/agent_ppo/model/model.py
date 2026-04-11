#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Simple MLP policy network for Robot Vacuum.
清扫大作战策略网络。
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


class Model(nn.Module):
    """Dual-head MLP for Robot Vacuum.

    清扫大作战双头 MLP 策略网络。
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "robot_vacuum"
        self.device = device

        obs_dim = Config.DIM_OF_OBSERVATION  # 69
        act_num = Config.ACTION_NUM  # 8

        hidden_dim = 256

        # Shared backbone / 共享骨干网络
        self.input_proj = nn.Sequential(
            _make_fc(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.shared_blocks = nn.Sequential(
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
        )

        # Actor branch / 策略分支
        self.actor_mlp = nn.Sequential(
            _make_fc(hidden_dim, 128),
            nn.ReLU(),
            _make_fc(128, 64),
            nn.ReLU(),
        )
        self.actor_head = _make_fc(64, act_num, gain=0.01)

        # Critic branch / 价值分支
        self.critic_mlp = nn.Sequential(
            _make_fc(hidden_dim, 128),
            nn.ReLU(),
            _make_fc(128, 64),
            nn.ReLU(),
        )
        self.critic_head = _make_fc(64, 1, gain=0.01)

    def forward(self, s, inference=False):
        """Forward pass.

        前向传播。
        """
        x = s.to(torch.float32)
        h = self.input_proj(x)
        h = self.shared_blocks(h)

        actor_h = self.actor_mlp(h)
        critic_h = self.critic_mlp(h)

        logits = self.actor_head(actor_h)
        value = self.critic_head(critic_h)
        return [logits, value]

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()

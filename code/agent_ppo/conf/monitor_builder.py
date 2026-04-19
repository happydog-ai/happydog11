#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Monitor panel configuration builder for Robot Vacuum.
清扫大作战监控面板配置构建器。
"""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    """
    # This function is used to create monitoring panel configurations for custom indicators.
    # 该函数用于创建自定义指标的监控面板配置。
    """
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("清扫大作战")
        .add_group(
            group_name="算法指标",
            group_name_en="algorithm",
        )

        # =========================
        # 奖励总览
        # =========================
        .add_panel(
            name="累积回报",
            name_en="reward",
            type="line",
        )
        .add_metric(
            metrics_name="reward",
            expr="avg(reward{})",
        )
        .end_panel()

        # =========================
        # 奖励拆分项
        # =========================
        .add_panel(
            name="清扫奖励",
            name_en="cleaning_reward",
            type="line",
        )
        .add_metric(
            metrics_name="cleaning_reward",
            expr="avg(cleaning_reward{})",
        )
        .end_panel()

        .add_panel(
            name="连续清扫奖励",
            name_en="clean_streak_reward",
            type="line",
        )
        .add_metric(
            metrics_name="clean_streak_reward",
            expr="avg(clean_streak_reward{})",
        )
        .end_panel()

        .add_panel(
            name="接近污渍奖励",
            name_en="dirt_approach_reward",
            type="line",
        )
        .add_metric(
            metrics_name="dirt_approach_reward",
            expr="avg(dirt_approach_reward{})",
        )
        .end_panel()

        .add_panel(
            name="障碍物安全奖励",
            name_en="obstacle_reward",
            type="line",
        )
        .add_metric(
            metrics_name="obstacle_reward",
            expr="avg(obstacle_reward{})",
        )
        .end_panel()

        .add_panel(
            name="充电桩总奖励",
            name_en="charger_reward",
            type="line",
        )
        .add_metric(
            metrics_name="charger_reward",
            expr="avg(charger_reward{})",
        )
        .end_panel()

        .add_panel(
            name="接近充电桩奖励",
            name_en="charger_approach_reward",
            type="line",
        )
        .add_metric(
            metrics_name="charger_approach_reward",
            expr="avg(charger_approach_reward{})",
        )
        .end_panel()

        .add_panel(
            name="离开充电桩惩罚",
            name_en="leave_charger_penalty",
            type="line",
        )
        .add_metric(
            metrics_name="leave_charger_penalty",
            expr="avg(leave_charger_penalty{})",
        )
        .end_panel()

        .add_panel(
            name="充电奖励",
            name_en="charging_reward",
            type="line",
        )
        .add_metric(
            metrics_name="charging_reward",
            expr="avg(charging_reward{})",
        )
        .end_panel()

        .add_panel(
            name="电量耗尽惩罚",
            name_en="battery_dead_penalty",
            type="line",
        )
        .add_metric(
            metrics_name="battery_dead_penalty",
            expr="avg(battery_dead_penalty{})",
        )
        .end_panel()

        .add_panel(
            name="位置不变惩罚",
            name_en="no_move_penalty",
            type="line",
        )
        .add_metric(
            metrics_name="no_move_penalty",
            expr="avg(no_move_penalty{})",
        )
        .end_panel()

        .add_panel(
            name="时间惩罚",
            name_en="step_penalty",
            type="line",
        )
        .add_metric(
            metrics_name="step_penalty",
            expr="avg(step_penalty{})",
        )
        .end_panel()

        # =========================
        # 状态/策略辅助指标
        # =========================
        .add_panel(
            name="低电量模式占比",
            name_en="low_battery_mode",
            type="line",
        )
        .add_metric(
            metrics_name="low_battery_mode",
            expr="avg(low_battery_mode{})",
        )
        .end_panel()

        .add_panel(
            name="规则接管比例",
            name_en="used_rule_ratio",
            type="line",
        )
        .add_metric(
            metrics_name="used_rule_ratio",
            expr="avg(used_rule_ratio{})",
        )
        .end_panel()

        .add_panel(
            name="规则引导损失",
            name_en="guide_loss",
            type="line",
        )
        .add_metric(
            metrics_name="guide_loss",
            expr="avg(guide_loss{})",
        )
        .end_panel()

        .add_panel(
            name="规则引导系数",
            name_en="guide_coef",
            type="line",
        )
        .add_metric(
            metrics_name="guide_coef",
            expr="avg(guide_coef{})",
        )
        .end_panel()

        # =========================
        # PPO损失项
        # =========================
        .add_panel(
            name="总损失",
            name_en="total_loss",
            type="line",
        )
        .add_metric(
            metrics_name="total_loss",
            expr="avg(total_loss{})",
        )
        .end_panel()

        .add_panel(
            name="价值损失",
            name_en="value_loss",
            type="line",
        )
        .add_metric(
            metrics_name="value_loss",
            expr="avg(value_loss{})",
        )
        .end_panel()

        .add_panel(
            name="策略损失",
            name_en="policy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="policy_loss",
            expr="avg(policy_loss{})",
        )
        .end_panel()

        .add_panel(
            name="熵损失",
            name_en="entropy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="entropy_loss",
            expr="avg(entropy_loss{})",
        )
        .end_panel()

        .add_panel(
            name="KL散度",
            name_en="approx_kl",
            type="line",
        )
        .add_metric(
            metrics_name="approx_kl",
            expr="avg(approx_kl{})",
        )
        .end_panel()

        .add_panel(
            name="裁剪比例",
            name_en="clip_fraction",
            type="line",
        )
        .add_metric(
            metrics_name="clip_fraction",
            expr="avg(clip_fraction{})",
        )
        .end_panel()

        .add_panel(
            name="解释方差",
            name_en="explained_variance",
            type="line",
        )
        .add_metric(
            metrics_name="explained_variance",
            expr="avg(explained_variance{})",
        )
        .end_panel()

        .end_group()
        .build()
    )

    return config_dict
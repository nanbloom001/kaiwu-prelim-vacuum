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
    This function is used to create monitoring panel configurations for custom indicators.
    该函数用于创建自定义指标的监控面板配置。
    """
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("清扫大作战DIY")
        .add_group(
            group_name="算法指标",
            group_name_en="algorithm",
        )
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
            name="决策策略损失",
            name_en="decision_policy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="decision_policy_loss",
            expr="avg(decision_policy_loss{})",
        )
        .end_panel()
        .add_panel(
            name="路径风格损失",
            name_en="style_policy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="style_policy_loss",
            expr="avg(style_policy_loss{})",
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
            name="模仿损失",
            name_en="imitation_loss",
            type="line",
        )
        .add_metric(
            metrics_name="imitation_loss",
            expr="avg(imitation_loss{})",
        )
        .end_panel()
        .add_panel(
            name="教师混合系数",
            name_en="teacher_mix",
            type="line",
        )
        .add_metric(
            metrics_name="teacher_mix",
            expr="avg(teacher_mix{})",
        )
        .end_panel()
        .add_panel(
            name="模仿系数",
            name_en="imitation_coef",
            type="line",
        )
        .add_metric(
            metrics_name="imitation_coef",
            expr="avg(imitation_coef{})",
        )
        .end_panel()
        .add_panel(
            name="教师权重",
            name_en="teacher_weight",
            type="line",
        )
        .add_metric(
            metrics_name="teacher_weight",
            expr="avg(teacher_weight{})",
        )
        .end_panel()
        .add_panel(
            name="策略权重",
            name_en="policy_weight",
            type="line",
        )
        .add_metric(
            metrics_name="policy_weight",
            expr="avg(policy_weight{})",
        )
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict

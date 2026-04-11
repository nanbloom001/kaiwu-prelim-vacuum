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
            name="覆盖率",
            name_en="coverage_rate",
            type="line",
        )
        .add_metric(
            metrics_name="coverage_rate",
            expr="avg(coverage_rate{})",
        )
        .end_panel()
        .add_panel(
            name="清扫比例",
            name_en="clean_ratio",
            type="line",
        )
        .add_metric(
            metrics_name="clean_ratio",
            expr="avg(clean_ratio{})",
        )
        .end_panel()
        .add_panel(
            name="重复访问率",
            name_en="repeat_visit_ratio",
            type="line",
        )
        .add_metric(
            metrics_name="repeat_visit_ratio",
            expr="avg(repeat_visit_ratio{})",
        )
        .end_panel()
        .add_panel(
            name="回充成功次数",
            name_en="charge_success_cnt",
            type="line",
        )
        .add_metric(
            metrics_name="charge_success_cnt",
            expr="avg(charge_success_cnt{})",
        )
        .end_panel()
        .add_panel(
            name="回充成功率",
            name_en="charge_success_rate",
            type="line",
        )
        .add_metric(
            metrics_name="charge_success_rate",
            expr="avg(charge_success_rate{})",
        )
        .end_panel()
        .add_panel(
            name="低电量触发次数",
            name_en="low_battery_trigger_cnt",
            type="line",
        )
        .add_metric(
            metrics_name="low_battery_trigger_cnt",
            expr="avg(low_battery_trigger_cnt{})",
        )
        .end_panel()
        .add_panel(
            name="近敌步数",
            name_en="near_npc_steps",
            type="line",
        )
        .add_metric(
            metrics_name="near_npc_steps",
            expr="avg(near_npc_steps{})",
        )
        .end_panel()
        .add_panel(
            name="卡死次数",
            name_en="stuck_cnt",
            type="line",
        )
        .add_metric(
            metrics_name="stuck_cnt",
            expr="avg(stuck_cnt{})",
        )
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict

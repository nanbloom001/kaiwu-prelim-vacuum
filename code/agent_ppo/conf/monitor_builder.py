#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Monitor panel configuration builder for Robot Vacuum.
"""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("Robot Vacuum PPO")
        .add_group(group_name="Algorithm", group_name_en="algorithm")
        .add_panel(name="Reward", name_en="reward", type="line")
        .add_metric(metrics_name="reward", expr="avg(reward{})")
        .end_panel()
        .add_panel(name="Total Loss", name_en="total_loss", type="line")
        .add_metric(metrics_name="total_loss", expr="avg(total_loss{})")
        .end_panel()
        .add_panel(name="Value Loss", name_en="value_loss", type="line")
        .add_metric(metrics_name="value_loss", expr="avg(value_loss{})")
        .end_panel()
        .add_panel(name="Policy Loss", name_en="policy_loss", type="line")
        .add_metric(metrics_name="policy_loss", expr="avg(policy_loss{})")
        .end_panel()
        .add_panel(name="Entropy Loss", name_en="entropy_loss", type="line")
        .add_metric(metrics_name="entropy_loss", expr="avg(entropy_loss{})")
        .end_panel()
        .add_panel(name="Avg Episode Steps", name_en="avg_episode_steps", type="line")
        .add_metric(metrics_name="avg_episode_steps", expr="avg(avg_episode_steps{})")
        .end_panel()
        .add_panel(name="Avg Charge Count", name_en="avg_charge_count", type="line")
        .add_metric(metrics_name="avg_charge_count", expr="avg(avg_charge_count{})")
        .end_panel()
        .add_panel(name="Avg Clean Score", name_en="avg_cleaned_cells", type="line")
        .add_metric(metrics_name="avg_cleaned_cells", expr="avg(avg_cleaned_cells{})")
        .end_panel()
        .add_panel(name="Avg Remaining Charge", name_en="avg_remaining_charge", type="line")
        .add_metric(metrics_name="avg_remaining_charge", expr="avg(avg_remaining_charge{})")
        .end_panel()
        .add_panel(name="Invalid Move Rate", name_en="avg_invalid_move_rate", type="line")
        .add_metric(metrics_name="avg_invalid_move_rate", expr="avg(avg_invalid_move_rate{})")
        .end_panel()
        .add_panel(name="Charge Efficiency", name_en="avg_charge_efficiency", type="line")
        .add_metric(metrics_name="avg_charge_efficiency", expr="avg(avg_charge_efficiency{})")
        .end_panel()
        .add_panel(name="Avg Clean Per Step", name_en="avg_clean_per_step", type="line")
        .add_metric(metrics_name="avg_clean_per_step", expr="avg(avg_clean_per_step{})")
        .end_panel()
        .add_panel(name="Battery Fail Rate", name_en="battery_fail_rate", type="line")
        .add_metric(metrics_name="battery_fail_rate", expr="avg(battery_fail_rate{})")
        .end_panel()
        .add_panel(name="Collision Fail Rate", name_en="collision_fail_rate", type="line")
        .add_metric(metrics_name="collision_fail_rate", expr="avg(collision_fail_rate{})")
        .end_panel()
        .add_panel(name="Completed Rate", name_en="completed_rate", type="line")
        .add_metric(metrics_name="completed_rate", expr="avg(completed_rate{})")
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict

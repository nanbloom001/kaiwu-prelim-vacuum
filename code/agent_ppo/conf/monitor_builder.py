#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Monitor panel configuration builder for the hybrid win + yjy Robot Vacuum PPO.
"""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("清扫PPO")
        .add_group(group_name="算法指标", group_name_en="algorithm")
        .add_panel(name="回报", name_en="reward", type="line")
        .add_metric(metrics_name="reward", expr="avg(reward{})")
        .end_panel()
        .add_panel(name="总损失", name_en="total_loss", type="line")
        .add_metric(metrics_name="total_loss", expr="avg(total_loss{})")
        .end_panel()
        .add_panel(name="价值损失", name_en="value_loss", type="line")
        .add_metric(metrics_name="value_loss", expr="avg(value_loss{})")
        .end_panel()
        .add_panel(name="策略损失", name_en="policy_loss", type="line")
        .add_metric(metrics_name="policy_loss", expr="avg(policy_loss{})")
        .end_panel()
        .add_panel(name="熵损失", name_en="entropy_loss", type="line")
        .add_metric(metrics_name="entropy_loss", expr="avg(entropy_loss{})")
        .end_panel()
        .add_panel(name="克隆损失", name_en="bc_loss", type="line")
        .add_metric(metrics_name="bc_loss", expr="avg(bc_loss{})")
        .end_panel()
        .add_panel(name="混合系数", name_en="mix_alpha", type="line")
        .add_metric(metrics_name="mix_alpha", expr="avg(mix_alpha{})")
        .end_panel()
        .add_panel(name="平均步数", name_en="avg_episode_steps", type="line")
        .add_metric(metrics_name="avg_episode_steps", expr="avg(avg_episode_steps{})")
        .end_panel()
        .add_panel(name="平均充电次数", name_en="avg_charge_count", type="line")
        .add_metric(metrics_name="avg_charge_count", expr="avg(avg_charge_count{})")
        .end_panel()
        .add_panel(name="到达充电桩数", name_en="charger_arrived_count", type="line")
        .add_metric(metrics_name="charger_arrived_count", expr="avg(charger_arrived_count{})")
        .end_panel()
        .add_panel(name="首桩到达步", name_en="charger_first_arrival_step", type="line")
        .add_metric(metrics_name="charger_first_arrival_step", expr="avg(charger_first_arrival_step{})")
        .end_panel()
        .add_panel(name="平均清扫分", name_en="avg_cleaned_cells", type="line")
        .add_metric(metrics_name="avg_cleaned_cells", expr="avg(avg_cleaned_cells{})")
        .end_panel()
        .add_panel(name="平均剩余电量", name_en="avg_remaining_charge", type="line")
        .add_metric(metrics_name="avg_remaining_charge", expr="avg(avg_remaining_charge{})")
        .end_panel()
        .add_panel(name="非法移动率", name_en="avg_invalid_move_rate", type="line")
        .add_metric(metrics_name="avg_invalid_move_rate", expr="avg(avg_invalid_move_rate{})")
        .end_panel()
        .add_panel(name="充电效率", name_en="avg_charge_efficiency", type="line")
        .add_metric(metrics_name="avg_charge_efficiency", expr="avg(avg_charge_efficiency{})")
        .end_panel()
        .add_panel(name="步均清扫", name_en="avg_clean_per_step", type="line")
        .add_metric(metrics_name="avg_clean_per_step", expr="avg(avg_clean_per_step{})")
        .end_panel()
        .add_panel(name="电量失败率", name_en="battery_fail_rate", type="line")
        .add_metric(metrics_name="battery_fail_rate", expr="avg(battery_fail_rate{})")
        .end_panel()
        .add_panel(name="碰撞失败率", name_en="collision_fail_rate", type="line")
        .add_metric(metrics_name="collision_fail_rate", expr="avg(collision_fail_rate{})")
        .end_panel()
        .add_panel(name="完成率", name_en="completed_rate", type="line")
        .add_metric(metrics_name="completed_rate", expr="avg(completed_rate{})")
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict

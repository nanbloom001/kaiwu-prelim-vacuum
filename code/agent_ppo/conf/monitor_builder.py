#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 漏 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Monitor panel configuration builder for Robot Vacuum.
"""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("清扫大作战")
        .add_group(group_name="算法指标", group_name_en="algorithm")
        .add_panel(name="累计回报", name_en="reward", type="line")
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
        .add_panel(name="BC 损失", name_en="bc_loss", type="line")
        .add_metric(metrics_name="bc_loss", expr="avg(bc_loss{})")
        .end_panel()
        .add_panel(name="Residual Alpha", name_en="mix_alpha", type="line")
        .add_metric(metrics_name="mix_alpha", expr="avg(mix_alpha{})")
        .end_panel()
        .add_panel(name="Episode Score", name_en="score", type="line")
        .add_metric(metrics_name="score", expr="avg(score{})")
        .end_panel()
        .add_panel(name="Local Predict", name_en="local_predict_cnt", type="line")
        .add_metric(metrics_name="local_predict_cnt", expr="avg(local_predict_cnt{})")
        .end_panel()
        .add_panel(name="Local Frames", name_en="local_frame_cnt", type="line")
        .add_metric(metrics_name="local_frame_cnt", expr="avg(local_frame_cnt{})")
        .end_panel()
        .add_panel(name="Local Yields", name_en="local_yield_cnt", type="line")
        .add_metric(metrics_name="local_yield_cnt", expr="avg(local_yield_cnt{})")
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict

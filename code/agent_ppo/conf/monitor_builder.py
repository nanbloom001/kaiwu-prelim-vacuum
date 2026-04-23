#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
精简版训练监控面板配置。

设计原则：
- 只保留关键指标
- 相关指标按 3 个一组顺序排列，便于面板一行展示
- 中文优先，必要时保留英文 metric key
"""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def _add_line_panel(builder, title_cn, metric_name, expr=None):
    panel = builder.add_panel(name=title_cn, name_en=metric_name, type="line")
    panel.add_metric(metrics_name=metric_name, expr=expr or f"avg({metric_name}{{}})")
    return panel.end_panel()


def build_monitor():
    monitor = MonitorConfigBuilder()

    builder = monitor.title("Robot Vacuum PPO").add_group(
        group_name="关键训练指标",
        group_name_en="core_training_dashboard",
    )

    panels = [
        # 结果概览
        ("完成率", "completed_rate"),
        ("电池失败率", "battery_fail_rate"),
        ("碰撞失败率", "collision_fail_rate"),
        # 效率表现
        ("平均清扫分", "avg_cleaned_cells"),
        ("平均清扫效率", "avg_clean_per_step"),
        ("胜局 CPS", "cps_win"),
        # 步数与充电
        ("平均步数", "avg_episode_steps"),
        ("平均充电次数", "avg_charge_count"),
        ("胜局充电次数", "avg_charge_count_win"),
        # 充电行为
        ("平均剩余电量", "avg_remaining_charge"),
        ("每次充电清扫分", "avg_clean_per_charge_when_charged"),
        ("零充电失败率", "zero_charge_battery_fail_rate"),
        ("电池失败中零充电占比", "zero_charge_among_battery_fail_rate"),
        # 完成质量
        ("胜局平均清扫分", "avg_clean_score_win"),
        ("完成局充电次数", "avg_charge_count_completed"),
        ("电池失败局充电次数", "avg_charge_count_battery_fail"),
        ("专家权重非零率", "expert_weight_nonzero_rate"),
        ("预返航 bias 激活率", "pre_return_bias_active_rate"),
        ("返航 bias 激活率", "return_bias_active_rate"),
        # 规划与返航
        ("规划偏离率", "planner_policy_divergence_rate"),
        ("可靠规划偏离率", "avg_reliable_planner_divergence_rate"),
        ("返航停滞率", "return_stall_rate"),
        ("路由阶段停滞率", "avg_route_phase_return_stall_rate"),
        ("返航效率比", "return_efficiency_ratio"),
        ("路由阶段动作监督激活率", "route_phase_action_teacher_active_rate"),
        ("返航切换样本数", "return_entry_count"),
        ("预返航支持返航数", "readiness_supported_return_entry_count"),
        ("预返航命中率", "pre_return_readiness_hit_rate"),
        ("预返航到返航切换率", "readiness_to_return_transition_rate"),
        ("无预返航直接返航率", "direct_return_without_readiness_rate"),
        ("风险释放奖励均值", "avg_reward_risk_release_reward"),
        ("路由阶段风险恶化惩罚均值", "avg_reward_route_risk_growth_pen"),
        ("清扫态风险恶化影子均值", "avg_reward_clean_risk_shadow"),
        ("路由阶段影子风险均值", "avg_route_phase_shadow_risk"),
        ("路由奖励就绪率", "avg_route_phase_reward_ready_rate"),
        ("充电机会成本惩罚均值", "avg_reward_charge_opp_cost_pen"),
        ("采样Anchor占比", "sampled_profile_anchor_rate"),
        ("采样Mild占比", "sampled_profile_mild_rate"),
        ("采样Broad占比", "sampled_profile_broad_rate"),
        ("电池失败正收益率", "battery_positive_reward_rate"),
        ("低价值重踩率", "clean_floor_revisit_rate"),
        ("低价值重踩惩罚均值", "clean_floor_revisit_penalty_mean"),
        ("有效覆盖奖励均值", "effective_coverage_bonus_mean"),
        # 行为结构
        ("扩张占比", "mode_usage_expand"),
        ("收缩占比", "mode_usage_contract"),
        ("返航占比", "mode_usage_return"),
        ("覆盖效率20", "avg_coverage_efficiency_20"),
        # teacher 在线观测
        ("模式监督激活率", "mode_teacher_active_rate"),
        ("锚点监督激活率", "route_anchor_teacher_active_rate"),
        ("目标监督激活率", "target_teacher_active_rate"),
        ("返航动作监督激活率", "return_action_teacher_active_rate"),
        ("路由阶段监督损失", "route_phase_policy_teacher_loss"),
        # 约束系统
        ("电池约束系数", "lambda_battery"),
        ("电池过程成本", "battery_process_cost_mean"),
        ("高需能停滞率", "high_need_return_stall_rate"),
        # 分课程胜率
        ("Anchor胜率", "anchor_win_rate"),
        ("Mild胜率", "mild_win_rate"),
        ("Broad胜率", "broad_win_rate"),
        # 课程系统
        ("课程阶段", "curriculum_stage_idx"),
        ("课程进度", "curriculum_progress"),
        ("停滞等级", "curriculum_stagnation_level"),
        # 训练数值
        ("总奖励", "reward"),
        ("总损失", "total_loss"),
        ("熵损失", "entropy_loss"),
    ]

    for title_cn, metric_name in panels:
        builder = _add_line_panel(builder, title_cn, metric_name)

    return builder.end_group().build()

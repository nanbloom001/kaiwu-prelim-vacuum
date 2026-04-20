#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import importlib
import unittest
import json
import os
import sys
import types
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _install_workflow_stubs():
    common_python_mod = types.ModuleType("common_python")
    utils_mod = types.ModuleType("common_python.utils")
    common_func_mod = types.ModuleType("common_python.utils.common_func")
    workflow_mod = types.ModuleType("common_python.utils.workflow_disaster_recovery")
    tools_mod = types.ModuleType("tools")
    metrics_utils_mod = types.ModuleType("tools.metrics_utils")
    train_env_conf_validate_mod = types.ModuleType("tools.train_env_conf_validate")

    def create_cls(name, **defaults):
        attrs = dict(defaults)

        def __init__(self, **kwargs):
            for key, default in defaults.items():
                setattr(self, key, kwargs.get(key, default))

        attrs["__init__"] = __init__
        return type(name, (), attrs)

    def handle_disaster_recovery(func=None, *args, **kwargs):
        if callable(func):
            return func

        def _decorator(inner):
            return inner

        return _decorator

    common_func_mod.create_cls = create_cls
    workflow_mod.handle_disaster_recovery = handle_disaster_recovery
    metrics_utils_mod.get_training_metrics = lambda *args, **kwargs: {}
    train_env_conf_validate_mod.read_usr_conf = lambda *args, **kwargs: {}
    utils_mod.common_func = common_func_mod
    utils_mod.workflow_disaster_recovery = workflow_mod
    common_python_mod.utils = utils_mod

    sys.modules["common_python"] = common_python_mod
    sys.modules["common_python.utils"] = utils_mod
    sys.modules["common_python.utils.common_func"] = common_func_mod
    sys.modules["common_python.utils.workflow_disaster_recovery"] = workflow_mod
    sys.modules["tools"] = tools_mod
    sys.modules["tools.metrics_utils"] = metrics_utils_mod
    sys.modules["tools.train_env_conf_validate"] = train_env_conf_validate_mod


_install_workflow_stubs()


class CurriculumAndCheckpointScoreTests(unittest.TestCase):
    def test_reward_contribution_payload_balances_positive_and_negative_components(self):
        from agent_ppo.utils.reward_metrics import compute_reward_contribution_payload

        payload = compute_reward_contribution_payload({
            "cleaning": 0.06,
            "streak": 0.02,
            "explore": 0.04,
            "charge_route_progress_bonus": 0.02,
            "necessary_charge_bonus": 0.02,
            "frontier": 0.0,
            "cps_bonus": 0.0,
            "edge_follow_bonus": 0.02,
            "charger_access_discovery_bonus": 0.01,
            "charger_access_probe_bonus": 0.01,
            "charge_detour_cost": -0.03,
            "charge_interrupt_cost": -0.01,
            "skip_needed_charge_penalty": -0.01,
            "unnecessary_charge_penalty": 0.0,
            "planner_alignment": -0.005,
            "idle": -0.005,
            "npc": 0.0,
            "coverage_tangle_penalty": -0.01,
        })

        self.assertAlmostEqual(payload["reward_positive_total"], 0.20, places=5)
        self.assertAlmostEqual(payload["reward_negative_total"], 0.07, places=5)
        self.assertAlmostEqual(payload["reward_net_total"], 0.13, places=5)
        self.assertAlmostEqual(payload["reward_positive_share_cleaning"], 0.3, places=5)
        self.assertAlmostEqual(payload["reward_positive_share_explore"], 0.2, places=5)
        self.assertAlmostEqual(payload["reward_positive_share_necessary_charge_bonus"], 0.1, places=5)
        self.assertAlmostEqual(payload["reward_positive_share_edge_follow_bonus"], 0.1, places=5)
        self.assertAlmostEqual(payload["reward_positive_share_charger_access_discovery_bonus"], 0.05, places=5)
        self.assertAlmostEqual(payload["reward_positive_share_charger_access_probe_bonus"], 0.05, places=5)
        self.assertAlmostEqual(payload["reward_negative_share_charge_detour_cost"], 3.0 / 7.0, places=5)
        self.assertAlmostEqual(payload["reward_negative_share_coverage_tangle_penalty"], 1.0 / 7.0, places=5)
        self.assertAlmostEqual(payload["reward_charging_positive_total"], 0.06, places=5)
        self.assertAlmostEqual(payload["reward_charging_negative_total"], 0.05, places=5)
        self.assertAlmostEqual(payload["reward_charging_positive_share_charge_route_progress_bonus"], 1.0 / 3.0, places=5)
        self.assertAlmostEqual(payload["reward_charging_positive_share_necessary_charge_bonus"], 1.0 / 3.0, places=5)
        self.assertAlmostEqual(payload["reward_charging_positive_share_charger_access_discovery_bonus"], 1.0 / 6.0, places=5)
        self.assertAlmostEqual(payload["reward_charging_positive_share_charger_access_probe_bonus"], 1.0 / 6.0, places=5)

    def test_reward_contribution_payload_accounts_for_return_progress_and_stall_components(self):
        from agent_ppo.utils.reward_metrics import compute_reward_contribution_payload

        payload = compute_reward_contribution_payload({
            "charge_route_progress_bonus": 0.02,
            "return_progress_shaping_bonus": 0.03,
            "necessary_charge_bonus": 0.01,
            "charger_access_discovery_bonus": 0.0,
            "charger_access_probe_bonus": 0.0,
            "charge_detour_cost": -0.01,
            "charge_interrupt_cost": -0.01,
            "skip_needed_charge_penalty": -0.02,
            "high_need_return_stall_penalty": -0.03,
            "unnecessary_charge_penalty": 0.0,
            "cleaning": 0.0,
            "streak": 0.0,
            "explore": 0.0,
            "frontier": 0.0,
            "cps_bonus": 0.0,
            "edge_follow_bonus": 0.0,
            "planner_alignment": 0.0,
            "idle": 0.0,
            "npc": 0.0,
            "coverage_tangle_penalty": 0.0,
        })

        self.assertAlmostEqual(payload["reward_charging_positive_total"], 0.06, places=5)
        self.assertAlmostEqual(payload["reward_charging_negative_total"], 0.07, places=5)
        self.assertAlmostEqual(payload["reward_charging_positive_share_return_progress_shaping_bonus"], 0.5, places=5)
        self.assertAlmostEqual(payload["reward_negative_share_high_need_return_stall_penalty"], 3.0 / 7.0, places=5)
        self.assertAlmostEqual(payload["reward_charging_negative_share_high_need_return_stall_penalty"], 3.0 / 7.0, places=5)

    def test_recent_episode_metrics_use_charged_only_efficiency_and_zero_charge_fail_rate(self):
        from agent_ppo.workflow.curriculum_state import _aggregate_episode_records

        records = [
            {
                "result": "completed",
                "clean_score": 300.0,
                "finished_steps": 1000.0,
                "charge_count": 6.0,
                "remaining_charge": 50.0,
                "invalid_move_rate": 0.0,
                "charge_efficiency": 50.0,
                "clean_per_charge_when_charged": 50.0,
                "clean_per_step": 0.30,
                "expert_weight": 0.0,
                "avg_reward_explore": 0.04,
                "avg_reward_edge_follow_bonus": 0.02,
                "avg_reward_charger_access_discovery_bonus": 0.01,
                "avg_reward_charger_access_probe_bonus": 0.00,
                "avg_reward_coverage_tangle_penalty": -0.01,
                "profile": "mild",
            },
            {
                "result": "battery",
                "clean_score": 200.0,
                "finished_steps": 400.0,
                "charge_count": 0.0,
                "remaining_charge": 0.0,
                "total_reward": 5.0,
                "invalid_move_rate": 0.0,
                "charge_efficiency": 200.0,
                "clean_per_charge_when_charged": None,
                "clean_per_step": 0.50,
                "expert_weight": 0.0,
                "avg_reward_explore": 0.02,
                "avg_reward_edge_follow_bonus": 0.01,
                "avg_reward_charger_access_discovery_bonus": 0.03,
                "avg_reward_charger_access_probe_bonus": 0.02,
                "avg_reward_coverage_tangle_penalty": -0.02,
                "profile": "anchor",
            },
            {
                "result": "battery",
                "clean_score": 180.0,
                "finished_steps": 500.0,
                "charge_count": 2.0,
                "remaining_charge": 0.0,
                "total_reward": -3.0,
                "invalid_move_rate": 0.0,
                "charge_efficiency": 90.0,
                "clean_per_charge_when_charged": 90.0,
                "clean_per_step": 0.36,
                "expert_weight": 0.0,
                "avg_reward_explore": 0.01,
                "avg_reward_edge_follow_bonus": 0.03,
                "avg_reward_charger_access_discovery_bonus": 0.00,
                "avg_reward_charger_access_probe_bonus": 0.00,
                "avg_reward_coverage_tangle_penalty": -0.03,
                "profile": "anchor",
            },
        ]

        metrics = _aggregate_episode_records(records, min_episode_count=1)
        self.assertAlmostEqual(metrics["avg_charge_efficiency"], (50.0 + 200.0 + 90.0) / 3.0, places=5)
        self.assertAlmostEqual(metrics["avg_clean_per_charge_when_charged"], 70.0, places=5)
        self.assertAlmostEqual(metrics["zero_charge_battery_fail_rate"], 0.5, places=5)
        self.assertEqual(metrics["battery_fail_count"], 2)
        self.assertEqual(metrics["battery_positive_reward_count"], 1)
        self.assertAlmostEqual(metrics["battery_positive_reward_rate"], 0.5, places=5)
        self.assertAlmostEqual(metrics["avg_charge_count_completed"], 6.0, places=5)
        self.assertAlmostEqual(metrics["avg_charge_count_battery_fail"], 1.0, places=5)
        self.assertAlmostEqual(metrics["avg_reward_explore"], (0.04 + 0.02 + 0.01) / 3.0, places=5)
        self.assertAlmostEqual(metrics["avg_reward_edge_follow_bonus"], 0.02, places=5)
        self.assertAlmostEqual(metrics["avg_reward_charger_access_discovery_bonus"], (0.01 + 0.03 + 0.0) / 3.0, places=5)
        self.assertAlmostEqual(metrics["avg_reward_charger_access_probe_bonus"], (0.0 + 0.02 + 0.0) / 3.0, places=5)
        self.assertAlmostEqual(metrics["avg_reward_coverage_tangle_penalty"], -0.02, places=5)

    def test_recent_episode_metrics_include_new_return_reward_components(self):
        from agent_ppo.workflow.curriculum_state import _aggregate_episode_records

        records = [
            {
                "result": "completed",
                "clean_score": 100.0,
                "finished_steps": 100.0,
                "charge_count": 1.0,
                "remaining_charge": 50.0,
                "invalid_move_rate": 0.0,
                "charge_efficiency": 100.0,
                "clean_per_charge_when_charged": 100.0,
                "clean_per_step": 1.0,
                "expert_weight": 0.0,
                "late_return_rate": 0.0,
                "late_contract_rate": 0.0,
                "anchor_switch_rate": 0.0,
                "target_switch_rate": 0.0,
                "diag_rate_all": 0.0,
                "diag_rate_contract": 0.0,
                "diag_rate_return": 0.0,
                "return_progress_per_step": 0.2,
                "return_efficiency_ratio": 0.5,
                "return_stall_rate": 0.0,
                "recoverability_score_avg": 0.0,
                "recoverability_violation_rate": 0.0,
                "wall_hugging_clean_floor_rate": 0.0,
                "stale_boundary_follow_rate": 0.0,
                "narrow_unknown_commit_rate": 0.0,
                "missed_charge_opportunity_rate": 0.0,
                "charger_nearby_not_charged_rate": 0.0,
                "suboptimal_target_hold_rate": 0.0,
                "planner_policy_divergence_rate": 0.0,
                "avg_path_cross_count_50": 0.0,
                "avg_coverage_efficiency_20": 0.0,
                "avg_all_charger_known_path_count": 0.0,
                "avg_unknown_on_target_path_ratio": 0.0,
                "avg_planner_topk_reachable_count": 0.0,
                "avg_planner_known_route_count_total": 0.0,
                "avg_planner_best_target_route_diversity": 0.0,
                "avg_planner_best_target_tangle_cost": 0.0,
                "avg_planner_best_target_edge_break_cost": 0.0,
                "avg_planner_best_target_region_fragment_cost": 0.0,
                "avg_planner_multi_route_recoverability": 0.0,
                "battery_process_cost_mean": 0.0,
                "collision_process_cost_mean": 0.0,
                "high_need_return_stall_rate": 0.1,
                "avg_charge_need_score": 0.0,
                "avg_slack_confidence": 0.0,
                "battery_fail_severity": 0.0,
                "mode_usage_depart": 0.0,
                "mode_usage_expand": 0.0,
                "mode_usage_harvest": 0.0,
                "mode_usage_contract": 0.0,
                "mode_usage_return": 0.0,
                "mode_usage_evade": 0.0,
                "avg_reward_cleaning": 0.0,
                "avg_reward_streak": 0.0,
                "avg_reward_explore": 0.0,
                "avg_reward_frontier": 0.0,
                "avg_reward_charger_access_discovery_bonus": 0.0,
                "avg_reward_charger_access_probe_bonus": 0.0,
                "avg_reward_idle": 0.0,
                "avg_reward_npc": 0.0,
                "avg_reward_planner_alignment": 0.0,
                "avg_reward_charge_route_progress_bonus": 0.02,
                "avg_reward_return_progress_shaping_bonus": 0.03,
                "avg_reward_necessary_charge_bonus": 0.01,
                "avg_reward_unnecessary_charge_penalty": 0.0,
                "avg_reward_charge_detour_cost": -0.01,
                "avg_reward_charge_interrupt_cost": -0.01,
                "avg_reward_skip_needed_charge_penalty": -0.02,
                "avg_reward_high_need_return_stall_penalty": -0.03,
                "avg_reward_cps_bonus": 0.0,
                "avg_reward_coverage_tangle_penalty": 0.0,
                "avg_reward_edge_follow_bonus": 0.0,
            }
        ]

        metrics = _aggregate_episode_records(records, min_episode_count=1)
        self.assertAlmostEqual(metrics["avg_reward_return_progress_shaping_bonus"], 0.03, places=5)
        self.assertAlmostEqual(metrics["avg_reward_high_need_return_stall_penalty"], -0.03, places=5)
        self.assertAlmostEqual(metrics["reward_positive_share_return_progress_shaping_bonus"], 0.5, places=5)
        self.assertAlmostEqual(metrics["reward_negative_share_high_need_return_stall_penalty"], 3.0 / 7.0, places=5)

    def test_constraint_utils_compute_confidence_need_and_severity(self):
        from agent_ppo.utils.constraint_utils import (
            classify_battery_fail_severity,
            classify_battery_state,
            compute_battery_process_cost_step,
            compute_charge_need_score,
            compute_collision_process_cost_step,
            compute_slack_confidence,
            has_known_charge_route,
        )

        self.assertTrue(has_known_charge_route(1, True))
        self.assertFalse(has_known_charge_route(0, True))

        confidence = compute_slack_confidence(2, True, True, 0.1)
        self.assertAlmostEqual(confidence, 1.0, places=5)

        need_known = compute_charge_need_score(True, charge_margin_now=0.0, battery_ratio=0.10, future_recoverability_score=-0.5)
        self.assertGreaterEqual(need_known, 0.7)
        self.assertEqual(classify_battery_state(0.1), "safe")
        self.assertEqual(classify_battery_state(0.2), "planning")
        self.assertEqual(classify_battery_state(0.9), "critical")

        cost, need_score, state = compute_battery_process_cost_step(
            has_known_route=True,
            charger_slack=-6.0,
            slack_confidence=0.8,
            charge_margin_now=0.0,
            battery_ratio=0.12,
            future_recoverability_score=-0.4,
            high_need_stall_indicator=1.0,
        )
        self.assertEqual(state, "critical")
        self.assertGreater(need_score, 0.5)
        self.assertGreater(cost, 0.1)

        no_route_cost, _, no_route_state = compute_battery_process_cost_step(
            has_known_route=False,
            charger_slack=0.0,
            slack_confidence=0.0,
            charge_margin_now=20.0,
            battery_ratio=0.15,
            future_recoverability_score=-0.2,
            high_need_stall_indicator=0.0,
        )
        self.assertIn(no_route_state, {"planning", "critical"})
        self.assertGreaterEqual(no_route_cost, 0.0)

        fail_type, severity = classify_battery_fail_severity(
            fail_reason="battery",
            finished_steps=200.0,
            max_step=1000.0,
            clean_per_step=0.10,
            all_charger_known_path_count=0.0,
            avg_unknown_on_target_path_ratio=0.8,
            remaining_charge=0.0,
        )
        self.assertEqual(fail_type, "early_unrecoverable")
        self.assertAlmostEqual(severity, 1.0, places=5)

        late_type, late_severity = classify_battery_fail_severity(
            fail_reason="battery",
            finished_steps=950.0,
            max_step=1000.0,
            clean_per_step=0.80,
            all_charger_known_path_count=2.0,
            avg_unknown_on_target_path_ratio=0.05,
            remaining_charge=0.0,
            charge_count=2.0,
        )
        self.assertEqual(late_type, "late_near_completion")
        self.assertAlmostEqual(late_severity, 0.25, places=5)

        mid_type, mid_severity = classify_battery_fail_severity(
            fail_reason="battery",
            finished_steps=890.0,
            max_step=1000.0,
            clean_per_step=0.74,
            all_charger_known_path_count=2.0,
            avg_unknown_on_target_path_ratio=0.05,
            remaining_charge=0.0,
            charge_count=0.0,
        )
        self.assertEqual(mid_type, "mid_recoverability_loss")
        self.assertAlmostEqual(mid_severity, 0.6, places=5)
        self.assertAlmostEqual(compute_collision_process_cost_step(1.0), 0.15, places=5)

    def test_policy_sampling_sanitizes_nan_and_illegal_probs(self):
        from agent_ppo.utils.policy_sampling import sanitize_policy_probs, safe_sample_action

        legal = [1, 0, 1, 0]
        probs, used_fallback = sanitize_policy_probs([float("nan"), 0.8, -0.3, 0.2], legal)
        self.assertTrue(used_fallback)
        self.assertAlmostEqual(sum(probs), 1.0, places=5)
        self.assertEqual(probs[1], 0.0)
        self.assertEqual(probs[3], 0.0)

        sampled = safe_sample_action([float("nan"), 0.8, -0.3, 0.2], legal, use_max=False, rng_seed=7)
        self.assertIn(sampled["action"], [0, 2])
        self.assertTrue(sampled["used_fallback"])
        self.assertAlmostEqual(sum(sampled["probs"]), 1.0, places=5)

    def test_checkpoint_scoring_prefers_behavioral_health_and_cps_over_raw_clean_score(self):
        from agent_ppo.workflow.checkpoint_score import compute_checkpoint_scores

        healthy_window = {
            "_count": 24,
            "win_rate": 0.78,
            "battery_fail_rate": 0.08,
            "collision_fail_rate": 0.03,
            "late_return_rate": 0.04,
            "return_stall_rate": 0.22,
            "avg_clean_per_step": 0.88,
            "cps_win": 0.97,
            "late_contract_rate": 0.05,
            "recoverability_violation_rate": 0.09,
            "wall_hugging_clean_floor_rate": 0.03,
            "stale_boundary_follow_rate": 0.02,
            "narrow_unknown_commit_rate": 0.05,
            "missed_charge_opportunity_rate": 0.01,
            "suboptimal_target_hold_rate": 0.03,
            "planner_policy_divergence_rate": 0.18,
            "avg_clean_score": 760.0,
        }
        unhealthy_window = {
            **healthy_window,
            "avg_clean_score": 1180.0,
            "avg_clean_per_step": 0.54,
            "cps_win": 0.58,
            "battery_fail_rate": 0.18,
            "return_stall_rate": 0.49,
            "wall_hugging_clean_floor_rate": 0.14,
            "suboptimal_target_hold_rate": 0.16,
            "planner_policy_divergence_rate": 0.39,
        }
        learning = {
            "entropy_loss": 0.78,
            "value_clean_loss_trend_ratio": 0.96,
            "value_survive_loss_trend_ratio": 0.97,
            "mode_teacher_active_rate": 0.50,
            "route_anchor_teacher_active_rate": 0.82,
            "target_teacher_active_rate": 0.83,
            "return_action_teacher_active_rate": 0.06,
            "env_total_score": 880.0,
            "env_total_score_trend_ratio": 1.02,
        }

        healthy = compute_checkpoint_scores(healthy_window, learning)
        unhealthy = compute_checkpoint_scores(unhealthy_window, learning)

        self.assertIn("resume_score_behavior", healthy)
        self.assertIn("resume_score_safety", healthy)
        self.assertIn("submission_score_completion", healthy)
        self.assertGreater(healthy["resume_readiness_score"], unhealthy["resume_readiness_score"])
        self.assertGreater(healthy["checkpoint_preservation_score"], unhealthy["checkpoint_preservation_score"])
        self.assertGreater(healthy["resume_score_behavior"], unhealthy["resume_score_behavior"])
        self.assertGreater(healthy["resume_score_safety"], unhealthy["resume_score_safety"])

    def test_curriculum_fast_skip_and_regression_rules(self):
        from agent_ppo.workflow.curriculum_policy import choose_stage, choose_stage_decision, should_regress_stage, curriculum_gate_ratios

        bootstrap_metrics = {
            "_count": 10,
            "win_rate": 0.78,
            "battery_fail_rate": 0.05,
            "return_stall_rate": 0.22,
            "wall_hugging_clean_floor_rate": 0.03,
            "suboptimal_target_hold_rate": 0.03,
            "broad_win_rate": 0.67,
            "planner_policy_divergence_rate": 0.18,
        }
        fast_context = {
            "global_step_since_resume": 3200,
            "bootstrap_metrics": bootstrap_metrics,
            "window_metrics": None,
            "learning_metrics": {"entropy_loss": 0.84, "entropy_trend_ratio": 1.0},
            "resume_fast_track": True,
        }
        stage = choose_stage("warmup", fast_context, None)
        self.assertEqual(stage, "robust")

        soft_context = {
            "global_step_since_resume": 15000,
            "entered_global_step": 0,
            "window_metrics": {
                "_count": 40,
                "win_rate": 0.86,
                "battery_fail_rate": 0.05,
                "collision_fail_rate": 0.04,
                "avg_clean_per_step": 0.72,
                "avg_charge_count": 4.7,
                "avg_coverage_efficiency_20": 0.85,
                "return_stall_rate": 0.60,
                "planner_policy_divergence_rate": 0.88,
            },
            "bootstrap_metrics": None,
            "learning_metrics": {"entropy_loss": 0.88, "entropy_trend_ratio": 1.0},
            "resume_fast_track": False,
        }
        soft_decision = choose_stage_decision("warmup", soft_context, None)
        self.assertEqual(soft_decision["proposed_stage"], "blend")
        self.assertEqual(soft_decision["promotion_reason"], "soft_gate")

        timeout_context = {
            "global_step_since_resume": 28000,
            "entered_global_step": 0,
            "window_metrics": {
                "_count": 40,
                "win_rate": 0.74,
                "battery_fail_rate": 0.12,
                "collision_fail_rate": 0.05,
                "avg_clean_per_step": 0.60,
                "return_stall_rate": 0.72,
                "planner_policy_divergence_rate": 0.94,
            },
            "bootstrap_metrics": None,
            "learning_metrics": {"entropy_loss": 0.95, "entropy_trend_ratio": 1.03},
            "resume_fast_track": False,
        }
        timeout_decision = choose_stage_decision("warmup", timeout_context, None)
        self.assertEqual(timeout_decision["proposed_stage"], "blend")
        self.assertEqual(timeout_decision["promotion_reason"], "timeout_gate")

        entry_metrics = {
            "battery_fail_rate": 0.06,
            "return_stall_rate": 0.22,
            "env_total_score": 910.0,
        }
        current_metrics = {
            "battery_fail_rate": 0.14,
            "return_stall_rate": 0.34,
        }
        learning_metrics = {
            "entropy_loss": 1.01,
            "env_total_score": 780.0,
        }
        self.assertTrue(should_regress_stage("robust", entry_metrics, current_metrics, learning_metrics))

        gate = curriculum_gate_ratios(
            stage="warmup",
            metrics={"return_stall_rate": 0.05},
            learning_metrics={"entropy_loss": 0.8},
            global_step_since_resume=1200,
            entered_global_step=0,
        )
        self.assertAlmostEqual(gate["curriculum_gate_global_step_ratio"], 0.4, places=4)
        self.assertAlmostEqual(gate["curriculum_gate_return_stall_ratio_raw"], 8.0, places=4)
        self.assertAlmostEqual(gate["curriculum_gate_return_stall_ratio"], 2.0, places=4)
        self.assertAlmostEqual(gate["curriculum_return_stall_margin"], -0.35, places=4)

    def test_checkpoint_scoring_tolerates_missing_learning_fields(self):
        from agent_ppo.workflow.checkpoint_score import compute_checkpoint_scores

        window = {
            "_count": 24,
            "win_rate": 0.71,
            "battery_fail_rate": 0.10,
            "collision_fail_rate": 0.04,
            "return_stall_rate": 0.28,
            "avg_clean_per_step": 0.81,
            "cps_win": 0.89,
        }
        learning = {
            "entropy_loss": None,
            "route_anchor_teacher_active_rate": None,
            "target_teacher_active_rate": None,
            "value_clean_loss_trend_ratio": None,
            "env_total_score": 850.0,
            "env_total_score_trend_ratio": 1.0,
        }

        scores = compute_checkpoint_scores(window, learning)
        self.assertIn("checkpoint_preservation_score", scores)
        self.assertIsInstance(scores["checkpoint_preservation_score"], float)

    def test_shared_curriculum_state_uses_global_aggregation_to_avoid_too_slow_or_too_fast(self):
        from agent_ppo.workflow.curriculum_state import SharedCurriculumStateStore

        with TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"KAIWU_TRAINING_START_MODE": "scratch", "KAIWU_PRELOAD_MODEL": "0"},
            clear=False,
        ):
            store = SharedCurriculumStateStore(Path(tmp))

            strong_payload = {
                "window_metrics": {
                    "_count": 8,
                    "win_rate": 0.90,
                    "battery_fail_rate": 0.00,
                    "collision_fail_rate": 0.00,
                    "return_stall_rate": 0.18,
                    "zero_charge_battery_fail_rate": 0.0,
                    "wall_hugging_clean_floor_rate": 0.02,
                    "suboptimal_target_hold_rate": 0.02,
                    "planner_policy_divergence_rate": 0.15,
                    "broad_win_rate": 0.70,
                },
                "bootstrap_metrics": {
                    "_count": 8,
                    "win_rate": 0.90,
                    "battery_fail_rate": 0.00,
                    "return_stall_rate": 0.18,
                    "zero_charge_battery_fail_rate": 0.0,
                    "wall_hugging_clean_floor_rate": 0.02,
                    "suboptimal_target_hold_rate": 0.02,
                    "planner_policy_divergence_rate": 0.15,
                    "broad_win_rate": 0.70,
                },
                "learning_metrics": {
                    "entropy_loss": 0.82,
                    "entropy_trend_ratio": 1.0,
                    "env_total_score": 880.0,
                },
                "runtime": {"global_step_since_resume": 3500},
            }
            store.write_signal("helper-a", strong_payload)
            state = store.refresh_state()
            self.assertEqual(state["stage"], "warmup")

            helper_b = {
                **strong_payload,
                "window_metrics": {**strong_payload["window_metrics"], "_count": 12},
                "bootstrap_metrics": {**strong_payload["bootstrap_metrics"], "_count": 12},
            }
            store.write_signal("helper-b", helper_b)
            state = store.refresh_state()
            self.assertEqual(state["stage"], "warmup")
            state = store.refresh_state()
            self.assertEqual(state["stage"], "robust")

    def test_shared_curriculum_state_records_promotion_reason_and_transition_guard(self):
        from agent_ppo.workflow.curriculum_state import SharedCurriculumStateStore

        with TemporaryDirectory() as tmp:
            store = SharedCurriculumStateStore(Path(tmp))

            learning = {
                "entropy_loss": 0.88,
                "entropy_trend_ratio": 1.0,
                "env_total_score": 880.0,
                "global_step": 18000.0,
            }

            def make_episode(local_ep):
                return {
                    "episode_cnt_local": local_ep,
                    "result": "completed",
                    "profile": "mild",
                    "clean_score": 760.0,
                    "finished_steps": 1000.0,
                    "charge_count": 5.0,
                    "remaining_charge": 210.0,
                    "invalid_move_rate": 0.0,
                    "charge_efficiency": 160.0,
                    "clean_per_charge_when_charged": 160.0,
                    "clean_per_step": 0.72,
                    "expert_weight": 0.0,
                    "late_return_rate": 0.02,
                    "late_contract_rate": 0.0,
                    "anchor_switch_rate": 0.0,
                    "target_switch_rate": 0.0,
                    "diag_rate_all": 0.2,
                    "diag_rate_contract": 0.18,
                    "diag_rate_return": 0.16,
                    "return_progress_per_step": 0.08,
                    "return_efficiency_ratio": 0.18,
                    "return_stall_rate": 0.60,
                    "recoverability_score_avg": 0.90,
                    "recoverability_violation_rate": 0.03,
                    "wall_hugging_clean_floor_rate": 0.02,
                    "stale_boundary_follow_rate": 0.01,
                    "narrow_unknown_commit_rate": 0.01,
                    "missed_charge_opportunity_rate": 0.0,
                    "charger_nearby_not_charged_rate": 0.0,
                    "suboptimal_target_hold_rate": 0.01,
                    "planner_policy_divergence_rate": 0.88,
                    "avg_path_cross_count_50": 6.0,
                    "avg_coverage_efficiency_20": 0.85,
                    "avg_all_charger_known_path_count": 2.5,
                    "avg_unknown_on_target_path_ratio": 0.08,
                    "mode_usage_depart": 0.0,
                    "mode_usage_expand": 0.03,
                    "mode_usage_harvest": 0.45,
                    "mode_usage_contract": 0.42,
                    "mode_usage_return": 0.10,
                    "mode_usage_evade": 0.0,
                }

            payload = {
                "window_metrics": {},
                "bootstrap_metrics": {},
                "learning_metrics": learning,
                "runtime": {"global_step_since_resume": 18000},
                "recent_episode_metrics": [make_episode(i) for i in range(1, 41)],
            }
            store.write_signal("helper-a", payload)

            first = store.refresh_state()
            self.assertEqual(first["stage"], "warmup")
            second = store.refresh_state()
            self.assertEqual(second["stage"], "blend")
            self.assertEqual(second["last_promotion_reason"], "soft_gate")
            self.assertTrue(second["in_transition_guard"])
            self.assertEqual(second["transition_target_stage"], "blend")

    def test_shared_curriculum_state_resume_seed_restores_snapshot_and_learning_metrics(self):
        from agent_ppo.workflow.curriculum_state import SharedCurriculumStateStore
        from agent_ppo.workflow.state_layout import ensure_runtime_state_dirs

        with TemporaryDirectory() as tmp:
            code_dir = Path(tmp)
            layout = ensure_runtime_state_dirs(code_dir)
            snapshot_path = layout.current.prepared_resume_dir / "curriculum_state.snapshot.json"
            state_meta_path = layout.current.prepared_resume_dir / "resume.state.json"
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_path.write_text(
                json.dumps(
                    {
                        "stage": "blend",
                        "stage_version": 3,
                        "entered_global_step": 12000,
                        "last_global_metrics": {"win_rate": 0.88},
                        "last_bootstrap_metrics": {"win_rate": 0.90},
                        "last_learning_metrics": {"global_step": 191120, "lambda_battery": 0.6},
                        "in_transition_guard": True,
                        "transition_target_stage": "blend",
                        "transition_entered_global_step": 12000,
                        "source_session_id": "old-session",
                    },
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )
            state_meta_path.write_text(
                json.dumps(
                    {
                        "global_step": 191120,
                        "global_step_since_resume": 191120,
                        "session_id": "old-session",
                        "curriculum_state_snapshot_path": str(snapshot_path),
                        "last_learning_metrics": {"global_step": 191120, "lambda_battery": 0.7},
                    },
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )
            store = SharedCurriculumStateStore(code_dir)
            with patch.dict(os.environ, {"KAIWU_TRAINING_START_MODE": "resume"}, clear=False):
                state = store.seed_initial_state("new-session", "warmup")
            self.assertEqual(state["stage"], "blend")
            self.assertEqual(state["source_session_id"], "new-session")
            self.assertEqual(state["restored_from_session_id"], "old-session")
            self.assertEqual(state["global_step_since_resume"], 191120)
            self.assertAlmostEqual(state["last_learning_metrics"]["lambda_battery"], 0.7, places=5)

    def test_shared_curriculum_state_learning_metrics_ignore_null_overwrite(self):
        from agent_ppo.workflow.curriculum_state import SharedCurriculumStateStore

        with TemporaryDirectory() as tmp:
            store = SharedCurriculumStateStore(Path(tmp))
            store.seed_initial_state("sess-a", "warmup")
            store.write_signal(
                "helper-a",
                {
                    "session_id": "sess-a",
                    "window_metrics": {},
                    "bootstrap_metrics": {},
                    "learning_metrics": {"global_step": 100.0, "lambda_battery": 0.6, "last_finite_step": 99.0},
                    "runtime": {"global_step_since_resume": 100},
                },
            )
            state = store.refresh_state()
            self.assertAlmostEqual(state["last_learning_metrics"]["lambda_battery"], 0.6, places=5)

            store.write_signal(
                "helper-b",
                {
                    "session_id": "sess-a",
                    "window_metrics": {},
                    "bootstrap_metrics": {},
                    "learning_metrics": {"global_step": 101.0, "lambda_battery": None, "last_finite_step": None},
                    "runtime": {"global_step_since_resume": 101},
                },
            )
            state = store.refresh_state()
            self.assertAlmostEqual(state["last_learning_metrics"]["lambda_battery"], 0.6, places=5)
            self.assertAlmostEqual(state["last_learning_metrics"]["last_finite_step"], 99.0, places=5)
            self.assertAlmostEqual(state["last_learning_metrics"]["global_step"], 101.0, places=5)

    def test_shared_curriculum_state_learning_metrics_fallback_to_learner_log_global_step(self):
        from agent_ppo.workflow.curriculum_state import SharedCurriculumStateStore

        with TemporaryDirectory() as tmp:
            code_dir = Path(tmp) / "code"
            learner_log_dir = Path(tmp) / "train" / "log" / "learner"
            code_dir.mkdir(parents=True, exist_ok=True)
            learner_log_dir.mkdir(parents=True, exist_ok=True)
            (learner_log_dir / "learner_train_pid1_log_2026-04-20-14.log").write_text(
                '{"time":"2026-04-20 14:51:17","message":"learner train process now input ready size is 1024, train process now train count is 587, global step is 587, train once cost time is 99.67 ms"}\n',
                encoding="utf-8",
            )

            store = SharedCurriculumStateStore(code_dir)
            store.seed_initial_state("sess-a", "warmup")
            store.write_signal(
                "helper-a",
                {
                    "session_id": "sess-a",
                    "window_metrics": {},
                    "bootstrap_metrics": {},
                    "learning_metrics": {"global_step": 0.0, "lambda_battery": 0.6},
                    "runtime": {"global_step_since_resume": 0},
                },
            )
            state = store.refresh_state()
            self.assertAlmostEqual(state["last_learning_metrics"]["global_step"], 587.0, places=5)
            self.assertAlmostEqual(state["last_learning_metrics"]["lambda_battery"], 0.6, places=5)

    def test_shared_curriculum_state_learning_metrics_fallback_to_container_style_learner_log(self):
        from agent_ppo.workflow.curriculum_state import SharedCurriculumStateStore

        with TemporaryDirectory() as tmp:
            code_dir = Path(tmp) / "code"
            learner_log_dir = Path(tmp) / "log" / "learner"
            code_dir.mkdir(parents=True, exist_ok=True)
            learner_log_dir.mkdir(parents=True, exist_ok=True)
            (learner_log_dir / "learner_train_pid2_log_2026-04-20-15.log").write_text(
                '{"time":"2026-04-20 15:01:17","message":"learner train process now input ready size is 1024, train process now train count is 888, global step is 888, train once cost time is 80.00 ms"}\n',
                encoding="utf-8",
            )

            store = SharedCurriculumStateStore(code_dir)
            store.seed_initial_state("sess-a", "warmup")
            store.write_signal(
                "helper-a",
                {
                    "session_id": "sess-a",
                    "window_metrics": {},
                    "bootstrap_metrics": {},
                    "learning_metrics": {"global_step": 0.0},
                    "runtime": {"global_step_since_resume": 0},
                },
            )
            state = store.refresh_state()
            self.assertAlmostEqual(state["last_learning_metrics"]["global_step"], 888.0, places=5)

    def test_shared_curriculum_state_builds_full_window_from_global_recent_episodes(self):
        from agent_ppo.workflow.curriculum_state import SharedCurriculumStateStore

        with TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"KAIWU_TRAINING_START_MODE": "scratch", "KAIWU_PRELOAD_MODEL": "0"},
            clear=False,
        ):
            store = SharedCurriculumStateStore(Path(tmp))

            learning = {
                "entropy_loss": 0.80,
                "entropy_trend_ratio": 1.0,
                "env_total_score": 900.0,
                "global_step": 6200.0,
            }

            def make_episode(local_ep, fail_reason="completed", return_stall=0.22):
                return {
                    "episode_cnt_local": local_ep,
                    "result": fail_reason,
                    "profile": "mild",
                    "clean_score": 820.0,
                    "finished_steps": 1000.0,
                    "charge_count": 4.0,
                    "remaining_charge": 220.0,
                    "invalid_move_rate": 0.0,
                    "charge_efficiency": 205.0,
                    "clean_per_step": 0.82,
                    "expert_weight": 0.0,
                    "late_return_rate": 0.02,
                    "late_contract_rate": 0.01,
                    "anchor_switch_rate": 0.0,
                    "target_switch_rate": 0.0,
                    "diag_rate_all": 0.20,
                    "diag_rate_contract": 0.24,
                    "diag_rate_return": 0.18,
                    "return_progress_per_step": 0.21,
                    "return_efficiency_ratio": 0.55,
                    "return_stall_rate": return_stall,
                    "recoverability_score_avg": 0.90,
                    "recoverability_violation_rate": 0.02,
                    "wall_hugging_clean_floor_rate": 0.02,
                    "stale_boundary_follow_rate": 0.01,
                    "narrow_unknown_commit_rate": 0.02,
                    "missed_charge_opportunity_rate": 0.0,
                    "charger_nearby_not_charged_rate": 0.0,
                    "suboptimal_target_hold_rate": 0.01,
                    "planner_policy_divergence_rate": 0.15,
                    "avg_path_cross_count_50": 1.5,
                    "avg_coverage_efficiency_20": 0.88,
                    "avg_all_charger_known_path_count": 2.5,
                    "avg_unknown_on_target_path_ratio": 0.08,
                    "mode_usage_depart": 0.0,
                    "mode_usage_expand": 0.08,
                    "mode_usage_harvest": 0.55,
                    "mode_usage_contract": 0.22,
                    "mode_usage_return": 0.12,
                    "mode_usage_evade": 0.03,
                }

            for helper_idx in range(4):
                payload = {
                    "window_metrics": {},
                    "bootstrap_metrics": {},
                    "learning_metrics": learning,
                    "runtime": {"global_step_since_resume": 6200},
                    "recent_episode_metrics": [
                        make_episode(helper_idx * 10 + offset) for offset in range(1, 11)
                    ],
                }
                store.write_signal(f"helper-{helper_idx}", payload)

            first = store.refresh_state()
            self.assertGreaterEqual(first["global_episode_count"], 40)
            self.assertEqual(first["last_global_metrics"]["_count"], 40)
            self.assertAlmostEqual(first["last_global_metrics"]["avg_return_progress_per_step"], 0.21, places=5)
            self.assertAlmostEqual(first["last_global_metrics"]["avg_return_efficiency_ratio"], 0.55, places=5)
            self.assertAlmostEqual(first["last_global_metrics"]["avg_high_need_return_stall_rate"], 0.0, places=5)
            self.assertEqual(first["stage"], "warmup")

            second = store.refresh_state()
            self.assertEqual(second["stage"], "blend")

    def test_shared_curriculum_state_honors_initial_blend_freeze(self):
        from agent_ppo.workflow.curriculum_state import SharedCurriculumStateStore

        with TemporaryDirectory() as tmp, patch.dict(
            "os.environ",
            {
                "KAIWU_TRAINING_START_MODE": "scratch",
                "KAIWU_PRELOAD_MODEL": "0",
                "KAIWU_CURRICULUM_INITIAL_STAGE": "blend",
                "KAIWU_CURRICULUM_INITIAL_BLEND_FREEZE_STEPS": "5000",
            },
            clear=False,
        ):
            store = SharedCurriculumStateStore(Path(tmp))

            learning = {
                "entropy_loss": 0.80,
                "entropy_trend_ratio": 1.0,
                "env_total_score": 920.0,
                "global_step": 6200.0,
            }

            def make_episode(local_ep):
                return {
                    "episode_cnt_local": local_ep,
                    "result": "completed",
                    "profile": "broad",
                    "clean_score": 900.0,
                    "finished_steps": 1000.0,
                    "charge_count": 4.0,
                    "remaining_charge": 250.0,
                    "invalid_move_rate": 0.0,
                    "charge_efficiency": 220.0,
                    "clean_per_step": 0.90,
                    "expert_weight": 0.0,
                    "late_return_rate": 0.01,
                    "late_contract_rate": 0.01,
                    "anchor_switch_rate": 0.0,
                    "target_switch_rate": 0.0,
                    "diag_rate_all": 0.20,
                    "diag_rate_contract": 0.22,
                    "diag_rate_return": 0.18,
                    "return_progress_per_step": 0.25,
                    "return_efficiency_ratio": 0.60,
                    "return_stall_rate": 0.20,
                    "recoverability_score_avg": 0.92,
                    "recoverability_violation_rate": 0.02,
                    "wall_hugging_clean_floor_rate": 0.02,
                    "stale_boundary_follow_rate": 0.01,
                    "narrow_unknown_commit_rate": 0.02,
                    "missed_charge_opportunity_rate": 0.0,
                    "charger_nearby_not_charged_rate": 0.0,
                    "suboptimal_target_hold_rate": 0.01,
                    "planner_policy_divergence_rate": 0.15,
                    "avg_path_cross_count_50": 1.2,
                    "avg_coverage_efficiency_20": 0.90,
                    "avg_all_charger_known_path_count": 2.8,
                    "avg_unknown_on_target_path_ratio": 0.05,
                    "mode_usage_depart": 0.0,
                    "mode_usage_expand": 0.08,
                    "mode_usage_harvest": 0.55,
                    "mode_usage_contract": 0.22,
                    "mode_usage_return": 0.12,
                    "mode_usage_evade": 0.03,
                }

            for helper_idx in range(4):
                payload = {
                    "window_metrics": {},
                    "bootstrap_metrics": {},
                    "learning_metrics": learning,
                    "runtime": {"global_step_since_resume": 3200},
                    "recent_episode_metrics": [make_episode(helper_idx * 10 + offset) for offset in range(1, 11)],
                }
                store.write_signal(f"helper-{helper_idx}", payload)

            first = store.refresh_state()
            self.assertEqual(first["stage"], "blend")
            second = store.refresh_state()
            self.assertEqual(second["stage"], "blend")

            for helper_idx in range(4):
                payload = {
                    "window_metrics": {},
                    "bootstrap_metrics": {},
                    "learning_metrics": learning,
                    "runtime": {"global_step_since_resume": 6200},
                    "recent_episode_metrics": [make_episode(100 + helper_idx * 10 + offset) for offset in range(1, 11)],
                }
                store.write_signal(f"helper-{helper_idx}", payload)

            third = store.refresh_state()
            self.assertEqual(third["stage"], "blend")
            fourth = store.refresh_state()
            self.assertEqual(fourth["stage"], "robust")

    def test_warmup_strict_gate_requires_planner_and_zero_charge_health(self):
        from agent_ppo.workflow.curriculum_policy import choose_stage_decision

        context = {
            "global_step_since_resume": 8000,
            "window_metrics": {
                "_count": 40,
                "win_rate": 0.75,
                "battery_fail_rate": 0.10,
                "collision_fail_rate": 0.02,
                "return_stall_rate": 0.32,
                "planner_policy_divergence_rate": 0.86,
                "zero_charge_battery_fail_rate": 0.20,
            },
            "bootstrap_metrics": None,
            "learning_metrics": {"entropy_loss": 0.80, "entropy_trend_ratio": 1.0},
            "resume_fast_track": False,
            "training_start_mode": "resume",
            "preload_enabled": True,
            "entered_global_step": 0,
        }
        blocked = choose_stage_decision("warmup", context, None)
        self.assertEqual(blocked["proposed_stage"], "warmup")

        context["window_metrics"]["planner_policy_divergence_rate"] = 0.70
        context["window_metrics"]["zero_charge_battery_fail_rate"] = 0.45
        passed = choose_stage_decision("warmup", context, None)
        self.assertEqual(passed["proposed_stage"], "blend")
        self.assertEqual(passed["promotion_reason"], "strict_gate")

    def test_transition_guard_blocks_immediate_regression_and_marks_degraded_mainline(self):
        from agent_ppo.workflow.curriculum_state import SharedCurriculumStateStore

        with TemporaryDirectory() as tmp:
            store = SharedCurriculumStateStore(Path(tmp))
            base_state = store.read_state()
            base_state.update(
                {
                    "stage": "blend",
                    "stage_version": 2,
                    "source_session_id": "sess-1",
                    "entered_global_step": 1000,
                    "in_transition_guard": True,
                    "transition_target_stage": "blend",
                    "transition_entered_global_step": 2000,
                    "last_stage_transition_global_step": 2000,
                    "stage_entry_metrics": {
                        "battery_fail_rate": 0.20,
                        "return_stall_rate": 0.30,
                        "planner_policy_divergence_rate": 0.60,
                    },
                }
            )
            store.state_path.write_text(json.dumps(base_state, ensure_ascii=True), encoding="utf-8")

            signal_payload = {
                "session_id": "sess-1",
                "runtime": {"global_step_since_resume": 5000},
                "learning_metrics": {"entropy_loss": 0.85, "global_step": 5000.0},
                "window_metrics": {
                    "_count": 40,
                    "win_rate": 0.40,
                    "battery_fail_rate": 0.55,
                    "return_stall_rate": 0.60,
                    "planner_policy_divergence_rate": 0.86,
                    "zero_charge_battery_fail_rate": 0.60,
                },
                "bootstrap_metrics": {
                    "_count": 20,
                    "win_rate": 0.40,
                    "battery_fail_rate": 0.55,
                    "return_stall_rate": 0.60,
                    "planner_policy_divergence_rate": 0.86,
                    "zero_charge_battery_fail_rate": 0.60,
                },
            }
            store.write_signal("helper-1", signal_payload)

            first = store.refresh_state()
            self.assertEqual(first["stage"], "blend")
            self.assertFalse(first["degraded_mainline"])

            second = store.refresh_state()
            self.assertEqual(second["stage"], "blend")
            self.assertTrue(second["degraded_mainline"])

    def test_training_preload_resolution_prefers_latest_preload_metadata(self):
        from agent_ppo.workflow.preload_checkpoint import resolve_training_preload
        from agent_ppo.workflow.state_layout import ensure_runtime_state_dirs

        with TemporaryDirectory() as tmp:
            code_dir = Path(tmp)
            ckpt_dir = ensure_runtime_state_dirs(code_dir).preload_cache_dir
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / "model.ckpt-24843.pkl"
            ckpt_path.write_bytes(b"checkpoint")

            latest_preload = {
                "enabled": True,
                "checkpoint_id": "24843",
                "checkpoint_path": str(ckpt_path),
                "checkpoint_dir": str(ckpt_dir),
                "global_step": 24843,
                "episode_cnt": 320,
            }
            (ckpt_dir / "latest_preload.json").write_text(
                json.dumps(latest_preload, ensure_ascii=True),
                encoding="utf-8",
            )

            resolved = resolve_training_preload(code_dir, {})
            self.assertTrue(resolved["enabled"])
            self.assertEqual(resolved["checkpoint_id"], "24843")
            self.assertEqual(Path(resolved["checkpoint_path"]), ckpt_path)
            self.assertEqual(Path(resolved["checkpoint_dir"]), ckpt_dir)

    def test_training_resume_resolution_exposes_state_metadata_paths(self):
        from agent_ppo.workflow.preload_checkpoint import resolve_training_preload
        from agent_ppo.workflow.state_layout import ensure_runtime_state_dirs

        with TemporaryDirectory() as tmp:
            code_dir = Path(tmp)
            layout = ensure_runtime_state_dirs(code_dir)
            ckpt_dir = layout.preload_cache_dir
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / "model.ckpt-24843.pkl"
            ckpt_path.write_bytes(b"checkpoint")
            (ckpt_dir / "latest_preload.json").write_text(
                json.dumps(
                    {
                        "enabled": True,
                        "checkpoint_id": "24843",
                        "checkpoint_path": str(ckpt_path),
                        "checkpoint_dir": str(ckpt_dir),
                        "global_step": 24843,
                    },
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )
            state_path = layout.current.prepared_resume_dir / "resume.state.json"
            curriculum_path = layout.current.prepared_resume_dir / "curriculum_state.snapshot.json"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(
                json.dumps(
                    {
                        "checkpoint_path": str(layout.current.prepared_resume_dir / "model.pkl"),
                        "checkpoint_id": "24843",
                        "global_step": 24843,
                        "global_step_since_resume": 9999,
                        "curriculum_state_snapshot_path": str(curriculum_path),
                    },
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )
            (layout.current.prepared_resume_dir / "model.pkl").write_bytes(b"prepared")
            curriculum_path.write_text(json.dumps({"stage": "blend"}, ensure_ascii=True), encoding="utf-8")

            resolved = resolve_training_preload(
                code_dir,
                {
                    "KAIWU_TRAINING_START_MODE": "resume",
                    "KAIWU_RESUME_BUNDLE_DIR": str(layout.current.prepared_resume_dir),
                },
            )
            self.assertTrue(resolved["enabled"])
            self.assertEqual(resolved["training_start_mode"], "resume")
            self.assertEqual(Path(resolved["resume_state_metadata_path"]), state_path)
            self.assertEqual(Path(resolved["curriculum_state_snapshot_path"]), curriculum_path)

    def test_training_preload_resolution_respects_scratch_start_mode(self):
        from agent_ppo.workflow.preload_checkpoint import resolve_training_preload
        from agent_ppo.workflow.state_layout import ensure_runtime_state_dirs

        with TemporaryDirectory() as tmp:
            code_dir = Path(tmp)
            ckpt_dir = ensure_runtime_state_dirs(code_dir).preload_cache_dir
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / "model.ckpt-24843.pkl"
            ckpt_path.write_bytes(b"checkpoint")
            (ckpt_dir / "latest_preload.json").write_text(
                json.dumps(
                    {
                        "enabled": True,
                        "checkpoint_id": "24843",
                        "checkpoint_path": str(ckpt_path),
                        "checkpoint_dir": str(ckpt_dir),
                        "global_step": 24843,
                    },
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )

            resolved = resolve_training_preload(code_dir, {"KAIWU_TRAINING_START_MODE": "scratch"})
            self.assertFalse(resolved["enabled"])

    def test_training_resume_requires_explicit_bundle_source(self):
        from agent_ppo.workflow.preload_checkpoint import resolve_training_preload

        with TemporaryDirectory() as tmp:
            code_dir = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                resolve_training_preload(code_dir, {"KAIWU_TRAINING_START_MODE": "resume"})

    def test_clear_current_runtime_state_removes_current_and_legacy_runtime_files(self):
        from agent_ppo.workflow.state_layout import clear_current_runtime_state, ensure_runtime_state_dirs

        with TemporaryDirectory() as tmp:
            code_dir = Path(tmp)
            layout = ensure_runtime_state_dirs(code_dir)
            layout.current.curriculum_state_path.write_text("{}", encoding="utf-8")
            layout.current.run_session_manifest_path.write_text("{}", encoding="utf-8")
            (layout.current.curriculum_signal_dir / "sig.json").write_text("{}", encoding="utf-8")
            (layout.current.prepared_resume_dir / "model.pkl").write_bytes(b"prepared")
            (code_dir / "curriculum_state.json").write_text("{}", encoding="utf-8")
            (code_dir / ".current_run_session.json").write_text("{}", encoding="utf-8")
            (code_dir / ".curriculum_state.lock").write_text("", encoding="utf-8")

            clear_current_runtime_state(code_dir, clear_legacy_current=True)

            self.assertFalse(layout.current.curriculum_state_path.exists())
            self.assertFalse(layout.current.run_session_manifest_path.exists())
            self.assertFalse((layout.current.curriculum_signal_dir / "sig.json").exists())
            self.assertFalse((layout.current.prepared_resume_dir / "model.pkl").exists())
            self.assertFalse((code_dir / "curriculum_state.json").exists())
            self.assertFalse((code_dir / ".current_run_session.json").exists())
            self.assertFalse((code_dir / ".curriculum_state.lock").exists())

    def test_clear_external_framework_state_clears_assets_but_preserves_directory_skeleton(self):
        from agent_ppo.workflow.startup_mode import clear_external_framework_state

        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            ckpt_root = base / "data" / "ckpt" / "robot_vacuum_ppo"
            projects_root = base / "data" / "projects"
            (ckpt_root / "models" / "worker").mkdir(parents=True, exist_ok=True)
            (ckpt_root / "models_new" / "worker").mkdir(parents=True, exist_ok=True)
            (ckpt_root / "plugins" / "old").mkdir(parents=True, exist_ok=True)
            (ckpt_root / "convert_models_aisrv" / "tmp").mkdir(parents=True, exist_ok=True)
            (ckpt_root / "model.ckpt-10.pkl").write_bytes(b"x")
            (ckpt_root / "id_list").write_text("10\n", encoding="utf-8")
            (ckpt_root / "process_stop.done").write_text("1\n", encoding="utf-8")
            (ckpt_root / "process_stop.meta.json").write_text("{}", encoding="utf-8")
            projects_root.mkdir(parents=True, exist_ok=True)
            (projects_root / "sigterm_pids").write_text("1\n", encoding="utf-8")

            with patch("agent_ppo.workflow.startup_mode.Path", wraps=Path) as path_cls:
                def _side_effect(raw):
                    text = str(raw)
                    if text.startswith("/data/"):
                        return base / text.lstrip("/")
                    return Path(raw)
                path_cls.side_effect = _side_effect
                clear_external_framework_state("robot_vacuum", "ppo")

            self.assertFalse((ckpt_root / "model.ckpt-10.pkl").exists())
            self.assertFalse((ckpt_root / "id_list").exists())
            self.assertTrue((ckpt_root / "models").exists())
            self.assertTrue((ckpt_root / "models_new").exists())
            self.assertTrue((ckpt_root / "plugins").exists())
            self.assertTrue((ckpt_root / "convert_models_aisrv").exists())
            self.assertEqual(list((ckpt_root / "models").iterdir()), [])
            self.assertEqual(list((ckpt_root / "models_new").iterdir()), [])
            self.assertEqual(list((ckpt_root / "plugins").iterdir()), [])
            self.assertEqual(list((ckpt_root / "convert_models_aisrv").iterdir()), [])
            self.assertFalse((ckpt_root / "process_stop.done").exists())
            self.assertFalse((ckpt_root / "process_stop.meta.json").exists())
            self.assertFalse((projects_root / "sigterm_pids").exists())

    def test_seed_preload_from_resume_creates_compatible_ckpt(self):
        from agent_ppo.workflow.preload_checkpoint import seed_preload_from_resume, resolve_latest_preload
        from agent_ppo.workflow.state_layout import legacy_resume_snapshots_dir

        with TemporaryDirectory() as tmp:
            code_dir = Path(tmp)
            resume_path = legacy_resume_snapshots_dir(code_dir) / "resume-time-1.pkl"
            resume_path.parent.mkdir(parents=True, exist_ok=True)
            resume_path.write_bytes(b"resume")

            seeded = seed_preload_from_resume(code_dir, str(resume_path), checkpoint_id="0")
            self.assertTrue(seeded["enabled"])

            latest = resolve_latest_preload(code_dir)
            self.assertEqual(latest["checkpoint_id"], "0")
            self.assertTrue(Path(latest["checkpoint_path"]).exists())

    def test_seed_preload_from_resume_copies_resume_sidecars_when_present(self):
        from agent_ppo.workflow.preload_checkpoint import (
            RESUME_CURRICULUM_SNAPSHOT_FILE,
            RESUME_LATEST_STATE_FILE,
            seed_preload_from_resume,
        )
        from agent_ppo.workflow.state_layout import ensure_runtime_state_dirs, legacy_resume_snapshots_dir

        with TemporaryDirectory() as tmp:
            code_dir = Path(tmp)
            layout = ensure_runtime_state_dirs(code_dir)
            resume_path = legacy_resume_snapshots_dir(code_dir) / "resume-step-step0000001.pkl"
            resume_path.parent.mkdir(parents=True, exist_ok=True)
            resume_path.write_bytes(b"resume")
            curriculum_path = resume_path.with_suffix(".curriculum.json")
            curriculum_path.write_text(json.dumps({"stage": "blend"}, ensure_ascii=True), encoding="utf-8")
            state_path = resume_path.with_suffix(".state.json")
            state_path.write_text(
                json.dumps(
                    {
                        "global_step": 1,
                        "global_step_since_resume": 123,
                        "curriculum_state_snapshot_path": str(curriculum_path),
                    },
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )

            seed_preload_from_resume(code_dir, str(resume_path), checkpoint_id="0")

            copied_state = layout.current.prepared_resume_dir / RESUME_LATEST_STATE_FILE
            copied_curriculum = layout.current.prepared_resume_dir / RESUME_CURRICULUM_SNAPSHOT_FILE
            self.assertTrue(copied_state.exists())
            self.assertTrue(copied_curriculum.exists())
            payload = json.loads(copied_state.read_text(encoding="utf-8"))
            self.assertEqual(Path(payload["curriculum_state_snapshot_path"]), copied_curriculum)

    def test_seed_preload_from_resume_falls_back_to_latest_resume_file(self):
        from agent_ppo.workflow.preload_checkpoint import seed_preload_from_resume, resolve_latest_preload
        from agent_ppo.workflow.state_layout import legacy_resume_latest_checkpoint_path

        with TemporaryDirectory() as tmp:
            code_dir = Path(tmp)
            fallback_resume = legacy_resume_latest_checkpoint_path(code_dir)
            fallback_resume.write_bytes(b"resume")

            seeded = seed_preload_from_resume(code_dir, "/missing/path.pkl", checkpoint_id="0")
            self.assertTrue(seeded["enabled"])
            latest = resolve_latest_preload(code_dir)
            self.assertEqual(latest["checkpoint_id"], "0")
            self.assertTrue(Path(latest["checkpoint_path"]).exists())

    def test_lite_benchmark_cache_and_stage_mapping(self):
        from agent_ppo.eval.lite_benchmark_bootstrap import (
            _lite_cache_signature,
            _recommended_initial_stage,
            lite_benchmark_metadata_path,
            resolve_cached_lite_benchmark,
        )

        self.assertEqual(
            _recommended_initial_stage(
                {
                    "completed_rate": 0.72,
                    "battery_fail_rate": 0.08,
                    "collision_fail_rate": 0.02,
                    "broad_win_rate": 0.70,
                    "return_stall_rate": 0.35,
                }
            ),
            "robust",
        )
        self.assertEqual(
            _recommended_initial_stage(
                {
                    "completed_rate": 0.60,
                    "battery_fail_rate": 0.18,
                    "collision_fail_rate": 0.03,
                    "broad_win_rate": 0.40,
                    "return_stall_rate": 0.50,
                }
            ),
            "blend",
        )
        self.assertEqual(
            _recommended_initial_stage(
                {
                    "completed_rate": 0.45,
                    "battery_fail_rate": 0.35,
                    "collision_fail_rate": 0.10,
                    "broad_win_rate": 0.20,
                    "return_stall_rate": 0.60,
                }
            ),
            "warmup",
        )
        self.assertEqual(
            _recommended_initial_stage(
                {
                    "completed_rate": 0.95,
                    "battery_fail_rate": 0.0,
                    "collision_fail_rate": 0.0,
                    "broad_win_rate": 0.20,
                    "return_stall_rate": 0.66,
                }
            ),
            "blend",
        )

        with TemporaryDirectory() as tmp:
            code_dir = Path(tmp)
            checkpoint_path = code_dir / "agent_ppo" / "ckpt" / "model.ckpt-0.pkl"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(b"ckpt")
            meta_path = lite_benchmark_metadata_path(code_dir)
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_mtime_ns": checkpoint_path.stat().st_mtime_ns,
                "cache_signature": _lite_cache_signature(),
                "recommended_initial_stage": "blend",
                "saved_at": "2026-04-18 13:00:00",
            }
            meta_path.write_text(json.dumps(payload), encoding="utf-8")
            resolved = resolve_cached_lite_benchmark(code_dir, str(checkpoint_path))
            self.assertEqual(resolved["recommended_initial_stage"], "blend")

    def test_seed_initial_state_uses_session_scoped_stage(self):
        from agent_ppo.workflow.curriculum_state import SharedCurriculumStateStore

        with TemporaryDirectory() as tmp:
            store = SharedCurriculumStateStore(Path(tmp))
            first = store.seed_initial_state(
                "sess-a",
                "blend",
                lite_benchmark_used=True,
                lite_benchmark_metrics={"recommended_initial_stage": "blend"},
            )
            self.assertEqual(first["stage"], "blend")
            self.assertTrue(first["lite_benchmark_used"])
            second = store.seed_initial_state("sess-a", "warmup", lite_benchmark_used=False)
            self.assertEqual(second["stage"], "blend")
            third = store.seed_initial_state("sess-b", "warmup", lite_benchmark_used=False)
            self.assertEqual(third["stage"], "warmup")

    def test_curriculum_state_read_state_ignores_legacy_root_state_in_scratch_mode(self):
        from agent_ppo.workflow.curriculum_state import SharedCurriculumStateStore

        with TemporaryDirectory() as tmp:
            code_dir = Path(tmp)
            (code_dir / "curriculum_state.json").write_text(
                json.dumps({"source_session_id": "legacy-run", "stage": "robust"}, ensure_ascii=True),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"KAIWU_TRAINING_START_MODE": "scratch"}, clear=False):
                store = SharedCurriculumStateStore(code_dir)
                state = store.read_state()
            self.assertNotEqual(state.get("source_session_id"), "legacy-run")
            self.assertEqual(state.get("training_start_mode"), "scratch")

    def test_shared_curriculum_state_ignores_old_session_signals(self):
        from agent_ppo.workflow.curriculum_state import SharedCurriculumStateStore

        with TemporaryDirectory() as tmp:
            store = SharedCurriculumStateStore(Path(tmp))
            store.seed_initial_state("new-session", "warmup", lite_benchmark_used=False)

            old_signal = {
                "session_id": "old-session",
                "window_metrics": {
                    "_count": 40,
                    "win_rate": 0.95,
                    "battery_fail_rate": 0.0,
                    "collision_fail_rate": 0.0,
                    "return_stall_rate": 0.18,
                    "wall_hugging_clean_floor_rate": 0.01,
                    "suboptimal_target_hold_rate": 0.01,
                    "planner_policy_divergence_rate": 0.10,
                    "broad_win_rate": 0.80,
                },
                "bootstrap_metrics": {
                    "_count": 10,
                    "win_rate": 0.95,
                    "battery_fail_rate": 0.0,
                    "return_stall_rate": 0.18,
                    "wall_hugging_clean_floor_rate": 0.01,
                    "suboptimal_target_hold_rate": 0.01,
                    "planner_policy_divergence_rate": 0.10,
                    "broad_win_rate": 0.80,
                },
                "learning_metrics": {
                    "entropy_loss": 0.80,
                    "entropy_trend_ratio": 1.0,
                    "env_total_score": 900.0,
                },
                "runtime": {"global_step_since_resume": 6200},
                "recent_episode_metrics": [],
            }
            store.write_signal("helper-old", old_signal)

            refreshed = store.refresh_state()
            self.assertEqual(refreshed["stage"], "warmup")
            self.assertEqual(refreshed["global_episode_count"], 0)

    def test_claim_run_session_id_secondary_requires_initialized_shared_manifest(self):
        from agent_ppo.workflow.train_workflow import EpisodeRunner
        from agent_ppo.workflow.state_layout import ensure_runtime_state_dirs

        with TemporaryDirectory() as tmp, patch.dict(os.environ, {"KAIWU_AISRV_INDEX": "2"}, clear=False):
            runner = EpisodeRunner.__new__(EpisodeRunner)
            runner.signal_source_id = "aisrv-2-pid-123"
            runner.code_path = Path(tmp)
            runner.state_layout = ensure_runtime_state_dirs(runner.code_path)
            runner.training_start_mode = "scratch"
            runner.curriculum_store = type("Store", (), {})()

            with patch("agent_ppo.workflow.train_workflow.fcntl", None):
                with self.assertRaises(RuntimeError):
                    EpisodeRunner._claim_run_session_id(runner, "helper-local-fallback")

            manifest_path = runner.state_layout.current.run_session_manifest_path
            manifest_path.write_text(
                json.dumps({"run_session_id": "shared-run", "state_initialized": True}, ensure_ascii=True),
                encoding="utf-8",
            )

            with patch("agent_ppo.workflow.train_workflow.fcntl", None):
                session_id = EpisodeRunner._claim_run_session_id(runner, "helper-local-fallback")

            self.assertEqual(session_id, "shared-run")

    def test_profile_plan_for_runtime_tightens_and_releases_weights(self):
        from agent_ppo.workflow.curriculum_policy import profile_plan_for_runtime

        poor_state = {
            "stage": "blend",
            "global_step_since_resume": 1200,
            "last_global_metrics": {
                "battery_fail_rate": 0.26,
                "return_stall_rate": 0.58,
                "planner_policy_divergence_rate": 0.86,
                "avg_clean_per_step": 0.34,
                "broad_win_rate": 0.30,
            },
        }
        poor_plan = profile_plan_for_runtime("blend", poor_state)
        self.assertTrue(poor_plan["observation_phase_active"])
        self.assertTrue(poor_plan["tightened"])
        self.assertGreaterEqual(poor_plan["weight_map"]["anchor"], 0.40)
        self.assertLessEqual(poor_plan["weight_map"]["broad"], 0.20)

        strong_state = {
            "stage": "blend",
            "global_step_since_resume": 4500,
            "last_global_metrics": {
                "battery_fail_rate": 0.08,
                "return_stall_rate": 0.32,
                "planner_policy_divergence_rate": 0.48,
                "avg_clean_per_step": 0.58,
                "broad_win_rate": 0.62,
            },
        }
        strong_plan = profile_plan_for_runtime("blend", strong_state)
        self.assertTrue(strong_plan["observation_phase_active"])
        self.assertFalse(strong_plan["tightened"])
        self.assertGreater(strong_plan["weight_map"]["broad"], poor_plan["weight_map"]["broad"])
        self.assertLess(strong_plan["weight_map"]["anchor"], poor_plan["weight_map"]["anchor"])

    def test_profile_plan_for_runtime_uses_softer_weights_when_warmup_is_degraded(self):
        from agent_ppo.workflow.curriculum_policy import profile_plan_for_runtime

        degraded_state = {
            "stage": "warmup",
            "global_step_since_resume": 22000,
            "curriculum_stagnation_level": 3,
            "degraded_mainline": True,
            "last_global_metrics": {
                "battery_fail_rate": 0.40,
                "return_stall_rate": 0.58,
                "planner_policy_divergence_rate": 0.87,
                "zero_charge_battery_fail_rate": 0.70,
            },
        }

        plan = profile_plan_for_runtime("warmup", degraded_state)

        self.assertAlmostEqual(plan["weight_map"]["anchor"], 0.52, places=4)
        self.assertAlmostEqual(plan["weight_map"]["mild"], 0.33, places=4)
        self.assertAlmostEqual(plan["weight_map"]["broad"], 0.15, places=4)
        self.assertTrue(plan["tightened"])

    def test_train_workflow_preserves_reward_components_for_episode_diagnostics(self):
        try:
            from agent_ppo.workflow.train_workflow import EpisodeRunner
        except ModuleNotFoundError as exc:
            self.skipTest(f"train_workflow import unavailable in local test env: {exc}")

        runner = EpisodeRunner.__new__(EpisodeRunner)
        payload = runner._normalize_reward_payload(
            {
                "reward_total": -0.42,
                "reward_clean": 0.08,
                "reward_survive": -0.50,
                "cleaning": 0.12,
                "charge_detour_cost": -0.18,
                "planner_alignment": -0.15,
                "charge_interrupt_cost": -0.05,
                "battery_process_cost": 0.21,
                "charge_need_score": 0.67,
            }
        )
        self.assertAlmostEqual(payload["reward_cleaning"], 0.12, places=5)
        self.assertAlmostEqual(payload["reward_charge_detour_cost"], -0.18, places=5)
        self.assertAlmostEqual(payload["reward_planner_alignment"], -0.15, places=5)
        self.assertAlmostEqual(payload["constraint_battery_process_cost"], 0.21, places=5)
        self.assertAlmostEqual(payload["constraint_charge_need_score"], 0.67, places=5)

        diagnostics = EpisodeRunner._episode_sequence_diagnostics(
            [
                {"mode": 3, "target": 1, "route_anchor": 1, "charger_slack": 1.0, "future_recoverability_score": 0.5,
                 "anchor_return_dist": 5.0, "is_diag_action": 0.0, "wall_hugging_clean_floor": 0.0,
                 "stale_boundary_follow": 0.0, "narrow_unknown_commit": 0.0, "missed_charge_opportunity": 0.0,
                 "charger_nearby_not_charged": 0.0, "suboptimal_target_hold": 0.0,
                 "planner_policy_divergence": 1.0, "path_cross_count_50": 1.0, "coverage_efficiency_20": 0.5,
                 "all_charger_known_path_count": 1.0, "unknown_on_target_path_ratio": 0.0,
                 "reward_cleaning": 0.12, "reward_charge_detour_cost": -0.18, "reward_planner_alignment": -0.15,
                 "reward_charge_interrupt_cost": -0.05, "reward_idle": -0.10, "reward_frontier": 0.02,
                 "reward_streak": 0.01, "reward_necessary_charge_bonus": 0.02, "reward_npc": 0.0, "reward_charge_route_progress_bonus": 0.04,
                 "reward_cps_bonus": 0.0},
                {"mode": 4, "target": 1, "route_anchor": 1, "charger_slack": 1.0, "future_recoverability_score": 0.5,
                 "anchor_return_dist": 5.0, "is_diag_action": 0.0, "wall_hugging_clean_floor": 0.0,
                 "stale_boundary_follow": 0.0, "narrow_unknown_commit": 0.0, "missed_charge_opportunity": 0.0,
                 "charger_nearby_not_charged": 0.0, "suboptimal_target_hold": 0.0,
                 "planner_policy_divergence": 1.0, "path_cross_count_50": 1.0, "coverage_efficiency_20": 0.5,
                 "all_charger_known_path_count": 1.0, "unknown_on_target_path_ratio": 0.0,
                 "reward_cleaning": 0.08, "reward_charge_detour_cost": -0.12, "reward_planner_alignment": -0.10,
                 "reward_charge_interrupt_cost": -0.03, "reward_idle": -0.08, "reward_frontier": 0.01,
                 "reward_streak": 0.00, "reward_necessary_charge_bonus": 0.00, "reward_npc": 0.0, "reward_charge_route_progress_bonus": 0.02,
                 "reward_cps_bonus": 0.0},
            ]
        )
        self.assertAlmostEqual(diagnostics["avg_reward_cleaning"], 0.10, places=5)
        self.assertAlmostEqual(diagnostics["avg_reward_charge_detour_cost"], -0.15, places=5)
        self.assertAlmostEqual(diagnostics["avg_reward_planner_alignment"], -0.125, places=5)

    def test_retrain_reward_defaults_expose_new_charge_cost_and_terminal_knobs(self):
        import agent_ppo.conf.conf as conf_module

        env_keys = [
            "KAIWU_CHARGE_ROUTE_PROGRESS_SCALE",
            "KAIWU_NECESSARY_CHARGE_BONUS_SCALE",
            "KAIWU_UNNECESSARY_CHARGE_PENALTY",
            "KAIWU_CHARGE_DETOUR_COST_SCALE",
            "KAIWU_CHARGE_INTERRUPT_COST_SCALE",
            "KAIWU_SKIP_NEEDED_CHARGE_PENALTY",
            "KAIWU_REWARD_CLEANING_BASE",
            "KAIWU_REWARD_STREAK_BONUS_BASE",
            "KAIWU_REWARD_CPS_BONUS_SCALE",
            "KAIWU_REWARD_CPS_BONUS_BASELINE",
            "KAIWU_REWARD_CPS_BONUS_SPAN",
            "KAIWU_BATTERY_FAIL_TASK_REWARD_SCALE",
            "KAIWU_EARLY_BATTERY_FAIL_TASK_REWARD_SCALE",
            "KAIWU_REWARD_SCHEDULE_ENABLED",
            "KAIWU_REWARD_SCHEDULE_WARM_STEPS",
            "KAIWU_REWARD_SCHEDULE_TOTAL_STEPS",
            "KAIWU_NECESSARY_CHARGE_BONUS_SCALE_PEAK",
            "KAIWU_BATTERY_FAIL_TASK_REWARD_SCALE_PEAK",
            "KAIWU_EARLY_BATTERY_FAIL_TASK_REWARD_SCALE_PEAK",
            "KAIWU_BATTERY_SAFE_NEED_THRESHOLD",
            "KAIWU_BATTERY_CRITICAL_NEED_THRESHOLD",
            "KAIWU_LAMBDA_BATTERY_INIT",
            "KAIWU_LAMBDA_BATTERY_MIN",
            "KAIWU_COVERAGE_EFFICIENCY_BONUS_SCALE",
            "KAIWU_FRONTIER_REWARD_SCALE",
            "KAIWU_COVERAGE_TANGLE_PENALTY_SCALE",
            "KAIWU_EDGE_FOLLOW_BONUS_SCALE",
            "KAIWU_EDGE_FOLLOW_FRONTIER_THRESHOLD",
            "KAIWU_EDGE_FOLLOW_CROSS_COUNT_MAX",
            "KAIWU_CHARGER_ACCESS_DISCOVERY_BONUS_SCALE",
            "KAIWU_CHARGER_ACCESS_PROBE_BONUS_SCALE",
            "KAIWU_CHARGER_ACCESS_PROBE_FRONTIER_THRESHOLD",
            "KAIWU_CHARGER_ACCESS_PROBE_UNKNOWN_RATIO_MIN",
            "KAIWU_CURRICULUM_WARMUP_TIMEOUT_STEPS",
            "KAIWU_CURRICULUM_BLEND_TIMEOUT_STEPS",
            "KAIWU_CURRICULUM_ROBUST_TIMEOUT_STEPS",
            "KAIWU_CURRICULUM_BLEND_GUARD_STEPS",
            "KAIWU_CURRICULUM_ROBUST_GUARD_STEPS",
            "KAIWU_EPISODE_COMPLETED_BONUS",
            "KAIWU_EPISODE_FAIL_EARLY_SCALE",
        ]
        original = {key: os.environ.get(key) for key in env_keys}
        try:
            for key in env_keys:
                os.environ.pop(key, None)
            conf_module = importlib.reload(conf_module)
            self.assertAlmostEqual(conf_module.Config.CHARGE_ROUTE_PROGRESS_SCALE, 0.18, places=5)
            self.assertAlmostEqual(conf_module.Config.NECESSARY_CHARGE_BONUS_SCALE, 1.05, places=5)
            self.assertAlmostEqual(conf_module.Config.UNNECESSARY_CHARGE_PENALTY, 0.18, places=5)
            self.assertAlmostEqual(conf_module.Config.CHARGE_DETOUR_COST_SCALE, 0.045, places=5)
            self.assertAlmostEqual(conf_module.Config.CHARGE_INTERRUPT_COST_SCALE, 0.03, places=5)
            self.assertAlmostEqual(conf_module.Config.SKIP_NEEDED_CHARGE_PENALTY, 0.22, places=5)
            self.assertAlmostEqual(conf_module.Config.REWARD_CLEANING_BASE, 0.66, places=5)
            self.assertAlmostEqual(conf_module.Config.REWARD_STREAK_BONUS_BASE, 0.07, places=5)
            self.assertAlmostEqual(conf_module.Config.REWARD_CPS_BONUS_SCALE, 0.32, places=5)
            self.assertAlmostEqual(conf_module.Config.REWARD_CPS_BONUS_BASELINE, 0.58, places=5)
            self.assertAlmostEqual(conf_module.Config.REWARD_CPS_BONUS_SPAN, 0.22, places=5)
            self.assertAlmostEqual(conf_module.Config.FRONTIER_REWARD_SCALE, 0.08, places=5)
            self.assertAlmostEqual(conf_module.Config.BATTERY_FAIL_TASK_REWARD_SCALE, 0.50, places=5)
            self.assertAlmostEqual(conf_module.Config.EARLY_BATTERY_FAIL_TASK_REWARD_SCALE, 0.30, places=5)
            self.assertTrue(conf_module.Config.REWARD_SCHEDULE_ENABLED)
            self.assertEqual(conf_module.Config.REWARD_SCHEDULE_WARM_STEPS, 5000)
            self.assertEqual(conf_module.Config.REWARD_SCHEDULE_TOTAL_STEPS, 20000)
            self.assertAlmostEqual(conf_module.Config.NECESSARY_CHARGE_BONUS_SCALE_PEAK, 1.40, places=5)
            self.assertAlmostEqual(conf_module.Config.BATTERY_FAIL_TASK_REWARD_SCALE_PEAK, 0.40, places=5)
            self.assertAlmostEqual(conf_module.Config.EARLY_BATTERY_FAIL_TASK_REWARD_SCALE_PEAK, 0.20, places=5)
            self.assertAlmostEqual(conf_module.Config.BATTERY_SAFE_NEED_THRESHOLD, 0.12, places=5)
            self.assertAlmostEqual(conf_module.Config.BATTERY_CRITICAL_NEED_THRESHOLD, 0.28, places=5)
            self.assertAlmostEqual(conf_module.Config.LAMBDA_BATTERY_INIT, 0.60, places=5)
            self.assertAlmostEqual(conf_module.Config.LAMBDA_BATTERY_MIN, 0.60, places=5)
            self.assertAlmostEqual(conf_module.Config.COVERAGE_EFFICIENCY_BONUS_SCALE, 0.12, places=5)
            self.assertAlmostEqual(conf_module.Config.COVERAGE_TANGLE_PENALTY_SCALE, 0.10, places=5)
            self.assertAlmostEqual(conf_module.Config.EDGE_FOLLOW_BONUS_SCALE, 0.06, places=5)
            self.assertAlmostEqual(conf_module.Config.EDGE_FOLLOW_FRONTIER_THRESHOLD, 0.10, places=5)
            self.assertAlmostEqual(conf_module.Config.EDGE_FOLLOW_CROSS_COUNT_MAX, 8.0, places=5)
            self.assertAlmostEqual(conf_module.Config.CHARGER_ACCESS_DISCOVERY_BONUS_SCALE, 0.18, places=5)
            self.assertAlmostEqual(conf_module.Config.CHARGER_ACCESS_PROBE_BONUS_SCALE, 0.05, places=5)
            self.assertAlmostEqual(conf_module.Config.CHARGER_ACCESS_PROBE_FRONTIER_THRESHOLD, 0.05, places=5)
            self.assertAlmostEqual(conf_module.Config.CHARGER_ACCESS_PROBE_UNKNOWN_RATIO_MIN, 0.12, places=5)
            self.assertEqual(conf_module.Config.CURRICULUM_WARMUP_TIMEOUT_STEPS, 25000)
            self.assertEqual(conf_module.Config.CURRICULUM_BLEND_TIMEOUT_STEPS, 35000)
            self.assertEqual(conf_module.Config.CURRICULUM_ROBUST_TIMEOUT_STEPS, 50000)
            self.assertEqual(conf_module.Config.CURRICULUM_BLEND_GUARD_STEPS, 8000)
            self.assertEqual(conf_module.Config.CURRICULUM_ROBUST_GUARD_STEPS, 12000)
            self.assertEqual(conf_module.Config.CURRICULUM_STAGE_TRANSITION_COOLDOWN_STEPS, 8000)
            self.assertAlmostEqual(conf_module.Config.EPISODE_COMPLETED_BONUS, 6.0, places=5)
            self.assertAlmostEqual(conf_module.Config.EPISODE_FAIL_EARLY_SCALE, 1.2, places=5)
        finally:
            for key, value in original.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            importlib.reload(conf_module)

    def test_snapshot_defaults_include_step_snapshots_and_eight_hour_time_retention(self):
        import agent_ppo.conf.conf as conf_module

        env_keys = [
            "KAIWU_RESUME_STEP_SNAPSHOT_INTERVAL",
            "KAIWU_KEEP_STEP_RESUME_SNAPSHOTS",
            "KAIWU_RESUME_TIME_SNAPSHOT_INTERVAL_SECONDS",
        ]
        original = {key: os.environ.get(key) for key in env_keys}
        try:
            for key in env_keys:
                os.environ.pop(key, None)
            conf_module = importlib.reload(conf_module)
            self.assertEqual(conf_module.Config.RESUME_STEP_SNAPSHOT_INTERVAL, 6000)
            self.assertEqual(conf_module.Config.KEEP_STEP_RESUME_SNAPSHOTS, 80)
            self.assertEqual(conf_module.Config.RESUME_TIME_SNAPSHOT_INTERVAL_SECONDS, 600)
            self.assertEqual(conf_module.Config.KEEP_TIME_RESUME_SNAPSHOTS, 48)
        finally:
            for key, value in original.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            importlib.reload(conf_module)

    def test_reward_schedule_anneals_necessary_charge_and_battery_fail_scales(self):
        from agent_ppo.utils.reward_schedule import get_reward_schedule

        start = get_reward_schedule(0)
        warm = get_reward_schedule(5000)
        middle = get_reward_schedule(12000)
        end = get_reward_schedule(20000)
        late = get_reward_schedule(26000)

        self.assertAlmostEqual(start["scheduled_necessary_charge_bonus_scale"], 1.40, places=5)
        self.assertAlmostEqual(warm["scheduled_necessary_charge_bonus_scale"], 1.40, places=5)
        self.assertGreater(middle["scheduled_necessary_charge_bonus_scale"], 1.05)
        self.assertAlmostEqual(end["scheduled_necessary_charge_bonus_scale"], 1.05, places=5)
        self.assertAlmostEqual(late["scheduled_necessary_charge_bonus_scale"], 1.05, places=5)
        self.assertAlmostEqual(start["scheduled_battery_fail_task_reward_scale"], 0.40, places=5)
        self.assertAlmostEqual(start["scheduled_early_battery_fail_task_reward_scale"], 0.20, places=5)
        self.assertAlmostEqual(end["scheduled_battery_fail_task_reward_scale"], 0.50, places=5)
        self.assertAlmostEqual(end["scheduled_early_battery_fail_task_reward_scale"], 0.30, places=5)

    def test_warmup_profile_guard_holds_and_releases_on_battery_health(self):
        from agent_ppo.workflow.curriculum_policy import profile_plan_for_runtime

        guarded = {
            "stage": "warmup",
            "global_step_since_resume": 0,
            "last_global_metrics": {
                "battery_fail_rate": 0.38,
                "avg_charge_count": 2.8,
                "zero_charge_battery_fail_rate": 0.30,
            },
        }
        guarded_plan = profile_plan_for_runtime("warmup", guarded)
        self.assertAlmostEqual(guarded_plan["weight_map"]["anchor"], 0.65, places=4)
        self.assertAlmostEqual(guarded_plan["weight_map"]["mild"], 0.30, places=4)
        self.assertAlmostEqual(guarded_plan["weight_map"]["broad"], 0.05, places=4)
        self.assertFalse(guarded_plan["observation_phase_active"])

        released = {
            "stage": "warmup",
            "global_step_since_resume": 5000,
            "last_global_metrics": {
                "battery_fail_rate": 0.18,
                "avg_charge_count": 4.8,
                "zero_charge_battery_fail_rate": 0.10,
            },
        }
        released_plan = profile_plan_for_runtime("warmup", released)
        self.assertAlmostEqual(released_plan["weight_map"]["anchor"], 0.45, places=4)
        self.assertAlmostEqual(released_plan["weight_map"]["mild"], 0.40, places=4)
        self.assertAlmostEqual(released_plan["weight_map"]["broad"], 0.15, places=4)

        transition_state = {
            "stage": "blend",
            "global_step_since_resume": 15000,
            "in_transition_guard": True,
            "transition_target_stage": "blend",
            "transition_entered_global_step": 10000,
            "last_global_metrics": {
                "battery_fail_rate": 0.10,
                "return_stall_rate": 0.40,
                "planner_policy_divergence_rate": 0.70,
            },
        }
        transition_plan = profile_plan_for_runtime("blend", transition_state)
        self.assertFalse(transition_plan["observation_phase_active"])
        self.assertAlmostEqual(transition_plan["weight_map"]["anchor"], 0.45, places=4)
        self.assertAlmostEqual(transition_plan["weight_map"]["mild"], 0.40, places=4)
        self.assertAlmostEqual(transition_plan["weight_map"]["broad"], 0.15, places=4)

    def test_curriculum_stagnation_status_uses_stage_specific_thresholds(self):
        from agent_ppo.workflow.curriculum_policy import stagnation_status

        level, reasons = stagnation_status(
            stage="warmup",
            metrics={
                "avg_clean_per_step": 0.25,
                "planner_policy_divergence_rate": 0.88,
                "mode_usage_expand": 0.0,
                "return_stall_rate": 0.60,
            },
            global_step_since_resume=9000,
            entered_global_step=0,
            stagnant_windows=3,
        )
        self.assertEqual(level, 1)
        self.assertIn("cps", reasons)

        level2, reasons2 = stagnation_status(
            stage="warmup",
            metrics={
                "avg_clean_per_step": 0.22,
                "planner_policy_divergence_rate": 0.91,
                "mode_usage_expand": 0.0,
                "return_stall_rate": 0.63,
            },
            global_step_since_resume=12000,
            entered_global_step=0,
            stagnant_windows=6,
        )
        self.assertEqual(level2, 2)
        self.assertIn("planner", reasons2)

    def test_apply_terminal_outcome_to_step_records_writes_collision_cost_into_last_sample(self):
        if importlib.util.find_spec("numpy") is None:
            self.skipTest("numpy is not available in the host test environment")
        from agent_ppo.workflow.train_workflow import EpisodeRunner

        step_records = [
            {"reward_clean": 0.2, "reward_survive": 0.1, "reward_total": 0.1},
            {"reward_clean": 0.3, "reward_survive": 0.2, "reward_total": 0.1},
        ]
        EpisodeRunner._apply_terminal_outcome_to_step_records(
            step_records,
            {
                "task_reward_scale": 0.5,
                "task_terminal_bonus": 1.0,
                "battery_terminal_cost": 0.4,
                "collision_terminal_cost": 0.7,
                "final_reward": -0.1,
            },
        )

        self.assertAlmostEqual(step_records[0]["reward_clean"], 0.1, places=5)
        self.assertAlmostEqual(step_records[1]["reward_clean"], 1.15, places=5)
        self.assertAlmostEqual(step_records[1]["reward_survive"], 1.3, places=5)
        self.assertAlmostEqual(step_records[1]["reward_total"], -0.05, places=5)

    def test_battery_fail_outcome_adds_extra_cost_for_zero_charge_fail(self):
        if importlib.util.find_spec("numpy") is None:
            self.skipTest("numpy is not available in the host test environment")
        from agent_ppo.workflow.train_workflow import EpisodeRunner

        reward_schedule = {
            "scheduled_battery_fail_task_reward_scale": 0.5,
            "scheduled_early_battery_fail_task_reward_scale": 0.3,
        }

        base_cost, base_scale, base_zero = EpisodeRunner._compute_battery_fail_outcome(
            charge_count=2.0,
            battery_fail_type="mid_recoverability_loss",
            battery_fail_severity=0.6,
            reward_schedule=reward_schedule,
        )
        zero_cost, zero_scale, zero_flag = EpisodeRunner._compute_battery_fail_outcome(
            charge_count=0.0,
            battery_fail_type="mid_recoverability_loss",
            battery_fail_severity=0.6,
            reward_schedule=reward_schedule,
        )

        self.assertFalse(base_zero)
        self.assertTrue(zero_flag)
        self.assertGreater(zero_cost, base_cost)
        self.assertLessEqual(zero_scale, base_scale)

    def test_choose_stage_decision_resume_stabilization_holds_current_stage(self):
        from agent_ppo.workflow.curriculum_policy import choose_stage_decision

        decision = choose_stage_decision(
            current_stage="robust",
            context={
                "global_step_since_resume": 1000,
                "window_metrics": None,
                "bootstrap_metrics": None,
                "learning_metrics": {},
                "training_start_mode": "resume",
                "preload_enabled": True,
                "entered_global_step": 0,
            },
            stage_entry_metrics=None,
        )

        self.assertEqual(decision["proposed_stage"], "robust")

    def test_seed_preload_from_resume_rewrites_checkpoint_path_in_state_sidecar(self):
        from agent_ppo.workflow.preload_checkpoint import latest_resume_state_path, seed_preload_from_resume
        from agent_ppo.workflow.state_layout import legacy_resume_snapshots_dir

        with TemporaryDirectory() as tmp:
            code_dir = Path(tmp)
            resume_path = legacy_resume_snapshots_dir(code_dir) / "resume-step-step0000001.pkl"
            resume_path.parent.mkdir(parents=True, exist_ok=True)
            resume_path.write_bytes(b"resume")
            state_path = resume_path.with_suffix(".state.json")
            state_path.write_text(
                json.dumps(
                    {
                        "checkpoint_path": str(resume_path),
                        "curriculum_state_snapshot_path": str(resume_path.with_suffix(".curriculum.json")),
                    },
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )

            seed_preload_from_resume(code_dir, str(resume_path), checkpoint_id="0")

            copied_state = json.loads(latest_resume_state_path(code_dir).read_text(encoding="utf-8"))
            self.assertTrue(str(copied_state["checkpoint_path"]).endswith("runtime_state/current/prepared_resume/model.pkl"))

    def test_resolve_benchmark_checkpoint_falls_back_when_prepared_resume_missing(self):
        from agent_ppo.workflow.preload_checkpoint import resolve_benchmark_checkpoint

        with TemporaryDirectory() as tmp:
            code_dir = Path(tmp)
            legacy_resume = code_dir / "model.ckpt-resume.pkl"
            legacy_resume.write_bytes(b"legacy")

            resolved = resolve_benchmark_checkpoint(
                code_dir,
                explicit_checkpoint=None,
                config_resume_checkpoint=str(code_dir / "runtime_state" / "current" / "prepared_resume" / "model.pkl"),
            )
            self.assertEqual(Path(resolved), legacy_resume)

    def test_save_manual_bundle_writes_run_manifest_file(self):
        from agent_ppo.workflow.train_workflow import EpisodeRunner
        from agent_ppo.workflow.state_layout import RUN_SESSION_MANIFEST_FILE, ensure_runtime_state_dirs

        with TemporaryDirectory() as tmp:
            code_dir = Path(tmp)
            runner = EpisodeRunner.__new__(EpisodeRunner)
            runner.code_path = code_dir
            runner.state_layout = ensure_runtime_state_dirs(code_dir, "run-1")
            runner.run_state = runner.state_layout.for_run("run-1")
            runner.run_session_id = "run-1"
            runner.helper_session_id = "helper-1"
            runner.signal_source_id = "aisrv-1-pid-1"
            runner.training_start_mode = "resume"
            runner.episode_cnt = 12
            runner.latest_learning_metrics = {"global_step": 345}
            runner.latest_training_metrics = {}
            runner.resume_global_step_base = None
            runner.resume_global_step_offset = 0
            runner.resume_curriculum_snapshot_path = runner.run_state.resume_latest_curriculum_snapshot_path
            runner.logger = None
            runner.archive = type("Archive", (), {"log_event": lambda *args, **kwargs: None})()
            runner.curriculum_store = type("Store", (), {"read_state": lambda self=None: {"stage": "warmup"}})()

            def _write_json_atomic(path, payload):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")

            runner._write_json_atomic = _write_json_atomic
            runner._current_train_global_step = lambda: 345
            runner._global_step_since_resume = lambda: 345
            runner._capture_curriculum_resume_snapshot = lambda: {"stage": "warmup", "run_session_id": "run-1"}
            runner._resume_state_payload = lambda trigger, clean_score, checkpoint_path=None, curriculum_snapshot_path=None: {
                "trigger": trigger,
                "checkpoint_path": str(checkpoint_path),
                "curriculum_state_snapshot_path": str(curriculum_snapshot_path),
                "global_step": 345,
                "global_step_since_resume": 345,
            }

            checkpoint_path = code_dir / "checkpoint.pkl"
            checkpoint_path.write_bytes(b"checkpoint")
            EpisodeRunner._save_manual_bundle(runner, checkpoint_path, "manual", 12.5)

            bundle_dir = runner.state_layout.manual_saves_dir / "manual-ep000012-step0000345"
            self.assertTrue((bundle_dir / RUN_SESSION_MANIFEST_FILE).exists())

    def test_primary_helper_claim_run_session_initializes_current_manifest_and_state(self):
        from agent_ppo.workflow.train_workflow import EpisodeRunner
        from agent_ppo.workflow.state_layout import ensure_runtime_state_dirs
        import agent_ppo.workflow.train_workflow as train_workflow_mod

        with TemporaryDirectory() as tmp:
            code_dir = Path(tmp)
            runner = EpisodeRunner.__new__(EpisodeRunner)
            runner.code_path = code_dir
            runner.state_layout = ensure_runtime_state_dirs(code_dir)
            runner.training_start_mode = "scratch"
            runner.signal_source_id = "aisrv-1-pid-1"
            runner.curriculum_store = type(
                "Store",
                (),
                {
                    "seed_initial_state": lambda self, session_id, initial_stage, lite_benchmark_used=False, lite_benchmark_metrics=None: {
                        "source_session_id": session_id,
                        "stage": initial_stage,
                    }
                },
            )()

            with patch.dict(os.environ, {"KAIWU_AISRV_INDEX": "1", "KAIWU_TRAINING_START_MODE": "scratch"}, clear=False):
                with patch.object(train_workflow_mod, "clear_current_runtime_state") as clear_mock:
                    run_session_id = EpisodeRunner._claim_run_session_id(runner, "helper-a")

            manifest = json.loads(runner.state_layout.current.run_session_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(run_session_id, manifest["run_session_id"])
            self.assertEqual(manifest["training_start_mode"], "scratch")
            self.assertTrue(manifest["state_initialized"])
            clear_mock.assert_called_once()

    def test_secondary_helper_rejects_uninitialized_current_manifest(self):
        from agent_ppo.workflow.train_workflow import EpisodeRunner
        from agent_ppo.workflow.state_layout import ensure_runtime_state_dirs
        import agent_ppo.workflow.train_workflow as train_workflow_mod

        with TemporaryDirectory() as tmp:
            code_dir = Path(tmp)
            runner = EpisodeRunner.__new__(EpisodeRunner)
            runner.code_path = code_dir
            runner.state_layout = ensure_runtime_state_dirs(code_dir)
            runner.training_start_mode = "scratch"
            runner.signal_source_id = "aisrv-2-pid-2"
            manifest_path = runner.state_layout.current.run_session_manifest_path
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "run_session_id": "old-run",
                        "state_initialized": False,
                    },
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"KAIWU_AISRV_INDEX": "2", "KAIWU_TRAINING_START_MODE": "scratch"}, clear=False):
                with patch.object(train_workflow_mod, "fcntl", None):
                    with self.assertRaises(RuntimeError):
                        EpisodeRunner._claim_run_session_id(runner, "helper-b")

    def test_save_resume_artifacts_named_episode_snapshot_writes_snapshot(self):
        from agent_ppo.workflow.train_workflow import EpisodeRunner
        from agent_ppo.workflow.state_layout import ensure_runtime_state_dirs

        with TemporaryDirectory() as tmp:
            code_dir = Path(tmp)
            runner = EpisodeRunner.__new__(EpisodeRunner)
            runner.code_path = code_dir
            runner.state_layout = ensure_runtime_state_dirs(code_dir, "run-2")
            runner.run_state = runner.state_layout.for_run("run-2")
            runner.run_session_id = "run-2"
            runner.helper_session_id = "helper-2"
            runner.signal_source_id = "aisrv-1-pid-2"
            runner.session_dir = runner.run_state.session_best_dir / "helper-2"
            runner.session_dir.mkdir(parents=True, exist_ok=True)
            runner.resume_snapshot_dir = runner.run_state.resume_snapshots_dir
            runner.resume_snapshot_dir.mkdir(parents=True, exist_ok=True)
            runner.resume_latest_path = runner.run_state.resume_latest_checkpoint_path
            runner.resume_latest_meta_path = runner.run_state.resume_latest_meta_path
            runner.resume_latest_state_path = runner.run_state.resume_latest_state_path
            runner.resume_curriculum_snapshot_path = runner.run_state.resume_latest_curriculum_snapshot_path
            runner.latest_model_path = runner.run_state.resume_latest_dir / "latest_model.pkl"
            runner.preload_ckpt_dir = runner.state_layout.preload_cache_dir
            runner.preload_ckpt_dir.mkdir(parents=True, exist_ok=True)
            runner.preload_latest_meta_path = runner.state_layout.preload_latest_metadata_path
            runner.episode_cnt = 7
            runner.keep_episode_snapshots = 3
            runner.keep_step_snapshots = 3
            runner.keep_time_snapshots = 3
            runner.latest_learning_metrics = {"global_step": 222}
            runner.latest_training_metrics = {}
            runner.resume_global_step_base = None
            runner.resume_global_step_offset = 0
            runner.resume_restored_checkpoint_global_step = None
            runner.resume_fast_track = True
            runner.training_start_mode = "resume"
            runner._is_primary_resume_writer = True
            runner.archive = type("Archive", (), {"log_event": lambda *args, **kwargs: None})()
            runner.logger = type("Logger", (), {"info": lambda *args, **kwargs: None})()
            runner.agent = type("Agent", (), {"model": type("Model", (), {"state_dict": lambda self=None: {}})()})()
            runner._current_train_global_step = lambda: 222
            runner._capture_curriculum_resume_snapshot = lambda: {"stage": "warmup"}
            runner._write_resume_meta = lambda payload: runner._write_json_atomic(runner.resume_latest_meta_path, payload)
            runner._prune_preload_snapshots = lambda: None
            runner._prune_snapshots = lambda prefix, keep_count: None

            EpisodeRunner._save_resume_artifacts(runner, "episode", 3.2, with_named_snapshot=True)

            snapshot_path = runner.resume_snapshot_dir / "resume-episode-ep000007.pkl"
            self.assertTrue(snapshot_path.exists())

    def test_infer_mode_uses_slack_not_anchor_distance_for_contract_gate(self):
        if importlib.util.find_spec("numpy") is None:
            self.skipTest("numpy is not available in the host test environment")
        from agent_ppo.conf.conf import Config
        from agent_ppo.feature.preprocessor import Preprocessor

        prep = Preprocessor.__new__(Preprocessor)
        prep.battery = 180.0
        prep.battery_max = 200.0
        prep.nearest_npc_dist = 10.0
        prep.charger_slack = 50.0
        prep.future_recoverability_score = 1.0
        prep.anchor_return_dist = 0.0
        prep.route_contract_pressure = 0.0
        prep.total_charger = 4
        prep.steps_since_charge = Config.DEPART_STEPS + 10
        prep.local_dirt_density = 0.0
        prep.dirty_adjacent = 0
        prep._get_guidance = lambda: {
            "margin": 50.0,
            "all_charger_known_path_count": 4,
            "unknown_path_ratio": 0.0,
            "planner_topk_reachable_count": 4,
            "planner_multi_route_recoverability": 1.0,
        }

        inferred = Preprocessor._infer_mode(prep)

        self.assertNotEqual(inferred, Preprocessor.MODE_CONTRACT)

    def test_benchmark_aggregate_overall_contains_completed_and_failure_rates(self):
        if importlib.util.find_spec("numpy") is None:
            self.skipTest("numpy is not available in the host test environment")
        from agent_ppo.eval.benchmark import _aggregate_results, _load_benchmark_checkpoint
        from unittest.mock import Mock

        aggregated = _aggregate_results(
            [
                {"round": "r1", "result": "completed", "clean_score": 10.0, "steps": 10.0, "charge_count": 1.0, "dirt_ratio": 0.5, "invalid_move_rate": 0.0},
                {"round": "r1", "result": "battery", "clean_score": 5.0, "steps": 8.0, "charge_count": 0.0, "dirt_ratio": 0.3, "invalid_move_rate": 0.1},
                {"round": "r1", "result": "collision", "clean_score": 3.0, "steps": 6.0, "charge_count": 0.0, "dirt_ratio": 0.2, "invalid_move_rate": 0.2},
            ]
        )

        overall = aggregated["overall"]
        self.assertAlmostEqual(overall["completed_rate"], 1.0 / 3.0, places=4)
        self.assertAlmostEqual(overall["battery_fail_rate"], 1.0 / 3.0, places=4)
        self.assertAlmostEqual(overall["collision_fail_rate"], 1.0 / 3.0, places=4)
        self.assertAlmostEqual(overall["avg_invalid_move_rate"], 0.1, places=5)

        agent = Mock()
        agent.device = "cpu"
        agent.model = Mock()
        with self.assertRaises(FileNotFoundError):
            _load_benchmark_checkpoint(agent, "/missing/checkpoint.pkl", Mock())

    def test_lite_benchmark_cache_invalidates_when_checkpoint_mtime_changes(self):
        from agent_ppo.eval.lite_benchmark_bootstrap import (
            _lite_cache_signature,
            lite_benchmark_metadata_path,
            resolve_cached_lite_benchmark,
        )

        with TemporaryDirectory() as tmp:
            code_dir = Path(tmp)
            checkpoint_path = code_dir / "model.ckpt-1.pkl"
            checkpoint_path.write_bytes(b"v1")
            meta_path = lite_benchmark_metadata_path(code_dir)
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(
                json.dumps(
                    {
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_mtime_ns": checkpoint_path.stat().st_mtime_ns,
                        "cache_signature": _lite_cache_signature(),
                    },
                    ensure_ascii=True,
                ),
                encoding="utf-8",
            )
            self.assertIsNotNone(resolve_cached_lite_benchmark(code_dir, str(checkpoint_path)))

            newer_mtime = checkpoint_path.stat().st_mtime_ns + 10_000
            checkpoint_path.write_bytes(b"v2")
            os.utime(checkpoint_path, ns=(newer_mtime, newer_mtime))
            self.assertIsNone(resolve_cached_lite_benchmark(code_dir, str(checkpoint_path)))

    def test_compose_exports_critical_runtime_env_keys(self):
        candidates = [
            Path("/home/user/TcKaiwuFinal/train/.docker-compose.yaml"),
            Path("/workspace/train/.docker-compose.yaml"),
            Path(__file__).resolve().parents[3] / "train" / ".docker-compose.yaml",
        ]
        compose_path = next((path for path in candidates if path.exists()), None)
        if compose_path is None:
            self.skipTest("docker-compose file is not mounted in this test environment")
        compose_text = compose_path.read_text(encoding="utf-8")

        for key in (
            "KAIWU_CURRICULUM_RESUME_STABILIZE_STEPS",
            "KAIWU_CURRICULUM_WARMUP_TIMEOUT_STEPS",
            "KAIWU_CURRICULUM_BLEND_TIMEOUT_STEPS",
            "KAIWU_CURRICULUM_ROBUST_TIMEOUT_STEPS",
            "KAIWU_CURRICULUM_BLEND_GUARD_STEPS",
            "KAIWU_CURRICULUM_ROBUST_GUARD_STEPS",
            "KAIWU_RESUME_STEP_SNAPSHOT_INTERVAL",
            "KAIWU_KEEP_STEP_RESUME_SNAPSHOTS",
            "KAIWU_RESUME_TIME_SNAPSHOT_INTERVAL_SECONDS",
            "KAIWU_ARCHIVE_DIR",
        ):
            self.assertIn(key, compose_text)


if __name__ == "__main__":
    unittest.main()

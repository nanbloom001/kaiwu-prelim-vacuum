#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


class CurriculumAndCheckpointScoreTests(unittest.TestCase):
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

        self.assertGreater(healthy["resume_readiness_score"], unhealthy["resume_readiness_score"])
        self.assertGreater(healthy["checkpoint_preservation_score"], unhealthy["checkpoint_preservation_score"])

    def test_curriculum_fast_skip_and_regression_rules(self):
        from agent_ppo.workflow.curriculum_policy import choose_stage, should_regress_stage

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

        with TemporaryDirectory() as tmp:
            store = SharedCurriculumStateStore(Path(tmp))

            strong_payload = {
                "window_metrics": {
                    "_count": 8,
                    "win_rate": 0.90,
                    "battery_fail_rate": 0.00,
                    "collision_fail_rate": 0.00,
                    "return_stall_rate": 0.18,
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


if __name__ == "__main__":
    unittest.main()

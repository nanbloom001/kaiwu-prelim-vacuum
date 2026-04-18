#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import unittest
import json
import importlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


class CurriculumAndCheckpointScoreTests(unittest.TestCase):
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
        from agent_ppo.workflow.curriculum_policy import choose_stage, should_regress_stage, curriculum_gate_ratios

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

    def test_shared_curriculum_state_builds_full_window_from_global_recent_episodes(self):
        from agent_ppo.workflow.curriculum_state import SharedCurriculumStateStore

        with TemporaryDirectory() as tmp:
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
            self.assertEqual(first["stage"], "warmup")

            second = store.refresh_state()
            self.assertEqual(second["stage"], "blend")

    def test_shared_curriculum_state_honors_initial_blend_freeze(self):
        from agent_ppo.workflow.curriculum_state import SharedCurriculumStateStore

        with TemporaryDirectory() as tmp, patch.dict(
            "os.environ",
            {
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

    def test_training_preload_resolution_prefers_latest_preload_metadata(self):
        from agent_ppo.workflow.preload_checkpoint import resolve_training_preload

        with TemporaryDirectory() as tmp:
            code_dir = Path(tmp)
            ckpt_dir = code_dir / "agent_ppo" / "ckpt"
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

    def test_training_preload_resolution_respects_scratch_start_mode(self):
        from agent_ppo.workflow.preload_checkpoint import resolve_training_preload

        with TemporaryDirectory() as tmp:
            code_dir = Path(tmp)
            ckpt_dir = code_dir / "agent_ppo" / "ckpt"
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

    def test_seed_preload_from_resume_creates_compatible_ckpt(self):
        from agent_ppo.workflow.preload_checkpoint import seed_preload_from_resume, resolve_latest_preload

        with TemporaryDirectory() as tmp:
            code_dir = Path(tmp)
            resume_path = code_dir / "resume_snapshots" / "resume-time-1.pkl"
            resume_path.parent.mkdir(parents=True, exist_ok=True)
            resume_path.write_bytes(b"resume")

            seeded = seed_preload_from_resume(code_dir, str(resume_path), checkpoint_id="0")
            self.assertTrue(seeded["enabled"])

            latest = resolve_latest_preload(code_dir)
            self.assertEqual(latest["checkpoint_id"], "0")
            self.assertTrue(Path(latest["checkpoint_path"]).exists())

    def test_seed_preload_from_resume_falls_back_to_latest_resume_file(self):
        from agent_ppo.workflow.preload_checkpoint import seed_preload_from_resume, resolve_latest_preload

        with TemporaryDirectory() as tmp:
            code_dir = Path(tmp)
            fallback_resume = code_dir / "model.ckpt-resume.pkl"
            fallback_resume.write_bytes(b"resume")

            seeded = seed_preload_from_resume(code_dir, "/missing/path.pkl", checkpoint_id="0")
            self.assertTrue(seeded["enabled"])
            latest = resolve_latest_preload(code_dir)
            self.assertEqual(latest["checkpoint_id"], "0")
            self.assertTrue(Path(latest["checkpoint_path"]).exists())

    def test_lite_benchmark_cache_and_stage_mapping(self):
        from agent_ppo.eval.lite_benchmark_bootstrap import (
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
            meta_path = lite_benchmark_metadata_path(code_dir)
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "checkpoint_path": "/workspace/code/agent_ppo/ckpt/model.ckpt-0.pkl",
                "recommended_initial_stage": "blend",
                "saved_at": "2026-04-18 13:00:00",
            }
            meta_path.write_text(json.dumps(payload), encoding="utf-8")
            resolved = resolve_cached_lite_benchmark(code_dir, "/workspace/code/agent_ppo/ckpt/model.ckpt-0.pkl")
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
                "return_stall": -0.18,
                "planner_alignment": -0.15,
                "charge_margin_pressure": -0.05,
            }
        )
        self.assertAlmostEqual(payload["reward_cleaning"], 0.12, places=5)
        self.assertAlmostEqual(payload["reward_return_stall"], -0.18, places=5)
        self.assertAlmostEqual(payload["reward_planner_alignment"], -0.15, places=5)

        diagnostics = EpisodeRunner._episode_sequence_diagnostics(
            [
                {"mode": 3, "target": 1, "route_anchor": 1, "charger_slack": 1.0, "future_recoverability_score": 0.5,
                 "anchor_return_dist": 5.0, "is_diag_action": 0.0, "wall_hugging_clean_floor": 0.0,
                 "stale_boundary_follow": 0.0, "narrow_unknown_commit": 0.0, "missed_charge_opportunity": 0.0,
                 "charger_nearby_not_charged": 0.0, "suboptimal_target_hold": 0.0,
                 "planner_policy_divergence": 1.0, "path_cross_count_50": 1.0, "coverage_efficiency_20": 0.5,
                 "all_charger_known_path_count": 1.0, "unknown_on_target_path_ratio": 0.0,
                 "reward_cleaning": 0.12, "reward_return_stall": -0.18, "reward_planner_alignment": -0.15,
                 "reward_charge_margin_pressure": -0.05, "reward_idle": -0.10, "reward_frontier": 0.02,
                 "reward_streak": 0.01, "reward_charge": 0.0, "reward_npc": 0.0, "reward_return_progress": 0.04,
                 "reward_cps_bonus": 0.0},
                {"mode": 4, "target": 1, "route_anchor": 1, "charger_slack": 1.0, "future_recoverability_score": 0.5,
                 "anchor_return_dist": 5.0, "is_diag_action": 0.0, "wall_hugging_clean_floor": 0.0,
                 "stale_boundary_follow": 0.0, "narrow_unknown_commit": 0.0, "missed_charge_opportunity": 0.0,
                 "charger_nearby_not_charged": 0.0, "suboptimal_target_hold": 0.0,
                 "planner_policy_divergence": 1.0, "path_cross_count_50": 1.0, "coverage_efficiency_20": 0.5,
                 "all_charger_known_path_count": 1.0, "unknown_on_target_path_ratio": 0.0,
                 "reward_cleaning": 0.08, "reward_return_stall": -0.12, "reward_planner_alignment": -0.10,
                 "reward_charge_margin_pressure": -0.03, "reward_idle": -0.08, "reward_frontier": 0.01,
                 "reward_streak": 0.00, "reward_charge": 0.0, "reward_npc": 0.0, "reward_return_progress": 0.02,
                 "reward_cps_bonus": 0.0},
            ]
        )
        self.assertAlmostEqual(diagnostics["avg_reward_cleaning"], 0.10, places=5)
        self.assertAlmostEqual(diagnostics["avg_reward_return_stall"], -0.15, places=5)
        self.assertAlmostEqual(diagnostics["avg_reward_planner_alignment"], -0.125, places=5)

    def test_retrain_reward_defaults_expose_new_charge_and_terminal_knobs(self):
        import agent_ppo.conf.conf as conf_module

        env_keys = [
            "KAIWU_CHARGE_REWARD_BASE",
            "KAIWU_REWARD_CLEANING_BASE",
            "KAIWU_REWARD_STREAK_BONUS_BASE",
            "KAIWU_OVERCHARGE_PENALTY_SCALE",
            "KAIWU_COVERAGE_EFFICIENCY_BONUS_SCALE",
            "KAIWU_EPISODE_COMPLETED_BONUS",
            "KAIWU_EPISODE_FAIL_EARLY_SCALE",
        ]
        original = {key: os.environ.get(key) for key in env_keys}
        try:
            for key in env_keys:
                os.environ.pop(key, None)
            conf_module = importlib.reload(conf_module)
            self.assertAlmostEqual(conf_module.Config.CHARGE_REWARD_BASE, 0.60, places=5)
            self.assertAlmostEqual(conf_module.Config.REWARD_CLEANING_BASE, 0.90, places=5)
            self.assertAlmostEqual(conf_module.Config.REWARD_STREAK_BONUS_BASE, 0.10, places=5)
            self.assertAlmostEqual(conf_module.Config.OVERCHARGE_PENALTY_SCALE, 0.60, places=5)
            self.assertAlmostEqual(conf_module.Config.COVERAGE_EFFICIENCY_BONUS_SCALE, 0.12, places=5)
            self.assertAlmostEqual(conf_module.Config.EPISODE_COMPLETED_BONUS, 6.0, places=5)
            self.assertAlmostEqual(conf_module.Config.EPISODE_FAIL_EARLY_SCALE, 1.2, places=5)
        finally:
            for key, value in original.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            importlib.reload(conf_module)

    def test_warmup_observation_profile_matches_retrain_defaults(self):
        from agent_ppo.workflow.curriculum_policy import observation_phase_active, profile_plan_for_runtime

        state = {
            "stage": "warmup",
            "global_step_since_resume": 0,
            "last_global_metrics": {
                "battery_fail_rate": 0.05,
                "return_stall_rate": 0.20,
                "planner_policy_divergence_rate": 0.40,
                "avg_clean_per_step": 0.40,
                "broad_win_rate": 0.20,
            },
        }
        plan = profile_plan_for_runtime("warmup", state)
        self.assertTrue(plan["observation_phase_active"])
        self.assertAlmostEqual(plan["weight_map"]["anchor"], 0.55, places=4)
        self.assertAlmostEqual(plan["weight_map"]["mild"], 0.35, places=4)
        self.assertAlmostEqual(plan["weight_map"]["broad"], 0.10, places=4)
        self.assertFalse(plan["tightened"])
        self.assertTrue(observation_phase_active(7000, "warmup"))


if __name__ == "__main__":
    unittest.main()

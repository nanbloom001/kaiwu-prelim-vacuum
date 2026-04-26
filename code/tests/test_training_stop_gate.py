#!/usr/bin/env python3
"""Pure local tests for the training stop-gate helper."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_stop_gate_module():
    script_path = REPO_ROOT / "train" / "tools" / "training_stop_gate.py"
    spec = importlib.util.spec_from_file_location("training_stop_gate", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run_session(**overrides):
    payload = {
        "run_session_id": "20260426-110457",
        "launch_label": "s1_survival_gate-return-teacher-route-readiness-label-fixed-g40",
        "launch_instance_id": "s1_survival-20260426-110427-5012b82ffc43",
        "state_initialized": True,
        "train_phase": "s1_survival",
        "training_start_mode": "scratch",
    }
    payload.update(overrides)
    return payload


def _curriculum(source_session_id="20260426-110457", sample_window_metrics=None):
    return {
        "source_session_id": source_session_id,
        "train_phase": "s1_survival",
        "training_start_mode": "scratch",
        "sample_window_metrics": sample_window_metrics or {},
    }


class TrainingStopGateTests(unittest.TestCase):
    def test_bootstrap20_severe_collapse_stops_before_global40(self):
        module = _load_stop_gate_module()
        curriculum = _curriculum(
            sample_window_metrics={
                "bootstrap_10": {"_count": 10, "battery_fail_rate": 0.8, "zero_charge_battery_fail_rate": 0.6, "win_rate": 0.1},
                "bootstrap_20": {
                    "_count": 20,
                    "battery_fail_rate": 1.0,
                    "zero_charge_battery_fail_rate": 0.85,
                    "win_rate": 0.0,
                    "return_action_teacher_mask_nonzero_rate": 0.0,
                    "route_phase_teacher_from_critical_fallback_rate": 1.0,
                    "route_phase_teacher_from_return_reliable_rate": 0.0,
                },
                "global_40": {
                    "_count": 40,
                    "battery_fail_rate": 1.0,
                    "zero_charge_battery_fail_rate": 0.825,
                    "win_rate": 0.0,
                    "avg_clean_score": 143.875,
                },
            }
        )

        report = module.evaluate_stop_gate(_run_session(), curriculum)

        self.assertEqual(report["decision"], "STOP_BOOTSTRAP20_SEVERE_COLLAPSE")
        self.assertTrue(report["requires_cleanup"])
        self.assertEqual(report["metric_snapshot"]["bootstrap_20"]["battery_fail_rate"], 1.0)
        self.assertIn("global_40", report["sample_window_keys"])

    def test_noncollapse_bootstrap20_then_global40_is_accepted(self):
        module = _load_stop_gate_module()
        curriculum = _curriculum(
            source_session_id="20260426-090735",
            sample_window_metrics={
                "bootstrap_20": {
                    "_count": 20,
                    "battery_fail_rate": 0.4,
                    "zero_charge_battery_fail_rate": 0.3,
                    "win_rate": 0.55,
                },
                "global_40": {
                    "_count": 40,
                    "battery_fail_rate": 0.375,
                    "zero_charge_battery_fail_rate": 0.2,
                    "win_rate": 0.6,
                    "return_action_teacher_mask_nonzero_rate": 0.0,
                    "route_phase_teacher_from_critical_fallback_rate": 0.9799240368963646,
                    "route_phase_teacher_from_return_reliable_rate": 0.0,
                },
            },
        )
        run_session = _run_session(
            run_session_id="20260426-090735",
            launch_label="s1_survival_gate-return-teacher-route-readiness-diagnostic",
            launch_instance_id="s1_survival-20260426-090704-diagnostic",
        )

        report = module.evaluate_stop_gate(run_session, curriculum)

        self.assertEqual(report["decision"], "STOP_GLOBAL40_ACCEPTED")
        self.assertTrue(report["requires_cleanup"])
        self.assertEqual(report["metric_snapshot"]["global_40"]["win_rate"], 0.6)

    def test_stale_104351_expected_different_fresh_run_is_binding_mismatch(self):
        module = _load_stop_gate_module()
        run_session = _run_session(
            run_session_id="20260426-104351",
            launch_instance_id="",
            launch_label="s1_survival_gate-return-teacher-route-readiness-label-fixed-g40",
        )
        curriculum = _curriculum(
            source_session_id="20260426-104351",
            sample_window_metrics={
                "bootstrap_20": {
                    "_count": 20,
                    "battery_fail_rate": 0.9,
                    "zero_charge_battery_fail_rate": 0.7,
                    "win_rate": 0.1,
                }
            },
        )

        report = module.evaluate_stop_gate(
            run_session,
            curriculum,
            expected_run_session_id="20260426-110457",
            expected_launch_label="s1_survival_gate-return-teacher-route-readiness-label-fixed-g40",
            expected_launch_instance_id="s1_survival-20260426-110427-5012b82ffc43",
        )

        self.assertEqual(report["decision"], "STOP_BINDING_MISMATCH")
        self.assertTrue(any("run_session_id expected" in item for item in report["binding_mismatches"]))
        self.assertTrue(any("launch_instance_id expected" in item for item in report["binding_mismatches"]))

    def test_missing_window_at_timeout_stops_with_timeout_decision(self):
        module = _load_stop_gate_module()
        curriculum = _curriculum(sample_window_metrics={"bootstrap_10": {"_count": 10}})

        report = module.evaluate_stop_gate(
            _run_session(),
            curriculum,
            now_ts=200.0,
            start_ts=100.0,
            timeout_seconds=90.0,
            active_container_names=["kaiwu-train-aisrv-1"],
        )

        self.assertEqual(report["decision"], "STOP_TIMEOUT_MISSING_WINDOWS")
        self.assertTrue(report["requires_cleanup"])
        self.assertEqual(report["active_container_names"], ["kaiwu-train-aisrv-1"])

    def test_evaluate_from_paths_and_write_json(self):
        module = _load_stop_gate_module()
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_path = tmp_path / "run_session.json"
            curriculum_path = tmp_path / "curriculum_state.json"
            output_path = tmp_path / "report.json"
            run_path.write_text(json.dumps(_run_session()), encoding="utf-8")
            curriculum_path.write_text(
                json.dumps(
                    _curriculum(
                        sample_window_metrics={
                            "bootstrap_20": {"_count": 20, "battery_fail_rate": 0.1, "zero_charge_battery_fail_rate": 0.0, "win_rate": 0.9},
                            "global_40": {"_count": 40, "battery_fail_rate": 0.2, "zero_charge_battery_fail_rate": 0.0, "win_rate": 0.8},
                        }
                    )
                ),
                encoding="utf-8",
            )

            report = module.evaluate_stop_gate_from_paths(run_path, curriculum_path)
            module._write_json(output_path, report)
            written = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(written["decision"], "STOP_GLOBAL40_ACCEPTED")
        self.assertIn("thresholds", written)


if __name__ == "__main__":
    unittest.main()

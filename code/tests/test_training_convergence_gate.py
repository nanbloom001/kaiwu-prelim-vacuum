#!/usr/bin/env python3
"""Tests for the fixed-window training convergence gate tool."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


def _load_gate_module():
    script_path = Path(__file__).resolve().parents[2] / "train" / "tools" / "training_convergence_gate.py"
    spec = importlib.util.spec_from_file_location("training_convergence_gate", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TrainingConvergenceGateTest(unittest.TestCase):
    def test_global40_two_regressions_revert(self):
        module = _load_gate_module()
        fixture_path = Path(__file__).resolve().parents[2] / "train" / "tools" / "fixtures" / "convergence_global40_revert.json"

        report = module.build_gate_report(json.loads(fixture_path.read_text(encoding="utf-8")))

        self.assertEqual(report["decision"], "revert")
        self.assertEqual(report["node"], "global_40")
        self.assertEqual({item["metric"] for item in report["regressed_metrics"]}, {"avg_clean_per_step", "battery_fail_rate"})

    def test_global40_continue_requires_threshold_justification(self):
        module = _load_gate_module()
        fixture_path = Path(__file__).resolve().parents[2] / "train" / "tools" / "fixtures" / "convergence_global40_continue.json"

        report = module.build_gate_report(json.loads(fixture_path.read_text(encoding="utf-8")))

        self.assertEqual(report["decision"], "continue_to_global_80")
        rationale = report["rationale"]["continuation_justification"]
        self.assertTrue(rationale)
        self.assertTrue(all("metric" in item and "threshold" in item and "justification" in item for item in rationale))

    def test_writes_json_to_requested_path(self):
        module = _load_gate_module()
        fixture_path = Path(__file__).resolve().parents[2] / "train" / "tools" / "fixtures" / "convergence_global40_continue.json"
        with TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "gate.json"
            module._write_json(output_path, module.build_gate_report(json.loads(fixture_path.read_text(encoding="utf-8"))))
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["decision"], "continue_to_global_80")
        self.assertIn("thresholds", payload)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Tests for benchmark intervention candidate selector."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, cast
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = REPO_ROOT / "train" / "tools" / "recommend_benchmark_intervention.py"
BATTERY_FIXTURE = REPO_ROOT / ".sisyphus" / "evidence" / "benchmark-900" / "task-6-battery-fixture.json"
COVERAGE_FIXTURE = REPO_ROOT / ".sisyphus" / "evidence" / "benchmark-900" / "task-6-coverage-fixture.json"
BASELINE_SUMMARY = REPO_ROOT / ".sisyphus" / "evidence" / "benchmark-900" / "task-4-baseline-summary.json"
WAVE0_AUDIT = REPO_ROOT / ".sisyphus" / "evidence" / "benchmark-900" / "wave0" / "full-board-audit-merged.json"


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("recommend_benchmark_intervention", TOOL_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fixture(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _minimal_summary(**overall: Any) -> dict[str, Any]:
    merged_overall = {
        "anomaly_summary": {
            "avg_low_value_revisit_rate": 0.0,
            "avg_missed_charge_opportunity_rate": 0.0,
            "avg_positive_reward_while_no_progress_rate": 0.0,
            "avg_revisit_on_clean_floor_rate": 0.0,
        },
        "avg_clean_score": 420.0,
        "broad_win_rate": 0.8,
        "episode_count": 10,
        "return_stall_rate": 0.0,
    }
    merged_overall.update(overall)
    return {
        "benchmark_metadata": {"overall": merged_overall},
        "episode_count": merged_overall["episode_count"],
        "failure_buckets": {"battery depletion": 0, "collision/stuck loop": 0},
        "next_recommended_levers": [],
        "per_map": [],
    }


def _top_category(payload: dict[str, Any]) -> str:
    return str(payload["recommendations"][0]["category"])


class RecommendBenchmarkInterventionTests(unittest.TestCase):
    tool: ModuleType = cast(ModuleType, cast(object, None))
    build_recommendations: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] = cast(
        Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]], None
    )

    def setUp(self) -> None:
        self.tool = _load_tool()
        self.build_recommendations = cast(Any, getattr(self.tool, "build_recommendations"))

    def test_required_categories_are_present_exactly_once(self) -> None:
        summary = _fixture(BASELINE_SUMMARY)
        audit = _fixture(WAVE0_AUDIT)
        payload = self.build_recommendations(summary, audit)

        categories = [item["category"] for item in payload["recommendations"]]
        self.assertEqual(sorted(categories), sorted(getattr(self.tool, "CATEGORIES")))
        self.assertEqual(len(categories), len(set(categories)))
        self.assertEqual(payload["schema_version"], 1)

    def test_required_evidence_fixtures_select_expected_top_recommendation(self) -> None:
        for path in (BATTERY_FIXTURE, COVERAGE_FIXTURE):
            with self.subTest(path=path.name):
                fixture = _fixture(path)
                payload = self.build_recommendations(fixture["summary"], fixture["audit"])
                self.assertEqual(_top_category(payload), fixture["expected_top_category"])

    def test_each_failure_bucket_fixture_returns_expected_top_recommendation(self) -> None:
        cases: list[tuple[str, dict[str, Any], str]] = []

        battery = _fixture(BATTERY_FIXTURE)
        cases.append(("battery depletion", battery["summary"], "battery_safety"))

        collision = _minimal_summary(collision_fail_rate=0.7)
        collision["failure_buckets"] = {"battery depletion": 0, "collision/stuck loop": 40}
        collision["next_recommended_levers"] = [
            {"lever": "collision/stuck loop", "failure_count": 40, "reasons": ["40 collision/stuck failures"]}
        ]
        cases.append(("collision/stuck loop", collision, "collision_stuck"))

        missed = _minimal_summary()
        missed["benchmark_metadata"]["overall"]["anomaly_summary"]["avg_missed_charge_opportunity_rate"] = 0.25
        missed["next_recommended_levers"] = [
            {"lever": "missed charger", "failure_count": 0, "reasons": ["7 episodes with zero charges or missed-charge telemetry"]}
        ]
        cases.append(("missed charger", missed, "missed_charge"))

        coverage = _fixture(COVERAGE_FIXTURE)
        cases.append(("inefficient coverage", coverage["summary"], "coverage_efficiency"))

        return_stall = _minimal_summary(return_stall_rate=0.9, late_return_rate=0.2)
        cases.append(("return stall", return_stall, "return_stall"))

        checkpoint = _minimal_summary()
        checkpoint["benchmark_metadata"]["learner_log"] = {"has_checkpoint_issue": True}
        cases.append(("checkpoint issue", checkpoint, "checkpoint_model_load"))

        for name, summary, expected in cases:
            with self.subTest(name=name):
                payload = self.build_recommendations(summary, {})
                self.assertEqual(_top_category(payload), expected)

    def test_pure_positive_global_position_opportunity_beats_penalty_and_architecture(self) -> None:
        summary = _fixture(BASELINE_SUMMARY)
        audit = _fixture(WAVE0_AUDIT)
        payload = self.build_recommendations(summary, audit)
        by_category = {item["category"]: item for item in payload["recommendations"]}

        global_rank = by_category["global_position_signal_gap"]["rank"]
        penalty_like_rank = by_category["missed_charge"]["rank"]
        architecture_rank = by_category["representational_limit"]["rank"]

        self.assertLess(global_rank, penalty_like_rank)
        self.assertLess(global_rank, architecture_rank)
        self.assertEqual(by_category["global_position_signal_gap"]["modification_class"], "P1_information_additive")
        self.assertEqual(by_category["representational_limit"]["modification_class"], "R5_architecture")

    def test_coverage_recommendation_does_not_recommend_network_or_architecture(self) -> None:
        fixture = _fixture(COVERAGE_FIXTURE)
        payload = self.build_recommendations(fixture["summary"], fixture["audit"])
        top = payload["recommendations"][0]
        joined = " ".join(top["allowed_file_groups"] + top["existing_mechanisms_to_reuse"]).lower()

        self.assertEqual(top["category"], "coverage_efficiency")
        self.assertNotIn("model.py", joined)
        self.assertNotIn("architecture", joined)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

workflow_recovery = types.ModuleType("common_python.utils.workflow_disaster_recovery")
workflow_recovery.handle_disaster_recovery = lambda *_args, **_kwargs: False
sys.modules.setdefault("common_python", types.ModuleType("common_python"))
sys.modules.setdefault("common_python.utils", types.ModuleType("common_python.utils"))
sys.modules.setdefault("common_python.utils.workflow_disaster_recovery", workflow_recovery)

from agent_ppo.eval import benchmark


class BenchmarkDiagnosticTests(unittest.TestCase):
    def test_phase_event_summary_extracts_charge_and_return_turning_points(self):
        records = [
            self._rec(1, mode=2, slack=8.0),
            self._rec(2, mode=2, slack=5.0),
            self._rec(3, mode=3, slack=3.0, recoverability=0.4),
            self._rec(4, mode=4, slack=-1.0, recoverability=-0.1),
            self._rec(5, mode=4, slack=2.0, just_charged=1.0),
            self._rec(6, mode=2, slack=9.0, just_charged=1.0),
        ]

        summary = benchmark._phase_event_summary(records)

        self.assertEqual(summary["first_contract_step"], 3)
        self.assertEqual(summary["first_return_step"], 4)
        self.assertEqual(summary["first_negative_slack_step"], 4)
        self.assertEqual(summary["first_low_recoverability_step"], 4)
        self.assertEqual(summary["first_charge_step"], 5)
        self.assertEqual(summary["last_charge_step"], 6)
        self.assertEqual(summary["return_after_negative_slack_delay"], 0)

    def test_charge_timing_summary_counts_risk_while_cleaning(self):
        records = [
            self._rec(1, mode=2, slack=7.0, battery=80),
            self._rec(2, mode=2, slack=5.0, battery=70, risk_worsening=1.0),
            self._rec(3, mode=2, slack=-1.0, battery=60, risk_worsening=1.0, missed_charge=1.0),
            self._rec(4, mode=4, slack=-2.0, battery=50, charger_nearby=1.0),
            self._rec(5, mode=4, slack=4.0, battery=90, just_charged=1.0),
        ]

        summary = benchmark._charge_timing_summary(records)

        self.assertEqual(summary["min_charger_slack"], -2.0)
        self.assertEqual(summary["charge_event_count_logged"], 1)
        self.assertEqual(summary["negative_slack_rate"], 0.4)
        self.assertEqual(summary["risk_worsening_while_cleaning_rate"], 0.4)
        self.assertEqual(summary["missed_charge_opportunity_rate"], 0.2)
        self.assertEqual(summary["charger_nearby_not_charged_rate"], 0.2)
        self.assertTrue(summary["late_return_before_charge"])

    def test_reward_attribution_exposes_positive_reward_conflict(self):
        records = [
            self._rec(1, reward=0.5, reward_clean=0.5, reward_survive=0.0, reward_cleaning=0.4, reward_idle=-0.1),
            self._rec(2, reward=-0.2, reward_clean=-0.1, reward_survive=-0.1, reward_cleaning=0.1, reward_idle=-0.3),
        ]

        attribution = benchmark._subset_reward_attribution(records, lambda rec: True)

        self.assertEqual(attribution["sample_count"], 2)
        self.assertEqual(attribution["positive_reward_sum_mean"], 0.25)
        self.assertEqual(attribution["negative_reward_sum_mean"], -0.2)
        self.assertEqual(attribution["positive_total_reward_rate"], 0.5)
        self.assertEqual(attribution["conflict_score"], 0.05)

    def test_ai_summary_cards_include_metric_values_and_reward_conflict(self):
        snapshot = {
            "timestamp": "session-test",
            "checkpoint": "checkpoint.pkl",
            "overall": {
                "win_rate": 0.5,
                "reward_attribution": {
                    "low_value_revisit": {
                        "sample_count": 2,
                        "reward_total_mean": 0.1,
                        "positive_total_reward_rate": 0.5,
                        "conflict_score": 0.05,
                        "top_positive_reward_terms": [["reward_cleaning", 0.25]],
                        "top_negative_reward_terms": [["reward_idle", -0.2]],
                    }
                },
            },
            "per_round": {},
            "episodes": [
                {
                    "episode_id": "round_1_map1",
                    "result": "completed",
                    "anomaly_summary": {
                        "low_value_revisit_rate": 0.2,
                        "redundant_clean_path_rate": 0.1,
                        "positive_reward_while_no_progress_rate": 0.3,
                    },
                    "evidence_windows": {
                        "first_late_return_window": [{"step": 1}],
                        "first_low_value_revisit_window": [{"step": 7}],
                    },
                }
            ],
        }

        summary = benchmark._build_ai_summary(snapshot)

        card = summary["diagnosis_cards"]["low_value_revisit"]
        self.assertEqual(card["episode_count"], 1)
        self.assertEqual(card["example_episode_id"], "round_1_map1")
        self.assertEqual(card["primary_metric_values"]["low_value_revisit_rate"], 0.2)
        self.assertEqual(card["reward_conflict"]["positive_total_reward_rate"], 0.5)
        self.assertEqual(card["reward_conflict"]["conflict_score"], 0.05)
        self.assertEqual(card["first_evidence_window"], "first_low_value_revisit_window")
        self.assertIn("low_value_revisit", summary["issue_index"])
        self.assertEqual(summary["issue_index"]["low_value_revisit"]["primary_metric_values"]["low_value_revisit_rate"], 0.2)

    def _rec(
        self,
        step,
        *,
        mode=2,
        slack=0.0,
        battery=100,
        battery_max=100,
        recoverability=1.0,
        just_charged=0.0,
        risk_worsening=0.0,
        missed_charge=0.0,
        charger_nearby=0.0,
        reward=0.0,
        reward_clean=0.0,
        reward_survive=0.0,
        reward_cleaning=0.0,
        reward_idle=0.0,
    ):
        rec = {
            "step": step,
            "mode": mode,
            "battery": battery,
            "battery_max": battery_max,
            "battery_ratio": round(float(battery) / float(max(battery_max, 1)), 4),
            "charger_slack": slack,
            "future_recoverability_score": recoverability,
            "just_charged": just_charged,
            "risk_worsening_while_cleaning": risk_worsening,
            "missed_charge_opportunity": missed_charge,
            "charger_nearby_not_charged": charger_nearby,
            "reward": reward,
            "reward_clean": reward_clean,
            "reward_survive": reward_survive,
            "reward_cleaning": reward_cleaning,
            "reward_idle": reward_idle,
            "anomalies": {},
        }
        for key in benchmark.REWARD_COMPONENT_KEYS:
            rec.setdefault(f"reward_{key}", 0.0)
        return rec


if __name__ == "__main__":
    unittest.main()

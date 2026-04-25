#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "train" / "tools"
sys.path.insert(0, str(TOOLS_DIR))


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


corpus_tool = _load_module("build_fixed_observation_corpus", TOOLS_DIR / "build_fixed_observation_corpus.py")
compare_tool = _load_module("compare_fixed_observations", TOOLS_DIR / "compare_fixed_observations.py")
REQUIRED_MAPS = corpus_tool.REQUIRED_MAPS
REQUIRED_TAGS = corpus_tool.REQUIRED_TAGS
coverage_summary = corpus_tool.coverage_summary
infer_tags = corpus_tool.infer_tags
validate_corpus = corpus_tool.validate_corpus
compare_corpora = compare_tool.compare_corpora


def _fixture() -> dict[str, Any]:
    return json.loads((REPO_ROOT / "train" / "tools" / "fixtures" / "fixed_observation_corpus.json").read_text(encoding="utf-8"))


class FixedObservationToolTests(unittest.TestCase):
    def test_fixture_corpus_covers_required_maps_and_tags(self):
        corpus = _fixture()
        self.assertEqual(validate_corpus(corpus), [])
        coverage = coverage_summary(corpus)
        self.assertEqual(coverage["missing_maps"], [])
        self.assertEqual(coverage["missing_tags"], [])
        self.assertEqual(coverage["maps_present"], list(REQUIRED_MAPS))
        self.assertEqual(set(coverage["tags_present"]), set(REQUIRED_TAGS))

    def test_infer_tags_handles_existing_episode_shape(self):
        record = {
            "step": 1,
            "battery": 200,
            "battery_max": 200,
            "charger_candidates": [{"center": [1, 2]}],
            "nearest_npc_dist": 4,
            "zero_progress_streak": 3,
            "cleaned_this_step": 0,
            "target_progress_delta": 0,
        }
        tags = infer_tags(record, first_step=1)
        self.assertIn("start", tags)
        self.assertIn("charger_visible", tags)
        self.assertIn("npc_near", tags)
        self.assertIn("no_progress", tags)

    def test_comparator_blocks_unintended_action_drift_for_information_additive(self):
        baseline = _fixture()
        candidate = copy.deepcopy(baseline)
        candidate["nodes"][0]["observation"]["action_or_logit"]["action"] = 4
        report = compare_corpora(
            baseline,
            candidate,
            modification_class="P1_information_additive",
            intended_diff_fields={"feature_diff"},
        )
        self.assertTrue(report["action_or_logit_diff"]["changed"])
        self.assertFalse(report["promotion_allowed"])
        self.assertEqual(report["unintended_changes"][0]["reason"], "outside_modification_class")

    def test_comparator_allows_declared_feature_diff_for_information_additive(self):
        baseline = _fixture()
        candidate = copy.deepcopy(baseline)
        candidate["nodes"][0]["observation"]["features"]["battery"] = 199
        report = compare_corpora(
            baseline,
            candidate,
            modification_class="P1_information_additive",
            intended_diff_fields={"feature_diff"},
        )
        self.assertTrue(report["feature_diff"]["changed"])
        self.assertTrue(report["promotion_allowed"])
        self.assertEqual(len(report["intended_changes"]), 1)

    def test_comparator_allows_declared_p2_eval_only_action_and_override_diffs(self):
        baseline = _fixture()
        candidate = copy.deepcopy(baseline)
        candidate["nodes"][0]["observation"]["action_or_logit"]["action"] = 4
        candidate["nodes"][0]["observation"]["action_or_logit"]["greedy_action"] = 4
        candidate["nodes"][0]["observation"]["override"]["fallback_to_chebyshev"] = 1
        report = compare_corpora(
            baseline,
            candidate,
            modification_class="P2_eval_only_safety",
            intended_diff_fields={"action_or_logit_diff", "override_diff"},
        )
        self.assertTrue(report["action_or_logit_diff"]["changed"])
        self.assertTrue(report["override_diff"]["changed"])
        self.assertTrue(report["promotion_allowed"])
        self.assertEqual(len(report["intended_changes"]), 2)
        self.assertEqual(report["unintended_changes"], [])

    def test_comparator_blocks_undeclared_p2_eval_only_action_and_override_drift(self):
        baseline = _fixture()
        candidate = copy.deepcopy(baseline)
        candidate["nodes"][0]["observation"]["action_or_logit"]["action"] = 4
        candidate["nodes"][0]["observation"]["override"]["fallback_to_chebyshev"] = 1
        report = compare_corpora(
            baseline,
            candidate,
            modification_class="P2_eval_only_safety",
            intended_diff_fields=set(),
        )
        self.assertTrue(report["action_or_logit_diff"]["changed"])
        self.assertTrue(report["override_diff"]["changed"])
        self.assertFalse(report["promotion_allowed"])
        self.assertEqual({change["reason"] for change in report["unintended_changes"]}, {"not_declared_intended"})


if __name__ == "__main__":
    unittest.main()

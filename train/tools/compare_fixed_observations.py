#!/usr/bin/env python3
"""Compare fixed-observation baseline and candidate corpora."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DIFF_FIELDS = (
    "feature_diff",
    "guidance_diff",
    "teacher_mask_diff",
    "action_or_logit_diff",
    "override_diff",
)
SECTION_FOR_DIFF = {
    "feature_diff": "features",
    "guidance_diff": "guidance",
    "teacher_mask_diff": "teacher_mask",
    "action_or_logit_diff": "action_or_logit",
    "override_diff": "override",
}
ALLOWED_BY_MODIFICATION_CLASS: dict[str, set[str]] = {
    "P0_observe_only": set(),
    "P1_information_additive": {"feature_diff", "guidance_diff", "teacher_mask_diff"},
    "P2_eval_only_safety": {"action_or_logit_diff", "override_diff"},
    "R1_small_threshold": {"guidance_diff", "teacher_mask_diff", "override_diff"},
    "R2_reward_positive": {"feature_diff", "guidance_diff"},
    "R3_reward_penalty": {"feature_diff", "guidance_diff"},
    "R4_light_refactor": set(DIFF_FIELDS),
    "R5_architecture": set(DIFF_FIELDS),
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _node_key(node: dict[str, Any]) -> str:
    return str(node.get("node_id") or f"map{node.get('map_id')}-{node.get('round')}-step{node.get('step')}")


def _index_nodes(corpus: dict[str, Any]) -> dict[str, dict[str, Any]]:
    nodes = corpus.get("nodes") or []
    return {_node_key(node): node for node in nodes if isinstance(node, dict)}


def _section(node: dict[str, Any] | None, section: str) -> Any:
    if not node:
        return None
    observation = node.get("observation")
    if not isinstance(observation, dict):
        return None
    return observation.get(section)


def _compare_values(before: Any, after: Any) -> dict[str, Any]:
    if before == after:
        return {"changed": False, "changed_keys": [], "before": before, "after": after}
    changed_keys: list[str] = []
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(set(before) | set(after)):
            if before.get(key) != after.get(key):
                changed_keys.append(str(key))
    else:
        changed_keys = ["<value>"]
    return {"changed": True, "changed_keys": changed_keys, "before": before, "after": after}


def compare_corpora(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    modification_class: str,
    intended_diff_fields: set[str] | None = None,
) -> dict[str, Any]:
    if modification_class not in ALLOWED_BY_MODIFICATION_CLASS:
        raise ValueError(f"unsupported modification_class {modification_class!r}")
    intended_diff_fields = intended_diff_fields or set()
    allowed = ALLOWED_BY_MODIFICATION_CLASS[modification_class]

    baseline_nodes = _index_nodes(baseline)
    candidate_nodes = _index_nodes(candidate)
    node_ids = sorted(set(baseline_nodes) | set(candidate_nodes))
    node_reports: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, Any]] = {field: {"changed": False, "node_count": 0, "nodes": []} for field in DIFF_FIELDS}
    intended_changes: list[dict[str, Any]] = []
    unintended_changes: list[dict[str, Any]] = []

    for node_id in node_ids:
        base_node = baseline_nodes.get(node_id)
        cand_node = candidate_nodes.get(node_id)
        node_report: dict[str, Any] = {"node_id": node_id, "missing_in_baseline": base_node is None, "missing_in_candidate": cand_node is None}
        for diff_field, section in SECTION_FOR_DIFF.items():
            diff = _compare_values(_section(base_node, section), _section(cand_node, section))
            node_report[diff_field] = diff
            if not diff["changed"]:
                continue
            aggregate[diff_field]["changed"] = True
            aggregate[diff_field]["node_count"] = int(aggregate[diff_field]["node_count"]) + 1
            aggregate_nodes = aggregate[diff_field]["nodes"]
            if isinstance(aggregate_nodes, list):
                aggregate_nodes.append(node_id)
            change = {"node_id": node_id, "diff_field": diff_field, "changed_keys": diff["changed_keys"]}
            if diff_field in intended_diff_fields and diff_field in allowed:
                intended_changes.append(change)
            else:
                reason = "not_declared_intended" if diff_field in allowed else "outside_modification_class"
                unintended_changes.append({**change, "reason": reason})
        node_reports.append(node_report)

    return {
        "schema_version": 1,
        "generated_by": "train/tools/compare_fixed_observations.py",
        "modification_class": modification_class,
        "declared_intended_diff_fields": sorted(intended_diff_fields),
        "allowed_diff_fields_for_class": sorted(allowed),
        "node_count": len(node_ids),
        "missing_nodes": {
            "baseline_only": sorted(set(baseline_nodes) - set(candidate_nodes)),
            "candidate_only": sorted(set(candidate_nodes) - set(baseline_nodes)),
        },
        "feature_diff": aggregate["feature_diff"],
        "guidance_diff": aggregate["guidance_diff"],
        "teacher_mask_diff": aggregate["teacher_mask_diff"],
        "action_or_logit_diff": aggregate["action_or_logit_diff"],
        "override_diff": aggregate["override_diff"],
        "intended_changes": intended_changes,
        "unintended_changes": unintended_changes,
        "promotion_allowed": not unintended_changes,
        "node_diffs": node_reports,
    }


def cmd_compare(args: argparse.Namespace) -> int:
    report = compare_corpora(
        _load_json(Path(args.baseline)),
        _load_json(Path(args.candidate)),
        args.modification_class,
        set(args.intended_diff),
    )
    if args.output:
        _write_json(Path(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["promotion_allowed"] or args.allow_blocked_exit_zero else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare fixed-observation baseline and candidate corpora.")
    parser.add_argument("--baseline", required=True, help="Baseline corpus JSON")
    parser.add_argument("--candidate", required=True, help="Candidate corpus JSON")
    parser.add_argument("--modification-class", required=True, choices=sorted(ALLOWED_BY_MODIFICATION_CLASS))
    parser.add_argument(
        "--intended-diff",
        action="append",
        default=[],
        choices=DIFF_FIELDS,
        help="Diff field intentionally changed by this candidate; repeat as needed",
    )
    parser.add_argument("--output", help="Optional comparison report JSON path")
    parser.add_argument(
        "--allow-blocked-exit-zero",
        action="store_true",
        help="Still write promotion_allowed=false, but return exit 0 for evidence-generation workflows.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return cmd_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())

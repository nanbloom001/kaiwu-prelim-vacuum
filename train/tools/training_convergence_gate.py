#!/usr/bin/env python3
"""Deterministic fixed-window convergence gate for training comparisons.

The gate consumes JSON shaped like ``compare_training_runs.py --json`` and writes
machine-readable decisions for the existing Linux sample nodes.  It does not
start, stop, or mutate training; it only documents convergence definitions and
early-stop decisions from already captured fixed-window metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / ".sisyphus/evidence/benchmark-900/wave0/training-convergence-gate.json"

SAMPLE_POINTS: tuple[str, ...] = (
    "bootstrap_10",
    "bootstrap_20",
    "global_40",
    "global_80",
    "global_120",
    "global_160",
    "global_200",
)
NODE_EPISODES = {
    "bootstrap_10": 10,
    "bootstrap_20": 20,
    "global_40": 40,
    "global_80": 80,
    "global_120": 120,
    "global_160": 160,
    "global_200": 200,
}
ALLOWED_DECISIONS: tuple[str, ...] = (
    "keep",
    "revert",
    "continue_to_global_80",
    "targeted_resume",
    "escalate",
)
GLOBAL40_NODE = "global_40"

REGRESSION_THRESHOLDS: dict[str, dict[str, Any]] = {
    "avg_clean_per_step": {
        "direction": "higher_is_better",
        "threshold": 0.05,
        "reason": "local CPS must not drop more than 0.05 at the main global_40 judgment point",
    },
    "battery_fail_rate": {
        "direction": "lower_is_better",
        "threshold": 0.05,
        "reason": "battery failure rate must not rise more than 5 percentage points",
    },
    "zero_charge_battery_fail_rate": {
        "direction": "lower_is_better",
        "threshold": 0.05,
        "reason": "zero-charge battery failures are a primary survival regression signal",
    },
    "route_phase_return_stall_rate": {
        "direction": "lower_is_better",
        "threshold": 0.05,
        "reason": "route-phase return stall must not worsen materially in the local window",
    },
    "planner_policy_divergence_rate": {
        "direction": "lower_is_better",
        "threshold": 0.05,
        "reason": "planner-policy divergence is a known blocker and must not worsen materially",
    },
}


CONVERGENCE_DEFINITIONS = {
    "basic": {
        "definition": "directionally stable by global_40 using the existing local-window metrics",
        "nodes": ["bootstrap_10", "bootstrap_20", "global_40"],
        "window_policy": "bootstrap nodes use local_10; global_40 uses local_20",
        "gate": "global_40 emits exactly one allowed decision",
    },
    "full": {
        "definition": "stable or plateaued by global_80/global_120 after passing the global_40 main judgment",
        "nodes": ["global_80", "global_120"],
        "window_policy": "global nodes use local_20; prefix metrics are auxiliary diagnostics only",
        "late_recovery": "global_160/global_200 are late recovery checks only and do not wash out a failed global_40",
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _node_payload(payload: dict[str, Any], node: str) -> dict[str, Any] | None:
    sample_points = payload.get("sample_points") or {}
    node_payload = sample_points.get(node)
    return node_payload if isinstance(node_payload, dict) else None


def _sample_metrics(sample: dict[str, Any] | None) -> dict[str, Any] | None:
    if not sample:
        return None
    local_metrics = sample.get("local_metrics")
    if isinstance(local_metrics, dict) and local_metrics:
        merged = dict(local_metrics)
    else:
        metrics = sample.get("metrics")
        merged = dict(metrics) if isinstance(metrics, dict) else {}
    learning_metrics = sample.get("learning_metrics")
    if isinstance(learning_metrics, dict):
        merged.update(learning_metrics)
    return merged or None


def _metrics_for(payload: dict[str, Any], node: str, side: str) -> dict[str, Any] | None:
    node_payload = _node_payload(payload, node)
    if not node_payload:
        side_payload = payload.get(side)
        if isinstance(side_payload, dict):
            side_node = (side_payload.get("sample_points") or {}).get(node)
            return _sample_metrics(side_node)
        return None

    side_payload = node_payload.get(side)
    if isinstance(side_payload, dict):
        return _sample_metrics(side_payload)


    nested_side = payload.get(side)
    if isinstance(nested_side, dict):
        side_node = (nested_side.get("sample_points") or {}).get(node)
        return _sample_metrics(side_node)
    return None


def _available_nodes(payload: dict[str, Any]) -> list[str]:
    nodes: list[str] = []
    for node in SAMPLE_POINTS:
        if _node_payload(payload, node) or _metrics_for(payload, node, "target"):
            nodes.append(node)
    return nodes


def _metric_delta(metric: str, target_value: float, baseline_value: float) -> float:
    return target_value - baseline_value


def _is_regression(metric: str, delta: float) -> bool:
    spec = REGRESSION_THRESHOLDS[metric]
    threshold = float(spec["threshold"])
    if spec["direction"] == "higher_is_better":
        return delta <= -threshold
    return delta >= threshold


def _is_improvement(metric: str, delta: float) -> bool:
    spec = REGRESSION_THRESHOLDS[metric]
    threshold = float(spec["threshold"])
    if spec["direction"] == "higher_is_better":
        return delta >= threshold
    return delta <= -threshold


def _metric_record(metric: str, target_value: float, baseline_value: float) -> dict[str, Any]:
    delta = _metric_delta(metric, target_value, baseline_value)
    spec = REGRESSION_THRESHOLDS[metric]
    return {
        "metric": metric,
        "baseline": round(float(baseline_value), 6),
        "target": round(float(target_value), 6),
        "delta": round(float(delta), 6),
        "direction": spec["direction"],
        "threshold": float(spec["threshold"]),
        "threshold_rule": spec["reason"],
    }


def classify_metrics(target_metrics: dict[str, Any] | None, baseline_metrics: dict[str, Any] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    regressed: list[dict[str, Any]] = []
    improved: list[dict[str, Any]] = []
    missing: list[str] = []
    for metric in REGRESSION_THRESHOLDS:
        target_value = _as_float((target_metrics or {}).get(metric))
        baseline_value = _as_float((baseline_metrics or {}).get(metric))
        if target_value is None or baseline_value is None:
            missing.append(metric)
            continue
        record = _metric_record(metric, target_value, baseline_value)
        delta = float(record["delta"])
        if _is_regression(metric, delta):
            regressed.append(record)
        elif _is_improvement(metric, delta):
            improved.append(record)
    return regressed, improved, missing


def _continuation_justification(improved_metrics: list[dict[str, Any]], regressed_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if regressed_metrics:
        return []
    return [
        {
            "metric": item["metric"],
            "baseline": item["baseline"],
            "target": item["target"],
            "delta": item["delta"],
            "threshold": item["threshold"],
            "direction": item["direction"],
            "justification": (
                f"{item['metric']} improved by {abs(float(item['delta'])):.6f}, "
                f"meeting threshold {float(item['threshold']):.6f} for {item['direction']}"
            ),
        }
        for item in improved_metrics
    ]


def decide_global40(payload: dict[str, Any]) -> dict[str, Any]:
    target_metrics = _metrics_for(payload, GLOBAL40_NODE, "target")
    baseline_metrics = _metrics_for(payload, GLOBAL40_NODE, "baseline")
    regressed_metrics, improved_metrics, missing_metrics = classify_metrics(target_metrics, baseline_metrics)
    continuation = _continuation_justification(improved_metrics, regressed_metrics)

    if target_metrics is None or baseline_metrics is None:
        decision = "escalate"
        reason = "global_40 baseline or target local metrics are missing; no deterministic gate decision can be made"
    elif len(regressed_metrics) >= 2:
        decision = "revert"
        reason = "global_40 worsened at least two configured metrics beyond thresholds"
    elif regressed_metrics:
        decision = "targeted_resume"
        reason = "global_40 has one threshold regression; resume should target the named blocker before promotion"
    elif continuation:
        decision = "continue_to_global_80"
        reason = "global_40 has exact metric/threshold improvement justification for continuing to global_80"
    else:
        decision = "keep"
        reason = "global_40 is directionally stable without threshold regressions or promotion-strength improvements"

    return {
        "decision": decision,
        "node": GLOBAL40_NODE,
        "regressed_metrics": regressed_metrics,
        "improved_metrics": improved_metrics,
        "thresholds": REGRESSION_THRESHOLDS,
        "rationale": {
            "summary": reason,
            "basic_convergence": CONVERGENCE_DEFINITIONS["basic"],
            "full_convergence": CONVERGENCE_DEFINITIONS["full"],
            "continuation_justification": continuation,
            "missing_metrics": missing_metrics,
            "allowed_decisions": list(ALLOWED_DECISIONS),
        },
    }


def build_gate_report(payload: dict[str, Any]) -> dict[str, Any]:
    global40_decision = decide_global40(payload)
    report = {
        "decision": global40_decision["decision"],
        "node": global40_decision["node"],
        "regressed_metrics": global40_decision["regressed_metrics"],
        "improved_metrics": global40_decision["improved_metrics"],
        "thresholds": global40_decision["thresholds"],
        "rationale": global40_decision["rationale"],
        "definitions": CONVERGENCE_DEFINITIONS,
        "sample_points": {
            "supported": list(SAMPLE_POINTS),
            "available": _available_nodes(payload),
            "episode_thresholds": NODE_EPISODES,
            "local_window_semantics": {
                "bootstrap_10": "local_10",
                "bootstrap_20": "local_10",
                "global_40": "local_20",
                "global_80": "local_20",
                "global_120": "local_20",
                "global_160": "local_20 late_recovery_only",
                "global_200": "local_20 late_recovery_only",
            },
        },
    }
    if report["decision"] not in ALLOWED_DECISIONS:
        raise ValueError(f"unsupported decision {report['decision']!r}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Define and apply the fixed-window training convergence gate. "
            "Basic convergence means directionally stable by global_40; full convergence means stable or plateaued "
            "by global_80/global_120; global_160/global_200 are late-recovery checks only."
        )
    )
    parser.add_argument("--input", type=Path, help="Input JSON from compare_training_runs.py --json or an equivalent fixture")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Output JSON path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--print-definitions", action="store_true", help="Print supported nodes and basic/full convergence definitions as JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.print_definitions:
        print(
            json.dumps(
                {
                    "definitions": CONVERGENCE_DEFINITIONS,
                    "supported_sample_points": list(SAMPLE_POINTS),
                    "allowed_decisions": list(ALLOWED_DECISIONS),
                    "thresholds": REGRESSION_THRESHOLDS,
                },
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not args.input:
        raise SystemExit("--input is required unless --print-definitions is used")

    report = build_gate_report(_read_json(args.input))
    _write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

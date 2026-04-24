#!/usr/bin/env python3
"""
Analyze fixed holdout benchmark runner output into JSON and optional Markdown.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze holdout benchmark output.")
    parser.add_argument("--input", required=True, help="Path to runner JSON output.")
    parser.add_argument("--output-md", default=None, help="Optional path to write Markdown summary.")
    return parser.parse_args()


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def coerce_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_fail_reason(episode: dict[str, object]) -> str:
    for key in ("fail_reason", "done_reason", "status"):
        value = episode.get(key)
        if value:
            return str(value)
    return "unknown"


def build_metrics(episodes: list[dict[str, object]]) -> dict[str, object]:
    if not episodes:
        return {
            "status": "NO_EPISODES",
            "episode_count": 0,
            "avg_clean_score": 0.0,
            "score_p10": 0.0,
            "score_p50": 0.0,
            "score_p90": 0.0,
            "completed_rate": 0.0,
            "battery_fail_rate": 0.0,
            "collision_fail_rate": 0.0,
            "avg_clean_per_step": 0.0,
            "avg_finished_steps": 0.0,
            "avg_charge_count": 0.0,
            "avg_remaining_charge": 0.0,
        }

    clean_scores = [coerce_float(ep.get("clean_score", ep.get("score", 0.0))) for ep in episodes]
    finished_steps = [coerce_float(ep.get("finished_steps", ep.get("step", 0.0))) for ep in episodes]
    charge_counts = [coerce_float(ep.get("charge_count", 0.0)) for ep in episodes]
    remaining_charge = [coerce_float(ep.get("remaining_charge", ep.get("battery", 0.0))) for ep in episodes]
    clean_per_step = []
    for score, steps in zip(clean_scores, finished_steps):
        clean_per_step.append(score / steps if steps > 0 else 0.0)

    fail_reasons = [normalize_fail_reason(ep) for ep in episodes]
    return {
        "status": "OK",
        "episode_count": len(episodes),
        "avg_clean_score": round(mean(clean_scores), 4),
        "score_p10": round(percentile(clean_scores, 0.10), 4),
        "score_p50": round(percentile(clean_scores, 0.50), 4),
        "score_p90": round(percentile(clean_scores, 0.90), 4),
        "completed_rate": round(mean([1.0 if reason == "completed" else 0.0 for reason in fail_reasons]), 4),
        "battery_fail_rate": round(mean([1.0 if reason == "battery" else 0.0 for reason in fail_reasons]), 4),
        "collision_fail_rate": round(mean([1.0 if reason == "collision" else 0.0 for reason in fail_reasons]), 4),
        "avg_clean_per_step": round(mean(clean_per_step), 6),
        "avg_finished_steps": round(mean(finished_steps), 4),
        "avg_charge_count": round(mean(charge_counts), 4),
        "avg_remaining_charge": round(mean(remaining_charge), 4),
    }


def derive_risks(payload: dict[str, object], combined: dict[str, object]) -> list[dict[str, object]]:
    risks = list(payload.get("risks") or [])
    mutation_guard = payload.get("model_mutation_guard") or {}
    if mutation_guard.get("mutation_detected"):
        risks.append(
            {
                "code": "MODEL_MUTATION_DETECTED",
                "severity": "error",
                "message": "Tracked model/checkpoint files drifted between pre/post snapshots.",
            }
        )
    if combined.get("status") == "NO_EPISODES":
        risks.append(
            {
                "code": "NO_EPISODES",
                "severity": "warning",
                "message": "Runner output contains no executed episodes yet; decision inputs are contract-only.",
            }
        )
    return risks


def render_markdown(result: dict[str, object]) -> str:
    combined = result["combined"]
    per_map = result["per_map"]
    lines = [
        "# Holdout Benchmark Analysis",
        "",
        f"- Run ID: `{result['run_id']}`",
        f"- Checkpoint: `{(result.get('checkpoint') or {}).get('path')}`",
        f"- Maps: `{result['maps']}`",
        f"- Episodes per map: `{result['episodes_per_map']}`",
        f"- Combined status: `{combined['status']}`",
        "",
        "## Combined",
        "",
        f"- Episode count: `{combined['episode_count']}`",
        f"- Avg clean score: `{combined['avg_clean_score']}`",
        f"- P10 / P50 / P90: `{combined['score_p10']}` / `{combined['score_p50']}` / `{combined['score_p90']}`",
        f"- Completed / Battery fail / Collision fail: `{combined['completed_rate']}` / `{combined['battery_fail_rate']}` / `{combined['collision_fail_rate']}`",
        "",
        "## Per Map",
        "",
        "| Map | Status | Episodes | Avg Score | P10 | P90 | Battery Fail | Collision Fail |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for map_id in sorted(per_map, key=lambda item: int(item)):
        row = per_map[map_id]
        lines.append(
            "| {map_id} | {status} | {episode_count} | {avg_clean_score} | {score_p10} | {score_p90} | {battery_fail_rate} | {collision_fail_rate} |".format(
                map_id=map_id,
                **row,
            )
        )

    risks = result.get("risks") or []
    lines.extend(["", "## Risks", ""])
    if risks:
        for risk in risks:
            lines.append(f"- `{risk.get('severity', 'info')}` `{risk.get('code', 'UNKNOWN')}`: {risk.get('message', '')}")
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))

    episodes = list(payload.get("episodes") or [])
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for episode in episodes:
        map_id = str(episode.get("map_id", "unknown"))
        grouped[map_id].append(episode)

    contract = payload.get("contract") or {}
    configured_maps = [str(map_id) for map_id in (contract.get("maps") or payload.get("maps") or [])]
    for map_id in configured_maps:
        grouped.setdefault(map_id, [])

    combined = build_metrics(episodes)
    per_map = {map_id: build_metrics(map_episodes) for map_id, map_episodes in sorted(grouped.items(), key=lambda item: int(item[0]))}

    result = {
        "run_id": payload.get("run_id"),
        "checkpoint": payload.get("checkpoint"),
        "maps": contract.get("maps") or payload.get("maps") or [],
        "episodes_per_map": contract.get("episodes_per_map") or payload.get("episodes_per_map"),
        "fixed_config": contract.get("fixed_config") or {},
        "combined": combined,
        "per_map": per_map,
        "risks": derive_risks(payload, combined),
        "decision_inputs": payload.get("decision_inputs") or {},
    }

    if args.output_md:
        output_md = Path(args.output_md).resolve()
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(result), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

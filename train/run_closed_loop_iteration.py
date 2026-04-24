#!/usr/bin/env python3
"""Safe Train -> Benchmark -> Analyze -> Accept/Rollback loop planner.

This runner is intentionally conservative: dry-run prints the exact command
templates for a closed-loop iteration, and decision mode evaluates existing
metrics/analysis JSON only. It never starts Docker unless a future operator
copies a printed template manually.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_CHECKPOINT = "code/model.ckpt-resume.pkl"
DEFAULT_HOLDOUT_OUTPUT = "train/context/HOLDOUT_BENCHMARK_LOOP.json"
DEFAULT_ANALYSIS_MD = "train/context/HOLDOUT_BENCHMARK_LOOP.md"
HOLDOUT_MAPS = "4,7"
EPISODES_PER_MAP = 10
T4_COMMIT = "5d578b9"
T5_COMMIT = "88ad9c8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan and decide safe closed-loop training/holdout iterations."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the safe loop plan and command templates without starting Docker.",
    )
    parser.add_argument(
        "--decision-json",
        default=None,
        help="Metrics/analysis JSON to evaluate into ACCEPT, REJECT, or CONTINUE/ESCALATE.",
    )
    parser.add_argument(
        "--until-score-gt",
        type=float,
        default=900.0,
        help="Strict holdout combined average score target. Default: 900.",
    )
    parser.add_argument(
        "--candidate-commit",
        default="HEAD",
        help="Commit to name in rollback/revert commands. Default: HEAD.",
    )
    parser.add_argument(
        "--consecutive-nonwinning-rounds",
        type=int,
        default=0,
        help="Number of consecutive rounds whose holdout combined average did not exceed the target.",
    )
    parser.add_argument(
        "--holdout-output",
        default=DEFAULT_HOLDOUT_OUTPUT,
        help="Benchmark JSON path used in printed command templates.",
    )
    parser.add_argument(
        "--analysis-md",
        default=DEFAULT_ANALYSIS_MD,
        help="Analyzer Markdown path used in printed command templates.",
    )
    return parser.parse_args()


def run_git(args: list[str]) -> str:
    env = dict(os.environ)
    env["GIT_MASTER"] = "1"
    try:
        completed = subprocess.run(
            ["git", *args],
            check=True,
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"
    return completed.stdout.strip() or "UNKNOWN"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_block(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if isinstance(value, dict):
        combined = value.get("combined")
        if isinstance(combined, dict):
            return combined
        return value
    return {}


def candidate_block(payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("candidate", "current", "analysis", "result"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return payload


def combined_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    combined = payload.get("combined")
    return combined if isinstance(combined, dict) else payload


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def count_high_score_battery_deaths(payload: dict[str, Any]) -> float:
    classification = payload.get("failure_classification")
    if not isinstance(classification, dict):
        return 0.0
    categories = classification.get("categories")
    if not isinstance(categories, dict):
        return 0.0
    bucket = categories.get("high_score_battery_death")
    if not isinstance(bucket, dict):
        return 0.0
    return as_float(bucket.get("count"))


def infrastructure_reject_reasons(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    mutation_guard = payload.get("model_mutation_guard")
    if isinstance(mutation_guard, dict) and mutation_guard.get("mutation_detected"):
        reasons.append("eval model mutation detected by benchmark mutation guard")

    for risk in payload.get("risks") or []:
        if not isinstance(risk, dict):
            continue
        code = str(risk.get("code", "")).upper()
        severity = str(risk.get("severity", "")).lower()
        message = str(risk.get("message", ""))
        combined = f"{code} {message}".upper()
        if "LEAK" in combined:
            reasons.append("holdout leakage reported by benchmark/analyzer risk")
        if "MODEL_MUTATION" in code or ("MUTATION" in combined and severity == "error"):
            reasons.append("eval model mutation reported by benchmark/analyzer risk")
    return sorted(set(reasons))


def escalation_ladder(rounds: int) -> list[str]:
    ladder = [
        "1. pure-positive / low-risk: parameter-only or guard-only adjustments",
        "2. reward add/rewrite: add or reshape targeted reward signals",
        "3. small refactor: local behavior/planner cleanup with bounded blast radius",
        "4. network only last: architecture/model changes after safer levers fail",
    ]
    if rounds >= 3:
        return ["Escalate now: 3+ consecutive non->900 rounds reached.", *ladder]
    return [f"Do not escalate yet: {rounds}/3 consecutive non->900 rounds.", *ladder]


def rollback_command(commit: str) -> str:
    return f"git revert {commit}"


def print_dry_run(args: argparse.Namespace) -> int:
    head = run_git(["rev-parse", "--short", "HEAD"])
    branch = run_git(["branch", "--show-current"])
    status = run_git(["status", "--short"])
    status_text = status if status != "UNKNOWN" and status else "clean"

    benchmark_command = (
        f"python train/run_holdout_benchmark.py --maps {HOLDOUT_MAPS} "
        f"--episodes-per-map {EPISODES_PER_MAP} --checkpoint {DEFAULT_CHECKPOINT} "
        f"--output {args.holdout_output} --dry-run"
    )
    analyzer_command = (
        f"python train/analyze_holdout_benchmark.py --input {args.holdout_output} "
        f"--output-md {args.analysis_md}"
    )

    print("CLOSED LOOP DRY RUN - no Docker command is executed")
    print(f"Git anchor: branch={branch} head={head}")
    print("Dirty status snapshot:")
    print(status_text)
    print("")
    print("Stop condition:")
    print(f"- Continue until holdout combined avg > {args.until_score_gt:.1f}; no fixed max-round stop is allowed.")
    print("")
    print("Training command templates:")
    print("- QUICK 20-minute template:")
    print("  cd train")
    print("  docker compose -p kaiwu-train -f .docker-compose.yaml --profile distributed up -d")
    print("  # observe for 20 minutes, then stop explicitly")
    print("  docker compose -p kaiwu-train -f .docker-compose.yaml --profile distributed down")
    print("- FULL 50-minute template:")
    print("  cd train")
    print("  docker compose -p kaiwu-train -f .docker-compose.yaml --profile distributed up -d")
    print("  # observe for 50 minutes, then stop explicitly")
    print("  docker compose -p kaiwu-train -f .docker-compose.yaml --profile distributed down")
    print("")
    print("Holdout benchmark command template:")
    print(f"- {benchmark_command}")
    print("Analyzer command template:")
    print(f"- {analyzer_command}")
    print("Rollback/revert command template:")
    print(f"- {rollback_command(args.candidate_commit)}")
    print("")
    print("Candidate separation requirement:")
    print(f"- If commits {T4_COMMIT} and {T5_COMMIT} are both present, benchmark/evaluate them separately before combining effects.")
    print("")
    print("Accept thresholds:")
    print("- ACCEPT if holdout combined avg improves >=20 or exceeds 900, completed rate drop <=0.05, and clean_per_step drop <=2%.")
    print("Reject thresholds:")
    print("- REJECT on battery fail rate increase >0.05, collision fail rate increase >0.02, high-score battery deaths increase, holdout leakage, or eval model mutation.")
    print("")
    print("Optimization escalation ladder:")
    for item in escalation_ladder(args.consecutive_nonwinning_rounds):
        print(f"- {item}")
    return 0


def decide(args: argparse.Namespace) -> int:
    path = Path(args.decision_json).resolve()
    payload = load_json(path)
    candidate_payload = candidate_block(payload)
    candidate = combined_metrics(candidate_payload)
    baseline = metric_block(payload, "baseline")
    commit = str(payload.get("candidate_commit") or args.candidate_commit)

    candidate_score = as_float(candidate.get("avg_clean_score"))
    baseline_score = as_float(baseline.get("avg_clean_score"), candidate_score)
    completed_drop = as_float(baseline.get("completed_rate"), as_float(candidate.get("completed_rate"))) - as_float(
        candidate.get("completed_rate")
    )
    baseline_cps = as_float(baseline.get("avg_clean_per_step"), as_float(candidate.get("avg_clean_per_step")))
    candidate_cps = as_float(candidate.get("avg_clean_per_step"))
    cps_drop_ratio = 0.0 if baseline_cps <= 0 else (baseline_cps - candidate_cps) / baseline_cps
    battery_fail_delta = as_float(candidate.get("battery_fail_rate")) - as_float(baseline.get("battery_fail_rate"))
    collision_fail_delta = as_float(candidate.get("collision_fail_rate")) - as_float(baseline.get("collision_fail_rate"))
    high_score_death_delta = count_high_score_battery_deaths(candidate_payload) - count_high_score_battery_deaths(
        payload.get("baseline") if isinstance(payload.get("baseline"), dict) else {}
    )

    reject_reasons = infrastructure_reject_reasons(candidate_payload)
    if battery_fail_delta > 0.05:
        reject_reasons.append(f"battery fail rate increased by {battery_fail_delta:.4f} > 0.05")
    if collision_fail_delta > 0.02:
        reject_reasons.append(f"collision fail rate increased by {collision_fail_delta:.4f} > 0.02")
    if high_score_death_delta > 0:
        reject_reasons.append("high-score battery deaths increased")

    improvement = candidate_score - baseline_score
    acceptance_score_ok = improvement >= 20.0 or candidate_score > args.until_score_gt
    stability_ok = completed_drop <= 0.05 and cps_drop_ratio <= 0.02

    if acceptance_score_ok and completed_drop > 0.05:
        reject_reasons.append(f"completed rate dropped by {completed_drop:.4f} > 0.05")
    if acceptance_score_ok and cps_drop_ratio > 0.02:
        reject_reasons.append(f"clean_per_step dropped by {cps_drop_ratio:.4%} > 2.0000%")

    if reject_reasons:
        decision = "REJECT"
        reason = "; ".join(reject_reasons)
    elif acceptance_score_ok and stability_ok:
        decision = "ACCEPT"
        reason = (
            f"score gate passed (improvement={improvement:.4f}, target_exceeded={candidate_score > args.until_score_gt}) "
            f"with completed drop {completed_drop:.4f} and clean_per_step drop {cps_drop_ratio:.4%} within limits"
        )
    else:
        decision = "CONTINUE/ESCALATE" if args.consecutive_nonwinning_rounds >= 3 else "CONTINUE"
        reason = (
            f"holdout combined avg {candidate_score:.4f} has not exceeded {args.until_score_gt:.4f}; "
            f"improvement={improvement:.4f}, completed_drop={completed_drop:.4f}, clean_per_step_drop={cps_drop_ratio:.4%}"
        )

    print(f"Decision: {decision}")
    print(f"Reason: {reason}")
    print(f"Holdout combined avg: {candidate_score:.4f}")
    print(f"Strict stop condition met: {str(candidate_score > args.until_score_gt).lower()}")
    print(f"Rollback command: {rollback_command(commit)}")
    print("Escalation ladder:")
    for item in escalation_ladder(args.consecutive_nonwinning_rounds):
        print(f"- {item}")
    return 0 if decision in {"ACCEPT", "CONTINUE", "CONTINUE/ESCALATE"} else 2


def main() -> int:
    args = parse_args()
    if args.dry_run:
        return print_dry_run(args)
    if args.decision_json:
        return decide(args)
    print("ERROR: provide --dry-run or --decision-json", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

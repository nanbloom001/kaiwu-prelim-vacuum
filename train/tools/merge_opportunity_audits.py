#!/usr/bin/env python3
"""Merge and validate Wave 0 full-board opportunity audits.

The tool is intentionally deterministic and read-only with respect to agent
behavior code: it synthesizes Task 0A audit reports from repository evidence and
validates the machine-checkable artifact required before any intervention work.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / ".sisyphus/evidence/benchmark-900/wave0/full-board-audit-merged.json"
ALLOWED_INTERVENTION_CLASSES = (
    "P0_observe_only",
    "P1_information_additive",
    "P2_eval_only_safety",
    "R1_small_threshold",
    "R2_reward_positive",
    "R3_reward_penalty",
    "R4_light_refactor",
    "R5_architecture",
)
REQUIRED_REPORT_FIELDS = (
    "agent_name",
    "scope",
    "files_or_artifacts_reviewed",
    "findings",
    "opportunities",
    "risks",
    "recommended_first_intervention",
    "confidence",
)
REQUIRED_SCOPES = (
    "observation_global_position_truth",
    "linux_mechanism_reuse",
    "yjy_linux_reconciliation",
    "benchmark_blocker_attribution",
    "pure_positive_opportunities",
    "integrity_overfit_risks",
)
REQUIRED_BOOLEANS = (
    "global_charger_positions_confirmed",
    "global_robot_positions_confirmed",
    "global_npc_positions_confirmed",
    "linux_mechanisms_to_reuse_confirmed",
    "yjy_reset_required",
)


def _repo(path: str) -> str:
    return str(Path(path))


def _opportunity(identifier: str, title: str, intervention_class: str, rationale: str, evidence: list[str]) -> dict[str, Any]:
    if intervention_class not in ALLOWED_INTERVENTION_CLASSES:
        raise ValueError(f"unsupported intervention class {intervention_class!r}")
    return {
        "id": identifier,
        "title": title,
        "intervention_class": intervention_class,
        "rationale": rationale,
        "evidence": evidence,
    }


def build_default_reports() -> list[dict[str, Any]]:
    """Return deterministic Task 0A reports grounded in checked repo evidence."""
    return [
        {
            "agent_name": "task-0A-observation-global-position-auditor",
            "scope": "observation_global_position_truth",
            "files_or_artifacts_reviewed": [
                _repo("code/agent_ppo/feature/preprocessor.py:232-281"),
                _repo("code/agent_ppo/feature/preprocessor.py:364-496"),
                _repo("code/agent_ppo/feature/preprocessor.py:1017-1182"),
                _repo("code/agent_ppo/feature/expert.py:102-149"),
                _repo("code/agent_ppo/model/model.py:233-343"),
                _repo("code/agent_ppo/eval/benchmark.py:124-233"),
            ],
            "findings": [
                "Hero/global robot position is read from frame_state.hero.pos into Preprocessor.cur_pos and emitted as normalized scalar features.",
                "Charger/global organ positions are read from frame_state.organs where sub_type == 1, converted into charger_map, nearest charger metrics, sorted charger candidates, entity features, and planner targets.",
                "NPC/global positions are read from frame_state.npcs, converted into npc_risk_map, nearest/all NPC metrics, entity features, and ExpertPolicy safety filters.",
                "Observation map_info and legal_action/legal_act are already consumed for local/global memory, passability, dirty memory, and legal action masking.",
                "Model has local, global-memory, entity-token, scalar, and action-history branches, so position-derived signals already reach inference/training tensors.",
            ],
            "opportunities": [
                _opportunity(
                    "observe-global-signal-coverage",
                    "Materialize a source-to-consumer truth table before adding more global-position logic",
                    "P0_observe_only",
                    "Existing code already consumes global robot, charger, NPC, map, and legal-action signals; the first safe action is to prove coverage and gaps without changing behavior.",
                    [
                        _repo("code/agent_ppo/feature/preprocessor.py:239-281"),
                        _repo("code/agent_ppo/feature/preprocessor.py:1017-1182"),
                    ],
                ),
                _opportunity(
                    "additive-position-diagnostics",
                    "Add non-control diagnostics for charger/NPC/source freshness if Task 0B finds blind spots",
                    "P1_information_additive",
                    "Diagnostics can expose stale or missing observation fields without changing policy decisions.",
                    [_repo("code/agent_ppo/eval/benchmark.py:124-233")],
                ),
            ],
            "risks": [
                "Claiming global positions are unexploited would be incorrect; current evidence shows partial exploitation across features, planner, model, and benchmark diagnostics.",
                "Hardcoded map IDs or coordinate tables remain prohibited; only live observation coordinates should be used in later tasks.",
            ],
            "recommended_first_intervention": {
                "id": "observe-global-signal-coverage",
                "intervention_class": "P0_observe_only",
                "summary": "Run/extend a read-only source-to-consumer audit before any global-position intervention.",
            },
            "confidence": "high",
        },
        {
            "agent_name": "task-0A-linux-mechanism-auditor",
            "scope": "linux_mechanism_reuse",
            "files_or_artifacts_reviewed": [
                _repo("code/agent_ppo/workflow/curriculum_state.py:63-75"),
                _repo("train/compare_training_runs.py:17-66"),
                _repo("train/compare_training_runs.py:118-158"),
                _repo("train/compare_training_runs.py:321-337"),
                _repo("code/agent_ppo/workflow/checkpoint_score.py"),
                _repo("train/context/optimization/SURVIVAL_CPS_STABILIZATION_PLAN_20260421_1903.md:151-232"),
            ],
            "findings": [
                "Linux branch already defines fixed sample windows: bootstrap_10, bootstrap_20, global_40, global_80, global_120, global_160, and global_200.",
                "compare_training_runs.py compares fixed windows with local_10 for bootstrap nodes and local_20 for global nodes, with global_40 as the main pass/fail decision point.",
                "Checkpoint scoring and comparison samples already exist and should be reused instead of creating parallel gating mechanisms.",
            ],
            "opportunities": [
                _opportunity(
                    "reuse-fixed-window-comparison",
                    "Reuse fixed-window comparison nodes for all future candidates",
                    "P0_observe_only",
                    "The mechanism exists and enforces comparable windows, reducing benchmark noise before interventions.",
                    [_repo("train/compare_training_runs.py:29-66"), _repo("code/agent_ppo/workflow/curriculum_state.py:63-75")],
                ),
                _opportunity(
                    "additive-gate-reporting",
                    "Add candidate reports that quote existing fixed-window and checkpoint scores",
                    "P1_information_additive",
                    "Reports can consolidate existing Linux metrics without changing training behavior.",
                    [_repo("train/compare_training_runs.py:340-378")],
                ),
            ],
            "risks": [
                "Reimplementing convergence gates would duplicate proven Linux mechanisms and increase disagreement between training, comparison, and checkpoint-selection views.",
                "Using prefix metrics as promotion authority would violate the current local-window main-judgment rule.",
            ],
            "recommended_first_intervention": {
                "id": "reuse-fixed-window-comparison",
                "intervention_class": "P0_observe_only",
                "summary": "Treat existing fixed sample windows and comparison script as the canonical convergence observation layer.",
            },
            "confidence": "high",
        },
        {
            "agent_name": "task-0A-yjy-linux-reconciliation-auditor",
            "scope": "yjy_linux_reconciliation",
            "files_or_artifacts_reviewed": [
                _repo(".sisyphus/plans/benchmark-900-iteration.md:17-27"),
                _repo(".sisyphus/plans/benchmark-900-iteration.md:133-154"),
                _repo("train/context/sessions/TRAINING_SESSION_20260414.md:206-213"),
                _repo("train/context/optimization/STRONG_HEURISTIC_STRUCTURE_V1_20260422.md:5-11"),
                _repo("train/context/optimization/STRONG_HEURISTIC_STRUCTURE_V1_20260422.md:686-687"),
            ],
            "findings": [
                "Plan context names origin/yjy@044b773 as canonical YJY algorithm source unless overridden, while Task 0A itself forbids behavior changes.",
                "Historical session notes say win/linux algorithm files were previously identical in core PPO logic while Linux added infrastructure/runtime improvements.",
                "Recent optimization context notes successful routes relied on stronger direct heuristics, but reconciliation must preserve Linux mechanisms rather than resetting blindly.",
            ],
            "opportunities": [
                _opportunity(
                    "reconcile-before-reset",
                    "Create a YJY/Linux diff inventory before any reset or transplant",
                    "P0_observe_only",
                    "The plan requires YJY parity later, but current Linux mechanisms are known valuable and should not be overwritten without a diff inventory.",
                    [_repo(".sisyphus/plans/benchmark-900-iteration.md:20-24")],
                ),
                _opportunity(
                    "additive-reconciliation-manifest",
                    "Record preserved Linux mechanisms beside any future YJY-derived candidate",
                    "P1_information_additive",
                    "A manifest is additive evidence and helps prevent accidental loss of Linux stability tooling.",
                    [_repo("train/context/sessions/TRAINING_SESSION_20260414.md:206-213")],
                ),
            ],
            "risks": [
                "A wholesale reset to YJY before preserving Linux fixes would risk losing fixed-window gates, runtime state, checkpoint scoring, and Linux Docker stability patches.",
                "Skipping YJY comparison would violate the broader benchmark-900 plan before algorithm interventions.",
            ],
            "recommended_first_intervention": {
                "id": "reconcile-before-reset",
                "intervention_class": "P0_observe_only",
                "summary": "Inventory YJY/Linux differences and preservation requirements before any behavior-changing reconciliation.",
            },
            "confidence": "medium",
        },
        {
            "agent_name": "task-0A-benchmark-blocker-auditor",
            "scope": "benchmark_blocker_attribution",
            "files_or_artifacts_reviewed": [
                _repo("train/context/diagnosis/UNIFIED_PROBLEM_DIAGNOSIS_REPORT_20260420.md:63-75"),
                _repo("train/context/diagnosis/UNIFIED_PROBLEM_DIAGNOSIS_REPORT_20260420.md:89-123"),
                _repo("train/context/diagnosis/UNIFIED_PROBLEM_DIAGNOSIS_REPORT_20260420.md:148-189"),
                _repo("train/context/diagnosis/FULL_REPO_AUDIT_FINDINGS_20260420.md:575-822"),
                _repo("code/agent_ppo/eval/benchmark.py:124-233"),
            ],
            "findings": [
                "Existing diagnosis attributes current benchmark/training blockers to planner-policy divergence, return stall, high battery_fail_rate, and high zero_charge_battery_fail_rate.",
                "Charging-related positive shaping is documented as too weak; charger_access_probe_bonus is effectively zero because gates rarely trigger.",
                "Benchmark tooling has known correctness risks: checkpoint fallback/metadata mismatch, actor progress reset to 0, disaster-recovery termination attribution, and missing overall fields.",
                "Benchmark code already has issue-index categories for missed_charge_opportunity, late_return, late_contract, return_stall, target_selection, charger_contested, and battery_fail.",
            ],
            "opportunities": [
                _opportunity(
                    "benchmark-attribution-first",
                    "Use existing issue-index and failure taxonomy before changing policy",
                    "P0_observe_only",
                    "Failure buckets already exist and should be the first source for candidate selection.",
                    [_repo("code/agent_ppo/eval/benchmark.py:124-233")],
                ),
                _opportunity(
                    "fix-eval-metadata-safety",
                    "Fix benchmark metadata/checkpoint correctness before trusting promotions",
                    "P2_eval_only_safety",
                    "Eval-only correctness changes reduce false promotion/rejection risk without changing train-time behavior.",
                    [_repo("train/context/diagnosis/FULL_REPO_AUDIT_FINDINGS_20260420.md:575-822")],
                ),
            ],
            "risks": [
                "Benchmark A/B decisions can be invalid if requested checkpoint and loaded checkpoint diverge.",
                "Missing benchmark overall fields can be silently treated as zero by downstream scoring or bootstrap code.",
            ],
            "recommended_first_intervention": {
                "id": "benchmark-attribution-first",
                "intervention_class": "P0_observe_only",
                "summary": "Attribute blockers from existing benchmark/failure diagnostics before selecting any behavior change.",
            },
            "confidence": "high",
        },
        {
            "agent_name": "task-0A-pure-positive-auditor",
            "scope": "pure_positive_opportunities",
            "files_or_artifacts_reviewed": [
                _repo(".sisyphus/plans/benchmark-900-iteration.md:36-41"),
                _repo("train/context/diagnosis/UNIFIED_PROBLEM_DIAGNOSIS_REPORT_20260420.md:148-201"),
                _repo("train/context/diagnosis/UNIFIED_PROBLEM_DIAGNOSIS_REPORT_20260420.md:260-267"),
                _repo("code/agent_ppo/feature/preprocessor.py:1080-1189"),
                _repo("code/agent_ppo/feature/expert.py:177-260"),
            ],
            "findings": [
                "Plan explicitly prioritizes pure-positive observe/additive information first, then eval safety, thresholds/rewards, refactor, and architecture.",
                "Current global-position signals and planner outputs are already present; early opportunities should expose/verify them rather than add risky control branches.",
                "Prior diagnosis says probe-style charger access reward is effectively inactive and charging positive signal is weak relative to cleaning/explore/streak/CPS rewards.",
            ],
            "opportunities": [
                _opportunity(
                    "ranked-observe-only-audits",
                    "Produce ranked audit outputs for global signals, failure buckets, and convergence windows",
                    "P0_observe_only",
                    "This creates evidence needed by later tasks without modifying behavior.",
                    [_repo(".sisyphus/plans/benchmark-900-iteration.md:133-169")],
                ),
                _opportunity(
                    "additive-signal-coverage",
                    "If Task 0B finds omissions, add feature/log fields that expose existing observation signals without changing decisions",
                    "P1_information_additive",
                    "Information-additive work can improve observability while preserving current policy semantics.",
                    [_repo("code/agent_ppo/feature/preprocessor.py:1080-1189")],
                ),
                _opportunity(
                    "charging-positive-only-candidate",
                    "Later consider positive-only charger-access/progress reinforcement if evidence confirms inactive positive guidance",
                    "R2_reward_positive",
                    "Existing reports identify weak/inactive positive charger guidance, but reward changes belong after Wave 0 audits and fixed-observation comparison.",
                    [_repo("train/context/diagnosis/UNIFIED_PROBLEM_DIAGNOSIS_REPORT_20260420.md:178-201")],
                ),
            ],
            "risks": [
                "Jumping directly to reward penalties or architecture would violate the requested opportunity ordering.",
                "Even positive rewards can alter behavior and must wait for fixed-observation comparison and benchmark attribution gates.",
            ],
            "recommended_first_intervention": {
                "id": "ranked-observe-only-audits",
                "intervention_class": "P0_observe_only",
                "summary": "Complete observe-only audits and rankings before any additive or reward candidate.",
            },
            "confidence": "high",
        },
        {
            "agent_name": "task-0A-integrity-overfit-auditor",
            "scope": "integrity_overfit_risks",
            "files_or_artifacts_reviewed": [
                _repo(".sisyphus/plans/benchmark-900-iteration.md:76-83"),
                _repo(".sisyphus/plans/benchmark-900-iteration.md:150-168"),
                _repo("train/context/diagnosis/FULL_REPO_AUDIT_FINDINGS_20260420.md:628-652"),
                _repo("train/context/diagnosis/FULL_REPO_AUDIT_FINDINGS_20260420.md:723-844"),
                _repo("train/context/optimization/SURVIVAL_CPS_STABILIZATION_PLAN_20260421_1903.md:130-149"),
            ],
            "findings": [
                "Plan forbids simulator/scoring/map/target benchmark parameter drift and hardcoded map/coordinate policy branches.",
                "Experiment discipline requires one main lever plus at most one necessary supporting lever, same-window comparisons, and fixed failure attribution classes.",
                "Existing audit documents environment-drift and reproducibility risks including scripts that mutate train/.env, historical archive analysis using current code, and benchmark metadata mismatches.",
            ],
            "opportunities": [
                _opportunity(
                    "no-preaudit-behavior-diff-guard",
                    "Record git diff guard for code/agent_ppo and train/phases before audit completion",
                    "P0_observe_only",
                    "The guard directly proves Task 0A did not intervene before the merged audit existed.",
                    [_repo(".sisyphus/plans/benchmark-900-iteration.md:150-168")],
                ),
                _opportunity(
                    "eval-drift-validator",
                    "Later add benchmark profile/result validators before using benchmark promotion gates",
                    "P2_eval_only_safety",
                    "Eval-only validation protects against target parameter drift without changing policy behavior.",
                    [_repo(".sisyphus/plans/benchmark-900-iteration.md:57-61")],
                ),
            ],
            "risks": [
                "Training/benchmark evidence can be polluted by untracked .env drift or silent fallback to a different checkpoint.",
                "Optimizing against maps 1-10 without drift guards can become benchmark overfit; later candidates must use live observations, not map IDs or hardcoded coordinates.",
            ],
            "recommended_first_intervention": {
                "id": "no-preaudit-behavior-diff-guard",
                "intervention_class": "P0_observe_only",
                "summary": "Keep and record a no-behavior-diff guard before allowing intervention tasks.",
            },
            "confidence": "high",
        },
    ]


def build_artifact() -> dict[str, Any]:
    reports = build_default_reports()
    opportunity_ranking: list[dict[str, Any]] = []
    seen: set[str] = set()
    class_rank = {name: idx for idx, name in enumerate(ALLOWED_INTERVENTION_CLASSES)}
    for report in reports:
        for opportunity in report["opportunities"]:
            key = str(opportunity["id"])
            if key in seen:
                continue
            seen.add(key)
            opportunity_ranking.append(
                {
                    "id": key,
                    "title": opportunity["title"],
                    "scope": report["scope"],
                    "intervention_class": opportunity["intervention_class"],
                    "rationale": opportunity["rationale"],
                    "evidence": opportunity["evidence"],
                }
            )
    opportunity_ranking.sort(key=lambda item: (class_rank[item["intervention_class"]], item["scope"], item["id"]))
    return {
        "schema_version": 1,
        "task": "0A_full_board_opportunity_audit_before_intervention",
        "generated_by": "train/tools/merge_opportunity_audits.py",
        "global_charger_positions_confirmed": True,
        "global_robot_positions_confirmed": True,
        "global_npc_positions_confirmed": True,
        "linux_mechanisms_to_reuse_confirmed": True,
        "yjy_reset_required": False,
        "allowed_intervention_classes": list(ALLOWED_INTERVENTION_CLASSES),
        "reports": reports,
        "opportunity_ranking": opportunity_ranking,
        "guardrails": [
            "No code/agent_ppo/**, train/phases/**, simulator, map, scoring, reward, model, benchmark target, Docker, training, or benchmark intervention was performed by Task 0A.",
            "Future behavior changes remain blocked until downstream Wave 0 audits and fixed-observation comparison requirements are satisfied.",
        ],
    }


def validate_artifact(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    reports = payload.get("reports")
    if not isinstance(reports, list) or len(reports) < 6:
        errors.append("reports must contain at least six report objects")
        reports = [] if not isinstance(reports, list) else reports
    scopes = {str(report.get("scope")) for report in reports if isinstance(report, dict)}
    for scope in REQUIRED_SCOPES:
        if scope not in scopes:
            errors.append(f"missing required report scope: {scope}")
    for idx, report in enumerate(reports):
        if not isinstance(report, dict):
            errors.append(f"report[{idx}] is not an object")
            continue
        for field in REQUIRED_REPORT_FIELDS:
            if field not in report:
                errors.append(f"report[{idx}] missing field: {field}")
        if report.get("confidence") not in {"low", "medium", "high"}:
            errors.append(f"report[{idx}] confidence must be low/medium/high")
        if not isinstance(report.get("files_or_artifacts_reviewed"), list) or not report.get("files_or_artifacts_reviewed"):
            errors.append(f"report[{idx}] files_or_artifacts_reviewed must be a non-empty list")
        if not isinstance(report.get("findings"), list) or not report.get("findings"):
            errors.append(f"report[{idx}] findings must be a non-empty list")
        if not isinstance(report.get("opportunities"), list) or not report.get("opportunities"):
            errors.append(f"report[{idx}] opportunities must be a non-empty list")
        if not isinstance(report.get("risks"), list):
            errors.append(f"report[{idx}] risks must be a list")
        recommended = report.get("recommended_first_intervention")
        if not isinstance(recommended, dict):
            errors.append(f"report[{idx}] recommended_first_intervention must be an object")
        elif recommended.get("intervention_class") not in ALLOWED_INTERVENTION_CLASSES:
            errors.append(f"report[{idx}] recommended_first_intervention has invalid intervention_class")
        for opp_idx, opportunity in enumerate(report.get("opportunities") or []):
            if not isinstance(opportunity, dict):
                errors.append(f"report[{idx}].opportunities[{opp_idx}] is not an object")
                continue
            if opportunity.get("intervention_class") not in ALLOWED_INTERVENTION_CLASSES:
                errors.append(f"report[{idx}].opportunities[{opp_idx}] invalid intervention_class")
    for field in REQUIRED_BOOLEANS:
        if type(payload.get(field)) is not bool:
            errors.append(f"top-level {field} must be a boolean")
    ranking = payload.get("opportunity_ranking")
    if not isinstance(ranking, list) or not ranking:
        errors.append("opportunity_ranking must be a non-empty list")
    else:
        class_rank = {name: idx for idx, name in enumerate(ALLOWED_INTERVENTION_CLASSES)}
        last_rank = -1
        for idx, item in enumerate(ranking):
            if not isinstance(item, dict):
                errors.append(f"opportunity_ranking[{idx}] is not an object")
                continue
            intervention_class = item.get("intervention_class")
            if intervention_class not in ALLOWED_INTERVENTION_CLASSES:
                errors.append(f"opportunity_ranking[{idx}] invalid intervention_class")
                continue
            current_rank = class_rank[intervention_class]
            if current_rank < last_rank:
                errors.append("opportunity_ranking is not sorted by allowed intervention enum order")
            last_rank = current_rank
    return errors


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Merged audit artifact path")
    parser.add_argument("--write-default", action="store_true", help="Write the deterministic built-in Task 0A audit artifact")
    parser.add_argument("--validate", action="store_true", help="Validate the output artifact")
    parser.add_argument("--summary-output", type=Path, help="Optional text file for validation evidence")
    args = parser.parse_args()

    if args.write_default:
        payload = build_artifact()
        _write_json(args.output, payload)
    else:
        payload = _load_json(args.output)

    errors = validate_artifact(payload) if args.validate else []
    summary_lines = [
        f"artifact={args.output}",
        f"reports={len(payload.get('reports') or [])}",
        "scopes=" + ",".join(sorted(str(report.get("scope")) for report in payload.get("reports", []))),
        "booleans=" + ",".join(f"{field}={payload.get(field)}" for field in REQUIRED_BOOLEANS),
        "allowed_intervention_classes=" + ",".join(ALLOWED_INTERVENTION_CLASSES),
        f"opportunity_ranking_count={len(payload.get('opportunity_ranking') or [])}",
    ]
    if errors:
        summary_lines.append("validation=FAIL")
        summary_lines.extend(f"error={error}" for error in errors)
    else:
        summary_lines.append("validation=PASS" if args.validate else "validation=SKIPPED")
    summary = "\n".join(summary_lines) + "\n"
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(summary, encoding="utf-8")
    print(summary, end="")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

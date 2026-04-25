#!/usr/bin/env python3
"""Audit observation-provided global-position inputs and static overfit guards.

This tool is intentionally read-only for agent behavior.  It accepts an episode
artifact (or any JSON artifact containing observation-like payloads), counts the
available observation source classes, maps those sources to current code
consumers, and records a conservative static guard against policy/control logic
that branches on map identity or hardcoded coordinate lookup tables.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / ".sisyphus/evidence/benchmark-900/global-position-audit.json"
DEFAULT_CONTEXT = REPO_ROOT / ".sisyphus/evidence/benchmark-900/wave0/full-board-audit-merged.json"

SCAN_TARGETS = (
    REPO_ROOT / "code/agent_ppo/agent.py",
    REPO_ROOT / "code/agent_ppo/feature",
    REPO_ROOT / "code/agent_ppo/utils",
    REPO_ROOT / "code/agent_ppo/workflow",
    REPO_ROOT / "code/agent_ppo/model",
    REPO_ROOT / "code/agent_ppo/algorithm",
)

POLICY_BRANCH_RE = re.compile(r"\b(if|elif|while|match|case)\b.*\b(map_id|map_random)\b")
COORDINATE_TABLE_RE = re.compile(
    r"\b(map|coord|coordinate|position|pos).*?(table|lookup|by_map|per_map|coords|positions)\b.*[=:].*[\[{].*\(\s*\d+\s*,\s*\d+\s*\)",
    re.IGNORECASE,
)
MAP_KEY_COORD_RE = re.compile(r"\b\d+\s*:\s*[\[(]\s*\(?\s*\d+\s*,\s*\d+\s*\)?")


@dataclass(frozen=True)
class SourceSpec:
    key: str
    label: str
    expected_paths: tuple[str, ...]
    consumers: tuple[dict[str, Any], ...]


def _consumer(kind: str, path: str, evidence: str, note: str) -> dict[str, Any]:
    return {"kind": kind, "path": path, "evidence": evidence, "note": note}


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec(
        key="own_robot",
        label="own robot global position",
        expected_paths=("observation.frame_state.heroes.pos", "observation.frame_state.hero.pos"),
        consumers=(
            _consumer("feature", "code/agent_ppo/feature/preprocessor.py", "pb2struct cur_pos; scalar x/z; local/global channel anchoring", "source=observation live hero position"),
            _consumer("expert", "code/agent_ppo/feature/expert.py", "filter_actions/get_charger_signal/_weighted_astar_full use prep.cur_pos", "source=observation live robot position"),
            _consumer("reward", "code/agent_ppo/feature/preprocessor.py", "reward_process uses cur_visit_count/current cell/guidance derived from cur_pos", "source=observation live robot position"),
            _consumer("model", "code/agent_ppo/model/model.py", "_split_obs consumes scalar/local/global/entity tensors", "position-derived tensors reach model"),
            _consumer("benchmark", "code/agent_ppo/eval/benchmark.py", "episode/anomaly diagnostics consume position-derived metrics", "diagnostic consumer only"),
        ),
    ),
    SourceSpec(
        key="charger_organs",
        label="charger/organ global positions",
        expected_paths=("observation.frame_state.organs[*].pos", "observation.frame_state.organs[*].sub_type"),
        consumers=(
            _consumer("feature", "code/agent_ppo/feature/preprocessor.py", "_refresh_static_maps, _nearest_charger_metrics, _collect_charger_info, _build_entity_feature", "source=observation live organ coordinates"),
            _consumer("expert", "code/agent_ppo/feature/expert.py", "update_chargers, _plan_to_charger_cached, _evaluate_charger_candidates", "planner consumes observation charger centers"),
            _consumer("reward", "code/agent_ppo/feature/preprocessor.py", "charger_slack, route_anchor, recoverability and missed-charge reward context", "reward context derived from live charger positions"),
            _consumer("model", "code/agent_ppo/model/model.py", "global charger channel and charger entity tokens", "position-derived charger signals reach model"),
            _consumer("benchmark", "code/agent_ppo/eval/benchmark.py", "issue-index metrics: missed_charge_opportunity, target_selection, charger_contested", "benchmark attribution consumes derived charger metrics"),
        ),
    ),
    SourceSpec(
        key="npc_other_robot",
        label="NPC/other robot global positions",
        expected_paths=("observation.frame_state.npcs[*].pos",),
        consumers=(
            _consumer("feature", "code/agent_ppo/feature/preprocessor.py", "_refresh_static_maps, _nearest_npc_metrics, _collect_npc_info, entity features", "source=observation live npc coordinates"),
            _consumer("expert", "code/agent_ppo/feature/expert.py", "filter_actions and _build_cost_map NPC danger costs", "safety filter/planner consume live NPC positions"),
            _consumer("reward", "code/agent_ppo/feature/preprocessor.py", "npc proximity, evade mode, collision process cost", "reward context derived from live NPC positions"),
            _consumer("model", "code/agent_ppo/model/model.py", "NPC entity tokens and global npc_risk channel", "position-derived NPC signals reach model"),
            _consumer("benchmark", "code/agent_ppo/eval/benchmark.py", "collision/failure attribution metrics", "diagnostic consumer only"),
        ),
    ),
    SourceSpec(
        key="map_info",
        label="local map_info view",
        expected_paths=("observation.map_info",),
        consumers=(
            _consumer("feature", "code/agent_ppo/feature/preprocessor.py", "_update_memory, _compute_actual_legal_actions, _build_local_channels, _build_global_channels", "source=observation local map view"),
            _consumer("expert", "code/agent_ppo/feature/expert.py", "_build_cost_map consumes explored/passable maps populated from map_info", "reachability uses observation-updated maps"),
            _consumer("reward", "code/agent_ppo/feature/preprocessor.py", "frontier/dirt/wall/stale-boundary reward context", "reward context derived from map_info"),
            _consumer("model", "code/agent_ppo/model/model.py", "local and global-memory branches", "map tensors reach model"),
        ),
    ),
    SourceSpec(
        key="legal_action",
        label="legal action mask",
        expected_paths=("observation.legal_action", "observation.legal_act"),
        consumers=(
            _consumer("feature", "code/agent_ppo/feature/preprocessor.py", "pb2struct _legal_act; get_legal_action merges observation mask with local passability", "source=observation action mask"),
            _consumer("expert", "code/agent_ppo/feature/expert.py", "filter_actions, get_charger_signal, get_logit_bias", "safety/planning respects legal mask"),
            _consumer("model", "code/agent_ppo/agent.py", "_legal_soft_max and safe_sample_action mask model logits", "policy output is masked by legal actions"),
        ),
    ),
    SourceSpec(
        key="battery",
        label="battery and battery_max",
        expected_paths=("observation.frame_state.heroes.battery", "observation.frame_state.heroes.battery_max", "observation.env_info.remaining_charge", "observation.env_info.battery_max"),
        consumers=(
            _consumer("feature", "code/agent_ppo/feature/preprocessor.py", "pb2struct battery/battery_max; scalar battery_ratio", "source=observation/env_info battery values"),
            _consumer("expert", "code/agent_ppo/feature/expert.py", "get_charger_signal, get_teacher_guidance, get_emergency_fallback", "battery gates planner/teacher/fallback signals"),
            _consumer("reward", "code/agent_ppo/feature/preprocessor.py", "reward_process battery risk/slack/charge context", "reward context derived from live battery"),
            _consumer("checkpoint/curriculum gate", "code/agent_ppo/workflow/checkpoint_score.py", "battery_fail_rate and zero_charge gates", "aggregate gate consumer"),
            _consumer("checkpoint/curriculum gate", "code/agent_ppo/workflow/curriculum_state.py", "battery_fail_rate, zero_charge_battery_fail_rate, battery_positive_reward_rate", "window aggregate consumer"),
        ),
    ),
    SourceSpec(
        key="path_reachability",
        label="path/reachability data derived from live observations",
        expected_paths=("derived.charger_path", "derived.reachable", "derived.unknown_path_ratio", "derived.planner_topk_reachable_count"),
        consumers=(
            _consumer("feature", "code/agent_ppo/feature/preprocessor.py", "_sort_charger_candidates, _build_local_channels return_guidance, scalar reachable/astar/unknown fields", "derived from observation coordinates and map memory"),
            _consumer("expert", "code/agent_ppo/feature/expert.py", "get_charger_signal returns charger_path/reachable/unknown_path_ratio/planner_topk_reachable_count", "planner-derived reachability"),
            _consumer("reward", "code/agent_ppo/feature/preprocessor.py", "reward_process known_route/slack_confidence/narrow_unknown_commit/suboptimal_target_hold", "reward consumes reachability signals"),
            _consumer("model", "code/agent_ppo/model/model.py", "scalar/entity/local/global tensors include route guidance and reachability", "derived path signals reach model"),
            _consumer("benchmark", "code/agent_ppo/eval/benchmark.py", "issue-index metrics include narrow_unknown_commit and planner_policy_divergence", "benchmark attribution consumes reachability metrics"),
            _consumer("checkpoint/curriculum gate", "code/agent_ppo/workflow/checkpoint_score.py", "reliable_planner_divergence_rate scoring/gates", "aggregate reachability/planner gate"),
        ),
    ),
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{id(payload)}.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def iter_nodes(payload: Any) -> Iterable[Any]:
    yield payload
    if isinstance(payload, dict):
        for value in payload.values():
            yield from iter_nodes(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from iter_nodes(value)


def _dict_get(obj: Any, key: str) -> Any:
    return obj.get(key) if isinstance(obj, dict) else None


def _has_pos(obj: Any) -> bool:
    pos = _dict_get(obj, "pos")
    return isinstance(pos, dict) and ("x" in pos or "z" in pos)


def _count_own_robot(payload: Any) -> int:
    count = 0
    for node in iter_nodes(payload):
        if not isinstance(node, dict):
            continue
        heroes = node.get("heroes") if isinstance(node.get("heroes"), dict) else None
        hero = node.get("hero") if isinstance(node.get("hero"), dict) else None
        for candidate in (heroes, hero):
            if _has_pos(candidate):
                count += 1
    return count


def _count_organs(payload: Any) -> int:
    count = 0
    for node in iter_nodes(payload):
        organs = node.get("organs") if isinstance(node, dict) else None
        if isinstance(organs, list):
            count += sum(1 for organ in organs if _has_pos(organ) and int(_dict_get(organ, "sub_type") or 1) == 1)
    return count


def _count_npcs(payload: Any) -> int:
    count = 0
    for node in iter_nodes(payload):
        npcs = node.get("npcs") if isinstance(node, dict) else None
        if isinstance(npcs, list):
            count += sum(1 for npc in npcs if _has_pos(npc))
    return count


def _count_key(payload: Any, *keys: str) -> int:
    count = 0
    for node in iter_nodes(payload):
        if isinstance(node, dict) and any(key in node and node.get(key) is not None for key in keys):
            count += 1
    return count


def count_source(spec_key: str, payload: Any) -> int:
    if spec_key == "own_robot":
        return _count_own_robot(payload)
    if spec_key == "charger_organs":
        return _count_organs(payload)
    if spec_key == "npc_other_robot":
        return _count_npcs(payload)
    if spec_key == "map_info":
        return _count_key(payload, "map_info")
    if spec_key == "legal_action":
        return _count_key(payload, "legal_action", "legal_act")
    if spec_key == "battery":
        return _count_key(payload, "battery", "battery_max", "remaining_charge")
    if spec_key == "path_reachability":
        return _count_key(payload, "charger_path", "reachable", "unknown_path_ratio", "planner_topk_reachable_count")
    raise KeyError(spec_key)


def build_source_audit(payload: Any) -> list[dict[str, Any]]:
    sources = []
    for spec in SOURCE_SPECS:
        count = count_source(spec.key, payload)
        entry: dict[str, Any] = {
            "key": spec.key,
            "label": spec.label,
            "source": "observation" if spec.key != "path_reachability" else "derived_from_observation",
            "expected_paths": list(spec.expected_paths),
            "artifact_count": count,
            "consumers": list(spec.consumers),
        }
        if count <= 0:
            entry["unavailable_reason"] = "source class not present in supplied artifact; code consumers are still mapped from repository evidence"
        sources.append(entry)
    return sources


def _iter_python_files() -> Iterable[Path]:
    for target in SCAN_TARGETS:
        if target.is_file():
            yield target
        elif target.is_dir():
            yield from sorted(target.rglob("*.py"))


def _is_allowed_map_line(path: Path, line: str) -> bool:
    rel = path.relative_to(REPO_ROOT).as_posix()
    stripped = line.strip()
    if rel.endswith("archive_analysis.py") and "map_id" in stripped:
        return True
    if "per_map_scores" in stripped:
        return True
    if stripped == 'if map_id != "?":':
        return True
    if "map_random" in stripped and ("env_info.get" in stripped or "map_random" in stripped and "self.map_random" in stripped and "=" in stripped):
        return True
    return False


def _is_allowed_coordinate_line(line: str) -> bool:
    allowed_tokens = ("ACTION_DELTAS", "DELTAS", "SAMPLE_WINDOW_EPISODES", "RETURN_WINDOW_ALIAS_KEYS")
    stripped = line.strip()
    return any(token in stripped for token in allowed_tokens)


def run_static_guard() -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    files_scanned = 0
    for path in _iter_python_files():
        files_scanned += 1
        rel = path.relative_to(REPO_ROOT).as_posix()
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            lines = path.read_text(errors="replace").splitlines()
        for lineno, line in enumerate(lines, start=1):
            if POLICY_BRANCH_RE.search(line) and not _is_allowed_map_line(path, line):
                findings.append(
                    {
                        "type": "forbidden_map_policy_branch",
                        "path": rel,
                        "line": lineno,
                        "text": line.strip(),
                    }
                )
            if (_is_allowed_coordinate_line(line)):
                continue
            if COORDINATE_TABLE_RE.search(line) or MAP_KEY_COORD_RE.search(line):
                findings.append(
                    {
                        "type": "possible_coordinate_lookup_table",
                        "path": rel,
                        "line": lineno,
                        "text": line.strip(),
                    }
                )
    return {
        "status": "pass" if not findings else "fail",
        "files_scanned": files_scanned,
        "scan_roots": [target.relative_to(REPO_ROOT).as_posix() for target in SCAN_TARGETS],
        "allowed": [
            "source=observation live coordinates parsed from frame_state/map_info/legal_action/battery",
            "normal env/map parsing and offline archive grouping that does not control policy decisions",
        ],
        "forbidden": [
            "policy/control branching on map_id or map_random",
            "hardcoded per-map coordinate lookup tables used for decisions",
        ],
        "findings": findings,
    }


def format_static_guard_text(guard: dict[str, Any]) -> str:
    lines = [
        "Task 0B static guard: global-position overfit policy/control scan",
        f"status={guard['status']}",
        f"files_scanned={guard['files_scanned']}",
        "scan_roots=" + ",".join(guard["scan_roots"]),
        "allowed=" + "; ".join(guard["allowed"]),
        "forbidden=" + "; ".join(guard["forbidden"]),
    ]
    if guard["findings"]:
        lines.append("findings:")
        for finding in guard["findings"]:
            lines.append(f"- {finding['type']} {finding['path']}:{finding['line']} {finding['text']}")
    else:
        lines.append("findings: none")
    return "\n".join(lines) + "\n"


def build_audit(artifact_path: Path, payload: Any, context_path: Path | None) -> dict[str, Any]:
    context_payload = None
    if context_path and context_path.exists():
        context_payload = load_json(context_path)
    static_guard = run_static_guard()
    return {
        "schema_version": 1,
        "task": "0B_audit_and_exploit_observation_provided_global_positions",
        "generated_by": "train/tools/audit_global_position_inputs.py",
        "baseline_episode_artifact": artifact_path.relative_to(REPO_ROOT).as_posix() if artifact_path.is_relative_to(REPO_ROOT) else str(artifact_path),
        "context_artifact": context_path.relative_to(REPO_ROOT).as_posix() if context_path and context_path.exists() and context_path.is_relative_to(REPO_ROOT) else (str(context_path) if context_path else None),
        "context_summary": {
            "global_robot_positions_confirmed_by_0A": bool((context_payload or {}).get("global_robot_positions_confirmed")),
            "global_charger_positions_confirmed_by_0A": bool((context_payload or {}).get("global_charger_positions_confirmed")),
            "global_npc_positions_confirmed_by_0A": bool((context_payload or {}).get("global_npc_positions_confirmed")),
        },
        "source_fields": build_source_audit(payload),
        "static_guard": static_guard,
        "verdict": "pass" if static_guard["status"] == "pass" else "fail",
        "notes": [
            "This audit is observe-only and does not modify code/agent_ppo behavior.",
            "Live coordinates are acceptable only with explicit source=observation provenance; hardcoded benchmark/map coordinates remain forbidden.",
        ],
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit observation global-position source fields, consumers, and static overfit guards.")
    parser.add_argument("baseline_episode_artifact", type=Path, help="JSON episode/artifact path to audit")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Audit JSON output path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT, help=f"0A context artifact path (default: {DEFAULT_CONTEXT})")
    parser.add_argument("--static-guard-output", type=Path, default=None, help="Optional text evidence path for the static guard")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    artifact_path = args.baseline_episode_artifact.resolve()
    if not artifact_path.exists():
        raise FileNotFoundError(f"baseline episode artifact not found: {artifact_path}")
    payload = load_json(artifact_path)
    audit = build_audit(artifact_path, payload, args.context.resolve() if args.context else None)
    write_json(args.output.resolve(), audit)
    if args.static_guard_output:
        guard_path = args.static_guard_output.resolve()
        guard_path.parent.mkdir(parents=True, exist_ok=True)
        guard_path.write_text(format_static_guard_text(audit["static_guard"]), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "verdict": audit["verdict"], "static_guard": audit["static_guard"]["status"]}, sort_keys=True))
    return 0 if audit["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

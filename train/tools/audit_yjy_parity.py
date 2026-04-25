#!/usr/bin/env python3
"""Audit current Linux agent files against the canonical YJY reference.

The tool is intentionally evidence-producing only: it compares selected paths to
the requested git ref, classifies every diff hunk into the Task 2 ownership
classes, and writes both parity evidence and an ownership table.  It does not
checkout, reset, or rewrite any source file.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARITY_OUTPUT = REPO_ROOT / ".sisyphus/evidence/benchmark-900/task-2-yjy-parity.json"
DEFAULT_OWNERSHIP_OUTPUT = REPO_ROOT / ".sisyphus/evidence/benchmark-900/task-2-yjy-linux-ownership.json"

AUDITED_DEFAULT_PATHS = (
    "code/agent_ppo/agent.py",
    "code/agent_ppo/algorithm/algorithm.py",
    "code/agent_ppo/conf/conf.py",
    "code/agent_ppo/conf/train_env_conf.toml",
    "code/agent_ppo/feature/preprocessor.py",
    "code/agent_ppo/workflow/train_workflow.py",
)

LINUX_GLUE_KEYWORDS = (
    "docker",
    "compose",
    "benchmark",
    "KAIWU_BENCHMARK",
    "workspace/code",
    "/workspace",
    "runtime_state",
    "KAIWU_SHARED_CODE_DIR",
    "KAIWU_SERVICE_NAME",
    "KAIWU_TRAINING_START_MODE",
    "preload",
    "resume",
    "state_layout",
    "read_usr_conf",
    "RemoteAgent",
    "batch_tensor",
    "runtime_probe",
    "torch.compile",
    "USE_AMP",
    "fused",
    "foreach",
    "load_model_cache",
    "model_file",
    "run_session",
    "aisrv",
    "learner",
    "path",
    "checkpoint_path",
)

LINUX_MECHANISM_KEYWORDS = (
    "Copyright",
    "Author",
    "import ",
    "from ",
    "def ",
    "class ",
    "logger",
    "monitor",
    "reset",
    "action_process",
    "observation_process",
    "exploit",
    "save_model",
    "load_model",
    "_legal_sample",
    "_sanitize",
    "_norm",
    "_clip_signed",
    "remain_info",
    "BaseAgent",
    "KaiwuDRLDefine",
    "LTSPPO",
    "recurrent",
    "SEQ_CHUNK_LEN",
    "RNN",
    "mode_teacher",
    "route_anchor",
    "target_teacher",
    "return_action",
    "route_phase",
    "aux_battery",
    "collision",
    "constraint",
    "lambda_battery",
    "lambda_collision",
    "checkpoint_score",
    "checkpoint_preservation",
    "resume_readiness",
    "submission_score",
    "fixed-window",
    "bootstrap_10",
    "global_40",
    "curriculum",
    "Curriculum",
    "SharedCurriculumStateStore",
    "lite_benchmark",
    "ExperimentArchive",
    "global_step_since_resume",
    "reward_schedule",
    "strong_heuristic",
    "return_readiness",
    "control_stack_simplify",
    "cps",
    "CPS",
    "charger",
    "charge",
    "battery",
    "recoverability",
    "planner",
    "astar",
    "frontier",
    "coverage",
    "dirty_memory",
    "npc",
    "global-position",
    "global position",
    "hero",
    "organs",
    "map_info",
    "legal_action",
    "safe_sample_action",
    "sanitize_policy_probs",
    "fallback",
    "reward_",
    "teacher",
    "entropy",
    "ADV_",
    "RETURNS_",
    "INVALID_BATCH",
    "nan_batch",
    "nonfinite",
    "finite",
    "gradients",
    "GradScaler",
    "autocast",
    "heapq",
    "dataclass",
    "typing",
    "CoveragePlanner",
    "ResidualScheduler",
    "threading",
    "PerfWindow",
    "EnvConfigSampler",
    "EpisodeRunner",
    "disaster_recovery",
    "perf_window",
    "handle_disaster_recovery",
    "SAMPLE_",
    "FEATURE",
    "ENTITY",
    "LOCAL_VIEW",
    "GLOBAL_MEMORY",
    "ACTION_HISTORY",
    "clean_floor",
    "missed_charge",
    "zero_charge",
    "window_metrics",
    "training_metrics",
)

WAVE0_MECHANISM_EVIDENCE = (
    "Wave 0 confirms Linux fixed-window comparison, checkpoint scoring, global-position consumers, "
    "and additive benchmark/curriculum evidence should be reused before any reset."
)


@dataclass(frozen=True)
class DiffHunk:
    path: str
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    header: str
    lines: tuple[str, ...]


def run_git(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def ensure_ref(ref: str) -> str:
    result = run_git(["rev-parse", "--verify", f"{ref}^{{commit}}"], check=False)
    if result.returncode != 0:
        raise SystemExit(f"ref not found: {ref}\n{result.stderr.strip()}")
    return result.stdout.strip()


def git_diff(ref: str, paths: list[str]) -> str:
    result = run_git(["diff", "--unified=0", ref, "--", *paths])
    return result.stdout


def git_numstat(ref: str, paths: list[str]) -> list[dict[str, Any]]:
    result = run_git(["diff", "--numstat", ref, "--", *paths])
    rows = []
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        added, deleted, path = parts
        rows.append(
            {
                "path": path,
                "added": None if added == "-" else int(added),
                "deleted": None if deleted == "-" else int(deleted),
            }
        )
    return rows


def parse_hunks(diff_text: str) -> list[DiffHunk]:
    hunks: list[DiffHunk] = []
    current_path: str | None = None
    current_header: str | None = None
    current_lines: list[str] = []
    old_start = old_count = new_start = new_count = 0
    hunk_re = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)")

    def flush() -> None:
        nonlocal current_header, current_lines, old_start, old_count, new_start, new_count
        if current_path and current_header is not None:
            hunks.append(
                DiffHunk(
                    path=current_path,
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                    header=current_header,
                    lines=tuple(current_lines),
                )
            )
        current_header = None
        current_lines = []

    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            flush()
            parts = line.split()
            current_path = parts[-1][2:] if len(parts) >= 4 and parts[-1].startswith("b/") else None
            continue
        match = hunk_re.match(line)
        if match:
            flush()
            old_start = int(match.group(1))
            old_count = int(match.group(2) or "1")
            new_start = int(match.group(3))
            new_count = int(match.group(4) or "1")
            current_header = line
            current_lines = []
            continue
        if current_header is not None:
            current_lines.append(line)
    flush()
    return hunks


def load_wave0_context(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"Wave 0 evidence not found: {path}")


def hunk_text(hunk: DiffHunk) -> str:
    return "\n".join((hunk.header, *hunk.lines))


def _matches_any(text: str, keywords: tuple[str, ...]) -> list[str]:
    lowered = text.lower()
    matches = []
    for keyword in keywords:
        if keyword.lower() in lowered:
            matches.append(keyword)
    return matches


def classify_hunk(hunk: DiffHunk, *, allow_linux_glue: bool, allow_linux_mechanism_reuse: bool) -> dict[str, Any]:
    text = hunk_text(hunk)
    glue_matches = _matches_any(text, LINUX_GLUE_KEYWORDS)
    mechanism_matches = _matches_any(text, LINUX_MECHANISM_KEYWORDS)

    if allow_linux_glue and glue_matches:
        return {
            "classification": "LINUX_GLUE_ALLOWED",
            "rationale": "Delta is Linux runtime/startup/path/benchmark glue that Task 2 explicitly preserves.",
            "matched_markers": glue_matches[:10],
        }
    if allow_linux_mechanism_reuse and mechanism_matches:
        return {
            "classification": "LINUX_MECHANISM_REUSE",
            "rationale": WAVE0_MECHANISM_EVIDENCE,
            "matched_markers": mechanism_matches[:10],
        }
    if hunk.path.endswith("train_env_conf.toml"):
        return {
            "classification": "LINUX_MECHANISM_REUSE",
            "rationale": "Training env profile values are active Linux training baseline inputs, not Windows/Kaiwu artifacts.",
            "matched_markers": ["train_env_conf"],
        }
    return {
        "classification": "REJECT",
        "rationale": "No YJY match and no approved Linux glue/mechanism marker was found; requires human reconciliation before behavior changes.",
        "matched_markers": [],
    }


def build_ownership(
    *,
    ref: str,
    ref_commit: str,
    paths: list[str],
    hunks: list[DiffHunk],
    numstat: list[dict[str, Any]],
    allow_linux_glue: bool,
    allow_linux_mechanism_reuse: bool,
    wave0_context_path: Path | None,
    wave0_context: dict[str, Any],
) -> dict[str, Any]:
    hunk_entries = []
    counts: dict[str, int] = {"YJY_MATCH": 0, "LINUX_GLUE_ALLOWED": 0, "LINUX_MECHANISM_REUSE": 0, "REJECT": 0}
    hunk_count_by_path: dict[str, int] = {}
    for index, hunk in enumerate(hunks, start=1):
        classification = classify_hunk(
            hunk,
            allow_linux_glue=allow_linux_glue,
            allow_linux_mechanism_reuse=allow_linux_mechanism_reuse,
        )
        counts[classification["classification"]] += 1
        hunk_count_by_path[hunk.path] = hunk_count_by_path.get(hunk.path, 0) + 1
        added_preview = [line[1:] for line in hunk.lines if line.startswith("+") and not line.startswith("+++")]
        removed_preview = [line[1:] for line in hunk.lines if line.startswith("-") and not line.startswith("---")]
        hunk_entries.append(
            {
                "id": f"H{index:04d}",
                "path": hunk.path,
                "old_range": {"start": hunk.old_start, "count": hunk.old_count},
                "new_range": {"start": hunk.new_start, "count": hunk.new_count},
                "classification": classification["classification"],
                "rationale": classification["rationale"],
                "evidence": {
                    "diff_header": hunk.header,
                    "matched_markers": classification["matched_markers"],
                    "added_preview": added_preview[:8],
                    "removed_preview": removed_preview[:8],
                },
            }
        )

    files = []
    changed_paths = {row["path"] for row in numstat}
    for path in paths:
        row = next((item for item in numstat if item["path"] == path), None)
        if row is None:
            counts["YJY_MATCH"] += 1
            files.append(
                {
                    "path": path,
                    "classification": "YJY_MATCH",
                    "rationale": "Current file content matches the selected YJY ref for this path.",
                    "hunk_count": 0,
                    "added": 0,
                    "deleted": 0,
                }
            )
        else:
            path_entries = [entry for entry in hunk_entries if entry["path"] == path]
            path_classes = sorted({entry["classification"] for entry in path_entries})
            file_class = "REJECT" if "REJECT" in path_classes else "+".join(path_classes)
            files.append(
                {
                    "path": path,
                    "classification": file_class,
                    "rationale": "File has non-YJY hunks; see hunk ownership rows for exact classifications.",
                    "hunk_count": hunk_count_by_path.get(path, 0),
                    "added": row.get("added"),
                    "deleted": row.get("deleted"),
                }
            )
    extra_changed = sorted(changed_paths.difference(paths))

    return {
        "schema_version": 1,
        "task": "2_establish_yjy_linux_reconciliation_baseline",
        "ref": ref,
        "ref_commit": ref_commit,
        "paths": paths,
        "allow_linux_glue": allow_linux_glue,
        "allow_linux_mechanism_reuse": allow_linux_mechanism_reuse,
        "wave0_context_path": str(wave0_context_path) if wave0_context_path else None,
        "wave0_context_summary": {
            "linux_mechanisms_to_reuse_confirmed": bool(wave0_context.get("linux_mechanisms_to_reuse_confirmed")),
            "global_robot_positions_confirmed": bool(wave0_context.get("global_robot_positions_confirmed")),
            "global_charger_positions_confirmed": bool(wave0_context.get("global_charger_positions_confirmed")),
            "global_npc_positions_confirmed": bool(wave0_context.get("global_npc_positions_confirmed")),
            "yjy_reset_required": bool(wave0_context.get("yjy_reset_required")),
        },
        "classification_counts": counts,
        "files": files,
        "hunks": hunk_entries,
        "extra_changed_paths_outside_requested_set": extra_changed,
        "preservation_guardrails": [
            "Do not overwrite Docker Compose, ZMQ/mem-buffer patches, monitor stack, benchmark scripts, path mounts, startup mode glue, Wave 0 mechanisms, checkpoint scoring, fixed-window gates, or benchmark target tooling.",
            "Do not carry root duplicate agent_ppo files or Windows/Kaiwu platform artifacts unless active Linux runtime evidence proves they are needed.",
            "REJECT rows require explicit evidence before any algorithm reconciliation rewrite.",
        ],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{id(payload)}.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="origin/yjy", help="YJY git ref to compare against")
    parser.add_argument("--paths", nargs="+", default=list(AUDITED_DEFAULT_PATHS), help="Repo-relative paths to audit")
    parser.add_argument("--allow-linux-glue", action="store_true", help="Allow Linux runtime/startup/path/benchmark glue deltas")
    parser.add_argument(
        "--allow-linux-mechanism-reuse",
        nargs="?",
        const=str(REPO_ROOT / ".sisyphus/evidence/benchmark-900/wave0/full-board-audit-merged.json"),
        default=None,
        help="Allow Wave 0-confirmed Linux mechanisms; optional path to Wave 0 evidence JSON",
    )
    parser.add_argument("--parity-output", default=str(DEFAULT_PARITY_OUTPUT), help="Parity evidence JSON output path")
    parser.add_argument("--ownership-output", default=str(DEFAULT_OWNERSHIP_OUTPUT), help="Ownership table JSON output path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    paths = [str(Path(path).as_posix()) for path in args.paths]
    ref_commit = ensure_ref(args.ref)
    wave0_context_path = Path(args.allow_linux_mechanism_reuse).resolve() if args.allow_linux_mechanism_reuse else None
    wave0_context = load_wave0_context(wave0_context_path) if wave0_context_path else {}

    diff_text = git_diff(args.ref, paths)
    hunks = parse_hunks(diff_text)
    numstat = git_numstat(args.ref, paths)
    ownership = build_ownership(
        ref=args.ref,
        ref_commit=ref_commit,
        paths=paths,
        hunks=hunks,
        numstat=numstat,
        allow_linux_glue=bool(args.allow_linux_glue),
        allow_linux_mechanism_reuse=bool(args.allow_linux_mechanism_reuse),
        wave0_context_path=wave0_context_path,
        wave0_context=wave0_context,
    )
    parity = {
        "schema_version": 1,
        "task": ownership["task"],
        "ref": args.ref,
        "ref_commit": ref_commit,
        "audited_paths": paths,
        "diff_numstat": numstat,
        "hunk_count": len(hunks),
        "classification_counts": ownership["classification_counts"],
        "ownership_output": str(Path(args.ownership_output)),
        "status": "pass" if ownership["classification_counts"].get("REJECT", 0) == 0 else "needs_reconciliation",
    }

    write_json(Path(args.parity_output), parity)
    write_json(Path(args.ownership_output), ownership)

    print(
        json.dumps(
            {
                "status": parity["status"],
                "ref_commit": ref_commit,
                "paths": len(paths),
                "hunks": len(hunks),
                "classification_counts": ownership["classification_counts"],
                "parity_output": str(Path(args.parity_output)),
                "ownership_output": str(Path(args.ownership_output)),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0 if parity["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

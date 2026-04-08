#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Background archive agent for log and checkpoint persistence.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from agent_ppo.utils.archive_analysis import build_checkpoint_analysis, write_checkpoint_reports
from agent_ppo.utils.experiment_archive import ExperimentArchive, utc_now_iso


TRAIN_PERF_PATTERN = re.compile(
    r"global step is (?P<global_step>\d+), "
    r"train once cost time is (?P<train_once_ms>[\d.]+) ms "
    r"\(data_fetch: (?P<data_fetch_ms>[\d.]+) ms, real_train: (?P<real_train_ms>[\d.]+) ms\).*"
    r"sample_production_and_consumption_ratio is (?P<ratio>[\d.]+), "
    r"replay buffer monitor is \{'buffer_utilization': '(?P<buffer>[^']+)'\}"
)


class ArchiveAgent:
    def __init__(self):
        self.archive = ExperimentArchive(service_name=os.getenv("KAIWU_SERVICE_NAME") or "archive-agent")
        self.host_train_root = Path(os.getenv("KAIWU_HOST_TRAIN_DIR") or "/workspace/host_train").resolve()
        self.code_root = Path(os.getenv("KAIWU_CODE_ROOT") or "/workspace/code").resolve()
        self.sync_seconds = int(os.getenv("KAIWU_ARCHIVE_SYNC_SECONDS") or "30")
        self.idle_seconds = int(os.getenv("KAIWU_ARCHIVE_IDLE_SECONDS") or "300")
        self._last_finalized_run_id: str | None = None

    def run_forever(self) -> None:
        while True:
            run_state = self.archive.get_run_state(create_if_missing=False)
            if run_state is None:
                time.sleep(self.sync_seconds)
                continue

            run_dir = self.archive.get_run_dir(create_if_missing=False)
            if run_dir is None:
                time.sleep(self.sync_seconds)
                continue

            self._sync_host_artifacts(run_dir)
            self.archive.write_manifest(
                {
                    "archive_agent_seen_at": utc_now_iso(),
                    "host_train_root": str(self.host_train_root),
                }
            )

            if self._should_finalize(run_state, run_dir):
                self._finalize_run(run_state, run_dir)
            time.sleep(self.sync_seconds)

    def _sync_host_artifacts(self, run_dir: Path) -> None:
        self._copy_tree(self.host_train_root / "log", run_dir / "human" / "raw_logs")
        self._copy_tree(self.host_train_root / "backup_model", run_dir / "human" / "checkpoints" / "backup_model")
        self._copy_file(self.host_train_root / ".env", run_dir / "human" / "config" / "train" / ".env")
        self._copy_file(
            self.host_train_root / ".docker-compose.yaml",
            run_dir / "human" / "config" / "train" / "docker-compose.yaml",
        )
        self._copy_file(self.code_root / "kaiwu.json", run_dir / "human" / "config" / "code" / "kaiwu.json")

    def _should_finalize(self, run_state: dict[str, Any], run_dir: Path) -> bool:
        if run_state.get("status") == "finalized":
            self._last_finalized_run_id = run_state.get("run_id")
            return False
        if self._last_finalized_run_id == run_state.get("run_id"):
            return False

        latest_mtime = self._latest_mtime(self.host_train_root / "log", run_dir / "ai" / "streams")
        if latest_mtime is None:
            return False
        idle_for = time.time() - latest_mtime
        return idle_for >= self.idle_seconds

    def _finalize_run(self, run_state: dict[str, Any], run_dir: Path) -> None:
        merged_streams = self._merge_streams(run_dir)
        parsed_learner_perf = self._parse_learner_perf(run_dir)

        train_window_records = merged_streams.get("train_window", [])
        train_window_records.extend(parsed_learner_perf)
        if train_window_records:
            train_window_records.sort(key=lambda item: item.get("ts", ""))
            self._write_jsonl(run_dir / "ai" / "train_window.jsonl", train_window_records)

        checkpoint_analysis = build_checkpoint_analysis(merged_streams.get("episode_summary", []))
        write_checkpoint_reports(run_dir, checkpoint_analysis)
        summary = self._build_summary(run_state, merged_streams, parsed_learner_perf, checkpoint_analysis)
        summary_path = run_dir / "ai" / "summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._write_report(run_dir, summary)
        self._compress_human_logs(run_dir)
        self.archive.mark_run_finalized({"summary_path": str(summary_path)})
        self._last_finalized_run_id = run_state.get("run_id")

    def _merge_streams(self, run_dir: Path) -> dict[str, list[dict[str, Any]]]:
        streams_dir = run_dir / "ai" / "streams"
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        if not streams_dir.exists():
            return grouped

        for path in sorted(streams_dir.glob("*.jsonl")):
            stream_name = path.name.split(".")[0]
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                grouped[stream_name].append(payload)

        for stream_name, records in grouped.items():
            records.sort(key=lambda item: item.get("ts", ""))
            self._write_jsonl(run_dir / "ai" / f"{stream_name}.jsonl", records)

        return grouped

    def _parse_learner_perf(self, run_dir: Path) -> list[dict[str, Any]]:
        parsed_records: list[dict[str, Any]] = []
        learner_dir = run_dir / "human" / "raw_logs" / "learner"
        if not learner_dir.exists():
            return parsed_records

        for path in sorted(learner_dir.glob("*.log")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                message = str(payload.get("message") or "")
                match = TRAIN_PERF_PATTERN.search(message)
                if not match:
                    continue
                parsed_records.append(
                    {
                        "ts": str(payload.get("time") or utc_now_iso()),
                        "service": "learner",
                        "pid": payload.get("pid"),
                        "record_type": "learner_perf",
                        "global_step": int(match.group("global_step")),
                        "train_once_ms": float(match.group("train_once_ms")),
                        "data_fetch_ms": float(match.group("data_fetch_ms")),
                        "real_train_ms": float(match.group("real_train_ms")),
                        "sample_production_and_consumption_ratio": float(match.group("ratio")),
                        "buffer_utilization": match.group("buffer"),
                    }
                )
        return parsed_records

    def _build_summary(
        self,
        run_state: dict[str, Any],
        merged_streams: dict[str, list[dict[str, Any]]],
        parsed_learner_perf: list[dict[str, Any]],
        checkpoint_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        episodes = merged_streams.get("episode_summary", [])
        checkpoints = merged_streams.get("checkpoint_index", [])
        events = merged_streams.get("key_events", [])

        fail_reason_counter = Counter(item.get("fail_reason", "unknown") for item in episodes)
        avg_total_score = self._mean([self._to_float(item.get("total_score")) for item in episodes])
        avg_finished_steps = self._mean([self._to_float(item.get("finished_steps")) for item in episodes])
        avg_charge_count = self._mean([self._to_float(item.get("charge_count")) for item in episodes])

        bottleneck = "unknown"
        mean_fetch_ms = self._mean([self._to_float(item.get("data_fetch_ms")) for item in parsed_learner_perf])
        mean_train_ms = self._mean([self._to_float(item.get("real_train_ms")) for item in parsed_learner_perf])
        if mean_fetch_ms > 0 or mean_train_ms > 0:
            if mean_fetch_ms > mean_train_ms * 1.5:
                bottleneck = "pipeline_fetch_bound"
            elif mean_train_ms > mean_fetch_ms * 1.5:
                bottleneck = "learner_bound"
            else:
                bottleneck = "balanced"

        latest_checkpoint = None
        if checkpoints:
            latest_checkpoint = checkpoints[-1].get("checkpoint_id") or checkpoints[-1].get("id")

        return {
            "run_id": run_state.get("run_id"),
            "created_at": run_state.get("created_at"),
            "finalized_at": utc_now_iso(),
            "algorithm": run_state.get("algorithm"),
            "episode_count": len(episodes),
            "event_count": len(events),
            "checkpoint_count": len(checkpoints),
            "avg_total_score": avg_total_score,
            "avg_finished_steps": avg_finished_steps,
            "avg_charge_count": avg_charge_count,
            "fail_reason_counts": dict(fail_reason_counter),
            "best_checkpoint_id": checkpoint_analysis.get("best_checkpoint_id"),
            "latest_checkpoint_id": latest_checkpoint,
            "checkpoint_analysis": checkpoint_analysis,
            "bottleneck_classification": bottleneck,
            "mean_data_fetch_ms": mean_fetch_ms,
            "mean_real_train_ms": mean_train_ms,
        }

    def _write_report(self, run_dir: Path, summary: dict[str, Any]) -> None:
        report = (
            f"# Training Report\n\n"
            f"- Run ID: `{summary['run_id']}`\n"
            f"- Algorithm: `{summary['algorithm']}`\n"
            f"- Finalized At: `{summary['finalized_at']}`\n"
            f"- Episode Count: `{summary['episode_count']}`\n"
            f"- Checkpoint Count: `{summary['checkpoint_count']}`\n"
            f"- Avg Total Score: `{summary['avg_total_score']:.2f}`\n"
            f"- Avg Finished Steps: `{summary['avg_finished_steps']:.2f}`\n"
            f"- Avg Charge Count: `{summary['avg_charge_count']:.2f}`\n"
            f"- Best Checkpoint: `{summary.get('best_checkpoint_id')}`\n"
            f"- Latest Checkpoint: `{summary.get('latest_checkpoint_id')}`\n"
            f"- Bottleneck Classification: `{summary['bottleneck_classification']}`\n"
            f"- Fail Reason Counts: `{json.dumps(summary['fail_reason_counts'], ensure_ascii=True, sort_keys=True)}`\n"
        )
        (run_dir / "human" / "report.md").write_text(report, encoding="utf-8")

    def _compress_human_logs(self, run_dir: Path) -> None:
        raw_logs_dir = run_dir / "human" / "raw_logs"
        if not raw_logs_dir.exists():
            return
        archive_base = run_dir / "human" / "raw_logs"
        archive_path = shutil.make_archive(str(archive_base), "gztar", root_dir=raw_logs_dir)
        digest_path = run_dir / "human" / "raw_logs_bundle.txt"
        digest_path.write_text(archive_path, encoding="utf-8")

    def _copy_tree(self, src: Path, dst: Path) -> None:
        if not src.exists():
            return
        for path in src.rglob("*"):
            if not path.is_file():
                continue
            relative_path = path.relative_to(src)
            self._copy_file(path, dst / relative_path)

    def _copy_file(self, src: Path, dst: Path) -> None:
        if not src.exists():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            src_stat = src.stat()
            dst_stat = dst.stat()
            if src_stat.st_size == dst_stat.st_size and int(src_stat.st_mtime) == int(dst_stat.st_mtime):
                return
        shutil.copy2(src, dst)

    def _latest_mtime(self, *paths: Path) -> float | None:
        latest = None
        for base in paths:
            if not base.exists():
                continue
            if base.is_file():
                mtime = base.stat().st_mtime
                latest = mtime if latest is None else max(latest, mtime)
                continue
            for path in base.rglob("*"):
                if not path.is_file():
                    continue
                mtime = path.stat().st_mtime
                latest = mtime if latest is None else max(latest, mtime)
        return latest

    def _write_jsonl(self, path: Path, records: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def _mean(self, values: list[float]) -> float:
        values = [value for value in values if value is not None]
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def _to_float(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0


def main() -> None:
    ArchiveAgent().run_forever()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Experiment archive and structured logging helpers for Robot Vacuum.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


_STATE_LOCK = Lock()
_CHECKPOINT_PATTERN = re.compile(r"model\.ckpt-(?P<checkpoint_id>[^./\\]+)\.pkl")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_checkpoint_id(text: str | None) -> str | None:
    if not text:
        return None
    match = _CHECKPOINT_PATTERN.search(text)
    if match:
        return match.group("checkpoint_id")
    if text.isdigit():
        return text
    return None


def infer_fail_reason(
    terminated: bool,
    truncated: bool,
    battery: int | None = None,
    extra_info: dict[str, Any] | None = None,
) -> str:
    if truncated:
        return "completed"

    extra_info = extra_info or {}
    result_message = str(extra_info.get("result_message") or "").lower()
    if "battery" in result_message or "charge" in result_message:
        return "battery"
    if "collision" in result_message or "npc" in result_message or "robot" in result_message:
        return "collision"
    if terminated and battery is not None and battery <= 0:
        return "battery"
    if terminated:
        return "collision"
    return "unknown"


class ExperimentArchive:
    def __init__(self, archive_root: str | Path | None = None, service_name: str | None = None):
        self.code_root = Path(__file__).resolve().parents[2]
        self.archive_root = Path(
            archive_root or os.getenv("KAIWU_ARCHIVE_DIR") or self.code_root.parent / "train" / "archive"
        ).resolve()
        self.service_name = service_name or os.getenv("KAIWU_SERVICE_NAME") or "default"
        self._run_state: dict[str, Any] | None = None
        self._warned_errors: set[str] = set()

    def get_run_state(self, create_if_missing: bool = True) -> dict[str, Any] | None:
        try:
            with _STATE_LOCK:
                current_state_path = self.archive_root / ".current_run.json"
                if current_state_path.exists():
                    try:
                        state = json.loads(current_state_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        state = None
                    if state and state.get("status") != "finalized":
                        self._run_state = state
                        self._ensure_run_layout(state)
                        return deepcopy(state)
                    if not create_if_missing:
                        return None
                elif not create_if_missing:
                    return None

                state = self._create_run_state()
                self._run_state = state
                self._write_json(current_state_path, state)
                self._ensure_run_layout(state)
                return deepcopy(state)
        except Exception as exc:
            self._warn_once("get_run_state", exc)
            if self._run_state:
                return deepcopy(self._run_state)
            if not create_if_missing:
                return None
            state = self._create_run_state()
            self._run_state = state
            return deepcopy(state)

    def ensure_run(self, manifest_patch: dict[str, Any] | None = None) -> dict[str, Any]:
        state = self.get_run_state(create_if_missing=True)
        assert state is not None
        try:
            self.write_manifest(manifest_patch)
        except Exception as exc:
            self._warn_once("ensure_run", exc)
        return state

    def get_run_dir(self, create_if_missing: bool = True) -> Path | None:
        state = self.get_run_state(create_if_missing=create_if_missing)
        if not state:
            return None
        return self.archive_root / state["run_id"]

    def write_manifest(self, manifest_patch: dict[str, Any] | None = None) -> Path | None:
        try:
            run_dir = self.get_run_dir(create_if_missing=True)
            if run_dir is None:
                return None
            manifest_path = run_dir / "ai" / "run_manifest.json"
            manifest = self._default_manifest()
            if manifest_path.exists():
                try:
                    manifest.update(json.loads(manifest_path.read_text(encoding="utf-8")))
                except json.JSONDecodeError:
                    pass
            if manifest_patch:
                manifest.update(manifest_patch)
            manifest["updated_at"] = utc_now_iso()
            self._write_json(manifest_path, manifest)
            return manifest_path
        except Exception as exc:
            self._warn_once("write_manifest", exc)
            return None

    def log_jsonl(self, stream_name: str, payload: dict[str, Any]) -> Path | None:
        try:
            run_dir = self.get_run_dir(create_if_missing=True)
            if run_dir is None:
                return None

            record = {
                "ts": utc_now_iso(),
                "service": self.service_name,
                "pid": os.getpid(),
            }
            record.update(payload)

            stream_path = run_dir / "ai" / "streams" / f"{stream_name}.{self.service_name}.{os.getpid()}.jsonl"
            self._append_jsonl(stream_path, record)
            return stream_path
        except Exception as exc:
            self._warn_once(f"log_jsonl:{stream_name}", exc)
            return None

    def log_event(self, event_type: str, payload: dict[str, Any] | None = None) -> Path | None:
        payload = payload or {}
        record = {"event_type": event_type}
        record.update(payload)
        return self.log_jsonl("key_events", record)

    def log_episode_summary(self, payload: dict[str, Any]) -> Path | None:
        return self.log_jsonl("episode_summary", payload)

    def log_train_window(self, payload: dict[str, Any]) -> Path | None:
        return self.log_jsonl("train_window", payload)

    def log_checkpoint(self, payload: dict[str, Any]) -> Path | None:
        return self.log_jsonl("checkpoint_index", payload)

    def mark_run_finalized(self, summary_patch: dict[str, Any] | None = None) -> Path | None:
        try:
            with _STATE_LOCK:
                current_state_path = self.archive_root / ".current_run.json"
                if not current_state_path.exists():
                    return None
                try:
                    state = json.loads(current_state_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    return None
                if summary_patch:
                    state.update(summary_patch)
                state["status"] = "finalized"
                state["finalized_at"] = utc_now_iso()
                self._write_json(current_state_path, state)
            return current_state_path
        except Exception as exc:
            self._warn_once("mark_run_finalized", exc)
            return None

    def _create_run_state(self) -> dict[str, Any]:
        project_code, project_version = self._read_kaiwu_meta()
        algorithm = os.getenv("KAIWU_ALGORITHM") or "unknown"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = f"{project_code}-{algorithm}-{timestamp}"
        return {
            "run_id": run_id,
            "project_code": project_code,
            "project_version": project_version,
            "algorithm": algorithm,
            "created_at": utc_now_iso(),
            "status": "running",
        }

    def _default_manifest(self) -> dict[str, Any]:
        state = self.get_run_state(create_if_missing=True)
        assert state is not None
        project_code, project_version = self._read_kaiwu_meta()
        return {
            "run_id": state["run_id"],
            "created_at": state["created_at"],
            "status": state["status"],
            "project_code": project_code,
            "project_version": project_version,
            "algorithm": os.getenv("KAIWU_ALGORITHM") or "unknown",
            "service_name": self.service_name,
            "code_root": str(self.code_root),
            "env": self._collect_env_snapshot(),
        }

    def _collect_env_snapshot(self) -> dict[str, str]:
        snapshot = {}
        for key, value in sorted(os.environ.items()):
            if key.startswith("KAIWU_") or key in {
                "MONITOR_TRPC_PORT",
                "USER_ID",
                "MONITOR_ID",
                "TRACKER_ID",
                "CLIENT_ENV",
            }:
                snapshot[key] = value
        return snapshot

    def _read_kaiwu_meta(self) -> tuple[str, str]:
        kaiwu_json = self.code_root / "kaiwu.json"
        if not kaiwu_json.exists():
            return "robot_vacuum", "unknown"
        try:
            payload = json.loads(kaiwu_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return "robot_vacuum", "unknown"
        project_code = str(payload.get("project_code") or "robot_vacuum")
        project_version = str(payload.get("version") or "unknown")
        return project_code, project_version

    def _ensure_run_layout(self, state: dict[str, Any]) -> None:
        self.archive_root.mkdir(parents=True, exist_ok=True)
        run_dir = self.archive_root / state["run_id"]
        for rel_path in (
            "human/raw_logs",
            "human/config/code",
            "human/config/train",
            "human/checkpoints",
            "ai",
            "ai/streams",
        ):
            (run_dir / rel_path).mkdir(parents=True, exist_ok=True)
        self._snapshot_code_config(run_dir)

    def _snapshot_code_config(self, run_dir: Path) -> None:
        target_dir = run_dir / "human" / "config" / "code"
        snapshot_flag = target_dir / ".snapshot_complete"
        if snapshot_flag.exists():
            return

        for relative_path in (
            "kaiwu.json",
            "train_test.py",
            "conf/algo_conf_robot_vacuum.toml",
            "conf/app_conf_robot_vacuum.toml",
            "conf/configure_app.toml",
            "agent_ppo/conf/conf.py",
            "agent_ppo/conf/train_env_conf.toml",
            "agent_ppo/conf/monitor_builder.py",
        ):
            src = self.code_root / relative_path
            if not src.exists():
                continue
            dst = target_dir / relative_path.replace("/", "_")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        snapshot_flag.write_text(utc_now_iso(), encoding="utf-8")

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        rendered = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
        last_error: Exception | None = None
        for _ in range(3):
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
            try:
                tmp_path.write_text(rendered, encoding="utf-8")
                os.replace(tmp_path, path)
                return
            except FileNotFoundError as exc:
                last_error = exc
                time.sleep(0.05)
            except OSError as exc:
                last_error = exc
                time.sleep(0.05)
            finally:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass
        if last_error is not None:
            raise last_error

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        rendered = json.dumps(payload, ensure_ascii=True) + "\n"
        last_error: Exception | None = None
        for _ in range(3):
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(rendered)
                return
            except FileNotFoundError as exc:
                last_error = exc
                time.sleep(0.05)
            except OSError as exc:
                last_error = exc
                time.sleep(0.05)
        if last_error is not None:
            raise last_error

    def _warn_once(self, stage: str, exc: Exception) -> None:
        key = f"{stage}:{type(exc).__name__}:{exc}"
        if key in self._warned_errors:
            return
        self._warned_errors.add(key)
        print(
            f"[ExperimentArchive] {stage} failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )

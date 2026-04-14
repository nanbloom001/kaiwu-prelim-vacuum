#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Helpers for resolving container-local routing decisions at runtime."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import socket
import sys
import time
from dataclasses import dataclass


def resolve_service_index(current_ip: str | None, service_ips: list[str | None]) -> int | None:
    """Return the 1-based service index whose IP matches the current container IP."""
    if not current_ip:
        return None

    for index, service_ip in enumerate(service_ips, start=1):
        if service_ip and service_ip == current_ip:
            return index

    return None


def select_aisrv_gpu_target(
    aisrv_index: int | None,
    gpu1_num: int,
    gpu2_num: int,
    gpu1: str,
    gpu2: str,
    gpu3: str,
) -> str | None:
    """Map an aisrv replica index to the configured physical GPU group."""
    if aisrv_index is None or aisrv_index <= 0:
        return None

    gpu1_limit = max(gpu1_num, 0)
    gpu2_limit = gpu1_limit + max(gpu2_num, 0)

    if aisrv_index <= gpu1_limit:
        return gpu1
    if aisrv_index <= gpu2_limit:
        return gpu2
    return gpu3


def _resolve_name_ip(*names: str) -> str | None:
    original_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(1.0)
    try:
        for name in names:
            if not name:
                continue
            try:
                _, _, addresses = socket.gethostbyname_ex(name)
            except (OSError, socket.timeout):
                continue
            if addresses:
                return addresses[0]
        return None
    finally:
        socket.setdefaulttimeout(original_timeout)


def _read_current_container_name(explicit_name: str | None) -> str:
    if explicit_name:
        return explicit_name

    env_name = os.environ.get("HOSTNAME")
    if env_name:
        return env_name

    try:
        with open("/etc/hostname", "r", encoding="utf-8") as file_obj:
            current_name = file_obj.read().strip()
    except (FileNotFoundError, PermissionError, OSError) as exc:
        raise RuntimeError(f"Failed to read container hostname: {exc}") from exc

    if not current_name:
        raise RuntimeError("Failed to read container hostname: /etc/hostname is empty")

    return current_name


def _validate_gpu_value(value: str, name: str) -> str:
    candidate = value.strip()
    if not candidate:
        raise ValueError(f"{name} cannot be empty")
    if not re.fullmatch(r"[A-Za-z0-9._:-]+(?:,[A-Za-z0-9._:-]+)*", candidate):
        raise ValueError(f"{name} contains unsupported characters: {value!r}")
    return candidate


@dataclass(frozen=True)
class AisrvRoutingResult:
    current_name: str
    current_ip: str | None
    aisrv_index: int | None
    target_gpu: str | None

    @property
    def is_resolved(self) -> bool:
        return self.aisrv_index is not None and self.target_gpu is not None


def _collect_service_ips(project_name: str, service_count: int) -> list[str | None]:
    return [
        _resolve_name_ip(
            f"{project_name}-aisrv-{index}",
            f"{project_name}_aisrv_{index}",
        )
        for index in range(1, service_count + 1)
    ]


def detect_aisrv_routing(
    current_name: str,
    service_count: int,
    project_name: str,
    gpu1_num: int,
    gpu2_num: int,
    gpu1: str,
    gpu2: str,
    gpu3: str,
    resolution_attempts: int = 1,
    retry_delay_seconds: float = 0.0,
) -> AisrvRoutingResult:
    """Resolve the current aisrv replica index and matching target GPU."""
    attempts = max(resolution_attempts, 1)
    last_result = AisrvRoutingResult(
        current_name=current_name,
        current_ip=None,
        aisrv_index=None,
        target_gpu=None,
    )

    for attempt in range(attempts):
        current_ip = _resolve_name_ip(current_name)
        service_ips = _collect_service_ips(project_name, service_count)
        aisrv_index = resolve_service_index(current_ip, service_ips)
        target_gpu = select_aisrv_gpu_target(
            aisrv_index=aisrv_index,
            gpu1_num=gpu1_num,
            gpu2_num=gpu2_num,
            gpu1=gpu1,
            gpu2=gpu2,
            gpu3=gpu3,
        )
        last_result = AisrvRoutingResult(
            current_name=current_name,
            current_ip=current_ip,
            aisrv_index=aisrv_index,
            target_gpu=target_gpu,
        )
        if last_result.is_resolved:
            return last_result
        if attempt + 1 < attempts and retry_delay_seconds > 0:
            time.sleep(retry_delay_seconds)

    return last_result


def build_shell_exports(result: AisrvRoutingResult) -> str:
    """Render the routing result as shell exports for docker-compose commands."""
    exports = {"KAIWU_AISRV_CURRENT_NAME": result.current_name}
    if result.current_ip:
        exports["KAIWU_AISRV_CURRENT_IP"] = result.current_ip
    if result.aisrv_index is not None:
        exports["KAIWU_AISRV_INDEX"] = str(result.aisrv_index)
    if result.target_gpu:
        exports["KAIWU_AISRV_TARGET_GPU"] = result.target_gpu
    if result.target_gpu:
        exports["NVIDIA_VISIBLE_DEVICES"] = result.target_gpu
        exports["CUDA_VISIBLE_DEVICES"] = result.target_gpu

    return "\n".join(
        f"export {key}={shlex.quote(value)}"
        for key, value in exports.items()
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resolve aisrv replica index and target GPU.")
    parser.add_argument("--current-name", default=None)
    parser.add_argument("--service-count", type=int, required=True)
    parser.add_argument("--project-name", default=os.environ.get("COMPOSE_PROJECT_NAME", "kaiwu-train"))
    parser.add_argument("--gpu1-num", type=int, required=True)
    parser.add_argument("--gpu2-num", type=int, required=True)
    parser.add_argument("--gpu1", required=True)
    parser.add_argument("--gpu2", required=True)
    parser.add_argument("--gpu3", required=True)
    parser.add_argument("--resolution-attempts", type=int, default=1)
    parser.add_argument("--retry-delay-seconds", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for resolving aisrv replica routing and target GPU."""
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    try:
        current_name = _read_current_container_name(args.current_name)
        gpu1 = _validate_gpu_value(args.gpu1, "gpu1")
        gpu2 = _validate_gpu_value(args.gpu2, "gpu2")
        gpu3 = _validate_gpu_value(args.gpu3, "gpu3")
    except (RuntimeError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1

    result = detect_aisrv_routing(
        current_name=current_name,
        service_count=args.service_count,
        project_name=args.project_name,
        gpu1_num=args.gpu1_num,
        gpu2_num=args.gpu2_num,
        gpu1=gpu1,
        gpu2=gpu2,
        gpu3=gpu3,
        resolution_attempts=args.resolution_attempts,
        retry_delay_seconds=args.retry_delay_seconds,
    )

    if not result.is_resolved:
        sys.stderr.write(
            "Failed to resolve aisrv routing: "
            f"current_name={result.current_name!r} current_ip={result.current_ip!r} "
            f"project_name={args.project_name!r} service_count={args.service_count}\n"
        )
        return 1

    sys.stdout.write(build_shell_exports(result))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
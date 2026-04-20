#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from __future__ import annotations

import re
from pathlib import Path


RUN_SCOPED_STOP_PATCH_MARKER = "# RUN_SCOPED_STOP_PATCH_V1"
TRAIN_TEST_STOP_PATCH_MARKER = "# RUN_SCOPED_TRAIN_TEST_STOP_PATCH_V1"


COMMON_SH_STOP_FUNCTION_PATTERN = re.compile(
    r"function check_process_stop_done\(\)\n\{\n.*?\n\}\n",
    re.DOTALL,
)


COMMON_SH_STOP_FUNCTION_NEW = """function check_process_stop_done()
{
    kaiwudrl_configure_file=$project_dir/kaiwudrl/conf/kaiwudrl/configure.toml
    current_run_manifest=${KAIWU_CURRENT_RUN_MANIFEST:-/workspace/code/runtime_state/current/run_session.json}
    process_stop_done_file="/data/ckpt/${app}_${algo}/process_stop.done"
    process_stop_meta_file="/data/ckpt/${app}_${algo}/process_stop.meta.json"
    boot_ts=${KAIWU_RUN_BOOT_TS:-0}
    while true;
    do
        # 检测是否正常结束, 如果不是正常结束则提前退出
        if [ ! -f "$process_stop_done_file" ];
        then
            sleep 5
            continue
        fi

        stop_file_mtime=$(stat -c %Y "$process_stop_done_file" 2>/dev/null || echo 0)
        if [ "$boot_ts" -gt 0 ] && [ "$stop_file_mtime" -lt "$boot_ts" ];
        then
            log_info "ignore stale process_stop.done older than current run boot time"
            rm -f "$process_stop_done_file" "$process_stop_meta_file"
            sigterm_pids_file=$(sed -n 's/^[[:space:]]*sigterm_pids_file[[:space:]]*=[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p' "$kaiwudrl_configure_file" | head -n 1)
            if [[ "$sigterm_pids_file" ]];
            then
                : > "$sigterm_pids_file"
            fi
            sleep 1
            continue
        fi

        current_run_session_id=$(python3 - <<'PY'
import json
import os
from pathlib import Path

manifest = Path(os.environ.get("KAIWU_CURRENT_RUN_MANIFEST", "/workspace/code/runtime_state/current/run_session.json"))
try:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
except (FileNotFoundError, OSError, json.JSONDecodeError):
    print("")
else:
    print(str(payload.get("run_session_id") or "").strip())
PY
)
        stop_run_session_id=$(python3 - <<'PY'
import json
from pathlib import Path

target = Path("/data/ckpt/${app}_${algo}/process_stop.meta.json")
try:
    payload = json.loads(target.read_text(encoding="utf-8"))
except (FileNotFoundError, OSError, json.JSONDecodeError):
    print("")
else:
    print(str(payload.get("run_session_id") or "").strip())
PY
)
        if [[ -n "$current_run_session_id" && -n "$stop_run_session_id" && "$current_run_session_id" != "$stop_run_session_id" ]];
        then
            log_info "ignore stale process_stop.done for another run_session_id"
            rm -f "$process_stop_done_file" "$process_stop_meta_file"
            sigterm_pids_file=$(sed -n 's/^[[:space:]]*sigterm_pids_file[[:space:]]*=[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p' "$kaiwudrl_configure_file" | head -n 1)
            if [[ "$sigterm_pids_file" ]];
            then
                : > "$sigterm_pids_file"
            fi
            sleep 1
            continue
        fi

        log_info "train success, sending SIGTERM to sigterm_pids processes."
        sigterm_pids_file=$(sed -n 's/^[[:space:]]*sigterm_pids_file[[:space:]]*=[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p' "$kaiwudrl_configure_file" | head -n 1)
        if [[ "$sigterm_pids_file" ]];
        then
            touch "$sigterm_pids_file"
            pids=$(cat "$sigterm_pids_file")

            if [[ "$pids" ]];
            then
                echo "Sending SIGTERM to processes: $pids"
                kill -15 $pids
            fi
        fi

        # 这里需要暂停一定时间, 需要让model文件上传成功, 否则容器关闭了, model文件无法上传
        sleep 30

        # 读取文件中的值
        exit_code=$(cat "$process_stop_done_file")
        # 使用读取到的值作为exit的参数
        exit $exit_code
        break
    done
}
"""


TRAIN_TEST_STOP_FUNCTION_PATTERN = re.compile(
    r"def _check_process_stop_done\(\) -> bool:\n"
    r'    """Check if the process_stop\.done file exists\.\n\n'
    r"    :returns: True if the done file exists, False otherwise\.\n"
    r'    """\n'
    r'    done_file = f"/data/ckpt/\{CONFIG\.app\}_\{CONFIG\.algo\}/process_stop\.done"\n'
    r"    return os\.path\.exists\(done_file\)\n",
    re.MULTILINE,
)


TRAIN_TEST_STOP_FUNCTION_REPLACEMENT = """def _check_process_stop_done() -> bool:
    \"\"\"Check if the process_stop.done file belongs to the current run.\"\"\"
    done_file = f"/data/ckpt/{CONFIG.app}_{CONFIG.algo}/process_stop.done"
    if not os.path.exists(done_file):
        return False

    boot_ts = int(os.environ.get("KAIWU_RUN_BOOT_TS", "0") or "0")
    if boot_ts > 0:
        try:
            stop_mtime = int(os.path.getmtime(done_file))
        except OSError:
            return False
        if stop_mtime < boot_ts:
            return False

    manifest = Path(os.environ.get("KAIWU_CURRENT_RUN_MANIFEST", "/workspace/code/runtime_state/current/run_session.json"))
    stop_meta = Path(f"/data/ckpt/{CONFIG.app}_{CONFIG.algo}/process_stop.meta.json")
    try:
        current_payload = json.loads(manifest.read_text(encoding="utf-8"))
        current_run = str(current_payload.get("run_session_id") or "").strip()
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        current_run = ""

    try:
        stop_payload = json.loads(stop_meta.read_text(encoding="utf-8"))
        stop_run = str(stop_payload.get("run_session_id") or "").strip()
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        stop_run = ""

    if current_run and stop_run and current_run != stop_run:
        return False
    return True
"""


def inject_run_scoped_stop_guard(common_sh_source: str) -> tuple[str, bool]:
    if RUN_SCOPED_STOP_PATCH_MARKER in common_sh_source:
        return common_sh_source, False
    updated, changed = COMMON_SH_STOP_FUNCTION_PATTERN.subn(
        lambda _match: f"{RUN_SCOPED_STOP_PATCH_MARKER}\n{COMMON_SH_STOP_FUNCTION_NEW}\n",
        common_sh_source,
        count=1,
    )
    if changed != 1:
        raise ValueError("check_process_stop_done anchor not found in common.sh")
    return updated, True


def inject_train_test_stop_guard(train_test_source: str) -> tuple[str, bool]:
    if TRAIN_TEST_STOP_PATCH_MARKER in train_test_source:
        return train_test_source, False
    if "import json\n" not in train_test_source:
        train_test_source = train_test_source.replace("import glob\n", "import glob\nimport json\n", 1)
    if "from pathlib import Path\n" not in train_test_source:
        train_test_source = train_test_source.replace("import time\n", "import time\nfrom pathlib import Path\n", 1)
    updated, changed = TRAIN_TEST_STOP_FUNCTION_PATTERN.subn(
        lambda _match: f"{TRAIN_TEST_STOP_PATCH_MARKER}\n{TRAIN_TEST_STOP_FUNCTION_REPLACEMENT}",
        train_test_source,
        count=1,
    )
    if changed != 1:
        raise ValueError("_check_process_stop_done anchor not found in train_test_utils.py")
    return updated, True


def _patch_file(target: Path, injector) -> bool:
    original = target.read_text(encoding="utf-8")
    updated, changed = injector(original)
    if changed:
        target.write_text(updated, encoding="utf-8")
    return changed


def apply_run_scoped_stop_patches(root: Path, tools_root: Path) -> list[Path]:
    patched: list[Path] = []
    common_sh = tools_root / "common.sh"
    if common_sh.exists() and _patch_file(common_sh, inject_run_scoped_stop_guard):
        patched.append(common_sh)

    train_test_utils = root / "kaiwudrl/common/utils/train_test_utils.py"
    if train_test_utils.exists() and _patch_file(train_test_utils, inject_train_test_stop_guard):
        patched.append(train_test_utils)

    return patched


__all__ = [
    "RUN_SCOPED_STOP_PATCH_MARKER",
    "TRAIN_TEST_STOP_PATCH_MARKER",
    "inject_run_scoped_stop_guard",
    "inject_train_test_stop_guard",
    "apply_run_scoped_stop_patches",
]

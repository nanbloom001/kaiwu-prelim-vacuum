#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import tempfile
import unittest
from pathlib import Path

from agent_ppo.utils.run_scoped_stop_patch import (
    RUN_SCOPED_STOP_PATCH_MARKER,
    TRAIN_TEST_STOP_PATCH_MARKER,
    apply_run_scoped_stop_patches,
    inject_run_scoped_stop_guard,
    inject_train_test_stop_guard,
)


COMMON_SH_SOURCE = """function check_process_stop_done()
{
    kaiwudrl_configure_file=$project_dir/kaiwudrl/conf/kaiwudrl/configure.toml
    while true;
    do
        # 检测是否正常结束, 如果不是正常结束则提前退出
        if [ ! -f "/data/ckpt/${app}_${algo}/process_stop.done" ];
        then
            sleep 5
            continue
        fi

        log_info "train success, sending SIGTERM to sigterm_pids processes."
        sigterm_pids_file=$(grep -oP 'sigterm_pids_file\\s*=\\s*[\"'\\''']\\K[^\"'\\''\"]+' $kaiwudrl_configure_file 2>/dev/null || true)
        if [[ "$sigterm_pids_file" ]];
        then
            touch $sigterm_pids_file
            pids=$(cat $sigterm_pids_file)

            if [[ "$pids" ]];
            then
                echo "Sending SIGTERM to processes: $pids"
                kill -15 $pids
            fi
        fi

        # 这里需要暂停一定时间, 需要让model文件上传成功, 否则容器关闭了, model文件无法上传
        sleep 30

        # 读取文件中的值
        exit_code=$(cat /data/ckpt/${app}_${algo}/process_stop.done)
        # 使用读取到的值作为exit的参数
        exit $exit_code
        break
    done
}
"""


TRAIN_TEST_SOURCE = """#!/usr/bin/env python3
import glob
import os
import platform
import sys
import time
from multiprocessing import Process
from typing import List

from common_python.config.config_control import CONFIG

def _check_process_stop_done() -> bool:
    \"\"\"Check if the process_stop.done file exists.

    :returns: True if the done file exists, False otherwise.
    \"\"\"
    done_file = f"/data/ckpt/{CONFIG.app}_{CONFIG.algo}/process_stop.done"
    return os.path.exists(done_file)
"""


class RunScopedStopPatchTests(unittest.TestCase):
    def test_inject_run_scoped_stop_guard_is_idempotent(self):
        updated, changed = inject_run_scoped_stop_guard(COMMON_SH_SOURCE)
        updated_again, changed_again = inject_run_scoped_stop_guard(updated)

        self.assertTrue(changed)
        self.assertFalse(changed_again)
        self.assertIn(RUN_SCOPED_STOP_PATCH_MARKER, updated)
        self.assertIn("boot_ts=${KAIWU_RUN_BOOT_TS:-0}", updated)
        self.assertEqual(updated, updated_again)

    def test_inject_train_test_stop_guard_is_idempotent(self):
        updated, changed = inject_train_test_stop_guard(TRAIN_TEST_SOURCE)
        updated_again, changed_again = inject_train_test_stop_guard(updated)

        self.assertTrue(changed)
        self.assertFalse(changed_again)
        self.assertIn(TRAIN_TEST_STOP_PATCH_MARKER, updated)
        self.assertIn("json.loads", updated)
        self.assertIn("from pathlib import Path", updated)
        self.assertEqual(updated, updated_again)

    def test_apply_run_scoped_stop_patches_updates_both_targets(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "project"
            tools_root = Path(tmp_dir) / "tools"
            common_sh = tools_root / "common.sh"
            train_test = root / "kaiwudrl/common/utils/train_test_utils.py"
            common_sh.parent.mkdir(parents=True, exist_ok=True)
            train_test.parent.mkdir(parents=True, exist_ok=True)
            common_sh.write_text(COMMON_SH_SOURCE, encoding="utf-8")
            train_test.write_text(TRAIN_TEST_SOURCE, encoding="utf-8")

            patched = apply_run_scoped_stop_patches(root, tools_root)

            self.assertEqual(set(patched), {common_sh, train_test})
            self.assertIn(RUN_SCOPED_STOP_PATCH_MARKER, common_sh.read_text(encoding="utf-8"))
            self.assertIn(TRAIN_TEST_STOP_PATCH_MARKER, train_test.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env bash
set -euo pipefail

echo "[aisrv-runtime] replay_buffer_type=${KAIWU_EXPERIMENT_REPLAY_BUFFER_TYPE:-reverb} send_sample_size=${KAIWU_EXPERIMENT_SEND_SAMPLE_SIZE:-unset} predict_batch_size=${KAIWU_EXPERIMENT_PREDICT_BATCH_SIZE:-unset}"

python3 - <<'PY'
from pathlib import Path
import os

project = os.environ.get("KAIWU_PROJECT_CODE", "robot_vacuum")
root = Path(f"/data/projects/{project}")

(root / "kaiwudrl/aisrv_post_init_patch.py").write_text(
    "import os\n"
    "from pathlib import Path\n"
    "\n"
    "def format_toml_value(value):\n"
    "    value = str(value)\n"
    "    if value.lower() in {'true', 'false'}:\n"
    "        return value.lower()\n"
    "    try:\n"
    "        int(value)\n"
    "        return value\n"
    "    except ValueError:\n"
    "        return f'\"{value}\"'\n"
    "\n"
    "def replace_toml_key(target, key, env_name):\n"
    "    env_value = os.environ.get(env_name)\n"
    "    if env_value in (None, '') or not target.exists():\n"
    "        return\n"
    "    lines = target.read_text().splitlines()\n"
    "    replacement = f'{key} = {format_toml_value(env_value)}'\n"
    "    filtered = [line for line in lines if line.strip().split('=', 1)[0].strip() != key]\n"
    "    filtered.append(replacement)\n"
    "    target.write_text('\\n'.join(filtered) + '\\n')\n"
    "    print(f'[aisrv-post-init] {key} = {env_value}')\n"
    "\n"
    "project = os.environ.get('KAIWU_PROJECT_CODE', 'robot_vacuum')\n"
    "root = Path(f'/data/projects/{project}')\n"
    "configure_toml = root / 'kaiwudrl/conf/kaiwudrl/configure.toml'\n"
    "replace_toml_key(configure_toml, 'aisrv_connect_to_kaiwu_env_count', 'KAIWU_PARALLEL_ENV_PER_AISRV')\n"
    "replace_toml_key(configure_toml, 'predict_batch_size', 'KAIWU_EXPERIMENT_PREDICT_BATCH_SIZE')\n"
    "replace_toml_key(configure_toml, 'proxy_batch_size', 'KAIWU_EXPERIMENT_PREDICT_BATCH_SIZE')\n"
)
PY

project_default=/root/tools/conf/project_default.toml
project_root="/data/projects/${KAIWU_PROJECT_CODE:-robot_vacuum}"

if [ -n "${KAIWU_EXPERIMENT_REPLAY_BUFFER_TYPE:-}" ]; then
  sed -i "s|sh tools/change_sample_server.sh reverb|sh tools/change_sample_server.sh ${KAIWU_EXPERIMENT_REPLAY_BUFFER_TYPE}|g" /root/tools/common.sh
  echo "[aisrv-sed] patched common.sh to use ${KAIWU_EXPERIMENT_REPLAY_BUFFER_TYPE}"
fi

if [ -n "${KAIWU_EXPERIMENT_DUMP_MODEL_FREQ:-}" ]; then
  sed -i "s|^dump_model_freq = .*|dump_model_freq = ${KAIWU_EXPERIMENT_DUMP_MODEL_FREQ}|g" "${project_default}"
fi

if [ -n "${KAIWU_EXPERIMENT_MODEL_FILE_SYNC_PER_MINUTES:-}" ]; then
  sed -i "s|^model_file_sync_per_minutes = .*|model_file_sync_per_minutes = ${KAIWU_EXPERIMENT_MODEL_FILE_SYNC_PER_MINUTES}|g" "${project_default}"
fi

python3 - <<'PY'
from pathlib import Path

target = Path("/root/tools/start_train_client.sh")
text = target.read_text()
anchor = "sh tools/change_alloc_process_count.sh kaiwu_env $${parallel_env_per_aisrv}\n"
inserted = "change_config_in_file aisrv_connect_to_kaiwu_env_count $${parallel_env_per_aisrv} $${kaiwudrl_configure_file} int\n"
if inserted not in text:
    if anchor not in text:
        print(f"[aisrv-startup-patch] anchor not found in {target}, skip in-place patch")
    else:
        text = text.replace(anchor, anchor + inserted, 1)
        target.write_text(text)
        print("[aisrv-startup-patch] inserted aisrv_connect_to_kaiwu_env_count before start_aisrv")
else:
    print("[aisrv-startup-patch] aisrv_connect_to_kaiwu_env_count already patched")
PY

bash -lc "/root/tools/start_train_client.sh aisrv & sleep 10 && python3 ${project_root}/kaiwudrl/aisrv_post_init_patch.py && wait"

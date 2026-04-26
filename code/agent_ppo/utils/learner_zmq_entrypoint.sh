#!/usr/bin/env bash
set -euo pipefail

echo "[learner-runtime] replay_buffer_type=${KAIWU_EXPERIMENT_REPLAY_BUFFER_TYPE:-reverb} train_batch_size=${KAIWU_EXPERIMENT_TRAIN_BATCH_SIZE:-unset} send_sample_size=${KAIWU_EXPERIMENT_SEND_SAMPLE_SIZE:-unset} predict_batch_size=${KAIWU_EXPERIMENT_PREDICT_BATCH_SIZE:-unset}"

python3 - <<'PY'
from pathlib import Path
import os
import sys

project = os.environ.get("KAIWU_PROJECT_CODE", "robot_vacuum")
root = Path(f"/data/projects/{project}")
sys.path.insert(0, "/workspace/code")

from agent_ppo.utils.zmq_patch import patch_zmq_runtime_files


def write_post_init_patch(target: Path) -> None:
    target.write_text(
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
        "        try:\n"
        "            float(value)\n"
        "            return value\n"
        "        except ValueError:\n"
        "            pass\n"
        "    return f'\"{value}\"'\n"
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
        "    print(f'[post-init] {key} = {env_value}')\n"
        "\n"
        "project = os.environ.get('KAIWU_PROJECT_CODE', 'robot_vacuum')\n"
        "root = Path(f'/data/projects/{project}')\n"
        "app_conf = root / 'conf/configure_app.toml'\n"
        "learner_conf = root / 'kaiwudrl/conf/kaiwudrl/learner.toml'\n"
        "configure_toml = root / 'kaiwudrl/conf/kaiwudrl/configure.toml'\n"
        "replace_toml_key(app_conf, 'train_batch_size', 'KAIWU_EXPERIMENT_TRAIN_BATCH_SIZE')\n"
        "replace_toml_key(app_conf, 'replay_buffer_capacity', 'KAIWU_EXPERIMENT_REPLAY_BUFFER_CAPACITY')\n"
        "replace_toml_key(app_conf, 'preload_ratio', 'KAIWU_EXPERIMENT_PRELOAD_RATIO')\n"
        "replace_toml_key(app_conf, 'dump_model_freq', 'KAIWU_EXPERIMENT_DUMP_MODEL_FREQ')\n"
        "replace_toml_key(app_conf, 'model_file_sync_per_minutes', 'KAIWU_EXPERIMENT_MODEL_FILE_SYNC_PER_MINUTES')\n"
        "replace_toml_key(app_conf, 'reverb_rate_limiter', 'KAIWU_EXPERIMENT_REVERB_RATE_LIMITER')\n"
        "replace_toml_key(app_conf, 'reverb_sampler', 'KAIWU_EXPERIMENT_REVERB_SAMPLER')\n"
        "replace_toml_key(configure_toml, 'dump_model_freq', 'KAIWU_EXPERIMENT_DUMP_MODEL_FREQ')\n"
        "replace_toml_key(configure_toml, 'model_file_sync_per_minutes', 'KAIWU_EXPERIMENT_MODEL_FILE_SYNC_PER_MINUTES')\n"
        "replace_toml_key(learner_conf, 'pytorch_read_data_from_reverb_type', 'KAIWU_PYTORCH_READ_DATA_FROM_REVERB_TYPE')\n"
        "replace_toml_key(learner_conf, 'replay_buffer_cache_multiplier', 'KAIWU_EXPERIMENT_REPLAY_BUFFER_CACHE_MULTIPLIER')\n"
        "replace_toml_key(configure_toml, 'predict_batch_size', 'KAIWU_EXPERIMENT_PREDICT_BATCH_SIZE')\n"
        "replace_toml_key(configure_toml, 'proxy_batch_size', 'KAIWU_EXPERIMENT_PREDICT_BATCH_SIZE')\n"
    )


replay_buffer_type = os.environ.get("KAIWU_EXPERIMENT_REPLAY_BUFFER_TYPE", "reverb")
patched = patch_zmq_runtime_files(root, replay_buffer_type)
for target in patched:
    print(f"[learner-zmq] patched spawn context in {target}")

if replay_buffer_type.strip().lower() == "zmq":
    for relative in (
        "kaiwudrl/common/utils/mem_buffer.py",
        "kaiwudrl/common/utils/mem_buffer_ratio.py",
    ):
        target = root / relative
        if not target.exists():
            continue
        text = target.read_text()
        original = text
        text = text.replace(
            "        self.device = \"cpu\"\n        if torch.cuda.is_available():\n            self.device = \"cuda\"\n",
            "        # ZMQ: keep mem_buffer CPU-only to avoid parent-process device init.\n        self.device = \"cpu\"\n",
            1,
        )
        text = text.replace(
            '        self.device = "cuda" if torch.cuda.is_available() else "cpu"\n',
            '        # ZMQ: keep mem_buffer CPU-only to avoid parent-process device init.\n        self.device = "cpu"\n',
            1,
        )
        if text != original:
            target.write_text(text)
            print(f"[learner-zmq] forced CPU-only mem_buffer init in {target}")

    mem_buffer_target = root / "kaiwudrl/common/utils/mem_buffer.py"
    if mem_buffer_target.exists():
        text = mem_buffer_target.read_text()
        original = text
        text = text.replace(
            "from multiprocessing import Value, Array, Queue",
            "from multiprocessing import Value, Array, Queue, Lock",
            1,
        )
        text = text.replace(
            "self._data_status = [Value(ctypes.c_bool, False, lock=True) for _ in range(max_sample_num)]",
            "self._data_status_arr = Array(ctypes.c_bool, max_sample_num, lock=False)  # lock_free_data_status\n        self._data_status_lock = Lock()",
            1,
        )
        text = text.replace("with self._data_status[partition_index].get_lock():", "with self._data_status_lock:")
        text = text.replace("with self._data_status[index].get_lock():", "with self._data_status_lock:")
        text = text.replace("with self._data_status[idx].get_lock():", "with self._data_status_lock:")
        text = text.replace("with self._data_status[i].get_lock():", "with self._data_status_lock:")
        text = text.replace("self._data_status[partition_index].value", "self._data_status_arr[partition_index]")
        text = text.replace("self._data_status[index].value", "self._data_status_arr[index]")
        text = text.replace("self._data_status[idx].value", "self._data_status_arr[idx]")
        text = text.replace("self._data_status[i].value", "self._data_status_arr[i]")
        if text != original:
            mem_buffer_target.write_text(text)
            print(f"[learner-zmq] patched mem_buffer lock-free status array in {mem_buffer_target}")

write_post_init_patch(root / "kaiwudrl/post_init_patch.py")
PY

project_default=/root/tools/conf/project_default.toml
project_root="/data/projects/${KAIWU_PROJECT_CODE:-robot_vacuum}"

if [ -n "${KAIWU_EXPERIMENT_DUMP_MODEL_FREQ:-}" ]; then
  sed -i "s|^dump_model_freq = .*|dump_model_freq = ${KAIWU_EXPERIMENT_DUMP_MODEL_FREQ}|g" "${project_default}"
fi

if [ -n "${KAIWU_EXPERIMENT_MODEL_FILE_SYNC_PER_MINUTES:-}" ]; then
  sed -i "s|^model_file_sync_per_minutes = .*|model_file_sync_per_minutes = ${KAIWU_EXPERIMENT_MODEL_FILE_SYNC_PER_MINUTES}|g" "${project_default}"
fi

if [ -n "${KAIWU_EXPERIMENT_REPLAY_BUFFER_TYPE:-}" ]; then
  sed -i "s|sh tools/change_sample_server.sh reverb|sh tools/change_sample_server.sh ${KAIWU_EXPERIMENT_REPLAY_BUFFER_TYPE}|g" /root/tools/common.sh
  echo "[learner-sed] patched common.sh to use ${KAIWU_EXPERIMENT_REPLAY_BUFFER_TYPE}"
fi

bash -lc "/root/tools/start_train_client.sh learner & sleep 10 && python3 ${project_root}/kaiwudrl/post_init_patch.py && wait"

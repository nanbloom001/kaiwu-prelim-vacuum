# MODEL_SIGNER_V2
"""
Thread-based model file signer that replicates ModelFileSave.process_model_file().

Produces signed zip packages matching the competition platform format:
  - Includes full code directory (agent_diy/, agent_ppo/), conf/, and ckpt/
  - kaiwu.json with all required metadata fields
  - Naming: {project}-{algo}-{step}-{datetime}-{version}.zip

Data sources (tried in order):
  1. User-checkpoint tar.gz files in CONFIG.user_ckpt_dir (preferred —
     created by after_save_model() when aisrv sends save_model requests)
  2. Framework .pkl checkpoints in CONFIG.restore_dir (fallback —
     always available, metadata constructed from CONFIG values)
"""
import threading
import json
import os
import time
import shutil
import re
from datetime import datetime, timezone
from common_python.config.config_control import CONFIG


class _BoolValue:
    def __init__(self):
        self._value = False

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self._value = v


class ModelSignerThread:
    def __init__(self, logger):
        self.logger = logger
        self.exit_flag = _BoolValue()
        self._thread = None
        self._processed = set()
        self._start_time = time.monotonic()

        # Framework paths
        self._user_ckpt_dir = CONFIG.user_ckpt_dir
        self._ckpt_dir = f'{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}'
        self._output_dir = os.path.join(
            CONFIG.standard_upload_file_dir, 'signed'
        )
        self._cos_target_dir = CONFIG.cos_local_target_dir
        self._copy_dirs = [
            d.strip() for d in CONFIG.copy_dir.split(',') if d.strip()
        ]
        self._code_dir = os.environ.get(
            'KAIWU_CODE_MOUNT', '/workspace/code'
        )

        # Signing key
        self._private_key = None
        self._interval_sec = 60

        os.makedirs(self._output_dir, exist_ok=True)
        self._init_signing_key()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name='model-signer'
        )
        self._thread.start()
        self.logger.info('[ModelSigner] Started in-process signing thread')

    def stop(self):
        self.exit_flag.value = True
        self.logger.info('[ModelSigner] Stop requested')

    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Key management
    # ------------------------------------------------------------------

    def _init_signing_key(self):
        from kaiwudrl.common.utils.common_func import (
            load_private_key_by_data,
            generate_private_key,
        )

        if CONFIG.private_key_content:
            try:
                self._private_key = load_private_key_by_data(
                    CONFIG.private_key_content
                )
                self.logger.info('[ModelSigner] Loaded platform private key')
                return
            except Exception:
                pass

        key_dir = os.path.join(self._output_dir, '.keys')
        os.makedirs(key_dir, exist_ok=True)
        priv_path = os.path.join(key_dir, 'private_key.pem')
        if os.path.exists(priv_path):
            with open(priv_path, 'r') as f:
                self._private_key = load_private_key_by_data(f.read())
            self.logger.info(
                f'[ModelSigner] Loaded existing private key from {priv_path}'
            )
        else:
            self._private_key, _ = generate_private_key(key_dir)
            self.logger.info(
                f'[ModelSigner] Generated new private key at {key_dir}'
            )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run_loop(self):
        while not self.exit_flag.value:
            try:
                self._process_tar_gz_files()
                self._process_pkl_checkpoints()
            except Exception as e:
                self.logger.error(f'[ModelSigner] Error: {e}')
            time.sleep(self._interval_sec)

    # ------------------------------------------------------------------
    # Source 1: tar.gz files from user_ckpt_dir (preferred)
    # ------------------------------------------------------------------

    def _process_tar_gz_files(self):
        id_list_path = os.path.join(self._user_ckpt_dir, 'id_list')
        if not os.path.exists(id_list_path):
            return
        with open(id_list_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            path = line.strip()
            if not path or not path.endswith('.tar.gz'):
                continue
            if path in self._processed:
                continue
            if self._process_tar_gz(path):
                self._processed.add(path)

    def _process_tar_gz(self, tar_gz_path):
        from kaiwudrl.common.utils.common_func import (
            tar_file_extract,
            python_exec_shell,
            compute_directory_hash,
            get_map_content,
            generate_private_signature_by_data,
            base64_encode,
            get_first_last_line_from_file,
            make_single_dir,
        )
        from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine

        if not os.path.exists(tar_gz_path):
            return False

        model_file_name = tar_gz_path.split('/')[-1]
        base_name = model_file_name.replace('.tar.gz', '')

        staging = self._cos_target_dir
        self._clean_dir(staging)
        make_single_dir(staging)

        shutil.copy(tar_gz_path, staging)
        for copy_dir in self._copy_dirs:
            python_exec_shell(f'cp -r {copy_dir} {staging}/')
        if os.path.isdir(self._code_dir):
            for agent_dir in ('agent_diy', 'agent_ppo'):
                src = os.path.join(self._code_dir, agent_dir)
                if os.path.isdir(src):
                    dst = os.path.join(staging, agent_dir)
                    shutil.copytree(src, dst, dirs_exist_ok=True)

        tar_file_extract(
            f'{staging}/{model_file_name}', f'{staging}/ckpt'
        )
        python_exec_shell(
            f'cd {staging}/ckpt && cd */ && mv * ../ && cd ../ && rm -rf */'
        )

        ckpt_id_list = f'{staging}/ckpt/{KaiwuDRLDefine.KAIWU_MODEL_ID_LIST}'
        checkpoint_id = None
        if os.path.exists(ckpt_id_list):
            try:
                _, last_line = get_first_last_line_from_file(ckpt_id_list)
                m = re.search(r'(?<=model\.ckpt-)\d+', last_line or '')
                if m:
                    checkpoint_id = m.group()
            except Exception:
                pass

        if checkpoint_id is None:
            self._clean_dir(staging)
            return False

        json_file = os.path.join(
            os.path.dirname(tar_gz_path), f'kaiwu_{checkpoint_id}.json'
        )
        for _ in range(3):
            if os.path.exists(json_file):
                break
            time.sleep(1)
        else:
            self._clean_dir(staging)
            return False

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError):
            self._clean_dir(staging)
            return False

        try:
            os.remove(json_file)
        except OSError:
            pass

        self._patch_eval_config(staging, checkpoint_id)
        data['model_file_name'] = f'{base_name}.zip'
        ckpt_hash, model_files = compute_directory_hash(f'{staging}/ckpt')
        data['model_file_hash'] = ckpt_hash
        data['model_file_path'] = ['ckpt/' + fn for fn in model_files]

        output = get_map_content(data)
        sig = generate_private_signature_by_data(output, self._private_key)
        data['signature'] = base64_encode(sig)

        kaiwu_json = (
            f'{staging}/ckpt/'
            f'{KaiwuDRLDefine.KAIWUDRL_MODEL_FILE_JSON_FILE_NAME}.json'
        )
        with open(kaiwu_json, 'w') as f:
            json.dump(data, f)

        zip_path = f'{self._output_dir}/{base_name}.zip'
        python_exec_shell(
            f"cd {staging} && zip -r -q {zip_path} . "
            f"-x 'conf/kaiwudrl/*' -x 'conf/kaiwudrl/'"
        )
        with open(f'{self._output_dir}/{base_name}.zip.json', 'w') as f:
            json.dump(data, f)

        try:
            os.remove(tar_gz_path)
        except OSError:
            pass
        self._clean_dir(staging)

        self.logger.info(
            f'[ModelSigner] Created {base_name}.zip '
            f'(step={checkpoint_id})'
        )
        self._cleanup_old_zips()
        return True

    # ------------------------------------------------------------------
    # Source 2: framework .pkl checkpoints (fallback)
    # ------------------------------------------------------------------

    def _process_pkl_checkpoints(self):
        id_list_path = os.path.join(self._ckpt_dir, 'id_list')
        if not os.path.exists(id_list_path):
            return
        with open(id_list_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            ckpt_id = line.strip()
            if not ckpt_id or ckpt_id.startswith('all'):
                continue
            if ckpt_id in self._processed:
                continue
            pkl_path = os.path.join(self._ckpt_dir, f'{ckpt_id}.pkl')
            if not os.path.exists(pkl_path):
                continue
            if self._sign_pkl_checkpoint(ckpt_id, pkl_path):
                self._processed.add(ckpt_id)

    def _sign_pkl_checkpoint(self, ckpt_id, pkl_path):
        from kaiwudrl.common.utils.common_func import (
            compute_directory_hash,
            get_map_content,
            generate_private_signature_by_data,
            base64_encode,
            python_exec_shell,
            make_single_dir,
        )
        from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
        from kaiwudrl.common.checkpoint.model_file_common import (
            format_reward_config,
        )

        step = int(ckpt_id.split('-')[-1]) if '-' in ckpt_id else 0
        pkl_name = f'{ckpt_id}.pkl'

        # Build staging directory
        staging = self._cos_target_dir
        self._clean_dir(staging)
        make_single_dir(staging)

        # Copy checkpoint file
        ckpt_sub = os.path.join(staging, 'ckpt')
        os.makedirs(ckpt_sub)
        shutil.copy2(pkl_path, os.path.join(ckpt_sub, pkl_name))

        # Copy id_list into ckpt/
        src_id_list = os.path.join(self._ckpt_dir, 'id_list')
        if os.path.exists(src_id_list):
            shutil.copy2(src_id_list, os.path.join(ckpt_sub, 'id_list'))

        # Copy conf/
        for copy_dir in self._copy_dirs:
            python_exec_shell(f'cp -r {copy_dir} {staging}/')

        # Copy agent code directories only (not test files or model artifacts)
        if os.path.isdir(self._code_dir):
            for agent_dir in ('agent_diy', 'agent_ppo'):
                src = os.path.join(self._code_dir, agent_dir)
                if os.path.isdir(src):
                    dst = os.path.join(staging, agent_dir)
                    shutil.copytree(src, dst, dirs_exist_ok=True)

        # Patch eval config
        self._patch_eval_config(staging, str(step))

        # Build metadata
        now = datetime.now(timezone.utc)
        local_now = datetime.now().astimezone()
        time_str = local_now.strftime('%Y_%m_%d_%H_%M_%S')
        version = getattr(CONFIG, 'kaiwu_project_version', '') or os.environ.get(
            'KAIWU_PROJECT_VERSION', '13.0.1'
        )
        base_name = f'{CONFIG.app}-{CONFIG.algo}-{step}-{time_str}-{version}'

        data = {
            'created_at': local_now.isoformat(),
            'train_time': int(time.monotonic() - self._start_time),
            'train_step': step,
            'platform': KaiwuDRLDefine.KAIWUDRL_MODEL_FILE_MAGIC,
            'business': CONFIG.business,
            'user_id': CONFIG.user_id,
            'team_id': CONFIG.team_id,
            'project_code': CONFIG.app,
            'project_version': version,
            'task_id': CONFIG.task_id,
            'algorithm': CONFIG.algo,
        }

        data['model_file_name'] = f'{base_name}.zip'
        ckpt_hash, model_files = compute_directory_hash(ckpt_sub)
        data['model_file_hash'] = ckpt_hash
        data['model_file_path'] = ['ckpt/' + fn for fn in model_files]

        # Sign
        output = get_map_content(data)
        sig = generate_private_signature_by_data(output, self._private_key)
        data['signature'] = base64_encode(sig)

        # Optional reward_config
        reward_config = format_reward_config()
        train_time_env = os.environ.get('KAIWU_TRAIN_TIME', '')
        if reward_config:
            data['reward_config'] = f'{reward_config}_runtime_{train_time_env}'

        # Write kaiwu.json in ckpt/
        with open(os.path.join(ckpt_sub, 'kaiwu.json'), 'w') as f:
            json.dump(data, f)

        # Create zip
        zip_path = f'{self._output_dir}/{base_name}.zip'
        python_exec_shell(
            f"cd {staging} && zip -r -q {zip_path} . "
            f"-x 'conf/kaiwudrl/*' -x 'conf/kaiwudrl/'"
        )

        # Write .zip.json
        with open(f'{self._output_dir}/{base_name}.zip.json', 'w') as f:
            json.dump(data, f)

        self._clean_dir(staging)

        self.logger.info(
            f'[ModelSigner] Created {base_name}.zip '
            f'(step={step}, hash={ckpt_hash[:12]}...)'
        )
        self._cleanup_old_zips()
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clean_dir(self, path):
        if not os.path.exists(path):
            return
        for entry in os.listdir(path):
            entry_path = os.path.join(path, entry)
            try:
                if os.path.isdir(entry_path):
                    shutil.rmtree(entry_path)
                else:
                    os.remove(entry_path)
            except FileNotFoundError:
                continue

    def _patch_eval_config(self, staging, checkpoint_id):
        conf_file = f'{staging}/conf/configure_app.toml'
        if not os.path.exists(conf_file):
            return
        try:
            with open(conf_file, 'r') as f:
                lines = f.readlines()
        except OSError:
            return
        filtered = [
            l for l in lines
            if not l.startswith('eval_model_dir')
            and not l.startswith('eval_model_id')
        ]
        eval_dir = f'/data/projects/{CONFIG.app}/ckpt'
        filtered.append(f'\neval_model_dir = "{eval_dir}"\n')
        filtered.append(f'eval_model_id = "{checkpoint_id}"\n')
        try:
            with open(conf_file, 'w') as f:
                f.writelines(filtered)
        except OSError:
            pass

    def _cleanup_old_zips(self):
        import glob as _glob
        zips = sorted(
            _glob.glob(os.path.join(self._output_dir, '*.zip')),
            key=os.path.getmtime,
        )
        for old_zip in zips[:-50]:
            old_json = old_zip + '.json'
            try:
                os.remove(old_zip)
                if os.path.exists(old_json):
                    os.remove(old_json)
            except OSError:
                pass

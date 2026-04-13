# KaiwuDRL Framework API Inventory

> Date: 2026-04-13
> Source: Container image `kaiwu-pub.tencentcloudcr.com/project/robot_vacuum/edu/win_gpu:13.0.1`
> Path inside container: `/data/projects/robot_vacuum/kaiwudrl/`

## 1. Resume vs Preload Model — Critical Finding

### Preload Model (Framework Built-in)

**Config**: `configure_app.toml` → `preload_model`, `preload_model_dir`, `preload_model_id`

**Call chain**:
```
trainer.py:360          # CONFIG.preload_model == true
  → LoadModelCommon.preload_model_file()
    → check_path_id_valid(dir, id)     # validates dir exists, id >= 0
    → agent_wrapper.preload_model_file(dir, id)    # PyTorch version
      → agent.load_model(path=dir, id=id, framework=True)  # calls YOUR load_model
      → self.train_count = preload_model_id
      → self.preload_model_train_count = preload_model_id  # fixes sample stat
```

**What preload does that current resume does NOT**:
1. **Fixes training count** — sets `train_count` and `preload_model_train_count` so that sample consumption ratio (`sample_production_and_consumption_ratio`) is correct
2. **Loads on both learner AND aisrv** — aisrv predictor_local.py also calls `preload_model_file`, so both sides start with the same weights (no stale random-initialization period)
3. **Hard fails on error** — if preload fails, `learner_process_stop(error_code=-1)` exits immediately (safer than silently training with wrong weights)

**Resume gap analysis**:
| Aspect | preload_model | current resume |
|--------|--------------|----------------|
| Load path | `agent.load_model(path, id)` via framework | `torch.load` + `load_state_dict` directly |
| File naming | Must be `model.ckpt-{int_id}.pkl` | Arbitrary (`model.ckpt-resume.pkl`) |
| Train count fix | Yes (auto) | No (causes stat drift) |
| aisrv sync | Yes (both learner+aisrv) | No (learner only) |
| Failure handling | Hard exit | Print warning, continue |

**Recommendation**: Wrap resume via preload by:
1. Saving resume snapshots as `model.ckpt-{episode_cnt}.pkl` in `preload_model_dir`
2. Setting `preload_model = true`, `preload_model_dir`, `preload_model_id` in `configure_app.toml`
3. Removing the custom resume `torch.load` logic from `agent.__init__`

---

## 2. BaseAgent Interface (`kaiwudrl/interface/`)

### `none_agent.py` — Abstract base class

Required methods (must implement):
- `learn(list_sample_data) -> dict` — training
- `predict(list_obs_data) -> list` — inference
- `exploit(list_obs_data) -> list` — exploitation
- `save_model(path, id, **kwargs)` — save checkpoint
- `load_model(path, id, **kwargs)` — load checkpoint

Optional methods:
- `reset(list_obs_data)` — per-episode reset
- `init_config(list_obs_data)` — config initialization
- `get_training_metrics(**kwargs)` — custom metrics
- `observation_process(env_obs)` — obs preprocessing
- `action_process(act_data)` — action postprocessing
- `send_sample_data(list_sample_data, **kwargs)` — send data to learner (aisrv-side)

Auto-dispatch config (add new obs methods here):
```python
_OBS_DISPATCH_METHODS = [
    {"method_name": "predict", "is_abstract": True},
    {"method_name": "exploit", "is_abstract": True},
    {"method_name": "reset", "is_abstract": False},
    {"method_name": "init_config", "is_abstract": False},
]
```

### `remote_agent.py` — Cluster mode (current mode)

Key capabilities:
- `send_sample_data()` — aisrv sends samples to learner via reverb (auto-serialization via `SampleData.FIELD_DIMS`)
- `get_training_metrics()` — aisrv queries learner for metrics via zmq
- `load_opponent_agent(path, id)` — load opponent model for self-play
- Method interceptors: business methods saved as `_business_xxx`, replaced with framework dispatchers

### `agent_context.py` — AgentContext dataclass

```python
@dataclass
class AgentContext:
    agent_id: int
    done: bool
    policy_conf: dict
    policy: dict
    main_id: str
    pred_output: dict
    expr_processor: dict
    start_time: float
    reward: object
```

---

## 3. StandardAgentWrapperPytorch (`common/algorithms/standard_agent_wrapper_pytorch.py`)

Framework layer between trainer and agent. Key methods:

### Training lifecycle
- `train(current_sync_model_version_from_learner)` — full training loop (data fetch + learn + after_train)
- `train_local(data, extra_tensors)` — single train step (no framework logic)
- `before_train()` — hook before training
- `after_train()` — increments `train_count`, saves model every `dump_model_freq`, returns `(has_model_file_changed, model_file_id)`

### Model management
- `save_param_by_source(path, id, source)` — saves model (framework/user triggered)
- `load_model_by_source(path, id, source)` — loads model (handles FRAMEWORK vs USER source)
- `preload_model_file(preload_model_dir, preload_model_id)` — preload logic (calls `agent.load_model`)
- `add_file_to_queue()` — manages max model file count (deletes oldest)

### Statistics
- `train_stat` property → `(train_count, preload_model_train_count)`
- `train_count` — total training steps
- `preload_model_train_count` — steps from preload (used for ratio calculation)
- `data_fetch_cost_time` — data fetch timing
- `real_train_cost_time` — actual training timing

### Key: `after_train()` controls model saving frequency
```python
def after_train(self):
    self.train_count += 1
    if self.train_count % CONFIG.dump_model_freq == 0:
        self.save_param_by_source(path=..., id=self.train_count, source=FRAMEWORK)
        self.add_file_to_queue()
        update_id_list(self.train_count, framework=True)
        has_model_file_changed = True
    return has_model_file_changed, self.train_count
```

---

## 4. Trainer (`server/learner/trainer.py`)

### Startup sequence (relevant parts):
1. `start_learner_process_by_type()` — COS model download (disabled in our setup)
2. `model_file_saver` — fork model save process (disabled when `push_to_cos=false`)
3. **`preload_model_file()`** — if `CONFIG.preload_model`, loads initial model
4. ZMQ server bind for aisrv communication
5. First model save (PyTorch) — `save_param_by_source(id=0)`, `update_id_list`
6. If preload: `update_id_list(preload_model_id)` else `update_id_list(0)`
7. `clear_user_ckpt_dir()` — clean user checkpoint dir (patched to skip for symlinks)

### Training loop:
- `train_detail()` → calls `agent_wrapper.train(strategy.get_current_sync_model_version_from_learner())`
- Returns `(train_success, app_monitor_data, has_model_file_changed, model_file_id)`
- On-policy: `process_policy_specific(model_file_id)` triggers model sync to aisrv/actor

---

## 5. On-Policy Strategy (`server/learner/on_policy_strategy.py`)

### Flow after successful training:
1. `learner_on_policy_process(True)`:
   - `replay_buffer.reset()` — clear samples (on-policy: use once)
   - `learner_push_model_to_modelpool()` — push checkpoint to model pool (with retry)
   - Send `MODEL_VERSION_CHANGE_REQUEST` to all aisrv/actor via ZMQ
   - Wait for `MODEL_VERSION_CHANGE_RESPONSE` from all
   - Heartbeat keep-alive with actor processes

### Sample filtering (on-policy):
- Samples carry `model_version` in last column
- Only samples with `model_version == current_sync_model_version_from_learner` are used
- Accumulated until `batch_size` reached, then train

---

## 6. Useful Interfaces Not Currently Used

### `agent.get_training_metrics(**kwargs)`
- aisrv calls this via zmq to query learner for custom metrics
- Returns dict from learner agent
- **Useful for**: Exposing custom resume/training state to aisrv

### `agent.load_opponent_agent(path, id)`
- Loads a separate model for self-play/opponent evaluation
- Only works in `REMOTE_AISRV_PREDICT` or `REMOTE_ACTOR_PREDICT` mode
- **Useful for**: Loading a frozen "best model" as opponent for evaluation

### `MultiModelManager` (`server/common/multi_model_common.py`)
- Manages multiple model versions simultaneously
- Configured via `CONFIG.init_model_file_list` and `CONFIG.init_model_file_id_list`
- Each model loaded from separate code package zip
- **Useful for**: A/B testing model versions, loading historical opponents

### `ReplayBufferWrapper`
- `reset(step, tf_sess)` — clear replay buffer (called in on-policy)
- `get_current_size()` — current buffer size
- `get_recv_speed()` — sample ingestion speed
- `get_insert_stats()` — insertion statistics
- **Useful for**: Monitoring replay health, custom reset logic

### `ModelFileSync` (`common/checkpoint/model_file_sync.py`)
- `push_checkpoint_to_model_pool()` — push to model pool for on-policy sync
- `pull_checkpoint_from_model_pool_by_on_policy()` — pull from model pool
- **Useful for**: Custom model distribution in multi-agent setups

### `SampleProcessor` interface (`interface/sample_processor.py`)
- `should_train()` — whether to train
- `proc_exprs()` — process experiences, returns `(train_data, train_frame_cnt, drop_frame_cnt)`
- **Useful for**: Custom sample filtering/preprocessing logic

### `Policy` interface (`interface/policy.py`)
- `send_pred_data()` / `get_pred_result()` — prediction data flow
- `need_train()` — training trigger
- `send_train_data()` / `send_train_data_to_sample_server()` — training data flow
- `stop()` — cleanup
- **Useful for**: Custom aisrv-side policy control

### `AgentContext`
- Passed to policy methods, contains `agent_id`, `done`, `pred_output`, etc.
- **Useful for**: Accessing per-agent state in policy callbacks

---

## 7. Model Path Layout (Framework Conventions)

```
/data/ckpt/{app}_{algo}/          # Framework checkpoint dir (CONFIG.ckpt_dir)
  model.ckpt-0.pkl
  model.ckpt-{train_count}.pkl
  checkpoint                       # TensorFlow checkpoint file
  id_list                          # Track saved model IDs

/data/user_ckpt/                   # User checkpoint dir (CONFIG.user_ckpt_dir)
  model.ckpt-{id}.pkl             # Saved by agent.save_model with source=USER

/data/restore/{app}_{algo}/       # Restore dir (CONFIG.restore_dir)
  model.ckpt-{id}.pkl             # Used by aisrv/actor for loading

/data/model_pools/                # Model pool for on-policy sync
  {pid}/
    model_{model_id}_pid_{pid}/
      ckpt/
        model.ckpt-{id}.pkl
      agent_{algo}/
        agent.py                   # Historical model's code
```

---

## 8. CONFIG Keys of Interest (from configure_app.toml)

| Key | Default | Description |
|-----|---------|-------------|
| `preload_model` | `false` | Enable preload model |
| `preload_model_dir` | `"{agent_name}/ckpt"` | Preload model directory |
| `preload_model_id` | `1000` | Preload model step ID |
| `dump_model_freq` | (varies) | Save model every N train steps |
| `train_batch_size` | (varies) | Training batch size |
| `max_save_model_file_count` | (varies) | Max checkpoint files to keep |
| `push_to_cos` | `true` | Enable COS upload (disabled in our setup) |
| `algorithm_on_policy_or_off_policy` | `"on_policy"` | On/off policy mode |
| `on_policy_error_max_retry_rounds` | (varies) | Max retry for on-policy sync |
| `on_policy_error_retry_count_when_modelpool` | (varies) | Max retry for modelpool push/pull |

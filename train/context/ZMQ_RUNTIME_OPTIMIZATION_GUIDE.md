# ZMQ 与运行效率优化操作说明

本文档面向后续接手本仓库的 AI 或工程师，说明如何在 `win_YJY` 分支上迁移/维护 ZMQ 样本链路和其他低侵入运行效率优化。目标是让训练能稳定启动、减少样本链路开销、降低无效模型同步/保存频率，并避免重复踩已有坑。

## 当前状态

- 当前分支：`win_YJY`。
- 训练模式：Docker Compose 分布式训练，入口主要在 `train/.docker-compose.yaml` 和 `train/.env`。
- 样本链路：已使用 `KAIWU_EXPERIMENT_REPLAY_BUFFER_TYPE=zmq`。
- 单机资源假设：单 GPU，AISRV 推理仍在 CPU，learner 训练使用 GPU。
- 当前主要瓶颈：AISRV/环境侧 CPU，而不是 learner 的 data fetch；learner 日志里 `data_fetch` 通常约 1-2 ms，`real_train` 约 18-30 ms。
- 已知重要约束：不要把监控配置 `server_req_base_url` 改成容器名或 `0.0.0.0`，正确值应保持 `http://127.0.0.1:${MONITOR_TRPC_PORT}`，详见 `train/context/MONITOR_CONFIG_NOTE.md`。

## 关键文件

- `train/.env`：运行时参数入口，控制 AISRV 数量、并行环境、ZMQ/reverb、batch、模型保存频率等。
- `train/.docker-compose.yaml`：容器启动脚本，负责把 `.env` 参数注入官方运行环境，并在容器内 patch 生成配置。
- `code/conf/configure_app.toml`：项目侧默认训练配置，需要和 `.env` 中关键参数保持一致。
- `code/agent_ppo/utils/zmq_patch.py`：ZMQ runtime patch 工具，用于修补官方 Kaiwu 运行时里 ZMQ + multiprocessing 的兼容问题。
- `code/agent_ppo/conf/conf.py`：模型/训练端 CPU 线程、batch tensor、optimizer 等轻量优化开关。
- `code/agent_ppo/algorithm/algorithm.py`、`code/agent_ppo/feature/preprocessor.py`、`code/agent_ppo/workflow/train_workflow.py`：算法本体，不应为了 ZMQ 性能优化随意改行为逻辑。

## 推荐操作步骤

1. 确认分支和工作区

```powershell
git branch --show-current
git status --short
```

要求分支应为 `win_YJY`。如果工作区已有其他人的未提交改动，只能追加兼容性改动，不要回滚或覆盖。

2. 设置 `.env` 中的 ZMQ 和吞吐参数

推荐基线：

```env
KAIWU_AISRV_NUM=2
KAIWU_GAMECORE_NUM=8
KAIWU_PARALLEL_ENV_PER_AISRV=4
KAIWU_EXPERIMENT_REPLAY_BUFFER_TYPE=zmq
KAIWU_PYTORCH_READ_DATA_FROM_REVERB_TYPE=1
KAIWU_EXPERIMENT_TRAIN_BATCH_SIZE=4096
KAIWU_EXPERIMENT_SEND_SAMPLE_SIZE=4096
KAIWU_EXPERIMENT_REPLAY_BUFFER_CACHE_MULTIPLIER=4
KAIWU_EXPERIMENT_REPLAY_BUFFER_CAPACITY=10000
KAIWU_EXPERIMENT_PRELOAD_RATIO=0.1
KAIWU_EXPERIMENT_DUMP_MODEL_FREQ=5000
KAIWU_EXPERIMENT_MODEL_FILE_SYNC_PER_MINUTES=3
KAIWU_EXPERIMENT_PREDICT_BATCH_SIZE=64
```

说明：

- `KAIWU_EXPERIMENT_REPLAY_BUFFER_TYPE=zmq` 是启用 ZMQ 的核心开关。
- `TRAIN_BATCH_SIZE=4096` 和 `SEND_SAMPLE_SIZE=4096` 保持一致，避免 learner 和 AISRV 样本粒度不匹配。
- `DUMP_MODEL_FREQ=5000` 和 `MODEL_FILE_SYNC_PER_MINUTES=3` 用于降低频繁保存/同步模型带来的 IO 和 CPU 抖动。
- 当前平台只有单 GPU，不要引入多 GPU/DDP 改造。

3. 同步 `code/conf/configure_app.toml`

至少确认以下值和 `.env` 一致：

```toml
replay_buffer_capacity = 10000
preload_ratio = 0.1
train_batch_size = 4096
dump_model_freq = 5000
model_file_sync_per_minutes = 3
```

注意：官方启动脚本会在容器内生成或覆盖部分配置，所以只改 `configure_app.toml` 不够，必须同时保证 compose 的 post-init patch 能把 `.env` 参数写入容器内配置。

4. 在 `train/.docker-compose.yaml` 中注入参数

compose 的公共环境区应传入这些变量：

```yaml
KAIWU_EXPERIMENT_REPLAY_BUFFER_TYPE: ${KAIWU_EXPERIMENT_REPLAY_BUFFER_TYPE:-reverb}
KAIWU_EXPERIMENT_TRAIN_BATCH_SIZE: ${KAIWU_EXPERIMENT_TRAIN_BATCH_SIZE:-4096}
KAIWU_EXPERIMENT_SEND_SAMPLE_SIZE: ${KAIWU_EXPERIMENT_SEND_SAMPLE_SIZE:-4096}
KAIWU_EXPERIMENT_DUMP_MODEL_FREQ: ${KAIWU_EXPERIMENT_DUMP_MODEL_FREQ:-5000}
KAIWU_EXPERIMENT_MODEL_FILE_SYNC_PER_MINUTES: ${KAIWU_EXPERIMENT_MODEL_FILE_SYNC_PER_MINUTES:-3}
KAIWU_EXPERIMENT_PREDICT_BATCH_SIZE: ${KAIWU_EXPERIMENT_PREDICT_BATCH_SIZE:-64}
KAIWU_EXPERIMENT_REPLAY_BUFFER_CACHE_MULTIPLIER: ${KAIWU_EXPERIMENT_REPLAY_BUFFER_CACHE_MULTIPLIER:-4}
KAIWU_EXPERIMENT_REPLAY_BUFFER_CAPACITY: ${KAIWU_EXPERIMENT_REPLAY_BUFFER_CAPACITY:-10000}
```

learner 和 AISRV 启动前都要 patch `/root/tools/common.sh`：

```bash
sed -i "s|sh tools/change_sample_server.sh reverb|sh tools/change_sample_server.sh ${KAIWU_EXPERIMENT_REPLAY_BUFFER_TYPE}|g" /root/tools/common.sh
```

这样官方 `start_train_client.sh` 才会按 ZMQ 启动 sample server。只改项目配置但不 patch `common.sh`，很容易表面配置是 ZMQ，实际仍按 reverb 启动。

5. 保留 ZMQ runtime patch

`code/agent_ppo/utils/zmq_patch.py` 的职责是运行时修补官方 Kaiwu 文件，不直接修改仓库中的官方依赖。核心逻辑：

- 只在 `replay_buffer_type == "zmq"` 时生效。
- 对 `kaiwudrl/common/replay_buffer/zmq_replay_buffer.py` 和 `kaiwudrl/server/learner/learner_server.py` 注入 spawn-context prelude。
- 避免 Linux 默认 `fork` 在父进程提前初始化 CUDA 后导致 ZMQ/multiprocessing 异常。

compose 里 learner 启动阶段应调用：

```python
from agent_ppo.utils.zmq_patch import patch_zmq_runtime_files
patched = patch_zmq_runtime_files(root, replay_buffer_type)
```

已有实现还会把 mem_buffer 初始化设备强制为 CPU：

```python
self.device = "cpu"
```

这个修改是有意的：ZMQ 样本缓冲不应在父进程里初始化 CUDA，真正训练张量会在 trainer/learner 里再转到 GPU。

6. 保留 post-init patch

官方容器启动后会生成：

- `/data/projects/robot_vacuum/conf/configure_app.toml`
- `/data/projects/robot_vacuum/kaiwudrl/conf/kaiwudrl/configure.toml`
- `/data/projects/robot_vacuum/kaiwudrl/conf/kaiwudrl/learner.toml`

因此 compose 中需要在 `start_train_client.sh` 启动后再执行 post-init patch，把 `.env` 参数写回这些生成配置。不要只依赖构建前的静态文件。

## 避坑清单

- 不要修改监控前端请求地址：`server_req_base_url` 必须保留 `http://127.0.0.1:${MONITOR_TRPC_PORT}`。改成容器服务名、宿主机 IP 或 `0.0.0.0` 会导致终端面板 `Fail to fetch`。
- 不要让 mem_buffer 在父进程中初始化 CUDA。ZMQ + multiprocessing 默认 fork 很容易触发 CUDA fork 问题，应保持 mem_buffer CPU-only。
- 不要引入多 GPU 逻辑。当前平台只有单 GPU，多 GPU 改造会增加启动和调试风险，收益为零。
- 不要单独提高 AISRV/gamecore 数量。当前 CPU 已接近瓶颈，盲目增加 `KAIWU_GAMECORE_NUM` 或 `KAIWU_AISRV_NUM` 可能只会提高上下文切换和排队。
- 不要把 `sample_production_and_consumption_ratio` 误解成消费/生产比。当前代码和日志语义应按“生产/消费比”理解；该值很高说明 learner 反复消费/复用缓冲样本，不能直接当作环境生产速度。
- 不要频繁保存模型。`dump_model_freq=500` 会造成过多模型写入和同步，应优先使用 `5000` 作为当前低侵入基线。
- 不要只看 stdout 的 `docker compose logs`。业务日志主要在容器内 `/data/projects/robot_vacuum/log/*.log`，本地同步在 `train/log/`。
- 不要把算法失败归因于 ZMQ。ZMQ 优化主要解决样本链路和启动稳定性；电量失败、碰撞、CPS 平台期仍要从 planner/reward/feature 处理。

## 测试与验收

1. 静态检查

```powershell
python -m py_compile code\agent_ppo\utils\zmq_patch.py
docker compose --env-file train/.env -f train/.docker-compose.yaml -p kaiwu-train config
```

验收点：

- compose config 能正常生成。
- `KAIWU_EXPERIMENT_REPLAY_BUFFER_TYPE` 显示为 `zmq`。
- batch、capacity、dump/model sync 变量显示为预期值。

2. 启动训练

```powershell
docker compose --env-file train/.env -f train/.docker-compose.yaml -p kaiwu-train up -d
```

如果需要重建容器：

```powershell
docker compose --env-file train/.env -f train/.docker-compose.yaml -p kaiwu-train up -d --force-recreate
```

3. 检查容器内配置是否真正生效

```powershell
docker exec kaiwu-train-learner-1 bash -lc "grep -E 'train_batch_size|dump_model_freq|model_file_sync_per_minutes|replay_buffer_capacity|preload_ratio' /data/projects/robot_vacuum/conf/configure_app.toml"
docker exec kaiwu-train-learner-1 bash -lc "grep -E 'dump_model_freq|model_file_sync_per_minutes' /data/projects/robot_vacuum/kaiwudrl/conf/kaiwudrl/configure.toml"
```

预期：

- `train_batch_size = 4096`
- `replay_buffer_capacity = 10000`
- `preload_ratio = 0.1`
- `dump_model_freq = 5000`
- `model_file_sync_per_minutes = 3`

4. 检查 ZMQ patch 日志

```powershell
docker logs kaiwu-train-learner-1 --tail 300
docker logs kaiwu-train-aisrv-1 --tail 300
```

或读取业务日志：

```powershell
docker exec kaiwu-train-learner-1 bash -lc "grep -E 'learner-runtime|learner-zmq|learner-sed|post_init' /data/projects/robot_vacuum/log/learner.log | tail -80"
docker exec kaiwu-train-aisrv-1 bash -lc "grep -E 'aisrv-runtime|aisrv-sed|post_init' /data/projects/robot_vacuum/log/aisrv.log | tail -80"
```

验收点：

- 能看到 `replay_buffer_type=zmq`。
- 能看到 `patched spawn context` 或已 patch 的等价日志。
- 没有 `RuntimeError: Cannot re-initialize CUDA in forked subprocess`。
- 没有 sample server 启动失败、ZMQ bind/connect 失败。

5. 检查 learner 是否使用 GPU

```powershell
docker exec kaiwu-train-learner-1 bash -lc "grep -E 'cuda:0|Loaded.*cuda|device' /data/projects/robot_vacuum/log/learner.log | tail -50"
nvidia-smi
```

预期：

- learner 加载模型在 `cuda:0`。
- AISRV 仍可能显示 `cpu`，这是当前低侵入方案的预期状态。

6. 检查训练是否真实推进

```powershell
docker exec kaiwu-train-learner-1 bash -lc "grep -E 'global step is|sample_production_and_consumption_ratio|algorithm.learn' /data/projects/robot_vacuum/log/learner.log | tail -80"
docker exec kaiwu-train-aisrv-1 bash -lc "grep -E 'GAMEOVER|training_metrics|DEATH_TRAJ' /data/projects/robot_vacuum/log/aisrv.log | tail -120"
```

验收点：

- `global step` 持续增长。
- `data_fetch` 通常应在毫秒级，若长期很高才说明样本链路仍是瓶颈。
- `GAMEOVER` 持续出现，说明环境在生产样本。
- 如果失败仍集中在 `reason:battery` 且 `mode=charge slack<0`，这是算法返航问题，不是 ZMQ 问题。

7. 粗略吞吐评估

用 learner 日志看训练吞吐：

```powershell
docker exec kaiwu-train-learner-1 bash -lc "grep 'global step is' /data/projects/robot_vacuum/log/learner.log | tail -20"
```

关注：

- `train once cost time`
- `data_fetch`
- `real_train`
- `sample_production_and_consumption_ratio`

当前基线下，`data_fetch` 约 1-2 ms 属于正常，瓶颈更多在 AISRV/环境 CPU。

## 回滚方案

如果 ZMQ 运行异常，需要临时回滚到 reverb：

1. 修改 `train/.env`

```env
KAIWU_EXPERIMENT_REPLAY_BUFFER_TYPE=reverb
```

2. 保留 `zmq_patch.py` 不动

该文件只在 `replay_buffer_type == "zmq"` 时生效，回滚到 reverb 后不会 patch ZMQ runtime。

3. 重建或重启容器

```powershell
docker compose --env-file train/.env -f train/.docker-compose.yaml -p kaiwu-train up -d --force-recreate
```

4. 验证日志中 sample server 回到 reverb

```powershell
docker logs kaiwu-train-learner-1 --tail 300
docker logs kaiwu-train-aisrv-1 --tail 300
```

## 后续优化边界

ZMQ 和运行时优化只能提升样本传输/训练链路稳定性，不会自动解决当前算法 plateau。若目标是提升胜率和 CPS，应另行处理：

- 首充电桩显式发现阶段。
- 更早的 `dist_to_charger + safety_margin` 返航硬约束。
- 电量失败样本加权或 reward 加强。
- 充电模式下 planner 与 PPO 的控制权重新平衡。
- NPC 下一步风险预测。

这些属于算法优化，不应混在 ZMQ runtime patch 中一起做，避免难以定位回归来源。

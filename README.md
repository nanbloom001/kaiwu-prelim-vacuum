# Robot Vacuum Training Notes

当前这套 `CPU + ZMQ` 训练不是手工进容器启动，而是通过 `docker compose` 启动。

## 正确启动方式

在仓库根目录执行：

```powershell
docker compose --profile distributed --env-file train/.env -f train/.docker-compose.yaml -p kaiwu-train up -d --force-recreate
```

说明：

- 必须带 `--profile distributed`，否则 `learner` 和 `aisrv` 不会按分布式配置重建。
- 必须带 `--force-recreate`，否则旧容器可能继续沿用旧的 `reverb` 启动命令，导致这次的 `zmq` 改动不生效。
- 不需要手工进入容器执行 `/workspace/code/agent_ppo/utils/*.sh`。
- `docker compose up` 会自动调用：
  - `code/agent_ppo/utils/learner_zmq_entrypoint.sh`
  - `code/agent_ppo/utils/aisrv_zmq_entrypoint.sh`

## 不建议的方式

不要只执行普通重启：

```powershell
docker restart kaiwu-train-learner-1
docker restart kaiwu-train-aisrv-1
docker restart kaiwu-train-aisrv-2
```

也不要只执行不带 `--profile distributed` 的：

```powershell
docker compose --env-file train/.env -f train/.docker-compose.yaml -p kaiwu-train up -d --force-recreate
```

这两种方式都可能让容器继续使用旧配置，表现为日志里仍然是 `reverb`。

## 启动后快速检查

看 learner 日志是否出现这些关键字：

```powershell
docker logs --tail 200 kaiwu-train-learner-1
```

预期应看到：

- `change kaiwudrl/conf/kaiwudrl/learner.toml zmq success`
- `change kaiwudrl/conf/kaiwudrl/configure.toml zmq success`
- `learner train replay_buff, use zmq`
- `ZmqReplayBuffer`

看 aisrv 日志：

```powershell
docker logs --tail 200 kaiwu-train-aisrv-1
docker logs --tail 200 kaiwu-train-aisrv-2
```

预期应看到：

- `aisrv-runtime`
- `change kaiwudrl/conf/kaiwudrl/learner.toml zmq success`
- `kaiwu_env_proxy use zmq`

## 当前已验证结论

- 当前可用链路是 `CPU + ZMQ`
- `learner` 与 `aisrv` 已能按 `zmq` 正常启动
- 稳定训练阶段相对之前 `CPU + reverb` 有明显提速

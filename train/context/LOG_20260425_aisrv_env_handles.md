# AISRV benchmark env handle investigation - 2026-04-25

## Question

Why does benchmark business code still receive only one environment per AISRV even when `KAIWU_PARALLEL_ENV_PER_AISRV`
and gamecore count are greater than 1?

## Local evidence

- The 2-AISRV/5-GC-per-AISRV run created 10 gamecores and each AISRV process saw 5 gamecore addresses.
- `train/log/aisrv/ai_server_container*_log_2026-04-25-23.log` shows each AISRV server starting multiple
  `AiSrvHandle` instances for gamecore addresses.
- The user workflow still entered only once per AISRV:
  - `kaiwu_rl_helper start agent 0`
  - `workflow(envs, agents, ...)`
  - benchmark manifest recorded `env_count=1`, `agent_count=1`
- The benchmark downgrade was therefore expected:
  - `workers_per_aisrv_requested=5`
  - `workers_per_aisrv_effective=1`
  - `worker_mode=serial`
  - `worker_downgrade_reason=requested=5, env_count=1, agent_count=1`

## Config path check

`KAIWU_PARALLEL_ENV_PER_AISRV` is currently passed as a container environment variable, but the active compose runtime
patch only writes predictor batch settings into Kaiwu TOML files:

- `predict_batch_size`
- `proxy_batch_size`

No active code path was found that patches a framework config key to make `workflow(envs, agents, ...)` receive more
than one env/agent handle.

## Linux branch comparison

`origin/linux-LTSPPO-charge-constraint` has `code/agent_ppo/eval/benchmark_parallel.py`.

That implementation also does not force multi-env handles inside one workflow call. It calculates:

```python
effective_slots = min(requested_slots, len(envs), len(agents))
```

The Linux handoff explicitly records the same observation:

- `available_env_handles = 1`
- `available_agent_handles = 1`
- `effective_envs_per_worker = 1`
- real speedup comes from outer multi-AISRV concurrency, not explicit multi-env handles in Python workflow.

## Conclusion

The current bottleneck is not the benchmark Python thread pool. The Kaiwu framework invocation exposes only one
business env/agent handle per `kaiwu_rl_helper` workflow. Extra gamecores are connected at the AISRV server layer, but
they are not surfaced as additional independent `envs[]` entries to PPO benchmark code.

The practical benchmark acceleration path is to run more AISRV workflow workers and distribute episode tasks across
them, preferably using the Linux branch's dynamic task queue pattern. Per-AISRV `workers_per_aisrv > 1` should be
treated as best-effort only and expected to downgrade to 1 unless framework startup/config is changed deeper than the
project-level compose patches.

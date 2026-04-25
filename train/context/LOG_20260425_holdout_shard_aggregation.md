# 2026-04-25 Holdout shard aggregation hardening

## Root cause

`train/run_holdout_benchmark.py` already prepared deterministic shard assignments, but the sharded non-dry path still waited for a single `code/holdout_result.json` artifact. That artifact is only written by serial worker mode. In sharded mode each AISRV writes only:

- `code/holdout_shards/results/shard_<index>.json`
- `code/holdout_shards/done/.done_shard_<index>`

So the runner could not complete real sharded execution correctly and had no fail-closed coverage validation.

## Changes

- Added runner-local shard helpers in `train/run_holdout_benchmark.py` for:
  - canonical JSON writes,
  - expected shard path discovery,
  - waiting for all done markers,
  - strict shard JSON loading,
  - shared metric aggregation copied from `code/agent_ppo/eval/holdout_benchmark.py` without importing torch runtime,
  - canonical shard-assignment metadata emission.
- Replaced sharded docker wait/copy behavior so the runner now:
  1. waits for every expected done marker,
  2. reads every expected shard result file,
  3. validates shard invariants and exact `(map_id, ep_idx)` coverage,
  4. sorts episodes deterministically by `(map_id, ep_idx)`,
  5. writes canonical aggregate output atomically to `code/holdout_result.json`, then copies it to the requested output path.
- Added hidden no-Docker verification interface:
  - `--aggregate-shards-only`
  - `--shard-root <path>`

## Fail-closed validation now enforced

- missing done marker
- missing shard result file
- invalid JSON / wrong JSON shape
- wrong execution metadata (`mode`, `shard_index`, `shard_count`)
- mismatched invariant shard fields (`schema_version`, `checkpoint`, `contract`, `round_def`, `maps`, `episodes_per_map`)
- missing episode key fields (`map_id`, `ep_idx`)
- duplicate `(map_id, ep_idx)`
- unexpected or missing episode coverage
- per-map counts not equal to requested `episodes_per_map`

## Aggregate schema notes

The canonical aggregate preserves serial-compatible top-level fields:

- `schema_version`
- `timestamp`
- `checkpoint`
- `elapsed_seconds`
- `contract`
- `round_def`
- `maps`
- `episodes_per_map`
- `overall`
- `per_map`
- `episodes`
- `execution`

Sharded-only metadata is emitted under `execution`:

- `mode = "sharded"`
- `expected_shards`
- `completed_shards`
- `shard_assignments`
- `source_files`
- `errors = []`

## Verification

- `python -m py_compile train/run_holdout_benchmark.py train/analyze_holdout_benchmark.py` ✅
- happy fake-shard aggregation (20 unique episodes) ✅
  - output: `.sisyphus/evidence/task-3-aggregate.json`
- missing shard result fails closed ✅
  - output: `.sisyphus/evidence/task-3-missing.txt`
- duplicate `(map_id, ep_idx)` fails closed ✅
  - output: `.sisyphus/evidence/task-3-duplicate.txt`

## Scope kept unchanged

- No PPO/reward/planner/model logic changed.
- No Docker or real holdout benchmark was started.
- `train/analyze_holdout_benchmark.py` was left unchanged because the aggregate remained analyzer-compatible.

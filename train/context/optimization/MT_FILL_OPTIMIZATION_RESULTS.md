# Multi-Thread Fill Optimization Results

Date: 2026-04-13

## Background

Prior optimizations achieved 372 step/min with 3 fill workers + 1 coordinator (target_fill=0.10, cache_multiplier=16, batch_size=2048). This session tested further tuning parameters.

## Committed Configuration (working baseline)

All changes below are in the committed `linux` branch:
- `target_fill=0.10` (down from 0.75)
- `KAIWU_EXPERIMENT_FILL_THREADS=3` (3 workers + 1 coordinator)
- `KAIWU_EXPERIMENT_REPLAY_BUFFER_CACHE_MULTIPLIER=16`
- Coordinator sleep = 0.3s
- `reverb_num_workers_per_iterator` via replace_toml_key

## R1 Baseline (3 workers, batch=2048, coord=0.3s)

Container started with committed config. Results from fresh Reverb (no cached data):

| Time  | Step | cost_ms | fetch_ms | train_ms | ratio | buffer_util    |
|-------|------|---------|----------|----------|-------|----------------|
| 02:17 | 63   | 174     | 122      | 52       | 10.7  | 30142/32768    |
| 02:18 | 428  | 98      | 63       | 35       | 30.5  | 22528/32768    |
| 02:19 | 775  | 98      | 60       | 38       | 34.7  | 26732/32768    |
| 02:20 | 1195 | 86      | **47**   | 39       | 39.6  | 0/32768        |
| 02:21 | 1631 | 331     | **286**  | 45       | 42.7  | 7718/32768     |
| 02:22 | 2065 | 184     | **135**  | 49       | 44.7  | 6144/32768     |

**Steady-state step/min**: ~420 (02:18-02:20)
**Mean fetch_ms**: ~57ms (steady state), with spikes to 286ms when buffer drains
**Key observation**: Buffer drained to 0 at 02:20, causing 331ms spike at 02:21. The coordinator's 0.3s sleep means up to 300ms latency before swap can occur after buffer drains.

## Planned but Not Tested

### R2: Coordinator interval 0.3s -> 0.1s
- **Change**: `time.sleep(0.3)` -> `time.sleep(0.1)` in docker-compose.yaml line 457
- **Expected**: Reduce fetch spike from 286ms to <150ms, lower mean_fetch
- **Risk**: Low (slightly more CPU wake-ups)
- **Status**: Patch applied but container recreation interrupted by user

### R3: batch_size 2048 -> 1024
- **Change**: `KAIWU_EXPERIMENT_TRAIN_BATCH_SIZE=1024`
- **Expected**: ~2x step/min (fetch halved), but may affect convergence
- **Risk**: Medium (training quality impact)
- **Status**: Not started

### R4: fill_threads 3 -> 2
- **Change**: `KAIWU_EXPERIMENT_FILL_THREADS=2`
- **Expected**: Reduce GIL contention, modest improvement
- **Risk**: Low
- **Status**: Not started

### R5: Combine best params
- Combine R2+R3 or R2+R4 based on individual results
- **Status**: Not started

## Optimization History Summary

| Stage | Config | step/min | fetch_ms | Notes |
|-------|--------|----------|----------|-------|
| Original baseline | cache=4, single thread | 126 | 707 | Bottleneck: target_fill=0.75 |
| After target_fill fix | cache=16, target_fill=0.10 | 194 | 532 | Large improvement |
| After MT fill | 3 workers, cache=16 | 372 | 282 | ~3x over original |
| R1 baseline (today) | Same as MT fill, fresh start | ~420 | 57 (spikes 286) | Reverb had cached data initially |

## Key Insights

1. **Buffer drain is the main cause of fetch spikes**: When buffer hits 0, coordinator needs up to 0.3s + fill time to recover
2. **Coordinator sleep is a tunable knob**: Reducing from 0.3s to 0.1s should directly reduce spike recovery time
3. **batch_size reduction**: Halving from 2048 to 1024 is the highest-impact untested change
4. **Worker count**: 3 workers may be slightly over-provisioned; 2 may reduce contention
5. **GPU utilization**: Only ~15% at current throughput - massive headroom for faster data pipeline

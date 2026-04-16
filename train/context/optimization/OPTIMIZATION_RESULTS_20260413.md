# KaiwuDRL Training Optimization Results

**Date**: 2026-04-13  
**System**: 4×A10 GPU (23GB each), 118 CPU cores, 503GB RAM  
**Framework**: KaiwuDRL v13.0.1, PPO, reverb replay buffer (V1 dataset)  
**Model**: CNN+MLP (69D input → 256 → 128 → 8 actions)

## Baseline Configuration

| Parameter | Value |
|-----------|-------|
| gamecores | 8 |
| aisrvs | 2 (GPU1, GPU2) |
| parallel_env_per_aisrv | 4 |
| batch_size | 4096 |
| cache_multiplier | 16 |
| coordinator_sleep | 0.3s |
| data_fetch (avg) | ~320ms |
| real_train (avg) | ~45ms |
| sample_production_consumption_ratio | ~60 |
| steps/min (est.) | ~164 |
| GPU0 utilization | 16-30% |

## Experiment Results Summary

### 1. Coordinator Sleep 0.3→0.05 + Cache 16→4 ✅ DEPLOYED

**Rationale**: V1 coordinator sleeps 0.3s between buffer swaps — reduces swap latency by 6×. Cache=4 keeps buffer smaller (16384→16384 retained, but better utilization).

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| data_fetch (avg) | ~320ms | ~200ms | **-38%** |
| real_train (avg) | ~45ms | ~47ms | +4% |
| steps/min | ~164 | ~218 | **+33%** |
| ratio | ~60 | ~33 | **-45%** |
| buffer utilization | 4-12% | 20-44% | +3× |

**Verdict**: Best single optimization. coordinator_sleep was the biggest win.

### 2. Gamecore Scaling (8→16, 2 aisrv, 4 env) ⚠️ NO IMPROVEMENT

Only 8 of 16 gamecores were active (2 aisrv × 4 env = 8 slots). Performance identical to 8gc config. Extra gamecores sat idle at 0% CPU.

### 3. Gamecore Scaling (18gc, 3 aisrv on GPU1-3, 6 env) ❌ WORSE

| Metric | 8gc/2aisrv | 18gc/3aisrv |
|--------|-----------|-------------|
| steps/min | 218 | 198 |
| ratio | 33 | 14 |
| data_fetch | ~200ms (stable) | ~120ms→420ms (degrading) |
| GPU0 util | — | **66%** (best observed) |

**Verdict**: More data production (ratio=14 is great) but reverb contention with 3 aisrv writers degrades learner throughput over time.

### 4. Gamecore Scaling (16gc, 2 aisrv, 8 env) ❌ WORSE

| Metric | 8gc/4env | 16gc/8env |
|--------|---------|----------|
| steps/min | 218 | 207 |
| ratio | 33 | 16 |
| data_fetch | ~200ms | ~338ms (degrading) |
| aisrv CPU | ~280% | ~612% |

**Verdict**: All 16 gamecores active, aisrvs saturated at 600%+ CPU. Buffer overflow (8807/16384) caused GIL contention.

### 5. Gamecore Scaling (12gc, 3 aisrv, 4 env) ❌ WORSE

| Metric | 8gc/2aisrv | 12gc/3aisrv |
|--------|-----------|-------------|
| steps/min | 218 | 213 |
| ratio | 33 | 22 |
| data_fetch | ~200ms | ~361ms |

**Verdict**: Conservative scaling still degraded by reverb contention from 3rd aisrv.

### 6. torch.compile ❌ NO IMPROVEMENT

Applied `torch.compile(model, mode="reduce-overhead")`. real_train unchanged at ~45ms. Model too small (69→256→128→8) for compile benefits — kernel launch latency dominates.

### 7. Fill Parallelism (6 workers, 1024 fetch) ❌ CATASTROPHIC

| Metric | 3 workers | 6 workers |
|--------|----------|----------|
| data_fetch | ~200ms | **2315ms** |
| real_train | ~45ms | **234ms** |

**Verdict**: Python GIL contention destroyed performance. 6 threads fighting for deque access.

### 8. Optimized V1 (pin_memory + fixed locking) ❌ WORSE

Combined V1's MT-fill with V2's pin_memory + async H2D + single-lock fix. data_fetch increased from ~320ms to ~570ms. pin_memory overhead outweighed benefits for small tensors, and `next(iter())` pattern prevents async prefetch.

### 9. Batch Size 4096→2048 ❌ NO IMPROVEMENT

Steps/min unchanged (218). Each step processes half the data → same throughput but less training per unit time.

## Final Deployed Configuration

```env
KAIWU_GAMECORE_NUM=8
KAIWU_AISRV_NUM=2
KAIWU_PARALLEL_ENV_PER_AISRV=4
KAIWU_EXPERIMENT_REPLAY_BUFFER_CACHE_MULTIPLIER=4
KAIWU_EXPERIMENT_COORDINATOR_SLEEP=0.05
KAIWU_EXPERIMENT_TRAIN_BATCH_SIZE=4096
```

**Result**: ~218 steps/min (33% improvement over baseline)

## Key Findings

### 1. Data Production Rate is the Fundamental Bottleneck
The `sample_production_and_consumption_ratio` of ~33-60 means the learner consumes data 33-60× faster than 8 gamecores + 2 aisrvs can produce it. Pipeline optimizations can't fix a production deficit.

### 2. Scaling Gamecores Has Diminishing Returns Due to Reverb Contention
More gamecores require more aisrvs, which means more concurrent writers to the reverb server running in the learner process. The gRPC overhead grows faster than the data production benefit.

### 3. V1 Pipeline Architecture Limits Throughput
- `next(iter(reverb_dataset))` creates and discards a Python generator each step
- Double buffer swap requires coordinator sleep cycle (now 0.05s, was 0.3s)
- Python GIL prevents true parallelism between fill workers and training
- Each batch requires 4096 deque popleft calls + list comprehension + torch.stack

### 4. GPU Utilization is CPU-Limited, Not GPU-Limited
- Model (69→256→128→8) is too small to saturate GPU compute
- GPU0 varies 16-66% based on data availability, not compute
- GPUs 1-3 show 0% util — aisrvs run predictions on CPU since the model is tiny
- torch.compile, AMP, fused optimizers all ineffective for such a small model

### 5. The `next(iter())` Pattern is a Structural Bottleneck
`replay_buffer_wrapper.py:249` calls `next(iter(self.reverb_dataset))` each step. This prevents:
- Persistent iterator reuse (marginal benefit, ~1-10μs save)
- CUDA async prefetch (V2's async H2D is killed by this pattern)
- DataLoader-style prefetch_factor optimization

## Recommendations for Further Improvement

1. **ZMQ mode** (highest potential, highest risk): Bypasses reverb entirely. Uses shared memory + LearnerServer subprocesses. Would eliminate gRPC overhead. Requires careful integration with existing hot-patches.

2. **Framework-level iterator fix**: Patch `replay_buffer_wrapper.py` to keep a persistent iterator instead of `next(iter())`. Would enable V2's async prefetch benefits.

3. **Multi-learner training**: Split batch across GPU0+GPU3 with gradient sync. Framework may not support this natively.

4. **Algorithmic changes**: Larger model would better utilize GPUs. But CLAUDE.md says not to modify PPO algorithm until pipeline is stable.

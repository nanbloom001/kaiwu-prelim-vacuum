# Training Session Report — 2026-04-14

## Session Overview

| Item | Value |
|------|-------|
| Start time | 2026-04-14 22:49 |
| End time | 2026-04-14 23:47 |
| Duration | ~58 min |
| Starting checkpoint | model.ckpt-resume.pkl (from previous session) |
| Final checkpoint | model.ckpt-resume.pkl (latest_model.pkl, step ~103k) |
| Total learner steps | 103,760 |
| Total episodes (2 aisrv) | 404 |
| Branch | linux (commits a3a9a1c) |

## Configuration

### Hyperparameters (conf.py)

```
LR = 0.00005
BETA = 0.012
CLIP = 0.15
GAMMA = 0.99
LAMBDA = 0.95
VF_COEF = 0.5
ENTROPY_FLOOR = 0.5
ENTROPY_FLOOR_COEF = 1.0
GRAD_CLIP = 0.5
```

### Infrastructure

- AMP: enabled
- JIT trace: enabled
- torch.compile: enabled (reduce-overhead)
- Batch tensor: enabled
- Fused optimizer: enabled
- Model caching: enabled
- Replay buffer: ZMQ, target_fill=50000

### Curriculum Sampling

- Episode 1-40: warmup (70% anchor, 22% mild, 8% broad)
- Episode 41-200: blend (35% anchor, 40% mild, 25% broad)
- Episode 201-400: robust (10% anchor, 35% mild, 55% broad)
- Episode 400+: eval_hard (5% anchor, 20% mild, 75% broad_eval)

Broad profile: battery_max ∈ [120,160,200,260,320,420,560,720], charger_count ∈ [1,4], robot_count ∈ [1,4]

## Episode Results

### Overall

| Instance | Episodes | WIN | FAIL | Win Rate |
|----------|----------|-----|------|----------|
| aisrv-1 (pid 424) | 201 | 153 | 48 | 76.1% |
| aisrv-2 (pid 426) | 203 | 150 | 53 | 73.9% |
| **Combined** | **404** | **303** | **101** | **75.0%** |

### Death Breakdown

| Cause | aisrv-1 | aisrv-2 | Total | % of failures |
|-------|---------|---------|-------|---------------|
| Battery | 45 | 47 | 92 | 91% |
| Collision | 3 | 6 | 9 | 9% |

### Score Distribution

| Metric | WIN | FAIL |
|--------|-----|------|
| Avg score | 582 | 342 |
| Min score | 92 | 45 |
| Max score | 1196 | 1318 |

Note: Some FAIL episodes scored higher than some WINs (FAIL max=1318 vs WIN max=1196). These are episodes where the robot cleaned well but ran out of battery before completion.

### Instant Deaths

~46 of 101 FAIL episodes (46%) ended at <=300 steps with avg_score ~165. These are episodes where the robot died very early, typically on hard configurations (low battery_max, few chargers, many NPCs).

## Learner Training Metrics

### Entropy Collapse (CRITICAL)

```
22:51  entropy=2.01   policy=-8.4    value=90.5
22:55  entropy=1.93   policy=-13.8   value=129.6
23:00  entropy=1.77   policy=-19.0   value=190.1
23:05  entropy=1.56   policy=-18.4   value=189.8
23:10  entropy=1.21   policy=-20.3   value=223.1
23:15  entropy=0.99   policy=-22.2   value=251.4
23:20  entropy=0.61   policy=-22.0   value=280.4
23:25  entropy=0.30   policy=-23.3   value=320.1
23:30  entropy=0.18   policy=-22.3   value=339.4
23:35  entropy=0.13   policy=-22.0   value=360.4
23:40  entropy=0.06   policy=-25.8   value=404.1
23:47  entropy=0.05   policy=-24.7   value=391.5
```

Entropy dropped from 2.01 → 0.05 over 58 minutes. ENTROPY_FLOOR=0.5 was breached at ~23:22 (step ~50k) and ENTROPY_FLOOR_COEF=1.0 was insufficient to prevent collapse. By end of training, entropy was 10x below the floor.

### Training Performance

```
Initial:  real_train=3669ms (cold start), data_fetch=4ms
Stable:   real_train=25-30ms, data_fetch=3-5ms
Ratio:    sample_production_and_consumption_ratio = 270x (oversampling)
```

Learner throughput was excellent. Data pipeline was stable throughout.

## Death Trajectory Analysis

### Methodology

DEATH_TRAJ logs capture the last 20 steps before battery death, recording battery, charger_slack, nearest NPC distance, mode, and action at each step.

### Key Finding: NPC Not a Factor

ALL 47 battery deaths from aisrv-1 had `npc_dist >= 10` at every step in the death window. NPC avoidance was never a factor in any charging failure. Deaths are purely navigation/timing issues.

### Slack at Death Window Entry

`charger_slack = battery - chebyshev_charger_dist - max(8, 0.04*battery_max)`

| Slack range | Count | % | Interpretation |
|-------------|-------|---|----------------|
| <= -5 | 11 | 23% | Already doomed, return triggered way too late |
| <= 0 | 19 | 40% | Return triggered too late |
| 0 to 3 | 10 | 21% | Tight margin, barely insufficient |
| > 3 | 18 | 38% | Looked OK but still ran out |

### Approaching Charger Analysis

All 18 deaths where slack > 3 were actively approaching the charger (slack decreasing throughout). Average slack was 6.7, but they ran out of battery ~9 steps later. The robot was heading in the right direction — just not early enough.

### Battery Max Distribution in Deaths

```
battery_max=300:  ~55% of deaths (most sampled in broad profile)
battery_max=200:  ~26%
battery_max=320:  ~21%
battery_max=120:  rare but near-impossible to survive
```

### Oscillation Pattern

25% of deaths showed <=3 unique actions in the 20-step death window, indicating the robot was stuck in a repeated pattern near the charger but unable to reach it. The remaining 75% showed varied actions as the robot navigated toward the charger.

## Root Cause Analysis

### Root Cause 1: charger_slack Uses Chebyshev Distance (HIGH)

`preprocessor.py:224-225` computes `charger_slack` using Chebyshev (straight-line) distance. For a charger 30 Chebyshev cells away with walls between, the actual walkable path may be 60+ steps. The slack appears positive when the robot is actually in danger.

**Impact**: `urgency_penalty`, `low_battery_flag`, `charge_pressure` in reward function all trigger too late.

### Root Cause 2: Expert A* Underestimates Wall Detours in Unexplored Areas (HIGH)

Expert's cost map assigns `_UNEXPLORED_COST=1.8` to unexplored cells, but has no knowledge of walls there. When the robot walks through unexplored territory toward a charger, it may hit unknown walls and need costly detours. `BASE_RETURN_MARGIN=18` is insufficient to cover these detours on complex maps.

### Root Cause 3: Entropy Collapse Overrides Expert Bias (HIGH)

Entropy collapsed from 2.01 to 0.05 despite ENTROPY_FLOOR=0.5. The expert logit bias (3.0-8.0 for non-emergency) is too weak to redirect a very confident (low entropy) policy. The policy learned to keep cleaning because the reward signal for charging urgency was too delayed (see Root Cause 1).

## Recommendations for Next Training Round

### Priority 1: Fix Entropy Collapse

- Increase `ENTROPY_FLOOR_COEF` from 1.0 to 5.0 (or use multiplicative scaling)
- Consider: `effective_beta = var_beta * max(1.0, ENTROPY_FLOOR / entropy_val)`

### Priority 2: Increase Return Margin

- Increase `BASE_RETURN_MARGIN` from 18 → 28
- This gives ~10 extra battery steps for wall detours

### Priority 3: Fix charger_slack Distance Metric

- Use expert's A* path distance instead of Chebyshev in `preprocessor.charger_slack`
- This makes the reward signal accurately reflect true distance to charger

### Priority 4: Increase Non-Emergency Expert Bias

- Raise soft bias floor from 3.0 → 6.0 in `expert.get_logit_bias()`
- Helps overcome low-entropy policy when charge pressure is building

### Priority 5: Constrain Hard Configurations

- When `battery_max < 160`, guarantee `charger_count >= 2`
- Prevents nearly-impossible configurations that waste training samples

## Model Checkpoints

| File | Source | Notes |
|------|--------|-------|
| `latest_model.pkl` | Auto-saved at ep 220 (aisrv-1) | Final model, step ~103k |
| `model.ckpt-resume.pkl` | Copied from latest_model.pkl | Ready for next round |
| `model.ckpt-resume.meta.json` | Updated | clean_score=544, ep=220 |

Intermediate checkpoints (step 40k-100k) were inside the container at `/data/ckpt/` and were lost when `docker compose down` was executed.

## Branch Sync Status

All algorithm files are identical between `origin/win` and `linux`:

- `expert.py`: identical (trailing newline only)
- `preprocessor.py`: identical (trailing newline only)
- `conf.py`: hyperparameters identical (linux adds LEARNER_* infrastructure)
- `agent.py`: algorithm logic identical (linux adds AMP/JIT/caching/probes)
- `algorithm.py`: loss computation identical (linux adds AMP/batch tensor)
- `train_workflow.py`: episode logic identical (linux adds PerfWindow/timing)

# Optimization Round 2 — 2026-04-12

## Previous State

- ~4805 episodes trained, best robust=1468, best avg=640
- Maps 1-3 only, battery_max=300, max_step=1000
- Evaluation showed: maps 4-10 never charged, NPC collision common, wall stuck, random cleaning

## Changes Made

### 1. Expert Policy Rewrite (`expert.py`)

Replaced A* with weighted Dijkstra + NPC danger cost map:

- **Cost map**: 128x128 with base cost 1.0 (passable), INF (wall/unexplored), +exponential danger near NPCs
  - `danger = 15.0 * exp(-dist/2.0)` within Chebyshev radius 8
  - Unexplored cells = impassable (conservative planning)
- **Graduated fallback**: full avoidance → reduced (30%) → no avoidance
- **NPC safety filter**: blocks stepping onto NPC + moving toward NPC within distance 3
- **Dynamic replanning**: cost map rebuilt each step with updated vision + NPC positions

### 2. Reward Reshaping (`preprocessor.py`)

| Component | Before | After | Rationale |
|-----------|--------|-------|-----------|
| Cleaning | 2.0 × cleaned | unchanged | Primary signal |
| Streak | +0.15 × min(streak,5) | unchanged | Cleaning continuity |
| **Edge-following** | none | **+0.08×walls + 0.12×dirty_neighbors** | Encourage boundary walking |
| Explore | 0.02 × min(cells,4) | 0.05 × min(cells,6) | Stronger discovery signal |
| **Frontier** | 0.03 × density × (1-explored) | **0.10 × density × (0.3+0.7×clean)** | No decay, scales with progress |
| Charger approach | 0.03 × pressure × delta | **0.10** × pressure × delta | 3× stronger charging incentive |
| **Charge bonus** | none | **+0.5 × just_charged** | Direct reward for charging |
| NPC penalty | -0.04 × risk | -0.5 × risk² | Quadratic, much stronger |
| **Revisit (frontier)** | -0.15 × clip(v-1,0,5) | **-0.05 × clip(v-1,0,2)** | Allow purposeful frontier-walking |
| **Revisit (off-frontier)** | same | **-0.20 × clip(v-1,0,4)** | Penalize wasteful wandering |
| Stuck | -0.2 × invalid | -0.3 × invalid - 0.15 × stuck_steps/10 | Escalating penalty |
| Idle | -0.03 × clip(no_prog/30,0,1) | -0.1 × clip(no_prog/15,0,1) | Faster convergence |

### 3. Episode-End Bonus (`train_workflow.py`)

- Efficiency: `0.4×clean_ratio + 0.2×score_per_step` → `0.6×clean_ratio + 0.3×score_per_step`

### 4. Bug Fix: Directional Dirty Coordinates (`preprocessor.py`)

- **DIRS order mismatch**: was rotated 4 positions relative to ACTION_DELTAS
- **Coordinate swap**: `dirty_memory[gz, gx]` → `dirty_memory[gx, gz]`
- Both fixed; 8D directional_dirty feature now correctly aligns with action directions

### 5. Dead Code Removal (`preprocessor.py`)

Removed `get_action_biases()` and helpers (`_alignment_scores`, `_local_dirty_vector`, `_directional_patch_scores`, `_patch_mean`) — never called by any code path. Agent decisions use only model logits + expert override.

### 6. Mode Trigger Adjustment

- EVADE mode trigger: `nearest_npc_dist <= 2.0` → `<= 4.0` (earlier evasion)
- NPC risk flag: `(4-d)/4` → `(8-d)/8` (wider awareness range)

### 7. Training Config (`train_env_conf.toml`)

- Maps: `[1, 2, 3]` → `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`

## Resume Compatibility

All changes preserve model weight compatibility:
- SCALAR_DIM=74 unchanged, FEATURES=[1323,192,74,8] unchanged
- Network architecture unchanged
- `model.ckpt-resume.pkl` loads successfully
- Reward/bias changes are training-only, no feature dimension impact

## Training Status

- Session: `20260412-015356`
- Training started successfully, no Python errors
- Early episodes show mixed results (expected during adaptation to new rewards)

## Files Modified

| File | Changes |
|------|---------|
| `code/agent_ppo/feature/expert.py` | Full rewrite: Dijkstra + NPC cost map |
| `code/agent_ppo/feature/preprocessor.py` | Reward reshaping, bug fix, dead code removal, edge detection |
| `code/agent_ppo/workflow/train_workflow.py` | Episode efficiency bonus increase |
| `code/agent_ppo/conf/train_env_conf.toml` | Maps expanded to 1-10 |
| `ENVIRONMENT_MODEL.md` | New: complete environment model document |

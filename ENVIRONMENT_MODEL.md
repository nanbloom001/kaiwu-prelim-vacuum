# Robot Vacuum RL Environment Model

## 1. Overview

Robot Vacuum (清扫大作战) is a grid-world RL environment where an agent controls a vacuum robot to clean dirty tiles while managing battery life, avoiding NPC (official robot) collisions, and efficiently exploring the map.

**Key constraints:**
- 128×128 grid world
- 10 public maps (+ 5 hidden for final evaluation)
- Partial observability (21×21 local view)
- 8-directional movement
- Battery depletes per step, requires charging at charger stations
- Collision with NPC ends episode immediately

---

## 2. State Space (Observation)

Total feature dimension: **1619** (sum of below)

### 2.1 Local View (3 × 21 × 21 = 1323D)

21×21 grid centered on agent's position. 3 channels:

| Channel | Meaning | Values |
|---------|---------|--------|
| 0 | Obstacle | 1 = obstacle/wall, 0 = passable |
| 1 | Cleaned | 1 = cleaned, 0 = not cleaned |
| 2 | Dirty | 1 = has dirt, 0 = no dirt |

Each cell is exactly one of: obstacle (0), cleaned (1), or dirty (2). The 3 channels are one-hot derived from the raw map.

### 2.2 Global Memory (3 × 8 × 8 = 192D)

Coarse 8×8 representation of the full 128×128 map (each block = 16×16 cells). 3 channels:

| Channel | Meaning | Range |
|---------|---------|-------|
| 0 | Explored ratio | [0, 1] per block — fraction of cells ever observed |
| 1 | Dirt memory | [0, 1] — 1 if any cell in block was seen dirty and not yet cleaned |
| 2 | Visit heat | [0, 1] — normalized visit frequency (visit_count / step) |

### 2.3 Scalar Features (74D)

#### Core State (8D)
| # | Feature | Range | Description |
|---|---------|-------|-------------|
| 0 | step_norm | [0,1] | step_no / 2000 |
| 1 | step_ratio | [0,1] | step_no / max_step |
| 2 | battery_ratio | [0,1] | battery / battery_max |
| 3 | battery_max_norm | [0,1] | battery_max normalized to [100, 999] |
| 4 | clean_ratio | [0,1] | dirt_cleaned / total_dirt |
| 5 | remaining_dirt | [0,1] | 1 - clean_ratio |
| 6 | pos_x | [0,1] | x / 127 |
| 7 | pos_z | [0,1] | z / 127 |

#### Local Perception (7D)
| # | Feature | Range | Description |
|---|---------|-------|-------------|
| 8 | nearest_dirt_dist | [0,1] | Chebyshev distance to nearest dirt in local view / 10 |
| 9 | dirt_delta | {0,1} | 1 if approaching nearest dirt |
| 10 | local_dirt_density | [0,1] | Fraction of local view that is dirty |
| 11 | local_obstacle_density | [0,1] | Fraction of local view that is obstacle |
| 12 | local_frontier_density | [0,1] | Fraction of passable cells adjacent to unexplored |
| 13 | revisit_ratio | [0,1] | cur_visit_count / 6, clipped |
| 14 | stuck_steps | [0,1] | consecutive stuck steps / 20 |

#### Movement & Progress (5D)
| # | Feature | Range | Description |
|---|---------|-------|-------------|
| 15 | no_progress_steps | [0,1] | steps without cleaning / 80 |
| 16 | invalid_move_ema | [0,1] | EMA of invalid move rate |
| 17 | actual_legal_ratio | [0,1] | ratio of actually legal actions |
| 18 | charge_count | [0,1] | number of charges / 50 |
| 19 | just_charged | {0,1} | 1 if just charged this step |

#### Charger & Battery (7D)
| # | Feature | Range | Description |
|---|---------|-------|-------------|
| 20 | nearest_charger_dist | [0,1] | distance / 128 |
| 21 | nearest_charger_dx | [-1,1] | signed dx / 128 |
| 22 | nearest_charger_dz | [-1,1] | signed dz / 128 |
| 23 | charger_slack | [-1,1] | battery - dist - reserve / battery_max |
| 24 | low_battery_flag | {0,1} | battery <= 18% or slack <= 4 |
| 25 | charge_pressure | [0,1] | (8 - slack) / 8 |
| 26 | nearest_npc_dist | [0,1] | distance / 128 |

#### NPC (3D)
| # | Feature | Range | Description |
|---|---------|-------|-------------|
| 27 | nearest_npc_dx | [-1,1] | signed dx / 128 |
| 28 | nearest_npc_dz | [-1,1] | signed dz / 128 |
| 29 | npc_risk_flag | [0,1] | (4 - dist) / 4 |

#### Environment Config (5D)
| # | Feature | Range | Description |
|---|---------|-------|-------------|
| 30 | total_charger_norm | [0,1] | charger_count normalized [1,4] |
| 31 | npc_count_norm | [0,1] | npc_count normalized [1,4] |
| 32 | max_step_norm | [0,1] | max_step / 2000 |
| 33 | explored_ratio | [0,1] | global explored fraction |
| 34 | dirty_memory_ratio | [0,1] | global dirty memory fraction |

#### Immediate Feedback (4D)
| # | Feature | Range | Description |
|---|---------|-------|-------------|
| 35 | cleaned_this_step | [0,1] | cells cleaned / 12 |
| 36 | mode_clean | {0,1} | current mode = CLEAN |
| 37 | mode_charge | {0,1} | current mode = CHARGE |
| 38 | mode_evade | {0,1} | current mode = EVADE |

#### Extended NPC/Charger (18D)
- NPC slots 2-4: (dist, dx, dz) × 3 = 9D (padding with sentinel if fewer NPCs)
- Charger slots 2-4: (dist, dx, dz) × 3 = 9D (padding with sentinel if fewer chargers)

#### Directional Dirty (8D)
- For each of 8 directions: average dirty_memory along that ray (5 cells deep)

#### Last Action One-Hot (9D)
- 8 directions + 1 "no action" slot

### 2.4 Legal Action Mask (8D)

Binary mask indicating which of 8 directions are legal. Merged from:
- Server-provided `legal_action` (framework level)
- Locally computed `actual_legal_actions` (based on local view)

If all merged actions are 0, falls back to `actual_legal_actions`.

---

## 3. Action Space

| Action ID | Direction | (dx, dz) |
|-----------|-----------|----------|
| 0 | Right | (1, 0) |
| 1 | Up-Right | (1, -1) |
| 2 | Up | (0, -1) |
| 3 | Up-Left | (-1, -1) |
| 4 | Left | (-1, 0) |
| 5 | Down-Left | (-1, 1) |
| 6 | Down | (0, 1) |
| 7 | Down-Right | (1, 1) |

### Movement Rules

- **Straight moves (0,2,4,6):** Legal if target cell is passable
- **Diagonal moves (1,3,5,7):** Legal if target cell passable AND at least one adjacent cardinal cell passable (anti-corner-clipping)
- **Invalid move:** Agent stays in place, still costs 1 step + 1 battery

---

## 4. Transition Dynamics

### 4.1 Position Update
- Agent position = `hero.pos.{x, z}`
- After action: if legal, move to target cell; if illegal, stay in place

### 4.2 Cleaning
- If agent moves onto a dirty cell → cell becomes clean, `dirt_cleaned` increments
- Cleaning is automatic upon entering a dirty cell

### 4.3 Charging
- Charger stations are 3×3 blocks (defined by organ position + width/height)
- When agent is on a charger cell → battery increases (exact rate controlled by game engine)
- `charge_count` increments when charging occurs

### 4.4 Battery
- Each step costs 1 battery (whether move succeeds or not)
- Battery starts at `battery_max`
- If battery reaches 0 → episode ends (battery failure)

### 4.5 NPC (Official Robots)
- NPCs spawn randomly on roads at episode start
- NPCs move autonomously each step (random walk or game logic)
- **If agent's next position = NPC's position → collision → episode ends**
- NPC positions visible in frame_state (global, not just local view)

### 4.6 Map Information
- 10 public maps (ID 1-10), 5 hidden maps (ID 11-15)
- Map is 128×128 grid
- Cells: 0=obstacle, 1=cleaned, 2=dirty
- Agent only sees 21×21 local view centered on itself

---

## 5. Episode Termination

| Condition | Type | Description |
|-----------|------|-------------|
| step >= max_step | Truncated | Normal completion |
| battery == 0 | Terminated | Battery failure |
| Collision with NPC | Terminated | Collision failure |
| Abnormal error | Terminated | Unknown failure |

---

## 6. Reward Structure (Post-Optimization)

### 6.1 Per-Step Reward

| Component | Formula | Purpose |
|-----------|---------|---------|
| Cleaning | `2.0 × cleaned_this_step` | Primary objective |
| Cleaning streak | `0.15 × min(cleaned>0, 1) × min(streak, 5)` | Encourage systematic cleaning |
| Exploration | `0.05 × min(new_cells, 6)` | Discover new areas |
| Frontier | `0.03 × frontier_density × (1 - explored_ratio)` | Guide toward unexplored boundaries |
| Charger approach | `0.03 × charge_pressure × delta_slack` | Charge when needed |
| NPC penalty | `-0.5 × npc_risk²` | Avoid NPC collision (quadratic) |
| Revisit penalty | `-0.15 × clip(visit-1, 0, 5)` | Reduce redundant coverage |
| Invalid move | `-0.3 × last_invalid` | Avoid wall collision |
| Stuck escalation | `-0.15 × stuck_steps/10` | Escalating penalty for being stuck |
| Idle | `-0.1 × clip(no_progress/15, 0, 1)` | Penalize no-cleaning streaks |

**Total per-step reward clipped to [-4.0, 6.0]**

### 6.2 Episode-End Bonus

| Outcome | Bonus |
|---------|-------|
| Completed (max_step reached) | +1.5 |
| Battery failure | -2.5 |
| Collision with NPC | -4.0 |
| Unknown failure | -3.0 |

Plus efficiency bonus: `0.4 × cleaning_ratio + 0.2 × min(score/steps, 1.5)`

---

## 7. Mode System

Agent infers its current mode from state (used for action bias):

| Mode | Trigger | Action Bias |
|------|---------|-------------|
| CLEAN | Default (battery ok, NPC far) | Dirt + frontier + visit avoidance |
| CHARGE | `battery_ratio ≤ 0.16` or `charger_slack ≤ 4` | Charger approach + dirt-in-path |
| EVADE | `nearest_npc_dist ≤ 4.0` | NPC avoidance + charge when needed |

---

## 8. Expert Override (Safety Net)

Minimal safety intervention only when RL clearly fails:

| Layer | Trigger | Action |
|-------|---------|--------|
| L1: NPC filter | Always active | Block actions that step onto NPC or move directly toward NPC within distance 2 |
| L2: Battery emergency | `battery ≤ 2.0 × charger_dist + 40` | A* pathfinding to nearest charger (conservative: unexplored = impassable) |

---

## 9. Training Configuration

| Parameter | Default | Curriculum Range |
|-----------|---------|------------------|
| Maps | [1-10] | Subset sampling in broad profile |
| Robot count | 1 | 1-4 (mild ±1, broad full range) |
| Charger count | 4 | 1-4 (mild ±1, broad full range) |
| Max step | 1000 | 500-2000 |
| Battery max | 300 | 120-720 |

### Curriculum Stages

| Stage | Episodes | Anchor/Mild/Broad |
|-------|----------|-------------------|
| Warmup | 1-40 | 70/22/8 |
| Blend | 41-200 | 35/40/25 |
| Robust | 201+ | 15/35/50 |

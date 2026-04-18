# Charge Reward Pre-Constraint Report 2026-04-18

## Summary

This document records the **pre-constraint** reward redesign that was implemented and tested before the team decided to explore a stronger "survival as constraint" direction.

Its purpose is:

- preserve the exact intent of the older reward path
- make later regression or branch comparison easier
- clarify what this older path improved
- clarify why it was still insufficient

This report reflects the version that:

- kept the standard PPO objective
- kept survival mixed into the overall reward
- tried to fix behavior by rebalancing reward terms
- started a fresh scratch run with the revised reward

It does **not** include the later "charge-constraint" proposal as part of the design.

## Goal Of The Old Scheme

The old scheme tried to solve the following observed pathologies without changing the optimization paradigm:

- too many charge actions
- very high `mode_usage_contract`
- almost zero `mode_usage_expand`
- high `planner_policy_divergence_rate`
- low `avg_clean_per_step` / low `cps_win`
- many cases where reward structure locally looked healthier, but global behavior still converged to "safe but inefficient"

The design goal was:

1. keep survivability acceptable
2. improve CPS and cleaning throughput
3. reduce over-conservative return/contract behavior
4. make reward attribution more interpretable

The core assumption of the old scheme was:

**the main issue was still reward shaping imbalance, and a sufficiently improved scalar reward could steer the policy back toward healthier behavior without introducing constraints.**

## Implemented Reward Direction

The old scheme pushed reward in four directions:

### 1. Increase productive positive reward

The reward made these terms more important:

- `REWARD_CLEANING_BASE`
- `REWARD_STREAK_BONUS_BASE`
- `EXPLORE_REWARD_SCALE`
- `FRONTIER_REWARD_SCALE`
- `RETURN_PROGRESS_REWARD_SCALE`
- `RECOVERABILITY_REWARD_SCALE`

Intent:

- reward actual cleaning more strongly
- reward staying productive for multiple steps
- reward continued forward progress
- avoid a policy that only learns conservative survival

### 2. Weaken continuous negative shaping

The reward reduced the dominance of:

- `PLANNER_DIVERGENCE_PENALTY`
- `RETURN_STALL_BASE_PENALTY`
- `RETURN_STALL_EMA_SCALE`
- `NPC_PENALTY_SCALE`
- `IDLE_PENALTY_SCALE`
- `UNKNOWN_PATH_RISK_PENALTY`
- `NARROW_UNKNOWN_RISK_PENALTY`

Intent:

- stop negative shaping from crushing all useful behavior
- avoid cases where `npc + planner + stall` dominated almost every `REWARD_TOP`
- let `cleaning / streak / return_progress` actually appear in the top positive terms

### 3. Add direct cost terms for obviously bad charging behavior

The old scheme explicitly enabled:

- `OVERCHARGE_PENALTY_SCALE`
- `COVERAGE_EFFICIENCY_BONUS_SCALE`

Intent:

- penalize excessive charging frequency
- reward more regular and less redundant coverage

### 4. Strengthen terminal semantics

The old scheme also changed episode-end bonuses:

- stronger `EPISODE_COMPLETED_BONUS`
- explicit fail penalties for:
  - battery
  - collision
  - unknown

Intent:

- make end-of-episode semantics less ambiguous
- reduce situations where poor behavior still produced acceptable total reward

## Concrete Configuration Shape

The pre-constraint scheme moved defaults broadly toward:

- stronger cleaning and streak rewards
- smaller but still present planner/stall penalties
- smaller safety-pressure penalties except at truly critical margin
- enabled overcharge penalty
- enabled coverage efficiency bonus

Representative values from the implemented path:

- `REWARD_CLEANING_BASE = 0.90`
- `REWARD_STREAK_BONUS_BASE = 0.10`
- `EXPLORE_REWARD_SCALE = 0.10`
- `FRONTIER_REWARD_SCALE = 0.18`
- `RETURN_PROGRESS_REWARD_SCALE = 0.35`
- `RECOVERABILITY_REWARD_SCALE = 0.25`
- `PLANNER_ALIGNMENT_REWARD = 0.05`
- `PLANNER_DIVERGENCE_PENALTY = 0.03`
- `RETURN_STALL_BASE_PENALTY = 0.04`
- `RETURN_STALL_EMA_SCALE = 0.015`
- `CHARGE_MARGIN_LOW_PENALTY = 0.03`
- `CHARGE_MARGIN_CRITICAL_PENALTY = 0.18`
- `MISSED_CHARGE_PENALTY = 0.03`
- `CHARGER_NEARBY_NOT_CHARGED_PENALTY = 0.02`
- `OVERCHARGE_PENALTY_SCALE = 0.60`
- `COVERAGE_EFFICIENCY_BONUS_SCALE = 0.12`
- `NPC_PENALTY_SCALE = 1.0`
- `IDLE_PENALTY_SCALE = 0.02`
- `EPISODE_COMPLETED_BONUS = 6.0`
- `EPISODE_BATTERY_FAIL_BONUS = -12.0`
- `EPISODE_COLLISION_FAIL_BONUS = -16.0`
- `EPISODE_UNKNOWN_FAIL_BONUS = -10.0`

The implementation also made `charge_reward` configurable through:

- `CHARGE_REWARD_BASE`

instead of leaving it as a fixed hardcoded multiplier.

## What Improved

The old scheme was not a failure. It did improve several things.

### A. Local reward attribution became much healthier

After the redesign, `REWARD_TOP` more often showed:

- positive:
  - `cleaning`
  - `streak`
  - `return_progress`
- negative:
  - smaller `planner_alignment`
  - smaller `return_stall`
  - smaller `npc`

This was a real improvement over the earlier state where negative shaping dominated nearly every episode.

### B. Training became easier to interpret

The training pipeline preserved reward components and episode summaries now exposed:

- per-step reward components
- episode-level top positive and negative reward contributors
- more direct diagnosis of which terms dominated behavior

This made it much easier to see what the policy was actually optimizing.

### C. The policy no longer looked completely pathological at the local step level

Compared with the worst earlier runs:

- survivability improved
- reward breakdowns looked less obviously wrong
- the policy did not immediately collapse or diverge

## What Still Failed

Despite the improvements above, the old scheme still produced a bad global fixed point.

### 1. High survival but low task quality

The scratch run under the old scalar reward showed a very characteristic pattern:

- high win rate
- low battery fail
- but very low `avg_clean_per_step`
- very high `avg_charge_count`
- very high `mode_usage_contract`
- nearly zero `mode_usage_expand`
- very high `planner_policy_divergence_rate`
- high `return_stall_rate`

In plain terms:

**the policy learned to survive and finish, but in a highly conservative and inefficient way.**

### 2. Overcharging was still not truly suppressed

Even after enabling explicit overcharge penalty, the system still produced windows with:

- very high charge counts
- very high remaining charge at episode end

This indicated that the overcharge penalty was not structurally strong enough to alter the dominant behavior pattern.

### 3. Planner compliance did not recover

Even with weaker planner penalties, the policy still failed to become planner-consistent.

This showed that:

- lowering planner punishment reduced local domination
- but did **not** make the policy genuinely learn planned, regular movement

### 4. Expand behavior did not recover

This was one of the clearest failure signals.

Despite the redesign:

- `mode_usage_expand` remained near zero

The policy still preferred:

- contract
- harvest
- safe local completion

instead of broader structured exploration.

## Why The Team Moved Beyond This Scheme

The main reason was not that the old reward was "broken".

The main reason was:

**it improved local shaping but did not change the global ranking of behaviors.**

The policy still learned that:

- safe completion
- high contract usage
- frequent charging
- weak planning

was a better equilibrium than:

- high CPS
- planned charging
- regular coverage
- healthy expand behavior

This means the problem is not just one bad weight.

It means the scalar reward still mixes together:

- survival
- charging
- planning
- coverage efficiency

in a way that allows a bad but stable compromise policy.

That is why the discussion moved toward:

- survival as a constraint
- charging cost as a more explicit structured objective
- reducing cross-term competition inside one scalar reward

## Recommendation For Regression / Comparison

If the team wants to compare future branches against this older path, use this pre-constraint scheme as the reference for:

- "improved local reward attribution but still wrong global behavior"
- "scalar reward only"
- "survival still mixed directly into reward"

Recommended comparison dimensions:

- `avg_clean_per_step`
- `cps_win`
- `avg_charge_count`
- `avg_remaining_charge`
- `mode_usage_expand`
- `mode_usage_contract`
- `planner_policy_divergence_rate`
- `return_stall_rate`
- `battery_fail_rate`
- `collision_fail_rate`

The old scheme should be considered a **useful baseline**, not the final solution.

## Final Assessment

The pre-constraint reward redesign was:

- technically meaningful
- diagnostically valuable
- locally better than the previous reward
- but globally insufficient

It should be preserved as a regression baseline, because it demonstrates an important lesson:

**a scalar reward can look locally healthier and still converge to the wrong long-horizon behavior.**

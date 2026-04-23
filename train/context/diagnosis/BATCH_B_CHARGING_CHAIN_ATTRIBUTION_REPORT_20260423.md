# Batch B Charging Chain Attribution Report 2026-04-23

## Scope

- Repo: `/home/user/TcKaiwuFinal`
- Run session: `20260423-123114`
- Phase: `s1_survival_strong_heuristic_slice2a_v1`
- Start mode: `scratch`
- Focus:
  - explain why Batch B is producing poor live outcomes
  - distinguish between:
    - route-phase gate / penalty overload
    - late or skipped charging under high need

This report is intentionally diagnostic-only.
It does not propose code changes yet.

## Current Window Snapshot

At the time of analysis:

- `global_episode_count = 249`

`bootstrap_20`:

- `avg_clean_per_step = 0.4845`
- `battery_fail_rate = 0.50`
- `zero_charge_battery_fail_rate = 0.30`
- `win_rate = 0.40`
- `avg_route_phase_reward_ready_rate = 0.2156`
- `avg_route_phase_shadow_risk = 0.1708`
- `avg_reward_route_phase_risk_growth_penalty = -0.00114`
- `avg_reward_skip_needed_charge_penalty = -0.01033`
- `route_phase_return_stall_rate = 0.4913`
- `planner_policy_divergence_rate = 0.8211`
- `avg_route_phase_planner_divergence_rate = 0.7425`
- `avg_charge_count_battery_fail = 1.20`
- `battery_positive_reward_rate = 0.0`

`global_40`:

- `avg_clean_per_step = 0.4431`
- `battery_fail_rate = 0.50`
- `zero_charge_battery_fail_rate = 0.25`
- `win_rate = 0.45`
- `avg_route_phase_reward_ready_rate = 0.1854`
- `avg_route_phase_shadow_risk = 0.1428`
- `avg_reward_route_phase_risk_growth_penalty = -0.00120`
- `avg_reward_skip_needed_charge_penalty = -0.01058`
- `route_phase_return_stall_rate = 0.4784`
- `planner_policy_divergence_rate = 0.8340`
- `avg_route_phase_planner_divergence_rate = 0.7480`
- `avg_charge_count_battery_fail = 1.00`
- `battery_positive_reward_rate = 0.0`

## Primary Conclusion

Batch B is clearly active:

- `route_phase_reward_ready_rate` is much higher than in Batch A
- route-phase risk-growth reward is non-zero
- Batch A economics fixes remain intact:
  - `battery_positive_reward_rate = 0.0`

However, the dominant live failure mechanism is **not** best explained as
\"route-phase penalty overload\".

The stronger explanation is:

> The policy is still charging too late or skipping charging entirely under high need.
> High `route_phase_reward_ready_rate` mainly marks episodes that have already entered
> a dangerous return context while undercharged.

In other words:

- Batch B is activating in the right kinds of risky contexts
- but the policy is arriving there too late, with too little charge margin
- and the strongest live symptom is still **zero/low-charge battery collapse**

## Why The Earlier \"Gate Too Wide\" Explanation Is Insufficient

It is true that `route_phase_reward_ready_rate` rose materially in Batch B.
But that metric is not a pure intervention count.

`route_phase_reward_ready` means:

- already in `CONTRACT/RETURN`
- route context exists
- `route_phase_shadow_risk >= threshold`

So a high ready rate already encodes:

- higher shadow risk
- more route-phase time

Also, the actual route-phase penalty is narrower still:

- it only fires when readiness is true
- and `route_phase_shadow_risk` is worsening

Therefore:

- \"high ready correlates with failure\" is not enough to prove
  that the gate itself is the root cause
- we need to compare it against charging behavior and other penalties

## Attribution By Ready-Rate Buckets

Using the most recent `120` completed episodes:

### Bucket A: `avg_route_phase_reward_ready_rate >= 0.30`

- count: `23`
- `win_rate = 0.1304`
- `battery_fail_rate = 0.8261`
- `zero_charge_battery_fail_rate = 0.5652`
- `charge_count = 1.1304`
- `avg_route_phase_shadow_risk = 0.3370`
- `avg_route_phase_reward_ready_rate = 0.4313`
- `avg_reward_skip_needed_charge_penalty = -0.01244`
- `avg_reward_route_phase_risk_growth_penalty = -0.00287`
- `route_phase_return_stall_rate = 0.4460`
- `planner_policy_divergence_rate = 0.7522`
- `route_phase_planner_divergence_rate = 0.6549`
- `clean_per_step = 0.5914`
- `total_reward = -1.9121`

### Bucket B: `0.10 <= ready < 0.30`

- count: `67`
- `win_rate = 0.4478`
- `battery_fail_rate = 0.5224`
- `zero_charge_battery_fail_rate = 0.2537`
- `charge_count = 4.3284`
- `avg_reward_skip_needed_charge_penalty = -0.01393`
- `avg_reward_route_phase_risk_growth_penalty = -0.00145`
- `route_phase_return_stall_rate = 0.5080`
- `planner_policy_divergence_rate = 0.8350`
- `route_phase_planner_divergence_rate = 0.7626`
- `clean_per_step = 0.4505`

### Bucket C: `ready < 0.10`

- count: `30`
- `win_rate = 0.8667`
- `battery_fail_rate = 0.0667`
- `zero_charge_battery_fail_rate = 0.0`
- `charge_count = 9.9`
- `avg_route_phase_shadow_risk = 0.0402`
- `avg_route_phase_reward_ready_rate = 0.0504`
- `avg_reward_skip_needed_charge_penalty = -0.00591`
- `avg_reward_route_phase_risk_growth_penalty = -0.00043`
- `route_phase_return_stall_rate = 0.4541`
- `planner_policy_divergence_rate = 0.8631`
- `route_phase_planner_divergence_rate = 0.7418`
- `clean_per_step = 0.3936`
- `total_reward = 182.9420`

### Interpretation

The strongest directional difference is:

- high-ready bucket:
  - very low `charge_count`
  - very high `zero_charge_battery_fail_rate`
  - much stronger `skip_needed_charge_penalty`
- low-ready bucket:
  - high `charge_count`
  - near-zero battery fail

By contrast, the route/stall explanation is weaker:

- high-ready bucket does **not** have obviously worse stall than low-ready
- high-ready bucket does **not** have worse planner divergence than low-ready
- in fact, high-ready episodes often show:
  - lower divergence
  - similar or slightly better stall
  - but much worse charging behavior

This makes the charging explanation stronger than the route-overload explanation.

## Battery Failure Type Attribution

Recent `120` episodes:

- `mid_recoverability_loss = 54`
- `early_unrecoverable = 2`

### `mid_recoverability_loss`

- `charge_count = 1.0556`
- `avg_route_phase_reward_ready_rate = 0.2717`
- `avg_route_phase_shadow_risk = 0.2105`
- `avg_reward_skip_needed_charge_penalty = -0.0152`
- `avg_reward_route_phase_risk_growth_penalty = -0.0021`
- `route_phase_return_stall_rate = 0.4897`
- `planner_policy_divergence_rate = 0.8128`
- `route_phase_planner_divergence_rate = 0.7462`
- `clean_per_step = 0.5126`

### `early_unrecoverable`

- `charge_count = 0.0`
- `avg_route_phase_reward_ready_rate = 0.6117`
- `avg_route_phase_shadow_risk = 0.6759`
- `avg_reward_skip_needed_charge_penalty = -0.0234`
- `avg_reward_route_phase_risk_growth_penalty = -0.0035`
- `route_phase_return_stall_rate = 0.4507`
- `planner_policy_divergence_rate = 0.5519`
- `route_phase_planner_divergence_rate = 0.3447`
- `clean_per_step = 0.7033`

### Interpretation

Even in the failure buckets, the recurring pattern is:

- very low charge count
- strong skipped-charge penalty
- dangerous return context already active

The dominant failure type is still not:

- \"route planner diverged so hard that return collapsed\"

It is more accurately:

- \"the agent kept harvesting, entered danger undercharged, then could not recover\"

## Within-Profile Checks

The pattern is not just caused by `broad` dominating.

### Anchor

High-ready:

- count `3`
- `win_rate = 0.3333`
- `battery_fail_rate = 0.6667`
- `zero_charge_battery_fail_rate = 0.6667`
- `charge_count = 1.3333`
- `avg_reward_skip_needed_charge_penalty = -0.0181`

Low-ready:

- count `7`
- `win_rate = 1.0`
- `battery_fail_rate = 0.0`
- `zero_charge_battery_fail_rate = 0.0`
- `charge_count = 16.5714`
- `avg_reward_skip_needed_charge_penalty = -0.0068`

### Mild

High-ready:

- count `3`
- `win_rate = 0.0`
- `battery_fail_rate = 1.0`
- `zero_charge_battery_fail_rate = 0.3333`
- `charge_count = 1.0`
- `avg_reward_skip_needed_charge_penalty = -0.0131`

Low-ready:

- count `8`
- `win_rate = 1.0`
- `battery_fail_rate = 0.0`
- `zero_charge_battery_fail_rate = 0.0`
- `charge_count = 7.25`
- `avg_reward_skip_needed_charge_penalty = -0.0051`

### Broad

High-ready:

- count `17`
- `win_rate = 0.1176`
- `battery_fail_rate = 0.8235`
- `zero_charge_battery_fail_rate = 0.5882`
- `charge_count = 1.1176`
- `avg_reward_skip_needed_charge_penalty = -0.0113`
- `route_phase_return_stall_rate = 0.4231`
- `route_phase_planner_divergence_rate = 0.6200`
- `clean_per_step = 0.6139`

Low-ready:

- count `15`
- `win_rate = 0.7333`
- `battery_fail_rate = 0.1333`
- `zero_charge_battery_fail_rate = 0.0`
- `charge_count = 8.2`
- `avg_reward_skip_needed_charge_penalty = -0.0060`
- `route_phase_return_stall_rate = 0.4262`
- `route_phase_planner_divergence_rate = 0.7322`
- `clean_per_step = 0.3761`

### Interpretation

The same direction holds inside every profile:

- high-ready subsets are the low-charge, high-battery-fail subsets
- low-ready subsets are the high-charge, high-win subsets

In `broad`, the evidence is especially clean:

- high-ready does **not** have worse route divergence or worse stall
- but it does have far lower charge count and far worse battery outcome

So the attribution survives profile stratification.

## Episode-Level Pattern

The highest-ready episodes are almost all:

- battery failures
- low or zero charge count
- high CPS
- negative total reward

Representative pattern:

- `ready_rate ~ 0.35 to 0.93`
- `charge_count = 0 to 2`
- `clean_per_step ~ 0.61 to 0.93`
- `result = battery`

This is the exact shape expected from:

- aggressive harvesting
- delayed charging
- undercharged entry into dangerous return context

It is **not** the shape expected from:

- low productivity route dithering

## Final Attribution

The current best explanation is:

1. Batch B successfully raised route-phase readiness in the intended high-risk contexts
2. But the policy still tends to keep harvesting too long
3. It enters route-phase danger undercharged
4. That produces:
   - low `charge_count`
   - high `skip_needed_charge_penalty`
   - high `zero_charge_battery_fail_rate`
5. High `ready_rate` is therefore mostly a marker of already-dangerous, undercharged return entry

So the dominant live failure mechanism is:

> **late or skipped charging under high need**

not:

> **route-phase penalty overload**

## Practical Implication For Next Tuning Round

The next tuning round should not begin by simply narrowing `route_phase_reward_ready` again.

That would risk suppressing a signal that is activating in the right contexts.

The next investigation should instead target the charging decision chain:

1. Why the policy is still not charging soon enough under high need
2. Why `skip_needed_charge_penalty` remains the dominant negative charging signal
3. Why high-risk buckets still collapse to `charge_count ~ 1`
4. Whether `charge_need_score / battery_state / planner readiness` still reach the action layer too late

## Decision

Current run status:

- Batch B is **active**
- Batch B is **not yet successful**
- Primary blocker:
  - charging remains too late / skipped under high need
- Secondary effects:
  - route-phase risk signals are visible
  - but they are not the main causal explanation for the failure pattern

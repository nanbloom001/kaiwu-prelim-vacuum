# Fixed8 Generalization Setup - 2026-04-23

## Purpose

This setup freezes the slice2a curriculum difficulty so training distribution no longer mixes sampled max step, charger count, battery max, or official robot count. It is intended to test whether the current reward/control stack can learn a stable charging policy under one explicit survival setting before reintroducing broader curriculum sampling.

## Phase

Use:

```bash
python3 train/run_training_phase.py s1_survival_strong_heuristic_slice2a_fixed8_v1 --seed-label fixed8 --start-mode scratch
```

Dry-run:

```bash
python3 train/run_training_phase.py s1_survival_strong_heuristic_slice2a_fixed8_v1 --seed-label dry --dry-run
```

## Fixed Training Distribution

- Train maps: `1,2,3,4,5,6,7,8`
- Generalization benchmark maps: `9,10`
- Max step: `1000`
- Charger count: `3`
- Battery max: `150`
- Official robot count: `4`
- Map randomization: enabled across the 8 training maps

## Implementation Contract

The phase enables `KAIWU_ENV_FIXED_DIFFICULTY=1`. In this mode, `EnvConfigSampler` still samples curriculum profiles for attribution, but every sampled profile is forced back to the same fixed environment distribution. This keeps profile metrics observable without allowing mild/broad profiles to silently change the task difficulty.

The generalization split is configured through benchmark env vars:

- `KAIWU_BENCHMARK_MAPS=9,10`
- `KAIWU_BENCHMARK_ROUNDS_JSON=[{"name":"fixed8_generalization","desc":"3 chargers / 4 robots / 1000 steps / 150 battery","charger_count":3,"robot_count":4,"max_step":1000,"battery_max":150}]`

## Expected Readout

During training, `episode_end` logs should report only:

- `chargers:3`
- `robots:4`
- `battery_max:150`
- `max_step:1000`
- `map` in `1..8`

During benchmark/generalization evaluation, episodes should run only on maps `9` and `10` under the same fixed difficulty.

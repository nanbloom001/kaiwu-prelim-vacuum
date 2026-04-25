# Task 0 Fixed Benchmark Contract Evidence

Date: 2026-04-25

Source document: `train/context/LTSPPO_BENCHMARK_ADAPTATION_20260425.md`

The adaptation document explicitly fixes the local holdout benchmark contract as follows:

| Parameter | Required value | Evidence |
| --- | ---: | --- |
| `max_step` | `1000` | Listed under "Local fixed benchmark contract for win_YJY" |
| `battery_max` | `150` | Listed under "Local fixed benchmark contract for win_YJY" |
| `robot_count` | `4` | Listed under "Local fixed benchmark contract for win_YJY" |
| `charger_count` | `3` | Listed under "Local fixed benchmark contract for win_YJY" |
| maps | `[4, 7]` | Listed under "Local fixed benchmark contract for win_YJY" |
| episodes per map | `10` | Listed under "Local fixed benchmark contract for win_YJY" |
| total episodes | `2 * 10 = 20` | Listed under "Local fixed benchmark contract for win_YJY" |
| `map_random` | `false` during selected-map episodes | Listed under "Local fixed benchmark contract for win_YJY" |

The document also states that LTSPPO's multi-round defaults must not override this local contract and that training maps remain `[1, 2, 3, 5, 6, 8, 9, 10]` while benchmark maps `[4, 7]` stay out of the training sampler.

Minimum AI-analysis log schema was defined in the same document and includes: episode id, map id, score, done/fail reason, step, action, planner mode/target, battery, charger distance/slack, reward components, death replay path, and checkpoint id.

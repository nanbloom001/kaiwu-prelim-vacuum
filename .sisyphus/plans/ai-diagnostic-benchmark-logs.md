# AI-Diagnostic Benchmark Logs

## TL;DR
> **Summary**: Upgrade the fixed `[4,7]` holdout benchmark logs from basic score telemetry into compact, versioned evidence that lets AI classify why a model fails: battery timing, charger discovery, NPC/path blockage, loops, low-value revisits, planner-policy divergence, and map imbalance.
> **Deliverables**:
> - Backward-compatible schema v2 in `code/agent_ppo/eval/holdout_benchmark.py`
> - Analyzer compatibility and diagnostic classification improvements in `train/analyze_holdout_benchmark.py`
> - Compact `evidence_windows`, `final_window`, `missing_signals`, and optional `ai_summary.json`
> - QA fixtures/evidence proving old-schema compatibility, new-schema diagnostics, missing-field resilience, and inference-only safety
> **Effort**: Medium
> **Parallel**: YES - 3 waves
> **Critical Path**: Task 1 → Task 2 → Task 4 → Task 5

## Context
### Original Request
用户要求：“请你规划如何优化日志，目标是能辅助AI判断模型存在的问题”。

### Interview Summary
- Current benchmark is `code/agent_ppo/eval/holdout_benchmark.py`, fixed to holdout maps `[4,7]`, `max_step=1000`, `battery_max=150`, `robot_count=4`, `charger_count=3`.
- Current analyzer is `train/analyze_holdout_benchmark.py`, with failure classes: `charger_unknown`, `return_too_late`, `optimistic_route_budget`, `npc_or_path_blocked`, `repeated_invalid_move`, `high_score_battery_death`, `late_battery_death`, `unknown`.
- Current schema mismatch: benchmark emits `result` and `steps`; analyzer primarily expects `fail_reason|done_reason|status` and `finished_steps|step`.
- Current `guidance` is fragile because `holdout_benchmark._extract_guidance()` tries `Preprocessor._get_guidance()`, but current `Preprocessor` does not expose that method.

### Metis Review (gaps addressed)
- Plan must define pre-action vs post-action semantics explicitly.
- Plan must preserve old schema while adding canonical aliases.
- Plan must add `field_availability` / `missing_signals` to avoid false confidence when optional signals are absent.
- Plan must cap log volume with compact windows, not duplicate full trajectories into summaries.
- Plan must forbid PPO/reward/planner behavior changes.

## Work Objectives
### Core Objective
Make benchmark logs and analyzer outputs decision-useful for AI diagnosis without changing model behavior, reward shaping, training workflow, checkpoint save/load behavior, or submission tooling.

### Deliverables
- `code/agent_ppo/eval/holdout_benchmark.py`: additive schema v2, canonical aliases, step telemetry, lifecycle summaries, evidence windows, missing-signal reporting.
- `train/analyze_holdout_benchmark.py`: old/new schema tolerant parsing, step-log enrichment, stronger deterministic failure classification, missing-signal warnings.
- `train/context/CHANGELOG.md`: one-line summary appended after implementation.
- `.sisyphus/evidence/`: command outputs and compact JSON/Markdown evidence from QA.

### Definition of Done (verifiable conditions with commands)
- `python -m py_compile code/agent_ppo/eval/holdout_benchmark.py train/analyze_holdout_benchmark.py`
- `python train/analyze_holdout_benchmark.py --input train/context/HOLDOUT_BENCHMARK_20260425_0941.json --output-md .sisyphus/evidence/ai-log-old-schema-analysis.md` completes without crashing.
- Synthetic new-schema fixture analysis completes and classifies at least one battery/charger failure deterministically.
- A dry-run benchmark command still works: `python train/run_holdout_benchmark.py --dry-run --maps 4,7 --episodes-per-map 1 --checkpoint code/model.ckpt-resume.pkl --output .sisyphus/evidence/ai-log-dryrun.json`.
- No new raw benchmark logs are written under `train/context/`; only compact summaries or existing context files are touched.

### Must Have
- Backward compatibility for existing `result`, `steps`, `overall`, `per_map`, `episodes`, `step_log` fields.
- New canonical aliases: `fail_reason`, `finished_steps`, `max_step`, `battery_max`, `status`, `schema_version=2`.
- Step records split into:
  - `decision_context`: pre-action planner/policy/mask state.
  - `outcome_state`: post-action battery/dirt/position/reward state.
- Episode summaries include compact `final_window` and targeted `evidence_windows`, capped to 50 rows per episode summary.
- Analyzer emits `missing_signals`/warnings rather than silently defaulting absent optional fields to meaningful-looking zeros.

### Must NOT Have (guardrails, AI slop patterns, scope boundaries)
- Must not change PPO policy, network, reward shaping, planner scoring, training workflow semantics, checkpoint save/load behavior, Docker compose, or packaging.
- Must not copy Linux LTSPPO algorithm/reward code wholesale.
- Must not require LTSPPO-only fields such as `mode_prob`, `target_prob`, `route_anchor_prob`; they must be optional with fallbacks.
- Must not store raw step logs or large artifacts in `train/context/`.
- Must not remove old fields or break old holdout JSON files.

## Verification Strategy
> ZERO HUMAN INTERVENTION - all verification is agent-executed.
- Test decision: tests-after + Python compile + analyzer fixture commands; no formal test framework setup required for this diagnostic patch.
- QA policy: Every task has agent-executed scenarios.
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.{ext}`

## Execution Strategy
### Parallel Execution Waves
> Target: 5-8 tasks per wave. <3 per wave (except final) = under-splitting.
> Extract shared dependencies as Wave-1 tasks for max parallelism.

Wave 1: Task 1 schema contract, Task 2 benchmark emitter telemetry, Task 3 analyzer parser compatibility.
Wave 2: Task 4 analyzer classification enrichment, Task 5 AI summary/evidence windows, Task 6 QA fixtures and changelog.
Wave 3: Final verification wave F1-F4.

### Dependency Matrix (full, all tasks)
- Task 1 blocks Tasks 2, 3, 4, 5, 6.
- Task 2 blocks Tasks 4, 5, 6.
- Task 3 blocks Task 4 and Task 6.
- Task 4 blocks Task 6.
- Task 5 blocks Task 6.
- Task 6 blocks Final Verification Wave.

### Agent Dispatch Summary (wave → task count → categories)
- Wave 1 → 3 tasks → `quick`, `quick`, `quick`
- Wave 2 → 3 tasks → `unspecified-low`, `unspecified-low`, `quick`
- Final → 4 review tasks → `oracle`, `unspecified-high`, `unspecified-high`, `deep`

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [x] 1. Define schema v2 contract and canonical aliases

  **What to do**: In `code/agent_ppo/eval/holdout_benchmark.py`, add a local schema contract section near `SCHEMA_VERSION` and bump local benchmark schema to `SCHEMA_VERSION = 2`. Add canonical aliases in every episode result returned by `_run_eval_episode()` and `_empty_episode_result()`:
  - Preserve existing `result` and `steps`.
  - Add `fail_reason` equal to the normalized terminal reason currently stored in `result`.
  - Add `done_reason` equal to `fail_reason`.
  - Add `status = "completed"` when `fail_reason == "completed"`, otherwise `"failed"`.
  - Add `finished_steps` equal to existing `steps`.
  - Add `max_step`, `battery_max`, `robot_count`, `charger_count` from `round_def`.
  - Add `field_availability` and `missing_signals` containers initialized even if empty.
  - Add a compact `contract` object to `snapshot` with maps, episodes per map, fixed config, and checkpoint metadata; do not remove existing top-level keys.

  **Must NOT do**: Do not remove, rename, or alter existing `result`, `steps`, `overall`, `per_map`, `episodes`, or `step_log`. Do not edit `train/run_holdout_benchmark.py` unless compile reveals direct incompatibility.

  **Recommended Agent Profile**:
  - Category: `quick` - Reason: localized additive schema edits in one runtime file.
  - Skills: [] - No special skill required.
  - Omitted: [`git-master`] - No commit requested by this plan task.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: [2, 3, 4, 5, 6] | Blocked By: []

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `code/agent_ppo/eval/holdout_benchmark.py:38` - current `SCHEMA_VERSION = 1`.
  - Pattern: `code/agent_ppo/eval/holdout_benchmark.py:150-162` - current snapshot shape with `overall`, `per_map`, `episodes`.
  - Pattern: `code/agent_ppo/eval/holdout_benchmark.py:327-348` - current episode result dict with `result` and `steps`.
  - Pattern: `code/agent_ppo/eval/holdout_benchmark.py:389-423` - current aggregate keys that must remain compatible.
  - Analyzer contract: `train/analyze_holdout_benchmark.py:100-105` - analyzer normalizes `fail_reason`, `done_reason`, `status`.
  - Analyzer contract: `train/analyze_holdout_benchmark.py:161-164` - analyzer expects `finished_steps` or `step`, not `steps`.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `python -m py_compile code/agent_ppo/eval/holdout_benchmark.py`
  - [ ] Static read confirms `_run_eval_episode()` result contains both old (`result`, `steps`) and new (`fail_reason`, `done_reason`, `status`, `finished_steps`) keys.
  - [ ] Static read confirms `_empty_episode_result()` includes the same canonical alias keys.
  - [ ] Static read confirms no raw logs are redirected to `train/context/`.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```
  Scenario: Schema aliases compile and coexist
    Tool: Bash
    Steps: Run `python -m py_compile code/agent_ppo/eval/holdout_benchmark.py` from repo root.
    Expected: Exit code 0.
    Evidence: .sisyphus/evidence/task-1-schema-compile.txt

  Scenario: Backward compatibility static check
    Tool: Bash
    Steps: Use a Python one-liner to read `code/agent_ppo/eval/holdout_benchmark.py` and assert strings `"result"`, `"steps"`, `"fail_reason"`, `"finished_steps"`, `"field_availability"`, `"missing_signals"` are present.
    Expected: One-liner exits 0 and prints PASS.
    Evidence: .sisyphus/evidence/task-1-schema-static.txt
  ```

  **Commit**: NO | Message: `feat(benchmark): add diagnostic schema aliases` | Files: [code/agent_ppo/eval/holdout_benchmark.py]

- [x] 2. Add pre-action decision context and post-action outcome state to step logs

  **What to do**: In `_run_eval_episode()` in `code/agent_ppo/eval/holdout_benchmark.py`, restructure each step record so AI can distinguish what the agent knew before acting from what happened after acting:
  - Before `env.step(act)`, capture `policy_info = agent.planner.update(...)`, `act_data`, selected action, legal mask, safe mask, policy/planner distributions if available.
  - Add `decision_context` with: `step`, `pos_before`, `last_action`, `chosen_action`, `policy_action` from `act_data.action`, `greedy_action` from `act_data.d_action`, `mix_alpha`, `policy_top1`, `planner_top1`, `planner_match`, `action_entropy`, `legal_action_count`, `safe_action_count`, `target_mode`, `should_charge`, `charger_distance`, `charger_slack`, `battery`, `battery_ratio`, `on_charger`, `nearest_npc_distance`, `frontier_density`, `local_unknown_ratio`, `local_dirty_ratio`, `new_known_cells`.
  - After `env.step(act)` and after calling `agent.observation_process(env_obs)` for the next obs, add `outcome_state` with: `pos_after`, `battery`, `battery_delta`, `dirt_cleaned`, `cleaned_delta`, `reward`, `total_reward`, `done`, `terminated`, `truncated`, `nearest_charger_dist`, `nearest_npc_dist`, `cur_revisit_count`, `observed_cells_count`, `known_charger_count`.
  - Keep current top-level simple fields (`step`, `action`, `reward`, `battery`, `charger_slack`, etc.) for compatibility, but source them consistently from `outcome_state` after preprocessing the new observation.
  - Replace fragile `_extract_guidance(fm)` dependency with `_extract_policy_info(policy_info)` and `_extract_action_diagnostics(act_data, policy_info, selected_action)`. If fields are missing, add their names to `missing_signals`.

  **Must NOT do**: Do not change action selection logic. Do not call `agent.learn()`, `agent.save_model()`, or mutate planner behavior. Do not log full `passable_map`/`observed_map` arrays per step.

  **Recommended Agent Profile**:
  - Category: `quick` - Reason: localized emitter enhancement with careful field extraction.
  - Skills: [] - No special skill required.
  - Omitted: [`frontend-ui-ux`] - No UI work.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: [4, 5, 6] | Blocked By: [1]

  **References**:
  - Runtime loop: `code/agent_ppo/eval/holdout_benchmark.py:242-307` - current action/predict/step/log loop.
  - Current step record: `code/agent_ppo/eval/holdout_benchmark.py:275-293` - current flat step schema.
  - Agent output: `code/agent_ppo/agent.py:203-230` - `guided_predict()` returns `ActData` with `prob`, `policy_prob`, `planner_prob`, `mix_alpha`, `action_mask`.
  - Agent action conversion: `code/agent_ppo/agent.py:175-178` - `action_process()` semantics.
  - Planner signals: `code/agent_ppo/algorithm/algorithm.py` `PolicyInfo` - fields include `safe_action_mask`, `action_scores`, `target_mode`, `charger_distance`, `charger_slack`, `should_charge`, `on_charger`, `frontier_density`, `local_unknown_ratio`, `new_known_cells`.
  - Preprocessor signals: `code/agent_ppo/feature/preprocessor.py:43-78` - state counters and charger arrival fields.

  **Acceptance Criteria**:
  - [ ] `python -m py_compile code/agent_ppo/eval/holdout_benchmark.py`
  - [ ] Step log rows include both `decision_context` and `outcome_state` keys.
  - [ ] Step log rows still include old top-level keys: `step`, `action`, `reward`, `battery`, `battery_max`, `dirt_cleaned`, `total_dirt`, `charger_slack`, `nearest_charger_dist`, `nearest_npc_dist`.
  - [ ] Missing optional fields are represented in `missing_signals`, not silently as misleading zeros.

  **QA Scenarios**:
  ```
  Scenario: Emitter compiles with diagnostic helpers
    Tool: Bash
    Steps: Run `python -m py_compile code/agent_ppo/eval/holdout_benchmark.py`.
    Expected: Exit code 0.
    Evidence: .sisyphus/evidence/task-2-step-telemetry-compile.txt

  Scenario: Static schema includes pre/post semantics
    Tool: Bash
    Steps: Use Python to assert `decision_context`, `outcome_state`, `missing_signals`, `policy_top1`, `planner_top1`, and `battery_delta` occur in `code/agent_ppo/eval/holdout_benchmark.py`.
    Expected: PASS.
    Evidence: .sisyphus/evidence/task-2-step-telemetry-static.txt
  ```

  **Commit**: NO | Message: `feat(benchmark): add decision and outcome step telemetry` | Files: [code/agent_ppo/eval/holdout_benchmark.py]

- [x] 3. Make analyzer tolerant of old and new benchmark schemas

  **What to do**: In `train/analyze_holdout_benchmark.py`, update parsing helpers so old and new outputs classify consistently:
  - Update `normalize_fail_reason()` to check `fail_reason`, `done_reason`, `status`, then `result`.
  - Update step count parsing to prefer `finished_steps`, then `steps`, then `step`.
  - Update `snapshot_example()` and `build_metrics()` to use the same helper for finished steps.
  - Add `schema_quality` or `missing_signals` section in analyzer output summarizing absent optional fields such as `step_log`, `decision_context`, `outcome_state`, `evidence_windows`, `field_availability`.
  - Do not require step logs for basic metrics.

  **Must NOT do**: Do not break analysis of `train/context/HOLDOUT_BENCHMARK_20260425_0941.json` or dry-run/no-episode outputs. Do not require real benchmark execution for analyzer tests.

  **Recommended Agent Profile**:
  - Category: `quick` - Reason: parser/helper compatibility change.
  - Skills: [] - No special skill required.
  - Omitted: [`git-master`] - No commit requested by plan.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: [4, 6] | Blocked By: [1]

  **References**:
  - Analyzer reason parser: `train/analyze_holdout_benchmark.py:100-105`.
  - Analyzer metrics: `train/analyze_holdout_benchmark.py:161-184`.
  - Snapshot example: `train/analyze_holdout_benchmark.py:318-330`.
  - Failure classifier: `train/analyze_holdout_benchmark.py:351-426`.

  **Acceptance Criteria**:
  - [ ] `python -m py_compile train/analyze_holdout_benchmark.py`
  - [ ] Analyzer old-schema fixture with only `result` and `steps` produces status `OK` and nonzero episode count.
  - [ ] Analyzer new-schema fixture with `fail_reason` and `finished_steps` produces the same metrics.
  - [ ] Analyzer reports missing optional diagnostic fields as warnings/quality notes, not crashes.

  **QA Scenarios**:
  ```
  Scenario: Old schema still analyzes
    Tool: Bash
    Steps: Run analyzer against `train/context/HOLDOUT_BENCHMARK_20260425_0941.json` and write Markdown to `.sisyphus/evidence/task-3-old-schema.md`.
    Expected: Command exits 0; Markdown exists; no Python traceback.
    Evidence: .sisyphus/evidence/task-3-old-schema.txt

  Scenario: New schema fixture analyzes
    Tool: Bash
    Steps: Generate a tiny JSON fixture in `.sisyphus/evidence/task-3-new-schema-fixture.json` with one episode using `fail_reason`, `finished_steps`, and `evidence_windows`; run analyzer on it.
    Expected: Command exits 0 and reports `episode_count=1` or equivalent OK status.
    Evidence: .sisyphus/evidence/task-3-new-schema.txt
  ```

  **Commit**: NO | Message: `fix(analyzer): tolerate benchmark schema aliases` | Files: [train/analyze_holdout_benchmark.py]

- [x] 4. Add episode lifecycle summaries and deterministic failure evidence

  **What to do**: Add derived episode-level diagnostics in `code/agent_ppo/eval/holdout_benchmark.py`, then consume them in `train/analyze_holdout_benchmark.py`:
  - In benchmark, derive from collected `step_records`:
    - `charger_known_first_step`, `charger_known_final`, `known_charger_count_final`
    - `charger_arrived_count`, `charger_first_arrival_step`, `charger_arrival_steps`
    - `first_should_charge_step`, `attempted_charge_step_count`, `first_return_mode_step`
    - `min_battery`, `min_battery_step`, `min_charger_slack`, `max_negative_charger_slack`
    - `action_histogram`, `last_actions`, `repeat_action_max_streak`
    - `revisit_ratio`, `max_revisit_count`, `unique_cells_visited`, `observed_ratio_final`
  - Add capped `final_window` with last 50 compact step rows.
  - Add `evidence_windows` with at most 11 rows each (center ±5): `first_low_slack_window`, `first_should_charge_window`, `first_missed_charge_window`, `first_loop_window`, `last_failure_window`.
  - In analyzer, use these fields before replay-only heuristics to classify: `charger_unknown`, `return_too_late`, `optimistic_route_budget`, `repeated_invalid_move`, `npc_or_path_blocked`, `late_battery_death`.

  **Must NOT do**: Do not duplicate full per-step JSONL in episode summary. Do not classify based on absent fields as if evidence exists; emit `insufficient_signal` where necessary.

  **Recommended Agent Profile**:
  - Category: `unspecified-low` - Reason: medium-complex derived metrics across emitter and analyzer.
  - Skills: [] - No special skill required.
  - Omitted: [`ultrabrain`] - Logic is detailed but not deeply ambiguous.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: [6] | Blocked By: [2, 3]

  **References**:
  - Current `step_records` collection: `code/agent_ppo/eval/holdout_benchmark.py:235` and append at `295`.
  - Current episode result dict: `code/agent_ppo/eval/holdout_benchmark.py:327-348`.
  - Analyzer failure classifier: `train/analyze_holdout_benchmark.py:351-426`.
  - Minimum AI schema: `train/context/LTSPPO_BENCHMARK_ADAPTATION_20260425.md:57-83`.
  - Linux inspiration: `origin/linux-LTSPPO-charge-constraint:code/agent_ppo/eval/benchmark.py` had `evidence_windows`, `reward_attribution`, `anomaly_summary`; port only compact concepts, not full schema.

  **Acceptance Criteria**:
  - [ ] `python -m py_compile code/agent_ppo/eval/holdout_benchmark.py train/analyze_holdout_benchmark.py`
  - [ ] Static read confirms `final_window` and `evidence_windows` are capped and derived from `step_records`.
  - [ ] Analyzer classifies a synthetic battery-failure fixture with negative `min_charger_slack` as `optimistic_route_budget` or equivalent deterministic category.
  - [ ] Analyzer classifies a synthetic no-charger-discovered battery failure as `charger_unknown`.

  **QA Scenarios**:
  ```
  Scenario: Battery route-budget classification
    Tool: Bash
    Steps: Create `.sisyphus/evidence/task-4-route-budget-fixture.json` with one failed episode, `fail_reason=battery`, `min_charger_slack=-5`, and `evidence_windows`; run analyzer.
    Expected: Failure classification includes `optimistic_route_budget` with count 1 or clear reason referencing negative slack.
    Evidence: .sisyphus/evidence/task-4-route-budget.txt

  Scenario: Charger unknown classification
    Tool: Bash
    Steps: Create `.sisyphus/evidence/task-4-charger-unknown-fixture.json` with `fail_reason=battery`, `charger_arrived_count=0`, `charger_known_final=false`, no attempted charge; run analyzer.
    Expected: Failure classification includes `charger_unknown` with count 1.
    Evidence: .sisyphus/evidence/task-4-charger-unknown.txt
  ```

  **Commit**: NO | Message: `feat(benchmark): add lifecycle evidence windows` | Files: [code/agent_ppo/eval/holdout_benchmark.py, train/analyze_holdout_benchmark.py]

- [x] 5. Add optional AI summary and reward-attribution-lite without requiring LTSPPO-only fields

  **What to do**: Add a compact, optional AI-facing summary from data already captured:
  - In `code/agent_ppo/eval/holdout_benchmark.py`, create helper `_build_ai_summary(snapshot)` that returns: `schema_version`, `timestamp`, `checkpoint`, `overall`, `per_map`, `top_failure_modes`, `missing_signals`, `example_evidence_windows`, and `recommended_next_analysis`.
  - Write `ai_summary.json` in the same session dir as `result.json`. Do not write to `train/context/`.
  - Add `reward_attribution_lite` per episode using scalar reward trends and optional reward dict if `agent.last_reward` becomes dict-like. If reward is scalar, emit `{"available": false, "reason": "scalar_reward_only", "total_reward": ...}`.
  - Add `anomaly_summary_lite` per episode with rates derivable from step records: `low_slack_rate`, `no_clean_step_rate`, `revisit_rate`, `planner_policy_mismatch_rate`, `npc_near_rate`, `loop_suspect_rate`.
  - In analyzer Markdown, include `AI Diagnostic Quality` section with missing signals and which failure classes are reliable/unreliable for the input.

  **Must NOT do**: Do not require `mode_prob`, `target_prob`, `route_anchor_prob`, or full LTSPPO reward components. Do not append to global `eval_results.json` unless an explicit env var such as `KAIWU_BENCHMARK_EXPORT_AI_SUMMARY=1` is present; default should avoid new global mutable aggregate artifacts.

  **Recommended Agent Profile**:
  - Category: `unspecified-low` - Reason: additive summarization helpers and analyzer markdown output.
  - Skills: [] - No special skill required.
  - Omitted: [`librarian`] - No external library docs needed.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: [6] | Blocked By: [2]

  **References**:
  - Current snapshot creation: `code/agent_ppo/eval/holdout_benchmark.py:150-167`.
  - Current result writing: `code/agent_ppo/eval/holdout_benchmark.py:164-167`.
  - Current analyzer markdown writer: inspect `train/analyze_holdout_benchmark.py` lower half for output Markdown function before editing.
  - Linux concept reference: `origin/linux-LTSPPO-charge-constraint:code/agent_ppo/eval/benchmark.py` `_build_ai_summary` and `reward_attribution` helpers.

  **Acceptance Criteria**:
  - [ ] `python -m py_compile code/agent_ppo/eval/holdout_benchmark.py train/analyze_holdout_benchmark.py`
  - [ ] Static read confirms `ai_summary.json` is written under `session_dir`, not `train/context/`.
  - [ ] Static read confirms `reward_attribution_lite` handles scalar reward with `available=false`.
  - [ ] Analyzer Markdown includes an `AI Diagnostic Quality` or equivalent section.

  **QA Scenarios**:
  ```
  Scenario: AI summary helper compiles and is session-local
    Tool: Bash
    Steps: Run py_compile and static assertions for `_build_ai_summary`, `ai_summary.json`, `reward_attribution_lite`, and `session_dir / "ai_summary.json"` in benchmark source.
    Expected: PASS.
    Evidence: .sisyphus/evidence/task-5-ai-summary-static.txt

  Scenario: Analyzer quality section appears
    Tool: Bash
    Steps: Run analyzer against a synthetic fixture with missing optional fields and write Markdown.
    Expected: Markdown contains diagnostic quality/missing signal warnings; command exits 0.
    Evidence: .sisyphus/evidence/task-5-analyzer-quality.md
  ```

  **Commit**: NO | Message: `feat(benchmark): add AI diagnostic summary` | Files: [code/agent_ppo/eval/holdout_benchmark.py, train/analyze_holdout_benchmark.py]

- [x] 6. Run compile, analyzer fixtures, dry-run, and record lightweight changelog

  **What to do**: Execute final task-level QA after Tasks 1-5:
  - Compile changed Python files.
  - Run analyzer on old existing context JSON.
  - Run analyzer on synthetic new-schema fixtures created under `.sisyphus/evidence/`.
  - Run `train/run_holdout_benchmark.py --dry-run` to verify wrapper contract remains intact.
  - Append one line to `train/context/CHANGELOG.md` in required format: `YYYY-MM-DD HH:MM | ...`.
  - If QA reveals stale training/model artifacts in git status, do not touch them; report as pre-existing/unrelated unless this task created them.

  **Must NOT do**: Do not run the real Docker benchmark unless explicitly requested. Do not commit. Do not move raw logs into `train/context/`.

  **Recommended Agent Profile**:
  - Category: `quick` - Reason: verification and changelog only.
  - Skills: [] - No special skill required.
  - Omitted: [`git-master`] - No commit requested.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: [Final Verification Wave] | Blocked By: [3, 4, 5]

  **References**:
  - Changelog rule: `train/context/AGENTS.md` - append small changes to `CHANGELOG.md` as `YYYY-MM-DD HH:MM | one-line summary`.
  - Existing old-schema JSON: `train/context/HOLDOUT_BENCHMARK_20260425_0941.json`.
  - Dry-run command: `python train/run_holdout_benchmark.py --dry-run --maps 4,7 --episodes-per-map 1 --checkpoint code/model.ckpt-resume.pkl --output .sisyphus/evidence/ai-log-dryrun.json`.

  **Acceptance Criteria**:
  - [ ] `python -m py_compile code/agent_ppo/eval/holdout_benchmark.py train/analyze_holdout_benchmark.py`
  - [ ] Analyzer old-schema command exits 0.
  - [ ] Analyzer synthetic new-schema commands exit 0.
  - [ ] Dry-run benchmark command exits 0 and writes `.sisyphus/evidence/ai-log-dryrun.json`.
  - [ ] `train/context/CHANGELOG.md` has exactly one new relevant line.

  **QA Scenarios**:
  ```
  Scenario: Full no-Docker verification
    Tool: Bash
    Steps: Run compile, old-schema analyzer, synthetic fixture analyzers, and dry-run benchmark command from repo root.
    Expected: All commands exit 0; evidence files exist under `.sisyphus/evidence/`.
    Evidence: .sisyphus/evidence/task-6-full-qa.txt

  Scenario: Scope guard check
    Tool: Bash
    Steps: Run `git status --short` and inspect changed paths.
    Expected: Only planned source/analyzer/changelog/evidence changes are attributable; bulky raw logs/checkpoints are not newly added.
    Evidence: .sisyphus/evidence/task-6-scope-git-status.txt
  ```

  **Commit**: NO | Message: `test(benchmark): verify AI diagnostic logging` | Files: [train/context/CHANGELOG.md, .sisyphus/evidence/*]

## Final Verification Wave (MANDATORY — after ALL implementation tasks)
> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.** Rejection or user feedback -> fix -> re-run -> present again -> wait for okay.
- [x] F1. Plan Compliance Audit — oracle
- [x] F2. Code Quality Review — unspecified-high
- [x] F3. Real Manual QA — unspecified-high (+ no Playwright; CLI-only project path)
- [x] F4. Scope Fidelity Check — deep
## Commit Strategy
- Do not commit unless user explicitly requests it after implementation.
- If commits are requested later, use small atomic commits:
  1. `feat(benchmark): add diagnostic schema aliases`
  2. `feat(benchmark): add AI diagnostic step telemetry`
  3. `fix(analyzer): classify benchmark diagnostic schema`
  4. `test(benchmark): verify diagnostic log compatibility`
- Do not include bulky runtime artifacts, checkpoints, raw `code/eval_logs/`, or Docker outputs.

## Success Criteria
- Existing holdout benchmark outputs remain parseable.
- New outputs include enough evidence for AI to explain whether a failure is likely charger unknown, late return, route-budget optimism, NPC/path block, repeated/stuck loop, low-value revisit, planner-policy divergence, or map imbalance.
- Analyzer produces explicit diagnostic quality/missing-signal information instead of false certainty.
- No model behavior, training behavior, reward shaping, or checkpoint behavior changes are introduced.

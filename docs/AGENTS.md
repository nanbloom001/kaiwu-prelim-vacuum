# Repository Guidelines

## Project Structure & Module Organization
This repository is a Python reinforcement-learning project for the Robot Vacuum task.

- `train_test.py`: main local training/smoke-test entry.
- `agent_ppo/`: primary PPO implementation (`agent.py`, `algorithm/`, `feature/`, `model/`, `workflow/`, `conf/`).
- `agent_diy/`: DIY/template agent with the same module layout as `agent_ppo/`.
- `conf/`: global app/algo TOML config (`app_conf_robot_vacuum.toml`, `algo_conf_robot_vacuum.toml`, `configure_app.toml`).
- `docs/`: design notes and execution plans.

Keep new logic inside the relevant agent package and keep config updates close to the module that consumes them.

## Build, Test, and Development Commands
Run commands from repository root (`E:\competition\26fwwb`).

- `python train_test.py`: run the training smoke test for the algorithm selected in `train_test.py` (`ppo` or `diy`).
- `python -m compileall agent_ppo agent_diy`: quick syntax check before commit.
- `rg "pattern" agent_ppo agent_diy conf`: fast code/config search.

Before running, confirm `algorithm_name` in `train_test.py` matches your target implementation.

## Coding Style & Naming Conventions
Follow existing Python style in this repo:

- 4-space indentation, UTF-8 files, module-level imports at top.
- Class names: `PascalCase` (for example, `Config`, `Model`).
- Functions/variables: `snake_case`.
- Constants/config fields: `UPPER_CASE` (for example, `ACTION_NUM`, `GAMMA`).
- Keep bilingual comments only when needed; prefer concise, implementation-focused notes.

No formatter/linter config is committed here; keep diffs small and consistent with surrounding code.

## Testing Guidelines
There is no dedicated unit-test suite in this repository currently. Use `python train_test.py` as the required regression check for behavior changes, especially after edits in:

- `agent_ppo/feature/preprocessor.py`
- `agent_ppo/model/model.py`
- `agent_ppo/conf/conf.py`

When adding tests later, place them under a top-level `tests/` directory and use `test_*.py` naming.

## Commit & Pull Request Guidelines
Git history is not available in this workspace snapshot, so follow a clear, conventional format:

- Commit messages: imperative, scoped, concise (for example, `ppo: fix observation dimension mismatch`).
- PRs should include: purpose, key file changes, how to run (`python train_test.py`), and before/after metrics or logs for training-impacting changes.
- Link issue/task IDs when available and include screenshots only for visualization/report updates in `docs/`.

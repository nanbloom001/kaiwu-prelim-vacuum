# Contributing

Thanks for your interest in contributing to kaiwuFinal!

## Development Workflow

1. Fork the repository and create a feature branch:
   `git checkout -b feat/your-feature`
2. Make your changes. Keep commits small and focused.
3. Run the syntax check before committing:
   ```bash
   python -m py_compile $(git ls-files '*.py')
   ```
4. Commit with a clear message (prefix: `feat:`, `fix:`, `docs:`, `refactor:`,
   `chore:`), then open a pull request against `main`.

## What Not To Commit

- Model checkpoints (`*.pkl`, `*.meta.json`) — see the model policy in
  `README.md`.
- Logs, backups (`*.bak`), archives (`*.zip`, `*.tar*`), binary artifacts.
- Environment files (`.env`) or any secrets. If you find a leaked secret,
  report it per `SECURITY.md` — do not post it in issues.

## Code Style

- Python 3, PEP 8. Keep functions documented with concise docstrings
  (English or Chinese).
- Do not break the Kaiwu platform contract: `code/` directory layout,
  `kaiwu.json`, `agent_<algorithm>` naming, and the `train_test.py` entry
  must remain compatible with the platform.

## Branching

- `main` is the default branch and must always stay releasable.
- Work in `feat/*` or `fix/*` branches and merge via pull request.

## Licensing

By contributing, you agree that your contributions are licensed under the
MIT License (see `LICENSE`). Files derived from Kaiwu platform templates keep
their original headers — see `NOTICE.md`.

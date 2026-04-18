Model archive: v6-geo-bestmodel-1051

Source checkpoint
- Original file: /home/user/TcKaiwuFinal/code/session_best/20260417-002439/best_model.pkl
- Archived as: model.ckpt-resume.pkl

Why this checkpoint was selected
- This checkpoint was selected from the current fresh-run geometry-aware training without stopping the ongoing run.
- It was chosen as the primary resume candidate because it represents the latest stable session-level best model rather than a single high-variance peak episode.
- Session summary at selection time:
  - best_robust_score: 3266.05
  - best_avg_score: 1154.65
  - updated_at: 2026-04-17 09:59:17
  - episode_cnt: 1051

Selection rationale
- Compared with earlier peak snapshots, this checkpoint is more suitable for continued training because it comes from a later and more mature part of training.
- Log-based neighborhood analysis showed that the surrounding training window was stronger and steadier than early high-score peaks.
- This point is intended as the main "continue training" candidate, not as a fixed benchmark winner.

Observed trend around this checkpoint
- Training stage: eval_hard
- Recent windows contained many broad_eval wins with high clean score, including 1300+ to 1600+ runs.
- The strategy already demonstrates strong ceiling performance in harder distributions.

Known remaining issue
- Battery fail is still the dominant failure mode.
- Many failure trajectories still enter return mode with negative slack, which means return timing and return execution efficiency are not fully solved.

Recommended usage
- Primary candidate for the next training resume / warm-start experiment.
- Recommended when the goal is stable continuation rather than selecting the single most aggressive peak checkpoint.

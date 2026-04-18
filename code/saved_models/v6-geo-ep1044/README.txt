Model archive: v6-geo-ep1044

Source checkpoint
- Original file: /home/user/TcKaiwuFinal/code/resume_snapshots/resume-time-20260417-100046.pkl
- Archived as: model.ckpt-resume.pkl

Why this checkpoint was selected
- This checkpoint was selected from the current fresh-run geometry-aware training without stopping the ongoing run.
- It corresponds to the time-triggered snapshot taken immediately after:
  - episode 1044
  - profile broad_eval
  - clean_score 1693
  - result WIN
- It was chosen as a high-value peak snapshot that is still close to the current mature training regime.

Selection rationale
- This is not simply the highest single score in the whole run.
- It was kept because the local training window around episode 1044 was stronger and more stable than earlier peak checkpoints.
- Local window analysis around episode 1044 showed:
  - window size: 36 episodes
  - local win rate: 66.7%
  - local average clean score: 921.8
- In the same neighborhood, the death density was lower than earlier peak regions:
  - deaths in +/-20 window: 12
  - battery: 11
  - collision: 1

Why it is useful
- This checkpoint is a good "aggressive but still trainable" resume candidate.
- It preserves a high-performing broad_eval behavior pattern while staying in a later, more representative part of training than early peak snapshots.

Known remaining issue
- Battery fail is still the primary remaining weakness.
- Even around this checkpoint, some broad_eval failures still die inside return mode with already-negative slack.

Recommended usage
- Secondary candidate for the next training resume / warm-start experiment.
- Recommended as a comparison point against the more conservative session-level best model.

Model archive: v6-ltsppo-ep188

Source checkpoint
- Original file: /home/user/TcKaiwuFinal/code/session_best/20260416-192132/best-ep000188-score00979.pkl
- Archived as: model.ckpt-resume.pkl

Why this checkpoint was selected
- This checkpoint was manually benchmarked against two other LTSPPO candidates from the same training session:
  - best-ep000202-score01110.pkl
  - best_model.pkl
- Under the same fixed benchmark setting, ep188 produced the best overall balance of win rate, clean score, and harder-round stability.
- It outperformed the previously archived LTSPPO baseline v6-ltsppo-ep72.

Benchmark command
- bash train/run_benchmark_parallel.sh ../code/session_best/20260416-192132/best-ep000188-score00979.pkl --workers 4 --envs-per-worker 10 --max-wait 1800

Benchmark result summary
- Overall:
  - Win Rate: 47.5%
  - Avg Clean Score: 634.0
  - Wins: 19/40
- Per round:
  - round_1: WR 70%, CS 625.1, BatteryFail 20%, CollisionFail 10%
  - round_2: WR 60%, CS 747.3, BatteryFail 30%, CollisionFail 10%
  - round_3: WR 40%, CS 544.0, BatteryFail 50%, CollisionFail 10%
  - round_4: WR 20%, CS 619.5, BatteryFail 70%, CollisionFail 10%

Comparison against other candidates in the same batch
- ep202: WR 45.0%, Avg CS 610.1, 18/40
- best_model: WR 40.0%, Avg CS 618.2, 16/40
- previous archived LTSPPO baseline (v6-ltsppo-ep72): WR 30.0%, Avg CS 460.0, 12/40

Recommended usage
- Primary resume candidate for the next LTSPPO round.
- Chosen because it appears stronger than the earlier LTSPPO baseline, but still has headroom for further training.

Known remaining issue
- Battery fail is still the dominant failure mode, especially in harder/longer rounds.
- The next round should focus on earlier return timing, stronger/longer teacher support, and better charger-slack signal alignment.

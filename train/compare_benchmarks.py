#!/usr/bin/env python3
"""
Compare evaluation benchmark results across checkpoints.

Usage:
    python compare_benchmarks.py                    # show all benchmarks
    python compare_benchmarks.py latest             # show latest
    python compare_benchmarks.py 0 1               # compare benchmark 0 vs 1
    python compare_benchmarks.py path/to/file.json  # read specific file
"""

import json
import sys
from pathlib import Path


def load_results(path: str | None = None) -> dict:
    candidates = []
    if path:
        candidates.append(Path(path))
    candidates.extend([
        Path(__file__).resolve().parent / "eval_results.json",
        Path("/workspace/code/eval_results.json"),
        Path(__file__).resolve().parents[1] / "code" / "eval_results.json",
    ])
    for p in candidates:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    print("No eval_results.json found. Run 'bash train/run_benchmark.sh' first.")
    sys.exit(1)


def fmt_pct(v: float) -> str:
    return f"{v:.0%}" if v >= 0 else " N/A"


def fmt_num(v: float) -> str:
    return f"{v:.0f}" if isinstance(v, (int, float)) else " N/A"


ROUND_ORDER = ["round_1", "round_2", "round_3", "round_4"]
ROUND_DISPLAY = {
    "round_1": "R1: 4C/3R/1000",
    "round_2": "R2: 3C/4R/1200",
    "round_3": "R3: 2C/4R/1600",
    "round_4": "R4: 2C/4R/2000",
}


def _get_per_round(bench: dict) -> dict:
    """Get per-round metrics, supporting both old 'scenarios' and new 'per_round' keys."""
    return bench.get("per_round", bench.get("scenarios", {}))


def print_benchmark(idx: int, bench: dict):
    overall = bench.get("overall", {})
    per_round = _get_per_round(bench)
    rounds_desc = bench.get("rounds", {})

    print(f"\n{'='*75}")
    print(f"  Benchmark #{idx}")
    print(f"  {bench.get('timestamp', '?')}  |  {bench.get('elapsed_seconds', '?')}s")
    print(f"  checkpoint={bench.get('checkpoint', '?')}  git={bench.get('git_commit', '?')}")
    print(f"{'='*75}")

    # Per-round details
    print(f"  {'Round':<18} {'WR':>6} {'CS':>8} {'Steps':>8} {'BatF':>6} {'ColF':>6} {'Eps':>5}")
    print(f"  {'-'*18} {'-'*6} {'-'*8} {'-'*8} {'-'*6} {'-'*6} {'-'*5}")
    for name in ROUND_ORDER:
        if name in per_round:
            m = per_round[name]
            label = ROUND_DISPLAY.get(name, name)
            print(f"  {label:<18} {fmt_pct(m.get('win_rate', 0)):>6} "
                  f"{fmt_num(m.get('avg_clean_score', 0)):>8} "
                  f"{fmt_num(m.get('avg_steps', 0)):>8} "
                  f"{fmt_pct(m.get('battery_fail_rate', 0)):>6} "
                  f"{fmt_pct(m.get('collision_fail_rate', 0)):>6} "
                  f"{m.get('episode_count', 0):>5}")

    print(f"  {'OVERALL':<18} {fmt_pct(overall.get('win_rate', 0)):>6} "
          f"{fmt_num(overall.get('avg_clean_score', 0)):>8} "
          f"{fmt_num(overall.get('avg_steps', 0)):>8} "
          f"{'':>6} {'':>6} "
          f"{overall.get('episode_count', 0):>5}")
    print(f"  Wins: {overall.get('win_episode_count', '?')}/{overall.get('episode_count', '?')}")


def compare_benchmarks(bench_a: dict, bench_b: dict):
    per_round_a = _get_per_round(bench_a)
    per_round_b = _get_per_round(bench_b)
    overall_a = bench_a.get("overall", {})
    overall_b = bench_b.get("overall", {})

    print(f"\n{'='*80}")
    print(f"  A: {bench_a.get('timestamp', '?')}  ckpt={bench_a.get('checkpoint', '?')}  git={bench_a.get('git_commit', '?')}")
    print(f"  B: {bench_b.get('timestamp', '?')}  ckpt={bench_b.get('checkpoint', '?')}  git={bench_b.get('git_commit', '?')}")
    print(f"{'='*80}")

    def delta(a, b):
        d = b - a
        if abs(d) < 0.0005:
            return "  =  "
        sign = "+" if d > 0 else ""
        return f"{sign}{d:.3f}"

    print(f"  {'Round':<18} {'Metric':<8} {'A':>8} {'B':>8} {'Delta':>8}")
    print(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for name in ROUND_ORDER:
        ma = per_round_a.get(name, {})
        mb = per_round_b.get(name, {})
        if not ma or not mb:
            continue
        label = ROUND_DISPLAY.get(name, name)
        for metric, mlabel in [("win_rate", "WR"), ("avg_clean_score", "CS"),
                                ("battery_fail_rate", "BatF"), ("collision_fail_rate", "ColF")]:
            va = ma.get(metric, 0)
            vb = mb.get(metric, 0)
            print(f"  {label:<18} {mlabel:<8} {va:>8.3f} {vb:>8.3f} {delta(va, vb):>8}")

    for metric, label in [("win_rate", "WR"), ("avg_clean_score", "CS"), ("avg_steps", "Steps")]:
        va = overall_a.get(metric, 0)
        vb = overall_b.get(metric, 0)
        print(f"  {'OVERALL':<18} {label:<8} {va:>8.3f} {vb:>8.3f} {delta(va, vb):>8}")

    wr_d = overall_b.get("win_rate", 0) - overall_a.get("win_rate", 0)
    cs_d = overall_b.get("avg_clean_score", 0) - overall_a.get("avg_clean_score", 0)
    if wr_d > 0.05 and cs_d > 20:
        verdict = "IMPROVED"
    elif wr_d < -0.05 and cs_d < -20:
        verdict = "REGRESSED"
    else:
        verdict = "MIXED/STABLE"
    print(f"\n  Verdict: {verdict}")


def main():
    results = load_results(sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] not in ("latest",) else None)
    benchmarks = results.get("benchmarks", results.get("snapshots", []))

    if not benchmarks:
        print("No benchmarks found in file.")
        return

    if len(sys.argv) >= 3 and sys.argv[1] != "latest":
        try:
            idx_a, idx_b = int(sys.argv[1]), int(sys.argv[2])
        except ValueError:
            print("Usage: compare_benchmarks.py [idx_a idx_b | latest | path]")
            return
        if idx_a >= len(benchmarks) or idx_b >= len(benchmarks):
            print(f"Only {len(benchmarks)} benchmarks (indices 0-{len(benchmarks)-1})")
            return
        print_benchmark(idx_a, benchmarks[idx_a])
        print_benchmark(idx_b, benchmarks[idx_b])
        compare_benchmarks(benchmarks[idx_a], benchmarks[idx_b])
        return

    print(f"\nFound {len(benchmarks)} benchmark(s):")
    for i, b in enumerate(benchmarks):
        o = b.get("overall", {})
        print(f"  [{i}] {b.get('timestamp', '?')}  "
              f"WR={fmt_pct(o.get('win_rate', 0))}  CS={fmt_num(o.get('avg_clean_score', 0))}  "
              f"wins={o.get('win_episode_count', '?')}/{o.get('episode_count', '?')}  "
              f"ckpt={b.get('checkpoint', '?')}")

    if len(benchmarks) >= 1:
        print_benchmark(len(benchmarks) - 1, benchmarks[-1])
    if len(benchmarks) >= 2:
        compare_benchmarks(benchmarks[-2], benchmarks[-1])


if __name__ == "__main__":
    main()

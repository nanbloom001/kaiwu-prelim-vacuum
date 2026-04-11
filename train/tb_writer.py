"""Convert training logs to TensorBoard events - reads raw GAMEOVER records for full resolution."""
import os
import re
import time
from pathlib import Path

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

BASE = Path("D:/TcKaiwuFinal")
AISRV_LOG_DIR = BASE / "train" / "log" / "aisrv"
TB_DIR = BASE / "train" / "tb_logs"


def _make_scalar(tag, value, step, wall_time):
    return Event(
        wall_time=wall_time,
        step=step,
        summary=Summary(value=[Summary.Value(tag=tag, simple_value=float(value))])
    )


def _parse_time(s):
    try:
        from datetime import datetime
        return datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S").timestamp()
    except Exception:
        return time.time()


def _parse_gameover_logs():
    """Parse all GAMEOVER records from aisrv logs, including map_id."""
    records = []
    for logf in AISRV_LOG_DIR.glob("aisrv_kaiwu_rl_helper_pid*_log_*.log"):
        with open(logf, "r", encoding="utf-8") as f:
            for line in f:
                if "GAMEOVER" not in line:
                    continue
                m = re.search(
                    r'"time": "([^"]+)".*ep:(\d+) steps:(\d+) result:(\w+).*clean_score:([\d.]+)',
                    line,
                )
                if not m:
                    continue
                # Extract map_id from the new format: map:X
                map_m = re.search(r'map:(\S+)', line)
                map_id = map_m.group(1) if map_m else None
                records.append({
                    "time": m.group(1),
                    "ep": int(m.group(2)),
                    "steps": int(m.group(3)),
                    "result": m.group(4),
                    "clean_score": float(m.group(5)),
                    "map_id": map_id,
                })
    records.sort(key=lambda x: x["ep"])
    return records


def _parse_training_metrics():
    """Parse training_metrics from aisrv logs."""
    ms = []
    for logf in AISRV_LOG_DIR.glob("aisrv_kaiwu_rl_helper_pid*_log_*.log"):
        with open(logf, "r", encoding="utf-8") as f:
            for line in f:
                if "training_metrics" not in line:
                    continue
                m = re.search(
                    r'"time": "([^"]+)".*train_global_step\': ?([\d.]+).*episode_cnt\': ?([\d.]+).*reward.*?([\d.-]+).*clean_score.*?([\d.]+).*charge_count.*?([\d.]+)',
                    line,
                )
                if m and float(m.group(2)) > 0:
                    ms.append({
                        "time": m.group(1)[:19],
                        "global_step": int(float(m.group(2))),
                        "episode_cnt": int(float(m.group(3))),
                        "reward": float(m.group(4)),
                        "clean_score_avg": float(m.group(5)),
                        "charge_count": float(m.group(6)),
                    })
    ms.sort(key=lambda x: x["time"])
    return ms


def convert():
    records = _parse_gameover_logs()
    if not records:
        print("No GAMEOVER records found in logs")
        return

    metrics = _parse_training_metrics()

    # Clean old events
    for old in TB_DIR.glob("events.out.tfevents.*"):
        old.unlink(missing_ok=True)

    os.makedirs(TB_DIR, exist_ok=True)
    writer = EventFileWriter(str(TB_DIR))

    # 1. Per-episode clean_score (full resolution)
    for r in records:
        wt = _parse_time(r["time"])
        writer.add_event(_make_scalar("clean_score", r["clean_score"], r["ep"], wt))

    # 2. Running max per episode
    running_max = 0
    for r in records:
        running_max = max(running_max, r["clean_score"])
        wt = _parse_time(r["time"])
        writer.add_event(_make_scalar("running_max", running_max, r["ep"], wt))

    # 3. Rolling average (window=10) per episode
    window = []
    for r in records:
        window.append(r["clean_score"])
        if len(window) > 10:
            window.pop(0)
        wt = _parse_time(r["time"])
        writer.add_event(_make_scalar("rolling_avg_10", sum(window) / len(window), r["ep"], wt))

    # 4. Rolling average (window=30) per episode
    window30 = []
    for r in records:
        window30.append(r["clean_score"])
        if len(window30) > 30:
            window30.pop(0)
        wt = _parse_time(r["time"])
        writer.add_event(_make_scalar("rolling_avg_30", sum(window30) / len(window30), r["ep"], wt))

    # 5. Win rate rolling (window=20) per episode
    win_window = []
    for r in records:
        win_window.append(1 if r["result"] == "WIN" else 0)
        if len(win_window) > 20:
            win_window.pop(0)
        wt = _parse_time(r["time"])
        writer.add_event(_make_scalar("win_rate_20", sum(win_window) / len(win_window) * 100, r["ep"], wt))

    # 6. Average steps per episode
    step_window = []
    for r in records:
        step_window.append(r["steps"])
        if len(step_window) > 20:
            step_window.pop(0)
        wt = _parse_time(r["time"])
        writer.add_event(_make_scalar("avg_steps_20", sum(step_window) / len(step_window), r["ep"], wt))

    # 7. Training metrics (reward, charge_count, clean_score_avg) by global_step
    for m in metrics:
        wt = _parse_time(m["time"])
        step = m["global_step"]
        writer.add_event(_make_scalar("metrics/reward", m["reward"], step, wt))
        writer.add_event(_make_scalar("metrics/charge_count", m["charge_count"], step, wt))
        writer.add_event(_make_scalar("metrics/clean_score_avg", m["clean_score_avg"], step, wt))

    # ---------------------------------------------------------------
    # 8. Per-map metrics + cross-map variance (generalization tracking)
    # ---------------------------------------------------------------
    import numpy as np

    has_map_data = any(r["map_id"] is not None for r in records)
    if has_map_data:
        # Group records by map_id
        map_groups: dict[str, list] = {}
        for r in records:
            mid = r["map_id"]
            if mid is None:
                continue
            map_groups.setdefault(mid, []).append(r)

        # 8a. Per-map clean_score and rolling average
        for mid, mrecs in sorted(map_groups.items()):
            tag_prefix = f"map_{mid}"
            for r in mrecs:
                wt = _parse_time(r["time"])
                writer.add_event(_make_scalar(f"{tag_prefix}/clean_score", r["clean_score"], r["ep"], wt))
            # Rolling avg (window=10) per map
            map_window = []
            for r in mrecs:
                map_window.append(r["clean_score"])
                if len(map_window) > 10:
                    map_window.pop(0)
                wt = _parse_time(r["time"])
                writer.add_event(
                    _make_scalar(f"{tag_prefix}/rolling_avg_10", sum(map_window) / len(map_window), r["ep"], wt)
                )

        # 8b. Cross-map variance metrics (sliding window of 20 episodes)
        # For each episode, compute variance of recent per-map averages
        map_score_windows: dict[str, list[float]] = {}
        for r in records:
            mid = r["map_id"]
            if mid is None:
                continue
            map_score_windows.setdefault(mid, []).append(r["clean_score"])
            if len(map_score_windows[mid]) > 15:
                map_score_windows[mid] = map_score_windows[mid][-15:]

            # Only compute when we have >= 3 maps with data
            map_avgs = {}
            for m, scores in map_score_windows.items():
                if len(scores) >= 3:
                    map_avgs[m] = sum(scores) / len(scores)

            if len(map_avgs) >= 3:
                avg_vals = list(map_avgs.values())
                wt = _parse_time(r["time"])
                writer.add_event(_make_scalar("cross_map/variance", float(np.std(avg_vals)), r["ep"], wt))
                writer.add_event(_make_scalar("cross_map/min_avg", min(avg_vals), r["ep"], wt))
                writer.add_event(_make_scalar("cross_map/spread", max(avg_vals) - min(avg_vals), r["ep"], wt))
                writer.add_event(_make_scalar("cross_map/mean_avg", sum(avg_vals) / len(avg_vals), r["ep"], wt))

    writer.close()
    print(f"TensorBoard: {len(records)} episodes, {len(metrics)} metrics -> {TB_DIR}")


if __name__ == "__main__":
    convert()

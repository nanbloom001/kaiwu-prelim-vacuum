import re

# Parse both log files for complete metrics
files = [
    'D:/TcKaiwuFinal/train/log/aisrv/aisrv_kaiwu_rl_helper_pid298_log_2026-04-13-02.log',
    'D:/TcKaiwuFinal/train/log/aisrv/aisrv_kaiwu_rl_helper_pid302_log_2026-04-13-02.log',
]

data = []
seen_steps = set()

for fpath in files:
    for line in open(fpath):
        if 'training_metrics' not in line:
            continue
        m = re.search(
            r"'train_global_step': (\d+\.?\d*).*?'episode_cnt': (\d+\.?\d*).*?"
            r"'clean_score': (\d+\.?\d*).*?'max_steps': (\d+\.?\d*).*?"
            r"'finished_steps': (\d+\.?\d*).*?'charge_count': (\d+\.?\d*).*?"
            r"'remaining_charge': (\d+\.?\d*).*?'entropy_loss': (\d+\.?\d*)",
            line
        )
        if m:
            step = int(float(m.group(1)))
            if step in seen_steps:
                continue
            seen_steps.add(step)
            data.append(dict(
                step=step,
                ep=int(float(m.group(2))),
                clean=float(m.group(3)),
                maxs=float(m.group(4)),
                fins=float(m.group(5)),
                chgc=float(m.group(6)),
                remc=float(m.group(7)),
                ent=float(m.group(8))
            ))

data.sort(key=lambda d: d['step'])

# Calculate CPS and comp rate
for d in data:
    d['cps'] = d['clean'] / max(d['fins'], 1)
    d['comp'] = d['fins'] / max(d['maxs'], 1)

# Print every entry
print(f"{'#':>3} {'Step':>6} {'EP':>5} {'CPS':>6} {'Clean':>7} {'FinSt':>7} {'MaxSt':>7} {'CompRate':>8} {'ChgCnt':>6} {'RemChg':>6} {'Entropy':>7}")
print("-" * 85)
for i, d in enumerate(data):
    print(f"{i+1:>3} {d['step']:>6} {d['ep']:>5} {d['cps']:>6.3f} {d['clean']:>7.0f} {d['fins']:>7.0f} {d['maxs']:>7.0f} {d['comp']:>8.1%} {d['chgc']:>6.1f} {d['remc']:>6.0f} {d['ent']:>7.3f}")

# Find best checkpoints by different criteria
print("\n=== BEST CHECKPOINTS ===")

# 1. Best CPS with comp >= 95%
valid = [d for d in data if d['comp'] >= 0.95]
if valid:
    best_cps = max(valid, key=lambda d: d['cps'])
    print(f"\nBest CPS (comp>=95%): step={best_cps['step']} CPS={best_cps['cps']:.3f} comp={best_cps['comp']:.1%} clean={best_cps['clean']:.0f} chg={best_cps['chgc']:.1f} ent={best_cps['ent']:.3f}")

# 2. Best combined score: CPS * comp_rate
for d in data:
    d['combined'] = d['cps'] * d['comp']
best_combined = max(data, key=lambda d: d['combined'])
print(f"Best CPS*CompRate:   step={best_combined['step']} CPS={best_combined['cps']:.3f} comp={best_combined['comp']:.1%} combined={best_combined['combined']:.3f} clean={best_combined['clean']:.0f} chg={best_combined['chgc']:.1f} ent={best_combined['ent']:.3f}")

# 3. Best CPS overall
best_cps_all = max(data, key=lambda d: d['cps'])
print(f"Best CPS (any comp): step={best_cps_all['step']} CPS={best_cps_all['cps']:.3f} comp={best_cps_all['comp']:.1%} clean={best_cps_all['clean']:.0f} chg={best_cps_all['chgc']:.1f} ent={best_cps_all['ent']:.3f}")

# 4. Best comp rate (highest clean score among 100% comp)
perfect = [d for d in data if d['comp'] >= 0.99]
if perfect:
    best_perfect = max(perfect, key=lambda d: d['clean'])
    print(f"Best 100% comp:      step={best_perfect['step']} CPS={best_perfect['cps']:.3f} comp={best_perfect['comp']:.1%} clean={best_perfect['clean']:.0f} chg={best_perfect['chgc']:.1f} ent={best_perfect['ent']:.3f}")

# 5. Moving average (smoothed top 5)
if len(data) >= 5:
    for i in range(len(data) - 4):
        window = data[i:i+5]
        avg_cps = sum(d['cps'] for d in window) / 5
        avg_comp = sum(d['comp'] for d in window) / 5
        avg_combined = avg_cps * avg_comp
        data[i]['avg5_combined'] = avg_combined
        data[i]['avg5_cps'] = avg_cps
        data[i]['avg5_comp'] = avg_comp

    smoothed = data[:-4]
    best_smooth = max(smoothed, key=lambda d: d.get('avg5_combined', 0))
    print(f"Best 5-pt avg:       step={best_smooth['step']} avgCPS={best_smooth['avg5_cps']:.3f} avgComp={best_smooth['avg5_comp']:.1%} avgCombined={best_smooth['avg5_combined']:.3f} ent={best_smooth['ent']:.3f}")

# Print entropy collapse phases
print("\n=== ENTROPY PHASES ===")
phases = [
    (2.0, 1.5, "Healthy exploration"),
    (1.5, 1.0, "Declining"),
    (1.0, 0.5, "Collapsing"),
    (0.5, 0.0, "Collapsed"),
]
for hi, lo, label in phases:
    entries = [d for d in data if lo <= d['ent'] < hi]
    if entries:
        steps = f"{entries[0]['step']}-{entries[-1]['step']}"
        avg_cps = sum(d['cps'] for d in entries) / len(entries)
        avg_comp = sum(d['comp'] for d in entries) / len(entries)
        avg_chg = sum(d['chgc'] for d in entries) / len(entries)
        print(f"  [{hi:.1f}-{lo:.1f}] steps {steps}: avgCPS={avg_cps:.3f} avgComp={avg_comp:.1%} avgChg={avg_chg:.1f} ({label})")

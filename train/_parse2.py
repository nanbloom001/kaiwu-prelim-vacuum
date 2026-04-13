import sys, os

# Use the Windows temp path
tmppath = os.environ.get('TEMP', 'C:/Users/lenovo/AppData/Local/Temp')
raw_file = os.path.join(tmppath, 'metrics_raw.txt')

fields = []
for line in open(raw_file):
    line = line.strip()
    if ':' in line:
        key, val = line.rsplit(':', 1)
        key = key.strip().replace("'", "").replace(" ", "")
        val = float(val.strip())
        fields.append((key, val))

# Group into entries of 8
data = []
for i in range(0, len(fields) - 7, 8):
    entry = {}
    for j in range(8):
        k, v = fields[i + j]
        entry[k] = v
    if 'train_global_step' in entry:
        s = int(entry['train_global_step'])
        entry['step'] = s
        entry['cps'] = entry.get('clean_score', 0) / max(entry.get('finished_steps', 1), 1)
        entry['comp'] = entry.get('finished_steps', 0) / max(entry.get('max_steps', 1), 1)
        data.append(entry)

# Remove duplicates by step
seen = set()
unique = []
for d in data:
    if d['step'] not in seen:
        seen.add(d['step'])
        unique.append(d)
data = sorted(unique, key=lambda d: d['step'])

print(f"Total entries: {len(data)}")
print(f"{'#':>3} {'Step':>6} {'EP':>5} {'CPS':>6} {'Clean':>7} {'FinSt':>7} {'MaxSt':>7} {'CompRate':>8} {'ChgCnt':>6} {'RemChg':>6} {'Entropy':>7}")
print("-" * 85)
for i, d in enumerate(data):
    print(f"{i+1:>3} {d['step']:>6} {int(d.get('episode_cnt',0)):>5} {d['cps']:>6.3f} {d.get('clean_score',0):>7.0f} {d.get('finished_steps',0):>7.0f} {d.get('max_steps',0):>7.0f} {d['comp']:>8.1%} {d.get('charge_count',0):>6.1f} {d.get('remaining_charge',0):>6.0f} {d.get('entropy_loss',0):>7.3f}")

# Best checkpoints
print("\n=== BEST CHECKPOINTS ===")

# Best CPS with comp >= 95%
valid = [d for d in data if d['comp'] >= 0.95]
if valid:
    best_cps = max(valid, key=lambda d: d['cps'])
    print(f"Best CPS (comp>=95%): step={best_cps['step']} CPS={best_cps['cps']:.3f} comp={best_cps['comp']:.1%} clean={best_cps.get('clean_score',0):.0f} chg={best_cps.get('charge_count',0):.1f} ent={best_cps.get('entropy_loss',0):.3f}")

# Best combined = CPS * comp
for d in data:
    d['combined'] = d['cps'] * d['comp']
best_comb = max(data, key=lambda d: d['combined'])
print(f"Best CPS*CompRate:   step={best_comb['step']} CPS={best_comb['cps']:.3f} comp={best_comb['comp']:.1%} combined={best_comb['combined']:.3f} clean={best_comb.get('clean_score',0):.0f} ent={best_comb.get('entropy_loss',0):.3f}")

# Best CPS overall
best_all = max(data, key=lambda d: d['cps'])
print(f"Best CPS (any):      step={best_all['step']} CPS={best_all['cps']:.3f} comp={best_all['comp']:.1%} clean={best_all.get('clean_score',0):.0f} chg={best_all.get('charge_count',0):.1f} ent={best_all.get('entropy_loss',0):.3f}")

# Best clean score with 100% comp
perfect = [d for d in data if d['comp'] >= 0.99]
if perfect:
    best_p = max(perfect, key=lambda d: d.get('clean_score', 0))
    print(f"Best clean@100%comp: step={best_p['step']} CPS={best_p['cps']:.3f} comp={best_p['comp']:.1%} clean={best_p.get('clean_score',0):.0f} chg={best_p.get('charge_count',0):.1f} ent={best_p.get('entropy_loss',0):.3f}")

# 5-entry moving average best
if len(data) >= 5:
    best_avg = None
    best_avg_val = 0
    for i in range(len(data) - 4):
        w = data[i:i+5]
        avg_cps = sum(d['cps'] for d in w) / 5
        avg_comp = sum(d['comp'] for d in w) / 5
        val = avg_cps * avg_comp
        if val > best_avg_val:
            best_avg_val = val
            best_avg = (i, avg_cps, avg_comp, val)
    if best_avg:
        i, ac, acomp, aval = best_avg
        print(f"Best 5-pt window:    step~{data[i+2]['step']} avgCPS={ac:.3f} avgComp={acomp:.1%} avgComb={aval:.3f} ent={data[i+2].get('entropy_loss',0):.3f}")

# Entropy phases
print("\n=== ENTROPY PHASES ===")
for hi, lo, label in [(2.0, 1.5, "Healthy"), (1.5, 1.0, "Declining"), (1.0, 0.5, "Collapsing"), (0.5, 0.0, "Collapsed")]:
    ents = [d for d in data if lo <= d.get('entropy_loss', 2) < hi]
    if ents:
        steps = f"{ents[0]['step']}-{ents[-1]['step']}"
        ac = sum(d['cps'] for d in ents) / len(ents)
        acomp = sum(d['comp'] for d in ents) / len(ents)
        achg = sum(d.get('charge_count', 0) for d in ents) / len(ents)
        print(f"  [{hi:.1f}-{lo:.1f}] steps {steps}: avgCPS={ac:.3f} avgComp={acomp:.1%} avgChg={achg:.1f} ({label})")

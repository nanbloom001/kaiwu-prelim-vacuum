import re, sys

lines = open('D:/TcKaiwuFinal/train/cps_raw.txt').readlines()
data = []
for line in lines:
    m = re.search(r"'train_global_step': (\d+\.?\d*).*?'episode_cnt': (\d+\.?\d*).*?'clean_score': (\d+\.?\d*).*?'max_steps': (\d+\.?\d*).*?'finished_steps': (\d+\.?\d*).*?'charge_count': (\d+\.?\d*).*?'remaining_charge': (\d+\.?\d*).*?'entropy_loss': (\d+\.?\d*)", line)
    if m:
        data.append(dict(
            step=float(m.group(1)), ep=float(m.group(2)),
            clean=float(m.group(3)), maxs=float(m.group(4)),
            fins=float(m.group(5)), chgc=float(m.group(6)),
            remc=float(m.group(7)), ent=float(m.group(8))
        ))

print(f"Parsed {len(data)} entries\n")

# Print header
print(f"{'Step':>6} {'EP':>5} {'CPS':>6} {'Clean':>7} {'MaxSt':>7} {'FinSt':>7} {'CompRate':>8} {'ChgCnt':>6} {'RemChg':>6} {'Entropy':>7}")
print("-" * 78)

for d in data:
    cps = d['clean'] / max(d['fins'], 1)
    comp = d['fins'] / max(d['maxs'], 1)
    print(f"{int(d['step']):>6} {int(d['ep']):>5} {cps:>6.3f} {d['clean']:>7.0f} {d['maxs']:>7.0f} {d['fins']:>7.0f} {comp:>8.1%} {d['chgc']:>6.1f} {d['remc']:>6.0f} {d['ent']:>7.3f}")

# Summary
print("\n--- Trend Analysis ---")
if len(data) >= 2:
    first = data[0]
    mid = data[len(data)//2]
    last = data[-1]

    cps_first = first['clean'] / max(first['fins'], 1)
    cps_mid = mid['clean'] / max(mid['fins'], 1)
    cps_last = last['clean'] / max(last['fins'], 1)

    comp_first = first['fins'] / max(first['maxs'], 1)
    comp_mid = mid['fins'] / max(mid['maxs'], 1)
    comp_last = last['fins'] / max(last['maxs'], 1)

    print(f"CPS:        early={cps_first:.3f}  mid={cps_mid:.3f}  latest={cps_last:.3f}")
    print(f"CompRate:   early={comp_first:.1%}  mid={comp_mid:.1%}  latest={comp_last:.1%}")
    print(f"Entropy:    early={first['ent']:.3f}  mid={mid['ent']:.3f}  latest={last['ent']:.3f}")
    print(f"ChargeCnt:  early={first['chgc']:.1f}  mid={mid['chgc']:.1f}  latest={last['chgc']:.1f}")
    print(f"RemCharge:  early={first['remc']:.0f}  mid={mid['remc']:.0f}  latest={last['remc']:.0f}")

    # Check for incomplete episodes (fin < max means death)
    deaths_recent = [d for d in data[-8:] if d['fins'] < d['maxs'] - 10]
    print(f"\nRecent incomplete episodes (deaths): {len(deaths_recent)}/8")

    # charge_count trend
    chg_early = sum(d['chgc'] for d in data[:5]) / 5
    chg_late = sum(d['chgc'] for d in data[-5:]) / 5
    print(f"Charge count trend: early_avg={chg_early:.1f}  late_avg={chg_late:.1f}")

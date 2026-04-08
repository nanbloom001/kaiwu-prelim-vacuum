"""Training data collection script - outputs raw data to TRAINING_DATA.json"""
import json, re, math, subprocess
from datetime import datetime
from pathlib import Path

BASE = Path(r"D:/TcKaiwuFinal")
AISRV_LOG_DIR = BASE / "train" / "log" / "aisrv"

def parse_gameover_records():
    records = []
    for logf in AISRV_LOG_DIR.glob("aisrv_kaiwu_rl_helper_pid*_log_*.log"):
        with open(logf, 'r', encoding='utf-8') as f:
            for line in f:
                if 'GAMEOVER' not in line:
                    continue
                m = re.search(
                    r'"time": "([^"]+)".*ep:(\d+) steps:(\d+) result:(\w+).*clean_score:([\d.]+)',
                    line,
                )
                if m:
                    records.append({
                        'time': m.group(1),
                        'ep': int(m.group(2)),
                        'steps': int(m.group(3)),
                        'result': m.group(4),
                        'clean_score': float(m.group(5)),
                    })
    records.sort(key=lambda x: x['time'])
    return records

def parse_training_metrics():
    ms = []
    for logf in AISRV_LOG_DIR.glob("aisrv_kaiwu_rl_helper_pid*_log_*.log"):
        with open(logf, 'r', encoding='utf-8') as f:
            for line in f:
                if 'training_metrics' not in line:
                    continue
                m = re.search(
                    r'"time": "([^"]+)".*train_global_step\': ?([\d.]+).*episode_cnt\': ?([\d.]+).*reward.*?([\d.-]+).*clean_score.*?([\d.]+).*charge_count.*?([\d.]+)',
                    line,
                )
                if m and float(m.group(2)) > 0:
                    ms.append({
                        'time': m.group(1)[:19],
                        'global_step': int(float(m.group(2))),
                        'episode_cnt': int(float(m.group(3))),
                        'reward': float(m.group(4)),
                        'clean_score_avg': float(m.group(5)),
                        'charge_count': float(m.group(6)),
                    })
    ms.sort(key=lambda x: x['time'])
    return ms

def get_container_status():
    try:
        r = subprocess.run(
            ['docker', 'ps', '--filter', 'name=kaiwu-train', '--format', '{{.Names}} {{.Status}}'],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip().split('\n') if r.stdout.strip() else []
    except Exception:
        return []

def calc_stats(records):
    scores = [r['clean_score'] for r in records]
    steps = [r['steps'] for r in records]
    n = len(scores)
    avg = sum(scores) / n
    std = (sum((s - avg) ** 2 for s in scores) / n) ** 0.5
    return {
        'total': n,
        'avg': round(avg, 2),
        'max': max(scores),
        'min': min(scores),
        'median': sorted(scores)[n // 2],
        'std': round(std, 2),
        'win_count': sum(1 for r in records if r['result'] == 'WIN'),
        'avg_steps': round(sum(steps) / n, 1),
    }

def calc_recent(records, n):
    recent = records[-n:]
    scores = [r['clean_score'] for r in recent]
    avg = sum(scores) / len(scores)
    return {
        'avg': round(avg, 2),
        'max': max(scores),
        'min': min(scores),
        'scores': scores,
        'episodes': [{'ep': r['ep'], 'score': r['clean_score'], 'steps': r['steps'], 'result': r['result']} for r in recent],
    }

def calc_time_series(records):
    ts = []
    max_ep = max(r['ep'] for r in records)
    for i in range(0, max_ep + 1, 50):
        bucket = [r for r in records if i <= r['ep'] < i + 50]
        if bucket:
            bs = [r['clean_score'] for r in bucket]
            ts.append({
                'ep_range': f'{i}-{i + 49}',
                'count': len(bucket),
                'avg_score': round(sum(bs) / len(bs), 2),
                'max_score': max(bs),
                'min_score': min(bs),
                'avg_steps': round(sum(r['steps'] for r in bucket) / len(bucket), 1),
            })
    return ts

def calc_distribution(scores):
    dist = {'0-50': 0, '50-100': 0, '100-150': 0, '150-200': 0, '200-300': 0, '300+': 0}
    for s in scores:
        if s < 50:
            dist['0-50'] += 1
        elif s < 100:
            dist['50-100'] += 1
        elif s < 150:
            dist['100-150'] += 1
        elif s < 200:
            dist['150-200'] += 1
        elif s < 300:
            dist['200-300'] += 1
        else:
            dist['300+'] += 1
    n = len(scores)
    dist_pct = {k: round(v / n * 100, 1) for k, v in dist.items()}
    return dist, dist_pct

def calc_convergence(records):
    def wavg(recs, n):
        if len(recs) < n:
            return None
        return round(sum(r['clean_score'] for r in recs[-n:]) / n, 2)

    conv = {
        'recent_10_avg': wavg(records, 10),
        'recent_20_avg': wavg(records, 20),
        'recent_30_avg': wavg(records, 30),
        'recent_50_avg': wavg(records, 50),
        'first_30_avg': wavg(records[:30], 30),
    }
    if conv['first_30_avg'] and conv['recent_30_avg']:
        conv['growth_first_vs_recent_pct'] = round(
            (conv['recent_30_avg'] - conv['first_30_avg']) / conv['first_30_avg'] * 100, 2
        )
    if conv['recent_10_avg'] and conv['recent_30_avg']:
        conv['growth_10_vs_30_pct'] = round(
            (conv['recent_10_avg'] - conv['recent_30_avg']) / conv['recent_30_avg'] * 100, 2
        )
    return conv

def calc_anomalies(records, avg, std):
    anomalies = []
    for r in records:
        if r['clean_score'] > avg + 2 * std:
            anomalies.append({'ep': r['ep'], 'score': r['clean_score'], 'steps': r['steps'], 'type': 'high_spike'})
        elif r['clean_score'] < max(0, avg - 2 * std):
            anomalies.append({'ep': r['ep'], 'score': r['clean_score'], 'steps': r['steps'], 'type': 'low_drop'})
    return anomalies

def main():
    records = parse_gameover_records()
    if not records:
        print("No GAMEOVER records found")
        return

    metrics = parse_training_metrics()
    containers = get_container_status()

    all_scores = [r['clean_score'] for r in records]
    stats = calc_stats(records)
    recent30 = calc_recent(records, 30)
    ts = calc_time_series(records)
    dist, dist_pct = calc_distribution(all_scores)
    conv = calc_convergence(records)
    anomalies = calc_anomalies(records, stats['avg'], stats['std'])

    top5 = sorted(records, key=lambda x: x['clean_score'], reverse=True)[:5]
    top5_out = [{'ep': r['ep'], 'score': r['clean_score'], 'steps': r['steps'], 'time': r['time'][:19]} for r in top5]

    start_t = datetime.strptime(records[0]['time'][:19], '%Y-%m-%d %H:%M:%S')
    end_t = datetime.strptime(records[-1]['time'][:19], '%Y-%m-%d %H:%M:%S')
    runtime = round((end_t - start_t).total_seconds() / 60, 1)

    output = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'runtime_minutes': runtime,
        'episodes': len(records),
        'plan': 'B Phase1++ (bat150, battery_fail=-8, 50% threshold)',
        'containers': containers,
        'stats': stats,
        'time_series': ts,
        'distribution': dist,
        'distribution_pct': dist_pct,
        'recent_scores': recent30['scores'],
        'recent_episodes': recent30['episodes'],
        'convergence': conv,
        'top_scores': top5_out,
        'anomalies': anomalies,
        'training_metrics_series': metrics,
    }

    out_path = BASE / 'train' / 'TRAINING_DATA.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Collected {len(records)} episodes, {runtime}min, avg={stats['avg']}, max={stats['max']}")

if __name__ == '__main__':
    main()

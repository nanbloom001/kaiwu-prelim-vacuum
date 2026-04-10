#!/usr/bin/env python3
"""Simple training status monitor - reads from log files"""
import re
import time
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("/home/user/TcKaiwuFinal/train/log")

GAMEOVER_PATTERN = re.compile(
    r'\[GAMEOVER\] ep:(?P<ep>\d+) steps:(?P<steps>\d+) result:(?P<result>\w+) '
    r'.*?clean_score:(?P<score>[-\d.]+)'
)

def get_latest_episodes(log_files, count=20):
    """Get latest episodes from log files"""
    episodes = []
    for log_file in log_files:
        try:
            content = log_file.read_text()
            for match in GAMEOVER_PATTERN.finditer(content):
                ep = {
                    'ep': int(match.group('ep')),
                    'steps': int(match.group('steps')),
                    'result': match.group('result'),
                    'score': float(match.group('score')),
                }
                episodes.append(ep)
        except:
            pass
    # Sort by episode number and get latest
    episodes.sort(key=lambda x: x['ep'], reverse=True)
    return episodes[:count]

def get_training_status():
    """Get overall training status"""
    status = {
        'learner_running': False,
        'aisrv_count': 0,
        'gamecore_count': 0,
        'total_episodes': 0,
        'latest_episodes': [],
        'avg_score': 0,
        'max_score': 0,
    }

    # Count containers
    import subprocess
    try:
        result = subprocess.run(['docker', 'compose', '-p', 'kaiwu-train', 'ps', '-q'],
                              capture_output=True, text=True, cwd=LOG_DIR)
        if result.returncode == 0:
            containers = result.stdout.strip().count('\n') + 1 if result.stdout.strip() else 0
    except:
        containers = 0

    # Get aisrv logs
    aisrv_logs = list((LOG_DIR / "aisrv").glob("aisrv_kaiwu_rl_helper_pid*.log"))
    episodes = get_latest_episodes(aisrv_logs, 50)

    if episodes:
        status['total_episodes'] = max(ep['ep'] for ep in episodes)
        scores = [ep['score'] for ep in episodes[:20]]  # Last 20
        status['avg_score'] = sum(scores) / len(scores) if scores else 0
        status['max_score'] = max(ep['score'] for ep in episodes)
        status['latest_episodes'] = episodes[:10]

    return status

def print_status():
    """Print training status to console"""
    status = get_training_status()

    print("\n" + "="*60)
    print(f"  Kaiwu Training Status - {datetime.now().strftime('%H:%M:%S')}")
    print("="*60)

    print(f"\n  Total Episodes: {status['total_episodes']}")
    print(f"  Avg Score (last 20): {status['avg_score']:.1f}")
    print(f"  Max Score: {status['max_score']:.1f}")

    print(f"\n  Latest Episodes:")
    print("  " + "-"*50)
    print(f"  {'Ep':<6} {'Steps':<8} {'Result':<6} {'Score':<10}")
    print("  " + "-"*50)

    for ep in status['latest_episodes']:
        result_color = "✓" if ep['result'] == "WIN" else "✗"
        print(f"  {ep['ep']:<6} {ep['steps']:<8} {result_color:<6} {ep['score']:<10.1f}")

    print("\n  Containers:")
    import subprocess
    try:
        result = subprocess.run(['docker', 'compose', '-p', 'kaiwu-train', 'ps'],
                              capture_output=True, text=True, cwd=LOG_DIR)
        lines = result.stdout.split('\n')
        for line in lines[1:]:  # Skip header
            if 'Up' in line or 'healthy' in line:
                parts = line.split()
                if len(parts) >= 4:
                    name = parts[0].replace('kaiwu-train-', '')
                    state = parts[3] if len(parts) > 3 else ''
                    if state:
                        print(f"    {name:<25} {state}")
    except:
        print("    (Unable to get container status)")

    print("="*60 + "\n")

def main():
    """Main monitoring loop"""
    try:
        while True:
            print_status()
            time.sleep(10)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    main()

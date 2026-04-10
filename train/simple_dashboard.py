#!/usr/bin/env python3
"""Simple training dashboard - reads from log files only"""
import re
import time
import json
from datetime import datetime
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

LOG_DIR = Path("/home/user/TcKaiwuFinal/train/log")

GAMEOVER_PATTERN = re.compile(
    r'\[GAMEOVER\] ep:(?P<ep>\d+) steps:(?P<steps>\d+) result:(?P<result>\w+) '
    r'.*?clean_score:(?P<score>[-\d.]+).*?invalid_move_rate:(?P<invalid>[-\d.]+) '
    r'profile:(?P<profile>\w+)'
)

def get_training_status():
    """Get current training status from logs"""
    aisrv_logs = list((LOG_DIR / "aisrv").glob("aisrv_kaiwu_rl_helper_pid*_log_*"))

    episodes = []
    for log_file in aisrv_logs:
        try:
            for line in read_lines(log_file, 500):
                match = GAMEOVER_PATTERN.search(line)
                if match:
                    episodes.append({
                        'ep': int(match.group('ep')),
                        'steps': int(match.group('steps')),
                        'result': match.group('result'),
                        'score': float(match.group('score')),
                        'invalid': float(match.group('invalid')),
                        'profile': match.group('profile'),
                    })
        except:
            pass

    # Sort and deduplicate
    episodes.sort(key=lambda x: x['ep'], reverse=True)
    seen = set()
    unique_episodes = []
    for ep in episodes:
        if ep['ep'] not in seen:
            seen.add(ep['ep'])
            unique_episodes.append(ep)

    latest = unique_episodes[:50] if unique_episodes else []

    # Calculate stats
    if latest:
        total_episodes = latest[0]['ep']
        scores = [e['score'] for e in latest[:30]]
        avg_score = sum(scores) / len(scores) if scores else 0
        max_score = max(e['score'] for e in latest)
        wins = sum(1 for e in latest if e['result'] == 'WIN')
    else:
        total_episodes = 0
        avg_score = 0
        max_score = 0
        wins = 0

    return {
        'total_episodes': total_episodes,
        'avg_score': avg_score,
        'max_score': max_score,
        'wins': wins,
        'latest': latest[:20]
    }

def read_lines(path, count=100):
    """Read last N lines from file"""
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            return lines[-count:] if len(lines) > count else lines
    except:
        return []

def render_html(status):
    """Render HTML dashboard"""
    episodes_html = ""
    for ep in status['latest']:
        result_class = 'win' if ep['result'] == 'WIN' else 'fail'
        episodes_html += f"""
        <tr>
            <td>{ep['ep']}</td>
            <td class="{result_class}">{ep['result']}</td>
            <td>{ep['score']:.1f}</td>
            <td>{ep['steps']}</td>
            <td>{ep['invalid']:.2f}</td>
            <td>{ep['profile']}</td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Kaiwu Training Monitor</title>
        <meta charset="utf-8">
        <meta http-equiv="refresh" content="10">
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                     background: #1a1a2e; color: #eee; margin: 0; padding: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            h1 {{ color: #4ecdc4; margin-bottom: 10px; }}
            .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
            .stat-card {{ background: #16213e; padding: 20px; border-radius: 8px; text-align: center; }}
            .stat-label {{ font-size: 12px; color: #8892b0; text-transform: uppercase; }}
            .stat-value {{ font-size: 32px; font-weight: bold; color: #4ecdc4; }}
            table {{ width: 100%; border-collapse: collapse; background: #16213e; border-radius: 8px; overflow: hidden; }}
            th {{ background: #0f3460; padding: 12px; text-align: left; color: #4ecdc4; }}
            td {{ padding: 10px; border-bottom: 1px solid #0f3460; }}
            tr:hover {{ background: #1a1a2e; }}
            .win {{ color: #4ecdc4; }}
            .fail {{ color: #ff6b6b; }}
            .timestamp {{ color: #8892b0; font-size: 12px; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Kaiwu Training Monitor</h1>

            <div class="stats">
                <div class="stat-card">
                    <div class="stat-label">Total Episodes</div>
                    <div class="stat-value">{status['total_episodes']}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Avg Score (30)</div>
                    <div class="stat-value">{status['avg_score']:.1f}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Max Score</div>
                    <div class="stat-value">{status['max_score']:.0f}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Wins (20)</div>
                    <div class="stat-value">{status['wins']}</div>
                </div>
            </div>

            <table>
                <thead>
                    <tr>
                        <th>Episode</th>
                        <th>Result</th>
                        <th>Score</th>
                        <th>Steps</th>
                        <th>Invalid Rate</th>
                        <th>Profile</th>
                    </tr>
                </thead>
                <tbody>
                    {episodes_html}
                </tbody>
            </table>

            <div class="timestamp">
                Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Auto-refresh: 10s
            </div>
        </div>
    </body>
    </html>
    """
    return html

class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/":
            self.send_response(404)
            self.end_headers()
            return

        try:
            status = get_training_status()
            html = render_html(status).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(html)
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f"Error: {e}".encode())

    def log_message(self, format, *args):
        pass  # Suppress log messages

def main():
    PORT = 18080
    server = HTTPServer(('0.0.0.0', PORT), DashboardHandler)
    print(f"Starting Kaiwu Training Monitor on http://0.0.0.0:{PORT}")
    print(f"Reading logs from: {LOG_DIR}")
    server.serve_forever()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Serve a lightweight local training dashboard from GreptimeDB metrics.

This avoids the fragile official local monitor UI and renders a self-hosted
HTML page with inline SVG charts. It only relies on the Python standard
library and the Prometheus-compatible API exposed by GreptimeDB.
"""

from __future__ import annotations

import argparse
import collections
import html
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable


DEFAULT_PROM_BASE = "http://127.0.0.1:4000/v1/prometheus"
BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
CODE_DIR = REPO_DIR / "code"
AISRV_LOG_DIR = BASE_DIR / "log" / "aisrv"
LEARNER_LOG_DIR = BASE_DIR / "log" / "learner"
RESUME_META_PATH = CODE_DIR / "model.ckpt-resume.meta.json"
RESUME_SNAPSHOT_DIR = CODE_DIR / "resume_snapshots"
OFFICIAL_MONITOR_URL = (
    "http://127.0.0.1:11000/p/v5/exp/monitor"
    "?domain_id=1&exp_id=1&task_uuid=1&task_id=0&platform=competition_stage"
)

GAMEOVER_PATTERN = re.compile(
    r"\[GAMEOVER\] ep:(?P<ep>\d+) steps:(?P<steps>\d+) result:(?P<result>\w+) "
    r".*?clean_score:(?P<score>[-\d.]+).*?invalid_move_rate:(?P<invalid>[-\d.]+) "
    r"profile:(?P<profile>\w+)"
)
CHECKPOINT_PATTERN = re.compile(r"checkpoint_id (?P<ckpt>\d+) success")
LEARNER_STEP_PATTERN = re.compile(r"global step is (?P<step>\d+)")
LEARNER_SAVE_PATTERN = re.compile(r"model\.ckpt-(?P<ckpt>\d+)\.pkl successfully")


@dataclass(frozen=True)
class MetricSpec:
    title: str
    query: str
    unit: str = ""
    decimals: int = 2


SUMMARY_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("Train Global Step", "sum(kaiwu_train_global_step{})", decimals=0),
    MetricSpec("Episode Count", "sum(kaiwu_episode_cnt{})", decimals=0),
    MetricSpec("Clean Score", "avg(kaiwu_clean_score{})"),
    MetricSpec("Charge Count", "avg(kaiwu_charge_count{})"),
    MetricSpec("Remaining Charge", "avg(kaiwu_remaining_charge{})"),
    MetricSpec("Finished Steps", "avg(kaiwu_finished_steps{})"),
    MetricSpec(
        "Prod/Cons Ratio",
        "avg(kaiwu_sample_production_and_consumption_ratio{})",
        decimals=3,
    ),
)


CHART_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("Train Global Step", "sum(kaiwu_train_global_step{})", decimals=0),
    MetricSpec("Episode Count", "sum(kaiwu_episode_cnt{})", decimals=0),
    MetricSpec("Clean Score", "avg(kaiwu_clean_score{})"),
    MetricSpec("Charge Count", "avg(kaiwu_charge_count{})"),
    MetricSpec("Remaining Charge", "avg(kaiwu_remaining_charge{})"),
    MetricSpec("Finished Steps", "avg(kaiwu_finished_steps{})"),
    MetricSpec(
        "Prod/Cons Ratio",
        "avg(kaiwu_sample_production_and_consumption_ratio{})",
        decimals=3,
    ),
)


class PrometheusClient:
    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # Create opener without proxy for local connections
        # Use empty ProxyHandler to bypass all proxy environment variables
        proxy_handler = urllib.request.ProxyHandler({})
        self.opener = urllib.request.build_opener(proxy_handler)
        self.opener.addheaders = [("Accept", "application/json")]

    def _fetch_json(self, path: str, params: dict[str, str | int | float]) -> dict:
        query = urllib.parse.urlencode(params)
        url = f"{self.base_url}{path}?{query}"
        request = urllib.request.Request(url)
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                payload = response.read().decode("utf-8")
            return json.loads(payload)
        except (urllib.error.URLError, TimeoutError) as exc:
            # Log error for debugging but don't crash
            return {"status": "error", "error": str(exc), "data": {"result": []}}

    def instant_query(self, prom_query: str) -> float | None:
        payload = self._fetch_json("/api/v1/query", {"query": prom_query})
        result = payload.get("data", {}).get("result", [])
        if not result:
            return None
        try:
            return float(result[0]["value"][1])
        except (KeyError, IndexError, TypeError, ValueError):
            return None

    def range_query(self, prom_query: str, start_ts: int, end_ts: int, step: int) -> list[tuple[int, float]]:
        payload = self._fetch_json(
            "/api/v1/query_range",
            {
                "query": prom_query,
                "start": start_ts,
                "end": end_ts,
                "step": step,
            },
        )
        result = payload.get("data", {}).get("result", [])
        if not result:
            return []
        series = []
        for item in result[0].get("values", []):
            try:
                ts = int(float(item[0]))
                value = float(item[1])
            except (IndexError, TypeError, ValueError):
                continue
            series.append((ts, value))
        return series


def fmt_value(value: float | None, decimals: int) -> str:
    if value is None:
        return "n/a"
    if decimals <= 0:
        return str(int(round(value)))
    return f"{value:.{decimals}f}"


def unix_to_local(ts: int) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def read_last_lines(path: Path, limit: int) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return list(collections.deque(handle, maxlen=limit))


def get_latest_files(folder: Path, pattern: str, count: int) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(folder.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)[:count]


def parse_resume_info() -> dict[str, str]:
    info = {
        "trigger": "n/a",
        "episode_cnt": "n/a",
        "clean_score": "n/a",
        "saved_at": "n/a",
        "latest_snapshot": "n/a",
    }
    if RESUME_META_PATH.exists():
        try:
            payload = json.loads(RESUME_META_PATH.read_text(encoding="utf-8"))
            info["trigger"] = str(payload.get("trigger", "n/a"))
            info["episode_cnt"] = str(payload.get("episode_cnt", "n/a"))
            info["clean_score"] = str(payload.get("clean_score", "n/a"))
            info["saved_at"] = str(payload.get("saved_at", "n/a"))
        except json.JSONDecodeError:
            info["saved_at"] = "meta parse failed"
    if RESUME_SNAPSHOT_DIR.exists():
        snapshots = sorted(RESUME_SNAPSHOT_DIR.glob("*.pkl"), key=lambda item: item.stat().st_mtime, reverse=True)
        if snapshots:
            info["latest_snapshot"] = snapshots[0].name
    return info


def parse_learner_info() -> dict[str, str]:
    info = {
        "global_step": "n/a",
        "latest_ckpt": "n/a",
        "file": "n/a",
    }
    latest_files = get_latest_files(LEARNER_LOG_DIR, "learner_train_pid*_log_*", 1)
    if not latest_files:
        return info
    latest = latest_files[0]
    lines = read_last_lines(latest, 250)
    info["file"] = latest.name
    for line in reversed(lines):
        step_match = LEARNER_STEP_PATTERN.search(line)
        if step_match and info["global_step"] == "n/a":
            info["global_step"] = step_match.group("step")
        save_match = LEARNER_SAVE_PATTERN.search(line)
        if save_match and info["latest_ckpt"] == "n/a":
            info["latest_ckpt"] = save_match.group("ckpt")
        if info["global_step"] != "n/a" and info["latest_ckpt"] != "n/a":
            break
    return info


def parse_recent_episodes(limit: int = 12) -> tuple[list[dict[str, str]], str]:
    rows: list[dict[str, str]] = []
    latest_checkpoint = "n/a"
    for path in get_latest_files(AISRV_LOG_DIR, "aisrv_kaiwu_rl_helper_pid*_log_*", 2):
        for line in read_last_lines(path, 220):
            ckpt_match = CHECKPOINT_PATTERN.search(line)
            if ckpt_match:
                latest_checkpoint = ckpt_match.group("ckpt")
            gameover_match = GAMEOVER_PATTERN.search(line)
            if not gameover_match:
                continue
            rows.append(
                {
                    "ep": gameover_match.group("ep"),
                    "steps": gameover_match.group("steps"),
                    "result": gameover_match.group("result"),
                    "score": gameover_match.group("score"),
                    "invalid": gameover_match.group("invalid"),
                    "profile": gameover_match.group("profile"),
                }
            )
    rows.sort(key=lambda item: int(item["ep"]), reverse=True)
    dedup: list[dict[str, str]] = []
    seen_eps: set[str] = set()
    for row in rows:
        if row["ep"] in seen_eps:
            continue
        seen_eps.add(row["ep"])
        dedup.append(row)
        if len(dedup) >= limit:
            break
    dedup.sort(key=lambda item: int(item["ep"]), reverse=True)
    return dedup, latest_checkpoint


def render_detail_panel(title: str, items: list[tuple[str, str]]) -> str:
    rows = "".join(
        '<div class="detail-row">'
        f'<div class="detail-key">{html.escape(key)}</div>'
        f'<div class="detail-val">{html.escape(value)}</div>'
        "</div>"
        for key, value in items
    )
    return f'<section class="detail-card"><h3>{html.escape(title)}</h3>{rows}</section>'


def render_recent_episode_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return (
            '<section class="detail-card detail-span-2"><h3>Recent Episodes</h3>'
            '<div class="chart-empty">No recent episode data</div></section>'
        )
    body = "".join(
        "<tr>"
        f"<td>{html.escape(row['ep'])}</td>"
        f"<td>{html.escape(row['result'])}</td>"
        f"<td>{html.escape(row['score'])}</td>"
        f"<td>{html.escape(row['invalid'])}</td>"
        f"<td>{html.escape(row['steps'])}</td>"
        f"<td>{html.escape(row['profile'])}</td>"
        "</tr>"
        for row in rows
    )
    return (
        '<section class="detail-card detail-span-2"><h3>Recent Episodes</h3>'
        '<div class="table-wrap"><table><thead><tr>'
        "<th>EP</th><th>Result</th><th>Clean</th><th>Invalid</th><th>Steps</th><th>Profile</th>"
        f"</tr></thead><tbody>{body}</tbody></table></div></section>"
    )


def render_svg_chart(title: str, series: list[tuple[int, float]], decimals: int) -> str:
    width = 560
    height = 220
    left = 52
    right = 18
    top = 22
    bottom = 34
    plot_w = width - left - right
    plot_h = height - top - bottom

    if len(series) < 2:
        return (
            '<div class="chart-card">'
            f"<h3>{html.escape(title)}</h3>"
            '<div class="chart-empty">No data yet</div>'
            "</div>"
        )

    xs = [point[0] for point in series]
    ys = [point[1] for point in series]
    min_y = min(ys)
    max_y = max(ys)
    if math.isclose(min_y, max_y):
        pad = max(abs(max_y) * 0.05, 1.0)
        min_y -= pad
        max_y += pad

    def scale_x(ts: int) -> float:
        return left + (ts - xs[0]) / max(xs[-1] - xs[0], 1) * plot_w

    def scale_y(value: float) -> float:
        return top + (max_y - value) / max(max_y - min_y, 1e-9) * plot_h

    polyline = " ".join(f"{scale_x(ts):.2f},{scale_y(val):.2f}" for ts, val in series)
    last_value = ys[-1]
    last_x = scale_x(xs[-1])
    last_y = scale_y(last_value)

    grid_lines = []
    y_labels = []
    for index in range(5):
        ratio = index / 4
        y = top + ratio * plot_h
        value = max_y - ratio * (max_y - min_y)
        grid_lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" class="grid"/>'
        )
        y_labels.append(
            f'<text x="{left - 8}" y="{y + 4:.2f}" class="axis-label" text-anchor="end">'
            f"{html.escape(fmt_value(value, decimals))}</text>"
        )

    x_labels = [
        f'<text x="{left}" y="{height - 10}" class="axis-label" text-anchor="start">{unix_to_local(xs[0])}</text>',
        f'<text x="{width - right}" y="{height - 10}" class="axis-label" text-anchor="end">{unix_to_local(xs[-1])}</text>',
    ]

    return (
        '<div class="chart-card">'
        f"<h3>{html.escape(title)}</h3>"
        f'<div class="chart-meta">Latest: {html.escape(fmt_value(last_value, decimals))}</div>'
        f'<svg viewBox="0 0 {width} {height}" class="chart-svg" role="img" aria-label="{html.escape(title)}">'
        f'{"".join(grid_lines)}'
        f'{"".join(y_labels)}'
        f'{"".join(x_labels)}'
        f'<polyline fill="none" stroke="#146ef5" stroke-width="3" points="{polyline}"></polyline>'
        f'<circle cx="{last_x:.2f}" cy="{last_y:.2f}" r="4.5" fill="#e94f37"></circle>'
        "</svg>"
        "</div>"
    )


def render_summary_cards(client: PrometheusClient, metrics: Iterable[MetricSpec]) -> str:
    cards = []
    for metric in metrics:
        value = client.instant_query(metric.query)
        cards.append(
            '<div class="card">'
            f'<div class="card-title">{html.escape(metric.title)}</div>'
            f'<div class="card-value">{html.escape(fmt_value(value, metric.decimals))}</div>'
            "</div>"
        )
    return "".join(cards)


def render_page(client: PrometheusClient, minutes: int, step: int, refresh_seconds: int) -> str:
    now_ts = int(time.time())
    start_ts = now_ts - minutes * 60
    cards_html = render_summary_cards(client, SUMMARY_METRICS)
    chart_sections: list[str] = []
    errors: list[str] = []
    recent_rows, loaded_checkpoint = parse_recent_episodes()
    learner_info = parse_learner_info()
    resume_info = parse_resume_info()
    status_panels = "".join(
        [
            render_detail_panel(
                "Training Runtime",
                [
                    ("Learner Global Step", learner_info["global_step"]),
                    ("Learner Latest CKPT", learner_info["latest_ckpt"]),
                    ("AISRV Loaded CKPT", loaded_checkpoint),
                    ("Learner Log File", learner_info["file"]),
                ],
            ),
            render_detail_panel(
                "Resume State",
                [
                    ("Trigger", resume_info["trigger"]),
                    ("Episode", resume_info["episode_cnt"]),
                    ("Clean Score", resume_info["clean_score"]),
                    ("Saved At", resume_info["saved_at"]),
                    ("Latest Snapshot", resume_info["latest_snapshot"]),
                ],
            ),
        ]
    )
    episode_table = render_recent_episode_table(recent_rows)

    for metric in CHART_METRICS:
        try:
            series = client.range_query(metric.query, start_ts, now_ts, step)
            chart_sections.append(render_svg_chart(metric.title, series, metric.decimals))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            errors.append(f"{metric.title}: {exc}")

    error_block = ""
    if errors:
        joined = "".join(f"<li>{html.escape(msg)}</li>" for msg in errors)
        error_block = f'<div class="errors"><strong>Query errors</strong><ul>{joined}</ul></div>'

    updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Kaiwu Local Monitor</title>
  <meta http-equiv="refresh" content="{refresh_seconds}">
  <style>
    :root {{
      --bg: #f3f6fb;
      --panel: #ffffff;
      --text: #10213a;
      --muted: #61758a;
      --line: #d7e0ea;
      --accent: #146ef5;
      --accent-2: #0a936f;
      --warn: #b76e00;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(20, 110, 245, 0.08), transparent 28%),
        linear-gradient(180deg, #eef4ff 0%, var(--bg) 100%);
      color: var(--text);
    }}
    .page {{
      max-width: 1480px;
      margin: 0 auto;
      padding: 24px;
    }}
    .hero {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
      margin-bottom: 20px;
    }}
    h1 {{
      margin: 0;
      font-size: 28px;
      line-height: 1.2;
    }}
    .sub {{
      color: var(--muted);
      margin-top: 8px;
      font-size: 14px;
    }}
    .links {{
      text-align: right;
      font-size: 13px;
      color: var(--muted);
    }}
    .tip {{
      margin-top: 10px;
      display: inline-block;
      padding: 8px 10px;
      border-radius: 10px;
      background: #fff6e7;
      color: #815300;
      font-size: 12px;
      border: 1px solid #f1dbaf;
    }}
    .links a {{
      color: var(--accent);
      text-decoration: none;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px 18px;
      box-shadow: 0 10px 24px rgba(16, 33, 58, 0.05);
    }}
    .card-title {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .card-value {{
      font-size: 28px;
      font-weight: 700;
      letter-spacing: -0.02em;
    }}
    .details {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 16px;
      margin-bottom: 18px;
    }}
    .detail-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px 18px;
      box-shadow: 0 10px 24px rgba(16, 33, 58, 0.05);
    }}
    .detail-card h3 {{
      margin: 0 0 12px;
      font-size: 17px;
    }}
    .detail-span-2 {{
      grid-column: 1 / -1;
    }}
    .detail-row {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 8px 0;
      border-top: 1px solid #eef2f7;
    }}
    .detail-row:first-of-type {{
      border-top: 0;
      padding-top: 0;
    }}
    .detail-key {{
      color: var(--muted);
      font-size: 13px;
    }}
    .detail-val {{
      text-align: right;
      font-size: 13px;
      font-weight: 600;
      color: var(--text);
      word-break: break-word;
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      text-align: left;
      padding: 10px 8px;
      border-top: 1px solid #eef2f7;
      white-space: nowrap;
    }}
    th {{
      color: var(--muted);
      font-weight: 600;
      border-top: 0;
      padding-top: 0;
    }}
    .charts {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(580px, 1fr));
      gap: 16px;
    }}
    .chart-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px 16px 10px;
      box-shadow: 0 10px 24px rgba(16, 33, 58, 0.05);
    }}
    .chart-card h3 {{
      margin: 0 0 4px;
      font-size: 16px;
    }}
    .chart-meta {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
    }}
    .chart-svg {{
      width: 100%;
      height: auto;
      display: block;
      overflow: visible;
    }}
    .grid {{
      stroke: var(--line);
      stroke-width: 1;
    }}
    .axis-label {{
      fill: var(--muted);
      font-size: 11px;
    }}
    .errors {{
      background: #fff5f4;
      border: 1px solid #f3c9c3;
      color: #8c2f21;
      border-radius: 12px;
      padding: 12px 14px;
      margin-bottom: 18px;
    }}
    .errors ul {{
      margin: 8px 0 0;
      padding-left: 18px;
    }}
    .chart-empty {{
      min-height: 180px;
      display: grid;
      place-items: center;
      color: var(--muted);
      font-size: 14px;
    }}
    @media (max-width: 900px) {{
      .hero {{
        display: block;
      }}
      .links {{
        text-align: left;
        margin-top: 12px;
      }}
      .charts {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <div>
        <h1>Kaiwu Local Monitor</h1>
        <div class="sub">Updated at {html.escape(updated_at)}. Window: last {minutes} minutes. Auto refresh: every {refresh_seconds}s.</div>
        <div class="tip">Official local page needs the full query string. Opening only <code>/p/v5/exp/monitor</code> will render the built-in 404 page.</div>
      </div>
      <div class="links">
        <div>Prometheus API: <a href="{html.escape(client.base_url)}/api/v1/query?query=sum(kaiwu_train_global_step%7B%7D)">{html.escape(client.base_url)}</a></div>
        <div>Official local page: <a href="{html.escape(OFFICIAL_MONITOR_URL)}">{html.escape(OFFICIAL_MONITOR_URL)}</a></div>
      </div>
    </section>
    {error_block}
    <section class="cards">{cards_html}</section>
    <section class="details">{status_panels}{episode_table}</section>
    <section class="charts">{''.join(chart_sections)}</section>
  </main>
</body>
</html>
"""


def build_handler(client: PrometheusClient, minutes: int, step: int, refresh_seconds: int):
    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path not in ("/", "/index.html"):
                self.send_response(404)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"not found")
                return

            try:
                body = render_page(client, minutes, step, refresh_seconds).encode("utf-8")
            except Exception as exc:  # noqa: BLE001
                self.send_response(500)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(f"dashboard render failed: {exc}\n".encode("utf-8"))
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args) -> None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = fmt % args
            print(f"[{timestamp}] {self.address_string()} {message}")

    return DashboardHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local Kaiwu metrics dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host. Default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=18080, help="Bind port. Default: 18080")
    parser.add_argument(
        "--prom-base",
        default=DEFAULT_PROM_BASE,
        help=f"Prometheus-compatible base URL. Default: {DEFAULT_PROM_BASE}",
    )
    parser.add_argument("--minutes", type=int, default=30, help="History window in minutes. Default: 30")
    parser.add_argument("--step", type=int, default=60, help="Query range step in seconds. Default: 60")
    parser.add_argument("--refresh-seconds", type=int, default=30, help="Page refresh interval. Default: 30")
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout in seconds. Default: 10")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = PrometheusClient(args.prom_base, args.timeout)
    handler = build_handler(client, args.minutes, args.step, args.refresh_seconds)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving local dashboard at http://{args.host}:{args.port}")
    print(f"Reading metrics from {args.prom_base}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

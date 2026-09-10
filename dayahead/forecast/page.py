"""
The published page.

Writes a single self-contained HTML file to docs/index.html, which GitHub
Pages serves without any build step or running process.

Why static rather than a dashboard
----------------------------------

The forecast updates once a day. There is nothing to be interactive about,
and a hosted app that sleeps after inactivity turns into a dead link on a CV
within weeks. A file in the repository renders identically in a year's time
whether or not anything is still running, which is the correct shape for a
daily batch product.

The page carries its own timestamp so that a stale page is visibly stale
rather than silently wrong. If the daily job fails, the previous page stays
published and the gap shows up in the track record rather than being papered
over with an error state.

No external assets, no CDN, no JavaScript library. Charts are inline SVG
drawn from the data, so the page has no dependency that can rot.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .. import config as cfg

DOCS = cfg.ROOT / "docs"
PAGE = DOCS / "index.html"

INK = "#1a1a1a"
MUTED = "#6b6b6b"
LINE = "#2b5797"
BAND = "#2b5797"
TRUTH = "#c1440e"
GRID = "#e4e4e4"


def _svg_line_chart(x_labels, series, bands=None, width=880, height=300,
                    ylab="EUR/MWh") -> str:
    """
    A line chart as inline SVG.

    Hand-drawn rather than pulled from a plotting library because the page
    must remain a single file with no external requests.
    """
    pad_l, pad_r, pad_t, pad_b = 55, 15, 15, 45
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    values = [v for s in series for v in s["values"] if np.isfinite(v)]
    if bands:
        values += [v for v in bands["low"] + bands["high"] if np.isfinite(v)]
    if not values:
        return f'<svg width="{width}" height="{height}"></svg>'

    lo, hi = min(values), max(values)
    span = hi - lo or 1.0
    lo, hi = lo - 0.08 * span, hi + 0.08 * span
    span = hi - lo

    n = max(len(x_labels), 2)
    sx = lambda i: pad_l + plot_w * i / (n - 1)
    sy = lambda v: pad_t + plot_h * (1 - (v - lo) / span)

    parts = [f'<svg width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
             'role="img">']

    # Horizontal gridlines and y labels.
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        v = lo + span * frac
        y = sy(v)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" '
                     f'y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" '
                     f'font-size="11" fill="{MUTED}">{v:,.0f}</text>')

    if lo < 0 < hi:
        y0 = sy(0)
        parts.append(f'<line x1="{pad_l}" y1="{y0:.1f}" x2="{width - pad_r}" '
                     f'y2="{y0:.1f}" stroke="{MUTED}" stroke-width="1" '
                     'stroke-dasharray="3 3"/>')

    if bands:
        top = " ".join(f"{sx(i):.1f},{sy(v):.1f}"
                       for i, v in enumerate(bands["high"]) if np.isfinite(v))
        bottom = " ".join(f"{sx(i):.1f},{sy(v):.1f}"
                          for i, v in reversed(list(enumerate(bands["low"])))
                          if np.isfinite(v))
        if top and bottom:
            parts.append(f'<polygon points="{top} {bottom}" fill="{BAND}" '
                         'fill-opacity="0.14"/>')

    for s in series:
        pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}"
                       for i, v in enumerate(s["values"]) if np.isfinite(v))
        if pts:
            parts.append(f'<polyline points="{pts}" fill="none" '
                         f'stroke="{s.get("colour", LINE)}" stroke-width="2" '
                         'stroke-linejoin="round"/>')

    step = max(1, n // 12)
    for i, label in enumerate(x_labels):
        if i % step:
            continue
        parts.append(f'<text x="{sx(i):.1f}" y="{height - pad_b + 18:.0f}" '
                     f'text-anchor="middle" font-size="11" fill="{MUTED}">'
                     f'{label}</text>')

    parts.append(f'<text x="14" y="{pad_t + 4}" font-size="11" fill="{MUTED}">'
                 f'{ylab}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _load() -> tuple[pd.DataFrame, dict, dict]:
    from ..forecast.daily import LOG_PATH, load_log, track_record_summary

    log = load_log()
    if len(log):
        log["delivery_hour_local"] = pd.to_datetime(log["delivery_hour_local"],
                                                    utc=True, errors="coerce")
    summary = track_record_summary(log) if len(log) else {"scored_hours": 0}

    champion = {}
    champ_path = cfg.REPORTS / "champion.json"
    if champ_path.exists():
        with champ_path.open(encoding="utf-8") as f:
            champion = json.load(f)
    return log, summary, champion


def build_page() -> str:
    log, summary, champion = _load()
    now = datetime.now(timezone.utc)

    # Tomorrow's forecast, meaning the most recent delivery day in the log.
    tomorrow_html = "<p class='empty'>No forecast issued yet.</p>"
    if len(log):
        latest_day = log["delivery_hour_local"].dt.tz_convert(
            cfg.LOCAL_TZ).dt.normalize().max()
        block = log[log["delivery_hour_local"].dt.tz_convert(
            cfg.LOCAL_TZ).dt.normalize() == latest_day].sort_values(
            "delivery_hour_local")
        if len(block):
            labels = [f"{h:02d}" for h in block["hour"]]
            chart = _svg_line_chart(
                labels,
                [{"values": block["y_pred"].astype(float).tolist()}],
                bands={"low": block["q10"].astype(float).tolist(),
                       "high": block["q90"].astype(float).tolist()},
            )
            tomorrow_html = (
                f"<p class='sub'>Delivery day "
                f"{latest_day:%A %d %B %Y}, local time. Shaded band is the "
                f"calibrated 80% interval.</p>{chart}"
            )

    # Recent predicted against realised.
    history_html = "<p class='empty'>Nothing scored yet. Outturn arrives a day after issuance.</p>"
    scored = log[log["y_true"].notna()] if len(log) else log
    if len(scored):
        recent = scored.sort_values("delivery_hour_local").tail(24 * 21)
        labels = [d.tz_convert(cfg.LOCAL_TZ).strftime("%d %b")
                  for d in recent["delivery_hour_local"]]
        chart = _svg_line_chart(
            labels,
            [{"values": recent["y_true"].astype(float).tolist(),
              "colour": TRUTH},
             {"values": recent["y_pred"].astype(float).tolist(),
              "colour": LINE}],
        )
        history_html = (
            "<p class='sub'><span class='key truth'></span>realised "
            "<span class='key pred'></span>forecast, last three weeks</p>"
            + chart
        )

    if summary.get("scored_hours"):
        stats = f"""
        <div class="stats">
          <div><span class="n">{summary['mae']:.1f}</span><span class="l">MAE, EUR/MWh</span></div>
          <div><span class="n">{summary['coverage']:.0%}</span><span class="l">interval coverage, nominal 80%</span></div>
          <div><span class="n">{summary['scored_hours']:,}</span><span class="l">hours scored</span></div>
          <div><span class="n">{summary['pending_hours']:,}</span><span class="l">awaiting outturn</span></div>
        </div>"""
    else:
        stats = ("<p class='empty'>The track record begins with the first "
                 "scheduled run.</p>")

    model = champion.get("champion", "N-HiTS")
    locked = champion.get("locked_test", {})
    locked_line = ""
    if locked:
        locked_line = (
            f"<p class='sub'>On a held-out year never used in development, "
            f"MAE {locked['champion_mae']:.2f} EUR/MWh against "
            f"{locked['backtest_mae']:.2f} in backtesting. Substituting "
            f"realised inputs for published forecasts would improve it by "
            f"{locked['leak_improvement_pct']:.0f}%, which is what the "
            f"gate-closure constraint costs.</p>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Day-ahead electricity price forecast, DE/LU</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI",
          Roboto, Helvetica, Arial, sans-serif;
          color: {INK}; background: #fdfdfc; margin: 0; padding: 0 20px 60px; }}
  main {{ max-width: 920px; margin: 0 auto; }}
  h1 {{ font-size: 26px; font-weight: 600; margin: 44px 0 6px; }}
  h2 {{ font-size: 18px; font-weight: 600; margin: 40px 0 4px; }}
  p {{ margin: 6px 0 14px; }}
  .sub {{ color: {MUTED}; font-size: 14px; }}
  .empty {{ color: {MUTED}; font-size: 14px; font-style: italic; }}
  .stats {{ display: flex; flex-wrap: wrap; gap: 28px; margin: 18px 0 8px; }}
  .stats .n {{ display: block; font-size: 24px; font-weight: 600; }}
  .stats .l {{ display: block; font-size: 12px; color: {MUTED}; }}
  .key {{ display: inline-block; width: 22px; height: 3px;
          vertical-align: middle; margin: 0 6px 0 14px; }}
  .key.truth {{ background: {TRUTH}; }}
  .key.pred {{ background: {LINE}; }}
  footer {{ margin-top: 48px; padding-top: 16px; border-top: 1px solid {GRID};
            color: {MUTED}; font-size: 13px; }}
  a {{ color: {LINE}; }}
  svg {{ max-width: 100%; height: auto; }}
</style>
</head>
<body>
<main>

  <h1>Day-ahead electricity price forecast</h1>
  <p class="sub">German and Luxembourg bidding zone. Issued daily before the
  noon auction, using only information published at gate closure.</p>

  <h2>Tomorrow</h2>
  {tomorrow_html}

  <h2>Track record</h2>
  <p class="sub">Every forecast below was published before its outturn was
  known. The log is committed to the repository and is append only.</p>
  {stats}
  {history_html}

  <h2>Method</h2>
  <p class="sub">Champion model: {model}, selected on pinball loss across 59
  rolling-origin backtest folds, with conformally calibrated intervals.
  Inputs are the published day-ahead forecasts of load, wind and solar, and
  prices from auctions that have already cleared. Realised outturn is never
  used.</p>
  {locked_line}

  <footer>
    Page generated {now:%Y-%m-%d %H:%M} UTC.
    Data: Bundesnetzagentur SMARD.
    If this timestamp is old, the daily job has not run and the figures above
    are the last ones produced.
  </footer>

</main>
</body>
</html>
"""


def write_page() -> dict:
    DOCS.mkdir(parents=True, exist_ok=True)
    html = build_page()
    PAGE.write_text(html, encoding="utf-8")
    return {"path": str(PAGE), "bytes": len(html.encode("utf-8"))}

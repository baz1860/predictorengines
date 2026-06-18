#!/usr/bin/env python3
"""Single-page offline dashboard (v2 M8) -> dashboard.html.

Renders, with inline SVG only (no external JS/CSS, opens via file://):
  * bankroll curve (settled-bet history)
  * CLV trend (closing-line value, if snapshots exist)
  * calibration plot (the fitted isotonic maps vs the diagonal)
  * today's bet queue (bet_queue.csv)
  * title-odds movers (champion % now vs the previous snapshot)

Reads only local files, so it runs in the daily pipeline offline. Re-run anytime:
  python3 report.py
"""
import html
import json
from datetime import date
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "dashboard.html"
TITLE_HISTORY = DATA / "title_history.csv"
START_BANKROLL = 100.0

C_BG, C_FG, C_MUTE = "#0f1419", "#e6e6e6", "#8a96a3"
C_GRID, C_POS, C_NEG, C_LINE = "#243140", "#2fbf71", "#e5484d", "#4c9aff"


def _poly(vals, w, h, pad=24):
    """Map a y-series (x=index) to SVG polyline points + y-range."""
    n = len(vals)
    lo, hi = min(vals), max(vals)
    if hi == lo:
        hi = lo + 1
    def X(i):
        return pad + (w - 2 * pad) * (i / max(n - 1, 1))
    def Y(v):
        return h - pad - (h - 2 * pad) * ((v - lo) / (hi - lo))
    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals))
    return pts, lo, hi, X, Y


def line_card(title, vals, color=C_LINE, fmt="{:.2f}", baseline=None):
    if not vals or len(vals) < 2:
        return _card(title, "<p class='mute'>Not enough data yet.</p>")
    w, h = 560, 190
    pts, lo, hi, X, Y = _poly(vals, w, h)
    grid = ""
    for frac in (0, 0.5, 1):
        y = 24 + (h - 48) * frac
        val = hi - (hi - lo) * frac
        grid += (f"<line x1='24' y1='{y:.1f}' x2='{w-24}' y2='{y:.1f}' "
                 f"stroke='{C_GRID}'/><text x='2' y='{y+4:.1f}' "
                 f"fill='{C_MUTE}' font-size='10'>{fmt.format(val)}</text>")
    base = ""
    if baseline is not None and lo <= baseline <= hi:
        yb = Y(baseline)
        base = (f"<line x1='24' y1='{yb:.1f}' x2='{w-24}' y2='{yb:.1f}' "
                f"stroke='{C_MUTE}' stroke-dasharray='4 3'/>")
    svg = (f"<svg viewBox='0 0 {w} {h}' width='100%'>{grid}{base}"
           f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{pts}'/>"
           f"<circle cx='{X(len(vals)-1):.1f}' cy='{Y(vals[-1]):.1f}' r='3' "
           f"fill='{color}'/></svg>")
    return _card(title, svg)


def calibration_card():
    f = DATA / "calibration.json"
    if not f.exists():
        return _card("Calibration", "<p class='mute'>Not fitted "
                     "(python3 validate.py --calibrate).</p>")
    maps = json.loads(f.read_text())
    w = h = 230
    pad = 28
    def X(v):
        return pad + (w - 2 * pad) * v
    def Y(v):
        return h - pad - (h - 2 * pad) * v
    parts = [f"<line x1='{X(0):.0f}' y1='{Y(0):.0f}' x2='{X(1):.0f}' y2='{Y(1):.0f}' "
             f"stroke='{C_MUTE}' stroke-dasharray='4 3'/>"]
    colors = {"home": C_LINE, "draw": "#f5a623", "away": C_POS}
    for side, col in colors.items():
        m = maps.get(side)
        if not m:
            continue
        pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(m["x"], m["y"]))
        parts.append(f"<polyline fill='none' stroke='{col}' stroke-width='2' "
                     f"points='{pts}'/>")
    axis = (f"<text x='{w/2:.0f}' y='{h-4}' fill='{C_MUTE}' font-size='10' "
            f"text-anchor='middle'>predicted →</text>")
    legend = "  ".join(f"<tspan fill='{c}'>● {s}</tspan>"
                       for s, c in colors.items())
    svg = (f"<svg viewBox='0 0 {w} {h}' width='100%'>{''.join(parts)}{axis}"
           f"<text x='{pad}' y='16' font-size='10'>{legend}</text></svg>")
    return _card("Calibration (isotonic map vs diagonal)", svg)


def title_card():
    f = HERE / "tournament_odds.csv"
    if not f.exists():
        return _card("Title odds", "<p class='mute'>No tournament_odds.csv.</p>")
    df = pd.read_csv(f).sort_values("champion", ascending=False).head(12)
    today = str(date.today())
    # movers vs previous snapshot
    prev = {}
    hist = pd.read_csv(TITLE_HISTORY) if TITLE_HISTORY.exists() else pd.DataFrame()
    if not hist.empty:
        before = hist[hist["date"] < today]
        if not before.empty:
            last = before[before["date"] == before["date"].max()]
            prev = dict(zip(last["team"], last["champion"]))
    # snapshot today (replace any existing today rows)
    snap = df[["team", "champion"]].copy()
    snap.insert(0, "date", today)
    allh = pd.concat([hist[hist["date"] != today] if not hist.empty else hist, snap],
                     ignore_index=True)
    allh.to_csv(TITLE_HISTORY, index=False)

    mx = df["champion"].max() or 1
    rows = ""
    for r in df.itertuples(index=False):
        pct = r.champion * 100
        bw = 100 * r.champion / mx
        mv = ""
        if r.team in prev:
            d = (r.champion - prev[r.team]) * 100
            if abs(d) >= 0.1:
                col = C_POS if d > 0 else C_NEG
                mv = f"<span style='color:{col}'>{'▲' if d>0 else '▼'}{abs(d):.1f}</span>"
        rows += (f"<tr><td>{html.escape(r.team)}</td>"
                 f"<td class='bar'><span style='width:{bw:.0f}%'></span></td>"
                 f"<td class='num'>{pct:.1f}%</td><td class='num'>{mv}</td></tr>")
    note = "" if prev else "<p class='mute'>First snapshot — movers appear next run.</p>"
    return _card("Title odds (▲▼ vs last snapshot)",
                 f"<table class='tbl'>{rows}</table>{note}")


def fixtures_card():
    f = HERE / "predictions_worldcup_2026.csv"
    if not f.exists():
        return _card("Today's fixtures", "<p class='mute'>No "
                     "predictions_worldcup_2026.csv yet (run predictor.py "
                     "--worldcup).</p>")
    df = pd.read_csv(f)
    if df.empty:
        return _card("Today's fixtures", "<p class='mute'>No fixtures "
                     "predicted yet.</p>")
    today = str(date.today())
    fx = df[df["date"] == today]
    if fx.empty:
        future = df[df["date"] > today]
        if not future.empty:
            fx = future[future["date"] == future["date"].min()]
    head = "".join(f"<th>{c}</th>" for c in
                   ["Match", "Win%", "Draw%", "Loss%", "BTTS%", "Likely"])
    body = ""
    for r in fx.itertuples(index=False):
        match = f"{html.escape(str(r.home))} v {html.escape(str(r.away))}"
        btts = getattr(r, "p_btts", None)
        btts_cell = f"{btts*100:.1f}" if btts is not None else "-"
        body += (f"<tr><td>{match}</td>"
                 f"<td class='num'>{r.p_home*100:.1f}</td>"
                 f"<td class='num'>{r.p_draw*100:.1f}</td>"
                 f"<td class='num'>{r.p_away*100:.1f}</td>"
                 f"<td class='num'>{btts_cell}</td>"
                 f"<td>{html.escape(str(r.likely_score))}</td></tr>")
    return _card("Today's fixtures",
                 f"<table class='tbl'><tr>{head}</tr>{body}</table>")


def queue_card():
    f = HERE / "bet_queue.csv"
    if not f.exists():
        return _card("Today's bet queue", "<p class='mute'>No bet_queue.csv yet "
                     "(run edge.py).</p>")
    q = pd.read_csv(f)
    if q.empty:
        return _card("Today's bet queue", "<p class='mute'>No bets queued "
                     "(no imminent positive-edge picks).</p>")
    adj = q["adjustments"].iloc[0] if "adjustments" in q.columns else "raw-model"
    head = "".join(f"<th>{c}</th>" for c in
                   ["match", "bet", "odds", "edge", "stake £", "squad adj"])
    body = ""
    for r in q.itertuples(index=False):
        body += (f"<tr><td>{html.escape(str(r.match))}</td>"
                 f"<td>{html.escape(str(r.bet))}</td>"
                 f"<td class='num'>{r.odds:.2f}</td>"
                 f"<td class='num'>{r.edge*100:+.1f}%</td>"
                 f"<td class='num'>{r.stake:.2f}</td>"
                 f"<td>{html.escape(str(getattr(r,'squad_adj','-')))}</td></tr>")
    return _card(f"Today's bet queue  <span class='mute'>({adj})</span>",
                 f"<table class='tbl'><tr>{head}</tr>{body}</table>")


def _card(title, body):
    return f"<section class='card'><h2>{title}</h2>{body}</section>"


def main():
    # bankroll + CLV from the ledger
    ledger_f = DATA / "ledger.csv"
    bankroll_vals, clv_vals, summary = [START_BANKROLL], [], ""
    if ledger_f.exists():
        led = pd.read_csv(ledger_f)
        settled = led[led["status"].isin(["won", "lost"])].copy()
        ba = pd.to_numeric(settled["bankroll_after"], errors="coerce").dropna()
        if len(ba):
            bankroll_vals = [START_BANKROLL] + ba.tolist()
        try:
            from core.clv import compute_clv
            c = compute_clv(settled).dropna()
            if len(c):
                clv_vals = (c.cumsum() / range(1, len(c) + 1)).tolist()  # rolling mean
        except Exception:
            pass
        wins = (settled["status"] == "won").sum()
        pnl = pd.to_numeric(settled["pnl"], errors="coerce").sum()
        cur = bankroll_vals[-1]
        summary = (f"Bankroll <b>£{cur:.2f}</b> &nbsp; settled {len(settled)} "
                   f"({wins} won) &nbsp; net £{pnl:+.2f}")
    cards = [
        _card("Summary", f"<p>{summary or 'No settled bets yet.'}</p>"),
        line_card("Bankroll curve (£)", bankroll_vals, C_POS, "£{:.0f}",
                  baseline=START_BANKROLL),
        (line_card("CLV trend (rolling mean %)", [v * 100 for v in clv_vals],
                   C_LINE, "{:+.1f}%", baseline=0.0) if clv_vals
         else _card("CLV trend", "<p class='mute'>No closing-odds snapshots yet "
                    "(clv.py --snapshot before kickoffs).</p>")),
        calibration_card(),
        fixtures_card(),
        queue_card(),
        title_card(),
    ]
    page = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>WC2026 model dashboard</title><style>
body{{background:{C_BG};color:{C_FG};font:14px/1.5 -apple-system,Segoe UI,Arial;margin:0;padding:24px}}
h1{{font-size:20px;margin:0 0 4px}} .sub{{color:{C_MUTE};margin:0 0 20px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}}
.card{{background:#161b22;border:1px solid {C_GRID};border-radius:10px;padding:14px}}
.card h2{{font-size:13px;font-weight:600;margin:0 0 10px;color:{C_FG}}}
.mute{{color:{C_MUTE}}} table.tbl{{width:100%;border-collapse:collapse;font-size:12px}}
.tbl td,.tbl th{{padding:3px 6px;border-bottom:1px solid {C_GRID};text-align:left}}
.tbl th{{color:{C_MUTE};font-weight:600}} .num{{text-align:right;font-variant-numeric:tabular-nums}}
td.bar{{width:50%}} td.bar span{{display:inline-block;height:9px;background:{C_LINE};border-radius:2px}}
</style></head><body>
<h1>World Cup 2026 — model dashboard</h1>
<p class='sub'>Generated {date.today()} · offline · python3 report.py</p>
<div class='grid'>{''.join(cards)}</div></body></html>"""
    OUT.write_text(page)
    print(f"Wrote {OUT.name} ({len(page)//1024} KB)")


if __name__ == "__main__":
    main()

"""Live dashboard data — the JSON behind the in-app Home view.

Single source of truth for both the live `/api/dashboard` route and the offline
`report.py` page. Reads only local files (suite ledger, predictions, bet queue,
tournament odds, calibration), so it works offline and in the daily pipeline.

Everything is defensive: any missing file degrades that card to an empty/`None`
section rather than raising, so the dashboard always renders.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from . import bankroll_store

ROOT = bankroll_store.ROOT
DATA = ROOT / "data"


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


# ── KPIs + bankroll curve ─────────────────────────────────────────────────
def _bankroll_section() -> dict:
    summ = bankroll_store.status_summary()
    start = summ.get("start", 100.0)
    ledger = bankroll_store.load_ledger()

    curve = [{"i": 0, "v": start}]
    if not ledger.empty:
        settled = ledger[ledger["status"].isin(["won", "lost", "push"])].copy()
        ba = pd.to_numeric(settled.get("bankroll_after"), errors="coerce").dropna()
        for i, v in enumerate(ba.tolist(), start=1):
            curve.append({"i": i, "v": round(float(v), 2)})

    t = summ["totals"]
    settled_n = t["settled_count"]
    staked = 0.0
    if not ledger.empty:
        closed = ledger[ledger["status"].isin(["won", "lost"])]
        staked = float(pd.to_numeric(closed.get("stake"), errors="coerce").fillna(0).sum())
    roi = (t["net_pnl"] / staked) if staked else 0.0
    hit = (t["won"] / settled_n) if settled_n else 0.0

    return {
        "bankroll": summ["bankroll"],
        "peak": summ["peak"],
        "start": start,
        "net_pnl": t["net_pnl"],
        "open_stake": t["open_stake"],
        "open_count": t["open_count"],
        "settled_count": settled_n,
        "won": t["won"],
        "roi": round(roi, 4),
        "hit_rate": round(hit, 4),
        "curve": curve,
        "by_sport": summ.get("by_sport", []),
    }


def _clv_section() -> list:
    ledger = bankroll_store.load_ledger()
    if ledger.empty:
        return []
    try:
        from clv import compute_clv  # noqa: E402  (project-root module)
        settled = ledger[ledger["status"].isin(["won", "lost"])].copy()
        c = compute_clv(settled).dropna()
        if len(c):
            roll = (c.cumsum() / range(1, len(c) + 1))
            return [{"i": i, "v": round(float(v) * 100, 3)} for i, v in enumerate(roll.tolist())]
    except Exception:
        pass
    return []


def _calibration_section() -> dict | None:
    f = DATA / "calibration.json"
    if not f.exists():
        return None
    try:
        maps = json.loads(f.read_text())
    except Exception:
        return None
    out = {}
    for side in ("home", "draw", "away"):
        m = maps.get(side)
        if m and "x" in m and "y" in m:
            out[side] = [{"x": float(x), "y": float(y)} for x, y in zip(m["x"], m["y"])]
    return out or None


def _fixtures_section() -> dict:
    df = _read_csv(ROOT / "predictions_worldcup_2026.csv")
    if df.empty:
        return {"day": None, "rows": []}
    today = str(date.today())
    fx = df[df["date"] == today]
    if fx.empty:
        future = df[df["date"] > today]
        if not future.empty:
            fx = future[future["date"] == future["date"].min()]
    if fx.empty:
        return {"day": None, "rows": []}
    rows = []
    for r in fx.itertuples(index=False):
        btts = getattr(r, "p_btts", None)
        rows.append({
            "match": f"{r.home} v {r.away}",
            "p_home": round(float(r.p_home), 4),
            "p_draw": round(float(r.p_draw), 4),
            "p_away": round(float(r.p_away), 4),
            "p_btts": round(float(btts), 4) if btts is not None and pd.notna(btts) else None,
            "likely": str(getattr(r, "likely_score", "")),
        })
    return {"day": str(fx["date"].iloc[0]), "rows": rows}


def _queue_section() -> dict:
    q = _read_csv(ROOT / "bet_queue.csv")
    if q.empty:
        return {"adjustments": None, "rows": []}
    adj = q["adjustments"].iloc[0] if "adjustments" in q.columns else "raw-model"
    rows = []
    for r in q.itertuples(index=False):
        rows.append({
            "match": str(r.match),
            "bet": str(r.bet),
            "odds": round(float(r.odds), 2),
            "edge": round(float(r.edge), 4),
            "stake": round(float(r.stake), 2),
        })
    return {"adjustments": str(adj), "rows": rows}


def _title_section() -> dict:
    df = _read_csv(ROOT / "tournament_odds.csv")
    if df.empty or "champion" not in df.columns:
        return {"rows": []}
    df = df.sort_values("champion", ascending=False).head(12)
    today = str(date.today())
    prev = {}
    hist = _read_csv(DATA / "title_history.csv")
    if not hist.empty:
        before = hist[hist["date"] < today]
        if not before.empty:
            last = before[before["date"] == before["date"].max()]
            prev = dict(zip(last["team"], last["champion"]))
    rows = []
    for r in df.itertuples(index=False):
        delta = None
        if r.team in prev:
            d = (r.champion - prev[r.team]) * 100
            if abs(d) >= 0.05:
                delta = round(d, 2)
        rows.append({
            "team": str(r.team),
            "champion": round(float(r.champion), 4),
            "delta": delta,
        })
    return {"rows": rows, "first_snapshot": not prev}


def _match_label(home, away) -> str:
    home, away = str(home or ""), str(away or "")
    if not away or "OUTRIGHT" in away.upper():
        return home or "—"
    return f"{home} v {away}"


def _market(bet, side) -> str:
    b = str(bet or "").lower()
    s = str(side or "").lower()
    if any(k in b for k in ("under", "over", "goals", "2.5", "o/u")):
        return "Totals"
    if "btts" in b or "both teams" in b:
        return "BTTS"
    if any(k in b for k in ("outright", "champion", "winner", "to win group", "to lift")):
        return "Outright"
    if "draw" in b or "win" in b or s in ("home", "away", "draw"):
        return "1X2 / ML"
    return "Other"


# ── bet-history explorer ───────────────────────────────────────────────────
def build_history() -> dict:
    ledger = bankroll_store.load_ledger()
    out = {
        "rows": [], "pnl_curve": [], "summary": {},
        "by_market": [], "by_sport": [],
        "options": {"sports": [], "engines": [], "markets": [], "statuses": []},
    }
    if ledger.empty:
        return out

    ledger = ledger.copy()
    ledger["stake_n"] = pd.to_numeric(ledger["stake"], errors="coerce").fillna(0.0)
    ledger["pnl_n"] = pd.to_numeric(ledger["pnl"], errors="coerce").fillna(0.0)
    ledger["market"] = [_market(b, s) for b, s in zip(ledger["bet"], ledger["side"])]

    rows = []
    for r in ledger.itertuples(index=False):
        rows.append({
            "placed_on": str(getattr(r, "placed_on", "") or ""),
            "engine": str(r.engine or ""),
            "sport": str(r.sport or ""),
            "match_date": str(r.match_date or ""),
            "match": _match_label(r.home, r.away),
            "bet": str(r.bet or ""),
            "market": r.market,
            "odds": round(float(r.odds), 2) if pd.notna(r.odds) else None,
            "stake": round(float(r.stake_n), 2),
            "status": str(r.status or ""),
            "pnl": round(float(r.pnl_n), 2),
        })
    out["rows"] = rows

    # cumulative P&L over settled bets, in ledger order
    closed = ledger[ledger["status"].isin(["won", "lost", "push"])]
    cum = 0.0
    curve = [{"i": 0, "v": 0.0}]
    for i, p in enumerate(closed["pnl_n"].tolist(), start=1):
        cum = round(cum + float(p), 2)
        curve.append({"i": i, "v": cum})
    out["pnl_curve"] = curve

    settled_won = closed[closed["status"].isin(["won", "lost"])]
    staked = float(settled_won["stake_n"].sum())
    net = float(closed["pnl_n"].sum())
    out["summary"] = {
        "total": int(len(ledger)),
        "open": int((ledger["status"] == "open").sum()),
        "settled": int(len(closed)),
        "won": int((closed["status"] == "won").sum()),
        "net_pnl": round(net, 2),
        "staked": round(staked, 2),
        "roi": round(net / staked, 4) if staked else 0.0,
        "hit_rate": round(int((closed["status"] == "won").sum()) / len(settled_won), 4) if len(settled_won) else 0.0,
    }

    def _agg(group_col):
        agg = []
        for key, grp in ledger.groupby(ledger[group_col].fillna("—")):
            c = grp[grp["status"].isin(["won", "lost"])]
            staked_g = float(c["stake_n"].sum())
            net_g = float(c["pnl_n"].sum())
            agg.append({
                group_col: key or "—",
                "bets": int(len(grp)),
                "settled": int(len(c)),
                "won": int((c["status"] == "won").sum()),
                "net_pnl": round(net_g, 2),
                "roi": round(net_g / staked_g, 4) if staked_g else 0.0,
                "hit_rate": round(int((c["status"] == "won").sum()) / len(c), 4) if len(c) else 0.0,
            })
        return sorted(agg, key=lambda x: x["net_pnl"], reverse=True)

    out["by_market"] = _agg("market")
    out["by_sport"] = _agg("sport")
    out["options"] = {
        "sports": sorted([s for s in ledger["sport"].fillna("").unique() if s]),
        "engines": sorted([e for e in ledger["engine"].fillna("").unique() if e]),
        "markets": sorted(ledger["market"].unique().tolist()),
        "statuses": sorted([s for s in ledger["status"].fillna("").unique() if s]),
    }
    return out


# ── fixtures / schedule (World Cup) ─────────────────────────────────────────
def build_fixtures() -> dict:
    df = _read_csv(ROOT / "predictions_worldcup_2026.csv")
    if df.empty:
        return {"days": []}
    today = str(date.today())
    df = df[df["date"] >= today].sort_values(["date", "home"])

    # map queued picks onto matches by "home v away"
    q = _read_csv(ROOT / "bet_queue.csv")
    picks: dict[str, list] = {}
    if not q.empty:
        for r in q.itertuples(index=False):
            picks.setdefault(str(r.match), []).append({
                "bet": str(r.bet), "odds": round(float(r.odds), 2),
                "edge": round(float(r.edge), 4),
            })

    days = []
    for d, grp in df.groupby("date"):
        rows = []
        for r in grp.itertuples(index=False):
            match = f"{r.home} v {r.away}"
            btts = getattr(r, "p_btts", None)
            rows.append({
                "match": match, "home": str(r.home), "away": str(r.away),
                "p_home": round(float(r.p_home), 4),
                "p_draw": round(float(r.p_draw), 4),
                "p_away": round(float(r.p_away), 4),
                "p_btts": round(float(btts), 4) if btts is not None and pd.notna(btts) else None,
                "xg_home": round(float(getattr(r, "xg_home", 0) or 0), 2),
                "xg_away": round(float(getattr(r, "xg_away", 0) or 0), 2),
                "likely": str(getattr(r, "likely_score", "")),
                "picks": picks.get(match, []),
            })
        days.append({"date": str(d), "rows": rows})
    return {"days": days}


# ── outrights / title race ──────────────────────────────────────────────────
def build_outrights() -> dict:
    df = _read_csv(ROOT / "tournament_odds.csv")
    if df.empty or "champion" not in df.columns:
        return {"teams": [], "dates": []}
    df = df.sort_values("champion", ascending=False).head(16)

    hist = _read_csv(DATA / "title_history.csv")
    dates = sorted(hist["date"].unique().tolist()) if not hist.empty else []
    today = str(date.today())
    if dates and dates[-1] != today:
        dates = dates + [today]

    series_by_team: dict[str, dict] = {}
    prev = {}
    if not hist.empty:
        for d, grp in hist.groupby("date"):
            series_by_team[d] = dict(zip(grp["team"], grp["champion"]))
        before = hist[hist["date"] < today]
        if not before.empty:
            last = before[before["date"] == before["date"].max()]
            prev = dict(zip(last["team"], last["champion"]))

    teams = []
    for r in df.itertuples(index=False):
        team, champ = str(r.team), float(r.champion)
        delta = None
        if team in prev:
            d = (champ - prev[team]) * 100
            if abs(d) >= 0.05:
                delta = round(d, 2)
        series = []
        for d in dates:
            snap = series_by_team.get(d, {})
            v = snap.get(team)
            if d == today and v is None:
                v = champ
            series.append({"date": d, "v": round(float(v) * 100, 3) if v is not None else None})
        teams.append({"team": team, "champion": round(champ, 4), "delta": delta, "series": series})
    return {"teams": teams, "dates": dates, "first_snapshot": not prev}


def build_dashboard() -> dict:
    """Assemble the full dashboard payload."""
    return {
        "generated": str(date.today()),
        "bankroll": _bankroll_section(),
        "clv": _clv_section(),
        "calibration": _calibration_section(),
        "fixtures": _fixtures_section(),
        "queue": _queue_section(),
        "title": _title_section(),
    }

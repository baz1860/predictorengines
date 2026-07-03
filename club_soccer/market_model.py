#!/usr/bin/env python3
"""P6.2 — market movement + do-not-bet, from data/odds_history_club.csv (P6.1).

`line_history` reports how a (match, market, side)'s odds have moved since
the first BSD snapshot we captured — implied probabilities from
`1/odds_median`, NOT de-vigged across sides (snapshot_odds.py stores each
side as its own row, not grouped by snapshot instant, so pairing them for a
true multi-way de-vig would be inexact; raw implied-prob movement is a fine
signal for *direction*, which is all do_not_bet needs — documented
approximation).

`do_not_bet` mirrors wc_v4/market_model.py's idea in a simpler, deterministic
form per the plan: suppress a recommended bet when EITHER
  (a) the market has already moved >= 3 implied-prob points toward our side
      since the first snapshot (the value's gone — we'd be chasing steam), or
  (b) book disagreement is tiny (disp < 0.005) and our edge is small (< 4%)
      — books unanimous + thin edge is more likely model noise than a real
      mispricing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from . import snapshot_odds as SO

STEAM_THRESHOLD = 0.03     # 3 implied-prob points
THIN_DISP_THRESHOLD = 0.005
THIN_EDGE_THRESHOLD = 0.04
WARMUP_DAYS = 30           # do-not-bet only applies once this much history exists


def _load_history() -> pd.DataFrame | None:
    if not SO.ODDS_HISTORY_CSV.exists():
        return None
    h = pd.read_csv(SO.ODDS_HISTORY_CSV)
    if h.empty:
        return None
    h["snapshot_time"] = pd.to_datetime(h["snapshot_time"], utc=True,
                                        format="mixed", errors="coerce")
    return h.dropna(subset=["snapshot_time"])


def history_age_days() -> float:
    """Days of odds_history_club.csv snapshot coverage — used to gate
    do_not_bet's "warming up" period (edge.py prints a notice until this
    clears WARMUP_DAYS)."""
    hist = _load_history()
    if hist is None or hist.empty:
        return 0.0
    span = hist["snapshot_time"].max() - hist["snapshot_time"].min()
    return max(0.0, span.total_seconds() / 86400.0)


def line_history(home: str, away: str, match_date: str, market: str, side: str,
                 asof: str | None = None) -> dict:
    """Opening/current implied prob + movement for one (match, market, side).

    Returns {} when there's no snapshot history for it. `asof` caps the
    "current" snapshot for point-in-time use.
    """
    hist = _load_history()
    if hist is None:
        return {}
    ev = hist[(hist["match_date"] == str(match_date)) & (hist["home"] == home)
              & (hist["away"] == away) & (hist["market"] == market)
              & (hist["side"] == side)].sort_values("snapshot_time")
    if ev.empty:
        return {}
    asof_ts = pd.Timestamp(asof, tz="UTC") if asof else None
    cur = ev if asof_ts is None else ev[ev["snapshot_time"] <= asof_ts]
    if cur.empty:
        return {}
    open_row = ev.iloc[0]
    curr_row = cur.iloc[-1]
    p_open = 1.0 / float(open_row["odds_median"])
    p_curr = 1.0 / float(curr_row["odds_median"])
    return {
        "n_snapshots": int(len(ev)), "p_open": p_open, "p_curr": p_curr,
        "move": p_curr - p_open, "odds_open": float(open_row["odds_median"]),
        "odds_curr": float(curr_row["odds_median"]),
        "disp_curr": (float(curr_row["disp"]) if pd.notna(curr_row["disp"]) else None),
        "n_books_curr": int(curr_row["n_books"]),
    }


def do_not_bet(row: dict, asof: str | None = None) -> dict:
    """row needs: home, away, date (match_date), market, side, edge (model
    prob - book prob, as edge.py already computes). Returns
    {"suppress": bool, "reason": str|None, "line": line_history(...) or {}}."""
    line = line_history(row.get("home", ""), row.get("away", ""), row.get("date", ""),
                        str(row.get("market", "")), str(row.get("side", "")), asof)
    if not line:
        return {"suppress": False, "reason": None, "line": {}}

    edge = row.get("edge")
    try:
        edge = float(edge)
    except (TypeError, ValueError):
        edge = None

    reasons = []
    if line["move"] >= STEAM_THRESHOLD:
        reasons.append(f"market_moved_{line['move']:+.3f}_toward_our_side_since_open")
    disp = line.get("disp_curr")
    if disp is not None and disp < THIN_DISP_THRESHOLD and edge is not None and edge < THIN_EDGE_THRESHOLD:
        reasons.append(f"books_unanimous(disp={disp:.4f})_thin_edge({edge:+.3f})")

    return {"suppress": bool(reasons), "reason": "; ".join(reasons) if reasons else None,
           "line": line}


if __name__ == "__main__":  # pragma: no cover — manual smoke
    age = history_age_days()
    print(f"odds_history_club.csv spans {age:.1f} days "
          f"({'do_not_bet active' if age >= WARMUP_DAYS else f'warming up, {WARMUP_DAYS - age:.1f}d to go'})")

"""Availability uncertainty layer — report-only, ported from
wc_v4/availability.py's lineup_confidence/certainty-classification idea.

`player_features.PlayerFeatureStore._compute_team_adj` already turns a list
of absentees into a point-estimate attack/defense multiplier. This module
adds what a bookmaker reasons about on top of that point estimate: how
CERTAIN is the read (a "doubtful, fitness test" absence is a shakier signal
than a confirmed long-term injury), and does it involve a goalkeeper (higher-
variance than an outfield absence)? When confidence is low, the multiplier is
returned as a band, not a point, so staking can haircut size accordingly —
never a change to the point estimate itself.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# BSD gives a *status* field (e.g. "injured", "suspended") we mostly trust
# outright, plus free-text *reason*/return-date notes where a doubtful read
# shows up as hedging language — the same idea as wc_v4/availability.py's
# note-text classification, adapted to BSD's more structured shape.
_DOUBTFUL = re.compile(
    r"doubt|fitness test|day-to-day|game.time decision|assessed|"
    r"knock|late test|75%|50%|questionable|expected to (return|be fit)|"
    r"targeting return|could return|race against time", re.I)
_CERTAIN_STATUSES = {"injured", "suspended", "out", "unavailable", "banned"}

_SD_PER_DOUBTFUL = 0.03    # attack/defense-mult uncertainty per doubtful absence
_SD_FLOOR = 0.01
_GK_SD_BUMP = 0.04


def classify_absence(status: str, reason: str) -> dict:
    text = f"{status} {reason}".strip()
    doubtful = bool(_DOUBTFUL.search(text))
    certain = (not doubtful) and (str(status).strip().lower() in _CERTAIN_STATUSES
                                  or bool(text))
    return {"certain_out": certain, "doubtful": doubtful}


def lineup_confidence(absences: list[dict]) -> dict:
    """How confident is the expected-lineup read, in [0.25, 1.0].

    Certain-out absences don't erode confidence (BSD already told us they're
    out — clean signal); each doubtful one does.
    """
    n_certain = n_doubtful = 0
    for a in absences:
        c = classify_absence(str(a.get("status", "")), str(a.get("reason", "")))
        if c["doubtful"]:
            n_doubtful += 1
        elif c["certain_out"]:
            n_certain += 1
    conf = max(0.25, min(1.0, 1.0 - 0.18 * n_doubtful))
    return {"confidence": round(conf, 3), "n_certain_out": n_certain, "n_doubtful": n_doubtful}


def _gk_absent(absences: list[dict], player_position) -> bool:
    for a in absences:
        name = str(a.get("name") or "")
        if name and player_position(name) == "GK":
            return True
    return False


def availability_band(store, team: str, absences: list[dict]) -> dict:
    """Point-estimate attack/defense multipliers (from
    PlayerFeatureStore._compute_team_adj) plus an uncertainty band that
    widens with doubtful absences and further if a goalkeeper is out."""
    missing = [{"name": a.get("name"), "pos": a.get("pos", "")} for a in absences]
    point = store._compute_team_adj(team, missing)
    conf = lineup_confidence(absences)
    gk = _gk_absent(absences, store.player_position)
    sd = _SD_FLOOR + _SD_PER_DOUBTFUL * conf["n_doubtful"] + (_GK_SD_BUMP if gk else 0.0)
    sd = round(float(sd), 4)
    return {
        "team": team,
        "attack_mult": point["attack_mult"], "defense_mult": point["defense_mult"],
        "attack_mult_low": round(point["attack_mult"] - sd, 4),
        "attack_mult_high": round(point["attack_mult"] + sd, 4),
        "defense_mult_low": round(point["defense_mult"] - sd, 4),
        "defense_mult_high": round(point["defense_mult"] + sd, 4),
        "uncertainty_sd": sd,
        "lineup_confidence": conf["confidence"],
        "n_certain_out": conf["n_certain_out"], "n_doubtful": conf["n_doubtful"],
        "gk_absent": gk, "n_missing": point["n_missing"],
        "status": "report_only",
    }


def match_availability(store, event: dict) -> dict:
    """Home/away availability bands for one BSD event dict."""
    from bsd_client import unavailable_players as bsd_unavailable
    home_team = str(event.get("home_team") or "")
    away_team = str(event.get("away_team") or "")
    u = bsd_unavailable(event)
    return {
        "home": availability_band(store, home_team, u.get("home", [])),
        "away": availability_band(store, away_team, u.get("away", [])),
    }


def match_confidence(report: dict) -> float:
    """Match-level confidence = the weaker of the two sides' reads (a shaky
    lineup on either side makes the whole prediction shakier)."""
    return min(float(report["home"]["lineup_confidence"]),
              float(report["away"]["lineup_confidence"]))

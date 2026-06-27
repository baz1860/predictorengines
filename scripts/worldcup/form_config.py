#!/usr/bin/env python3
"""Shared form-layer config: league weighting, gains, and team aggregation.

One source of truth used by the multiplier writer, the backtest, and the fitter.
League strength scales each player's form contribution — an in-form Premier League
player should move a team's numbers more than an in-form MLS player. The two gains
(G_ATT/G_DEF) translate aggregated form deltas into lambda multipliers and are
*fitted* (form_fit.py) rather than hand-set; load_params() reads the fitted values.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.worldcup.player_form import POS_BASELINE, RATING_BASELINE

HERE = Path(__file__).resolve().parents[2]
PARAMS_FILE = HERE / "data" / "worldcup" / "form_params.json"
PLAYER_CLUB_FILE = HERE / "data" / "worldcup" / "player_club.json"

# attack vs defence influence by position
W_ATT = {"GK": 0.0, "DF": 0.3, "MF": 1.0, "FW": 1.3}
W_DEF = {"GK": 1.3, "DF": 1.2, "MF": 0.6, "FW": 0.2}

CLAMP_LO, CLAMP_HI = 0.80, 1.25
DEFAULT_G_ATT, DEFAULT_G_DEF = 0.12, 0.10   # hand values; fitter overwrites

# club-league strength by club_country (relative quality, ~0.6-1.0).
# Fixed prior — NOT fitted (too many params for 66 matches). Unknown/'' -> default.
LEAGUE_STRENGTH = {
    "England": 1.00, "Spain": 0.98, "Italy": 0.95, "Germany": 0.95,
    "France": 0.90, "Portugal": 0.84, "Netherlands": 0.82, "Belgium": 0.78,
    "Brazil": 0.80, "Argentina": 0.76, "Türkiye": 0.74, "Turkey": 0.74,
    "Mexico": 0.72, "USA": 0.70, "Saudi Arabia": 0.70, "Scotland": 0.70,
    "Czechia": 0.70, "Greece": 0.70, "Switzerland": 0.72, "Austria": 0.70,
    "Denmark": 0.72, "Norway": 0.68, "Sweden": 0.68, "Croatia": 0.68,
    "Japan": 0.66, "South Korea": 0.64, "Egypt": 0.62, "South Africa": 0.58,
}
DEFAULT_LEAGUE_STRENGTH = 0.65


def _norm_pos(p: str) -> str:
    p = (p or "").upper().strip()
    if p in ("GK", "G", "GOALKEEPER"):
        return "GK"
    if p in ("RB", "LB", "CB", "RWB", "LWB", "WB", "DEF", "DF", "SW", "RCB", "LCB"):
        return "DF"
    if p in ("CDM", "CM", "CAM", "DM", "AM", "LM", "RM", "MID", "MF", "RCM", "LCM"):
        return "MF"
    if p in ("ST", "CF", "LW", "RW", "FWD", "FW", "SS", "RS", "LS", "FORWARD"):
        return "FW"
    return {"G": "GK", "D": "DF", "M": "MF", "F": "FW"}.get(p[:1], "MF")


def _clamp(x: float) -> float:
    return max(CLAMP_LO, min(CLAMP_HI, x))


def load_player_club() -> dict:
    if PLAYER_CLUB_FILE.exists():
        return json.loads(PLAYER_CLUB_FILE.read_text())
    return {}


def league_strength(club_map: dict, pid: str) -> float:
    rec = club_map.get(str(pid)) or {}
    country = (rec.get("club_country") or "").strip()
    return LEAGUE_STRENGTH.get(country, DEFAULT_LEAGUE_STRENGTH)


def load_params() -> tuple[float, float]:
    """(g_att, g_def) — fitted values if present, else hand defaults."""
    if PARAMS_FILE.exists():
        p = json.loads(PARAMS_FILE.read_text())
        return float(p.get("g_att", DEFAULT_G_ATT)), float(p.get("g_def", DEFAULT_G_DEF))
    return DEFAULT_G_ATT, DEFAULT_G_DEF


def team_deltas(xi: list[dict], form_of, club_map: dict) -> tuple[float, float, int]:
    """League-weighted, gain-independent (att_delta, def_delta, matched).

    xi items: {"player_id", "pos"}. form_of(player_id) -> form dict | None.
    A weak-league XI yields smaller deltas (its form shrinks toward neutral)."""
    na = da = nd = dd = 0.0
    matched = 0
    for pl in xi:
        f = form_of(pl["player_id"])
        if not f:
            continue
        matched += 1
        pos = pl["pos"]
        lw = league_strength(club_map, pl["player_id"])
        bxg, bxa = POS_BASELINE.get(pos, POS_BASELINE["MF"])
        rt = f["rating"] - RATING_BASELINE
        att = rt + 1.5 * ((f["xg90"] + f["xa90"]) - (bxg + bxa))
        na += W_ATT[pos] * lw * att
        da += W_ATT[pos]
        nd += W_DEF[pos] * lw * rt
        dd += W_DEF[pos]
    att_d = na / da if da else 0.0
    def_d = nd / dd if dd else 0.0
    return att_d, def_d, matched


def multipliers(att_d: float, def_d: float,
                g_att: float, g_def: float) -> tuple[float, float]:
    """(attack_mult, defense_mult) from deltas + gains. defense_mult<1 = concede less."""
    return _clamp(1 + g_att * att_d), _clamp(1 - g_def * def_d)

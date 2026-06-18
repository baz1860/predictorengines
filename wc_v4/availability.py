"""M3 — Player availability & replacement value (World Cup).

Goal (V4_PLAN.md M3): "price absences like a bookmaker, not as a flat
team-strength nudge." V3 already turns confirmed absences into an Elo/att/def
adjustment (`squads.py` -> `data/squad_ratings.csv`). This module adds the parts a
bookmaker reasons about on top of that single number:

  * expected-lineup CONFIDENCE from how certain each reported absence is;
  * replacement value (the squad-power drop already encodes bench drop-off);
  * goalkeeper-specific impact (a missing GK is not a missing winger);
  * attack / defence / set-piece contribution splits;
  * rotation likelihood from fixture congestion;
  * and — the key M3 acceptance — UNCERTAINTY: when lineup confidence is low, the
    availability adjustment is returned as a band, not a point, so staking can
    widen out (M7) instead of trusting a shaky lineup read.

Everything here is REPORT-ONLY by default (guardrails #1 and #6): it surfaces
signal and uncertainty; it does not change a V3 default until the validation
harness shows it helps on held-out data.

Reuses `squads.py`'s small loaders (`data/squad_ratings.csv`, `data/squads.csv`,
`data/absences.csv`) — no need to touch the 11 MB EA player file here.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import squads as SQ  # noqa: E402  (load_absences, load_adj_split, POS_DEF_SHARE, norm)

DATA = ROOT / "data"
SQUADS_CSV = DATA / "squads.csv"
SQUAD_RATINGS = DATA / "squad_ratings.csv"

# Elo points of availability uncertainty contributed by a single *doubtful*
# absence, and a floor of model risk even when every absence is certain.
_SD_PER_DOUBTFUL = 6.0
_SD_FLOOR = 2.0

# Note keywords. "Certain out" → confident the player is unavailable; "doubtful"
# → the lineup read itself is uncertain (return possible, fitness test, etc.).
_CERTAIN_OUT = re.compile(
    r"out of (the )?(world cup|tournament|season)|ruled out|acl|cruciate|"
    r"season-ending|will miss the (world cup|tournament)|torn", re.I)
_DOUBTFUL = re.compile(
    r"possible return|targeting return|could return|doubt|fitness test|"
    r"expected to (return|be fit)|99%|race against time|late test|"
    r"out of (the )?opener|hamstring|knock|assessed", re.I)


# ── absences with certainty classification ────────────────────────────────────
def _nkey(name: Any) -> str:
    """Normalised whitespace-joined player key (squads.norm returns tokens)."""
    return " ".join(SQ.norm(name))


def _absences_df() -> pd.DataFrame:
    """team, player, note for every current absence (manual + API merged by
    squads.load_absences, which returns a DataFrame), tagged certain / doubtful
    from the note text."""
    raw = SQ.load_absences()  # DataFrame[team, player, note]
    rows: list[dict] = []
    for r in raw.itertuples(index=False):
        note = str(getattr(r, "note", "") or "")
        certain = bool(_CERTAIN_OUT.search(note))
        doubtful = bool(_DOUBTFUL.search(note)) and not certain
        rows.append({"team": str(r.team), "player": str(r.player), "note": note,
                     "certain_out": certain, "doubtful": doubtful})
    return pd.DataFrame(rows, columns=["team", "player", "note",
                                       "certain_out", "doubtful"])


# ── squad positions (for GK detection & per-position depth) ───────────────────
def _positions() -> dict[str, dict[str, str]]:
    """team -> {normalised player name -> position} from data/squads.csv."""
    if not SQUADS_CSV.exists():
        return {}
    df = pd.read_csv(SQUADS_CSV)
    out: dict[str, dict[str, str]] = {}
    for r in df.itertuples(index=False):
        out.setdefault(r.team, {})[_nkey(r.player)] = str(r.pos).upper()
    return out


def _position_of(team: str, player: str, pos_map: dict) -> str | None:
    team_pos = pos_map.get(team, {})
    key = _nkey(player)
    if key in team_pos:
        return team_pos[key]
    # fall back to a loose surname match (absence names are messier than squads)
    sk = key.split()[-1] if key.split() else key
    for name, pos in team_pos.items():
        if sk and sk in name.split():
            return pos
    return None


# ── lineup confidence ─────────────────────────────────────────────────────────
def lineup_confidence(team: str, absences: pd.DataFrame | None = None) -> dict:
    """How confident is the expected-lineup read for `team`?  In [0, 1].

    Full confidence (1.0) means no doubtful situations. Each *doubtful* absence
    (possible-return, fitness test, day-to-day knock) erodes confidence; certain-
    out players do NOT erode it (we know they're gone — that's a clean signal).
    """
    a = absences if absences is not None else _absences_df()
    ta = a[a["team"] == team] if not a.empty else a
    n_certain = int(ta["certain_out"].sum()) if not ta.empty else 0
    n_doubtful = int(ta["doubtful"].sum()) if not ta.empty else 0
    # logistic-ish erosion: each doubtful case removes a chunk of confidence
    conf = float(np.clip(1.0 - 0.18 * n_doubtful, 0.25, 1.0))
    return {"team": team, "confidence": round(conf, 3),
            "n_certain_out": n_certain, "n_doubtful": n_doubtful}


# ── goalkeeper-specific impact ────────────────────────────────────────────────
def gk_impact(team: str, absences: pd.DataFrame | None = None,
              pos_map: dict | None = None) -> dict:
    """Is a goalkeeper among the absences?  A missing first-choice GK carries
    outsize, defence-heavy impact (POS_DEF_SHARE['GK'] = 0.75), so we flag it
    explicitly rather than letting it average out across the squad."""
    a = absences if absences is not None else _absences_df()
    pos_map = pos_map if pos_map is not None else _positions()
    ta = a[a["team"] == team] if not a.empty else a
    gk_out = [r.player for r in ta.itertuples(index=False)
              if _position_of(team, r.player, pos_map) == "GK"]
    return {"team": team, "gk_absent": bool(gk_out), "keepers_out": gk_out,
            "def_share": SQ.POS_DEF_SHARE["GK"]}


# ── contribution splits + replacement value ───────────────────────────────────
def _ratings_row(team: str) -> dict | None:
    if not SQUAD_RATINGS.exists():
        return None
    df = pd.read_csv(SQUAD_RATINGS)
    row = df[df["team"] == team]
    return row.iloc[0].to_dict() if not row.empty else None


def contribution_splits(team: str) -> dict:
    """Split the team's availability adjustment into attack / defence / set-piece.

    att_adj and def_adj come straight from V3's position-aware split. Set-piece
    contribution is approximated from the defensive share (aerial/GK absences
    drive set-piece concessions) and is explicitly report-only — we don't have a
    set-piece dataset for international squads.
    """
    r = _ratings_row(team)
    if not r:
        return {"team": team, "available": False}
    att = float(r.get("att_adj", 0.0) or 0.0)
    dfn = float(r.get("def_adj", 0.0) or 0.0)
    setp = round(0.5 * dfn, 3)  # proxy — report-only
    return {"team": team, "available": True,
            "attack_adj": round(att, 3), "defence_adj": round(dfn, 3),
            "set_piece_adj_proxy": setp, "set_piece_status": "report_only"}


def replacement_value(team: str) -> dict:
    """Squad-power drop from full strength to currently-available, i.e. the value
    of who's missing net of who replaces them. `squad_ratings.csv` already encodes
    bench drop-off (power = best-18 mean), so the gap IS the replacement cost."""
    r = _ratings_row(team)
    if not r:
        return {"team": team, "available": False}
    pf = float(r.get("power_full", 0.0) or 0.0)
    pa = float(r.get("power_avail", 0.0) or 0.0)
    return {"team": team, "available": True,
            "power_full": round(pf, 2), "power_available": round(pa, 2),
            "replacement_drop": round(pf - pa, 2),
            "n_out": int(r.get("n_out", 0) or 0),
            "elo_adj": round(float(r.get("elo_adj", 0.0) or 0.0), 2)}


# ── rotation likelihood ───────────────────────────────────────────────────────
def rotation_likelihood(congestion: float | None) -> dict:
    """P(notable rotation) from recent fixture congestion (matches in the trailing
    ~14 days, from the M1 feature store). Heuristic and report-only: international
    windows rarely congest, so this mostly matters for club soccer, but it's wired
    here so the same call works across engines."""
    if congestion is None or (isinstance(congestion, float) and np.isnan(congestion)):
        return {"rotation_prob": None, "status": "no_data"}
    c = float(congestion)
    prob = float(np.clip(0.08 * max(c - 1.0, 0.0), 0.0, 0.6))
    return {"rotation_prob": round(prob, 3), "congestion": c,
            "status": "report_only"}


# ── the headline: availability adjustment WITH an uncertainty band ────────────
def availability_adjustment(team: str, congestion: float | None = None) -> dict:
    """Point estimate of the availability Elo adjustment PLUS an uncertainty band.

    The point estimate is V3's `elo_adj`. The band widens when lineup confidence
    is low (doubtful absences) or a goalkeeper is involved, so downstream staking
    can haircut size when the lineup read is shaky — the M3 acceptance that "the
    model reports uncertainty when lineup confidence is low".
    """
    a = _absences_df()
    pos_map = _positions()
    conf = lineup_confidence(team, a)
    gk = gk_impact(team, a, pos_map)
    rv = replacement_value(team)
    splits = contribution_splits(team)
    rot = rotation_likelihood(congestion)

    adj = rv.get("elo_adj", 0.0) if rv.get("available") else 0.0
    # SD grows with doubtful absences and a touch more if a keeper is in doubt.
    sd = _SD_FLOOR + _SD_PER_DOUBTFUL * conf["n_doubtful"]
    if gk["gk_absent"]:
        sd += _SD_PER_DOUBTFUL  # keeper situations are higher-variance
    sd = round(float(sd), 2)
    return {
        "team": team,
        "elo_adj": adj,
        "elo_adj_low": round(adj - sd, 2),
        "elo_adj_high": round(adj + sd, 2),
        "uncertainty_sd": sd,
        "lineup_confidence": conf["confidence"],
        "n_certain_out": conf["n_certain_out"],
        "n_doubtful": conf["n_doubtful"],
        "gk_absent": gk["gk_absent"],
        "replacement_drop": rv.get("replacement_drop"),
        "contribution": splits,
        "rotation": rot,
        "status": "report_only",
    }


def team_report(team: str, congestion: float | None = None) -> dict:
    """Everything M3 knows about one team's availability, for the explainability
    surface (guardrail #4)."""
    return availability_adjustment(team, congestion)


if __name__ == "__main__":  # pragma: no cover — manual smoke
    import argparse, json
    ap = argparse.ArgumentParser(description="V4 M3 availability report")
    ap.add_argument("team", nargs="?", help="team to report (default: all with absences)")
    args = ap.parse_args()
    if args.team:
        print(json.dumps(team_report(args.team), indent=2))
    else:
        a = _absences_df()
        for t in sorted(a["team"].unique()):
            r = availability_adjustment(t)
            print(f"{t:24s} adj={r['elo_adj']:+5.1f} "
                  f"[{r['elo_adj_low']:+5.1f},{r['elo_adj_high']:+5.1f}] "
                  f"conf={r['lineup_confidence']:.2f} "
                  f"doubtful={r['n_doubtful']} gk={r['gk_absent']}")

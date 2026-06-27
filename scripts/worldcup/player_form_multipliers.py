#!/usr/bin/env python3
"""Phase 2 — per-team attack/defence multipliers from player form.

Turns the Phase 1 form store into the same shape the model already understands:
per-team ``attack_mult`` (scales the team's own goals) and ``defense_mult``
(scales the OPPONENT's goals) — the convention used by club_soccer's
PlayerFeatureStore and squads.adjusted_sources.

Then it lays the form-based multipliers next to the existing **EA squad gap**
(squads.load_adj_split, availability-only) so we can see where form disagrees with
the static prior. That divergence is the whole question: if form just echoes the
EA gap it adds nothing; if it moves independently, it's new signal.

This is a REPORT. It does not touch predictions.

Usage:
    python3 -m scripts.worldcup.player_form_multipliers            # all teams in lineups.csv
    python3 -m scripts.worldcup.player_form_multipliers --team Argentina
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from scripts.worldcup.player_form import (  # noqa: E402
    CACHE, POS_BASELINE, RATING_BASELINE, compute_form,
    fetch_player_stats, _client,
)

LINEUPS = HERE / "data" / "worldcup" / "lineups.csv"

# attack vs defence influence by position (how much a player's form feeds each side)
W_ATT = {"GK": 0.0, "DF": 0.3, "MF": 1.0, "FW": 1.3}
W_DEF = {"GK": 1.3, "DF": 1.2, "MF": 0.6, "FW": 0.2}

# form -> multiplier gains (tuned so typical spreads sit inside the clamp)
G_ATT = 0.12
G_DEF = 0.10
CLAMP_LO, CLAMP_HI = 0.80, 1.25


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


# ── inputs ────────────────────────────────────────────────────────────────────

def load_xis() -> dict[str, list[dict]]:
    """team -> starter dicts for that team's MOST RECENT confirmed match.

    lineups.csv may hold several matches per team; we want the latest XI, not a
    concatenation of every game they played."""
    if not LINEUPS.exists():
        sys.exit(f"missing {LINEUPS}")
    rows = list(csv.DictReader(LINEUPS.open(newline="")))
    # latest fixture per team, keyed by (match_date, fixture_id)
    latest: dict[str, tuple] = {}
    for r in rows:
        team = r["team"]
        k = (r.get("match_date", ""), r.get("provider_fixture_id", ""))
        if team not in latest or k > latest[team]:
            latest[team] = k
    xis: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if str(r.get("starter")).strip().lower() != "true":
            continue
        team = r["team"]
        if (r.get("match_date", ""), r.get("provider_fixture_id", "")) != latest[team]:
            continue
        xis[team].append({
            "player_id": (r.get("provider_player_id") or "").strip(),
            "name": r.get("player", ""),
            "pos": _norm_pos(r.get("position")),
        })
    return xis


def load_form_cache() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    return {}


def get_form(pid: str, pos: str, cache: dict, get, key) -> dict | None:
    """Form for one player: cache first, else fetch live by player_id."""
    if pid in cache:
        return cache[pid]
    if not pid.isdigit():
        return None
    rows = fetch_player_stats(get, key, int(pid))
    if not rows:
        return None
    return compute_form(rows, pos)


# ── multipliers ───────────────────────────────────────────────────────────────

def team_multipliers(xi: list[dict], cache: dict, get, key) -> dict:
    """League-weighted, fitted-gain multipliers (via form_config — single source)."""
    from scripts.worldcup import form_config as FC
    club = FC.load_player_club()
    g_att, g_def = FC.load_params()
    forms = {}
    for pl in xi:
        f = get_form(pl["player_id"], pl["pos"], cache, get, key)
        if f:
            forms[pl["player_id"]] = f
    att_delta, def_delta, matched = FC.team_deltas(xi, forms.get, club)
    am, dm = FC.multipliers(att_delta, def_delta, g_att, g_def)
    return {
        "attack_mult": round(am, 3),
        "defense_mult": round(dm, 3),
        "att_delta": round(att_delta, 3),
        "def_delta": round(def_delta, 3),
        "matched": matched,
        "xi_size": len(xi),
    }


# ── EA-gap comparison ─────────────────────────────────────────────────────────

def ea_multipliers() -> tuple[dict, dict, str]:
    """Return (attack_mult_by_team, defense_mult_by_team, note) from the EA squad
    gap, converted to the same lambda-multiplier space for comparison."""
    try:
        import numpy as np
        from engines.worldcup.squads import load_adj_split
        from engines.worldcup.predictor import (load_matches, compute_elo,
                                                 fit_goal_model)
        attA, defD = load_adj_split()
        if not attA:
            return {}, {}, "EA gap not refreshed (run squads.py)"
        played, _ = load_matches()
        _, played = compute_elo(played)
        k = fit_goal_model(played)[1] / 400.0
        a_mult = {t: float(np.exp(2 * k * v)) for t, v in attA.items()}
        d_mult = {t: float(np.exp(-2 * k * v)) for t, v in defD.items()}
        return a_mult, d_mult, ""
    except Exception as exc:  # noqa: BLE001
        return {}, {}, f"EA gap unavailable: {exc}"


# ── report ────────────────────────────────────────────────────────────────────

# BSD team name -> predictor (results.csv) name, so the edge path can look them up
TEAM_ALIAS = {
    "Côte d'Ivoire": "Ivory Coast", "Czechia": "Czech Republic",
    "Türkiye": "Turkey", "Cabo Verde": "Cape Verde",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina", "USA": "United States",
}
FORM_MULTS_OUT = HERE / "data" / "worldcup" / "form_multipliers.json"


def write_multipliers(xis: dict, cache: dict, get, key) -> Path:
    out = {}
    for team, xi in xis.items():
        m = team_multipliers(xi, cache, get, key)
        name = TEAM_ALIAS.get(team, team)
        out[name] = {"attack_mult": m["attack_mult"],
                     "defense_mult": m["defense_mult"],
                     "matched": m["matched"], "xi_size": m["xi_size"]}
    FORM_MULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    FORM_MULTS_OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return FORM_MULTS_OUT


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2 form multipliers (PoC report)")
    ap.add_argument("--team", help="limit to one team name")
    ap.add_argument("--write", action="store_true",
                    help="write data/worldcup/form_multipliers.json for --form-adj")
    args = ap.parse_args()

    xis = load_xis()
    if args.team:
        xis = {t: v for t, v in xis.items() if t.lower() == args.team.lower()}
        if not xis:
            sys.exit(f"team {args.team!r} not in lineups.csv")
    cache = load_form_cache()
    get, key = _client()

    if args.write:
        path = write_multipliers(xis, cache, get, key)
        print(f"wrote {len(xis)} team multipliers -> {path}")
        return

    ea_att, ea_def, ea_note = ea_multipliers()

    print(f"\nForm-based team multipliers (XI from lineups.csv; cache={CACHE.name})")
    print("attack_mult scales own goals · defense_mult scales OPPONENT goals "
          "(>1 = concede more)")
    if ea_note:
        print(f"[EA gap comparison: {ea_note}]")
    print(f"\n{'team':14} {'matched':>7} │ {'FORM att':>8} {'FORM def':>8} │ "
          f"{'EA att':>7} {'EA def':>7} │ verdict")
    print("─" * 78)

    for team, xi in xis.items():
        m = team_multipliers(xi, cache, get, key)
        ea_a = ea_att.get(team)
        ea_d = ea_def.get(team)
        ea_a_s = f"{ea_a:.3f}" if ea_a is not None else "  —  "
        ea_d_s = f"{ea_d:.3f}" if ea_d is not None else "  —  "
        # does form move where the EA gap is flat?
        ea_flat = (ea_a is None or abs(ea_a - 1) < 0.01) and \
                  (ea_d is None or abs(ea_d - 1) < 0.01)
        form_moves = abs(m["attack_mult"] - 1) > 0.02 or abs(m["defense_mult"] - 1) > 0.02
        verdict = ("NEW SIGNAL (form moves, EA flat)" if ea_flat and form_moves
                   else "form ≈ neutral" if not form_moves
                   else "both move")
        print(f"{team[:14]:14} {m['matched']:>3}/{m['xi_size']:<3} │ "
              f"{m['attack_mult']:>8.3f} {m['defense_mult']:>8.3f} │ "
              f"{ea_a_s:>7} {ea_d_s:>7} │ {verdict}")

    print()


if __name__ == "__main__":
    main()

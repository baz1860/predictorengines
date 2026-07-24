#!/usr/bin/env python3
"""Evidence coverage: how much the model actually knows about a team.

Motivating failure (2026-07-21): Sturm Graz were priced at 18% away to Hearts
and won 4-0. Sturm Graz's Austrian Bundesliga is not ingested, so the club
existed in the model only through 24 UEFA matches. Every rating lookup in
model.py uses a silent default --

    params["elo"].get(home, BASE_ELO)      # 1500.0
    params["attack"].get(home, 0.0)        # league-average attack

-- so an unrated club is assigned the identity of a perfectly average team,
where "average" is the pooled mean across every fitted team INCLUDING English
League Two. Nothing warned, nothing flagged, and the card reported
lineup_confidence 1.0 for the fixture.

This module makes that absence visible. It computes, per team, how much
evidence the fit actually rests on, and assigns a tier:

    full       enough recent matches AND domestic-league data
    thin       known, but under-evidenced -- ratings are weakly identified
    defaulted  absent from the fit entirely -- ratings are pure defaults

A team playing only in Europe is NEVER "full" no matter how many matches it
has: 24 UEFA matches against opposition the model also cannot rate does not
identify a rating. That is precisely the Sturm Graz signature, and a naive
match-count threshold would have called it fully evidenced.

P0 is report-only. Tiers are computed, attached to predictions and surfaced
on bet suggestions; no probability is altered and no stake is blocked here.
Variance inflation for thin/defaulted teams lands in P5, after the baseline
has been measured against the tiers this module produces.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from .competitions import get as _get_comp

# ── thresholds ────────────────────────────────────────────────────────────
# A team needs this many matches inside RECENT_WINDOW_DAYS to be "full".
# 20 is roughly half a domestic league season -- enough for the attack/defence
# shrinkage (prior weight 4 matches) to be dominated by real observations
# rather than by the global prior.
MIN_RECENT_MATCHES = 20
RECENT_WINDOW_DAYS = 730          # two seasons

# Below this, even the "thin" label understates it -- the team is barely
# distinguishable from a default-rated one.
VERY_THIN_MATCHES = 5

TIER_FULL = "full"
TIER_THIN = "thin"
TIER_DEFAULTED = "defaulted"

_TIER_ORDER = {TIER_FULL: 0, TIER_THIN: 1, TIER_DEFAULTED: 2}


def _is_domestic(competition: str | None) -> bool:
    """Domestic = anything that is not a UEFA club competition.

    Unknown competitions count as domestic: during the 55-league expansion a
    league may appear in fixtures before it lands in the registry, and the
    safe reading of an unrecognised competition is "some domestic league",
    not "Europe". Mislabelling a domestic league as European would wrongly
    hold a well-evidenced team at `thin`; the reverse would wrongly promote
    a Sturm-Graz-shaped team to `full`. Prefer the conservative error.
    """
    comp = _get_comp(competition)
    if comp is None:
        return True
    return comp.kind != "europe"


def build_team_evidence(df: pd.DataFrame, as_of=None) -> dict[str, dict]:
    """Summarise per-team evidence from the played-fixtures frame.

    Called by model.fit(); the result is stored in model_params.json under
    "team_evidence" so that pricing can read it without touching fixtures.csv.
    """
    if df.empty:
        return {}
    dates = pd.to_datetime(df["date"], errors="coerce", utc=True)
    as_of = pd.Timestamp(as_of, tz="UTC") if as_of is not None else dates.max()
    cutoff = as_of - pd.Timedelta(days=RECENT_WINDOW_DAYS)

    ev: dict[str, dict] = {}

    def _touch(team: str) -> dict:
        rec = ev.get(team)
        if rec is None:
            rec = ev[team] = {"n": 0, "n_recent": 0, "n_domestic": 0,
                              "n_domestic_recent": 0, "comps": set(),
                              "last_date": None}
        return rec

    comp_domestic = {c: _is_domestic(c) for c in df["competition"].dropna().unique()}

    for comp, date, home, away in zip(df["competition"], dates,
                                      df["home"], df["away"]):
        recent = bool(pd.notna(date) and date >= cutoff)
        domestic = comp_domestic.get(comp, True)
        for team in (home, away):
            rec = _touch(team)
            rec["n"] += 1
            if recent:
                rec["n_recent"] += 1
            if domestic:
                rec["n_domestic"] += 1
                if recent:
                    rec["n_domestic_recent"] += 1
            if pd.notna(comp):
                rec["comps"].add(str(comp))
            if pd.notna(date):
                prev = rec["last_date"]
                if prev is None or date > prev:
                    rec["last_date"] = date

    out: dict[str, dict] = {}
    for team, rec in ev.items():
        last = rec["last_date"]
        out[team] = {
            "n": int(rec["n"]),
            "n_recent": int(rec["n_recent"]),
            "n_domestic": int(rec["n_domestic"]),
            "n_domestic_recent": int(rec["n_domestic_recent"]),
            "comps": sorted(rec["comps"]),
            "has_domestic": bool(rec["n_domestic"] > 0),
            "last_date": None if last is None else last.strftime("%Y-%m-%d"),
        }
    return out


def team_tier(evidence: dict | None) -> tuple[str, list[str]]:
    """Classify one team. Returns (tier, reasons)."""
    if not evidence:
        return TIER_DEFAULTED, ["no matches in the fitted dataset; "
                                "ratings are defaults (elo 1500, attack/defence 0)"]
    reasons: list[str] = []
    n_recent = int(evidence.get("n_recent", 0))
    n_dom_recent = int(evidence.get("n_domestic_recent", 0))

    if not evidence.get("has_domestic", False):
        reasons.append("no domestic-league data; known only from European "
                       "matches against opposition that is also unrated")
    if n_recent < MIN_RECENT_MATCHES:
        reasons.append(f"only {n_recent} match(es) in the last "
                       f"{RECENT_WINDOW_DAYS // 365} seasons "
                       f"(need {MIN_RECENT_MATCHES})")
    elif n_dom_recent < MIN_RECENT_MATCHES:
        reasons.append(f"only {n_dom_recent} recent domestic match(es) "
                       f"(need {MIN_RECENT_MATCHES})")
    if int(evidence.get("n", 0)) <= VERY_THIN_MATCHES:
        reasons.append("sample is near-negligible; rating is effectively the "
                       "league-agnostic default")

    return (TIER_FULL, []) if not reasons else (TIER_THIN, reasons)


def team_coverage(params: dict, team: str) -> dict:
    """Coverage record for one team, from a loaded params dict."""
    store = params.get("team_evidence") or {}
    ev = store.get(team)
    rated = team in (params.get("elo") or {})
    tier, reasons = team_tier(ev)
    # A team can be absent from team_evidence but present in the rating maps
    # only if params predate this module; treat that as thin, not defaulted,
    # so a stale artifact degrades to caution rather than to a false alarm.
    if ev is None and rated and "team_evidence" not in params:
        tier, reasons = TIER_THIN, ["params artifact predates coverage "
                                    "instrumentation; evidence unknown"]
    return {
        "team": team,
        "tier": tier,
        "rated": bool(rated),
        "n": int((ev or {}).get("n", 0)),
        "n_recent": int((ev or {}).get("n_recent", 0)),
        "n_domestic_recent": int((ev or {}).get("n_domestic_recent", 0)),
        "has_domestic": bool((ev or {}).get("has_domestic", False)),
        "competitions": list((ev or {}).get("comps", [])),
        "last_date": (ev or {}).get("last_date"),
        "reasons": reasons,
    }


def worst_tier(*tiers: str) -> str:
    """A match is only as trustworthy as its least-evidenced side."""
    return max(tiers, key=lambda t: _TIER_ORDER.get(t, 1))


def match_coverage(params: dict, home: str, away: str) -> dict:
    """Coverage for a fixture. `tier` is the worse of the two sides."""
    h = team_coverage(params, home)
    a = team_coverage(params, away)
    tier = worst_tier(h["tier"], a["tier"])
    notes: list[str] = []
    for side, rec in (("home", h), ("away", a)):
        for reason in rec["reasons"]:
            notes.append(f"{rec['team']} ({side}): {reason}")
    return {"tier": tier, "home": h, "away": a, "notes": notes,
            "reliable": tier == TIER_FULL}


def summarise(params: dict) -> dict:
    """Fleet-wide coverage counts -- for health.py and the daily run log."""
    store = params.get("team_evidence") or {}
    counts = {TIER_FULL: 0, TIER_THIN: 0}
    no_domestic = 0
    for _team, ev in store.items():
        tier, _ = team_tier(ev)
        counts[tier] = counts.get(tier, 0) + 1
        if not ev.get("has_domestic", False):
            no_domestic += 1
    return {"teams": len(store), "by_tier": counts,
            "euro_only_teams": no_domestic}


def main() -> None:
    import argparse
    import json

    from . import model as M

    ap = argparse.ArgumentParser(description="Report model evidence coverage.")
    ap.add_argument("--team", help="show coverage for one team")
    ap.add_argument("--match", nargs=2, metavar=("HOME", "AWAY"))
    ap.add_argument("--list-thin", action="store_true",
                    help="list every team below the full-evidence bar")
    args = ap.parse_args()

    params = M.load_params()
    if args.team:
        print(json.dumps(team_coverage(params, args.team), indent=2))
        return
    if args.match:
        print(json.dumps(match_coverage(params, *args.match), indent=2))
        return

    summary = summarise(params)
    print(f"teams fitted          : {summary['teams']}")
    print(f"  full evidence       : {summary['by_tier'].get(TIER_FULL, 0)}")
    print(f"  thin evidence       : {summary['by_tier'].get(TIER_THIN, 0)}")
    print(f"  no domestic data    : {summary['euro_only_teams']}")

    if args.list_thin:
        store = params.get("team_evidence") or {}
        rows = []
        for team, ev in store.items():
            tier, reasons = team_tier(ev)
            if tier != TIER_FULL:
                rows.append((ev.get("n", 0), team, "; ".join(reasons)))
        rows.sort()
        print()
        for n, team, why in rows:
            print(f"  {n:>4}  {team:<38} {why}")


if __name__ == "__main__":
    main()

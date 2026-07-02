#!/usr/bin/env python3
"""Squads and transfers, derived by OBSERVATION rather than a transfer feed.

A player belongs to the club he most recently appeared for (per the v2
player_stats_cache.json apps history — see player_features.py). No transfer
feed is consulted; a squad list is just "who showed up for whom, lately",
and a transfer is detected when that changes between a player's two most
recent appearances.

Outputs:
  data/squads_club.csv        current squad per team (report/card use)
  data/transfers_detected.csv report-only: recently-observed team changes
  data/transfers_manual.csv   operator-maintained overrides for a move known
                               ahead of a debut (empty header template,
                               applied on top of the observed squad)
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .player_features import PlayerFeatureStore

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SQUADS_CSV = DATA / "squads_club.csv"
TRANSFERS_DETECTED_CSV = DATA / "transfers_detected.csv"
TRANSFERS_MANUAL_CSV = DATA / "transfers_manual.csv"

CURRENT_SQUAD_WINDOW_DAYS = 120


def season_start(today) -> str:
    year = today.year if today.month >= 7 else today.year - 1
    return f"{year}-07-01"


def _load_manual_overrides() -> pd.DataFrame:
    if not TRANSFERS_MANUAL_CSV.exists():
        return pd.DataFrame(columns=["effective_date", "player", "from_team", "to_team"])
    return pd.read_csv(TRANSFERS_MANUAL_CSV)


def squad_asof(store: PlayerFeatureStore, asof: str,
              apply_manual: bool = True) -> pd.DataFrame:
    """Point-in-time squads: for each player, using only apps dated <= asof,
    the current squad of club T = players whose most recent such app (within
    CURRENT_SQUAD_WINDOW_DAYS of asof) was for T.

    Shared by build_squads() (asof=today, manual overrides applied) and
    minutes.py's xi_load computation (asof=an arbitrary historical match
    date, manual overrides OFF — a backtest must not see a real-world
    operator override that wasn't knowable at that point in time).
    """
    asof_date = pd.Timestamp(asof).date()
    cutoff = str(asof_date - timedelta(days=CURRENT_SQUAD_WINDOW_DAYS))
    szn_start = season_start(asof_date)
    manual_latest: dict[str, tuple[str, str]] = {}
    if apply_manual:
        manual = _load_manual_overrides()
        for r in manual.itertuples(index=False):
            eff = str(r.effective_date)
            if eff > str(asof_date):
                continue
            prev = manual_latest.get(r.player)
            if prev is None or eff > prev[0]:
                manual_latest[r.player] = (eff, r.to_team)

    rows = []
    for _, rec in store._player_records():
        apps = sorted((a for a in rec.get("apps", []) if a["date"] <= str(asof_date)),
                      key=lambda a: a["date"])
        if not apps:
            continue
        last = apps[-1]
        if last["date"] < cutoff:
            continue  # no appearance in the current-squad window — drop
        team = last["team"]
        if rec["name"] in manual_latest:
            team = manual_latest[rec["name"]][1]
        season_apps = [a for a in apps if a["date"] >= szn_start]
        thirty_cut = str(asof_date - timedelta(days=30))
        rows.append({
            "team": team,
            "player": rec["name"],
            "pos": rec.get("pos", "MF"),
            "last_seen": last["date"],
            "apps_season": len(season_apps),
            "mins_season": round(sum(a["mins"] for a in season_apps), 1),
            "mins_30d": round(sum(a["mins"] for a in apps if a["date"] >= thirty_cut), 1),
        })
    df = pd.DataFrame(rows, columns=["team", "player", "pos", "last_seen",
                                     "apps_season", "mins_season", "mins_30d"])
    return df.sort_values(["team", "mins_season"], ascending=[True, False]).reset_index(drop=True)


def build_squads(store: PlayerFeatureStore | None = None) -> pd.DataFrame:
    """Current squad per club from observed appearances (asof=today, with
    manual transfer overrides applied)."""
    store = store or PlayerFeatureStore().load()
    today = str(datetime.now(timezone.utc).date())
    return squad_asof(store, today, apply_manual=True)


def detect_transfers(store: PlayerFeatureStore | None = None) -> pd.DataFrame:
    """Players whose latest appearance's team differs from their previous
    appearance's team — report-only, no automatic model adjustment."""
    store = store or PlayerFeatureStore().load()
    rows = []
    for _, rec in store._player_records():
        apps = sorted(rec.get("apps", []), key=lambda a: a["date"])
        if len(apps) < 2:
            continue
        prev_team, last_team = apps[-2]["team"], apps[-1]["team"]
        if prev_team and last_team and prev_team != last_team:
            rows.append({"date": apps[-1]["date"], "player": rec["name"],
                        "from_team": prev_team, "to_team": last_team})
    df = pd.DataFrame(rows, columns=["date", "player", "from_team", "to_team"])
    return df.sort_values("date", ascending=False).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="write data/squads_club.csv and data/transfers_detected.csv")
    args = ap.parse_args()
    store = PlayerFeatureStore().load()
    squads = build_squads(store)
    transfers = detect_transfers(store)
    print(f"Squads: {len(squads)} players across {squads['team'].nunique()} teams")
    print(f"Detected transfers: {len(transfers)}")
    if not transfers.empty:
        print(transfers.head(10).to_string(index=False))
    if args.write:
        DATA.mkdir(exist_ok=True)
        squads.to_csv(SQUADS_CSV, index=False)
        transfers.to_csv(TRANSFERS_DETECTED_CSV, index=False)
        if not TRANSFERS_MANUAL_CSV.exists():
            TRANSFERS_MANUAL_CSV.write_text("effective_date,player,from_team,to_team\n")
        print(f"Wrote {SQUADS_CSV.name}, {TRANSFERS_DETECTED_CSV.name}")


if __name__ == "__main__":
    main()

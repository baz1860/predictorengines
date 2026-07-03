#!/usr/bin/env python3
"""Minutes-load features: who's tired, from observed appearance history.

Definitions (exact, per the plan):
  * Per player, as of date D: mins_7d = Σ minutes over apps with
    D-7 < date <= D (same construction for 14d/30d); mins_season = Σ since
    the season boundary (July 1 preceding D).
  * Likely XI of club T as of D = top 11 current-squad players by mins_30d
    (ties -> mins_season). Uses whatever exists if fewer than 11 have data.
  * Team features: xi_load_7d/14d/30d = mean of the likely XI's per-player
    minutes in the window, divided by 90 (units: matches-worth of minutes).
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

from .club_squads import squad_asof, season_start
from .player_features import PlayerFeatureStore

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PLAYER_MINUTES_CSV = DATA / "player_minutes.csv"

LIKELY_XI_SIZE = 11


def _window_minutes(apps: list[dict], asof_date, days: int) -> float:
    """Σ minutes over apps with asof_date - days < date <= asof_date."""
    total = 0.0
    for a in apps:
        d = pd.Timestamp(a["date"]).date()
        if d > asof_date:
            continue
        if (asof_date - d).days < days:
            total += float(a["mins"])
    return total


def player_minutes_row(apps: list[dict], asof: str) -> dict:
    """mins_7d/14d/30d/season + starts_season for one player's apps, using
    only apps dated <= asof (point-in-time safe)."""
    asof_date = pd.Timestamp(asof).date()
    szn_start = season_start(asof_date)
    season_apps = [a for a in apps if a["date"] <= str(asof_date) and a["date"] >= szn_start]
    return {
        "mins_7d": round(_window_minutes(apps, asof_date, 7), 1),
        "mins_14d": round(_window_minutes(apps, asof_date, 14), 1),
        "mins_30d": round(_window_minutes(apps, asof_date, 30), 1),
        "mins_season": round(sum(a["mins"] for a in season_apps), 1),
        "starts_season": len(season_apps),
    }


def build_player_minutes(store: PlayerFeatureStore, asof: str) -> pd.DataFrame:
    """Per-player minutes-load table as of `asof`, for players in that date's
    squads (squad_asof), team attribution via observed appearances."""
    squads = squad_asof(store, asof, apply_manual=(asof >= str(datetime.now(timezone.utc).date())))
    by_name = {rec["name"]: rec for _, rec in store._player_records()}
    rows = []
    for r in squads.itertuples(index=False):
        rec = by_name.get(r.player)
        apps = rec.get("apps", []) if rec else []
        mrow = player_minutes_row(apps, asof)
        rows.append({"asof": asof, "team": r.team, "player": r.player, "pos": r.pos, **mrow})
    return pd.DataFrame(rows, columns=["asof", "team", "player", "pos",
                                       "mins_7d", "mins_14d", "mins_30d",
                                       "mins_season", "starts_season"])


def likely_xi(minutes_df: pd.DataFrame, team: str) -> pd.DataFrame:
    """Top LIKELY_XI_SIZE players of `team` by mins_30d (ties -> mins_season)."""
    sub = minutes_df[minutes_df["team"] == team]
    return sub.sort_values(["mins_30d", "mins_season"], ascending=False).head(LIKELY_XI_SIZE)


def xi_loads(minutes_df: pd.DataFrame, team: str) -> dict[str, float]:
    """xi_load_7d/14d/30d for `team`'s likely XI — mean minutes in each
    window / 90 (units: matches-worth of minutes). 0.0 if no squad data."""
    xi = likely_xi(minutes_df, team)
    if xi.empty:
        return {"xi_load_7d": 0.0, "xi_load_14d": 0.0, "xi_load_30d": 0.0}
    return {
        "xi_load_7d": round(float(xi["mins_7d"].mean()) / 90.0, 4),
        "xi_load_14d": round(float(xi["mins_14d"].mean()) / 90.0, 4),
        "xi_load_30d": round(float(xi["mins_30d"].mean()) / 90.0, 4),
    }


def xi_loads_asof(store: PlayerFeatureStore, team: str, asof: str) -> dict[str, float]:
    """Point-in-time xi_load_7d/14d/30d for one team as of `asof`, for use
    from feature_store.py without materialising the full league-wide table."""
    minutes_df = build_player_minutes(store, asof)
    minutes_df = minutes_df[minutes_df["team"] == team]
    return xi_loads(minutes_df, team) if not minutes_df.empty else \
        {"xi_load_7d": 0.0, "xi_load_14d": 0.0, "xi_load_30d": 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None,
                    help="ISO date to compute as of (default: today)")
    ap.add_argument("--write", action="store_true",
                    help="write data/player_minutes.csv")
    args = ap.parse_args()
    asof = args.asof or str(datetime.now(timezone.utc).date())
    store = PlayerFeatureStore().load()
    df = build_player_minutes(store, asof)
    print(f"Player minutes as of {asof}: {len(df)} players, "
          f"{df['team'].nunique()} teams")
    for team in sorted(df["team"].unique())[:10]:
        loads = xi_loads(df, team)
        print(f"  {team:25s} xi_load_7d={loads['xi_load_7d']:.2f} "
              f"14d={loads['xi_load_14d']:.2f} 30d={loads['xi_load_30d']:.2f}")
    if args.write:
        DATA.mkdir(exist_ok=True)
        df.to_csv(PLAYER_MINUTES_CSV, index=False)
        print(f"Wrote {PLAYER_MINUTES_CSV}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Refresh nfl/data/ from nflverse (no API key needed).

Sources:
  * games + closing lines: nflverse/nfldata `data/games.csv` (1999-> every
    game, spread_line/total_line/moneylines/rest/QB starters/coaches).
  * team-week offense splits: nflverse-data release `stats_team` ->
    `stats_team_week_{season}.csv` (2003->). Defense-allowed splits are
    derived from the opponent's offensive row in the same game_id, so we
    never need to touch raw play-by-play.
  * QB-week splits: nflverse-data release `player_stats` ->
    `stats_player_week_{season}.csv` (2003->), filtered to position == QB.

Usage:
  python3 -m nfl.fetch_data                  # full refresh, 2003-> aggregates
  python3 -m nfl.fetch_data --since 2010      # smaller/faster aggregate refresh
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import urllib.request
from datetime import date

import pandas as pd

from .team_names import full_name

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
# Scratch downloads go to the system tmp dir, not DATA_DIR: some mounted
# filesystems (sandboxed sessions) allow writing but not deleting/renaming, so
# temp files must live somewhere they can be cleaned up.
TMP_DIR = tempfile.gettempdir()

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
STATS_TEAM_WEEK_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/stats_team/"
    "stats_team_week_{season}.csv"
)
STATS_PLAYER_WEEK_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/player_stats/"
    "stats_player_week_{season}.csv"
)

FIRST_EPA_SEASON = 2003  # per plan: fit structural params 2003-2014, walk-forward 2015+


def _download(url: str, dest: str, retries: int = 3) -> None:
    last_err = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "nfl-engine/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r, open(dest, "wb") as f:
                f.write(r.read())
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"failed to download {url}: {last_err}")


def fetch_games() -> pd.DataFrame:
    """Download + normalise games.csv. Splits into completed (data/games.csv)
    and scheduled (data/upcoming.csv), mirroring the cfb engine convention."""
    tmp = os.path.join(TMP_DIR, "_nfl_games_raw.csv")
    os.makedirs(DATA_DIR, exist_ok=True)
    _download(GAMES_URL, tmp)
    g = pd.read_csv(tmp, low_memory=False)
    os.remove(tmp)

    g["date"] = pd.to_datetime(g["gameday"]).dt.date
    g["home"] = g["home_team"].map(full_name)
    g["away"] = g["away_team"].map(full_name)
    g["neutral"] = 0  # nflverse marks neutral-site games via `location` == 'Neutral'
    g.loc[g["location"].astype(str).str.lower() == "neutral", "neutral"] = 1
    g["div_game"] = g["div_game"].fillna(0).astype(int)

    keep = [
        "game_id", "season", "game_type", "week", "date", "home", "away", "neutral",
        "home_score", "away_score", "home_rest", "away_rest", "div_game",
        "home_moneyline", "away_moneyline", "spread_line", "home_spread_odds",
        "away_spread_odds", "total_line", "under_odds", "over_odds",
        "roof", "surface", "temp", "wind",
        "home_qb_id", "away_qb_id", "home_qb_name", "away_qb_name",
        "home_coach", "away_coach", "home_team", "away_team",
    ]
    g = g[keep].rename(columns={"game_type": "season_type",
                                "home_team": "home_abbr", "away_team": "away_abbr"})
    g = g.sort_values("date").reset_index(drop=True)

    completed = g[g["home_score"].notna() & g["away_score"].notna()].copy()
    completed[["home_score", "away_score"]] = completed[["home_score", "away_score"]].astype(int)
    completed.to_csv(os.path.join(DATA_DIR, "games.csv"), index=False)

    upcoming = g[g["home_score"].isna() & (pd.to_datetime(g["date"]) >= pd.Timestamp(date.today()))].copy()
    upcoming.to_csv(os.path.join(DATA_DIR, "upcoming.csv"), index=False)

    print(f"{len(completed)} completed games ({completed['season'].min()}-{completed['season'].max()}) "
          f"-> data/games.csv")
    print(f"{len(upcoming)} upcoming games -> data/upcoming.csv")
    return completed


def fetch_epa_team_week(since: int = FIRST_EPA_SEASON, until: int | None = None) -> pd.DataFrame:
    """Aggregate stats_team_week_{season}.csv (offense splits) into per-team-week
    offense AND defense (opponent's offense in the same game_id) EPA splits."""
    until = until or date.today().year
    frames = []
    for season in range(since, until + 1):
        url = STATS_TEAM_WEEK_URL.format(season=season)
        tmp = os.path.join(TMP_DIR, f"_nfl_stw_{season}.csv")
        try:
            _download(url, tmp)
        except RuntimeError:
            continue  # season not yet released (e.g. current season pre-week-1)
        df = pd.read_csv(tmp, low_memory=False)
        os.remove(tmp)
        df = df[df["season_type"] != "POST"].copy() if "season_type" in df.columns else df
        frames.append(df)
    if not frames:
        raise RuntimeError("no stats_team_week data downloaded")
    stw = pd.concat(frames, ignore_index=True)

    off_cols = ["season", "week", "team", "game_id", "opponent_team",
                "attempts", "sacks_suffered", "passing_epa",
                "carries", "rushing_epa"]
    off = stw[off_cols].copy()
    off["dropbacks"] = off["attempts"].fillna(0) + off["sacks_suffered"].fillna(0)
    off["off_pass_epa"] = off["passing_epa"].fillna(0.0)
    off["off_rush_epa"] = off["rushing_epa"].fillna(0.0)
    off["off_carries"] = off["carries"].fillna(0)
    off = off[["season", "week", "team", "game_id", "opponent_team",
               "dropbacks", "off_pass_epa", "off_carries", "off_rush_epa"]]

    # defense-allowed = opponent's offensive output in the same game
    opp = off.rename(columns={
        "team": "_opp_team", "opponent_team": "team",
        "dropbacks": "def_dropbacks_faced", "off_pass_epa": "def_pass_epa_allowed",
        "off_carries": "def_carries_faced", "off_rush_epa": "def_rush_epa_allowed",
    })[["season", "week", "game_id", "team", "def_dropbacks_faced",
        "def_pass_epa_allowed", "def_carries_faced", "def_rush_epa_allowed"]]

    merged = off.merge(opp, on=["season", "week", "game_id", "team"], how="inner")
    merged["team"] = merged["team"].map(full_name)
    merged = merged.drop(columns=["opponent_team"]).sort_values(["season", "week", "team"])
    dest = os.path.join(DATA_DIR, "epa_team_week.csv")
    merged.to_csv(dest, index=False)
    print(f"{len(merged)} team-week EPA rows ({merged['season'].min()}-{merged['season'].max()}) -> {dest}")
    return merged


def fetch_qb_week(since: int = FIRST_EPA_SEASON, until: int | None = None) -> pd.DataFrame:
    """Aggregate stats_player_week_{season}.csv (QB rows only) into qb_week.csv."""
    until = until or date.today().year
    frames = []
    for season in range(since, until + 1):
        url = STATS_PLAYER_WEEK_URL.format(season=season)
        tmp = os.path.join(TMP_DIR, f"_nfl_spw_{season}.csv")
        try:
            _download(url, tmp)
        except RuntimeError:
            continue
        df = pd.read_csv(tmp, low_memory=False)
        os.remove(tmp)
        df = df[df["position"] == "QB"].copy()
        if "season_type" in df.columns:
            df = df[df["season_type"] != "POST"]
        frames.append(df)
    if not frames:
        raise RuntimeError("no stats_player_week data downloaded")
    spw = pd.concat(frames, ignore_index=True)

    spw["dropbacks"] = spw["attempts"].fillna(0) + spw["sacks_suffered"].fillna(0)
    out = spw[["season", "week", "team", "player_id", "player_display_name",
               "dropbacks", "passing_epa"]].rename(columns={"player_display_name": "qb_name"})
    out = out[out["dropbacks"] > 0].copy()
    out["passing_epa"] = out["passing_epa"].fillna(0.0)
    out["team"] = out["team"].map(full_name)
    out = out.sort_values(["season", "week", "team"])
    dest = os.path.join(DATA_DIR, "qb_week.csv")
    out.to_csv(dest, index=False)
    print(f"{len(out)} QB-week rows ({out['season'].min()}-{out['season'].max()}) -> {dest}")
    return out


def run_checks(games: pd.DataFrame) -> None:
    since2003 = games[games["season"] >= 2003]
    assert len(since2003) >= 6000, f"expected >= 6000 games 2003+, got {len(since2003)}"
    assert since2003["spread_line"].notna().mean() >= 0.99, "spread_line coverage < 99%"
    mean_margin = (since2003["home_score"] - since2003["away_score"]).mean()
    assert 0.5 <= mean_margin <= 4.0, f"mean home margin {mean_margin:.2f} out of expected band"
    print(f"checks OK: {len(since2003)} games 2003+, spread_line coverage "
          f"{since2003['spread_line'].notna().mean():.1%}, mean home margin {mean_margin:+.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=int, default=FIRST_EPA_SEASON,
                    help="first season to aggregate EPA/QB data for (default 2003)")
    ap.add_argument("--skip-epa", action="store_true", help="skip EPA/QB aggregation (games.csv only)")
    args = ap.parse_args()

    games = fetch_games()
    run_checks(games)
    if not args.skip_epa:
        fetch_epa_team_week(since=args.since)
        fetch_qb_week(since=args.since)
    return 0


if __name__ == "__main__":
    sys.exit(main())

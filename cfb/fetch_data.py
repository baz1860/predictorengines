#!/usr/bin/env python3
"""Refresh cfb/data/games.csv from the sportsdataverse cfbfastR-data GitHub mirror
(CFBD data, updated daily in season). Run weekly during the season."""
import glob
import os
import subprocess
import sys
import tempfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "https://github.com/sportsdataverse/cfbfastR-data"
TMP = "/tmp/cfbfastR-data"


def atomic_to_csv(df, dest, required_columns, *, allow_empty=False):
    """Validate a staged CSV before atomically replacing the last-good file."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if df.empty and not allow_empty:
        raise ValueError(f"refusing to replace {dest} with an empty dataset")
    fd, tmp = tempfile.mkstemp(prefix=f".{os.path.basename(dest)}.",
                               dir=os.path.dirname(dest))
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        staged = pd.read_csv(tmp, nrows=1)
        missing = set(required_columns) - set(staged.columns)
        if missing:
            raise ValueError(f"staged {dest} missing columns: {sorted(missing)}")
        os.replace(tmp, dest)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def main():
    if os.path.isdir(os.path.join(TMP, ".git")):
        subprocess.run(["git", "-C", TMP, "pull", "--quiet"], check=True)
    else:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO, TMP],
            check=True,
        )
        subprocess.run(["git", "-C", TMP, "sparse-checkout", "set", "schedules/csv"], check=True)

    frames = []
    for f in sorted(glob.glob(os.path.join(TMP, "schedules/csv/cfb_schedules_*.csv"))):
        frames.append(pd.read_csv(f, low_memory=False))
    g = pd.concat(frames, ignore_index=True)
    # Division-I games: FBS *and* full FCS schedules, so FCS teams can carry
    # their own Elo/power ratings (sub-FCS opponents get pooled in
    # elo.load_games).
    d1 = ("fbs", "fcs")
    g = g[g["home_division"].isin(d1) | g["away_division"].isin(d1)].copy()
    g["date"] = pd.to_datetime(g["start_date"]).dt.date

    cols = ["game_id", "season", "week", "season_type", "date", "neutral_site", "home_team",
            "home_division", "away_team", "away_division", "home_points", "away_points"]
    names = ["game_id", "season", "week", "season_type", "date", "neutral", "home_team",
             "home_div", "away_team", "away_div", "home_points", "away_points"]
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)

    done = g[(g["completed"] == True) & g["home_points"].notna() & g["away_points"].notna()]  # noqa: E712
    out = done[cols].copy()
    out.columns = names
    out[["home_points", "away_points"]] = out[["home_points", "away_points"]].astype(int)
    out = out.sort_values("date").reset_index(drop=True)
    dest = os.path.join(HERE, "data", "games.csv")
    atomic_to_csv(out, dest, names, allow_empty=False)
    print(f"{len(out)} completed games, {out['season'].min()}-{out['season'].max()} -> {dest}")

    from datetime import date as _date
    upc = g[(g["completed"] == False) & (g["date"] >= _date.today())][cols[:9]].copy()  # noqa: E712
    upc.columns = names[:9]
    upc = upc.sort_values("date").reset_index(drop=True)
    atomic_to_csv(upc, os.path.join(HERE, "data", "upcoming.csv"), names[:9],
                  allow_empty=True)
    print(f"{len(upc)} upcoming games -> data/upcoming.csv")

    build_closing_spreads(g)
    return 0


def build_closing_spreads(sched):
    """Consensus closing spreads (2006-2019 in the mirror) -> data/closing_spreads.csv.

    One row per game: median home line and median juice per side across books.
    """
    src = os.path.join(TMP, "betting/csv/cfb_line_odds.csv.gz")
    if not os.path.exists(src):
        subprocess.run(["git", "-C", TMP, "sparse-checkout", "add", "betting/csv"], check=True)
    df = pd.read_csv(src, low_memory=False)
    sp = df[df["market_type"] == "spread"].dropna(subset=["lines"]).copy()
    sp[["away_name", "home_name"]] = sp["game_desc"].str.split("@", n=1, expand=True)

    # abbr -> school name by voting across all games the abbr appears in
    votes = {}
    for r in sp[["abbr", "away_name", "home_name"]].drop_duplicates().itertuples():
        for cand in (r.away_name, r.home_name):
            votes.setdefault(r.abbr, {}).setdefault(cand, 0)
            votes[r.abbr][cand] += 1
    amap = {a: max(c, key=c.get) for a, c in votes.items()}
    sp["is_home"] = sp["abbr"].map(amap) == sp["home_name"]

    cons = sp.groupby(["season", "week", "home_name", "away_name"]).apply(
        lambda g: pd.Series({
            "home_line": g.loc[g["is_home"], "lines"].median(),
            "home_odds": g.loc[g["is_home"], "odds"].median(),
            "away_odds": g.loc[~g["is_home"], "odds"].median(),
            "n_books": g["book"].nunique(),
        }), include_groups=False).reset_index().dropna(subset=["home_line"])

    sc = sched[sched["completed"] == True]  # noqa: E712
    m = cons.merge(
        sc[["season", "week", "home_team", "away_team"]],
        left_on=["season", "week", "home_name", "away_name"],
        right_on=["season", "week", "home_team", "away_team"], how="inner",
    )[["season", "week", "home_team", "away_team", "home_line", "home_odds", "away_odds", "n_books"]]
    dest = os.path.join(HERE, "data", "closing_spreads.csv")
    if os.path.exists(dest):  # keep imported seasons the mirror doesn't cover (e.g. CFBD 2020+)
        old = pd.read_csv(dest)
        keep = old[old["season"] > m["season"].max()]
        m = pd.concat([m, keep[m.columns]], ignore_index=True)
    atomic_to_csv(m, dest,
                  ["season", "week", "home_team", "away_team", "home_line",
                   "home_odds", "away_odds", "n_books"], allow_empty=False)
    print(f"{len(m)} games with consensus closing spreads "
          f"({int(m['season'].min())}-{int(m['season'].max())}) -> {dest}")


if __name__ == "__main__":
    sys.exit(main())

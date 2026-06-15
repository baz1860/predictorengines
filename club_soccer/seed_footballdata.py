#!/usr/bin/env python3
"""Seed the Club Soccer engine from football-data.co.uk — FREE, no API key, no quota.

    python3 club_soccer/seed_footballdata.py --seasons 2022 2023 2024 2025

football-data.co.uk publishes one CSV per league per season with full-time
scores, shots, shots on target, and corners — everything the model and its
`form` component need — with zero API calls. It covers domestic LEAGUES only
(no cups, no UEFA, no upcoming fixtures); for those you still need API-Football
or a manual odds.csv.

What it does:
  1. Backs up the current fixtures.csv -> data/fixtures_synthetic.bak.csv
  2. Downloads each covered league for each season and maps it onto the engine's
     fixtures.csv schema (with real shot/corner stats).
  3. Refits the model and writes a fresh validation baseline from real results.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import model as M
from competitions import BY_NAME

DATA = HERE / "data"
FIXTURES = DATA / "fixtures.csv"
BACKUP = DATA / "fixtures_synthetic.bak.csv"
BASE = "https://www.football-data.co.uk/mmz4281"

# engine competition name -> football-data.co.uk league code
LEAGUE_CODES = {
    "Premier League": "E0",
    "Championship": "E1",
    "League One": "E2",
    "League Two": "E3",
    "Scottish Premiership": "SC0",
    "Scottish Championship": "SC1",
    "Scottish League One": "SC2",
    "Scottish League Two": "SC3",
    "Bundesliga": "D1",
    "Serie A": "I1",
    "Ligue 1": "F1",
    "La Liga": "SP1",
}

# friendly-name normalisation so predictions read naturally; unmapped names pass through
TEAM_ALIASES = {
    "Man City": "Manchester City", "Man United": "Manchester United",
    "Nott'm Forest": "Nottingham Forest", "Newcastle": "Newcastle United",
    "Wolves": "Wolverhampton", "Sheffield United": "Sheffield Utd",
    "Spurs": "Tottenham", "Ein Frankfurt": "Eintracht Frankfurt",
    "Bayern Munich": "Bayern Munich", "Dortmund": "Borussia Dortmund",
    "Leverkusen": "Bayer Leverkusen", "Inter": "Inter", "Milan": "AC Milan",
    "Paris SG": "Paris Saint-Germain", "Ath Madrid": "Atletico Madrid",
    "Ath Bilbao": "Athletic Bilbao", "Sociedad": "Real Sociedad",
}


def season_code(start_year: int) -> str:
    """2024 -> '2425' (the 2024/25 season)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def norm(name: str) -> str:
    return TEAM_ALIASES.get(str(name).strip(), str(name).strip())


def fid(code: str, date: str, home: str, away: str) -> int:
    """Deterministic 9-digit fixture id (football-data has none)."""
    h = hashlib.md5(f"{code}|{date}|{home}|{away}".encode()).hexdigest()
    return int(h[:8], 16)


def download(url: str, timeout: int = 30) -> pd.DataFrame | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            raw = r.read()
    except Exception as e:
        print(f"  ! {url.split('/')[-2:]}: {e}")
        return None
    try:
        return pd.read_csv(io.StringIO(raw.decode("latin-1")), on_bad_lines="skip")
    except Exception as e:
        print(f"  ! parse {url}: {e}")
        return None


def col(df: pd.DataFrame, name: str):
    return df[name] if name in df.columns else ""


def fixtures_for(comp_name: str, league_code: str, start_year: int) -> list[dict]:
    comp = BY_NAME[comp_name]
    code = season_code(start_year)
    df = download(f"{BASE}/{code}/{league_code}.csv")
    if df is None or "FTHG" not in df.columns:
        return []
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    dates = pd.to_datetime(col(df, "Date"), dayfirst=True, errors="coerce")
    rows = []
    for i, r in df.reset_index(drop=True).iterrows():
        d = dates.iloc[i]
        if pd.isna(d):
            continue
        date = d.strftime("%Y-%m-%d")
        home, away = norm(r["HomeTeam"]), norm(r["AwayTeam"])
        rows.append({
            "fixture_id": fid(league_code + code, date, home, away),
            "date": date, "season": start_year,
            "competition": comp.name, "competition_id": comp.api_id,
            "country": comp.country, "type": comp.kind,
            "home_id": "", "home": home, "away_id": "", "away": away,
            "home_goals": int(r["FTHG"]), "away_goals": int(r["FTAG"]),
            "status": "FT", "neutral": 0,
            "home_shots": _num(r.get("HS")), "away_shots": _num(r.get("AS")),
            "home_sot": _num(r.get("HST")), "away_sot": _num(r.get("AST")),
            "home_corners": _num(r.get("HC")), "away_corners": _num(r.get("AC")),
        })
    print(f"  {comp.name} {start_year}/{(start_year + 1) % 100:02d}: {len(rows)} matches")
    return rows


def _num(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", type=int, nargs="+", default=[2022, 2023, 2024, 2025],
                    help="season start years (default: 2022 2023 2024 2025)")
    ap.add_argument("--leagues", nargs="+", default=list(LEAGUE_CODES),
                    help="subset of competition names to pull (default: all covered)")
    ap.add_argument("--keep-backup", action="store_true")
    args = ap.parse_args()

    wanted = [l for l in args.leagues if l in LEAGUE_CODES]
    if not wanted:
        sys.exit(f"No covered leagues in {args.leagues}. Options: {list(LEAGUE_CODES)}")

    if FIXTURES.exists() and not (args.keep_backup and BACKUP.exists()):
        BACKUP.write_text(FIXTURES.read_text())
        print(f"Backed up current fixtures -> {BACKUP.name}")

    print(f"\nDownloading {len(wanted)} leagues x {len(args.seasons)} seasons "
          "from football-data.co.uk...")
    rows: list[dict] = []
    for comp_name in wanted:
        for yr in args.seasons:
            rows.extend(fixtures_for(comp_name, LEAGUE_CODES[comp_name], yr))

    if not rows:
        sys.exit("No data downloaded — check your connection or season values. "
                 f"Synthetic backup left intact; restore with: cp {BACKUP} {FIXTURES}")

    df = pd.DataFrame(rows).drop_duplicates(subset=["fixture_id"], keep="last")
    df.to_csv(FIXTURES, index=False)
    print(f"\nWrote {len(df)} real matches across {df['competition'].nunique()} "
          f"leagues -> {FIXTURES}")

    print("\nRefitting model...")
    params = M.fit()
    M.save_params(params)
    print(f"  fitted {params['fitted_matches']} matches, {len(params['teams'])} teams")

    print("\nWalk-forward validation (fresh baseline from real data):")
    # Run validate.py as its own process from club_soccer/ so the right module
    # loads regardless of sys.path and writes its own baseline + predictions.
    import subprocess
    subprocess.run([sys.executable, str(HERE / "validate.py"), "--update-baseline"],
                   cwd=str(HERE), check=False)
    print("  (uniform-1/3 baseline Brier ≈ 0.667 — beat it to have real skill)")
    print("\nDone. League data is real now. For cups/UEFA/upcoming fixtures you still "
          "need API-Football or a manual odds.csv.")


if __name__ == "__main__":
    main()

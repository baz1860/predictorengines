"""Evidence tiers: how much the model actually knows about a national team.

Ported from `club_soccer/coverage.py`, which was written after a real loss: Sturm
Graz were priced at 18% away to Hearts and won 4-0. Their league was not ingested,
so the club existed in the model only through 24 UEFA matches, and every rating
lookup used a silent default — an unrated team is handed the identity of a
perfectly average one. Nothing warned.

The international version of that failure is worse, not better, because the
population is more unequal. San Marino, Anguilla and Brazil are all "teams" to a
rating lookup, but the evidence behind their ratings differs by two orders of
magnitude.

Tiers
-----
    full        enough recent matches AND a spread of opponents
    thin        known, but under-evidenced — the rating is weakly identified
    defaulted   effectively absent — the rating is close to a pure prior

Opponent diversity is part of the test, not just match count. A team that plays
only its four regional neighbours can accumulate 40 matches without its rating
being pinned to the global scale — the same structural problem that made Sturm
Graz look average. A naive match-count threshold misses exactly that case.

Report-only. Nothing here changes a probability. It changes what we are willing to
bet on, which is the plan's §11 "gate betting per competition, per evidence tier".
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results.csv"

FULL, THIN, DEFAULTED = "full", "thin", "defaulted"

RECENT_YEARS = 4
MIN_MATCHES_FULL = 20
MIN_OPPONENTS_FULL = 10
MIN_MATCHES_THIN = 5
# A team whose opponents are themselves poorly connected is not well-anchored even
# with plenty of matches. Measured as the share of its recent opponents that are
# themselves above the "full" match threshold.
MIN_ANCHORED_SHARE = 0.30


@dataclass(frozen=True)
class TeamEvidence:
    team: str
    matches: int
    opponents: int
    anchored_share: float
    tier: str
    reason: str


def _recent(df: pd.DataFrame, asof: object = None) -> pd.DataFrame:
    df = df.dropna(subset=["home_score", "away_score"])
    end = pd.Timestamp(asof) if asof is not None else df.date.max()
    return df[df.date >= end - pd.DateOffset(years=RECENT_YEARS)]


def build(results: pd.DataFrame | None = None,
          asof: object = None) -> dict[str, TeamEvidence]:
    df = results if results is not None else pd.read_csv(RESULTS, parse_dates=["date"])
    if "date" in df and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df = df.assign(date=pd.to_datetime(df["date"], errors="coerce"))
    recent = _recent(df, asof)

    pairs = pd.concat([
        recent[["home_team", "away_team"]].rename(
            columns={"home_team": "team", "away_team": "opponent"}),
        recent[["away_team", "home_team"]].rename(
            columns={"away_team": "team", "home_team": "opponent"}),
    ]).dropna()

    counts = pairs.groupby("team").size()
    opponents = pairs.groupby("team").opponent.nunique()
    well_played = set(counts[counts >= MIN_MATCHES_FULL].index)

    anchored = (pairs.assign(anchor=pairs.opponent.isin(well_played))
                     .groupby("team").anchor.mean())

    out: dict[str, TeamEvidence] = {}
    for team in counts.index:
        n = int(counts[team])
        opp = int(opponents.get(team, 0))
        share = float(anchored.get(team, 0.0))
        if n >= MIN_MATCHES_FULL and opp >= MIN_OPPONENTS_FULL and share >= MIN_ANCHORED_SHARE:
            tier, reason = FULL, "enough recent matches against well-connected opponents"
        elif n >= MIN_MATCHES_THIN:
            bits = []
            if n < MIN_MATCHES_FULL:
                bits.append(f"only {n} recent matches")
            if opp < MIN_OPPONENTS_FULL:
                bits.append(f"only {opp} distinct opponents")
            if share < MIN_ANCHORED_SHARE:
                bits.append(f"only {share:.0%} of opponents are themselves "
                            f"well-evidenced")
            tier, reason = THIN, "; ".join(bits)
        else:
            tier, reason = DEFAULTED, f"only {n} recent matches — rating is near-prior"
        out[str(team)] = TeamEvidence(str(team), n, opp, share, tier, reason)
    return out


def tier_of(team: object, table: dict[str, TeamEvidence] | None = None) -> str:
    table = table if table is not None else build()
    entry = table.get(str(team))
    return entry.tier if entry else DEFAULTED


def fixture_tier(home: object, away: object,
                 table: dict[str, TeamEvidence] | None = None) -> str:
    """The WEAKER of the two sides. A fixture is only as evidenced as its worst team."""
    table = table if table is not None else build()
    order = {FULL: 2, THIN: 1, DEFAULTED: 0}
    a, b = tier_of(home, table), tier_of(away, table)
    return a if order[a] <= order[b] else b


def summary(table: dict[str, TeamEvidence] | None = None) -> pd.DataFrame:
    table = table if table is not None else build()
    df = pd.DataFrame([vars(t) for t in table.values()])
    return (df.groupby("tier")
              .agg(teams=("team", "size"), median_matches=("matches", "median"),
                   median_opponents=("opponents", "median"))
              .reindex([FULL, THIN, DEFAULTED]))


if __name__ == "__main__":
    table = build()
    print(summary(table).to_string())
    print("\nthin / defaulted teams (not bettable without a warning):")
    rows = sorted((t for t in table.values() if t.tier != FULL),
                  key=lambda t: (t.tier, -t.matches))
    for t in rows[:40]:
        print(f"  {t.tier:<10}{t.team:<28}{t.matches:>3}m {t.opponents:>3}opp  {t.reason}")
    print(f"  … {max(0, len(rows) - 40)} more")

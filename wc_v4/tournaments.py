"""Leak-free held-out samples for past World Cups (validation substrate).

The V4 gate was sample-starved on a single 64-match tournament (WC2022). This
module generalises the leak-free sample construction so any past World Cup can be
folded in, the same way `market_blend._wc2022_samples` builds 2022.

For each tournament we fit the SAME point-in-time model the suite ships — an
Elo+Poisson / Dixon-Coles blend trained strictly before the tournament's kickoff —
and emit one sample per match:

    Sample = (p_model[3], p_market[3] | None, actual_idx, date, stage)

`p_market` is present only when a same-schema odds file exists and passes a
lightweight schema check (`data/wc{YEAR}_odds.csv`, like `data/wc2022_odds.csv`).
With odds, a match can feed the market-blend gate; without valid odds, the match
still strengthens the *model-only* calibration evidence, which is the honest
thing it can contribute.

numpy + pandas only. No network: everything comes from `data/results.csv` and, if
present, a local odds CSV.
"""
from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, NamedTuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engines.worldcup.predictor import (load_matches, compute_elo, fit_goal_model,  # noqa: E402
                       expected_goals, HOME_ADV, DC_RHO)
from engines.worldcup.dixoncoles import fit_dc, outcome_probs  # noqa: E402
from engines.worldcup.edge import devig  # noqa: E402

DATA = ROOT / "data"
_SIDE_IDX = {"home": 0, "draw": 1, "away": 2}
ODDS_REQUIRED_COLUMNS = {
    "date", "home", "away", "odds_home", "odds_draw", "odds_away", "result90",
}
ODDS_PRICE_COLUMNS = ("odds_home", "odds_draw", "odds_away")
# API/odds name spellings -> results.csv spellings (extend as needed).
_NAME_MAP = {"USA": "United States", "Korea Republic": "South Korea"}


class Sample(NamedTuple):
    p_model: np.ndarray
    p_market: np.ndarray | None
    actual: int          # 0 home / 1 draw / 2 away (final-score 1X2)
    date: str
    stage: str           # "group" | "knockout"
    tournament: str


def _result_idx(hs: float, as_: float) -> int:
    return 0 if hs > as_ else (1 if hs == as_ else 2)


def point_in_time_model(cutoff: str) -> Callable[..., np.ndarray | None]:
    """Return probs(home, away, neutral) -> blended 1X2 vector, using only data
    strictly before `cutoff` (so scoring any tournament match is hindsight-free).
    Mirrors the elo+dc blend in market_blend exactly."""
    played, _ = load_matches()
    _, played = compute_elo(played)
    train = played[played["date"] < cutoff]
    beta = fit_goal_model(train)
    dc = fit_dc(train, anchor=cutoff, verbose=False)
    ratings, _ = compute_elo(train)

    def probs(home: str, away: str, neutral: bool = True) -> np.ndarray | None:
        home = _NAME_MAP.get(home, home)
        away = _NAME_MAP.get(away, away)
        if home not in ratings or home not in dc.att or away not in dc.att:
            return None
        adv = 0.0 if neutral else HOME_ADV
        le = expected_goals(ratings[home], ratings[away], beta, adv)
        ld = dc.lambdas(home, away)
        pe = np.array(outcome_probs(*le, DC_RHO)[:3])
        pdc = np.array(outcome_probs(*ld, dc.rho)[:3])
        return (pe + pdc) / 2.0

    return probs


def _path_label(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def validate_odds_csv(odds_csv: Path) -> dict[str, Any]:
    """Return schema/parse status for a local historical 1X2 odds file.

    This is intentionally strict but not clever: the existing historical files
    use fractional odds, plus `result90` in {home, draw, away}. Invalid files are
    left out of market-gate samples and reported by the validation harness.
    """
    path = Path(odds_csv)
    status: dict[str, Any] = {
        "path": _path_label(path),
        "exists": path.exists(),
        "valid": False,
        "rows": 0,
        "errors": [],
    }
    if not path.exists():
        status["errors"].append("missing")
        return status
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive report path
        status["errors"].append(f"read_error:{exc}")
        return status
    status["rows"] = int(len(df))
    missing = sorted(ODDS_REQUIRED_COLUMNS - set(df.columns))
    if missing:
        status["errors"].append("missing_columns:" + ",".join(missing))
    if df.empty:
        status["errors"].append("empty")
    if "result90" in df.columns:
        bad = sorted(set(df["result90"].dropna().astype(str)) - set(_SIDE_IDX))
        if bad:
            status["errors"].append("bad_result90:" + ",".join(bad))
    if not missing:
        for col in ODDS_PRICE_COLUMNS:
            bad_count = 0
            for v in df[col].dropna():
                try:
                    if float(Fraction(str(v))) + 1.0 <= 1.0:
                        bad_count += 1
                except (ValueError, ZeroDivisionError):
                    bad_count += 1
            if bad_count:
                status["errors"].append(f"bad_{col}:{bad_count}")
        keys = df[["date", "home", "away"]].astype(str)
        dupes = int(keys.duplicated().sum())
        if dupes:
            status["errors"].append(f"duplicate_matches:{dupes}")
    status["valid"] = not status["errors"]
    return status


def _odds_lookup(odds_csv: Path) -> dict[tuple, np.ndarray]:
    """(date, home, away) -> de-vigged p_market, from a wc20xx_odds.csv file."""
    if not validate_odds_csv(odds_csv)["valid"]:
        return {}
    df = pd.read_csv(odds_csv)
    to_dec = lambda s: float(Fraction(str(s))) + 1.0  # fractional -> decimal
    out: dict[tuple, np.ndarray] = {}
    for r in df.itertuples(index=False):
        try:
            p, _ = devig([to_dec(r.odds_home), to_dec(r.odds_draw),
                          to_dec(r.odds_away)])
        except (ValueError, ZeroDivisionError):
            continue
        home = _NAME_MAP.get(r.home, r.home)
        away = _NAME_MAP.get(r.away, r.away)
        out[(str(r.date), home, away)] = np.asarray(p, float)
    return out


def tournament_samples(cutoff: str, date_lo: str, date_hi: str,
                       knockout_from: str,
                       odds_csv: Path | None = None,
                       tournament: str = "FIFA World Cup") -> list[Sample]:
    """Leak-free samples for one World Cup, from results.csv (+ optional odds).

    cutoff        first day of the tournament (model trained strictly before it).
    date_lo/hi    inclusive date window selecting the tournament's matches.
    knockout_from matches on/after this date are tagged "knockout".
    odds_csv      optional wc20xx_odds.csv for p_market (enables the blend gate).
    """
    probs = point_in_time_model(cutoff)
    odds = _odds_lookup(odds_csv) if odds_csv else {}

    played, _ = load_matches()
    played["date"] = pd.to_datetime(played["date"])
    sel = played[(played["date"] >= date_lo) & (played["date"] <= date_hi)
                 & (played["tournament"] == tournament)].sort_values("date")

    samples: list[Sample] = []
    for r in sel.itertuples(index=False):
        pm = probs(r.home_team, r.away_team, bool(r.neutral))
        if pm is None:
            continue
        md = pd.Timestamp(r.date).strftime("%Y-%m-%d")
        stage = "knockout" if md >= knockout_from else "group"
        market = odds.get((md, _NAME_MAP.get(r.home_team, r.home_team),
                           _NAME_MAP.get(r.away_team, r.away_team)))
        samples.append(Sample(
            p_model=pm, p_market=market,
            actual=_result_idx(r.home_score, r.away_score),
            date=md, stage=stage, tournament=tournament))
    return samples


# Registry of the past World Cups we can replay. Add a row + drop a
# data/wc{YEAR}_odds.csv to fold a new tournament into the gate.
TOURNAMENTS: dict[str, dict[str, Any]] = {
    "WC2018": {"cutoff": "2018-06-14", "date_lo": "2018-06-14",
               "date_hi": "2018-07-15", "knockout_from": "2018-06-30",
               "odds_csv": DATA / "wc2018_odds.csv"},
    "WC2022": {"cutoff": "2022-11-20", "date_lo": "2022-11-20",
               "date_hi": "2022-12-18", "knockout_from": "2022-12-03",
               "odds_csv": DATA / "wc2022_odds.csv"},
}


def all_samples(which: tuple[str, ...] | None = None) -> dict[str, list[Sample]]:
    """Build samples for each registered tournament (default: all)."""
    keys = which or tuple(TOURNAMENTS)
    out: dict[str, list[Sample]] = {}
    for k in keys:
        cfg = TOURNAMENTS[k]
        oc = cfg.get("odds_csv")
        odds_status = validate_odds_csv(Path(oc)) if oc else {"valid": False}
        out[k] = tournament_samples(
            cfg["cutoff"], cfg["date_lo"], cfg["date_hi"],
            cfg["knockout_from"],
            odds_csv=oc if (oc and odds_status["valid"]) else None)
    return out


def odds_validation() -> dict[str, dict[str, Any]]:
    """Odds-file schema status for every registered tournament."""
    out: dict[str, dict[str, Any]] = {}
    for name, cfg in TOURNAMENTS.items():
        oc = cfg.get("odds_csv")
        out[name] = validate_odds_csv(Path(oc)) if oc else {
            "path": None, "exists": False, "valid": False,
            "rows": 0, "errors": ["no_odds_csv_configured"],
        }
    return out


if __name__ == "__main__":  # pragma: no cover
    for name, samps in all_samples().items():
        n = len(samps)
        with_odds = sum(1 for s in samps if s.p_market is not None)
        ko = sum(1 for s in samps if s.stage == "knockout")
        print(f"{name}: {n} matches | {with_odds} with market odds | "
              f"{ko} knockout")

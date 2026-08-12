"""Reconstructed preseason-prior and FCS-transition challengers.

The production prior uses CFBD team-talent and returning-PPA endpoints. When
those feeds are unavailable, this module evaluates a deliberately separate
challenger built from prior-year talent, current recruiting points, and incoming
transfer count. Historical CFBD responses are reconstructed after the fact, so
even a passing holdout remains non-runtime until the provenance and transition
gates are independently approved.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import elo as E
from . import fetch_cfbd
from . import priors

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
INPUTS_JSON = DATA / "prior_proxy_inputs.json"
REPORT_JSON = DATA / "prior_challenger_2025.json"
SELECTION_SEASONS = (2022, 2023, 2024)
HOLDOUT_SEASON = 2025
CURRENT_SEASON = 2026
GRID = {
    "previous_talent": (0.0, 30.0, 60.0, 90.0),
    "recruiting": (0.0, 30.0, 60.0, 90.0),
    "incoming_transfers": (0.0, 15.0, 30.0, 45.0),
}
TRANSITION_TEAMS = {
    2022: ("James Madison",),
    2023: ("Jacksonville State", "Sam Houston"),
    2024: ("Kennesaw State",),
    2025: ("Delaware", "Missouri State"),
    2026: ("North Dakota State", "Sacramento State"),
}
TRANSITION_MIN_HOLDOUT_GAMES = 30


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: str | Path, payload: object) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dest.name}.", dir=dest.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, dest)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return dest


def _target_fbs(games: pd.DataFrame, season: int) -> set[str]:
    if season == CURRENT_SEASON:
        return {team for team, div in E._schedule_divisions(season).items()
                if str(div).lower() == "fbs"}
    rows = games[games["season"] == season]
    return (set(rows.loc[rows["home_div"] == "fbs", "home"])
            | set(rows.loc[rows["away_div"] == "fbs", "away"]))


def _zscore(values: dict[str, float], teams: set[str]) -> dict[str, float]:
    complete = {team: float(values[team]) for team in teams if team in values}
    if len(complete) < 20:
        raise ValueError(f"feature coverage is inadequate: {len(complete)}/{len(teams)}")
    array = np.asarray(list(complete.values()), dtype=float)
    sd = float(array.std()) or 1.0
    mean = float(array.mean())
    return {team: (float(values.get(team, mean)) - mean) / sd for team in teams}


def aggregate_year(year: int, recruiting: list[dict], portal: list[dict],
                   games: pd.DataFrame, official: dict) -> dict:
    """Create a compact decision-time feature snapshot from raw CFBD rows."""
    teams = _target_fbs(games, year)
    cutoff = pd.Timestamp(f"{year}-08-02T00:00:00Z")
    recruiting_points = {
        str(row.get("team")): float(row["points"])
        for row in recruiting
        if row.get("team") and row.get("points") is not None
    }
    recruiting_z = _zscore(recruiting_points, teams)
    incoming = {team: 0 for team in teams}
    included_portal = 0
    for row in portal:
        transfer_date = pd.to_datetime(
            row.get("transferDate"), errors="coerce", utc=True)
        if pd.isna(transfer_date) or transfer_date > cutoff:
            continue
        included_portal += 1
        destination = str(row.get("destination") or "")
        if destination in incoming:
            incoming[destination] += 1
    incoming_z = _zscore(incoming, teams)
    previous_talent = {
        team: float(official.get((team, year - 1), {}).get("talent_z", 0.0))
        for team in teams
    }
    previous_talent_present = sum(
        "talent_z" in official.get((team, year - 1), {}) for team in teams)
    rows = {
        team: {
            "previous_talent_z": previous_talent[team],
            "previous_talent_present": (
                "talent_z" in official.get((team, year - 1), {})),
            "recruiting_points": recruiting_points.get(team),
            "recruiting_z": recruiting_z[team],
            "incoming_transfers": incoming[team],
            "incoming_transfers_z": incoming_z[team],
        }
        for team in sorted(teams)
    }
    return {
        "season": year,
        "cutoff": cutoff.isoformat(),
        "target_fbs_teams": len(teams),
        "coverage": {
            "previous_talent": previous_talent_present,
            "recruiting": sum(team in recruiting_points for team in teams),
            "portal_destination_or_zero": len(teams),
        },
        "source": {
            "recruiting_query": f"CFBD /recruiting/teams?year={year}",
            "recruiting_rows": len(recruiting),
            "recruiting_sha256": _canonical_hash(recruiting),
            "portal_query": f"CFBD /player/portal?year={year}",
            "portal_rows": len(portal),
            "portal_rows_through_cutoff": included_portal,
            "portal_sha256": _canonical_hash(portal),
        },
        "teams": rows,
    }


def fetch_inputs(start: int = 2022, end: int = CURRENT_SEASON,
                 path: str | Path = INPUTS_JSON, refresh_all: bool = False) -> dict:
    key = fetch_cfbd._key()
    if not key:
        raise ValueError("CFBD key is not configured")
    games = E.load_games()
    official = priors.load_features()
    previous = {}
    if not refresh_all:
        try:
            previous = load_inputs(path).get("seasons", {})
        except (OSError, json.JSONDecodeError):
            previous = {}
    seasons = {}
    for year in range(start, end + 1):
        # Completed historical seasons are frozen inputs (their SHA-256s are
        # recorded in this artifact) — don't re-download them on every run.
        if year < CURRENT_SEASON and str(year) in previous and not refresh_all:
            seasons[str(year)] = previous[str(year)]
            continue
        recruiting = fetch_cfbd.pull(f"/recruiting/teams?year={year}", key)
        portal = fetch_cfbd.pull(f"/player/portal?year={year}", key)
        if not isinstance(recruiting, list) or not isinstance(portal, list):
            raise ValueError(f"CFBD challenger response is malformed for {year}")
        seasons[str(year)] = aggregate_year(
            year, recruiting, portal, games, official)
    payload = {
        "schema_version": 1,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": (
            "Retrospective CFBD reconstruction fetched in 2026. Transfer rows "
            "are cutoff by transferDate; recruiting ranks have no publication "
            "timestamp and are not a true archived decision-time snapshot."
        ),
        "seasons": seasons,
    }
    _atomic_json(path, payload)
    return payload


def load_inputs(path: str | Path = INPUTS_JSON) -> dict:
    return json.loads(Path(path).read_text())


def offsets(payload: dict, coefficients: dict[str, float]) -> dict:
    out = {}
    for season_text, season in payload["seasons"].items():
        for team, row in season["teams"].items():
            out[(team, int(season_text))] = (
                coefficients["previous_talent"] * row["previous_talent_z"]
                + coefficients["recruiting"] * row["recruiting_z"]
                + coefficients["incoming_transfers"]
                * row["incoming_transfers_z"]
            )
    return out


def score(games: pd.DataFrame, prior_offsets: dict,
          seasons: tuple[int, ...], teams_by_season: dict | None = None) -> dict:
    _, history = E.run_elo(
        games, record_pregame=True, carry=0.65, prior_offsets=prior_offsets)
    diffs = np.asarray([row[2] for row in history], dtype=float)
    mask = ((games["week"] <= 4)
            & (games["home_div"] == "fbs")
            & (games["away_div"] == "fbs")
            & games["season"].isin(seasons))
    if teams_by_season:
        selected = []
        for row in games.itertuples():
            teams = teams_by_season.get(int(row.season), ())
            selected.append(row.home in teams or row.away in teams)
        mask &= np.asarray(selected, dtype=bool)
    chosen = mask.to_numpy()
    probability = 1.0 / (1.0 + 10.0 ** (-diffs[chosen] / 400.0))
    actual = (games["home_points"] > games["away_points"]).astype(float).to_numpy()[chosen]
    if not len(actual):
        return {"n_games": 0, "brier": None, "accuracy": None}
    return {
        "n_games": int(len(actual)),
        "brier": float(np.mean((probability - actual) ** 2)),
        "accuracy": float(np.mean((probability > 0.5) == (actual > 0.5))),
    }


def select_coefficients(payload: dict, games: pd.DataFrame) -> tuple[dict, dict]:
    best: tuple[float, dict, dict] | None = None
    keys = tuple(GRID)
    for values in itertools.product(*(GRID[key] for key in keys)):
        coefficients = dict(zip(keys, values))
        metric = score(games, offsets(payload, coefficients), SELECTION_SEASONS)
        candidate = (float(metric["brier"]), coefficients, metric)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    return best[1], best[2]


def transition_validation(games: pd.DataFrame) -> dict:
    candidates = range(1200, 1651, 50)
    selection_rows = []
    holdout_by_rating = {}
    original = E.NEW_TEAM_ELO
    try:
        for rating in candidates:
            E.NEW_TEAM_ELO = float(rating)
            selection = score(
                games, {}, tuple(year for year in TRANSITION_TEAMS if year < 2025),
                TRANSITION_TEAMS)
            holdout = score(games, {}, (HOLDOUT_SEASON,), TRANSITION_TEAMS)
            selection_rows.append({"rating": rating, **selection})
            holdout_by_rating[rating] = holdout
    finally:
        E.NEW_TEAM_ELO = original
    selected = min(selection_rows, key=lambda row: row["brier"])
    holdout = holdout_by_rating[selected["rating"]]
    champion = holdout_by_rating[int(original)]
    enough_data = holdout["n_games"] >= TRANSITION_MIN_HOLDOUT_GAMES
    return {
        "selection_seasons": "2022-2024",
        "selection_teams": [team for year in (2022, 2023, 2024)
                            for team in TRANSITION_TEAMS[year]],
        "selected_starting_elo": selected["rating"],
        "selection": {k: v for k, v in selected.items() if k != "rating"},
        "holdout_season": HOLDOUT_SEASON,
        "holdout_teams": list(TRANSITION_TEAMS[HOLDOUT_SEASON]),
        "holdout": holdout,
        "champion_1300_holdout": champion,
        "minimum_holdout_games": TRANSITION_MIN_HOLDOUT_GAMES,
        "sample_gate": enough_data,
        "runtime_approved": False,
        "reason": (
            "transition sample is too small for promotion"
            if not enough_data else "requires independent football-strength prior review"
        ),
        "current_2026_transition_teams": list(TRANSITION_TEAMS[CURRENT_SEASON]),
    }


def validate(payload: dict | None = None) -> dict:
    payload = payload or load_inputs()
    games = E.load_games()
    coefficients, selection = select_coefficients(payload, games)
    proxy = offsets(payload, coefficients)
    holdout = score(games, proxy, (HOLDOUT_SEASON,))
    baseline_selection = score(games, {}, SELECTION_SEASONS)
    baseline_holdout = score(games, {}, (HOLDOUT_SEASON,))
    improvement = baseline_holdout["brier"] - holdout["brier"]
    proxy_gate = (
        improvement >= 0.002
        and holdout["accuracy"] >= baseline_holdout["accuracy"]
    )
    coverage = payload["seasons"][str(CURRENT_SEASON)]["coverage"]
    target = payload["seasons"][str(CURRENT_SEASON)]["target_fbs_teams"]
    coverage_gate = coverage["recruiting"] / target >= 0.80
    transition = transition_validation(games)
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "rejected_for_runtime",
        "selection_seasons": list(SELECTION_SEASONS),
        "holdout_season": HOLDOUT_SEASON,
        "selected_coefficients": coefficients,
        "baseline": {"selection": baseline_selection, "holdout": baseline_holdout},
        "proxy": {"selection": selection, "holdout": holdout},
        "holdout_brier_improvement": improvement,
        "gates": {
            "forecast_lift": proxy_gate,
            "current_coverage": coverage_gate,
            "archived_point_in_time_provenance": False,
            "transition_sample": transition["sample_gate"],
        },
        "runtime_approved": False,
        "rejection_reasons": [
            "historical recruiting inputs are retrospectively reconstructed, not archived decision-time snapshots",
            transition["reason"],
        ],
        "current_2026": {
            "target_fbs_teams": target,
            "coverage": coverage,
            "input_fingerprint": _canonical_hash(payload["seasons"][str(CURRENT_SEASON)]),
        },
        "transition": transition,
        "input_artifact_sha256": _canonical_hash(payload),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fetch", action="store_true",
                        help="refresh compact CFBD challenger inputs "
                             "(current season only; frozen years are reused)")
    parser.add_argument("--refetch-all", action="store_true",
                        help="with --fetch, re-download frozen historical years too")
    parser.add_argument("--write", action="store_true",
                        help="write the frozen validation report")
    args = parser.parse_args()
    payload = (fetch_inputs(refresh_all=args.refetch_all)
               if args.fetch else load_inputs())
    report = validate(payload)
    if args.write:
        _atomic_json(REPORT_JSON, report)
        print(f"wrote {REPORT_JSON}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

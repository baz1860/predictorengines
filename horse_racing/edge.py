"""Decision-time win-board comparison for the pure horse-racing model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .model import load_artifact, predict_race
from .schema import (DataBundle, DataError, latest_odds_snapshot, load_bundle,
                     race_cutoff, race_row, runner_snapshot)

DEFAULT_EDGE = 0.03
MAX_BOARD_SPAN_SECONDS = 300
MAX_ODDS_AGE_SECONDS = 1800


def price_race(race_id: str, bundle: DataBundle | None = None,
               artifact: dict | None = None, data_dir: str | Path | None = None,
               source: str | None = None, min_edge: float = DEFAULT_EDGE) -> pd.DataFrame:
    bundle = bundle or load_bundle(data_dir)
    artifact = artifact or load_artifact(Path(data_dir) / "model_params.json" if data_dir else None)
    race = race_row(bundle, race_id)
    cutoff = race_cutoff(race, int(artifact.get("cutoff_minutes", 15)))
    runners = runner_snapshot(bundle, race_id, cutoff)
    pred = predict_race(race_id, bundle=bundle, artifact=artifact)
    odds = latest_odds_snapshot(bundle, race_id, cutoff, source=source)
    if odds.empty:
        raise DataError(f"race {race_id!r} has no open win odds at or before {cutoff}")

    active = set(runners["runner_id"].astype(str))
    quoted = set(odds["runner_id"].astype(str))
    board_complete = active == quoted
    board_span = (odds["captured_at"].max() - odds["captured_at"].min()).total_seconds()
    board_complete = board_complete and board_span <= MAX_BOARD_SPAN_SECONDS
    board_age = (cutoff - odds["captured_at"].max()).total_seconds()
    board_complete = board_complete and 0 <= board_age <= MAX_ODDS_AGE_SECONDS
    sizes = pd.to_numeric(odds["available_size"], errors="coerce")
    board_complete = board_complete and bool(
        (sizes.notna() & np.isfinite(sizes) & (sizes > 0)).all())

    # If either side supplies a field version, every quote must match the
    # active runner declaration. Blank-on-both remains valid for manual V1 data.
    runner_versions = runners.set_index("runner_id")["field_version"].to_dict()
    version_ok = True
    for row in odds.itertuples(index=False):
        rv = runner_versions.get(str(row.runner_id))
        ov = row.field_version
        if (pd.notna(rv) or pd.notna(ov)) and not (pd.notna(rv) and pd.notna(ov)
                                                   and float(rv) == float(ov)):
            version_ok = False
            break
    board_complete = board_complete and version_ok

    work = pred.merge(odds[["runner_id", "decimal_odds", "captured_at", "source",
                            "available_size", "field_version"]],
                      on="runner_id", how="left")
    work["p_book"] = 1.0 / work["decimal_odds"]
    overround = float(work["p_book"].sum()) if board_complete else np.nan
    work["p_market"] = work["p_book"] / overround if board_complete and overround > 0 else np.nan
    work["edge"] = work["p_model"] - work["p_market"]
    work["ev_per_unit"] = work["p_model"] * work["decimal_odds"] - 1.0
    work["kelly_frac"] = np.maximum(
        0.0, (work["p_model"] * work["decimal_odds"] - 1.0)
        / np.maximum(work["decimal_odds"] - 1.0, 1e-12))
    work["recommended"] = (board_complete & (work["edge"] >= float(min_edge))
                           & (work["ev_per_unit"] > 0))
    work["stake_gbp"] = 0.0  # analytical V1; suite staking intentionally disabled
    work["event_id"] = str(race_id)
    work["match_date"] = str(race.get("meeting_date") or pd.Timestamp(
        race["scheduled_off_utc"]).date())
    work["home"] = work["horse_name"]
    work["away"] = ""
    work["market"] = "win"
    work["side"] = "win"
    work["line"] = ""
    work["bet"] = work["horse_name"].map(lambda name: f"{name} to win")
    work["odds"] = work["decimal_odds"]
    work["board_complete"] = bool(board_complete)
    work["overround"] = overround
    return work.sort_values("ev_per_unit", ascending=False).reset_index(drop=True)

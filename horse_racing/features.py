"""Leakage-safe, race-relative feature construction."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, log

import numpy as np
import pandas as pd

from .schema import DataBundle, DataError, final_results, race_cutoff, runner_snapshot

FEATURE_SCHEMA_VERSION = 1
FEATURES = [
    "official_rating_rel", "official_rating_missing_rel",
    "weight_rel", "weight_missing_rel", "draw_rel", "draw_missing_rel",
    "age_rel", "age_missing_rel", "horse_form_rel", "horse_win_rel",
    "horse_starts_rel", "days_since_rel", "days_since_missing_rel",
    "trainer_win_rel", "jockey_win_rel", "surface_fit_rel",
    "distance_fit_rel", "course_fit_rel",
]


@dataclass
class DecayedStats:
    perf_sum: float = 0.0
    wins: float = 0.0
    weight: float = 0.0
    lifetime_starts: int = 0
    last_time: pd.Timestamp | None = None

    def _factor(self, at: pd.Timestamp, half_life_days: float) -> float:
        if self.last_time is None:
            return 1.0
        days = max(0.0, (at - self.last_time).total_seconds() / 86400.0)
        return exp(-log(2.0) * days / half_life_days)

    def values(self, at: pd.Timestamp, half_life_days: float,
               perf_prior: float, win_prior: float, prior_win: float) -> tuple[float, float]:
        f = self._factor(at, half_life_days)
        w = self.weight * f
        perf = self.perf_sum * f / (w + perf_prior) if w + perf_prior else 0.0
        win = (self.wins * f + win_prior * prior_win) / (w + win_prior)
        return perf, win

    def update(self, at: pd.Timestamp, performance: float, won: bool,
               half_life_days: float) -> None:
        f = self._factor(at, half_life_days)
        self.perf_sum = self.perf_sum * f + performance
        self.wins = self.wins * f + float(won)
        self.weight = self.weight * f + 1.0
        self.lifetime_starts += 1
        self.last_time = at


@dataclass
class HistoryState:
    horses: dict[str, DecayedStats] = field(default_factory=dict)
    trainers: dict[str, DecayedStats] = field(default_factory=dict)
    jockeys: dict[str, DecayedStats] = field(default_factory=dict)
    horse_surface: dict[tuple[str, str], DecayedStats] = field(default_factory=dict)
    horse_distance: dict[tuple[str, int], DecayedStats] = field(default_factory=dict)
    horse_course: dict[tuple[str, str], DecayedStats] = field(default_factory=dict)


def _stats(mapping: dict, key) -> DecayedStats:
    if key not in mapping:
        mapping[key] = DecayedStats()
    return mapping[key]


def distance_bucket(metres) -> int:
    try:
        return int(round(float(metres) / 400.0) * 400)
    except (TypeError, ValueError):
        return 0


def _active_snapshot_cache(bundle: DataBundle, cutoff_minutes: int) -> dict[str, pd.DataFrame]:
    out = {}
    for race in bundle.races.itertuples(index=False):
        row = bundle.races[bundle.races["race_id"].astype(str) == str(race.race_id)].iloc[0]
        out[str(race.race_id)] = runner_snapshot(
            bundle, str(race.race_id), race_cutoff(row, cutoff_minutes))
    return out


def _result_events(bundle: DataBundle, cutoff_minutes: int) -> list[dict]:
    """Full result versions ordered by the time each version became available."""
    if bundle.results.empty:
        return []
    df = bundle.results.copy()
    df["_version"] = df["record_version"].fillna(1).astype(int)
    events = []
    for (rid, version), rows in df.groupby(["race_id", "_version"], sort=False):
        effective = rows["result_updated_at"].where(rows["result_updated_at"].notna(),
                                                     rows["result_published_at"])
        if effective.isna().any():
            raise DataError(f"result version {rid}/{version} has no availability timestamp")
        events.append({"race_id": str(rid), "version": int(version),
                       "available_at": effective.max(), "rows": rows.copy()})
    by_race: dict[str, set[str]] = {}
    race_lookup = bundle.races.set_index("race_id", drop=False)
    for event in sorted(events, key=lambda e: (e["race_id"], e["version"])):
        runner_ids = set(event["rows"]["runner_id"].astype(str))
        winners = int((pd.to_numeric(event["rows"]["finish_position"],
                                     errors="coerce") == 1).sum())
        if len(runner_ids) < 2 or winners < 1:
            raise DataError(f"result version {event['race_id']}/{event['version']} is incomplete")
        expected = by_race.setdefault(event["race_id"], runner_ids)
        if runner_ids != expected:
            raise DataError(f"result correction {event['race_id']}/{event['version']} "
                            "must contain the same complete runner set")
        if event["race_id"] not in race_lookup.index:
            raise DataError(f"result version references unknown race {event['race_id']!r}")
        race = race_lookup.loc[event["race_id"]]
        own_cutoff = race_cutoff(race, cutoff_minutes)
        scheduled_off = pd.Timestamp(race["scheduled_off_utc"])
        if event["available_at"] <= own_cutoff:
            raise DataError(f"result version {event['race_id']}/{event['version']} "
                            "was timestamped at/before its own prediction cutoff")
        if event["available_at"] < scheduled_off:
            raise DataError(f"result version {event['race_id']}/{event['version']} "
                            "was timestamped before scheduled off")
    return sorted(events, key=lambda e: (e["available_at"], e["race_id"], e["version"]))


def _apply_race(state: HistoryState, race: pd.Series, runners: pd.DataFrame,
                results: pd.DataFrame) -> None:
    if runners.empty or results.empty:
        return
    joined = runners.merge(results[["runner_id", "finish_position", "completion_status"]],
                           on="runner_id", how="inner")
    pos = pd.to_numeric(joined["finish_position"], errors="coerce")
    joined = joined[pos.notna() & (pos > 0)].copy()
    if len(joined) < 2:
        return
    joined["finish_position"] = pd.to_numeric(joined["finish_position"])
    field_n = max(len(joined), int(joined["finish_position"].max()))
    if field_n <= 1:
        return
    at = pd.Timestamp(race["scheduled_off_utc"])
    surface = str(race.get("surface") or "").lower()
    course = str(race.get("course_id") or "")
    dist = distance_bucket(race.get("distance_metres"))
    for row in joined.itertuples(index=False):
        finish = float(row.finish_position)
        performance = 1.0 - 2.0 * (finish - 1.0) / (field_n - 1.0)
        won = finish == 1.0
        _stats(state.horses, str(row.horse_id)).update(at, performance, won, 180.0)
        _stats(state.trainers, str(row.trainer_id)).update(at, performance, won, 365.0)
        _stats(state.jockeys, str(row.jockey_id)).update(at, performance, won, 270.0)
        _stats(state.horse_surface, (str(row.horse_id), surface)).update(
            at, performance, won, 365.0)
        _stats(state.horse_distance, (str(row.horse_id), dist)).update(
            at, performance, won, 365.0)
        _stats(state.horse_course, (str(row.horse_id), course)).update(
            at, performance, won, 540.0)


def _rebuild_state(bundle: DataBundle, latest: dict[str, pd.DataFrame],
                   snapshots: dict[str, pd.DataFrame]) -> tuple[HistoryState, pd.Timestamp | None]:
    state = HistoryState()
    last_off = None
    races = bundle.races.set_index("race_id", drop=False)
    ordered = sorted(latest, key=lambda rid: pd.Timestamp(races.loc[rid, "scheduled_off_utc"]))
    for rid in ordered:
        if rid not in races.index:
            continue
        race = races.loc[rid]
        _apply_race(state, race, snapshots.get(rid, pd.DataFrame()), latest[rid])
        last_off = pd.Timestamp(race["scheduled_off_utc"])
    return state, last_off


def _raw_features(state: HistoryState, race: pd.Series, runners: pd.DataFrame,
                  cutoff: pd.Timestamp) -> pd.DataFrame:
    surface = str(race.get("surface") or "").lower()
    course = str(race.get("course_id") or "")
    dist = distance_bucket(race.get("distance_metres"))
    prior_win = 1.0 / max(len(runners), 1)
    records = []
    for row in runners.itertuples(index=False):
        h = state.horses.get(str(row.horse_id), DecayedStats())
        trainer = state.trainers.get(str(row.trainer_id), DecayedStats())
        jockey = state.jockeys.get(str(row.jockey_id), DecayedStats())
        hs = state.horse_surface.get((str(row.horse_id), surface), DecayedStats())
        hd = state.horse_distance.get((str(row.horse_id), dist), DecayedStats())
        hc = state.horse_course.get((str(row.horse_id), course), DecayedStats())
        h_form, h_win = h.values(cutoff, 180.0, 2.0, 5.0, prior_win)
        _tp, trainer_win = trainer.values(cutoff, 365.0, 10.0, 25.0, prior_win)
        _jp, jockey_win = jockey.values(cutoff, 270.0, 10.0, 25.0, prior_win)
        surface_fit, _ = hs.values(cutoff, 365.0, 3.0, 6.0, prior_win)
        distance_fit, _ = hd.values(cutoff, 365.0, 3.0, 6.0, prior_win)
        course_fit, _ = hc.values(cutoff, 540.0, 4.0, 8.0, prior_win)
        days_since = ((cutoff - h.last_time).total_seconds() / 86400.0
                      if h.last_time is not None else np.nan)
        records.append({
            "race_id": str(row.race_id), "runner_id": str(row.runner_id),
            "horse_id": str(row.horse_id), "horse_name": str(row.horse_name),
            "trainer_id": str(row.trainer_id), "jockey_id": str(row.jockey_id),
            "cutoff": cutoff, "official_rating": row.official_rating,
            "weight": row.weight_carried_kg, "draw": row.draw, "age": row.age,
            "horse_form": h_form, "horse_win": h_win,
            "horse_starts": np.log1p(h.lifetime_starts), "days_since": days_since,
            "trainer_win": trainer_win, "jockey_win": jockey_win,
            "surface_fit": surface_fit, "distance_fit": distance_fit,
            "course_fit": course_fit,
        })
    return pd.DataFrame(records)


def _race_relative(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
    out = raw[["race_id", "runner_id", "horse_id", "horse_name", "trainer_id",
               "jockey_id", "cutoff"]].copy()
    for col in ("official_rating", "weight", "draw", "age", "horse_form", "horse_win",
                "horse_starts", "days_since", "trainer_win", "jockey_win",
                "surface_fit", "distance_fit", "course_fit"):
        values = pd.to_numeric(raw[col], errors="coerce")
        missing = values.isna().astype(float)
        fill = float(values.mean()) if values.notna().any() else 0.0
        values = values.fillna(fill)
        sd = float(values.std(ddof=0))
        out[f"{col}_rel"] = (values - float(values.mean())) / (sd if sd > 1e-9 else 1.0)
        if col in {"official_rating", "weight", "draw", "age", "days_since"}:
            msd = float(missing.std(ddof=0))
            out[f"{col}_missing_rel"] = ((missing - float(missing.mean()))
                                                  / (msd if msd > 1e-9 else 1.0))
    return out


def build_feature_frame(bundle: DataBundle, race_ids: list[str] | None = None,
                        cutoff_minutes: int = 15, include_labels: bool = True) -> pd.DataFrame:
    """Build features in decision-time order while applying only published results.

    Full-version result corrections trigger a state rebuild. That is slower than
    ignoring corrections but keeps historical replay semantically correct.
    """
    wanted = set(map(str, race_ids)) if race_ids is not None else None
    races = bundle.races.copy()
    races["_decision_cutoff"] = races.apply(
        lambda row: race_cutoff(row, cutoff_minutes), axis=1)
    races = races.sort_values(["_decision_cutoff", "scheduled_off_utc", "race_id"])
    snapshots = _active_snapshot_cache(bundle, cutoff_minutes)
    events = _result_events(bundle, cutoff_minutes)
    event_i = 0
    latest: dict[str, pd.DataFrame] = {}
    versions: dict[str, int] = {}
    state = HistoryState()
    last_applied_off: pd.Timestamp | None = None
    race_lookup = bundle.races.set_index("race_id", drop=False)
    frames = []

    for race in races.itertuples(index=False):
        rid = str(race.race_id)
        race_s = race_lookup.loc[rid]
        cutoff = race_cutoff(race_s, cutoff_minutes)
        if pd.Timestamp(race_s["source_updated_at"]) > cutoff:
            raise DataError(f"race {rid!r} metadata was not available by prediction cutoff")
        while event_i < len(events) and events[event_i]["available_at"] <= cutoff:
            event = events[event_i]
            erid = event["race_id"]
            previous = versions.get(erid)
            latest[erid] = event["rows"]
            versions[erid] = event["version"]
            off = (pd.Timestamp(race_lookup.loc[erid, "scheduled_off_utc"])
                   if erid in race_lookup.index else None)
            if previous is None and off is not None and (
                    last_applied_off is None or off >= last_applied_off):
                _apply_race(state, race_lookup.loc[erid], snapshots.get(erid, pd.DataFrame()),
                            event["rows"])
                last_applied_off = off
            else:
                state, last_applied_off = _rebuild_state(bundle, latest, snapshots)
            event_i += 1

        if wanted is not None and rid not in wanted:
            continue
        runners = snapshots.get(rid, pd.DataFrame())
        if len(runners) < 2:
            continue
        frame = _race_relative(_raw_features(state, race_s, runners, cutoff))
        frame["scheduled_off_utc"] = pd.Timestamp(race_s["scheduled_off_utc"])
        frame["course_id"] = str(race_s.get("course_id") or "")
        frame["surface"] = str(race_s.get("surface") or "")
        if include_labels:
            results = final_results(bundle, rid)
            winners = set(results.loc[pd.to_numeric(results.get("finish_position"),
                                                      errors="coerce") == 1,
                                      "runner_id"].astype(str)) if not results.empty else set()
            frame["won"] = frame["runner_id"].isin(winners).astype(int)
        frames.append(frame)
    if not frames:
        cols = ["race_id", "runner_id", "horse_id", "horse_name", "cutoff",
                "scheduled_off_utc", *FEATURES]
        if include_labels:
            cols.append("won")
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames, ignore_index=True)
    for col in FEATURES:
        if col not in out:
            out[col] = 0.0
    return out

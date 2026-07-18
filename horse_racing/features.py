"""Leakage-safe, race-relative feature construction (V2).

V2 replaces the single-half-life history statistics with an explicit
multi-horizon state engine and adds layered feature families that can be
enabled independently for ablation experiments:

- ``core``          V1-parity features (rating, weight, draw, age, form, fits)
- ``form_multi``    multi-horizon form, top-3/top-half rates, DNF rate,
                    last-run performance, pair effects, state confidence
- ``class_struct``  class moves, rating/weight changes, handicap interaction
- ``suitability``   going affinity, course-distance affinity, continuous
                    distance similarity
- ``draw_hier``     hierarchically shrunk course x surface x distance draw bias
- ``weight_rating`` weight/age x distance interactions

Every feature is computed strictly from state accumulated by results whose
availability timestamp precedes the prediction cutoff. The chronological
replay and full-version correction semantics are unchanged from V1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, log

import numpy as np
import pandas as pd

from .schema import DataBundle, DataError, final_results, race_cutoff, runner_snapshot

FEATURE_SCHEMA_VERSION = 2

# Multi-horizon decay half-lives (days). The conditional logit learns its own
# mixture over these horizons, which approximates a learned decay curve; the
# global ``half_life_scale`` build parameter allows explicit decay search.
HORIZON_SHORT = 30.0
HORIZON_MID = 90.0
HORIZON_LONG = 365.0
HORSE_HORIZONS = (HORIZON_SHORT, HORIZON_MID, HORIZON_LONG)
CONNECTION_HORIZONS = (HORIZON_SHORT, HORIZON_LONG)
AFFINITY_HORIZON = 365.0
DRAW_HORIZON = 730.0

# Raw feature columns per family. ``FEATURES`` (the model schema) is derived
# from these via race-relative standardisation below.
MISSING_INDICATOR = {"official_rating", "weight", "draw", "age", "days_since",
                     "or_change", "weight_change"}
FAMILIES: dict[str, list[str]] = {
    "core": ["official_rating", "weight", "draw", "age", "horse_form",
             "horse_win", "horse_starts", "days_since", "trainer_win",
             "jockey_win", "surface_fit", "distance_fit", "course_fit"],
    "form_multi": ["form_short", "form_long", "form_n", "last_perf",
                   "top3_rate", "tophalf_rate", "dnf_rate",
                   "trainer_form_short", "jockey_form_short", "pair_win"],
    "class_struct": ["class_move", "or_change", "weight_change",
                     "or_x_handicap"],
    "suitability": ["going_fit", "course_dist_fit", "dist_delta_abs",
                    "dist_delta_signed"],
    "draw_hier": ["draw_effect", "draw_norm"],
    "weight_rating": ["weight_x_dist", "age_x_dist"],
}
ALL_FAMILIES = tuple(FAMILIES)


def feature_names(families: tuple[str, ...] | list[str] = ALL_FAMILIES) -> list[str]:
    """Model feature schema (race-relative columns) for a family selection."""
    unknown = sorted(set(families) - set(FAMILIES))
    if unknown:
        raise ValueError(f"unknown feature families: {unknown}")
    out: list[str] = []
    for family in ALL_FAMILIES:
        if family not in families:
            continue
        for col in FAMILIES[family]:
            out.append(f"{col}_rel")
            if col in MISSING_INDICATOR:
                out.append(f"{col}_missing_rel")
    return out


FEATURES = feature_names(ALL_FAMILIES)

_LN2 = log(2.0)


class DecayedSums:
    """Named exponentially-decayed sums sharing one clock."""

    __slots__ = ("half_life", "sums", "last_time")

    def __init__(self, half_life: float, keys: tuple[str, ...]):
        self.half_life = float(half_life)
        self.sums = dict.fromkeys(keys, 0.0)
        self.last_time: pd.Timestamp | None = None

    def _decay(self, at: pd.Timestamp) -> None:
        if self.last_time is not None:
            days = max(0.0, (at - self.last_time).total_seconds() / 86400.0)
            factor = exp(-_LN2 * days / self.half_life)
            for key in self.sums:
                self.sums[key] *= factor
        self.last_time = at

    def add(self, at: pd.Timestamp, **increments: float) -> None:
        self._decay(at)
        for key, value in increments.items():
            self.sums[key] += value

    def get(self, at: pd.Timestamp, key: str) -> float:
        if self.last_time is None:
            return 0.0
        days = max(0.0, (at - self.last_time).total_seconds() / 86400.0)
        return self.sums[key] * exp(-_LN2 * days / self.half_life)


_RUN_KEYS = ("perf", "win", "top3", "tophalf", "fin_w", "dnf", "all_w")


class EntityState:
    """Multi-horizon decayed run statistics for one entity.

    Maintains parallel accumulators per half-life so any horizon can be read
    in O(1) without storing event lists. Returns shrunk point estimates plus
    the decayed effective sample size as the confidence measure.
    """

    __slots__ = ("horizons", "lifetime_starts", "last_time", "last_perf")

    def __init__(self, half_lives: tuple[float, ...]):
        self.horizons = {hl: DecayedSums(hl, _RUN_KEYS) for hl in half_lives}
        self.lifetime_starts = 0
        self.last_time: pd.Timestamp | None = None
        self.last_perf: float | None = None

    def update(self, at: pd.Timestamp, perf: float | None, won: bool,
               top3: bool, tophalf: bool, finished: bool) -> None:
        for acc in self.horizons.values():
            if finished and perf is not None:
                acc.add(at, perf=perf, win=float(won), top3=float(top3),
                        tophalf=float(tophalf), fin_w=1.0, all_w=1.0)
            else:
                acc.add(at, dnf=1.0, all_w=1.0)
        self.lifetime_starts += 1
        self.last_time = at
        if finished and perf is not None:
            self.last_perf = perf

    def form(self, at: pd.Timestamp, half_life: float, prior_weight: float = 2.0) -> float:
        acc = self.horizons[half_life]
        w = acc.get(at, "fin_w")
        return acc.get(at, "perf") / (w + prior_weight) if w + prior_weight else 0.0

    def rate(self, at: pd.Timestamp, half_life: float, key: str,
             prior: float, prior_weight: float) -> float:
        acc = self.horizons[half_life]
        denom = "all_w" if key == "dnf" else "fin_w"
        w = acc.get(at, denom)
        return (acc.get(at, key) + prior * prior_weight) / (w + prior_weight)

    def effective_n(self, at: pd.Timestamp, half_life: float) -> float:
        return self.horizons[half_life].get(at, "all_w")


@dataclass
class HistoryState:
    """Explicit per-entity state layer (plan section 2)."""
    horses: dict[str, EntityState] = field(default_factory=dict)
    trainers: dict[str, EntityState] = field(default_factory=dict)
    jockeys: dict[str, EntityState] = field(default_factory=dict)
    pairs: dict[tuple[str, str], EntityState] = field(default_factory=dict)
    horse_surface: dict[tuple[str, str], EntityState] = field(default_factory=dict)
    horse_going: dict[tuple[str, str], EntityState] = field(default_factory=dict)
    horse_distance: dict[tuple[str, int], EntityState] = field(default_factory=dict)
    horse_course: dict[tuple[str, str], EntityState] = field(default_factory=dict)
    horse_course_dist: dict[tuple[str, str, int], EntityState] = field(default_factory=dict)
    horse_class: dict[str, DecayedSums] = field(default_factory=dict)
    horse_logdist: dict[str, DecayedSums] = field(default_factory=dict)
    horse_last: dict[str, dict] = field(default_factory=dict)
    draw_course: dict[tuple[str, str, int], DecayedSums] = field(default_factory=dict)
    draw_surface: dict[tuple[str, int], DecayedSums] = field(default_factory=dict)
    half_life_scale: float = 1.0

    def _entity(self, mapping: dict, key, horizons: tuple[float, ...]) -> EntityState:
        state = mapping.get(key)
        if state is None:
            scaled = tuple(h * self.half_life_scale for h in horizons)
            state = mapping[key] = EntityState(scaled)
        return state

    def _sums(self, mapping: dict, key, half_life: float,
              keys: tuple[str, ...]) -> DecayedSums:
        state = mapping.get(key)
        if state is None:
            state = mapping[key] = DecayedSums(half_life * self.half_life_scale, keys)
        return state

    def scaled(self, half_life: float) -> float:
        return half_life * self.half_life_scale


def going_group(going) -> str:
    raw = str(going or "").strip().lower()
    if not raw:
        return "unknown"
    if "heavy" in raw:
        return "heavy"
    if "soft" in raw or "yielding" in raw:
        return "soft"
    if "firm" in raw or "hard" in raw:
        return "firm"
    if "good" in raw:
        return "good"
    if raw in {"standard", "standard to slow", "standard to fast", "slow", "fast"} \
            or "standard" in raw:
        return "aw_standard"
    return "other"


def distance_bucket(metres) -> int:
    try:
        return int(round(float(metres) / 400.0) * 400)
    except (TypeError, ValueError):
        return 0


def _log_distance(metres) -> float | None:
    try:
        value = float(metres)
    except (TypeError, ValueError):
        return None
    return log(value) if value > 0 else None


def _draw_norms(draws: pd.Series) -> pd.Series:
    """Draw position mapped to [-1, 1] within the declared field."""
    values = pd.to_numeric(draws, errors="coerce")
    valid = values.dropna()
    if len(valid) < 3 or valid.max() == valid.min():
        return pd.Series(np.nan, index=draws.index)
    mid = (valid.max() + valid.min()) / 2.0
    half = (valid.max() - valid.min()) / 2.0
    return (values - mid) / half


def _active_snapshot_cache(bundle: DataBundle, cutoff_minutes: int) -> dict[str, pd.DataFrame]:
    out = {}
    for _, row in bundle.races.iterrows():
        rid = str(row["race_id"])
        out[rid] = runner_snapshot(bundle, rid, race_cutoff(row, cutoff_minutes))
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
        event["scheduled_off"] = scheduled_off
        if event["available_at"] <= own_cutoff:
            raise DataError(f"result version {event['race_id']}/{event['version']} "
                            "was timestamped at/before its own prediction cutoff")
        if event["available_at"] < scheduled_off:
            raise DataError(f"result version {event['race_id']}/{event['version']} "
                            "was timestamped before scheduled off")
    # Multiple races commonly become available in one provider batch. Apply
    # equal-availability events in race-time order so state decay is chronological
    # and the replay does not trigger a full rebuild for every arbitrary ID tie.
    return sorted(events, key=lambda e: (e["available_at"], e["scheduled_off"],
                                         e["race_id"], e["version"]))


def _apply_race(state: HistoryState, race: pd.Series, runners: pd.DataFrame,
                results: pd.DataFrame) -> None:
    if runners.empty or results.empty:
        return
    joined = runners.merge(results[["runner_id", "finish_position", "completion_status"]],
                           on="runner_id", how="inner")
    if joined.empty:
        return
    pos = pd.to_numeric(joined["finish_position"], errors="coerce")
    finishers = joined[pos.notna() & (pos > 0)].copy()
    if len(finishers) < 2:
        return
    finishers["finish_position"] = pd.to_numeric(finishers["finish_position"])
    field_n = max(len(finishers), int(finishers["finish_position"].max()))
    if field_n <= 1:
        return
    at = pd.Timestamp(race["scheduled_off_utc"])
    surface = str(race.get("surface") or "").lower()
    course = str(race.get("course_id") or "")
    going = going_group(race.get("going"))
    dist = distance_bucket(race.get("distance_metres"))
    logdist = _log_distance(race.get("distance_metres"))
    race_class = pd.to_numeric(race.get("race_class"), errors="coerce")
    draw_norm = _draw_norms(joined["draw"])
    finished_ids = set(finishers["runner_id"].astype(str))

    for i, row in enumerate(joined.itertuples(index=False)):
        hid = str(row.horse_id)
        tid, jid = str(row.trainer_id), str(row.jockey_id)
        finished = str(row.runner_id) in finished_ids
        if finished:
            finish = float(pd.to_numeric(row.finish_position))
            perf = 1.0 - 2.0 * (finish - 1.0) / (field_n - 1.0) if field_n > 1 else 0.0
            won = finish == 1.0
            top3 = finish <= 3.0
            tophalf = finish <= field_n / 2.0
        else:
            perf, won, top3, tophalf = None, False, False, False

        state._entity(state.horses, hid, HORSE_HORIZONS).update(
            at, perf, won, top3, tophalf, finished)
        state._entity(state.trainers, tid, CONNECTION_HORIZONS).update(
            at, perf, won, top3, tophalf, finished)
        state._entity(state.jockeys, jid, CONNECTION_HORIZONS).update(
            at, perf, won, top3, tophalf, finished)
        state._entity(state.pairs, (tid, jid), CONNECTION_HORIZONS).update(
            at, perf, won, top3, tophalf, finished)
        for mapping, key in ((state.horse_surface, (hid, surface)),
                             (state.horse_going, (hid, going)),
                             (state.horse_distance, (hid, dist)),
                             (state.horse_course, (hid, course)),
                             (state.horse_course_dist, (hid, course, dist))):
            state._entity(mapping, key, (AFFINITY_HORIZON,)).update(
                at, perf, won, top3, tophalf, finished)

        if not pd.isna(race_class):
            state._sums(state.horse_class, hid, AFFINITY_HORIZON,
                        ("class_sum", "w")).add(at, class_sum=float(race_class), w=1.0)
        if logdist is not None:
            state._sums(state.horse_logdist, hid, AFFINITY_HORIZON,
                        ("logd_sum", "w")).add(at, logd_sum=logdist, w=1.0)
        last = state.horse_last.setdefault(hid, {})
        last["official_rating"] = pd.to_numeric(row.official_rating, errors="coerce")
        last["weight"] = pd.to_numeric(row.weight_carried_kg, errors="coerce")
        last["at"] = at

        dn = draw_norm.iloc[i]
        if finished and perf is not None and not pd.isna(dn):
            for mapping, key in ((state.draw_course, (course, surface, dist)),
                                 (state.draw_surface, (surface, dist))):
                state._sums(mapping, key, DRAW_HORIZON, ("swp", "sww")).add(
                    at, swp=float(dn) * perf, sww=float(dn) ** 2)


def _rebuild_state(bundle: DataBundle, latest: dict[str, pd.DataFrame],
                   snapshots: dict[str, pd.DataFrame],
                   half_life_scale: float = 1.0) -> tuple[HistoryState, pd.Timestamp | None]:
    state = HistoryState(half_life_scale=half_life_scale)
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


def _draw_effect(state: HistoryState, course: str, surface: str, dist: int,
                 at: pd.Timestamp, draw_norm: float) -> float:
    """Hierarchically shrunk draw-bias slope times the runner's draw position.

    Course-level slope shrinks toward the surface-level slope, which shrinks
    toward zero. Priors keep the effect conservative unless the data support a
    strong, persistent bias (plan section 3D).
    """
    if pd.isna(draw_norm):
        return 0.0
    k_course, k_surface = 25.0, 50.0
    surf = state.draw_surface.get((surface, dist))
    if surf is not None:
        beta_surface = surf.get(at, "swp") / (surf.get(at, "sww") + k_surface)
    else:
        beta_surface = 0.0
    node = state.draw_course.get((course, surface, dist))
    if node is not None:
        sww = node.get(at, "sww")
        beta = (node.get(at, "swp") + k_course * beta_surface) / (sww + k_course)
    else:
        beta = beta_surface
    return beta * float(draw_norm)


def _raw_features(state: HistoryState, race: pd.Series, runners: pd.DataFrame,
                  cutoff: pd.Timestamp) -> pd.DataFrame:
    surface = str(race.get("surface") or "").lower()
    course = str(race.get("course_id") or "")
    going = going_group(race.get("going"))
    dist = distance_bucket(race.get("distance_metres"))
    logdist = _log_distance(race.get("distance_metres"))
    race_class = pd.to_numeric(race.get("race_class"), errors="coerce")
    handicap = pd.to_numeric(race.get("handicap_flag"), errors="coerce")
    handicap = float(handicap) if not pd.isna(handicap) else 0.0
    prior_win = 1.0 / max(len(runners), 1)
    hl = state.scaled
    draw_norm = _draw_norms(runners["draw"])
    empty_horse = EntityState(tuple(h * state.half_life_scale for h in HORSE_HORIZONS))
    empty_conn = EntityState(tuple(h * state.half_life_scale for h in CONNECTION_HORIZONS))
    empty_aff = EntityState((AFFINITY_HORIZON * state.half_life_scale,))

    records = []
    for i, row in enumerate(runners.itertuples(index=False)):
        hid = str(row.horse_id)
        tid, jid = str(row.trainer_id), str(row.jockey_id)
        h = state.horses.get(hid, empty_horse)
        trainer = state.trainers.get(tid, empty_conn)
        jockey = state.jockeys.get(jid, empty_conn)
        pair = state.pairs.get((tid, jid), empty_conn)
        hs = state.horse_surface.get((hid, surface), empty_aff)
        hg = state.horse_going.get((hid, going), empty_aff)
        hd = state.horse_distance.get((hid, dist), empty_aff)
        hc = state.horse_course.get((hid, course), empty_aff)
        hcd = state.horse_course_dist.get((hid, course, dist), empty_aff)

        days_since = ((cutoff - h.last_time).total_seconds() / 86400.0
                      if h.last_time is not None else np.nan)
        official_rating = pd.to_numeric(row.official_rating, errors="coerce")
        weight = pd.to_numeric(row.weight_carried_kg, errors="coerce")
        age = pd.to_numeric(row.age, errors="coerce")

        # class / change features from last-run state
        last = state.horse_last.get(hid, {})
        class_state = state.horse_class.get(hid)
        if class_state is not None and not pd.isna(race_class):
            w = class_state.get(cutoff, "w")
            class_avg = class_state.get(cutoff, "class_sum") / w if w > 1e-9 else np.nan
            class_move = float(race_class) - class_avg if not pd.isna(class_avg) else np.nan
        else:
            class_move = np.nan
        prev_or = last.get("official_rating", np.nan)
        or_change = (float(official_rating) - float(prev_or)
                     if not pd.isna(official_rating) and not pd.isna(prev_or) else np.nan)
        prev_weight = last.get("weight", np.nan)
        weight_change = (float(weight) - float(prev_weight)
                         if not pd.isna(weight) and not pd.isna(prev_weight) else np.nan)

        # continuous distance similarity
        logdist_state = state.horse_logdist.get(hid)
        if logdist_state is not None and logdist is not None:
            w = logdist_state.get(cutoff, "w")
            mean_logd = logdist_state.get(cutoff, "logd_sum") / w if w > 1e-9 else None
            dist_delta = logdist - mean_logd if mean_logd is not None else None
        else:
            dist_delta = None

        records.append({
            "race_id": str(row.race_id), "runner_id": str(row.runner_id),
            "horse_id": hid, "horse_name": str(row.horse_name),
            "trainer_id": tid, "jockey_id": jid, "cutoff": cutoff,
            # core (V1 parity)
            "official_rating": official_rating, "weight": weight,
            "draw": pd.to_numeric(row.draw, errors="coerce"), "age": age,
            "horse_form": h.form(cutoff, hl(HORIZON_MID)),
            "horse_win": h.rate(cutoff, hl(HORIZON_MID), "win", prior_win, 5.0),
            "horse_starts": np.log1p(h.lifetime_starts), "days_since": days_since,
            "trainer_win": trainer.rate(cutoff, hl(HORIZON_LONG), "win", prior_win, 25.0),
            "jockey_win": jockey.rate(cutoff, hl(HORIZON_LONG), "win", prior_win, 25.0),
            "surface_fit": hs.form(cutoff, hl(AFFINITY_HORIZON), 3.0),
            "distance_fit": hd.form(cutoff, hl(AFFINITY_HORIZON), 3.0),
            "course_fit": hc.form(cutoff, hl(AFFINITY_HORIZON), 4.0),
            # form_multi
            "form_short": h.form(cutoff, hl(HORIZON_SHORT)),
            "form_long": h.form(cutoff, hl(HORIZON_LONG)),
            "form_n": np.log1p(h.effective_n(cutoff, hl(HORIZON_MID))),
            "last_perf": h.last_perf if h.last_perf is not None else 0.0,
            "top3_rate": h.rate(cutoff, hl(HORIZON_MID), "top3", 3.0 * prior_win, 4.0),
            "tophalf_rate": h.rate(cutoff, hl(HORIZON_MID), "tophalf", 0.5, 4.0),
            "dnf_rate": h.rate(cutoff, hl(HORIZON_LONG), "dnf", 0.05, 5.0),
            "trainer_form_short": trainer.form(cutoff, hl(HORIZON_SHORT), 4.0),
            "jockey_form_short": jockey.form(cutoff, hl(HORIZON_SHORT), 4.0),
            "pair_win": pair.rate(cutoff, hl(HORIZON_LONG), "win", prior_win, 12.0),
            # class_struct
            "class_move": class_move, "or_change": or_change,
            "weight_change": weight_change,
            "or_x_handicap": (float(official_rating) * handicap
                              if not pd.isna(official_rating) else np.nan),
            # suitability
            "going_fit": hg.form(cutoff, hl(AFFINITY_HORIZON), 3.0),
            "course_dist_fit": hcd.form(cutoff, hl(AFFINITY_HORIZON), 4.0),
            "dist_delta_abs": abs(dist_delta) if dist_delta is not None else np.nan,
            "dist_delta_signed": dist_delta if dist_delta is not None else np.nan,
            # draw_hier
            "draw_effect": _draw_effect(state, course, surface, dist, cutoff,
                                        draw_norm.iloc[i]),
            "draw_norm": draw_norm.iloc[i],
            # weight_rating
            "weight_x_dist": (float(weight) * logdist
                              if not pd.isna(weight) and logdist is not None else np.nan),
            "age_x_dist": (float(age) * logdist
                           if not pd.isna(age) and logdist is not None else np.nan),
        })
    return pd.DataFrame(records)


_RAW_COLUMNS = [col for family in ALL_FAMILIES for col in FAMILIES[family]]


def _race_relative(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
    out = raw[["race_id", "runner_id", "horse_id", "horse_name", "trainer_id",
               "jockey_id", "cutoff"]].copy()
    for col in _RAW_COLUMNS:
        values = pd.to_numeric(raw[col], errors="coerce")
        missing = values.isna().astype(float)
        fill = float(values.mean()) if values.notna().any() else 0.0
        values = values.fillna(fill)
        sd = float(values.std(ddof=0))
        out[f"{col}_rel"] = (values - float(values.mean())) / (sd if sd > 1e-9 else 1.0)
        if col in MISSING_INDICATOR:
            msd = float(missing.std(ddof=0))
            out[f"{col}_missing_rel"] = ((missing - float(missing.mean()))
                                                  / (msd if msd > 1e-9 else 1.0))
    return out


def build_feature_frame(bundle: DataBundle, race_ids: list[str] | None = None,
                        cutoff_minutes: int = 15, include_labels: bool = True,
                        half_life_scale: float = 1.0) -> pd.DataFrame:
    """Build features in decision-time order while applying only published results.

    Full-version result corrections trigger a state rebuild. That is slower than
    ignoring corrections but keeps historical replay semantically correct.
    ``half_life_scale`` multiplies every decay half-life, enabling explicit
    decay-rate search without touching feature code.
    """
    if not (0.05 <= float(half_life_scale) <= 20.0):
        raise ValueError("half_life_scale must be within [0.05, 20]")
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
    state = HistoryState(half_life_scale=float(half_life_scale))
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
                state, last_applied_off = _rebuild_state(bundle, latest, snapshots,
                                                         float(half_life_scale))
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
        frame["going_group"] = going_group(race_s.get("going"))
        frame["distance_band"] = distance_bucket(race_s.get("distance_metres"))
        handicap = pd.to_numeric(race_s.get("handicap_flag"), errors="coerce")
        frame["handicap_flag"] = float(handicap) if not pd.isna(handicap) else 0.0
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

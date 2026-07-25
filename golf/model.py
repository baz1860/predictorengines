"""
golf/model.py  –  Composite player rating model (Strokes Gained approach).

Rating = strokes gained per round vs field average.
Positive rating → better player (gains strokes on field).

Composite formula:
    rating = 0.55 * sg_baseline  +  0.30 * course_fit  +  0.15 * recent_form

Scoring per round:
    score_vs_field ~ Normal(-rating, sigma)

Where sigma (≈ 3.0) is the round-to-round scoring variance for the course.
This captures the fact that golf has high variance — even a 2-stroke-better
player loses to a 150th-ranked player in a given week ~28% of the time.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent / "data"
ROUNDS_CSV = DATA_DIR / "rounds.csv"
PARAMS_JSON = DATA_DIR / "model_params.json"
MODEL_CONFIG_JSON = DATA_DIR / "model_config.json"
PUBLIC_STATS_CSV = DATA_DIR / "pgatour_stats.csv"
DATA_MANIFEST_JSON = DATA_DIR / "data_manifest.json"
WEATHER_FEATURES_JSON = DATA_DIR / "weather_features.json"
GLOBAL_PRIORS_CSV = DATA_DIR / "global_player_priors.csv"

# Weight parameters for composite rating
W_BASELINE = 0.55
W_COURSE   = 0.30
W_FORM     = 0.15

# Min rounds at course to use course fit (otherwise weight redistribution)
MIN_COURSE_ROUNDS = 2

# Recent-form decay half-life in rounds
FORM_TAU = 4.0

# Default round-to-round σ (strokes vs field) if not set per course
DEFAULT_SIGMA = 3.0

# Major championship σ adjustment
MAJOR_SIGMA_BUMP = 0.15


@dataclass
class Player:
    name: str
    dg_id: str = ""
    sg_baseline: float = 0.0    # season SG:Total vs average field
    sg_ott: float = 0.0         # SG: off the tee
    sg_app: float = 0.0         # SG: approach
    sg_atg: float = 0.0         # SG: around the green
    sg_putt: float = 0.0        # SG: putting
    datagolf_skill: float = 0.0 # DataGolf composite (if available)
    owgr: int = 999             # Official World Golf Ranking
    country: str = ""
    tee_time_r1: str = ""
    tee_time_r2: str = ""
    start_hole_r1: str = ""
    start_hole_r2: str = ""
    weather_wave_adj: float = 0.0
    weather_round_adj: dict = field(default_factory=dict)
    global_prior_adj: float = 0.0
    course_fit: float = 0.0     # SG at this specific course (filled by load_course_fit)
    course_rounds: int = 0      # How many rounds at this course
    recent_form: float = 0.0    # Exponentially-weighted recent SG
    # Computed composite
    rating: float = 0.0
    sigma: float = DEFAULT_SIGMA
    birdie_rate: float = 0.18  # birdie-or-better holes / holes played
    bogey_rate: float = 0.14   # exactly-bogey holes / holes played
    blowup_rate: float = 0.02  # double-bogey-or-worse holes / holes played
    scoring_shape_sample: int = 0
    scoring_shape_source: str = "field"


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val) if val not in ("", None) else default
    except (ValueError, TypeError):
        return default


def _safe_int(val, default: int = 999) -> int:
    try:
        return int(float(val)) if val not in ("", None) else default
    except (ValueError, TypeError):
        return default


# ─────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────

def load_players(path: Path | None = None) -> dict[str, Player]:
    """Load players.csv → dict keyed by lowercase name."""
    path = path or DATA_DIR / "players.csv"
    if not path.exists():
        return {}

    players: dict[str, Player] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            name = row.get("name", "").strip()
            if not name:
                continue

            # Prefer DataGolf skill if populated, otherwise SG:Total
            dg_skill = _safe_float(row.get("datagolf_skill"))
            sg_total = _safe_float(row.get("sg_total"))
            baseline = dg_skill if dg_skill != 0.0 else sg_total

            p = Player(
                name=name,
                dg_id=row.get("dg_id", ""),
                sg_baseline=baseline,
                sg_ott=_safe_float(row.get("sg_ott")),
                sg_app=_safe_float(row.get("sg_app")),
                sg_atg=_safe_float(row.get("sg_atg")),
                sg_putt=_safe_float(row.get("sg_putt")),
                datagolf_skill=dg_skill,
                owgr=_safe_int(row.get("owgr"), 999),
                country=row.get("country", ""),
            )
            players[name.lower()] = p

    return players


def load_field(
    path: Path | None = None,
    players: dict[str, Player] | None = None,
) -> list[Player]:
    """
    Load field.csv → list of Player objects for the current tournament.
    Merges SG ratings from players dict if available.
    """
    path = path or DATA_DIR / "field.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No field file at {path}. Run: python -m golf.fetch --espn"
        )

    players = players or {}
    field_players: list[Player] = []

    with open(path) as f:
        for row in csv.DictReader(f):
            name = row.get("name", "").strip()
            if not name:
                continue

            # Look up ratings from players.csv
            p = players.get(name.lower())
            if p is None:
                p = Player(name=name)
            p.dg_id = str(row.get("dg_id") or p.dg_id or "")

            # Override sigma from field.csv if set
            sigma_override = _safe_float(row.get("course_sigma"), 0.0)
            if sigma_override > 0:
                p.sigma = sigma_override
            p.owgr = _safe_int(row.get("world_rank") or row.get("owgr"), p.owgr)
            p.tee_time_r1 = row.get("tee_time_r1", "")
            p.tee_time_r2 = row.get("tee_time_r2", "")
            p.start_hole_r1 = row.get("start_hole_r1", "")
            p.start_hole_r2 = row.get("start_hole_r2", "")

            field_players.append(p)

    return field_players


def load_field_event(path: Path | None = None) -> str:
    """Event name recorded in field.csv ('' when absent). This is the
    ESPN-resolved name of the current tournament — odds boards are tagged
    with the same string, so pricers compare the two to reject stale boards."""
    path = path or DATA_DIR / "field.csv"
    if not path.exists():
        return ""
    with open(path) as f:
        for row in csv.DictReader(f):
            ev = (row.get("event") or "").strip()
            if ev:
                return ev
    return ""


def load_field_context(path: Path | None = None) -> dict:
    """Current event/course metadata embedded in ``field.csv`` by refresh."""
    path = path or DATA_DIR / "field.csv"
    if not path.exists():
        return {}
    try:
        with path.open() as handle:
            row = next(csv.DictReader(handle), None) or {}
    except (OSError, csv.Error):
        return {}
    return {
        "event_id": str(row.get("event_id") or "").strip(),
        "event": str(row.get("event") or "").strip(),
        "course": str(row.get("course") or "").strip(),
        "course_id": str(row.get("course_id") or "").strip(),
        "course_par": _safe_int(row.get("course_par"), 0),
        "course_yards": _safe_int(row.get("course_yards"), 0),
        "par3_holes": _safe_int(row.get("par3_holes"), 0),
        "par4_holes": _safe_int(row.get("par4_holes"), 0),
        "par5_holes": _safe_int(row.get("par5_holes"), 0),
    }


def load_course_history(
    course: str,
    path: Path | None = None,
) -> dict[str, tuple[float, int]]:
    """
    Load course_history.csv for a specific course.
    Returns dict: player_name_lower → (avg_sg_at_course, rounds_played)
    """
    path = path or DATA_DIR / "course_history.csv"
    if not path.exists():
        return {}

    course_lower = course.lower()
    history: dict[str, tuple[float, int]] = {}

    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("course", "").lower() != course_lower:
                continue
            name = row.get("player", "").strip().lower()
            sg = _safe_float(row.get("sg_at_course"))
            rounds = _safe_int(row.get("rounds_played"), 0)
            if name:
                history[name] = (sg, rounds)

    return history


def load_recent_form(path: Path | None = None) -> dict[str, float]:
    """
    Load recent_form.csv (optional) → player → exp-weighted SG.
    This file can be generated from DataGolf historical rounds or manually.
    """
    path = path or DATA_DIR / "recent_form.csv"
    if not path.exists():
        return {}

    form: dict[str, float] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            name = row.get("player", "").strip().lower()
            sg = _safe_float(row.get("weighted_sg"))
            if name:
                form[name] = sg
    return form


# ─────────────────────────────────────────────
# Rating computation
# ─────────────────────────────────────────────

def compute_ratings(
    players: list[Player],
    course: str = "",
    is_major: bool = False,
    course_history: dict | None = None,
    recent_form: dict | None = None,
) -> list[Player]:
    """
    Compute composite rating for each player in field.

    Modifies players in-place, returns same list sorted by rating descending.
    """
    course_history = course_history or {}
    recent_form    = recent_form or {}

    for p in players:
        key = p.name.lower()

        # ── course fit ──
        if key in course_history:
            sg_course, n_rounds = course_history[key]
            if n_rounds >= MIN_COURSE_ROUNDS:
                p.course_fit = sg_course
                p.course_rounds = n_rounds
            else:
                p.course_fit = 0.0
                p.course_rounds = n_rounds
        else:
            p.course_fit = 0.0
            p.course_rounds = 0

        # ── recent form ──
        if key in recent_form:
            p.recent_form = recent_form[key]

        # ── weight redistribution when course fit unavailable ──
        has_course = p.course_rounds >= MIN_COURSE_ROUNDS
        has_form   = p.recent_form != 0.0

        if has_course and has_form:
            w_b, w_c, w_f = W_BASELINE, W_COURSE, W_FORM
        elif has_course and not has_form:
            w_b, w_c, w_f = W_BASELINE + W_FORM, W_COURSE, 0.0
        elif not has_course and has_form:
            w_b, w_c, w_f = W_BASELINE + W_COURSE, 0.0, W_FORM
        else:
            w_b, w_c, w_f = 1.0, 0.0, 0.0

        p.rating = (
            w_b * p.sg_baseline
            + w_c * p.course_fit
            + w_f * p.recent_form
        )

        # ── sigma ──
        if p.sigma == DEFAULT_SIGMA and is_major:
            p.sigma += MAJOR_SIGMA_BUMP

    # Normalise ratings so field average = 0
    if players:
        mean_rating = sum(p.rating for p in players) / len(players)
        for p in players:
            p.rating -= mean_rating

    return sorted(players, key=lambda p: p.rating, reverse=True)


# ─────────────────────────────────────────────
# Utility: expected finish distribution
# (analytical approximation, used for sanity checks)
# ─────────────────────────────────────────────

def expected_win_prob_normal(rating: float, sigma: float, n_players: int) -> float:
    """
    Rough analytical win probability for a player with `rating` strokes
    advantage in a field of `n_players` where round scores are iid Normal.

    4-round tournament: total σ = σ_round * 2 (variance adds).
    This ignores the cut — use simulate.py for accurate cut-adjusted probs.
    """
    total_sigma = sigma * math.sqrt(4)  # 4 independent rounds
    # P(player beats one opponent) ~ Phi(rating / (total_sigma * sqrt(2)))
    p_beat_one = 0.5 * (1 + math.erf(rating / (total_sigma * math.sqrt(2))))
    # Win = beat all n-1 opponents (independence approximation)
    return p_beat_one ** (n_players - 1)


# ═════════════════════════════════════════════════════════════════════════
# v2: FITTED skill + variance model (fit from data/rounds.csv)
#
# Decompose every round:   score_to_par = mu + difficulty[t,r] - skill[p] + ε
#   • skill[p]        strokes-gained vs field (higher = better, scores lower)
#   • difficulty[t,r] per tournament-round level → field-strength & setup adjust
#   • ε ~ Normal(0, sigma[p])   per-player round-to-round variance
# Solved by time-decayed, ridge-shrunk sparse least squares (cfb/power.py
# analogue). Ridge on skill gives regression-to-mean for low-sample players.
# sigma, recent form, and course fit come from the fit residuals.
# ═════════════════════════════════════════════════════════════════════════

# Fit hyper-parameters (tuned further by validate.py)
SKILL_HALFLIFE_DAYS = 365.0     # decay for the durable skill estimate
RIDGE_SKILL = 8.0               # shrink skill→0 in equivalent-round weights
RIDGE_DIFF = 1.0               # light shrink on tournament-round levels
SIGMA_SHRINK_ROUNDS = 25.0      # Empirical-Bayes prior weight for per-player σ
FORM_HALFLIFE_DAYS = 21.0       # short-window recency for "form"
FORM_WINDOW_DAYS = 70           # only rounds inside this window feed form
FORM_K = 12.0                   # EB shrink for form (equivalent rounds)
FORM_WEIGHT = 0.7               # how much form nudges the rating (validate tunes)
COURSE_PROFILE_RIDGE = 40.0     # shrink player par/yardage sensitivity
COURSE_PROFILE_WEIGHT = 0.5     # conservative general-course adjustment
DEFAULT_SKILL_QUANTILE = 0.20   # rating for unknown players (weak-field default)

DEFAULT_MODEL_CONFIG = {
    "skill_halflife_days": SKILL_HALFLIFE_DAYS,
    "ridge_skill": RIDGE_SKILL,
    "sigma_shrink_rounds": SIGMA_SHRINK_ROUNDS,
    "form_halflife_days": FORM_HALFLIFE_DAYS,
    "form_weight": FORM_WEIGHT,
    "course_profile_ridge": COURSE_PROFILE_RIDGE,
    "course_profile_weight": COURSE_PROFILE_WEIGHT,
}

PUBLIC_STAT_BLEND = 0.15
GLOBAL_PRIOR_MAX_BLEND = 0.25
WEATHER_WAVE_MAX_ABS = 0.35
COURSE_PROFILE_MAX_ABS = 0.35


def load_model_config(path: Path | None = None) -> dict:
    """Champion fit hyperparameters, falling back to the validated constants."""
    path = path or MODEL_CONFIG_JSON
    cfg = dict(DEFAULT_MODEL_CONFIG)
    if path.exists():
        try:
            raw = json.loads(path.read_text())
            raw = raw.get("config", raw)
            for k in cfg:
                if k in raw:
                    cfg[k] = float(raw[k])
        except Exception:
            pass
    return cfg


def save_model_config(config: dict, metrics: dict | None = None,
                      path: Path | None = None) -> Path:
    path = path or MODEL_CONFIG_JSON
    payload = {"config": {k: float(config[k]) for k in DEFAULT_MODEL_CONFIG},
               "metrics": metrics or {},
               "source": "golf/validate.py --tune-config"}
    from .io_utils import atomic_write_text
    atomic_write_text(path, json.dumps(payload, indent=2))
    return path


# Plausible band for SG: Total per round; values beyond this are mis-scraped.
MAX_SANE_SG_TOTAL = 4.0

# Scoreboard markers — keep in sync with pgatour_stats._SCOREBOARD_KEYS.
_SCOREBOARD_KEYS = {
    "position", "roundscore", "totalscore", "thru", "teetime", "starthole",
    "leaderboardsortorder", "groupnumber", "scoresort", "currentround",
}


def _looks_like_scoreboard(raw_json: str | None) -> bool:
    """True if a stored stat row is actually a leaked live-leaderboard entry."""
    if not raw_json:
        return False
    try:
        blob = json.loads(raw_json)
    except (ValueError, TypeError):
        return False
    if not isinstance(blob, dict):
        return False
    return any(str(k).lower() in _SCOREBOARD_KEYS for k in blob)


def _public_stats_capture_date(path: Path) -> dt.date | None:
    """Point-in-time date for the current public-stat snapshot."""
    if path == PUBLIC_STATS_CSV and DATA_MANIFEST_JSON.exists():
        try:
            manifest = json.loads(DATA_MANIFEST_JSON.read_text())
            raw = (((manifest.get("inputs") or {}).get("pga_stats") or {})
                   .get("fetched_at"))
            if raw:
                return dt.datetime.fromisoformat(
                    str(raw).replace("Z", "+00:00")
                ).date()
        except (OSError, ValueError, TypeError):
            pass
    try:
        return dt.datetime.fromtimestamp(
            path.stat().st_mtime, tz=dt.timezone.utc
        ).date()
    except OSError:
        return None


def load_public_stat_priors(path: Path | None = None, *, asof=None) -> dict[str, dict]:
    """Load current public PGA Tour stat snapshots into player rating priors.

    The provider writes one row per player/stat. SG: Total is the preferred
    prior; otherwise we synthesize a conservative total from SG tee-to-green and
    putting, or category components. Values are already strokes gained per round.
    """
    path = path or PUBLIC_STATS_CSV
    if not path.exists():
        return {}
    # This file is a current snapshot, not a point-in-time history. Compare the
    # snapshot's own capture date to the fit cutoff: historical fits before the
    # capture stay clean, while a weekday refit after the last played round does
    # not silently discard a valid older snapshot.
    if asof is not None:
        import pandas as pd
        captured = _public_stats_capture_date(path)
        if captured is not None and pd.Timestamp(asof).date() < captured:
            return {}
    rows: dict[str, dict] = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            name = (r.get("player_name") or "").strip()
            stat = (r.get("stat_name") or "").strip().lower()
            try:
                value = float(r.get("value"))
            except (TypeError, ValueError):
                continue
            if not name:
                continue
            # Skip live-leaderboard rows that leaked in tagged as a stat: their
            # raw_json carries scoreboard keys and `value` is a position/score,
            # not strokes gained (see pgatour_stats._is_scoreboard_entry).
            if _looks_like_scoreboard(r.get("raw_json")):
                continue
            rows.setdefault(name, {})[stat] = value

    priors = {}
    for name, vals in rows.items():
        sg_total = vals.get("sg_total")
        if sg_total is None and "sg_t2g" in vals and "sg_putt" in vals:
            sg_total = vals["sg_t2g"] + vals["sg_putt"]
        if sg_total is None:
            parts = [vals.get(k) for k in ("sg_ott", "sg_app", "sg_arg", "sg_putt")]
            parts = [p for p in parts if p is not None]
            if parts:
                sg_total = sum(parts)
        if sg_total is None:
            continue
        # Per-round SG: Total sits in roughly [-4, +4]. Anything outside that is
        # a mis-scraped value (rank / scoreboard number) — drop, don't blend it.
        if not -MAX_SANE_SG_TOTAL <= sg_total <= MAX_SANE_SG_TOTAL:
            continue
        priors[name] = {
            "sg_total": round(float(sg_total), 4),
            "stats": {k: round(float(v), 4) for k, v in vals.items()},
        }
    return priors


def load_rounds_df(path: Path | None = None):
    """Read rounds.csv → DataFrame (raises if absent)."""
    import pandas as pd
    path = path or ROUNDS_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"No {path}. Seed it first: python -m golf.fetch --seed 2022 2023 2024 2025")
    df = pd.read_csv(path)
    if "course_name" in df:
        # New histories carry the real venue. Keep the legacy `course` alias for
        # downstream callers while ensuring event names can no longer win.
        df["course"] = df["course_name"].fillna(df.get("course", "")).astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "player", "score_to_par"])
    df["score_to_par"] = df["score_to_par"].astype(float)
    # Defense in depth: provider and integrity checks reject these first, but a
    # hand-edited CSV must not be able to explode fitted skill and variance.
    df = df[df["score_to_par"].between(-15.0, 30.0)]
    return df


def fit(rounds_df, asof=None, config: dict | None = None,
        include_public_stats: bool | None = None) -> dict:
    """Fit skill, difficulty, sigma, form and course-fit on rounds before `asof`.

    Returns a params dict (see save_params for the JSON shape). Walk-forward safe:
    only rounds strictly before `asof` are used.
    """
    import numpy as np
    import pandas as pd
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import lsqr

    cfg = load_model_config() if config is None else {**DEFAULT_MODEL_CONFIG, **config}
    skill_halflife = float(cfg["skill_halflife_days"])
    ridge_skill = float(cfg["ridge_skill"])
    sigma_shrink = float(cfg["sigma_shrink_rounds"])
    form_halflife = float(cfg["form_halflife_days"])
    form_weight = float(cfg["form_weight"])
    course_profile_ridge = float(cfg["course_profile_ridge"])
    course_profile_weight = float(cfg["course_profile_weight"])

    df = rounds_df
    if asof is not None:
        asof = pd.Timestamp(asof)
        df = df[df["date"] < asof]
    if len(df) < 500:
        raise ValueError(f"only {len(df)} rounds before {asof} — need ≥500")
    asof = asof or (df["date"].max() + pd.Timedelta(days=1))

    df = df.copy()
    source_ids = df.get("dg_id", pd.Series("", index=df.index)).fillna("").astype(str)
    source_ids = source_ids.str.replace(r"\.0$", "", regex=True).str.strip()
    df["player_key"] = np.where(
        source_ids.ne(""),
        "source:" + source_ids,
        "name:" + df["player"].map(_fold_name),
    )
    players = sorted(df["player_key"].unique())
    display_names = (
        df.sort_values("date")
        .groupby("player_key")["player"]
        .last()
        .to_dict()
    )
    pi = {p: i for i, p in enumerate(players)}
    np_ = len(players)
    # tournament-round group key
    df = df.assign(tr=df["tournament_id"].astype(str) + "|" + df["round"].astype(str))
    trs = sorted(df["tr"].unique())
    di = {t: i for i, t in enumerate(trs)}
    nd = len(trs)

    age = (asof - df["date"]).dt.days.values.astype(float)
    w = np.sqrt(0.5 ** (age / skill_halflife))   # weight on squared resid
    mu = float(np.average(df["score_to_par"].values, weights=w ** 2))
    y = df["score_to_par"].values - mu

    # Sparse design: each row has  +1·diff[tr]  −1·skill[p].  Columns 0..np-1 =
    # skill, np..np+nd-1 = difficulty. Rows scaled by w; ridge rows appended.
    pidx = df["player_key"].map(pi).values
    tidx = df["tr"].map(di).values + np_
    m = len(df)
    rows = np.repeat(np.arange(m), 2)
    cols = np.empty(2 * m, dtype=int); cols[0::2] = pidx; cols[1::2] = tidx
    vals = np.empty(2 * m, dtype=float); vals[0::2] = -w; vals[1::2] = w
    b = y * w

    # ridge rows: skill→0 (weight √RIDGE_SKILL), diff→0 (weight √RIDGE_DIFF)
    n_un = np_ + nd
    rr = np.arange(n_un) + m
    rc = np.arange(n_un)
    rv = np.r_[np.full(np_, math.sqrt(ridge_skill)),
               np.full(nd, math.sqrt(RIDGE_DIFF))]
    A = csr_matrix((np.r_[vals, rv], (np.r_[rows, rr], np.r_[cols, rc])),
                   shape=(m + n_un, n_un))
    bb = np.r_[b, np.zeros(n_un)]
    x = lsqr(A, bb, atol=1e-8, btol=1e-8, iter_lim=2000)[0]
    skill = x[:np_]
    diff = x[np_:]

    # residuals (unweighted, in stroke units) for σ / form / course fit
    pred = mu + diff[df["tr"].map(di).values] - skill[pidx]
    resid = df["score_to_par"].values - pred

    # ── per-player σ, Empirical-Bayes shrunk toward field σ ──
    var_field = float(np.average(resid ** 2, weights=w ** 2))
    sigma_field = math.sqrt(var_field)
    counts = np.bincount(pidx, minlength=np_).astype(float)
    sse = np.bincount(pidx, weights=resid ** 2, minlength=np_)
    var_p = np.divide(sse, counts, out=np.full(np_, var_field), where=counts > 0)
    var_shrunk = (counts * var_p + sigma_shrink * var_field) / \
                 (counts + sigma_shrink)
    sigma_p = np.sqrt(var_shrunk)

    # ── per-player right-tail proxy from double-bogey-or-worse holes ──
    hole_mask = (
        df.get("holes_scored", pd.Series(0, index=df.index)).fillna(0).values > 0
    )
    holes = df.get(
        "holes_scored", pd.Series(0, index=df.index)
    ).fillna(0).values.astype(float)
    birdies = df.get(
        "birdies_or_better", pd.Series(0, index=df.index)
    ).fillna(0).values.astype(float)
    bogeys = df.get(
        "bogeys", pd.Series(0, index=df.index)
    ).fillna(0).values.astype(float)
    doubles = df.get(
        "double_bogeys_or_worse", pd.Series(0, index=df.index)
    ).fillna(0).values.astype(float)
    total_holes = float(holes[hole_mask].sum())
    birdie_rate_field = (
        float(birdies[hole_mask].sum() / total_holes)
        if total_holes > 0 else 0.18
    )
    bogey_rate_field = (
        float(bogeys[hole_mask].sum() / total_holes)
        if total_holes > 0 else 0.14
    )
    double_bogey_rate_field = (
        float(doubles[hole_mask].sum() / total_holes)
        if total_holes > 0 else 0.02
    )
    holes_p = np.bincount(pidx, weights=holes, minlength=np_)
    birdies_p = np.bincount(pidx, weights=birdies, minlength=np_)
    bogeys_p = np.bincount(pidx, weights=bogeys, minlength=np_)
    doubles_p = np.bincount(pidx, weights=doubles, minlength=np_)
    # Twenty rounds of field-rate pseudo-observations prevent tiny scorecards
    # from creating extreme simulated tails.
    blowup_prior_holes = 360.0
    birdie_rate_p = (
        birdies_p + blowup_prior_holes * birdie_rate_field
    ) / (holes_p + blowup_prior_holes)
    bogey_rate_p = (
        bogeys_p + blowup_prior_holes * bogey_rate_field
    ) / (holes_p + blowup_prior_holes)
    blowup_rate_p = (
        doubles_p + blowup_prior_holes * double_bogey_rate_field
    ) / (holes_p + blowup_prior_holes)

    # ── par-3 / par-4 / par-5 player profile ──
    # Each observation is net of that tournament-round's field scoring rate for
    # the same par type. The profile therefore captures relative suitability,
    # not an easier setup or a hot scoring week. Shrink by 20 rounds of holes.
    par_profiles = [dict() for _ in range(np_)]
    par_profile_prior_holes = 360.0
    for par in (3, 4, 5):
        value_col = pd.to_numeric(
            df.get(f"par{par}_to_par", pd.Series(0, index=df.index)),
            errors="coerce",
        ).fillna(0.0)
        holes_col = pd.to_numeric(
            df.get(f"par{par}_holes", pd.Series(0, index=df.index)),
            errors="coerce",
        ).fillna(0.0)
        tr_total = value_col.groupby(df["tr"]).transform("sum")
        tr_holes = holes_col.groupby(df["tr"]).transform("sum")
        tr_rate = np.divide(
            tr_total.values,
            tr_holes.values,
            out=np.zeros(len(df), dtype=float),
            where=tr_holes.values > 0,
        )
        par_holes = holes_col.values.astype(float)
        par_sg = -(value_col.values.astype(float) - par_holes * tr_rate)
        player_holes = np.bincount(pidx, weights=par_holes, minlength=np_)
        player_sg = np.bincount(pidx, weights=par_sg, minlength=np_)
        player_rounds = np.bincount(
            pidx, weights=(par_holes > 0).astype(float), minlength=np_
        )
        par_skill = player_sg / (player_holes + par_profile_prior_holes)
        centers = np.divide(
            player_holes,
            player_rounds,
            out=np.zeros(np_, dtype=float),
            where=player_rounds > 0,
        )
        for i in range(np_):
            if player_holes[i] >= 72:
                par_profiles[i][f"par{par}_skill_per_hole"] = round(
                    float(par_skill[i]), 6
                )
                par_profiles[i][f"par{par}_center_holes"] = round(
                    float(centers[i]), 3
                )

    # ── major σ multiplier ──
    is_major = df["is_major"].astype(int).values == 1
    if is_major.sum() > 200:
        maj = math.sqrt(np.average(resid[is_major] ** 2, weights=(w[is_major]) ** 2))
        major_sigma_mult = round(max(0.9, min(1.3, maj / sigma_field)), 3)
    else:
        major_sigma_mult = 1.05

    # Exact-course effects failed honest ablation and are disabled. They were
    # sparse, unbounded, and particularly unsafe for rotating multi-course events.
    courses: dict[str, dict[str, float]] = {}
    single_course = (
        pd.to_numeric(
            df.get("multi_course", pd.Series(0, index=df.index)),
            errors="coerce",
        ).fillna(0).values == 0
    )
    course_net_resid = resid

    # ── player sensitivity to measured course par and yardage ──
    # Estimate within-player slopes on field/difficulty-adjusted residuals.
    # Centering each player's observed course mix keeps durable skill from being
    # counted again; ridge shrinkage prevents sparse schedules producing large
    # course-style claims. Exact-venue course fit takes precedence at prediction.
    par_values = pd.to_numeric(
        df.get("course_par", pd.Series(np.nan, index=df.index)), errors="coerce"
    ).values.astype(float)
    yard_values = pd.to_numeric(
        df.get("course_yards", pd.Series(np.nan, index=df.index)), errors="coerce"
    ).values.astype(float)
    valid_course = (
        np.isfinite(par_values) & np.isfinite(yard_values)
        & (par_values >= 68) & (par_values <= 75)
        & (yard_values >= 5000) & (yard_values <= 9000)
        & single_course
    )
    if valid_course.any():
        course_weights = w[valid_course] ** 2
        par_mean = float(np.average(par_values[valid_course], weights=course_weights))
        yard_mean = float(np.average(yard_values[valid_course], weights=course_weights))
        par_sd = float(np.sqrt(np.average(
            (par_values[valid_course] - par_mean) ** 2, weights=course_weights
        ))) or 1.0
        yard_sd = float(np.sqrt(np.average(
            (yard_values[valid_course] - yard_mean) ** 2, weights=course_weights
        ))) or 1.0
    else:
        par_mean, par_sd = 72.0, 1.0
        yard_mean, yard_sd = 7200.0, 500.0
    z_par = np.where(valid_course, (par_values - par_mean) / par_sd, 0.0)
    z_yards = np.where(valid_course, (yard_values - yard_mean) / yard_sd, 0.0)
    course_profile = [None] * np_
    player_order = np.argsort(pidx, kind="stable")
    player_stops = np.cumsum(counts.astype(int))
    player_start = 0
    for i, player_stop in enumerate(player_stops):
        player_rows = player_order[player_start:player_stop]
        player_start = player_stop
        rows_valid = player_rows[valid_course[player_rows]]
        n_profile = len(rows_valid)
        if n_profile < 12:
            continue
        ww = w[rows_valid] ** 2
        x = np.column_stack((z_par[rows_valid], z_yards[rows_valid]))
        center = np.average(x, axis=0, weights=ww)
        xc = x - center
        target = -course_net_resid[rows_valid]
        target = target - np.average(target, weights=ww)
        xtwx = (xc.T * ww) @ xc + course_profile_ridge * np.eye(2)
        beta = np.linalg.solve(xtwx, (xc.T * ww) @ target)
        course_profile[i] = {
            "par_slope": round(float(beta[0]), 5),
            "yards_slope": round(float(beta[1]), 5),
            "par_center_z": round(float(center[0]), 5),
            "yards_center_z": round(float(center[1]), 5),
            "rounds": n_profile,
        }
    recent_cut = asof - pd.Timedelta(days=FORM_WINDOW_DAYS)
    rmask = df["date"].values >= np.datetime64(recent_cut)
    fw = np.sqrt(0.5 ** (age / form_halflife)) * rmask
    fsum = np.bincount(pidx, weights=-course_net_resid * fw, minlength=np_)
    fwsum = np.bincount(pidx, weights=fw, minlength=np_)
    fcnt = np.bincount(pidx, weights=rmask.astype(float), minlength=np_)
    form_raw = np.divide(fsum, fwsum, out=np.zeros(np_), where=fwsum > 0)
    form = form_raw * (fcnt / (fcnt + FORM_K))

    default_skill = float(np.quantile(skill, DEFAULT_SKILL_QUANTILE))
    if include_public_stats is None:
        include_public_stats = False
    public_priors = load_public_stat_priors(asof=asof) if include_public_stats else {}

    return {
        "asof": str(pd.Timestamp(asof).date()),
        "mu": round(mu, 4),
        "sigma_field": round(sigma_field, 4),
        "major_sigma_mult": major_sigma_mult,
        "double_bogey_rate_field": round(double_bogey_rate_field, 6),
        "birdie_rate_field": round(birdie_rate_field, 6),
        "bogey_rate_field": round(bogey_rate_field, 6),
        "course_context": {
            "par_mean": round(par_mean, 4),
            "par_sd": round(par_sd, 4),
            "yards_mean": round(yard_mean, 2),
            "yards_sd": round(yard_sd, 2),
        },
        "skill_halflife_days": skill_halflife,
        "ridge_skill": ridge_skill,
        "sigma_shrink_rounds": sigma_shrink,
        "form_halflife_days": form_halflife,
        "form_weight": form_weight,
        "course_profile_ridge": course_profile_ridge,
        "course_profile_weight": course_profile_weight,
        "model_config": {k: float(cfg[k]) for k in DEFAULT_MODEL_CONFIG},
        "default_skill": round(default_skill, 4),
        "public_stat_blend": PUBLIC_STAT_BLEND,
        "public_stat_priors": public_priors,
        "fitted_rounds": int(m),
        "players": {
            p: {
                "display_name": str(display_names[p]),
                "source_player_id": p.removeprefix("source:")
                    if p.startswith("source:") else "",
                "skill": round(float(skill[i]), 4),
                "sigma": round(float(sigma_p[i]), 4),
                "form": round(float(form[i]), 4),
                "n_rounds": int(counts[i]),
                "birdie_rate": round(float(birdie_rate_p[i]), 6),
                "bogey_rate": round(float(bogey_rate_p[i]), 6),
                "blowup_rate": round(float(blowup_rate_p[i]), 6),
                "hole_sample": int(holes_p[i]),
                **({"par_profile": par_profiles[i]} if par_profiles[i] else {}),
                **({"course_profile": course_profile[i]} if course_profile[i] else {}),
            } for i, p in enumerate(players)
        },
        "courses": courses,
    }


def _assert_sane_priors(params: dict) -> None:
    """Refuse to persist scoreboard-contaminated priors. The loader already drops
    out-of-range rows, but a prior can only reach here via a clean fit — so a
    value outside ±MAX_SANE_SG_TOTAL means something bypassed the read guard
    (e.g. a hand-edited or legacy params dict). Fail loud rather than bake a
    leaderboard position into the rating that the card never re-fits away."""
    bad = []
    for name, row in (params.get("public_stat_priors") or {}).items():
        try:
            sg = float(row["sg_total"])
        except (KeyError, TypeError, ValueError):
            continue
        if not -MAX_SANE_SG_TOTAL <= sg <= MAX_SANE_SG_TOTAL:
            bad.append((name, sg))
    if bad:
        preview = ", ".join(f"{n}={v:g}" for n, v in bad[:5])
        raise ValueError(
            f"refusing to save model_params.json: {len(bad)} public_stat_prior(s) "
            f"outside ±{MAX_SANE_SG_TOTAL} SG — looks like scoreboard contamination "
            f"({preview}). Refit from a clean pgatour_stats.csv.")


def save_params(params: dict, path: Path | None = None) -> Path:
    _assert_sane_priors(params)
    path = path or PARAMS_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    from .io_utils import atomic_write_text
    atomic_write_text(path, json.dumps(params, indent=1))
    return path


def load_params(path: Path | None = None) -> dict | None:
    path = path or PARAMS_JSON
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


_TRANSLIT = str.maketrans({
    "ø": "o", "Ø": "o", "æ": "ae", "Æ": "ae", "ð": "d", "Ð": "d",
    "þ": "th", "Þ": "th", "ł": "l", "Ł": "l", "ß": "ss",
})


def _fold_name(name: str) -> str:
    """Accent-, case- and punctuation-insensitive key for matching player names
    across sources (e.g. 'Ludvig Aberg' from a book vs fitted 'Ludvig Åberg',
    'Hojgaard' vs 'Højgaard' — ø/æ do not decompose under NFKD, so transliterate
    first — and 'J J Spaun' vs 'J.J. Spaun', where punctuation differs)."""
    s = str(name).translate(_TRANSLIT)
    nfkd = unicodedata.normalize("NFKD", s)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Punctuation (dots in initials, apostrophes, hyphens) → space so e.g.
    # "J.J." and "J J" fold the same; whitespace is then collapsed.
    stripped = "".join(c if (c.isalnum() or c.isspace()) else " " for c in stripped)
    return " ".join(stripped.lower().split())


# Nickname / first-name aliases that fold-matching cannot resolve. Maps an
# alternate name (any source) to the canonical fitted name. Matched folded on
# both sides (see _FOLDED_ALIASES), so case/accents of the source don't matter.
NAME_ALIASES = {
    "Matthew Fitzpatrick": "Matt Fitzpatrick",
    "Christopher Gotterup": "Chris Gotterup",
    "Alexander Noren": "Alex Noren",
    "Joohyung Kim": "Tom Kim",
    "Jayden Trey Schaper": "Jayden Schaper",
    "Adrien Dumont": "Adrien Dumont de Chassart",
    "John Keefer": "Johnny Keefer",
    "Benjamin James": "Ben James",
    "Nicolas Echavarria": "Nico Echavarria",
    "Samuel Stevens": "Sam Stevens",
}

# Folded alias keys so lookups are case/accent-insensitive (e.g. a board's
# "SAMUEL STEVENS" still hits the "Samuel Stevens" entry).
_FOLDED_ALIASES = {_fold_name(k): v for k, v in NAME_ALIASES.items()}


def _folded_index(params: dict) -> dict[str, str | None]:
    """Folded display name → stable fitted identity, refusing ambiguity."""
    idx = params.get("_folded_index")
    if idx is None:
        idx = {}
        for n, row in params.get("players", {}).items():
            key = _fold_name(row.get("display_name") or n)
            if key in idx and idx[key] != n:
                idx[key] = None  # ambiguous: never silently pick the last player
            else:
                idx[key] = n
        for n in params.get("public_stat_priors", {}):
            key = _fold_name(n)
            if key not in idx:
                idx[key] = n
        params["_folded_index"] = idx
    return idx


def resolve_name(name: str, params: dict, source_player_id: str = "") -> str | None:
    """Stable fitted identity for a source id or unambiguous display name."""
    players = params.get("players", {})
    source_key = f"source:{str(source_player_id).strip()}"
    if source_player_id and source_key in players:
        return source_key
    if name in players:
        return name
    idx = _folded_index(params)
    hit = idx.get(_fold_name(name))
    if hit:
        return hit
    alias = _FOLDED_ALIASES.get(_fold_name(name))
    if alias:
        return idx.get(_fold_name(alias), alias if alias in players else None)
    return None


def _public_stat_prior(name: str, params: dict, canon: str | None = None) -> float | None:
    priors = params.get("public_stat_priors", {}) or {}
    for key in (canon, name):
        if key and key in priors:
            try:
                return float(priors[key]["sg_total"])
            except (KeyError, TypeError, ValueError):
                return None
    folded = _fold_name(name)
    for p_name, row in priors.items():
        if _fold_name(p_name) == folded:
            try:
                return float(row["sg_total"])
            except (KeyError, TypeError, ValueError):
                return None
    return None


_STAT_ALIGN_CACHE: dict[int, dict] = {}


def _public_stat_alignment(params: dict) -> dict | None:
    """Location/scale mapping public SG-total priors onto the fitted-skill scale.

    Fitted skill is measured against the full modelled pool (tour regulars sit
    well above 0), while public SG-total is measured against the PGA Tour field
    (tour regulars sit near 0). Blending them raw drags any player who *has* a
    public prior toward ~0 while players without one keep full skill, which
    inverts head-to-head ordering (a strong player with a prior can be rated
    below a weaker one without). Matching the prior's mean and spread to the
    fitted-skill distribution over the overlap set makes the blend a genuine
    regularisation in one coordinate system. Cached per params object.
    """
    key = id(params)
    if key in _STAT_ALIGN_CACHE:
        return _STAT_ALIGN_CACHE[key] or None
    players = params.get("players", {}) or {}
    fw = float(params.get("form_weight", FORM_WEIGHT))
    fits: list[float] = []
    pubs: list[float] = []
    for identity, d in players.items():
        if "skill" not in d:
            continue
        display_name = str(d.get("display_name") or identity)
        sp = _public_stat_prior(display_name, params, identity)
        if sp is None:
            continue
        fits.append(float(d["skill"]) + fw * float(d.get("form", 0.0)))
        pubs.append(sp)
    if len(fits) < 10:
        _STAT_ALIGN_CACHE[key] = {}
        return None
    import statistics as _st
    align = {
        "fit_mean": _st.fmean(fits), "fit_sd": _st.pstdev(fits) or 1.0,
        "pub_mean": _st.fmean(pubs), "pub_sd": _st.pstdev(pubs) or 1.0,
    }
    _STAT_ALIGN_CACHE[key] = align
    return align


def _aligned_public_prior(stat_prior: float, params: dict) -> float:
    """Rescale a public SG-total prior onto the fitted-skill scale."""
    a = _public_stat_alignment(params)
    if not a:
        return stat_prior
    z = (stat_prior - a["pub_mean"]) / a["pub_sd"]
    return a["fit_mean"] + z * a["fit_sd"]


def _public_stat_components(name: str, params: dict, canon: str | None = None) -> dict[str, float]:
    priors = params.get("public_stat_priors", {}) or {}
    candidates = [canon, name]
    folded = _fold_name(name)
    candidates.extend(
        p_name for p_name in priors
        if _fold_name(p_name) == folded
    )
    for key in candidates:
        row = priors.get(key) if key else None
        stats = (row or {}).get("stats") or {}
        out = {}
        for k, v in stats.items():
            try:
                out[str(k).lower()] = float(v)
            except (TypeError, ValueError):
                continue
        if out:
            return out
    return {}


def load_weather_features(path: Path | None = None) -> dict:
    """Load optional tournament/round weather features written by refresh.py."""
    path = path or WEATHER_FEATURES_JSON
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def load_global_player_priors(path: Path | None = None) -> dict[str, dict]:
    """Manual/global-tour priors for thin PGA samples.

    CSV columns: name, sg_total, sigma, source, notes. These are intentionally
    explicit rather than scraped, so majors can include LIV/DPWT/international
    players without forcing them through a weak PGA-only default.
    """
    path = path or GLOBAL_PRIORS_CSV
    if not path.exists():
        return {}
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            name = (row.get("name") or row.get("player") or "").strip()
            if not name:
                continue
            try:
                sg = float(row.get("sg_total") or row.get("skill"))
            except (TypeError, ValueError):
                continue
            rec = {"sg_total": max(-2.5, min(3.0, sg))}
            try:
                sig = float(row.get("sigma") or 0.0)
                if sig > 0:
                    rec["sigma"] = max(1.8, min(4.5, sig))
            except (TypeError, ValueError):
                pass
            rec["source"] = row.get("source", "")
            rec["notes"] = row.get("notes", "")
            out[_fold_name(name)] = rec
    return out


def _owgr_skill_prior(rank: int | None) -> float | None:
    """Conservative SG/round prior from world rank for thin global samples."""
    if rank is None or rank <= 0 or rank >= 999:
        return None
    # Smoothly maps roughly: #1≈2.2, #10≈1.3, #50≈0.7, #100≈0.4, #300≈0.0.
    return max(-0.6, min(2.2, 2.2 - 0.4 * math.log(float(rank))))


def _global_player_prior(name: str, canon: str | None = None,
                         priors: dict | None = None) -> dict | None:
    priors = priors if priors is not None else load_global_player_priors()
    for key in (name, canon):
        folded = _fold_name(key or "")
        if folded in priors:
            return priors[folded]
    return None


def _tee_hour(value: str, timezone: str = "") -> float | None:
    import re
    from zoneinfo import ZoneInfo

    text = str(value or "").strip()
    if not text:
        return None
    try:
        # ESPN tee times are commonly ISO-8601 UTC. Open-Meteo wave hours are
        # local course time, so convert before extracting the hour.
        if "T" in text:
            parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is not None and timezone and timezone != "auto":
                parsed = parsed.astimezone(ZoneInfo(timezone))
            return parsed.hour + parsed.minute / 60.0
        m12 = re.search(r"\b(\d{1,2}):(\d{2})\s*([AP]M)\b", text, re.I)
        if m12:
            hour = int(m12.group(1)) % 12
            if m12.group(3).lower() == "pm":
                hour += 12
            return hour + int(m12.group(2)) / 60.0
        m24 = re.search(r"\b(\d{1,2}):(\d{2})(?::\d{2})?\b", text)
        if m24:
            return int(m24.group(1)) + int(m24.group(2)) / 60.0
        hh, mm = text[:5].split(":")
        return int(hh) + int(mm) / 60.0
    except Exception:
        return None


def _weather_wave_adjustment(player: Player, weather_features: dict,
                             round_no: int = 1) -> float:
    if not weather_features:
        return 0.0
    rounds = weather_features.get("rounds") or {}
    r = rounds.get(str(round_no)) or weather_features
    wave = r.get("wave_penalty") or {}
    if not wave:
        return 0.0
    location = weather_features.get("location") or {}
    timezone = str(location.get("timezone") or weather_features.get("timezone") or "")
    tee = _tee_hour(
        getattr(player, f"tee_time_r{round_no}", ""),
        timezone=timezone,
    )
    if tee is None:
        return 0.0
    split = float(wave.get("split_hour", 12.0))
    side = "late" if tee >= split else "early"
    penalty = float(wave.get(f"{side}_penalty", 0.0))
    # Positive rating means strokes gained; a weather penalty lowers rating.
    return max(-WEATHER_WAVE_MAX_ABS, min(WEATHER_WAVE_MAX_ABS, -penalty))


def _weather_round_adjustments(player: Player, weather_features: dict) -> dict[int, float]:
    return {
        rnd: _weather_wave_adjustment(player, weather_features, rnd)
        for rnd in range(1, 5)
    }


def _course_profile_adjustment(
    player_params: dict,
    params: dict,
    course_par: int = 0,
    course_yards: int = 0,
    par3_holes: int = 0,
    par4_holes: int = 0,
    par5_holes: int = 0,
) -> float:
    profile = player_params.get("course_profile") or {}
    par_profile = player_params.get("par_profile") or {}
    context = params.get("course_context") or {}
    if not profile and not par_profile:
        return 0.0
    raw = 0.0
    if profile and course_par and course_yards:
        par_sd = float(context.get("par_sd") or 1.0)
        yards_sd = float(context.get("yards_sd") or 1.0)
        z_par = (
            float(course_par) - float(context.get("par_mean") or 72.0)
        ) / par_sd
        z_yards = (
            float(course_yards) - float(context.get("yards_mean") or 7200.0)
        ) / yards_sd
        raw += (
            float(profile.get("par_slope") or 0.0)
            * (z_par - float(profile.get("par_center_z") or 0.0))
            + float(profile.get("yards_slope") or 0.0)
            * (z_yards - float(profile.get("yards_center_z") or 0.0))
        )
    if par_profile:
        # Standard layouts have four par 3s; when the feed omits hole objects,
        # total par still identifies the usual par-5 count (par = 68 + n_par5).
        if not any((par3_holes, par4_holes, par5_holes)) and course_par:
            par3_holes = 4
            par5_holes = max(0, min(6, int(course_par) - 68))
            par4_holes = 18 - par3_holes - par5_holes
        for par, count in (
            (3, par3_holes),
            (4, par4_holes),
            (5, par5_holes),
        ):
            if count:
                raw += (
                    float(count)
                    - float(par_profile.get(f"par{par}_center_holes") or count)
                ) * float(par_profile.get(f"par{par}_skill_per_hole") or 0.0)
    weight = float(params.get("course_profile_weight", COURSE_PROFILE_WEIGHT))
    return max(-COURSE_PROFILE_MAX_ABS, min(COURSE_PROFILE_MAX_ABS, weight * raw))


def _rating_for_components(name: str, params: dict, course: str = "",
                           course_par: int = 0, course_yards: int = 0,
                           par3_holes: int = 0, par4_holes: int = 0,
                           par5_holes: int = 0,
                           world_rank: int | None = None,
                           source_player_id: str = "",
                           global_priors: dict | None = None,
                           feature_flags: dict | None = None) -> tuple[float, float, dict]:
    """(rating, sigma, components) for one player from fitted params."""
    canon = resolve_name(name, params, source_player_id)
    pl = params.get("players", {}).get(canon) if canon else None
    fw = params.get("form_weight", FORM_WEIGHT)
    flags = {
        "public_stat": False,
        "global_priors": False,
        "exact_course": False,
        "course_profile": True,
        **(feature_flags or {}),
    }
    stat_prior = _public_stat_prior(name, params, canon) if flags["public_stat"] else None
    manual_global = _global_player_prior(name, canon, global_priors) if flags["global_priors"] else None
    global_prior = _owgr_skill_prior(world_rank) if flags["global_priors"] else None
    components = {
        "base": 0.0,
        "form": 0.0,
        "public_stat": 0.0,
        "course_fit": 0.0,
        "course_profile": 0.0,
        "global_prior": 0.0,
    }
    if pl is None:
        rating = params.get("default_skill", -0.5)
        if manual_global is not None:
            rating = float(manual_global["sg_total"])
            components["global_prior"] = rating
        elif stat_prior is not None:
            rating = stat_prior
            components["public_stat"] = stat_prior
        elif global_prior is not None:
            rating = global_prior
            components["global_prior"] = global_prior
        sigma = manual_global.get("sigma") if manual_global and manual_global.get("sigma") else \
            params.get("sigma_field", DEFAULT_SIGMA) * 1.08
    else:
        skill = float(pl["skill"])
        form_adj = fw * float(pl.get("form", 0.0))
        rating = skill + form_adj
        components["base"] = skill
        components["form"] = form_adj
        if stat_prior is not None:
            # Public priors live on the PGA-Tour-relative SG scale; map them onto
            # the fitted-skill scale before blending so the regularisation can't
            # invert head-to-head ordering (see _public_stat_alignment).
            stat_prior = _aligned_public_prior(stat_prior, params)
            blend = float(params.get("public_stat_blend", PUBLIC_STAT_BLEND))
            components["public_stat"] = blend * (stat_prior - rating)
            rating = (1 - blend) * rating + blend * stat_prior
        gp = float(manual_global["sg_total"]) if manual_global is not None else global_prior
        if gp is not None:
            n_rounds = float(pl.get("n_rounds", 0.0) or 0.0)
            blend = min(GLOBAL_PRIOR_MAX_BLEND, 30.0 / (n_rounds + 120.0))
            components["global_prior"] = blend * (gp - rating)
            rating = (1 - blend) * rating + blend * gp
        sigma = pl.get("sigma", params.get("sigma_field", DEFAULT_SIGMA))
    course_rows = params.get("courses", {}).get(course, {}) if course else {}
    has_exact_course = bool(
        flags["exact_course"] and canon and canon in course_rows
    )
    if has_exact_course:
        cf = course_rows.get(canon, 0.0)
        components["course_fit"] = cf
        rating += cf
    elif pl is not None and flags["course_profile"]:
        profile_adj = _course_profile_adjustment(
            pl,
            params,
            course_par=course_par,
            course_yards=course_yards,
            par3_holes=par3_holes,
            par4_holes=par4_holes,
            par5_holes=par5_holes,
        )
        components["course_profile"] = profile_adj
        rating += profile_adj
    return rating, sigma, components


def rating_for(name: str, params: dict, course: str = "",
               course_par: int = 0, course_yards: int = 0,
               par3_holes: int = 0, par4_holes: int = 0,
               par5_holes: int = 0,
               world_rank: int | None = None) -> tuple[float, float]:
    """(rating, sigma) for one player from fitted params. Unknown → default."""
    rating, sigma, _components = _rating_for_components(
        name, params, course, course_par=course_par, course_yards=course_yards,
        par3_holes=par3_holes, par4_holes=par4_holes, par5_holes=par5_holes,
        world_rank=world_rank)
    return rating, sigma


def predict_field(field_names, params: dict, course: str = "",
                  course_par: int = 0, course_yards: int = 0,
                  par3_holes: int = 0, par4_holes: int = 0,
                  par5_holes: int = 0,
                  is_major: bool = False, weather_features: dict | None = None,
                  round_no: int = 1, feature_flags: dict | None = None) -> list[Player]:
    """Build rated Player objects for a field from fitted params.

    Accepts an iterable of names or Player objects. Ratings are centred on the
    field mean (= 0) so simulate.py reads them directly; σ keeps absolute scale.
    """
    maj_mult = params.get("major_sigma_mult", 1.0) if is_major else 1.0
    flags = {
        "weather": False,
        "public_stat": False,
        "global_priors": False,
        "exact_course": False,
        "course_profile": True,
        "scoring_shape": True,
        **(feature_flags or {}),
    }
    global_priors = load_global_player_priors() if flags["global_priors"] else {}
    weather_features = load_weather_features() if weather_features is None else weather_features
    out: list[Player] = []
    for item in field_names:
        name = item.name if isinstance(item, Player) else str(item)
        world_rank = getattr(item, "owgr", None) if isinstance(item, Player) else None
        source_player_id = (
            getattr(item, "dg_id", "") if isinstance(item, Player) else ""
        )
        rating, sigma, comps = _rating_for_components(
            name, params, course, course_par=course_par,
            course_yards=course_yards, par3_holes=par3_holes,
            par4_holes=par4_holes, par5_holes=par5_holes,
            world_rank=world_rank,
            source_player_id=source_player_id,
            global_priors=global_priors,
            feature_flags=flags)
        canon = resolve_name(name, params, source_player_id)
        pl = params.get("players", {}).get(canon, {}) if canon else {}
        p = Player(name=name)
        if isinstance(item, Player):
            p.dg_id = item.dg_id
            p.owgr = item.owgr
            p.country = item.country
            p.tee_time_r1 = item.tee_time_r1
            p.tee_time_r2 = item.tee_time_r2
            p.start_hole_r1 = item.start_hole_r1
            p.start_hole_r2 = item.start_hole_r2
        p.rating = rating
        p.sigma = sigma * maj_mult
        hole_sample = int(pl.get("hole_sample", 0) or 0)
        use_player_shape = flags["scoring_shape"] and hole_sample >= 360
        p.scoring_shape_sample = hole_sample
        p.scoring_shape_source = "player" if use_player_shape else "field"
        p.birdie_rate = float(
            pl.get("birdie_rate")
            if use_player_shape
            else params.get("birdie_rate_field", 0.18)
        )
        p.bogey_rate = float(
            pl.get("bogey_rate")
            if use_player_shape
            else params.get("bogey_rate_field", 0.14)
        )
        p.blowup_rate = float(
            pl.get("blowup_rate")
            if use_player_shape
            else params.get("double_bogey_rate_field", 0.02)
        )
        p.sg_baseline = pl.get("skill", rating)
        p.recent_form = pl.get("form", 0.0)
        p.course_fit = comps.get("course_fit", 0.0)
        p.global_prior_adj = comps.get("global_prior", 0.0)
        p.weather_round_adj = _weather_round_adjustments(p, weather_features) \
            if flags["weather"] else {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        p.weather_wave_adj = p.weather_round_adj.get(round_no, 0.0)
        out.append(p)
    if out:
        mean_r = sum(p.rating for p in out) / len(out)
        for p in out:
            p.rating -= mean_r
    return sorted(out, key=lambda p: p.rating, reverse=True)


# ─────────────────────────────────────────────
# Quick summary printer
# ─────────────────────────────────────────────

def print_ratings(players: list[Player], top_n: int = 30) -> None:
    print(f"\n{'Rank':<5} {'Player':<30} {'Rating':>7} {'Baseline':>9} {'CourseFit':>10} {'Form':>7} {'σ':>5} {'OWGR':>5}")
    print("-" * 80)
    for i, p in enumerate(players[:top_n], 1):
        print(
            f"{i:<5} {p.name:<30} {p.rating:>+7.3f} {p.sg_baseline:>+9.3f} "
            f"{p.course_fit:>+10.3f} {p.recent_form:>+7.3f} {p.sigma:>5.2f} {p.owgr:>5}"
        )


# ─────────────────────────────────────────────
# CLI (standalone rating inspection)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Fit / inspect golf player ratings")
    ap.add_argument("--fit", action="store_true",
                    help="Fit from data/rounds.csv and save model_params.json")
    ap.add_argument("--course", default="", help="Course name for fit lookup")
    ap.add_argument("--major", action="store_true", help="Apply major sigma adjustment")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    # ── Fit mode (v2) ──
    if args.fit:
        params = fit(load_rounds_df())
        save_params(params)
        print(f"Fitted {len(params['players'])} players from "
              f"{params['fitted_rounds']:,} rounds as of {params['asof']}: "
              f"mu={params['mu']:.2f}, σ_field={params['sigma_field']:.2f}, "
              f"major×{params['major_sigma_mult']}")
        ranked = sorted(params["players"].items(),
                        key=lambda kv: -(kv[1]["skill"] + params["form_weight"] * kv[1]["form"]))
        print(f"\n{'Rank':<5}{'Player':<26}{'Rating':>8}{'Skill':>8}{'Form':>7}{'σ':>6}{'N':>5}")
        print("-" * 65)
        for i, (name, pl) in enumerate(ranked[:args.top], 1):
            rating = pl["skill"] + params["form_weight"] * pl["form"]
            label = str(pl.get("display_name") or name)
            print(f"{i:<5}{label:<26}{rating:>+8.3f}{pl['skill']:>+8.3f}"
                  f"{pl['form']:>+7.3f}{pl['sigma']:>6.2f}{pl['n_rounds']:>5}")
        raise SystemExit(0)

    # ── Inspect a field with fitted params (fallback: legacy players.csv) ──
    params = load_params()
    field_p = load_field(players=load_players())
    if not field_p:
        print("No field.csv found. Run fetch.py --espn first.")
        raise SystemExit(1)

    if params:
        rated = predict_field(field_p, params, course=args.course, is_major=args.major)
    else:
        print("(no model_params.json — using legacy players.csv ratings)")
        ch = load_course_history(args.course) if args.course else {}
        rated = compute_ratings(field_p, course=args.course, is_major=args.major,
                                course_history=ch, recent_form=load_recent_form())
    print_ratings(rated, top_n=args.top)

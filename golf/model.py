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
    course_fit: float = 0.0     # SG at this specific course (filled by load_course_fit)
    course_rounds: int = 0      # How many rounds at this course
    recent_form: float = 0.0    # Exponentially-weighted recent SG
    # Computed composite
    rating: float = 0.0
    sigma: float = DEFAULT_SIGMA


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

            # Override sigma from field.csv if set
            sigma_override = _safe_float(row.get("course_sigma"), 0.0)
            if sigma_override > 0:
                p.sigma = sigma_override

            field_players.append(p)

    return field_players


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
COURSE_K = 12.0                 # EB shrink for course fit
DEFAULT_SKILL_QUANTILE = 0.20   # rating for unknown players (weak-field default)

DEFAULT_MODEL_CONFIG = {
    "skill_halflife_days": SKILL_HALFLIFE_DAYS,
    "ridge_skill": RIDGE_SKILL,
    "sigma_shrink_rounds": SIGMA_SHRINK_ROUNDS,
    "form_halflife_days": FORM_HALFLIFE_DAYS,
    "form_weight": FORM_WEIGHT,
    "course_k": COURSE_K,
}

PUBLIC_STAT_BLEND = 0.15


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
    path.write_text(json.dumps(payload, indent=2))
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


def load_public_stat_priors(path: Path | None = None) -> dict[str, dict]:
    """Load current public PGA Tour stat snapshots into player rating priors.

    The provider writes one row per player/stat. SG: Total is the preferred
    prior; otherwise we synthesize a conservative total from SG tee-to-green and
    putting, or category components. Values are already strokes gained per round.
    """
    path = path or PUBLIC_STATS_CSV
    if not path.exists():
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
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "player", "score_to_par"])
    df["score_to_par"] = df["score_to_par"].astype(float)
    return df


def fit(rounds_df, asof=None, config: dict | None = None) -> dict:
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
    course_k = float(cfg["course_k"])

    df = rounds_df
    if asof is not None:
        asof = pd.Timestamp(asof)
        df = df[df["date"] < asof]
    if len(df) < 500:
        raise ValueError(f"only {len(df)} rounds before {asof} — need ≥500")
    asof = asof or (df["date"].max() + pd.Timedelta(days=1))

    players = sorted(df["player"].unique())
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
    pidx = df["player"].map(pi).values
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

    # ── major σ multiplier ──
    is_major = df["is_major"].astype(int).values == 1
    if is_major.sum() > 200:
        maj = math.sqrt(np.average(resid[is_major] ** 2, weights=(w[is_major]) ** 2))
        major_sigma_mult = round(max(0.9, min(1.3, maj / sigma_field)), 3)
    else:
        major_sigma_mult = 1.05

    # ── recent form: −EB-shrunk weighted-mean recent residual (positive = hot) ──
    recent_cut = asof - pd.Timedelta(days=FORM_WINDOW_DAYS)
    rmask = df["date"].values >= np.datetime64(recent_cut)
    fw = np.sqrt(0.5 ** (age / form_halflife)) * rmask
    fsum = np.bincount(pidx, weights=-resid * fw, minlength=np_)
    fwsum = np.bincount(pidx, weights=fw, minlength=np_)
    fcnt = np.bincount(pidx, weights=rmask.astype(float), minlength=np_)
    form_raw = np.divide(fsum, fwsum, out=np.zeros(np_), where=fwsum > 0)
    form = form_raw * (fcnt / (fcnt + FORM_K))

    # ── course fit: −EB-shrunk mean residual per (player, course) ──
    courses: dict[str, dict[str, float]] = {}
    cdiff = df.assign(resid=resid)
    for course, grp in cdiff.groupby("course"):
        cp = grp.groupby("player")["resid"].agg(["mean", "count"])
        cp = cp[cp["count"] >= 4]
        if cp.empty:
            continue
        fit_vals = (-cp["mean"]) * (cp["count"] / (cp["count"] + course_k))
        courses[str(course)] = {p: round(float(v), 3)
                                for p, v in fit_vals.items() if abs(v) > 0.05}

    default_skill = float(np.quantile(skill, DEFAULT_SKILL_QUANTILE))
    public_priors = load_public_stat_priors()

    return {
        "asof": str(pd.Timestamp(asof).date()),
        "mu": round(mu, 4),
        "sigma_field": round(sigma_field, 4),
        "major_sigma_mult": major_sigma_mult,
        "skill_halflife_days": skill_halflife,
        "ridge_skill": ridge_skill,
        "sigma_shrink_rounds": sigma_shrink,
        "form_halflife_days": form_halflife,
        "form_weight": form_weight,
        "course_k": course_k,
        "model_config": {k: float(cfg[k]) for k in DEFAULT_MODEL_CONFIG},
        "default_skill": round(default_skill, 4),
        "public_stat_blend": PUBLIC_STAT_BLEND,
        "public_stat_priors": public_priors,
        "fitted_rounds": int(m),
        "players": {
            p: {
                "skill": round(float(skill[i]), 4),
                "sigma": round(float(sigma_p[i]), 4),
                "form": round(float(form[i]), 4),
                "n_rounds": int(counts[i]),
            } for i, p in enumerate(players)
        },
        "courses": courses,
    }


def save_params(params: dict, path: Path | None = None) -> Path:
    path = path or PARAMS_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(params, f, indent=1)
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


def _folded_index(params: dict) -> dict[str, str]:
    """Folded-name → canonical fitted name, cached on the params dict."""
    idx = params.get("_folded_index")
    if idx is None:
        idx = {_fold_name(n): n for n in params.get("players", {})}
        for n in params.get("public_stat_priors", {}):
            idx.setdefault(_fold_name(n), n)
        params["_folded_index"] = idx
    return idx


def resolve_name(name: str, params: dict) -> str | None:
    """Canonical fitted name for `name`, tolerant of accents/case. None if unknown."""
    players = params.get("players", {})
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


def rating_for(name: str, params: dict, course: str = "") -> tuple[float, float]:
    """(rating, sigma) for one player from fitted params. Unknown → default."""
    canon = resolve_name(name, params)
    pl = params.get("players", {}).get(canon) if canon else None
    fw = params.get("form_weight", FORM_WEIGHT)
    stat_prior = _public_stat_prior(name, params, canon)
    if pl is None:
        if stat_prior is not None:
            return stat_prior, params.get("sigma_field", DEFAULT_SIGMA) * 1.05
        return params.get("default_skill", -0.5), \
               params.get("sigma_field", DEFAULT_SIGMA) * 1.1
    rating = pl["skill"] + fw * pl.get("form", 0.0)
    if stat_prior is not None:
        blend = float(params.get("public_stat_blend", PUBLIC_STAT_BLEND))
        rating = (1 - blend) * rating + blend * stat_prior
    if course:
        rating += params.get("courses", {}).get(course, {}).get(canon, 0.0)
    return rating, pl.get("sigma", params.get("sigma_field", DEFAULT_SIGMA))


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


def predict_field(field_names, params: dict, course: str = "",
                  is_major: bool = False) -> list[Player]:
    """Build rated Player objects for a field from fitted params.

    Accepts an iterable of names or Player objects. Ratings are centred on the
    field mean (= 0) so simulate.py reads them directly; σ keeps absolute scale.
    """
    maj_mult = params.get("major_sigma_mult", 1.0) if is_major else 1.0
    out: list[Player] = []
    for item in field_names:
        name = item.name if isinstance(item, Player) else str(item)
        rating, sigma = rating_for(name, params, course)
        canon = resolve_name(name, params)
        pl = params.get("players", {}).get(canon, {}) if canon else {}
        p = Player(name=name)
        p.rating = rating
        p.sigma = sigma * maj_mult
        p.sg_baseline = pl.get("skill", rating)
        p.recent_form = pl.get("form", 0.0)
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
            print(f"{i:<5}{name:<26}{rating:>+8.3f}{pl['skill']:>+8.3f}"
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

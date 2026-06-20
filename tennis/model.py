"""tennis/model.py — surface-split Bradley-Terry skill model.

Each player carries a base skill and per-surface offsets (clay/grass/carpet
relative to hard). Match probability is

    logit P(A beats B) = skill_A − skill_B
                       + offset_A[s] − offset_B[s]
                       + form_weight · (form_A − form_B)
                       + h2h_weight · h2h_log_odds(A, B, s)

Parameters are fitted by penalised (ridge) logistic regression on the binary
match-outcome design, with time-decay sample weights so recent matches count
more, and a rank-based ridge target so low-sample players regress toward the
prior implied by their ranking rather than the field mean. Fitted with scipy's
L-BFGS over a sparse design (no scikit-learn dependency).

ATP and WTA are fitted separately → atp_model_params.json / wta_model_params.json.
"""
from __future__ import annotations

import json
import math
import unicodedata
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent / "data"
MATCHES_CSV = DATA_DIR / "matches.csv"

SURFACES = ("clay", "grass", "carpet")   # hard is the baseline (offset ≡ 0)
SURFACE_IDX = {s: i for i, s in enumerate(SURFACES)}

# ── fit hyperparameters (tuned later by validate.py) ──
DEFAULT_CONFIG = {
    "skill_halflife_days": 365.0,   # ≈ 52-week half-life on the skill estimate
    "ridge_skill": 6.0,             # shrink skill toward its rank prior
    "ridge_offset": 24.0,           # shrink surface offsets toward 0 (small samples)
    "rank_prior_coef": -0.12,       # skill prior = coef · log(median_rank)
    "form_halflife_days": 56.0,     # 8-week recency window for the form nudge
    "form_window_days": 56,
    "form_k": 8.0,                  # Empirical-Bayes shrink for form (equiv. matches)
    "form_weight": 0.5,             # how much form nudges the rating
    "min_surface_matches": 12,      # keep a surface offset only above this count
    "h2h_weight": 0.05,             # heavily shrunk head-to-head nudge
}

MIN_FIT_MATCHES = 20                # hard floor below which a fit is meaningless
DEFAULT_RANK = 9999


def _params_path(tour: str) -> Path:
    return DATA_DIR / f"{tour.lower()}_model_params.json"


# ─────────────────────────────────────────────
# Name folding (accent/case-insensitive matching across sources)
# ─────────────────────────────────────────────

_TRANSLIT = str.maketrans({
    "ø": "o", "Ø": "o", "æ": "ae", "Æ": "ae", "ð": "d", "Ð": "d",
    "þ": "th", "Þ": "th", "ł": "l", "Ł": "l", "ß": "ss",
})


def fold_name(name: str) -> str:
    s = str(name).translate(_TRANSLIT)
    nfkd = unicodedata.normalize("NFKD", s)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    return " ".join(stripped.lower().split())


def _folded_index(params: dict) -> dict[str, str]:
    idx = params.get("_folded_index")
    if idx is None:
        idx = {fold_name(n): n for n in params.get("skills", {})}
        params["_folded_index"] = idx
    return idx


def resolve_name(name: str, params: dict) -> Optional[str]:
    """Canonical fitted name for `name`, tolerant of accents/case. None if unknown."""
    skills = params.get("skills", {})
    if name in skills:
        return name
    return _folded_index(params).get(fold_name(name))


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

def load_matches_df(path: Path | None = None):
    """Read matches.csv → DataFrame (raises if absent)."""
    import pandas as pd
    path = path or MATCHES_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"No {path}. Seed it first: python -m tennis.fetch --seed 2020 2021 2022 2023 2024 2025")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "winner", "loser"])
    df["winner"] = df["winner"].astype(str)
    df["loser"] = df["loser"].astype(str)
    df["surface"] = df["surface"].astype(str).str.lower()
    for col in ("winner_rank", "loser_rank"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(DEFAULT_RANK)
    return df


# ─────────────────────────────────────────────
# Fit
# ─────────────────────────────────────────────

def fit(matches_df, tour: str | None = None, asof=None,
        config: dict | None = None) -> dict:
    """Fit skill + surface offsets + form on matches before `asof` (walk-forward
    safe). Returns a params dict (see `save_params` for the JSON shape)."""
    import numpy as np
    import pandas as pd
    from scipy.sparse import csr_matrix
    from scipy.special import expit
    from scipy.optimize import minimize

    cfg = {**DEFAULT_CONFIG, **(config or {})}
    df = matches_df
    if tour:
        df = df[df["tour"].astype(str).str.lower() == tour.lower()]
    if asof is not None:
        asof = pd.Timestamp(asof)
        df = df[df["date"] < asof]
    df = df.reset_index(drop=True)
    if len(df) < MIN_FIT_MATCHES:
        raise ValueError(f"only {len(df)} matches{f' for {tour}' if tour else ''}"
                         f" before {asof} — need ≥{MIN_FIT_MATCHES}")
    asof_ts = asof if asof is not None else (df["date"].max() + pd.Timedelta(days=1))

    players = sorted(set(df["winner"]) | set(df["loser"]))
    pi = {p: i for i, p in enumerate(players)}
    P = len(players)
    n_cols = P * (1 + len(SURFACES))   # skill block + 3 surface-offset blocks

    # ── rank prior (cold start): skill_prior = coef · log(median_rank) ──
    rank_lists: dict[str, list[float]] = {p: [] for p in players}
    for w, r in zip(df["winner"], df["winner_rank"]):
        if r and r < DEFAULT_RANK:
            rank_lists[w].append(float(r))
    for l, r in zip(df["loser"], df["loser_rank"]):
        if r and r < DEFAULT_RANK:
            rank_lists[l].append(float(r))
    prior = np.zeros(n_cols)
    coef = float(cfg["rank_prior_coef"])
    for p, i in pi.items():
        rs = rank_lists[p]
        med = float(np.median(rs)) if rs else float(DEFAULT_RANK)
        prior[i] = coef * math.log(max(med, 1.0))

    # ── time-decay sample weights ──
    age = (asof_ts - df["date"]).dt.days.to_numpy().astype(float)
    decay = 0.5 ** (age / float(cfg["skill_halflife_days"]))

    # ── sparse design: row = +skill_w −skill_l (+off_w[s] −off_l[s] if s≠hard) ──
    widx = df["winner"].map(pi).to_numpy()
    lidx = df["loser"].map(pi).to_numpy()
    surf = df["surface"].map(SURFACE_IDX)         # NaN for hard / unknown
    m = len(df)

    rows_i: list[np.ndarray] = []
    cols_i: list[np.ndarray] = []
    vals_i: list[np.ndarray] = []
    rng = np.arange(m)
    rows_i += [rng, rng]
    cols_i += [widx, lidx]
    vals_i += [np.ones(m), -np.ones(m)]
    has_surf = surf.notna().to_numpy()
    if has_surf.any():
        sidx = surf.fillna(0).to_numpy().astype(int)
        off_w = P + sidx * P + widx
        off_l = P + sidx * P + lidx
        sel = np.where(has_surf)[0]
        rows_i += [sel, sel]
        cols_i += [off_w[sel], off_l[sel]]
        vals_i += [np.ones(len(sel)), -np.ones(len(sel))]

    X = csr_matrix((np.concatenate(vals_i),
                    (np.concatenate(rows_i), np.concatenate(cols_i))),
                   shape=(m, n_cols))

    lam = np.empty(n_cols)
    lam[:P] = float(cfg["ridge_skill"])
    lam[P:] = float(cfg["ridge_offset"])
    w = decay

    def objective(beta):
        z = X.dot(beta)
        # weighted logistic NLL with all labels = 1 (winner beats loser)
        nll = float(np.dot(w, np.logaddexp(0.0, -z)))
        diff = beta - prior
        reg = 0.5 * float(np.dot(lam, diff * diff))
        grad_z = w * (expit(z) - 1.0)
        grad = X.T.dot(grad_z) + lam * diff
        return nll + reg, grad

    res = minimize(objective, prior.copy(), jac=True, method="L-BFGS-B",
                   options={"maxiter": 500, "maxfun": 50000})
    beta = res.x
    skills = beta[:P]

    # ── surface offsets: keep only well-sampled (player, surface) pairs ──
    surf_counts = {s: np.zeros(P) for s in SURFACES}
    for s_name, k in SURFACE_IDX.items():
        mask = surf.to_numpy() == k
        if mask.any():
            surf_counts[s_name] = (np.bincount(widx[mask], minlength=P)
                                   + np.bincount(lidx[mask], minlength=P)).astype(float)
    min_surf = int(cfg["min_surface_matches"])
    surface_offsets: dict[str, dict[str, float]] = {}
    for p, i in pi.items():
        d = {}
        for s_name in SURFACES:
            k = SURFACE_IDX[s_name]
            if surf_counts[s_name][i] >= min_surf:
                v = float(beta[P + k * P + i])
                if abs(v) > 1e-4:
                    d[s_name] = round(v, 4)
        if d:
            surface_offsets[p] = d

    # ── recent-form nudge: EB-shrunk mean (actual − expected) over the window ──
    form = _fit_form(df, skills, surface_offsets, pi, asof_ts, cfg)

    default_skill = float(np.quantile(skills, 0.20)) if P else 0.0
    counts = (np.bincount(widx, minlength=P) + np.bincount(lidx, minlength=P))

    return {
        "tour": (tour or "all").lower(),
        "asof": str(pd.Timestamp(asof_ts).date()),
        "n_matches": int(m),
        "n_players": int(P),
        "default_skill": round(default_skill, 4),
        "form_weight": float(cfg["form_weight"]),
        "h2h_weight": float(cfg["h2h_weight"]),
        "hyperparams": {k: float(cfg[k]) for k in DEFAULT_CONFIG},
        "skills": {p: round(float(skills[i]), 4) for p, i in pi.items()},
        "surface_offsets": surface_offsets,
        "form": form,
        "n_played": {p: int(counts[i]) for p, i in pi.items()},
        "meta": {"converged": bool(res.success), "iterations": int(res.nit)},
    }


def _fit_form(df, skills, surface_offsets, pi, asof_ts, cfg) -> dict[str, float]:
    """Per-player momentum nudge: Empirical-Bayes-shrunk, decay-weighted mean of
    (actual − model-expected) over recent matches. Positive = over-performing."""
    import numpy as np
    import pandas as pd

    window = pd.Timedelta(days=int(cfg["form_window_days"]))
    recent = df[df["date"] >= (asof_ts - window)]
    if recent.empty:
        return {}

    P = len(pi)
    half = float(cfg["form_halflife_days"])
    num = np.zeros(P)   # Σ weight·residual
    wsum = np.zeros(P)  # Σ weight
    cnt = np.zeros(P)

    def base_logit(name, surf):
        i = pi[name]
        s = skills[i]
        off = surface_offsets.get(name, {}).get(surf, 0.0)
        return s + off

    for _, row in recent.iterrows():
        w_name, l_name, s = row["winner"], row["loser"], row["surface"]
        age = (asof_ts - row["date"]).days
        wt = 0.5 ** (age / half)
        lw = base_logit(w_name, s)
        ll = base_logit(l_name, s)
        p_w = 1.0 / (1.0 + math.exp(-(lw - ll)))
        # winner: actual 1, expected p_w; loser: actual 0, expected p_l = 1 − p_w
        iw, il = pi[w_name], pi[l_name]
        num[iw] += wt * (1.0 - p_w); wsum[iw] += wt; cnt[iw] += 1
        num[il] += wt * (0.0 - (1.0 - p_w)); wsum[il] += wt; cnt[il] += 1

    k = float(cfg["form_k"])
    out: dict[str, float] = {}
    inv = {i: p for p, i in pi.items()}
    for i in range(P):
        if wsum[i] <= 0:
            continue
        raw = num[i] / wsum[i]
        shrunk = raw * (cnt[i] / (cnt[i] + k))
        if abs(shrunk) > 1e-4:
            out[inv[i]] = round(float(shrunk), 4)
    return out


# ─────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────

def _player_logit(name: str, surface: str, params: dict) -> float:
    """skill + surface offset + form_weight·form for one player (0-centred at the
    field default for unknowns)."""
    canon = resolve_name(name, params)
    if canon is None:
        return params.get("default_skill", 0.0)
    skill = params["skills"].get(canon, params.get("default_skill", 0.0))
    off = params.get("surface_offsets", {}).get(canon, {}).get(surface, 0.0)
    fw = params.get("form_weight", 0.0)
    form = params.get("form", {}).get(canon, 0.0)
    return skill + off + fw * form


def predict_match(player_a: str, player_b: str, surface: str, params: dict,
                  h2h_log_odds: float = 0.0) -> dict:
    """P(A beats B) on `surface` from fitted params.

    `h2h_log_odds` is an optional, already-computed head-to-head signal (see
    `h2h_log_odds_from_df`); it is multiplied by the fitted `h2h_weight`. Returns
    {"p_a", "p_b", "logit"}.
    """
    surface = (surface or "hard").lower()
    la = _player_logit(player_a, surface, params)
    lb = _player_logit(player_b, surface, params)
    logit = (la - lb) + params.get("h2h_weight", 0.0) * float(h2h_log_odds)
    p_a = 1.0 / (1.0 + math.exp(-logit))
    return {"p_a": p_a, "p_b": 1.0 - p_a, "logit": logit}


def h2h_log_odds_from_df(player_a: str, player_b: str, surface: str, df,
                         prior: float = 1.0) -> float:
    """Laplace-smoothed log-odds of A's historical H2H vs B on `surface`.

    Symmetric small-sample prior `prior` keeps an unplayed or one-sided record
    from blowing up; the caller shrinks the whole term via `h2h_weight`.
    """
    sub = df[(df["surface"].astype(str).str.lower() == surface.lower())]
    a_wins = int(((sub["winner"] == player_a) & (sub["loser"] == player_b)).sum())
    b_wins = int(((sub["winner"] == player_b) & (sub["loser"] == player_a)).sum())
    return math.log((a_wins + prior) / (b_wins + prior))


# ─────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────

def save_params(params: dict, tour: str | None = None, path: Path | None = None) -> Path:
    path = path or _params_path(tour or params.get("tour", "all"))
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in params.items() if not k.startswith("_")}
    with open(path, "w") as f:
        json.dump(clean, f, indent=1)
    return path


def load_params(tour: str = "atp", path: Path | None = None) -> dict | None:
    path = path or _params_path(tour)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def _ranked(params: dict, top: int = 30) -> list[tuple[str, float]]:
    fw = params.get("form_weight", 0.0)
    forms = params.get("form", {})
    rows = [(n, s + fw * forms.get(n, 0.0)) for n, s in params["skills"].items()]
    return sorted(rows, key=lambda kv: -kv[1])[:top]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Fit / inspect tennis skill ratings")
    ap.add_argument("--fit", action="store_true", help="Fit and save model_params.json")
    ap.add_argument("--tour", default="atp", choices=["atp", "wta"])
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    if args.fit:
        params = fit(load_matches_df(), tour=args.tour)
        out = save_params(params, tour=args.tour)
        print(f"Fitted {params['n_players']} {args.tour.upper()} players from "
              f"{params['n_matches']:,} matches as of {params['asof']} → {out}")
        print(f"\n{'Rank':<5}{'Player':<28}{'Rating':>8}")
        print("-" * 42)
        for i, (name, rating) in enumerate(_ranked(params, args.top), 1):
            print(f"{i:<5}{name:<28}{rating:>+8.3f}")
        raise SystemExit(0)

    params = load_params(args.tour)
    if not params:
        print(f"No fitted params for {args.tour}. Run: python -m tennis.model --fit --tour {args.tour}")
        raise SystemExit(1)
    print(f"{args.tour.upper()} model · {params['n_players']} players · "
          f"{params['n_matches']:,} matches · asof {params['asof']}")
    for i, (name, rating) in enumerate(_ranked(params, args.top), 1):
        print(f"{i:<5}{name:<28}{rating:>+8.3f}")

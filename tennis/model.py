"""tennis/model.py — surface-split Bradley-Terry skill model.

Each player carries a base skill and per-surface offsets (clay/grass/carpet
relative to hard). Match probability is

    logit P(A beats B) = skill_A − skill_B
                       + offset_A[s] − offset_B[s]

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

# Deliberately small parameter set. The former probability-space "form"
# residual and tiny H2H term mixed incompatible units and did not improve the
# validation baseline, so they are not part of the estimator.
DEFAULT_CONFIG = {
    "skill_halflife_days": 365.0,   # ≈ 52-week half-life on the skill estimate
    "ridge_skill": 6.0,             # shrink skill toward its rank prior
    "ridge_offset": 24.0,           # shrink surface offsets toward 0 (small samples)
    "rank_prior_coef": -0.12,       # skill prior = coef · log(median_rank)
    "min_surface_matches": 12,      # keep a surface offset only above this count
}

MIN_FIT_MATCHES = 20                # hard floor below which a fit is meaningless
DEFAULT_RANK = 9999
MAX_DATA_AGE_DAYS = 21

# Empirical-Bayes prior strength (in service points) for per-player serve/return
# rates, and the minimum points before a player's own rate is trusted at all.
SERVE_SHRINK_POINTS = 600.0
SERVE_MIN_POINTS = 250.0
DEFAULT_SERVE_BY_TOUR = {
    "atp": 0.64,
    "wta": 0.60,
}
GAMES_CAL_SAMPLE = 600              # training matches sampled for the games-level cal


def _fit_games_cal(df, params, tour) -> float:
    """Walk-forward-safe multiplicative correction for expected total games.

    The Markov games model over-predicts the total by a stable ~8% (server-hold
    and set-count idealisations); without this the over/under market is unusable.
    We sample completed training matches, compute the model's expected total games
    (with the matchup serve base when available, else the fixed baseline), and
    return Σactual / Σpredicted, clipped to a sane band. 1.0 ⇒ no correction."""
    import numpy as np
    from . import simulate as S
    from .scores import score_total_games

    d = df.copy()
    d = d[d["score"].astype(str).str.strip() != ""]
    if len(d) == 0:
        return 1.0
    if len(d) > GAMES_CAL_SAMPLE:
        d = d.sample(n=GAMES_CAL_SAMPLE, random_state=0)
    # Expected total games varies smoothly in (match prob, serve base, best_of),
    # so memoise on rounded buckets — hundreds of matches collapse to a few cells.
    cache: dict[tuple, float] = {}

    def exp_games(p_a, base, best_of):
        key = (round(p_a, 2), round(base, 2), best_of)
        if key not in cache:
            cache[key] = S.match_markets(key[0], best_of=best_of, base=key[1])["exp_total_games"]
        return cache[key]

    num = den = 0.0
    for r in d.itertuples(index=False):
        tg = score_total_games(getattr(r, "score", ""))
        if tg is None:
            continue
        a, b = sorted([str(r.winner), str(r.loser)], key=fold_name)
        surface = str(r.surface).lower()
        best_of = int(r.best_of) if str(getattr(r, "best_of", "")).strip() not in ("", "nan") else 3
        base = serve_base(a, b, surface, params)
        p_a = predict_match(a, b, surface, params)["p_a"]
        pred = exp_games(
            p_a, base if base is not None else S.base_serve(tour), best_of,
        )
        if pred > 0:
            num += tg
            den += pred
    if den <= 0:
        return 1.0
    return float(np.clip(num / den, 0.80, 1.20))


def _fit_serve(df, asof_ts, halflife_days: float, tour: str | None) -> dict:
    """Per-player decay-weighted, EB-shrunk serve-points-won and return-points-won
    rates from the `*_sv_pts/_sv_won` columns. Returns {} when the slice carries
    no point stats (e.g. a WTA MatchCharting-only feed).

    serve-points-won: fraction of a player's own service points won.
    return-points-won: fraction of the opponent's service points the player won
    when returning (= 1 − opponent's serve-points-won that match).
    """
    import numpy as np
    import pandas as pd

    need = ("w_sv_pts", "w_sv_won", "l_sv_pts", "l_sv_won")
    if not all(c in df.columns for c in need):
        return {}
    d = df.copy()
    for c in need:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=need)
    d = d[(d["w_sv_pts"] > 0) & (d["l_sv_pts"] > 0)]
    if len(d) < 50:
        return {}

    age = (asof_ts - d["date"]).dt.days.to_numpy().astype(float)
    wt = 0.5 ** (age / float(halflife_days))

    # tour averages (decay-weighted), overall + per surface
    tot_sv = float(np.sum((d["w_sv_pts"] + d["l_sv_pts"]) * wt))
    tot_won = float(np.sum((d["w_sv_won"] + d["l_sv_won"]) * wt))
    fallback = DEFAULT_SERVE_BY_TOUR.get(str(tour or "").lower(), 0.62)
    avg_spw = tot_won / tot_sv if tot_sv > 0 else fallback
    by_surface = {}
    for surf, g in d.assign(_w=wt).groupby("surface"):
        sv = float(np.sum((g["w_sv_pts"] + g["l_sv_pts"]) * g["_w"]))
        wn = float(np.sum((g["w_sv_won"] + g["l_sv_won"]) * g["_w"]))
        if sv > 0:
            by_surface[str(surf).lower()] = round(wn / sv, 4)

    # per-player accumulators: serve pts/won, return pts/won (return = opp serve)
    acc: dict[str, list[float]] = {}   # [sv_pts, sv_won, ret_pts, ret_won]

    def add(name, sv_pts, sv_won, ret_pts, ret_won, w):
        a = acc.setdefault(name, [0.0, 0.0, 0.0, 0.0])
        a[0] += sv_pts * w; a[1] += sv_won * w
        a[2] += ret_pts * w; a[3] += ret_won * w

    for r, w in zip(d.itertuples(index=False), wt):
        # winner serves w_sv_pts; returns against l_sv_pts (wins l_sv_pts - l_sv_won)
        add(r.winner, r.w_sv_pts, r.w_sv_won, r.l_sv_pts, r.l_sv_pts - r.l_sv_won, w)
        add(r.loser, r.l_sv_pts, r.l_sv_won, r.w_sv_pts, r.w_sv_pts - r.w_sv_won, w)

    avg_rpw = 1.0 - avg_spw
    K = SERVE_SHRINK_POINTS
    serve = {}
    for name, (svp, svw, rtp, rtw) in acc.items():
        if svp <= 0 or rtp <= 0:
            continue
        spw = (svw + K * avg_spw) / (svp + K)        # EB-shrunk to tour mean
        rpw = (rtw + K * avg_rpw) / (rtp + K)
        serve[name] = {"spw": round(float(spw), 4), "rpw": round(float(rpw), 4),
                       "sv_pts": int(round(svp))}
    return {"players": serve,
            "avg_spw": round(float(avg_spw), 4),
            "by_surface": by_surface}


def _params_path(tour: str) -> Path:
    return DATA_DIR / f"{tour.lower()}_model_params.json"


def assert_params_fresh(params: dict, asof=None) -> None:
    """Refuse to serve a fitted model whose own as-of date is stale."""
    import pandas as pd

    if not params.get("asof"):
        raise ValueError("Saved tennis model has no as-of date; refit before pricing.")
    try:
        model_asof = pd.Timestamp(params["asof"])
    except (TypeError, ValueError):
        raise ValueError(
            "Saved tennis model has an invalid as-of date; refit before pricing."
        ) from None
    reference = pd.Timestamp(asof) if asof is not None else pd.Timestamp.now().normalize()
    age = (reference - model_asof).days
    if age > MAX_DATA_AGE_DAYS:
        tour = str(params.get("tour") or "tennis").upper()
        raise ValueError(
            f"{tour} model is stale: fitted through {model_asof.date()} "
            f"({age} days old; maximum {MAX_DATA_AGE_DAYS}). Refresh results "
            "and refit before pricing."
        )


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
    """Read clean completed matches for modelling (raises if absent).

    Retirements and walkovers remain in the source CSV for settlement/audit but
    are excluded here so they are not trained as ordinary decisive wins.
    """
    import pandas as pd
    from .scores import is_retirement_or_walkover
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
    if "score" in df.columns:
        incomplete = df["score"].map(is_retirement_or_walkover)
        df = df[~incomplete].copy()
    for col in ("winner_rank", "loser_rank"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(DEFAULT_RANK)
    return df


# ─────────────────────────────────────────────
# Fit
# ─────────────────────────────────────────────

def fit(matches_df, tour: str | None = None, asof=None,
        config: dict | None = None, with_games_cal: bool = True) -> dict:
    """Fit skill + surface offsets + form on matches before `asof` (walk-forward
    safe). Returns a params dict (see `save_params` for the JSON shape).

    `with_games_cal` runs the (relatively expensive) total-games level calibration.
    Walk-forward validation refits — which only score the match-winner / set
    markets, all independent of the games calibration — pass False to stay fast;
    it defaults True for the production fit so the over/under stays calibrated."""
    import numpy as np
    import pandas as pd
    from scipy.sparse import csr_matrix
    from scipy.special import expit
    from scipy.optimize import minimize

    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if str(tour or "").lower() == "wta":
        cfg["rank_prior_coef"] = 0.0
    df = matches_df
    if tour:
        df = df[df["tour"].astype(str).str.lower() == tour.lower()]
    requested_asof = pd.Timestamp(asof) if asof is not None else pd.Timestamp.now().normalize()
    if asof is not None:
        df = df[df["date"] < requested_asof]
    df = df.reset_index(drop=True)
    if len(df) < MIN_FIT_MATCHES:
        raise ValueError(f"only {len(df)} matches{f' for {tour}' if tour else ''}"
                         f" before {asof} — need ≥{MIN_FIT_MATCHES}")
    latest = pd.Timestamp(df["date"].max())
    data_age = (requested_asof - latest).days
    if data_age > MAX_DATA_AGE_DAYS:
        label = f"{tour.upper()} " if tour else ""
        raise ValueError(
            f"{label}results are stale: newest match {latest.date()} is "
            f"{data_age} days before fit as-of {requested_asof.date()} "
            f"(maximum {MAX_DATA_AGE_DAYS})"
        )
    asof_ts = requested_asof

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
    # The current WTA feed has no ranking columns. Explicitly use a zero-centred
    # prior instead of pretending 9999 is a meaningful rank for every player.
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

    default_skill = float(np.quantile(skills, 0.20)) if P else 0.0
    counts = (np.bincount(widx, minlength=P) + np.bincount(lidx, minlength=P))

    serve = _fit_serve(df, asof_ts, float(cfg["skill_halflife_days"]), tour)

    out = {
        "tour": (tour or "all").lower(),
        "asof": str(pd.Timestamp(asof_ts).date()),
        "n_matches": int(m),
        "n_players": int(P),
        "default_skill": round(default_skill, 4),
        "hyperparams": {k: float(cfg[k]) for k in DEFAULT_CONFIG},
        "skills": {p: round(float(skills[i]), 4) for p, i in pi.items()},
        "surface_offsets": surface_offsets,
        "serve": serve,
        "n_played": {p: int(counts[i]) for p, i in pi.items()},
        "meta": {"converged": bool(res.success), "iterations": int(res.nit)},
    }
    # games-level calibration uses the assembled params (needs serve/skills set)
    out["games_cal"] = round(_fit_games_cal(df, out, tour), 4) if with_games_cal else 1.0
    return out


# ─────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────

def _player_logit(name: str, surface: str, params: dict) -> float:
    """Skill plus surface offset for one player."""
    canon = resolve_name(name, params)
    if canon is None:
        return params.get("default_skill", 0.0)
    skill = params["skills"].get(canon, params.get("default_skill", 0.0))
    off = params.get("surface_offsets", {}).get(canon, {}).get(surface, 0.0)
    return skill + off


def serve_base(player_a: str, player_b: str, surface: str, params: dict) -> float | None:
    """Matchup-specific average serve-point-win level for the two players on
    `surface`, from the fitted serve/return rates. Drives the *total-games* regime
    (two big servers hold more → more games) while the headline match prob stays
    pinned to the Bradley-Terry estimate. Returns None when serve stats are
    unavailable for either player (caller falls back to the symmetric baseline).

    Each player's serve points won vs this specific opponent is the standard
    opponent-adjusted combination
        p(i serving) = spw_i − (rpw_j − avg_rpw)
    and the base is the average of the two servers' values.
    """
    sv = params.get("serve") or {}
    players = sv.get("players") or {}
    ca = resolve_name(player_a, params) or player_a
    cb = resolve_name(player_b, params) or player_b
    ra, rb = players.get(ca), players.get(cb)
    if not ra or not rb:
        return None
    if ra.get("sv_pts", 0) < SERVE_MIN_POINTS or rb.get("sv_pts", 0) < SERVE_MIN_POINTS:
        return None
    avg_spw = float((sv.get("by_surface") or {}).get(
        (surface or "hard").lower(),
        sv.get("avg_spw", DEFAULT_SERVE_BY_TOUR.get(params.get("tour"), 0.62)),
    ))
    avg_rpw = 1.0 - avg_spw
    ps_a = ra["spw"] - (rb["rpw"] - avg_rpw)     # A serving vs B returning
    ps_b = rb["spw"] - (ra["rpw"] - avg_rpw)     # B serving vs A returning
    base = 0.5 * (ps_a + ps_b)
    return float(min(max(base, 0.55), 0.74))


def predict_match(player_a: str, player_b: str, surface: str, params: dict) -> dict:
    """P(A beats B) on `surface` from fitted params.

    The return value includes unresolved input names so consumers can refuse to
    stake a typo rather than presenting a confident-looking 50/50.
    """
    surface = (surface or "hard").lower()
    la = _player_logit(player_a, surface, params)
    lb = _player_logit(player_b, surface, params)
    logit = la - lb
    p_a = 1.0 / (1.0 + math.exp(-logit))
    unresolved = [
        name for name in (player_a, player_b) if resolve_name(name, params) is None
    ]
    return {
        "p_a": p_a, "p_b": 1.0 - p_a, "logit": logit,
        "unresolved": unresolved,
    }


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
    return sorted(params["skills"].items(), key=lambda kv: -kv[1])[:top]


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

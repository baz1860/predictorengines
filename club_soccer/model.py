#!/usr/bin/env python3
"""Club soccer match model.

Models: goals (attack/defence Poisson), elo, and ensemble (default blend).
The engine is intentionally data-file first: API fetchers update fixtures.csv,
while the model can always run from local CSV fallbacks.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from competitions import strength

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIXTURES = DATA / "fixtures.csv"
PARAMS = DATA / "model_params.json"
ENSEMBLE_WEIGHTS = DATA / "ensemble_weights.json"
MAX_GOALS = 10
BASE_ELO = 1500.0
HOME_ADV_ELO = 55.0
HALF_LIFE_DAYS = 365.0
DC_RHO = -0.08
RECENT_K = 6        # matches in the shots-on-target recency window
# ensemble blend (chosen by held-out walk-forward search, June 2026):
# goals (actual-goal attack/def), elo, xg (long-run SoT expected goals),
# xgf (xg + recent SoT form). The model-signal sprint retuned this toward Elo:
# time-split checks showed less overfitting than the heavier SoT/form blend.
DEFAULT_ENSEMBLE_W = {"goals": 0.20, "elo": 0.40, "xg": 0.20, "xgf": 0.20, "xpress": 0.0}
ENSEMBLE_W = DEFAULT_ENSEMBLE_W
ENSEMBLE_COMPONENTS = tuple(DEFAULT_ENSEMBLE_W)


def _normalise_weights(weights: dict) -> dict[str, float]:
    vals = {k: max(0.0, float(weights.get(k, 0.0))) for k in ENSEMBLE_COMPONENTS}
    s = sum(vals.values())
    if s <= 0:
        return dict(DEFAULT_ENSEMBLE_W)
    return {k: v / s for k, v in vals.items()}


def load_ensemble_weights(path: Path = ENSEMBLE_WEIGHTS) -> dict[str, float]:
    """Champion ensemble weights, falling back to the validated hardcoded blend."""
    if path.exists():
        try:
            raw = json.loads(path.read_text())
            weights = raw.get("weights", raw)
            if isinstance(weights, dict):
                return _normalise_weights(weights)
        except Exception:
            pass
    return dict(DEFAULT_ENSEMBLE_W)


def load_fixtures(path: Path = FIXTURES) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run club_soccer/fetch.py or add fixtures.csv.")
    df = pd.read_csv(path, parse_dates=["date"])
    for c in ("home_goals", "away_goals", "home_shots", "away_shots",
              "home_sot", "away_sot", "home_corners", "away_corners"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["neutral"] = pd.to_numeric(df.get("neutral", 0), errors="coerce").fillna(0).astype(int)
    return df


def played(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["home_goals", "away_goals"]).copy()


def upcoming(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["home_goals"].isna() | df["away_goals"].isna()].copy()


def team_names(df: pd.DataFrame | None = None) -> list[str]:
    df = load_fixtures() if df is None else df
    return sorted(set(df["home"].dropna()) | set(df["away"].dropna()))


def _weights(dates: pd.Series) -> np.ndarray:
    anchor = dates.max()
    age = (anchor - dates).dt.days.to_numpy(dtype=float)
    return np.exp(-math.log(2) * age / HALF_LIFE_DAYS)


def _poisson_pmf(lam: float) -> np.ndarray:
    g = np.arange(MAX_GOALS + 1)
    return np.exp(-lam) * np.power(lam, g) / np.array([math.factorial(int(i)) for i in g])


def score_matrix(lam_h: float, lam_a: float, rho: float = DC_RHO) -> np.ndarray:
    lam_h, lam_a = max(0.05, float(lam_h)), max(0.05, float(lam_a))
    M = np.outer(_poisson_pmf(lam_h), _poisson_pmf(lam_a))
    M[0, 0] *= 1 - lam_h * lam_a * rho
    M[1, 0] *= 1 + lam_a * rho
    M[0, 1] *= 1 + lam_h * rho
    M[1, 1] *= 1 - rho
    return M / M.sum()


def probs_from_matrix(M: np.ndarray) -> dict[str, float]:
    total = np.add.outer(np.arange(M.shape[0]), np.arange(M.shape[1]))
    return {
        "home": float(np.tril(M, -1).sum()),
        "draw": float(np.trace(M)),
        "away": float(np.triu(M, 1).sum()),
        "over25": float(M[total > 2].sum()),
        "under25": float(M[total <= 2].sum()),
        "btts_yes": float(M[1:, 1:].sum()),
        "btts_no": float(1.0 - M[1:, 1:].sum()),
    }


def top_scorelines(M: np.ndarray, n: int = 5) -> list[dict]:
    flat = sorted(((i, j, float(M[i, j])) for i in range(M.shape[0])
                   for j in range(M.shape[1])), key=lambda x: -x[2])
    return [{"score": f"{i}-{j}", "prob": round(p, 4)} for i, j, p in flat[:n]]


def fit(df: pd.DataFrame | None = None) -> dict:
    df = played(load_fixtures() if df is None else df).sort_values("date")
    if df.empty:
        raise ValueError("No played fixtures available to fit the model.")
    teams = sorted(set(df["home"]) | set(df["away"]))
    w = _weights(df["date"])
    avg_home = float(np.average(df["home_goals"], weights=w))
    avg_away = float(np.average(df["away_goals"], weights=w))
    global_avg = max(0.8, (avg_home + avg_away) / 2)

    rows = []
    for r, wt in zip(df.itertuples(index=False), w):
        comp_str = strength(r.competition)
        rows.append((r.home, "for", r.home_goals, wt, comp_str))
        rows.append((r.home, "against", r.away_goals, wt, comp_str))
        rows.append((r.away, "for", r.away_goals, wt, comp_str))
        rows.append((r.away, "against", r.home_goals, wt, comp_str))
    stats = {t: {"gf": 0.0, "ga": 0.0, "wf": 0.0, "wa": 0.0,
                 "xf": 0.0, "xa": 0.0, "wx": 0.0,
                 "xpf": 0.0, "xpa": 0.0, "wxp": 0.0}
             for t in teams}
    for t, typ, goals, wt, comp_str in rows:
        key, wk = ("gf", "wf") if typ == "for" else ("ga", "wa")
        stats[t][key] += float(goals) * wt
        stats[t][wk] += wt

    # SoT-based expected goals: conv = league goals per shot-on-target; a team's
    # xG-for/against is its SoT scaled by conv. Cleaner strength signal than goals.
    has_sot = "home_sot" in df.columns and "away_sot" in df.columns
    if has_sot:
        sot_sum = float(np.average(df["home_sot"].fillna(0) + df["away_sot"].fillna(0), weights=w))
        goal_sum = float(np.average(df["home_goals"] + df["away_goals"], weights=w))
        conv = goal_sum / sot_sum if sot_sum > 0 else 0.0
    else:
        conv = 0.0
    if conv > 0:
        for r, wt in zip(df.itertuples(index=False), w):
            hs, as_ = getattr(r, "home_sot", np.nan), getattr(r, "away_sot", np.nan)
            if pd.isna(hs) or pd.isna(as_):
                continue
            stats[r.home]["xf"] += float(hs) * conv * wt
            stats[r.home]["xa"] += float(as_) * conv * wt
            stats[r.away]["xf"] += float(as_) * conv * wt
            stats[r.away]["xa"] += float(hs) * conv * wt
            stats[r.home]["wx"] += wt
            stats[r.away]["wx"] += wt

    # Shot-pressure xG: local free-data challenger using SoT, non-SoT shots and
    # corners. Coefficients are fit on the training slice only, clipped to
    # plausible non-negative ranges so corner/volume noise cannot dominate.
    xp_coef = {"sot": conv if conv > 0 else 0.30, "non_sot": 0.02, "corner": 0.015}
    has_pressure = all(c in df.columns for c in (
        "home_shots", "away_shots", "home_sot", "away_sot",
        "home_corners", "away_corners"))
    if has_pressure:
        X_rows, y_rows, w_rows = [], [], []
        for r, wt in zip(df.itertuples(index=False), w):
            vals = [getattr(r, c, np.nan) for c in (
                "home_shots", "away_shots", "home_sot", "away_sot",
                "home_corners", "away_corners")]
            if any(pd.isna(v) for v in vals):
                continue
            hn = max(float(r.home_shots) - float(r.home_sot), 0.0)
            an = max(float(r.away_shots) - float(r.away_sot), 0.0)
            X_rows.append([float(r.home_sot), hn, float(r.home_corners)])
            y_rows.append(float(r.home_goals)); w_rows.append(float(wt))
            X_rows.append([float(r.away_sot), an, float(r.away_corners)])
            y_rows.append(float(r.away_goals)); w_rows.append(float(wt))
        if len(X_rows) >= 200:
            X = np.asarray(X_rows, dtype=float)
            yv = np.asarray(y_rows, dtype=float)
            sw = np.sqrt(np.asarray(w_rows, dtype=float))
            try:
                coef, *_ = np.linalg.lstsq(X * sw[:, None], yv * sw, rcond=None)
                xp_coef = {
                    "sot": float(np.clip(coef[0], 0.12, 0.60)),
                    "non_sot": float(np.clip(coef[1], 0.0, 0.08)),
                    "corner": float(np.clip(coef[2], 0.0, 0.08)),
                }
            except np.linalg.LinAlgError:
                pass
        for r, wt in zip(df.itertuples(index=False), w):
            vals = [getattr(r, c, np.nan) for c in (
                "home_shots", "away_shots", "home_sot", "away_sot",
                "home_corners", "away_corners")]
            if any(pd.isna(v) for v in vals):
                continue
            hx = (xp_coef["sot"] * float(r.home_sot)
                  + xp_coef["non_sot"] * max(float(r.home_shots) - float(r.home_sot), 0.0)
                  + xp_coef["corner"] * float(r.home_corners))
            ax = (xp_coef["sot"] * float(r.away_sot)
                  + xp_coef["non_sot"] * max(float(r.away_shots) - float(r.away_sot), 0.0)
                  + xp_coef["corner"] * float(r.away_corners))
            stats[r.home]["xpf"] += hx * wt
            stats[r.home]["xpa"] += ax * wt
            stats[r.away]["xpf"] += ax * wt
            stats[r.away]["xpa"] += hx * wt
            stats[r.home]["wxp"] += wt
            stats[r.away]["wxp"] += wt

    attack, defence, attack_xg, defence_xg = {}, {}, {}, {}
    attack_xpress, defence_xpress = {}, {}
    base_xf, base_xa = {}, {}
    for t in teams:
        gf = (stats[t]["gf"] + global_avg * 4) / (stats[t]["wf"] + 4)
        ga = (stats[t]["ga"] + global_avg * 4) / (stats[t]["wa"] + 4)
        attack[t] = float(math.log(max(0.25, gf) / global_avg))
        defence[t] = float(math.log(max(0.25, ga) / global_avg))
        xf = (stats[t]["xf"] + global_avg * 4) / (stats[t]["wx"] + 4)
        xa = (stats[t]["xa"] + global_avg * 4) / (stats[t]["wx"] + 4)
        attack_xg[t] = float(math.log(max(0.25, xf) / global_avg))
        defence_xg[t] = float(math.log(max(0.25, xa) / global_avg))
        xpf = (stats[t]["xpf"] + global_avg * 4) / (stats[t]["wxp"] + 4)
        xpa = (stats[t]["xpa"] + global_avg * 4) / (stats[t]["wxp"] + 4)
        attack_xpress[t] = float(math.log(max(0.25, xpf) / global_avg))
        defence_xpress[t] = float(math.log(max(0.25, xpa) / global_avg))
        base_xf[t], base_xa[t] = xf, xa

    # recency form: last RECENT_K matches' SoT-xG vs the team's season baseline,
    # as a log-ratio attack/defence nudge (the part long-run rates miss).
    recent = {t: [] for t in teams}
    if conv > 0:
        for r in df.sort_values("date").itertuples(index=False):
            hs, as_ = getattr(r, "home_sot", np.nan), getattr(r, "away_sot", np.nan)
            if pd.isna(hs) or pd.isna(as_):
                continue
            recent[r.home].append((float(hs) * conv, float(as_) * conv))
            recent[r.away].append((float(as_) * conv, float(hs) * conv))
    fatk, fdef = {}, {}
    for t in teams:
        last = recent[t][-RECENT_K:]
        if len(last) < 3:
            fatk[t] = fdef[t] = 0.0
            continue
        rf = float(np.mean([x[0] for x in last])); ra = float(np.mean([x[1] for x in last]))
        fatk[t] = float(np.clip(math.log(max(0.25, rf) / max(0.25, base_xf[t])), -0.4, 0.4))
        fdef[t] = float(np.clip(math.log(max(0.25, ra) / max(0.25, base_xa[t])), -0.4, 0.4))

    elo = {t: BASE_ELO for t in teams}
    for r in df.itertuples(index=False):
        h, a = r.home, r.away
        adv = 0.0 if int(r.neutral) else HOME_ADV_ELO
        exp_h = 1.0 / (1.0 + 10 ** ((elo[a] - (elo[h] + adv)) / 400.0))
        actual_h = 1.0 if r.home_goals > r.away_goals else (0.5 if r.home_goals == r.away_goals else 0.0)
        margin = abs(float(r.home_goals) - float(r.away_goals))
        comp_k = 18 + 20 * strength(r.competition)
        k = comp_k * (1.0 if margin <= 1 else min(1.75, 1 + margin / 4))
        delta = k * (actual_h - exp_h)
        elo[h] += delta
        elo[a] -= delta

    params = {"teams": teams, "global_avg": global_avg,
              "home_goal_adv": float(max(0.02, avg_home - avg_away)),
              "attack": attack, "defence": defence,
              "attack_xg": attack_xg, "defence_xg": defence_xg,
              "attack_xpress": attack_xpress, "defence_xpress": defence_xpress,
              "fatk": fatk, "fdef": fdef, "conv": float(conv),
              "xpress_coef": xp_coef,
              "elo": {k: float(v) for k, v in elo.items()},
              "fitted_matches": int(len(df))}
    return params


def save_params(params: dict, path: Path = PARAMS) -> None:
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(params, indent=2))


def load_params() -> dict:
    if PARAMS.exists():
        return json.loads(PARAMS.read_text())
    params = fit()
    save_params(params)
    return params


def _lambdas_goals(params: dict, home: str, away: str, competition: str | None, neutral: bool) -> tuple[float, float]:
    base = float(params["global_avg"])
    home_adv = 0.0 if neutral else float(params.get("home_goal_adv", 0.25)) / 2
    comp_adj = (strength(competition) - 0.75) * 0.12
    ah = params["attack"].get(home, 0.0); da = params["defence"].get(away, 0.0)
    aa = params["attack"].get(away, 0.0); dh = params["defence"].get(home, 0.0)
    # defence[t] = log(goals_conceded / global_avg): POSITIVE means a team concedes
    # more (weaker D), so the opponent's expected goals must ADD it. (Was `- da`,
    # which inverted it — cancels in aggregate but mis-ranks individual matches;
    # fixing it improved walk-forward Brier 0.6317 -> 0.6175.)
    return (base * math.exp(ah + da + home_adv + comp_adj),
            base * math.exp(aa + dh - home_adv + comp_adj))


def _lambdas_elo(params: dict, home: str, away: str, neutral: bool) -> tuple[float, float]:
    eh = params["elo"].get(home, BASE_ELO)
    ea = params["elo"].get(away, BASE_ELO)
    diff = (eh + (0 if neutral else HOME_ADV_ELO) - ea) / 400.0
    total = 2.55 + 0.20 * abs(diff)
    share = 1.0 / (1.0 + math.exp(-1.2 * diff))
    return max(0.15, total * share), max(0.15, total * (1 - share))


def _lambdas_xg(params: dict, home: str, away: str, competition: str | None,
                neutral: bool, form: bool = False) -> tuple[float, float]:
    """SoT-based expected-goals lambdas. With form=True, add the recent-SoT
    attack/defence nudge. Falls back to the goals attack/defence maps if a
    cached params dict predates the xg fields."""
    base = float(params["global_avg"])
    home_adv = 0.0 if neutral else float(params.get("home_goal_adv", 0.25)) / 2
    comp_adj = (strength(competition) - 0.75) * 0.12
    ax = params.get("attack_xg", params["attack"])
    dx = params.get("defence_xg", params["defence"])
    ah = ax.get(home, 0.0); da = dx.get(away, 0.0)
    aa = ax.get(away, 0.0); dh = dx.get(home, 0.0)
    if form:
        fa, fd = params.get("fatk", {}), params.get("fdef", {})
        ah += fa.get(home, 0.0); da += fd.get(away, 0.0)
        aa += fa.get(away, 0.0); dh += fd.get(home, 0.0)
    return (base * math.exp(ah + da + home_adv + comp_adj),
            base * math.exp(aa + dh - home_adv + comp_adj))


def _lambdas_xpress(params: dict, home: str, away: str, competition: str | None,
                    neutral: bool) -> tuple[float, float]:
    """Shot-pressure lambdas from SoT, non-SoT shots and corners.

    Falls back to the existing SoT-xG maps for cached params that predate the
    shot-pressure fields.
    """
    base = float(params["global_avg"])
    home_adv = 0.0 if neutral else float(params.get("home_goal_adv", 0.25)) / 2
    comp_adj = (strength(competition) - 0.75) * 0.12
    ax = params.get("attack_xpress", params.get("attack_xg", params["attack"]))
    dx = params.get("defence_xpress", params.get("defence_xg", params["defence"]))
    ah = ax.get(home, 0.0); da = dx.get(away, 0.0)
    aa = ax.get(away, 0.0); dh = dx.get(home, 0.0)
    return (base * math.exp(ah + da + home_adv + comp_adj),
            base * math.exp(aa + dh - home_adv + comp_adj))


def component_matrices(params: dict, home: str, away: str,
                       competition: str | None, neutral: bool) -> dict[str, np.ndarray]:
    return {
        "goals": score_matrix(*_lambdas_goals(params, home, away, competition, neutral)),
        "elo": score_matrix(*_lambdas_elo(params, home, away, neutral)),
        "xg": score_matrix(*_lambdas_xg(params, home, away, competition, neutral)),
        "xgf": score_matrix(*_lambdas_xg(params, home, away, competition, neutral, form=True)),
        "xpress": score_matrix(*_lambdas_xpress(params, home, away, competition, neutral)),
    }


def predict(home: str, away: str, competition: str | None = None,
            model: str = "ensemble", neutral: bool = False,
            params: dict | None = None) -> dict:
    params = load_params() if params is None else params
    teams = set(params["teams"])
    if home not in teams:
        raise ValueError(f"Unknown team: {home!r}")
    if away not in teams:
        raise ValueError(f"Unknown team: {away!r}")
    if home == away:
        raise ValueError("Pick two different teams.")
    if model == "ensemble":
        parts = component_matrices(params, home, away, competition, neutral)
        weights = load_ensemble_weights()
        M = sum(weights[k] * parts[k] for k in ENSEMBLE_COMPONENTS)
        M = M / M.sum()
    elif model == "goals":
        M = score_matrix(*_lambdas_goals(params, home, away, competition, neutral))
    elif model == "elo":
        M = score_matrix(*_lambdas_elo(params, home, away, neutral))
    elif model == "xg":
        M = score_matrix(*_lambdas_xg(params, home, away, competition, neutral, form=True))
    elif model == "xpress":
        M = score_matrix(*_lambdas_xpress(params, home, away, competition, neutral))
    else:
        raise ValueError("Unknown model: use ensemble, goals, elo, xg, or xpress.")
    probs = probs_from_matrix(M)
    xg_h = float(sum(i * M[i, :].sum() for i in range(M.shape[0])))
    xg_a = float(sum(j * M[:, j].sum() for j in range(M.shape[1])))
    return {"home": home, "away": away, "competition": competition or "",
            "model": model, "xg_home": round(xg_h, 2), "xg_away": round(xg_a, 2),
            "probs": {k: round(v, 4) for k, v in probs.items()},
            "scorelines": top_scorelines(M), "matrix": M}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("home", nargs="?")
    ap.add_argument("away", nargs="?")
    ap.add_argument("--competition", default="")
    ap.add_argument("--model", choices=["ensemble", "goals", "elo", "xg", "xpress"], default="ensemble")
    ap.add_argument("--neutral", action="store_true")
    ap.add_argument("--fit", action="store_true")
    args = ap.parse_args()
    if args.fit:
        params = fit()
        save_params(params)
        print(f"Saved {len(params['teams'])} teams from {params['fitted_matches']} matches -> {PARAMS}")
        return
    if not args.home or not args.away:
        ap.print_help()
        return
    out = predict(args.home, args.away, args.competition, args.model, args.neutral)
    print(f"{args.home} vs {args.away} ({args.competition or 'club soccer'})")
    print(f"Expected goals: {out['xg_home']:.2f} - {out['xg_away']:.2f}")
    p = out["probs"]
    print(f"Home {p['home']:.1%}  Draw {p['draw']:.1%}  Away {p['away']:.1%}")
    print(f"Over 2.5 {p['over25']:.1%}  BTTS {p['btts_yes']:.1%}")
    for s in out["scorelines"]:
        print(f"  {s['score']} {s['prob']:.1%}")


if __name__ == "__main__":
    main()

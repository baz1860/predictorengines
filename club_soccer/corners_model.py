"""Club-soccer CORNERS model — a soft-market companion to the goals model.

Mirrors `club_soccer/model.py` conventions: per-team attack/defence as log-ratios of
recency-weighted corner rates, a per-competition baseline (leagues differ in corner
volume), and a home adjustment. Corners are modelled as independent Poisson counts, so
the same machinery yields total-corner and team-corner Over/Under prices.

Why corners (not match odds): the WC 1X2 market is too efficient to beat (see
docs/model_improvements_changelog.md). Corners are a softer, less-watched market — but
they carry fatter margins and lower limits, so an edge must be real AND beat the corner
*closing* line. This module is validated on calibration, not ROI.

Data: club_soccer/data/fixtures.csv already has home_corners/away_corners
(16,775 matches, top European leagues, 2022-2026).

VALIDATION VERDICT (walk-forward, 2022-26): a pure team-strength corners model does
NOT beat the league-average baseline (MAE ~2.8 vs 2.71; O/U-9.5 log-loss ~0.71 vs
0.69) and is overconfident. Corners are driven by in-game state, not stable team
traits. Heavy shrinkage only gets it back to ~baseline. Conclusion: do not bet this
as-is.

SHOTS LEVER — TESTED, DOESN'T RESCUE IT (run --validate-shots):
  Added a Poisson GLM with team shot-strength alongside corner-strength. The fitted
  shot coefficient is ~0.08 (corner coef ~0.99): shots are collinear with corner rate
  and add almost nothing. The shots model STILL loses to the league baseline
  (O/U-9.5 log-loss ~0.706 vs 0.693; MAE 2.77 vs 2.71). Conclusion: pre-match team
  features — corners OR shots — do not beat quoting the competition average. Corners
  are governed by in-game state and noise, not stable pre-match team traits.

  What's left (not pre-match team strength): in-play/live models (game state), or
  market-specific inefficiencies (line shopping, specific situations). A pre-match
  strength model is the wrong tool for this market. Validate any future edge by CLV vs
  the CORNER closing line, never by ROI.

CLI:
  python -m club_soccer.corners_model --validate
  python -m club_soccer.corners_model --predict "Arsenal" "Chelsea" --comp "Premier League"
  python -m club_soccer.corners_model --teams --comp "Serie A"
"""
from __future__ import annotations
import argparse, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "club_soccer" / "data" / "fixtures.csv"

HALF_LIFE_DAYS = 540.0   # corners are style-driven & fairly stable; ~1.5y half-life
SHRINK_GAMES = 15.0      # Bayesian shrink team rates toward league mean (pseudo-games).
                         # Set firm: validation shows the team-corner signal is weak and
                         # easily overfits, so heavy shrinkage keeps it honest/calibrated.
MAX_C = 25               # corner-count grid


# ── data ──────────────────────────────────────────────────────────────────
def load_corners(path: Path = FIX) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ("home_corners", "away_corners"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["home_corners", "away_corners", "date", "home", "away"])
    return df.sort_values("date").reset_index(drop=True)


def _weights(dates: pd.Series, asof: pd.Timestamp, half_life: float) -> np.ndarray:
    age = (asof - dates).dt.days.clip(lower=0).to_numpy()
    return np.power(0.5, age / half_life)


# ── fit ───────────────────────────────────────────────────────────────────
def fit(df: pd.DataFrame, asof: pd.Timestamp | None = None,
        half_life: float = HALF_LIFE_DAYS) -> dict:
    if asof is None:
        asof = df["date"].max() + pd.Timedelta(days=1)
    df = df[df["date"] < asof].copy()
    w = _weights(df["date"], asof, half_life)
    df["w"] = w

    gw = df["w"].sum()
    global_avg = float((df["w"] * (df["home_corners"] + df["away_corners"]) / 2).sum() / gw)

    # home advantage as a log multiplier: 0.5*log(mean_home / mean_away)
    mh = (df["w"] * df["home_corners"]).sum() / gw
    ma = (df["w"] * df["away_corners"]).sum() / gw
    home_adv = 0.5 * math.log(max(mh, 0.1) / max(ma, 0.1))

    # per-competition baseline (league corner environment)
    comp_base = {}
    for comp, g in df.groupby("competition"):
        gwc = g["w"].sum()
        comp_base[comp] = float((g["w"] * (g["home_corners"] + g["away_corners"]) / 2).sum() / gwc)

    # team corner-for / corner-against weighted rates (home+away pooled), shrunk to league mean
    cf_sum, ca_sum, wsum = {}, {}, {}
    sf_sum, sa_sum = {}, {}                                    # shots for / against
    have_shots = df["home_shots"].notna().all() if "home_shots" in df else False
    def add(team, cf, ca, sf, sa, wt):
        cf_sum[team] = cf_sum.get(team, 0.0) + wt * cf
        ca_sum[team] = ca_sum.get(team, 0.0) + wt * ca
        sf_sum[team] = sf_sum.get(team, 0.0) + wt * (sf if sf == sf else 0.0)
        sa_sum[team] = sa_sum.get(team, 0.0) + wt * (sa if sa == sa else 0.0)
        wsum[team] = wsum.get(team, 0.0) + wt
    hs = pd.to_numeric(df.get("home_shots"), errors="coerce") if have_shots else None
    as_ = pd.to_numeric(df.get("away_shots"), errors="coerce") if have_shots else None
    df = df.assign(shf=hs, sha=as_) if have_shots else df.assign(shf=np.nan, sha=np.nan)
    for r in df.itertuples(index=False):
        add(r.home, r.home_corners, r.away_corners, r.shf, r.sha, r.w)
        add(r.away, r.away_corners, r.home_corners, r.sha, r.shf, r.w)

    shot_avg = float((df["w"] * ((df["shf"].fillna(0) + df["sha"].fillna(0)) / 2)).sum() / gw) if have_shots else 0.0
    attack, defence, shot_atk, shot_def = {}, {}, {}, {}
    for t in wsum:
        n = wsum[t]
        cf = (cf_sum[t] + SHRINK_GAMES * global_avg) / (n + SHRINK_GAMES)   # shrink
        ca = (ca_sum[t] + SHRINK_GAMES * global_avg) / (n + SHRINK_GAMES)
        attack[t] = math.log(max(0.25, cf) / global_avg)    # >0 => wins more corners
        defence[t] = math.log(max(0.25, ca) / global_avg)   # >0 => concedes more corners
        if have_shots and shot_avg > 0:
            sf = (sf_sum[t] + SHRINK_GAMES * shot_avg) / (n + SHRINK_GAMES)
            sa = (sa_sum[t] + SHRINK_GAMES * shot_avg) / (n + SHRINK_GAMES)
            shot_atk[t] = math.log(max(0.25, sf) / shot_avg)
            shot_def[t] = math.log(max(0.25, sa) / shot_avg)
    return {"global_avg": global_avg, "home_adv": home_adv, "comp_base": comp_base,
            "attack": attack, "defence": defence,
            "shot_atk": shot_atk, "shot_def": shot_def, "shot_avg": shot_avg,
            "half_life": half_life, "asof": str(asof.date()), "n_train": int(len(df))}


def lambdas(params: dict, home: str, away: str, comp: str | None = None,
            neutral: bool = False) -> tuple[float, float]:
    base = params["comp_base"].get(comp, params["global_avg"])
    h = 0.0 if neutral else params["home_adv"]
    ah, dh = params["attack"].get(home, 0.0), params["defence"].get(home, 0.0)
    aa, da = params["attack"].get(away, 0.0), params["defence"].get(away, 0.0)
    lam_h = base * math.exp(ah + da + h)
    lam_a = base * math.exp(aa + dh - h)
    return lam_h, lam_a


# ── shots-augmented Poisson GLM (the "next lever", now implemented) ─────────
# log lambda = log(comp_base) + theta . [1, corner_term, shot_term, home_sign]
#   corner_term = attack[att] + defence[dfd]   (team corner strength)
#   shot_term   = shot_atk[att] + shot_def[dfd] (team shot strength)
#   home_sign   = +1 attacker home, -1 attacker away
def _glm_row(params, att, dfd, is_home):
    ct = params["attack"].get(att, 0.0) + params["defence"].get(dfd, 0.0)
    st = params.get("shot_atk", {}).get(att, 0.0) + params.get("shot_def", {}).get(dfd, 0.0)
    return np.array([1.0, ct, st, 1.0 if is_home else -1.0])


def fit_glm(df: pd.DataFrame, params: dict):
    """Weighted Poisson IRLS with a per-competition log-base offset. Returns theta."""
    asof = pd.Timestamp(params["asof"]) + pd.Timedelta(days=1)
    w = _weights(df["date"], asof, params["half_life"])
    X, y, off, wt = [], [], [], []
    cb = params["comp_base"]; gb = params["global_avg"]
    for i, r in enumerate(df.itertuples(index=False)):
        base = math.log(max(0.5, cb.get(r.competition, gb)))
        X.append(_glm_row(params, r.home, r.away, True));  y.append(r.home_corners); off.append(base); wt.append(w[i])
        X.append(_glm_row(params, r.away, r.home, False)); y.append(r.away_corners); off.append(base); wt.append(w[i])
    X, y, off, wt = np.array(X), np.array(y, float), np.array(off), np.array(wt)
    theta = np.zeros(X.shape[1])
    for _ in range(50):
        mu = np.exp(np.clip(off + X @ theta, -6, 6))
        W = mu * wt
        z = X @ theta + (y - mu) / np.maximum(mu, 1e-9)
        XtW = X.T * W
        new = np.linalg.solve(XtW @ X + 1e-6 * np.eye(X.shape[1]), XtW @ z)
        if np.max(np.abs(new - theta)) < 1e-10:
            theta = new; break
        theta = new
    return theta


def lambdas_glm(params, theta, home, away, comp=None):
    base = math.log(max(0.5, params["comp_base"].get(comp, params["global_avg"])))
    lam_h = math.exp(base + _glm_row(params, home, away, True) @ theta)
    lam_a = math.exp(base + _glm_row(params, away, home, False) @ theta)
    return lam_h, lam_a


# ── markets ───────────────────────────────────────────────────────────────
def _pmf(lam):
    k = np.arange(MAX_C); return np.exp(-lam) * lam**k / np.array([math.factorial(i) for i in k])

def markets(lam_h: float, lam_a: float) -> dict:
    ph, pa = _pmf(lam_h), _pmf(lam_a)
    total = np.convolve(ph, pa)                       # P(total corners = k)
    exp_total = lam_h + lam_a
    out = {"exp_total": round(exp_total, 2),
           "exp_home": round(lam_h, 2), "exp_away": round(lam_a, 2),
           "most_likely_total": int(np.argmax(total))}
    for line in (8.5, 9.5, 10.5, 11.5, 12.5):
        p_over = total[np.arange(len(total)) > line].sum()
        out[f"over_{line}"] = round(float(p_over), 3)
    for side, p, lam in (("home", ph, lam_h), ("away", pa, lam_a)):
        for line in (3.5, 4.5, 5.5):
            out[f"{side}_over_{line}"] = round(float(p[np.arange(len(p)) > line].sum()), 3)
    return out


# ── validation (walk-forward, calibration-first) ───────────────────────────
def validate(half_life: float = HALF_LIFE_DAYS, split: float = 0.8):
    df = load_corners()
    cut = df["date"].quantile(split)
    train, test = df[df["date"] < cut], df[df["date"] >= cut]
    params = fit(train, asof=cut, half_life=half_life)

    # baseline: league-average total corners (no team info)
    base_total = params["global_avg"] * 2
    ae_model = ae_base = 0.0
    ll_model = ll_base = 0.0
    rel = []                       # (p_over_9.5, actual_over) for calibration
    base_over_rate = ( (train["home_corners"]+train["away_corners"]) > 9.5 ).mean()
    n = 0
    for r in test.itertuples(index=False):
        lh, la = lambdas(params, r.home, r.away, r.competition)
        mk = markets(lh, la)
        actual = r.home_corners + r.away_corners
        ae_model += abs(mk["exp_total"] - actual)
        ae_base += abs(base_total - actual)
        p = min(max(mk["over_9.5"], 1e-6), 1-1e-6)
        a = 1 if actual > 9.5 else 0
        ll_model += -(a*math.log(p) + (1-a)*math.log(1-p))
        pb = min(max(base_over_rate, 1e-6), 1-1e-6)
        ll_base += -(a*math.log(pb) + (1-a)*math.log(1-pb))
        rel.append((mk["over_9.5"], a)); n += 1

    # calibration table for Over 9.5
    ra = np.array(rel); ece = 0.0; bins = []
    for b in range(10):
        lo, hi = b/10, (b+1)/10
        m = (ra[:,0] >= lo) & (ra[:,0] < hi if b < 9 else ra[:,0] <= hi)
        if m.sum() == 0: continue
        conf, freq, cnt = ra[m,0].mean(), ra[m,1].mean(), int(m.sum())
        bins.append((lo, conf, freq, cnt)); ece += cnt/len(ra)*abs(conf-freq)

    print(f"Corners model validation  (train {len(train):,} -> test {len(test):,}, "
          f"half-life {half_life:.0f}d)\n")
    print(f"  Mean abs error, total corners:  model {ae_model/n:.2f}   league-avg {ae_base/n:.2f}")
    print(f"  Over/Under 9.5 log-loss:        model {ll_model/n:.4f}   base-rate {ll_base/n:.4f}")
    print(f"  Over 9.5 calibration error (ECE): {ece:.3f}")
    print(f"\n  Over 9.5 reliability (pred -> actual):")
    for lo, conf, freq, cnt in bins:
        print(f"    p~{conf:.2f}  actual {freq:.2f}  (n={cnt})")
    print(f"\n  league avg total corners: {base_total:.2f}   home_adv mult: {math.exp(params['home_adv']):.3f}")
    beats = (ll_model/n) < (ll_base/n) - 1e-3 and (ae_model/n) < (ae_base/n) - 0.02
    print("\n  VERDICT:", "team-strength corner signal beats the league baseline."
          if beats else
          "team-strength does NOT beat the league baseline — corners are dominated by "
          "game-state/noise, not stable team traits. Do not bet this model as-is; it would\n"
          "           lose to the corner line + margin. Next lever: shot-volume & match\n"
          "           favouritism features (see model docstring), with realistic low ceiling.")


def validate_shots(half_life: float = HALF_LIFE_DAYS, split: float = 0.8):
    """Head-to-head held-out: league baseline vs corners-only vs corners+shots GLM."""
    df = load_corners()
    cut = df["date"].quantile(split)
    train, test = df[df["date"] < cut], df[df["date"] >= cut]
    params = fit(train, asof=cut, half_life=half_life)
    theta = fit_glm(train, params)

    def llb(p, a): p = min(max(p, 1e-6), 1-1e-6); return -(a*math.log(p)+(1-a)*math.log(1-p))
    rows = {"baseline": [0,0,[]], "corners": [0,0,[]], "corners+shots": [0,0,[]]}
    base_over = ((train.home_corners+train.away_corners) > 9.5).mean()
    for r in test.itertuples(index=False):
        actual = r.home_corners + r.away_corners; a = 1 if actual > 9.5 else 0
        cb = params["comp_base"].get(r.competition, params["global_avg"])
        # baseline
        rows["baseline"][0] += abs(cb*2 - actual); rows["baseline"][1] += llb(base_over, a); rows["baseline"][2].append((base_over,a))
        # corners-only
        lh, la = lambdas(params, r.home, r.away, r.competition); mk = markets(lh, la)
        rows["corners"][0] += abs(mk["exp_total"]-actual); rows["corners"][1] += llb(mk["over_9.5"], a); rows["corners"][2].append((mk["over_9.5"],a))
        # corners+shots GLM
        lh2, la2 = lambdas_glm(params, theta, r.home, r.away, r.competition); mk2 = markets(lh2, la2)
        rows["corners+shots"][0] += abs(mk2["exp_total"]-actual); rows["corners+shots"][1] += llb(mk2["over_9.5"], a); rows["corners+shots"][2].append((mk2["over_9.5"],a))
    n = len(test)
    def ece(pairs):
        ra = np.array(pairs); e = 0.0
        for b in range(10):
            lo, hi = b/10, (b+1)/10
            m = (ra[:,0]>=lo)&(ra[:,0]<hi if b<9 else ra[:,0]<=hi)
            if m.sum(): e += m.sum()/len(ra)*abs(ra[m,0].mean()-ra[m,1].mean())
        return e
    print(f"Corners: baseline vs corners-only vs +shots GLM  (train {len(train):,} -> test {n:,})\n")
    print(f"  {'model':16s} {'MAE':>6} {'O/U9.5 logloss':>16} {'ECE':>7}")
    for k, (ae, ll, rel) in rows.items():
        print(f"  {k:16s} {ae/n:>6.3f} {ll/n:>16.4f} {ece(rel):>7.3f}")
    print(f"\n  GLM theta [intercept, corner, shot, home] = {np.round(theta,3)}")
    print(f"  (shot coefficient ~0 => shots add nothing beyond corner rates)")


def main():
    ap = argparse.ArgumentParser(description="Club-soccer corners model")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--validate-shots", action="store_true")
    ap.add_argument("--predict", nargs=2, metavar=("HOME","AWAY"))
    ap.add_argument("--comp", default=None)
    ap.add_argument("--teams", action="store_true")
    ap.add_argument("--half-life", type=float, default=HALF_LIFE_DAYS)
    a = ap.parse_args()
    if getattr(a, "validate_shots", False):
        validate_shots(half_life=a.half_life)
    elif a.validate:
        validate(half_life=a.half_life)
    elif a.teams:
        df = load_corners()
        if a.comp: df = df[df["competition"] == a.comp]
        print("\n".join(sorted(set(df["home"]) | set(df["away"]))))
    elif a.predict:
        params = fit(load_corners(), half_life=a.half_life)
        lh, la = lambdas(params, a.predict[0], a.predict[1], a.comp)
        mk = markets(lh, la)
        print(f"{a.predict[0]} vs {a.predict[1]}  ({a.comp or 'league avg base'})")
        for k, v in mk.items():
            print(f"  {k:18s} {v}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()

"""tennis/engine.py — in-process command API for the tennis engine.

Mirrors golf/engine.py: each command takes a params dict and returns a JSON-able
dict, dispatched by the adapter through app.engines._inproc.run_inprocess
(allowlist + secret redaction + finite-JSON guard).

Commands:
  schema    – player list (per tour), surfaces, markets, tour selector
  predict   – head-to-head P(A beats B) + set/games sub-markets (Markov chain)
  simulate  – full draw Monte-Carlo → outright win/final/SF/QF probabilities
  edge      – two-way de-vigged EV across odds.csv, fractional-Kelly staked
"""
from __future__ import annotations

import csv

from . import calibrate as C
from . import market as MK
from . import model as M
from . import portfolio as PORT
from . import simulate as S
from .providers import DATA_DIR

SURFACES = ["hard", "clay", "grass", "carpet"]
TOURS = [{"id": "atp", "label": "ATP (men)"}, {"id": "wta", "label": "WTA (women)"}]
DEFAULT_KELLY = 0.25
ODDS_CSV = DATA_DIR / "odds.csv"
DRAW_CSV = DATA_DIR / "draw.csv"


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _tour(p) -> str:
    t = str(p.get("tour") or p.get("model") or "atp").lower()
    return t if t in ("atp", "wta") else "atp"


def _load_params_or_raise(tour: str) -> dict:
    params = M.load_params(tour)
    if not params:
        raise ValueError(f"No fitted {tour.upper()} model. Run: "
                         f"python -m tennis.model --fit --tour {tour}")
    return params


def _all_names() -> list[str]:
    """Union of fitted player names across whichever tour params exist (for
    typeahead validation in the Predict tab)."""
    names: set[str] = set()
    for t in ("atp", "wta"):
        params = M.load_params(t)
        if params:
            names |= set(params.get("skills", {}).keys())
    return sorted(names)


def _sims_arg(p) -> int:
    try:
        n = int(p.get("sims", 50000))
    except (TypeError, ValueError):
        raise ValueError("sims must be a number")
    return max(2000, min(n, 200000))


def _h2h_fn(tour: str, weight_on: bool):
    """Return an h2h_log_odds(a, b, surface) closure backed by matches.csv, or
    None when H2H is off / no data is present (keeps predict cheap)."""
    if not weight_on:
        return None
    try:
        df = M.load_matches_df()
    except FileNotFoundError:
        return None
    df = df[df["tour"].astype(str).str.lower() == tour]
    if df.empty:
        return None
    return lambda a, b, s: M.h2h_log_odds_from_df(a, b, s, df)


# ─────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────

def cmd_schema(p=None):
    p = p or {}
    tour = _tour(p)
    params = M.load_params(tour)
    names = sorted(params.get("skills", {}).keys()) if params else _all_names()
    return {"kind": "match", "names": names, "tours": TOURS, "models": TOURS,
            "surfaces": SURFACES, "default_surface": "hard",
            "markets": ["match_winner", "set_hcp", "first_set", "total_games",
                        "win", "final", "sf", "qf"],
            "default_sims": 50000, "sim_options": [10000, 50000, 100000],
            "competitor_label": "Player",
            "fitted": {t: M.load_params(t) is not None for t in ("atp", "wta")}}


def cmd_predict(p):
    a, b = p.get("player_a"), p.get("player_b")
    if not a or not b:
        raise ValueError("predict needs player_a and player_b")
    tour = _tour(p)
    surface = (p.get("surface") or "hard").lower()
    best_of = int(p.get("best_of", 5 if tour == "atp" and p.get("slam") else 3))
    params = _load_params_or_raise(tour)

    h2h = 0.0
    hf = _h2h_fn(tour, bool(p.get("h2h", True)))
    if hf:
        h2h = hf(a, b, surface)
    pred = M.predict_match(a, b, surface, params, h2h_log_odds=h2h)
    p_a = pred["p_a"]
    mk = S.match_markets(p_a, best_of=best_of)

    calibrated = bool(p.get("calibrated", True))
    maps = C.load_maps() if calibrated else None
    p_a_disp = C.apply_one("match_winner", p_a, maps) if maps else p_a
    p_first = C.apply_one("first_set", mk["p_first_set"], maps) if maps else mk["p_first_set"]
    p_minus = C.apply_one("set_hcp", mk["p_a_minus_1_5_sets"], maps) if maps else mk["p_a_minus_1_5_sets"]

    rows = [
        {"market": "Match winner", "side": a, "p": round(p_a_disp, 4)},
        {"market": "Match winner", "side": b, "p": round(1 - p_a_disp, 4)},
        {"market": "First set", "side": a, "p": round(p_first, 4)},
        {"market": f"{a} −1.5 sets", "side": a, "p": round(p_minus, 4)},
        {"market": f"{a} +1.5 sets", "side": a, "p": round(mk["p_a_plus_1_5_sets"], 4)},
    ]
    columns = [
        {"key": "market", "label": "Market", "fmt": "text"},
        {"key": "side", "label": "Selection", "fmt": "text"},
        {"key": "p", "label": "Model", "fmt": "pct1"},
    ]
    note = (f"{tour.upper()} · {surface} · best-of-{best_of} · "
            f"exp games ≈ {mk['exp_total_games']:.1f}"
            + (" · calibrated" if maps else "")
            + (f" · H2H {h2h:+.2f}" if h2h else ""))
    return {
        "note": note,
        "outcomes": [{"label": a, "prob": p_a_disp}, {"label": b, "prob": 1 - p_a_disp}],
        "table": {"columns": columns, "rows": rows},
        "result": {a: p_a_disp, b: 1 - p_a_disp},
    }


def _load_draw(tour: str) -> list[tuple[str, str]]:
    if not DRAW_CSV.exists():
        raise ValueError(f"No draw. Add {DRAW_CSV} "
                         "(tour, surface, best_of, round, player_a, player_b) "
                         "or run: python -m tennis.fetch --draw-template")
    pairings: list[tuple[str, str]] = []
    surface = "hard"
    best_of = 3
    with open(DRAW_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("tour") or "").lower() not in ("", tour):
                continue
            a, b = (row.get("player_a") or "").strip(), (row.get("player_b") or "").strip()
            if a and b:
                pairings.append((a, b))
                surface = (row.get("surface") or surface).lower()
                try:
                    best_of = int(float(row.get("best_of") or best_of))
                except ValueError:
                    pass
    if not pairings:
        raise ValueError(f"No {tour.upper()} rows in {DRAW_CSV}.")
    return pairings, surface, best_of


def cmd_simulate(p):
    tour = _tour(p)
    params = _load_params_or_raise(tour)
    pairings, surface, best_of = _load_draw(tour)
    n = _sims_arg(p)
    import numpy as np
    rng = np.random.default_rng(int(p.get("seed", 0)) or None)
    hf = _h2h_fn(tour, bool(p.get("h2h", True)))
    res = S.simulate_draw(pairings, params, surface, best_of=best_of,
                          n_sims=n, rng=rng, h2h_fn=hf)
    rows = [{"player": k, "win": round(v["win"], 4), "final": round(v["final"], 4),
             "sf": round(v["sf"], 4), "qf": round(v["qf"], 4)}
            for k, v in res.items()]
    rows.sort(key=lambda r: -r["win"])
    columns = [
        {"key": "player", "label": "Player", "fmt": "text"},
        {"key": "win", "label": "Win", "fmt": "pct1"},
        {"key": "final", "label": "Final", "fmt": "pct"},
        {"key": "sf", "label": "SF", "fmt": "pct"},
        {"key": "qf", "label": "QF", "fmt": "pct"}]
    note = (f"{n:,} sims · {len(pairings)} first-round matches · "
            f"{tour.upper()} · {surface} · best-of-{best_of}")
    return {"note": note, "columns": columns, "rows": rows}


def _load_odds() -> list[dict]:
    if not ODDS_CSV.exists():
        raise ValueError(f"No odds. Add {ODDS_CSV} (tour, surface, best_of, "
                         "player_a, player_b, odds_a, odds_b) or run: "
                         "python -m tennis.fetch --odds-template")
    with open(ODDS_CSV, newline="") as f:
        return [r for r in csv.DictReader(f)
                if (r.get("player_a") or "").strip() and (r.get("player_b") or "").strip()]


def _kelly(p_model: float, odds: float, frac: float) -> float:
    b = odds - 1.0
    if b <= 0:
        return 0.0
    f = (b * p_model - (1.0 - p_model)) / b
    return max(0.0, f) * frac


def cmd_edge(p):
    tour = _tour(p)
    params = _load_params_or_raise(tour)
    odds_rows = _load_odds()
    bankroll = float(p.get("bankroll", 100.0))
    peak = float(p.get("peak", bankroll))
    kelly_frac = float(p.get("kelly", DEFAULT_KELLY))
    min_edge = float(p.get("min_edge", 0.0))
    hf = _h2h_fn(tour, bool(p.get("h2h", True)))
    maps = C.load_maps() if bool(p.get("calibrated", True)) else None
    blended = bool(p.get("market_blend", True))
    w_mkt = MK.blend_weights().get("match_winner", 0.5)

    rows = []
    for r in odds_rows:
        if (r.get("tour") or "").lower() not in ("", tour):
            continue
        a, b = r["player_a"].strip(), r["player_b"].strip()
        surface = (r.get("surface") or "hard").lower()
        try:
            oa, ob = float(r["odds_a"]), float(r["odds_b"])
        except (ValueError, KeyError):
            continue
        h2h = hf(a, b, surface) if hf else 0.0
        p_a = M.predict_match(a, b, surface, params, h2h_log_odds=h2h)["p_a"]
        if maps:
            p_a = C.apply_one("match_winner", p_a, maps)
        pm_a, pm_b = MK.devig_two_way(oa, ob)
        for (home, away, odds, p_model, p_mkt) in (
                (a, b, oa, p_a, pm_a), (b, a, ob, 1 - p_a, pm_b)):
            p_eff = MK.blend(p_model, p_mkt, w_mkt) if blended else p_model
            ev = p_eff * odds - 1.0
            stake = round(bankroll * _kelly(p_eff, odds, kelly_frac), 2) if ev > 0 else 0.0
            rows.append({
                "player": home, "opponent": away, "home": home, "away": away,
                "surface": surface, "market": "match_winner", "side": "win",
                "odds": round(odds, 3), "p_model": round(p_model, 4),
                "p_blend": round(p_eff, 4), "p_market": round(p_mkt, 4),
                "ev_per_unit": round(ev, 4), "stake_gbp": stake,
                "recommended": False})

    # stake only +EV bets, then apply simultaneous-Kelly portfolio discipline
    staked = PORT.apply_portfolio([r for r in rows if r["ev_per_unit"] > 0],
                                  bankroll=bankroll, peak=peak)
    stake_by = {(r["player"], r["opponent"]): r["stake_gbp"] for r in staked}
    rec_keys = {(r["player"], r["opponent"]) for r in staked
                if r["stake_gbp"] > 0 and r["ev_per_unit"] > min_edge}
    for r in rows:
        key = (r["player"], r["opponent"])
        r["stake_gbp"] = stake_by.get(key, 0.0)
        r["recommended"] = key in rec_keys

    rows.sort(key=lambda r: -r["ev_per_unit"])
    columns = [
        {"key": "player", "label": "Player", "fmt": "text"},
        {"key": "opponent", "label": "Opponent", "fmt": "text"},
        {"key": "market", "label": "Market", "fmt": "text"},
        {"key": "odds", "label": "Odds", "fmt": "num"},
        {"key": "p_model", "label": "Model", "fmt": "pct"},
        {"key": "p_blend", "label": "Blend", "fmt": "pct"},
        {"key": "p_market", "label": "Market", "fmt": "pct"},
        {"key": "ev_per_unit", "label": "EV", "fmt": "signed_num"},
        {"key": "stake_gbp", "label": "Stake", "fmt": "gbp"}]
    n_rec = sum(1 for r in rows if r["recommended"])
    note = (f"{n_rec} staked / {len(rows)} priced · {tour.upper()} · "
            f"{PORT.summary(staked, bankroll, peak)}"
            + (" · calibrated" if maps else "")
            + (" · market-blend" if blended else ""))
    return {"note": note, "columns": columns, "rows": rows}


COMMANDS = {"schema": cmd_schema, "predict": cmd_predict,
            "simulate": cmd_simulate, "edge": cmd_edge}

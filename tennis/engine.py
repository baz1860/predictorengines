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
    M.assert_params_fresh(params)
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

    pred = M.predict_match(a, b, surface, params)
    p_a = pred["p_a"]
    base = M.serve_base(a, b, surface, params)

    calibrated = bool(p.get("calibrated", True))
    maps = C.load_maps() if calibrated else None
    p_a_disp = C.apply_oriented(a, b, p_a, maps) if maps else p_a
    # Re-invert all sub-markets from the calibrated headline so the displayed
    # match/set/handicap board remains internally coherent.
    mk = S.match_markets(
        p_a_disp, best_of=best_of, base=base or S.base_serve(tour),
        games_cal=float(params.get("games_cal", 1.0)),
    )

    rows = [
        {"market": "Match winner", "side": a, "p": round(p_a_disp, 4)},
        {"market": "Match winner", "side": b, "p": round(1 - p_a_disp, 4)},
        {"market": "First set", "side": a, "p": round(mk["p_first_set"], 4)},
        {"market": f"{a} −1.5 sets", "side": a,
         "p": round(mk["p_a_minus_1_5_sets"], 4)},
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
            + (f" · unresolved: {', '.join(pred['unresolved'])}"
               if pred["unresolved"] else ""))
    return {
        "note": note,
        "outcomes": [{"label": a, "prob": p_a_disp}, {"label": b, "prob": 1 - p_a_disp}],
        "table": {"columns": columns, "rows": rows},
        "result": {a: p_a_disp, b: 1 - p_a_disp},
    }


def _load_draw_groups(tour: str, event_filter: str = "") -> list[dict]:
    if not DRAW_CSV.exists():
        raise ValueError(f"No draw. Add {DRAW_CSV} "
                         "(tour, surface, best_of, round, player_a, player_b) "
                         "or run: python -m tennis.fetch --draw-template")
    wanted = event_filter.lower().strip()
    groups: dict[str, dict] = {}
    with open(DRAW_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("tour") or "").lower() not in ("", tour):
                continue
            event = (row.get("tourney_name") or "").strip() or "Saved draw"
            if wanted and wanted not in event.lower():
                continue
            a, b = (row.get("player_a") or "").strip(), (row.get("player_b") or "").strip()
            if a and b:
                try:
                    best_of = int(float(row.get("best_of") or 3))
                except (TypeError, ValueError):
                    best_of = 3
                group = groups.setdefault(event, {
                    "event": event, "event_id": row.get("event_id") or "",
                    "surface": (row.get("surface") or "hard").lower(),
                    "best_of": best_of, "rows": []})
                group["rows"].append({
                    "round": (row.get("round") or "").strip(),
                    "player_a": a,
                    "player_b": b,
                    "state": (row.get("state") or "").strip(),
                    "winner": (row.get("winner") or "").strip(),
                    "surface": (row.get("surface") or group["surface"]).lower(),
                    "best_of": best_of,
                })
    if not groups:
        raise ValueError(f"No {tour.upper()} rows in {DRAW_CSV}.")
    return list(groups.values())


def cmd_simulate(p):
    tour = _tour(p)
    params = _load_params_or_raise(tour)
    event_filter = str(p.get("event") or p.get("tourney_name") or "")
    groups = _load_draw_groups(tour, event_filter)
    n = _sims_arg(p)
    import numpy as np
    rng = np.random.default_rng(int(p.get("seed", 0)) or None)
    rows = []
    total_draw_rows = 0
    locked = 0
    for group in groups:
        draw_rows = group["rows"]
        total_draw_rows += len(draw_rows)
        locked += sum(1 for r in draw_rows if r.get("state") == "post" and r.get("winner"))
        try:
            res = S.simulate_draw_rows(draw_rows, params, group["surface"],
                                       best_of=group["best_of"], n_sims=n,
                                       rng=rng)
        except ValueError as e:
            raise ValueError(f"{e}. Re-fetch the draw from ESPN or save a draw with "
                             "completed winners/state columns.") from None
        rows.extend({"event": group["event"], "player": k,
                     "win": round(v["win"], 4), "final": round(v["final"], 4),
                     "sf": round(v["sf"], 4), "qf": round(v["qf"], 4)}
                    for k, v in res.items())
    rows.sort(key=lambda r: -r["win"])
    columns = [
        *([{"key": "event", "label": "Event", "fmt": "text"}] if len(groups) > 1 else []),
        {"key": "player", "label": "Player", "fmt": "text"},
        {"key": "win", "label": "Win", "fmt": "pct1"},
        {"key": "final", "label": "Final", "fmt": "pct"},
        {"key": "sf", "label": "SF", "fmt": "pct"},
        {"key": "qf", "label": "QF", "fmt": "pct"}]
    note = (f"{n:,} sims per event · {len(groups)} event(s) · {total_draw_rows} draw rows"
            + (f" · {locked} locked result(s)" if locked else "")
            + f" · {tour.upper()}")
    return {"note": note, "columns": columns, "rows": rows}


def _load_odds() -> list[dict]:
    if not ODDS_CSV.exists():
        raise ValueError(f"No odds. Add {ODDS_CSV} (tour, surface, best_of, "
                         "player_a, player_b, odds_a, odds_b) or run: "
                         "python -m tennis.fetch --odds-template")
    with open(ODDS_CSV, newline="") as f:
        return [r for r in csv.DictReader(f)
                if (r.get("player_a") or "").strip() and (r.get("player_b") or "").strip()]


def cmd_edge(p):
    tour = _tour(p)
    params = _load_params_or_raise(tour)
    odds_rows = _load_odds()
    bankroll = float(p.get("bankroll", 100.0))
    peak = float(p.get("peak", bankroll))
    kelly_frac = float(p.get("kelly", DEFAULT_KELLY))
    min_edge = float(p.get("min_edge", 0.0))
    maps = C.load_maps() if bool(p.get("calibrated", True)) else None
    blended = bool(p.get("market_blend", True))
    w_mkt = MK.blend_weights().get("match_winner", 0.5)

    rows = []
    unresolved_rows = 0
    for r in odds_rows:
        if (r.get("tour") or "").lower() not in ("", tour):
            continue
        a, b = r["player_a"].strip(), r["player_b"].strip()
        surface = (r.get("surface") or "hard").lower()
        try:
            oa, ob = float(r["odds_a"]), float(r["odds_b"])
        except (ValueError, KeyError):
            continue
        prediction = M.predict_match(a, b, surface, params)
        if prediction["unresolved"]:
            unresolved_rows += 1
            continue
        p_a = prediction["p_a"]
        if maps:
            p_a = C.apply_oriented(a, b, p_a, maps)
        pm_a, pm_b = MK.devig_two_way(oa, ob)
        for (home, away, odds, p_model, p_mkt) in (
                (a, b, oa, p_a, pm_a), (b, a, ob, 1 - p_a, pm_b)):
            p_eff = MK.blend(p_model, p_mkt, w_mkt) if blended else p_model
            ev = p_eff * odds - 1.0
            stake = PORT.kelly_stake(p_eff, odds, bankroll, kelly_frac) if ev > 0 else 0.0
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
            + (" · market-blend" if blended else "")
            + (f" · skipped {unresolved_rows} unresolved match(es)"
               if unresolved_rows else ""))
    return {"note": note, "columns": columns, "rows": rows}


COMMANDS = {"schema": cmd_schema, "predict": cmd_predict,
            "simulate": cmd_simulate, "edge": cmd_edge}

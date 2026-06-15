#!/usr/bin/env python3
"""Isolated runner for the golf engine (v2).

Subprocess invoked by app/engines/golf.py with cwd + PYTHONPATH set to golf/, so
`import model`, `import simulate`, `import edge` resolve to the golf modules
without colliding with the root engine. JSON params on stdin, JSON on stdout.

Commands:
  schema    – field names, markets, sim options
  simulate  – fit-backed field projection (win/T5/T10/T20/cut) → predictions.csv
  predict   – head-to-head matchup probability for two players (joint sim)
  edge      – calibrated + market-blended edges across all markets, portfolio-
              staked; matchup/3-ball odds from matchups.csv / threeballs.csv
"""
import json
import sys
from pathlib import Path

import numpy as np

import model
import simulate as GSIM
import edge as GE
import portfolio as GPORT

DATA = Path(__file__).resolve().parent  # unused; golf/ is cwd


def _params():
    raw = sys.stdin.read().strip()
    return json.loads(raw) if raw else {}


def _field_names() -> list[str]:
    """Field for the current event: field.csv if present, else the latest event
    in rounds.csv (so the app still works before fetch.py --espn is run)."""
    from model import load_field, load_players
    try:
        field = load_field(players=load_players())
        names = [p.name for p in field]
        if names:
            return names
    except FileNotFoundError:
        pass
    import pandas as pd
    rounds = model.ROUNDS_CSV
    if rounds.exists():
        df = pd.read_csv(rounds)
        if not df.empty:
            tid = df.sort_values("date")["tournament_id"].iloc[-1]
            return sorted(df[df["tournament_id"] == tid]["player"].unique())
    raise ValueError("No field — run fetch.py --espn or seed rounds.csv.")


def _rated_field(course="", major=False):
    """Rated Player objects from the fitted model, with legacy fallback."""
    names = _field_names()
    params = model.load_params()
    if params:
        return model.predict_field(names, params, course=course, is_major=major), True
    # legacy path: players.csv composite ratings
    from model import compute_ratings, load_field, load_players, \
        load_course_history, load_recent_form
    field = load_field(players=load_players())
    ch = load_course_history(course) if course else {}
    return compute_ratings(field, course=course, is_major=major,
                           course_history=ch, recent_form=load_recent_form()), False


def cmd_schema():
    names = sorted(_field_names())
    return {"kind": "field", "names": names, "models": [],
            "default_sims": 50000, "sim_options": [10000, 50000, 100000],
            "markets": ["win", "top5", "top10", "top20", "cut", "matchup", "3ball"],
            "competitor_label": "Player",
            "fitted": model.load_params() is not None}


def _sims_arg(p):
    try:
        n = int(p.get("sims", 50000))
    except (TypeError, ValueError):
        raise ValueError("sims must be a number")
    return max(2000, min(n, 200000))


def cmd_simulate(p):
    n = _sims_arg(p)
    course = p.get("course", "") or ""
    major = bool(p.get("major", False))
    cut_rule = int(p.get("cut_rule", 65))
    rated, fitted = _rated_field(course, major)
    rng = np.random.default_rng(int(p.get("seed", 0)) or None)
    results = GSIM.simulate_tournament(rated, n_sims=n, cut_rule=cut_rule, rng=rng)
    GSIM.write_predictions(rated, results)
    rows = []
    for pl in rated:
        r = results[pl.name]
        rows.append({"name": pl.name, "rating": round(pl.rating, 2),
                     "sigma": round(pl.sigma, 2),
                     "win": round(r["win"], 4), "top5": round(r["top5"], 4),
                     "top10": round(r["top10"], 4), "top20": round(r["top20"], 4),
                     "cut": round(r["made_cut"], 4),
                     "avg_finish": round(r["avg_finish"], 1)})
    rows.sort(key=lambda x: -x["win"])
    columns = [
        {"key": "name", "label": "Player", "fmt": "text"},
        {"key": "rating", "label": "Rating", "fmt": "signed_num"},
        {"key": "sigma", "label": "σ", "fmt": "num1"},
        {"key": "win", "label": "Win", "fmt": "pct1"},
        {"key": "top5", "label": "Top 5", "fmt": "pct"},
        {"key": "top10", "label": "Top 10", "fmt": "pct"},
        {"key": "top20", "label": "Top 20", "fmt": "pct"},
        {"key": "cut", "label": "Make cut", "fmt": "pct"},
        {"key": "avg_finish", "label": "Avg fin", "fmt": "num1"}]
    src = "fitted model" if fitted else "legacy players.csv"
    note = (f"{n:,} sims · {len(rated)} players · {src}"
            + (f" · {course}" if course else ""))
    # When the field is no larger than the cut rule the cut never binds, so the
    # make-cut and wider top-N columns collapse to ~100% and are not meaningful.
    if not results.get("__cut_binds__", True):
        note += (f" · ⚠ cut does not bind (field {len(rated)} ≤ cut {cut_rule}): "
                 "make-cut/top-N not meaningful")
    return {"note": note, "columns": columns, "rows": rows}


def cmd_predict(p):
    """Head-to-head matchup probability for two named players."""
    a, b = p.get("player_a"), p.get("player_b")
    if not a or not b:
        raise ValueError("predict needs player_a and player_b")
    course = p.get("course", "") or ""
    major = bool(p.get("major", False))
    rated, _ = _rated_field(course, major)
    n = _sims_arg(p)
    res = GSIM.simulate_tournament(rated, n_sims=n, rng=np.random.default_rng(0),
                                   matchups=[(a, b)])
    d = res.get("__matchups__", {}).get((a, b))
    if not d:
        raise ValueError(f"Both players must be in the field: {a}, {b}")
    return {"note": f"{n:,} sims · {course or 'no course'}",
            "columns": [{"key": "player", "label": "Player", "fmt": "text"},
                        {"key": "p", "label": "P(finish better)", "fmt": "pct1"}],
            "rows": [{"player": a, "p": round(d[a], 4)},
                     {"player": b, "p": round(d[b], 4)}],
            "result": {a: d[a], b: d[b], "tie": d["tie"]}}


def cmd_edge(p):
    course = p.get("course", "") or ""
    major = bool(p.get("major", False))
    rated, _ = _rated_field(course, major)
    odds_data = GE.load_odds_csv()
    matchup_odds = GE.load_matchup_odds()
    threeball_odds = GE.load_threeball_odds()
    if not (odds_data or matchup_odds or threeball_odds):
        raise ValueError("No odds. Add golf/data/odds.csv (name, odds_win, "
                         "odds_top5, odds_top10, odds_top20, odds_cut) and/or "
                         "matchups.csv / threeballs.csv.")
    bankroll = float(p.get("bankroll", 100.0))
    peak = float(p.get("peak", bankroll))
    kelly = float(p.get("kelly", GE.DEFAULT_KELLY))
    calibrated = bool(p.get("calibrated", True))
    blended = bool(p.get("market_blend", True))
    min_edge = float(p.get("min_edge", 0.0))

    pairs = [(a, b) for (a, b) in matchup_odds]
    trios = [t for t in threeball_odds]
    n = _sims_arg(p)
    results = GSIM.simulate_tournament(rated, n_sims=n, cut_rule=int(p.get("cut_rule", 65)),
                                       rng=np.random.default_rng(0),
                                       matchups=pairs, threeballs=trios)
    rows = GE.price_all(rated, results, odds_data, matchup_odds, threeball_odds,
                        bankroll=bankroll, kelly=kelly, calibrated=calibrated,
                        blended=blended, min_edge=min_edge)
    # stake only +EV bets, then apply portfolio discipline
    staked = [r for r in rows if r["ev_per_unit"] > 0]
    staked = GPORT.apply_portfolio(staked, bankroll=bankroll, peak=peak)
    staked_keys = {(r["player"], r["side"]) for r in staked}
    stake_by = {(r["player"], r["side"]): r["stake_gbp"] for r in staked}
    for r in rows:
        r["stake_gbp"] = stake_by.get((r["player"], r["side"]), 0.0)
        r["recommended"] = (r["player"], r["side"]) in staked_keys

    columns = [
        {"key": "player", "label": "Player", "fmt": "text"},
        {"key": "market", "label": "Market", "fmt": "text"},
        {"key": "odds", "label": "Odds", "fmt": "num"},
        {"key": "p_model", "label": "Model", "fmt": "pct"},
        {"key": "p_market", "label": "Market", "fmt": "pct"},
        {"key": "ev_per_unit", "label": "EV", "fmt": "signed_num"},
        {"key": "stake_gbp", "label": "Stake", "fmt": "gbp"}]
    note = (f"{len([r for r in rows if r['recommended']])} staked / {len(rows)} "
            f"priced · {GPORT.summary(staked, bankroll, peak)}"
            f"{' · calibrated' if calibrated else ''}"
            f"{' · market-blend' if blended else ''}")
    # price_all already suppresses make-cut/top-N when the cut doesn't bind;
    # tell the user so the missing markets aren't mistaken for a data gap.
    if not results.get("__cut_binds__", True):
        note += (f" · ⚠ cut does not bind (field {len(rated)} ≤ cut rule): "
                 "make-cut suppressed")
    return {"note": note, "columns": columns, "rows": rows}


COMMANDS = {"schema": lambda p: cmd_schema(), "simulate": cmd_simulate,
            "predict": cmd_predict, "edge": cmd_edge}


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "schema"
    try:
        print(json.dumps(COMMANDS[cmd](_params())))
    except ValueError as e:
        print(json.dumps({"error": str(e)})); sys.exit(2)
    except Exception as e:  # noqa
        print(json.dumps({"error": f"{type(e).__name__}: {e}"})); sys.exit(1)


if __name__ == "__main__":
    main()

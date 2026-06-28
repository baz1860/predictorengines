"""In-process command API for the golf engine (refactor Phase 4).

The command logic that used to live in app/engines/runners/golf_runner.py, now imported
and called directly by the adapter (no subprocess). Functions take a params dict and
return a JSON-able dict; errors are plain exceptions that the adapter dispatches through
app.engines._inproc.run_inprocess (allowlist + redaction + finite-JSON).

Commands:
  schema    – field names, markets, sim options
  refresh   – free-source provider refresh → SQLite/CSV cache + manifest
  simulate  – fit-backed field projection (win/T5/T10/T20/cut) → predictions.csv
  predict   – head-to-head matchup probability for two players (joint sim)
  edge      – calibrated + market-blended edges across all markets, portfolio-staked
"""
import json
from pathlib import Path

import numpy as np

from . import edge as GE
from . import model
from . import portfolio as GPORT
from . import refresh as GREF
from . import round_pricer as GRP
from . import simulate as GSIM
from . import simulate_inplay as GSIP
from .providers.odds_manual import ManualOddsProvider

DATA_DIR = Path(__file__).parent / "data"


def _field_names() -> list[str]:
    """Field for the current event: field.csv if present, else the latest event
    in rounds.csv (so the app still works before fetch.py --espn is run)."""
    from .model import load_field, load_players
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
    from .model import compute_ratings, load_field, load_players, \
        load_course_history, load_recent_form
    field = load_field(players=load_players())
    ch = load_course_history(course) if course else {}
    return compute_ratings(field, course=course, is_major=major,
                           course_history=ch, recent_form=load_recent_form()), False


def _live_state(p) -> dict | None:
    """Resolve the in-play state for the current event, or None for pre-tournament.

    Order of precedence:
      1. ``p["pretournament"]`` truthy → force the pre-tournament projection.
      2. explicit ``p["rounds_done"]`` (+ optional ``p["scores_csv"]``).
      3. ``data/live_state.json`` written by refresh from the live leaderboard.

    Returns ``{"rounds_done", "scores", "source", "event_name"}`` where ``scores``
    maps lowercase player name → cumulative strokes-to-par, or None when there is
    no completed round to condition on.
    """
    if p.get("pretournament") or p.get("force_pretournament"):
        return None

    rounds_done = 0
    scores_path = None
    event_name = ""
    source = ""

    if p.get("rounds_done"):
        rounds_done = int(p["rounds_done"])
        scores_path = Path(p.get("scores_csv") or (DATA_DIR / "scores_live.csv"))
        source = "explicit params"
    else:
        state_file = DATA_DIR / "live_state.json"
        if state_file.exists():
            try:
                state = json.loads(state_file.read_text())
            except (ValueError, OSError):
                state = {}
            rounds_done = int(state.get("rounds_done") or 0)
            event_name = state.get("event_name", "")
            scores_path = DATA_DIR / state.get("scores_csv", "scores_live.csv")
            source = "live leaderboard"

    if rounds_done < 1 or not scores_path or not Path(scores_path).exists():
        return None
    if rounds_done >= GSIP.TOTAL_ROUNDS:
        return None  # tournament complete — nothing left to simulate

    scores = GSIP.load_scores(Path(scores_path))
    if not scores:
        return None
    return {"rounds_done": rounds_done, "scores": scores,
            "source": source, "event_name": event_name}


def _inplay_results(rated, state, n, rng, matchups=None, threeballs=None):
    """Run the in-play sim over the rated field; return (results, survivors).

    The results dict mirrors the pre-tournament sim's shape so edge.price_all can
    consume it unchanged: per-player win/top5/top10/top20/made_cut, the reserved
    ``__cut_binds__`` flag, and (when requested) score-aware ``__matchups__`` /
    ``__threeballs__``. The make-cut market is suppressed in-play — survivors are
    already through, so it is not a live betting market.
    """
    scores = state["scores"]
    survivors = [pl for pl in rated if pl.name.lower() in scores]
    if not survivors:
        raise ValueError(
            "No field players matched the live scores snapshot (name mismatch?). "
            "Re-run refresh, or pass pretournament=1 to force the pre-event model.")
    res = GSIP.simulate_inplay(survivors, scores, state["rounds_done"],
                               n_sims=n, rng=rng,
                               matchups=matchups, threeballs=threeballs)
    results = {"__cut_binds__": False}
    for name, r in res.items():
        if name.startswith("__"):
            results[name] = r          # pass reserved keys through unchanged
            continue
        results[name] = {
            "win": r["win"], "top5": r["top5"], "top10": r["top10"],
            "top20": r["top20"], "made_cut": 1.0, "missed_cut": 0.0,
            "avg_finish": r["avg_finish"], "current_score": r["current_score"],
            "n_sims": r["n_sims"],
        }
    return results, survivors


def cmd_schema(_p=None):
    names = sorted(_field_names())
    state = _live_state({})
    return {"kind": "field", "names": names, "models": [],
            "default_sims": 50000, "sim_options": [10000, 50000, 100000],
            "markets": ["win", "top5", "top10", "top20", "cut", "matchup", "3ball"],
            "competitor_label": "Player",
            "fitted": model.load_params() is not None,
            "live": state is not None,
            "rounds_done": state["rounds_done"] if state else 0}


def cmd_refresh(p):
    round_no = int(p.get("round", p.get("round_no", 1)) or 1)
    manifest = GREF.run_refresh(
        season=int(p["season"]) if p.get("season") else None,
        event=p.get("event_id", p.get("event", "")) or "",
        stats=bool(p.get("stats", False)),
        weather=bool(p.get("weather", False)),
        odds_api_sport=p.get("odds_api_sport", "") or "",
        manual_raw=p.get("manual_raw", "") or str(GREF.THREEBALLS_RAW),
        round_no=round_no,
        fit=bool(p.get("fit", False)),
        use_cache=bool(p.get("use_cache", False)),
    )
    provider_rows = manifest.get("provider_rows") or {}
    rows = [{"provider": key, "rows": value} for key, value in provider_rows.items()]
    qa = manifest.get("qa") or {}
    event = manifest.get("event") or {}
    warnings = len(qa.get("warnings") or [])
    errors = len(qa.get("errors") or [])
    note = "Free-source refresh"
    if event:
        note += f" · {event.get('name', 'current event')}"
    note += f" · {sum(int(r.get('rows') or 0) for r in rows):,} provider rows"
    if warnings:
        note += f" · {warnings} warning(s)"
    if errors:
        note += f" · {errors} error(s)"
    return {
        "note": note,
        "columns": [
            {"key": "provider", "label": "Provider", "fmt": "text"},
            {"key": "rows", "label": "Rows", "fmt": "num"},
        ],
        "rows": rows,
        "event": event,
        "provider_rows": provider_rows,
        "qa": qa,
        "manifest": manifest,
    }


def _sims_arg(p):
    try:
        n = int(p.get("sims", 50000))
    except (TypeError, ValueError):
        raise ValueError("sims must be a number")
    return max(2000, min(n, 200000))


def cmd_simulate(p):
    """Field projection. Auto-routes to the in-play sim once a round is complete.

    Before round 1 (no live scores) this is the pre-tournament projection off the
    fitted ratings. Once refresh has recorded a completed round, it conditions on
    the live leaderboard instead — fixing the rounds played and simulating only
    those remaining. Pass ``pretournament=1`` to force the pre-event projection.
    """
    state = _live_state(p)
    if state is not None:
        return cmd_simulate_inplay(p, _state=state)

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
    if not results.get("__cut_binds__", True):
        note += (f" · ⚠ cut does not bind (field {len(rated)} ≤ cut {cut_rule}): "
                 "make-cut/top-N not meaningful")
    return {"note": note, "columns": columns, "rows": rows}


def cmd_simulate_inplay(p, _state=None):
    """In-tournament projection conditioned on the live leaderboard.

    Fixes each surviving player's score through the completed rounds and
    simulates the remainder, so win/top-N reflect the current standings rather
    than the pre-event ratings. Reads live_state.json (written by refresh) unless
    ``rounds_done``/``scores_csv`` are passed explicitly.
    """
    state = _state if _state is not None else _live_state(p)
    if state is None:
        raise ValueError(
            "No completed-round scores available. Run refresh during a live event, "
            "or pass rounds_done=N with a scores_csv.")
    n = _sims_arg(p)
    course = p.get("course", "") or ""
    major = bool(p.get("major", False))
    rated, fitted = _rated_field(course, major)
    rng = np.random.default_rng(int(p.get("seed", 0)) or None)
    results, survivors = _inplay_results(rated, state, n, rng)
    GSIP.write_predictions_inplay(survivors, _results_for_writer(results), state["rounds_done"])

    rows = []
    for pl in survivors:
        r = results[pl.name]
        score = int(r["current_score"])
        rows.append({"name": pl.name, "rating": round(pl.rating, 2),
                     "score": f"{score:+d}" if score else "E",
                     "win": round(r["win"], 4), "top5": round(r["top5"], 4),
                     "top10": round(r["top10"], 4), "top20": round(r["top20"], 4),
                     "avg_finish": round(r["avg_finish"], 1)})
    rows.sort(key=lambda x: -x["win"])
    columns = [
        {"key": "name", "label": "Player", "fmt": "text"},
        {"key": "score", "label": "Thru", "fmt": "text"},
        {"key": "rating", "label": "Rating", "fmt": "signed_num"},
        {"key": "win", "label": "Win", "fmt": "pct1"},
        {"key": "top5", "label": "Top 5", "fmt": "pct"},
        {"key": "top10", "label": "Top 10", "fmt": "pct"},
        {"key": "top20", "label": "Top 20", "fmt": "pct"},
        {"key": "avg_finish", "label": "Avg fin", "fmt": "num1"}]
    rd = state["rounds_done"]
    left = GSIP.TOTAL_ROUNDS - rd
    src = "fitted model" if fitted else "legacy players.csv"
    ev_label = (state.get("event_name") + " · ") if state.get("event_name") else ""
    note = (f"{ev_label}in-play after R{rd} ({left} to play) · {n:,} sims · "
            f"{len(survivors)} survivors · {src} · live leaderboard")
    return {"note": note, "columns": columns, "rows": rows}


def _results_for_writer(results: dict) -> dict:
    """Adapt engine results back to the write_predictions_inplay shape."""
    return {k: v for k, v in results.items() if not k.startswith("__")}


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
    state = _live_state(p)
    inplay_note = ""
    if state is not None:
        # Live: every market — outrights, places, and tournament-long
        # matchups/3-balls — is priced off the same in-play sim, conditioned on
        # the leaderboard. (Round-by-round groups have their own path in
        # round_3balls.) Groups naming a cut player are skipped by the sim.
        results, _surv = _inplay_results(
            rated, state, n, np.random.default_rng(0),
            matchups=pairs, threeballs=trios)
        rd = state["rounds_done"]
        inplay_note = f" · in-play after R{rd} (live leaderboard)"
    else:
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
            f"{' · market-blend' if blended else ''}{inplay_note}")
    if not results.get("__cut_binds__", True) and state is None:
        note += (f" · ⚠ cut does not bind (field {len(rated)} ≤ cut rule): "
                 "make-cut suppressed")
    return {"note": note, "columns": columns, "rows": rows}


def cmd_round_3balls(p):
    params = model.load_params()
    if not params:
        raise ValueError("No model_params.json - run model.py --fit first.")
    round_no = int(p.get("round", p.get("round_no", 1)) or 1)
    event_id = p.get("event_id", "") or ""
    quotes = ManualOddsProvider().load_threeballs(event_id=event_id, round_no=round_no)
    if not quotes:
        raise ValueError("No 3-ball odds found in golf/data/threeballs.csv.")
    missing = GRP.field_mismatch(quotes, _field_names(), params)
    if missing:
        # Stale board (e.g. last week's tournament). Drop any prior edges file so
        # callers that re-read it (season.py) don't render the wrong event.
        GRP.OUT_CSV.unlink(missing_ok=True)
        raise ValueError(
            f"Round-group board does not match the current field: {len(missing)} "
            f"player(s) not in field.csv (stale board from another event?): "
            + ", ".join(missing[:12]) + ("…" if len(missing) > 12 else "")
            + ". Re-paste this event's tee groups into "
              "golf/data/threeballs_r1_raw.txt and rerun refresh."
        )
    bankroll = float(p.get("bankroll", 100.0))
    rows = GRP.price_round_groups(
        quotes,
        params,
        course=p.get("course", "") or "",
        is_major=bool(p.get("major", False)),
        sims=_sims_arg(p),
        bankroll=bankroll,
        kelly=float(p.get("kelly", 0.25)),
        min_rounds=int(p.get("min_rounds", 60)),
    )
    GRP.write_round_edges(rows)
    columns = [
        {"key": "round", "label": "Round", "fmt": "num"},
        {"key": "player", "label": "Player", "fmt": "text"},
        {"key": "odds", "label": "Odds", "fmt": "num"},
        {"key": "p_dead_heat_equiv", "label": "Model", "fmt": "pct"},
        {"key": "p_market", "label": "Market", "fmt": "pct"},
        {"key": "ev_pct", "label": "EV%", "fmt": "signed_num"},
        {"key": "kelly_stake", "label": "Stake", "fmt": "gbp"},
    ]
    markets = sorted({r.get("market", "3ball") for r in rows})
    label = "/".join(markets) if markets else "groups"
    return {"note": f"Round {round_no} {label} · {len(rows)} sides", "columns": columns, "rows": rows}


COMMANDS = {"schema": lambda p: cmd_schema(), "refresh": cmd_refresh,
            "simulate": cmd_simulate, "simulate_inplay": cmd_simulate_inplay,
            "predict": cmd_predict, "edge": cmd_edge,
            "round_3balls": cmd_round_3balls}

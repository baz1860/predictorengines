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
import csv
from pathlib import Path

import numpy as np

from . import edge as GE
from . import model
from . import portfolio as GPORT
from . import refresh as GREF
from . import round_pricer as GRP
from . import simulate as GSIM
from . import simulate_inplay as GSIP
from .providers.odds_manual import (ManualOddsProvider, board_event, norm_event,
                                    board_captured_at, threeballs_csv_path,
                                    threeballs_raw_path)

DATA_DIR = Path(__file__).parent / "data"


def _current_event_name() -> str:
    """Event name from field.csv ('' when unknown). Both field.csv and the
    board tags come from the same ESPN-resolved event name, so an exact
    (normalized) comparison is the staleness test."""
    return model.load_field_event()


def _current_event_context() -> dict:
    """Event provenance/rules embedded in field.csv by the refresh."""
    path = DATA_DIR / "field.csv"
    if not path.exists():
        return {}
    try:
        with path.open() as f:
            row = next(csv.DictReader(f), None) or {}
    except (OSError, csv.Error):
        return {}
    return {
        "event_id": str(row.get("event_id") or "").strip(),
        "event_name": str(row.get("event") or "").strip(),
        "cut_rule": int(row.get("cut_rule") or 65),
        "no_cut": str(row.get("no_cut") or "").strip().lower() in {"1", "true", "yes"},
    }


def _simulation_rules(p: dict) -> tuple[int, bool]:
    ctx = _current_event_context()
    return (
        int(p.get("cut_rule", ctx.get("cut_rule", 65)) or 65),
        bool(p.get("no_cut", ctx.get("no_cut", False))),
    )


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


def _rated_field(course="", major=False, options=None):
    """Rated Player objects from the fitted model, with legacy fallback."""
    from .model import load_field, load_players
    context = model.load_field_context()
    course = course or context.get("course", "")
    try:
        field_items = load_field(players=load_players())
    except FileNotFoundError:
        field_items = _field_names()
    params = model.load_params()
    if params:
        options = options or {}
        feature_flags = {
            "weather": bool(options.get("weather", False)),
            "public_stat": bool(options.get("public_stat", False)),
            "global_priors": bool(options.get("global_priors", False)),
            "exact_course": bool(options.get("exact_course", False)),
        }
        return model.predict_field(
            field_items,
            params,
            course=course,
            course_par=int(context.get("course_par") or 0),
            course_yards=int(context.get("course_yards") or 0),
            par3_holes=int(context.get("par3_holes") or 0),
            par4_holes=int(context.get("par4_holes") or 0),
            par5_holes=int(context.get("par5_holes") or 0),
            is_major=major,
            feature_flags=feature_flags,
        ), True
    # legacy path: players.csv composite ratings
    from .model import compute_ratings, load_course_history, load_recent_form
    field = load_field(players=load_players())
    ch = load_course_history(course) if course else {}
    return compute_ratings(field, course=course, is_major=major,
                           course_history=ch, recent_form=load_recent_form()), False


def _refresh_mtime() -> float | None:
    """Modification time of the last refresh (its manifest), or None if unknown."""
    try:
        return (DATA_DIR / "free_source_manifest.json").stat().st_mtime
    except OSError:
        return None


def _board_fresh(path: Path, ref: float | None, tol: float = 1800) -> bool:
    """True if the odds board was (re)written in the latest refresh cycle.

    A board written by the current refresh lands within seconds of the manifest,
    so anything older than `ref - tol` (default 30 min) is from a previous cycle
    and must not be priced against a live leaderboard. Unknown ref ⇒ treat as
    fresh (pre-tournament / no refresh marker to compare against).
    """
    if ref is None:
        return True
    captured = board_captured_at(path)
    if captured is None:
        return False
    return captured.timestamp() >= ref - tol


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
            event_id = state.get("event_id", "")
            scores_path = DATA_DIR / state.get("scores_csv", "scores_live.csv")
            source = "live leaderboard"

            current = _current_event_context()
            current_id = current.get("event_id", "")
            current_name = current.get("event_name", "")
            id_ok = bool(event_id and current_id and str(event_id) == str(current_id))
            name_ok = bool(event_name and current_name
                           and norm_event(event_name) == norm_event(current_name))
            if not (id_ok or name_ok):
                return None

    if rounds_done < 1 or not scores_path or not Path(scores_path).exists():
        return None
    total_rounds = int((state if 'state' in locals() else {}).get("total_rounds")
                       or GSIP.TOTAL_ROUNDS)
    if rounds_done >= total_rounds:
        return None  # tournament complete — nothing left to simulate

    scores = GSIP.load_scores(Path(scores_path))
    if not scores:
        return None
    return {"rounds_done": rounds_done, "scores": scores,
            "source": source, "event_name": event_name,
            "cut_rule": int((state if 'state' in locals() else {}).get("cut_rule") or 65),
            "no_cut": bool((state if 'state' in locals() else {}).get("no_cut", False)),
            "total_rounds": total_rounds}


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
                               matchups=matchups, threeballs=threeballs,
                               cut_rule=int(state.get("cut_rule", 65)),
                               no_cut=bool(state.get("no_cut", False)),
                               total_rounds=int(state.get("total_rounds", 4)))
    results = {"__cut_binds__": bool(res.get("__cut_binds__", False))}
    for name, r in res.items():
        if name.startswith("__"):
            results[name] = r          # pass reserved keys through unchanged
            continue
        results[name] = {
            "win": r["win"], "top5": r["top5"], "top10": r["top10"],
            "top20": r["top20"], "made_cut": r.get("made_cut", 1.0),
            "missed_cut": 1.0 - r.get("made_cut", 1.0),
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
        manual_raw=p.get("manual_raw", "") or str(threeballs_raw_path(round_no)),
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
    cut_rule, no_cut = _simulation_rules(p)
    rated, fitted = _rated_field(course, major, p)
    rng = np.random.default_rng(int(p.get("seed", 0)) or None)
    results = GSIM.simulate_tournament(
        rated, n_sims=n, cut_rule=cut_rule, no_cut=no_cut, rng=rng)
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
    rated, fitted = _rated_field(course, major, p)
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
    rated, _ = _rated_field(course, major, p)
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
    rated, _ = _rated_field(course, major, p)
    odds_data = GE.load_odds_csv()
    # Tournament edge prices 72-hole markets only. Single-round boards
    # (group_id tagged -r1/-r2/-r3) settle on one round and are priced by the
    # round card (cmd_round_3balls), so exclude them here.
    matchup_odds = GE.load_matchup_odds(tournament_only=True)
    threeball_odds = GE.load_threeball_odds(tournament_only=True)

    # Event-tag staleness guard: a matchup/3-ball board captured for another
    # event must not be priced against this one — field name-overlap can't
    # tell consecutive events apart (co-sanctioned weeks share most players),
    # so only a board tagged with the current event is trusted.
    stale_note = ""
    current_event = _current_event_name()
    if current_event:
        wrong = []
        for label, path in (("outright", DATA_DIR / "odds.csv"),
                            ("matchup", DATA_DIR / "matchups.csv"),
                            ("3-ball", DATA_DIR / "threeballs.csv")):
            odds_ref = (odds_data if label == "outright" else
                        matchup_odds if label == "matchup" else threeball_odds)
            if not odds_ref:
                continue
            tag = board_event(path)
            if norm_event(tag) != norm_event(current_event):
                wrong.append(f"{label} ({tag or 'untagged'})")
                if label == "outright":
                    odds_data = {}
                elif label == "matchup":
                    matchup_odds = {}
                else:
                    threeball_odds = {}
        if wrong:
            stale_note = (" · ⚠ board(s) not from '" + current_event + "' skipped: "
                          + ", ".join(wrong))

    # Live staleness guard: once we're pricing off the leaderboard, an odds board
    # that wasn't refreshed this cycle is dangerous — comparing yesterday's
    # even-money matchup line to a score-aware model manufactures huge fake
    # edges (e.g. backing a player who now leads by 10). Drop any board older
    # than the latest refresh so we never bet a stale price.
    state = _live_state(p)
    if state is not None:
        ref = _refresh_mtime()
        stale = []
        if odds_data and not _board_fresh(DATA_DIR / "odds.csv", ref):
            odds_data = {}
            stale.append("outright")
        if matchup_odds and not _board_fresh(DATA_DIR / "matchups.csv", ref):
            matchup_odds = {}
            stale.append("matchup")
        if threeball_odds and not _board_fresh(DATA_DIR / "threeballs.csv", ref):
            threeball_odds = {}
            stale.append("3-ball")
        if stale:
            stale_note += " · ⚠ stale board(s) skipped: " + ", ".join(stale)

    if not (odds_data or matchup_odds or threeball_odds):
        if stale_note:
            # Return an empty priced board (not an error) so callers persist
            # it and overwrite any previously written edge report.
            return {"note": "No fresh odds to price" + stale_note
                    + " — re-run refresh to pull the current board.",
                    "columns": [], "rows": []}
        raise ValueError("No odds. Add golf/data/odds.csv (name, odds_win, "
                         "odds_top5, odds_top10, odds_top20, odds_cut) and/or "
                         "matchups.csv / threeballs.csv.")
    # Record accepted, event-tagged prices for future CLV analysis. One-sided
    # place/cut lines remain raw implied prices; only complete outright boards
    # can be normalized honestly.
    if current_event and odds_data:
        history_boards = {}
        for mkt, col in (("win", "odds_win"), ("top5", "odds_top5"),
                         ("top10", "odds_top10"), ("top20", "odds_top20"),
                         ("cut", "odds_cut")):
            board = {od["name"]: od[col] for od in odds_data.values() if od.get(col)}
            if board:
                history_boards[mkt] = board
        GE.market.snapshot_fair(history_boards, event=current_event)
    bankroll = float(p.get("bankroll", 100.0))
    peak = float(p.get("peak", bankroll))
    kelly = float(p.get("kelly", GE.DEFAULT_KELLY))
    calibrated = bool(p.get("calibrated", True))
    calibration_note = ""
    if state is not None and "calibrated" not in p:
        # Production maps were fit on pre-tournament probabilities. Do not
        # silently apply them to conditioned in-play states without a separate
        # temporal in-play calibration study.
        calibrated = False
        calibration_note = " · in-play calibration disabled"
    elif calibrated and GE.calibrate.load_maps() is None:
        calibrated = False
        calibration_note = " · calibration unavailable (re-run honest validation + calibrate)"
    blended = bool(p.get("market_blend", True))
    min_edge = float(p.get("min_edge", 0.0))

    pairs = [(a, b) for (a, b) in matchup_odds]
    trios = [t for t in threeball_odds]
    n = _sims_arg(p)
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
        cut_rule, no_cut = _simulation_rules(p)
        results = GSIM.simulate_tournament(rated, n_sims=n, cut_rule=cut_rule,
                                           no_cut=no_cut,
                                           rng=np.random.default_rng(0),
                                           matchups=pairs, threeballs=trios)
    rows = GE.price_all(rated, results, odds_data, matchup_odds, threeball_odds,
                        bankroll=bankroll, kelly=kelly, calibrated=calibrated,
                        blended=blended, min_edge=min_edge)
    # Don't stake players with too little history to rate reliably: the model
    # falls back toward a default skill, so an "edge" against the book is
    # spurious — the book is pricing information the model can't see. Keep the
    # rows (probabilities/EV are informative) but never put money on them.
    min_rounds = int(p.get("min_rounds", 60))
    _params = model.load_params() or {}
    _counts = _params.get("players", {})

    def _thin(name: str) -> bool:
        canon = model.resolve_name(name, _params) if hasattr(model, "resolve_name") else name
        return int((_counts.get(canon or "") or {}).get("n_rounds", 0)) < min_rounds

    for r in rows:
        r["thin_sample"] = _thin(r["player"])
    # stake only +EV, well-sampled bets, then apply portfolio discipline
    staked = [r for r in rows if r["ev_per_unit"] > 0 and not r["thin_sample"]]
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
        {"key": "p_final", "label": "Used", "fmt": "pct"},
        {"key": "p_market", "label": "Market", "fmt": "pct"},
        {"key": "ev_per_unit", "label": "EV", "fmt": "signed_num"},
        {"key": "stake_gbp", "label": "Stake", "fmt": "gbp"}]
    note = (f"{len([r for r in rows if r['recommended']])} staked / {len(rows)} "
            f"priced · {GPORT.summary(staked, bankroll, peak)}"
            f"{' · calibrated' if calibrated else ''}"
            f"{' · market-blend' if blended else ''}{calibration_note}"
            f"{inplay_note}{stale_note}")
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
    board_csv = threeballs_csv_path(round_no)
    raw_name = threeballs_raw_path(round_no).name
    quotes = ManualOddsProvider().load_threeballs(event_id=event_id, round_no=round_no)
    if not quotes:
        raise ValueError(
            f"No round-group odds found for round {round_no}. Paste this round's "
            f"tee groups into golf/data/{raw_name} and rerun refresh.")
    # Event-tag staleness guard. The name-overlap check below cannot tell
    # consecutive events apart when their fields overlap (a co-sanctioned week
    # shares most of the tour), so the board must carry the event it was
    # captured for and it must be this one.
    current_event = _current_event_name()
    if current_event:
        tag = board_event(board_csv)
        if norm_event(tag) != norm_event(current_event):
            try:
                GRP.OUT_CSV.unlink(missing_ok=True)
            except OSError:
                pass
            if tag:
                raise ValueError(
                    f"Round-group board is from '{tag}' but the current event is "
                    f"'{current_event}' — stale board. Re-paste this event's tee "
                    f"groups into golf/data/{raw_name} and rerun refresh.")
            raise ValueError(
                "Round-group board has no event tag, so it can't be verified "
                f"against the current event ('{current_event}'). Rerun refresh "
                f"(boards it writes are tagged), or add an 'event' column to "
                f"golf/data/{board_csv.name} with the value '{current_event}'.")
    missing = set(GRP.field_mismatch(quotes, _field_names(), params))
    if missing:
        board_players = {q.player_name for q in quotes
                         if q.market in GRP.ROUND_GROUP_MARKETS}
        frac = len(missing) / max(1, len(board_players))
        # A large mismatch means the board is for the wrong event (stale) — refuse
        # and clear the edges file. A handful of unmatched names is just bookmaker
        # spelling drift (e.g. "Ryan Vools" vs field's "Ryan Voois"): drop only the
        # groups that name them and price the rest.
        if frac > 0.5:
            try:
                GRP.OUT_CSV.unlink(missing_ok=True)
            except OSError:
                try:
                    GRP.OUT_CSV.write_text("")
                except OSError:
                    pass
            raise ValueError(
                f"Round-group board does not match the current field: {len(missing)} "
                f"of {len(board_players)} player(s) not in field.csv (stale board "
                f"from another event?): " + ", ".join(sorted(missing)[:12])
                + ("…" if len(missing) > 12 else "")
                + ". Re-paste this event's tee groups into "
                  f"golf/data/{raw_name} and rerun refresh.")
        bad_groups = {q.group_id for q in quotes if q.player_name in missing}
        quotes = [q for q in quotes if q.group_id not in bad_groups]
        if not quotes:
            raise ValueError("No round groups left after dropping unmatched names.")
    bankroll = float(p.get("bankroll", 100.0))
    rows = GRP.price_round_groups(
        quotes,
        params,
        course=p.get("course", "") or "",
        is_major=bool(p.get("major", False)),
        sims=_sims_arg(p),
        bankroll=bankroll,
        kelly=float(p.get("kelly", GE.DEFAULT_KELLY)),
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

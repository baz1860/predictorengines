#!/usr/bin/env python3
"""Contract + behaviour tests for the tennis engine.

Fully isolated: builds a small synthetic dataset in a private temp directory and
points every tennis module's data path at it for the duration, then fits ATP +
WTA models and exercises the full adapter path (schema / predict / simulate /
edge / settlement) against the shared engine contract. The real tennis/data/ is
never read or written, so even a hard kill mid-run (e.g. `timeout`, SIGKILL, OOM)
cannot corrupt repo data — earlier this test wrote its synthetic fixture straight
to tennis/data/matches.csv and relied on an in-memory restore that a signal-kill
skipped, clobbering the real data.

Also checks the Markov-chain math invariants that don't need fitted data.

Run: python3 test_tennis_contract.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from contracts import (assert_finite_json, validate_edge_rows,
                       validate_prediction, validate_table)

REAL_DATA = ROOT / "tennis" / "data"
# Set to the private temp dir in main(); the fixture writer and all redirected
# tennis modules use this, so the real tennis/data/ is never touched.
DATA = REAL_DATA


def _redirect_tennis_data(tmp: Path) -> None:
    """Point every tennis data path (and the adapter's) at `tmp`. All these
    constants are read at call-time, so reassigning them on the imported modules
    fully redirects file IO for the rest of the process."""
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "api_cache").mkdir(exist_ok=True)
    import tennis.model as M
    import tennis.providers as PV
    import tennis.calibrate as C
    import tennis.validate as V
    import tennis.engine as E
    import tennis.market as MK
    import app.engines.tennis as TA
    M.DATA_DIR = tmp; M.MATCHES_CSV = tmp / "matches.csv"
    PV.DATA_DIR = tmp; PV.MATCHES_CSV = tmp / "matches.csv"; PV.CACHE_DIR = tmp / "api_cache"
    C.DATA_DIR = tmp; C.PRED_CSV = tmp / "validation_predictions.csv"
    C.CALIB_FILE = tmp / "calibration.json"
    V.DATA_DIR = tmp; V.PRED_CSV = tmp / "validation_predictions.csv"
    V.BASELINE_JSON = tmp / "validation_baseline.json"
    E.DATA_DIR = tmp; E.ODDS_CSV = tmp / "odds.csv"; E.DRAW_CSV = tmp / "draw.csv"
    MK.DATA_DIR = tmp; MK.BLEND_JSON = tmp / "market_blend.json"
    MK.ODDS_HISTORY = tmp / "odds_history.csv"
    TA.MATCHES_CSV = tmp / "matches.csv"

_results: list[tuple[str, str, str]] = []


def _check(name: str, fn) -> None:
    try:
        fn()
        _results.append((name, "PASS", ""))
    except Exception as e:  # noqa: BLE001
        _results.append((name, "FAIL", f"{type(e).__name__}: {e}"))


# ── synthetic fixture ─────────────────────────────────────────────────────────

NAMES = [f"Player {i:02d}" for i in range(16)]


def _build_fixture() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    skill = np.linspace(1.6, -1.6, 16)
    clay_spec = {"Player 12": 2.2}        # weak overall, clay monster
    surfaces = ["hard", "clay", "grass"]
    order = np.argsort(-skill).tolist()
    rows = []
    base = pd.Timestamp("2022-01-01")
    for k in range(7000):
        i, j = rng.choice(16, 2, replace=False)
        s = rng.choice(surfaces)
        si = skill[i] + (clay_spec.get(NAMES[i], 0.0) if s == "clay" else 0.0)
        sj = skill[j] + (clay_spec.get(NAMES[j], 0.0) if s == "clay" else 0.0)
        a_wins = rng.random() < 1 / (1 + np.exp(-(si - sj)))
        w, l = (i, j) if a_wins else (j, i)
        d = base + pd.Timedelta(days=int(rng.integers(0, 900)))
        rows.append(dict(date=d.date().isoformat(), tourney_id=f"t{k % 60}",
                         tourney_name="Synthetic Open", tour="atp", surface=s,
                         round="R32", best_of=3, winner=NAMES[w], loser=NAMES[l],
                         winner_rank=order.index(w) + 1, loser_rank=order.index(l) + 1,
                         winner_sets=2, loser_sets=0, score="6-3 6-4"))
    return pd.DataFrame(rows)


def _bracket_fixture() -> pd.DataFrame:
    """~60 synthetic 8-player single-elimination ATP tournaments (QF/SF/F rounds)
    so the outright (draw) backtest has reconstructable brackets."""
    rng = np.random.default_rng(11)
    skill = np.linspace(1.6, -1.6, 16)
    base = pd.Timestamp("2022-01-01")
    rows = []

    def play(i, j):
        pa = 1 / (1 + np.exp(-(skill[i] - skill[j])))
        return (i, j) if rng.random() < pa else (j, i)

    for t in range(60):
        field = list(rng.choice(16, 8, replace=False))
        d = base + pd.Timedelta(days=int(rng.integers(0, 900)))

        def add(w, l, rnd, d=d, t=t):
            rows.append(dict(date=d.date().isoformat(), tourney_id=f"CUP{t}",
                             tourney_name="Synthetic Cup", tour="atp", surface="hard",
                             round=rnd, best_of=3, winner=NAMES[w], loser=NAMES[l],
                             winner_rank=0, loser_rank=0, winner_sets=2, loser_sets=0,
                             score="6-3 6-4"))
        qfw = []
        for k in range(0, 8, 2):
            w, l = play(field[k], field[k + 1]); add(w, l, "QF"); qfw.append(w)
        sfw = []
        for k in range(0, 4, 2):
            w, l = play(qfw[k], qfw[k + 1]); add(w, l, "SF"); sfw.append(w)
        w, l = play(sfw[0], sfw[1]); add(w, l, "F")
    return pd.DataFrame(rows)


def _write_fixture() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    df = _build_fixture()
    atp = pd.concat([df, _bracket_fixture()], ignore_index=True)
    # WTA reuses the flat rows under a different tour tag so both fits exist.
    wta = df.copy(); wta["tour"] = "wta"
    pd.concat([atp, wta], ignore_index=True).to_csv(DATA / "matches.csv", index=False)

    # 8-player draw (4 first-round matches) in bracket order.
    draw = pd.DataFrame([
        dict(tour="atp", tourney_name="Synthetic Open", surface="hard", best_of=3,
             round="QF", player_a=NAMES[a], player_b=NAMES[b])
        for a, b in [(0, 7), (3, 4), (1, 6), (2, 5)]])
    draw.to_csv(DATA / "draw.csv", index=False)

    odds = pd.DataFrame([
        dict(tour="atp", surface="hard", best_of=3,
             player_a=NAMES[0], player_b=NAMES[15], odds_a=1.10, odds_b=8.0),
        dict(tour="atp", surface="clay", best_of=3,
             player_a=NAMES[12], player_b=NAMES[0], odds_a=3.5, odds_b=1.30)])
    odds.to_csv(DATA / "odds.csv", index=False)


# ── behaviour checks ──────────────────────────────────────────────────────────

def _markov_invariants() -> None:
    from tennis import simulate as S
    assert abs(S.match_win_prob(0.64, 0.64, 3) - 0.5) < 1e-9, "equal serves ≠ 0.5"
    assert abs(S.prob_win_tiebreak(0.6, 0.6) - 0.5) < 1e-9, "equal tiebreak ≠ 0.5"
    # stronger player rewarded more over best-of-5 than best-of-3
    p3 = S.match_win_prob(0.68, 0.63, 3)
    p5 = S.match_win_prob(0.68, 0.63, 5)
    assert 0.5 < p3 < p5, f"bo5 should favour the stronger player more: {p3} {p5}"
    # inversion reproduces the target
    ps = S.point_edge_for_target(0.72, 3)
    assert abs(S.match_win_prob(*ps, 3) - 0.72) < 1e-3, "edge inversion off"


def _model_behaviour() -> None:
    from tennis import model as M
    params = M.load_params("atp")
    assert params and params["n_players"] == 16, "fit did not produce 16 players"
    from scipy.stats import spearmanr
    true = np.linspace(1.6, -1.6, 16)
    fit = np.array([params["skills"][n] for n in NAMES])
    rho = spearmanr(fit, true).statistic
    assert rho > 0.7, f"fitted skill correlation too low: {rho:.3f}"
    # clay specialist: meaningfully closer / better on clay than on hard
    p_clay = M.predict_match("Player 12", "Player 00", "clay", params)["p_a"]
    p_hard = M.predict_match("Player 12", "Player 00", "hard", params)["p_a"]
    assert p_clay > p_hard, f"clay specialism not captured: clay {p_clay} hard {p_hard}"


def _adapter_contract() -> None:
    from app.engines import registry
    ad = registry.get("tennis")
    info = ad.info()
    assert info["id"] == "tennis" and info["capabilities"], "bad info()"
    assert_finite_json(info)

    pred = ad.predict({"player_a": NAMES[0], "player_b": NAMES[15],
                       "surface": "hard", "tour": "atp"})
    validate_prediction(pred)
    assert pred["outcomes"][0]["prob"] > 0.5, "stronger player should be favoured"

    sim = ad.simulate({"tour": "atp", "sims": 2000, "seed": 1})
    validate_table(sim)
    assert sim["rows"][0]["win"] >= sim["rows"][-1]["win"], "sim not sorted by win"
    assert abs(sum(r["win"] for r in sim["rows"]) - 1.0) < 0.05, "win probs ≠ 1"

    edge = ad.edge({"tour": "atp", "record": False})
    assert_finite_json(edge)
    validate_edge_rows(edge.get("rows") or [])


def _validation_and_calibration() -> None:
    from tennis import calibrate as C
    from tennis import model as M
    from tennis import validate as V
    df = M.load_matches_df()
    pred = V.walk_forward(df, "atp", since="2023-01-01", retrain_days=120,
                          verbose=False)
    assert len(pred) > 200, f"too few walk-forward predictions: {len(pred)}"
    rep = V.summarize(pred)
    # the model should beat the base rate (skill is measured vs the realised
    # base rate, so this holds even though synthetic names correlate with skill)
    assert rep["match_winner"]["skill"] > 0.05, \
        f"no walk-forward skill: {rep['match_winner']['skill']}"

    pred.to_csv(V.PRED_CSV, index=False)
    maps = C.fit_from_csv()
    assert "match_winner" in maps, "calibration did not fit match_winner"
    for p in (0.05, 0.4, 0.6, 0.95):
        pc = C.apply_one("match_winner", p, maps)
        assert 0.0 <= pc <= 1.0, f"calibrated prob out of range: {pc}"
    # isotonic maps are monotone non-decreasing
    lo = C.apply_one("match_winner", 0.2, maps)
    hi = C.apply_one("match_winner", 0.8, maps)
    assert hi >= lo - 1e-9, f"calibration not monotone: {lo} -> {hi}"


def _market_and_portfolio() -> None:
    from tennis import market as MK
    from tennis import portfolio as PORT
    pa, pb = MK.devig_two_way(1.80, 2.10)
    assert abs(pa + pb - 1.0) < 1e-9 and pa > pb, "two-way de-vig broken"
    # log-odds blend sits between model and market and is monotone in the model
    assert MK.blend(0.6, 0.5, 0.5) < 0.6, "blend should pull toward market"
    assert MK.blend(0.7, 0.5, 0.5) > MK.blend(0.6, 0.5, 0.5), "blend not monotone"
    # per-player cap: one player's total exposure capped at 10% of bankroll
    rows = [{"player": "A", "p_model": 0.7, "stake_gbp": 7.0},
            {"player": "A", "p_model": 0.4, "stake_gbp": 6.0},
            {"player": "B", "p_model": 0.6, "stake_gbp": 5.0}]
    out = PORT.apply_portfolio([dict(r) for r in rows], bankroll=100.0, peak=100.0)
    a_total = sum(r["stake_gbp"] for r in out if r["player"] == "A")
    assert a_total <= 10.0 + 1e-6, f"per-player cap not enforced: {a_total}"


def _outright_backtest() -> None:
    from tennis import calibrate as C
    from tennis import model as M
    from tennis import simulate as S
    from tennis import validate as V
    df = M.load_matches_df()
    # bracket reconstruction + a single sim sums to one champion
    ev = df[df["tourney_id"] == "CUP0"]
    root = V.reconstruct_bracket(ev)
    assert root is not None and len(S._bracket_leaves(root)) == 8, "bad bracket"
    params = M.load_params("atp")
    res = S.simulate_bracket(root, params, "hard", n_sims=1500,
                             rng=np.random.default_rng(2))
    assert abs(sum(v["win"] for v in res.values()) - 1.0) < 1e-9, "win probs ≠ 1"

    outp = V.walk_forward_outright(df, "atp", since="2022-06-01", sims=800,
                                   verbose=False)
    assert len(outp) > 100, f"too few outright rows: {len(outp)}"
    rep = V.summarize(outp)
    assert rep["win"]["skill"] > 0.02, f"no outright win skill: {rep['win']['skill']}"

    # outright calibration fits and the nesting guard keeps win ≤ final ≤ sf ≤ qf
    maps = C.fit_maps(outp)
    assert "win" in maps, "outright calibration did not fit win"
    cal = C.apply_outright({"win": 0.5, "final": 0.2, "sf": 0.9, "qf": 0.1}, maps)
    assert cal["win"] <= cal["final"] + 1e-9 <= cal["sf"] + 1e-9 <= cal["qf"] + 1e-9, \
        f"nesting guard violated: {cal}"


def _edge_blend_portfolio() -> None:
    from app.engines import registry
    ad = registry.get("tennis")
    edge = ad.edge({"tour": "atp", "record": False})
    assert_finite_json(edge)
    validate_edge_rows(edge.get("rows") or [])
    assert "market-blend" in edge["note"], "market blend not applied in edge"
    assert any("p_blend" in r for r in edge["rows"]), "edge rows missing p_blend"


def _calibrated_predict() -> None:
    # calibration.json now exists → predict should apply it and say so
    from app.engines import registry
    ad = registry.get("tennis")
    pred = ad.predict({"player_a": NAMES[0], "player_b": NAMES[15],
                       "surface": "hard", "tour": "atp"})
    validate_prediction(pred)
    assert "calibrated" in pred["note"], "calibration not applied in predict"


def _settlement() -> None:
    from app.engines import registry
    ad = registry.get("tennis")
    # Player 00 beat Player 15 somewhere in the fixture → a back-Player-00 bet wins.
    rows = pd.DataFrame([
        {"home": NAMES[0], "away": NAMES[15], "market": "match_winner",
         "side": "win", "match_date": "2021-01-01"},
        {"home": NAMES[15], "away": NAMES[0], "market": "match_winner",
         "side": "win", "match_date": "2021-01-01"}])
    graded = ad.grade_open_bets(rows)
    assert graded.get(0, ("",))[0] == "won", f"expected win, got {graded.get(0)}"
    assert graded.get(1, ("",))[0] == "lost", f"expected loss, got {graded.get(1)}"


def main() -> int:
    global DATA
    # Fingerprint the real source-of-truth file so we can prove the test never
    # touched it, regardless of how the run ends.
    real_matches = REAL_DATA / "matches.csv"
    real_before = real_matches.read_bytes() if real_matches.exists() else None

    tmp = Path(tempfile.mkdtemp(prefix="tennis_contract_"))
    DATA = tmp
    _redirect_tennis_data(tmp)
    try:
        _check("markov_invariants", _markov_invariants)
        _write_fixture()
        from tennis import model as M
        df = M.load_matches_df()
        for tour in ("atp", "wta"):
            M.save_params(M.fit(df, tour=tour), tour=tour)
        _check("model_behaviour", _model_behaviour)
        _check("adapter_contract", _adapter_contract)
        _check("settlement", _settlement)
        _check("market+portfolio", _market_and_portfolio)
        _check("outright_backtest", _outright_backtest)
        _check("edge_blend_portfolio", _edge_blend_portfolio)
        _check("validation+calibration", _validation_and_calibration)
        _check("calibrated_predict", _calibrated_predict)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # Hard safety net: the real data must be byte-identical to before the run.
    real_after = real_matches.read_bytes() if real_matches.exists() else None
    if real_before != real_after:
        print("FATAL: test modified the real tennis/data/matches.csv (isolation broken)")
        return 2

    width = max(len(n) for n, *_ in _results)
    print(f"{'CHECK'.ljust(width)}  STATUS  DETAIL")
    print("-" * (width + 24))
    fails = 0
    for name, status, detail in _results:
        if status == "FAIL":
            fails += 1
        print(f"{name.ljust(width)}  {status:<6}  {detail}")
    print("-" * (width + 24))
    print(f"{len(_results) - fails} pass · {fails} fail")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

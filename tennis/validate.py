"""tennis/validate.py — walk-forward backtest + regression gate (the yardstick).

Refit the model on matches STRICTLY before a retrain date, then predict every
match in the window that follows — no look-ahead. Each match is oriented
neutrally (players sorted by folded name, independent of who won) so the
match-winner calibration set isn't trivially biased toward p≈1.

Markets scored straight off completed matches:
  * match_winner — A beats B (binary)
  * set_hcp      — A wins in straight sets (covers −1.5 sets)
  * first_set    — A wins the opening set (parsed from the score string)

Outright markets (win/final/SF/QF) need a bracket-reconstruction backtest and
are a documented follow-up; calibrate.py already supports them once present.

Outputs:
  data/validation_predictions.csv   (feeds calibrate.py)
  data/validation_baseline.json     (headline Brier for --gate)

Usage:
  python -m tennis.validate [--tour atp|wta|both] [--since 2023-01-01]
                            [--retrain-days 28] [--gate]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import model as M
from . import simulate as S

DATA_DIR = Path(__file__).parent / "data"
PRED_CSV = DATA_DIR / "validation_predictions.csv"
BASELINE_JSON = DATA_DIR / "validation_baseline.json"

MATCH_MARKETS = ["match_winner", "set_hcp", "first_set"]
OUTRIGHT_MARKETS = ["win", "final", "sf", "qf"]
ALL_MARKETS = MATCH_MARKETS + OUTRIGHT_MARKETS
GATE_TOL = 0.005          # allowed headline (match_winner) Brier regression
EPS = 1e-12

# Knockout round labels, shallow (final) → deep (first round).
ROUND_SEQ = ["F", "SF", "QF", "R16", "R32", "R64", "R128", "R256"]
_ROUND_RANK = {r: i for i, r in enumerate(ROUND_SEQ)}


# ─────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────

def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability(p: np.ndarray, y: np.ndarray, bins=10) -> list[tuple]:
    edges = np.linspace(0, 1, bins + 1)
    out = []
    idx = np.digitize(p, edges[1:-1])
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        out.append((round(float(p[m].mean()), 3), round(float(y[m].mean()), 3), int(m.sum())))
    return out


def _first_set_winner(score: str, winner: str, loser: str) -> str | None:
    """Name of the player who won the opening set, parsed from the score string
    (games are listed winner-first per set). None if unparseable (walkover, etc.)."""
    s = str(score or "").strip()
    if not s:
        return None
    first = s.split()[0].split("(", 1)[0]
    if "-" not in first:
        return None
    a, _, b = first.partition("-")
    try:
        gw, gl = int(a), int(b)
    except ValueError:
        return None
    if gw == gl:
        return None
    return winner if gw > gl else loser


# ─────────────────────────────────────────────
# Walk-forward loop
# ─────────────────────────────────────────────

def walk_forward(df: pd.DataFrame, tour: str, since: str, retrain_days: int = 28,
                 config: dict | None = None, verbose: bool = True) -> pd.DataFrame:
    sub = df[df["tour"].astype(str).str.lower() == tour].sort_values("date")
    since_ts = pd.Timestamp(since)
    rows = []
    params = None
    last_fit = None
    n_refits = 0

    for r in sub.itertuples():
        date = pd.Timestamp(r.date)
        if date < since_ts:
            continue
        if params is None or last_fit is None or (date - last_fit).days >= retrain_days:
            try:
                params = M.fit(df, tour=tour, asof=date, config=config)
                last_fit = date
                n_refits += 1
                if verbose:
                    print(f"  refit {tour} asof {date.date()}  "
                          f"({params['n_matches']:,} matches, {params['n_players']} players)")
            except ValueError:
                continue
        winner, loser = str(r.winner), str(r.loser)
        surface = str(r.surface).lower()
        best_of = int(r.best_of) if not pd.isna(r.best_of) else 3
        a, b = sorted([winner, loser], key=M.fold_name)
        p_a = M.predict_match(a, b, surface, params)["p_a"]
        mk = S.match_markets(p_a, best_of=best_of)
        a_won = int(a == winner)
        loser_sets = int(r.loser_sets) if not pd.isna(r.loser_sets) else -1

        # set_hcp: A covers −1.5 sets ⇔ A wins in straight sets
        y_set = (1 if (a_won and loser_sets == 0) else 0) if loser_sets >= 0 else np.nan
        # first_set
        fsw = _first_set_winner(r.score, winner, loser)
        y_first = (1 if fsw == a else 0) if fsw is not None else np.nan

        rows.append({
            "tour": tour, "date": str(date.date()), "tourney_id": str(r.tourney_id),
            "player_a": a, "player_b": b,
            "p_match_winner": p_a, "y_match_winner": a_won,
            "p_set_hcp": mk["p_a_minus_1_5_sets"], "y_set_hcp": y_set,
            "p_first_set": mk["p_first_set"], "y_first_set": y_first,
        })
    if verbose:
        print(f"  {tour}: {len(rows):,} match predictions over {n_refits} refits")
    return pd.DataFrame(rows)


def summarize(pred: pd.DataFrame) -> dict:
    report = {}
    for mkt in ALL_MARKETS:
        pcol, ycol = f"p_{mkt}", f"y_{mkt}"
        if pcol not in pred.columns:
            continue
        sub = pred[[pcol, ycol]].dropna()
        if sub.empty:
            continue
        p = sub[pcol].to_numpy(dtype=float)
        y = sub[ycol].to_numpy(dtype=float)
        base = float(y.mean())
        b = brier(p, y)
        b_base = brier(np.full_like(y, base), y)
        report[mkt] = {
            "n": int(len(y)), "base_rate": round(base, 4),
            "brier": round(b, 5), "brier_base": round(b_base, 5),
            "skill": round(1 - b / b_base, 4) if b_base > 0 else 0.0,
            "logloss": round(logloss(p, y), 5),
            "reliability": reliability(p, y),
        }
    report["headline_brier"] = report.get("match_winner", {}).get("brier", 1.0)
    return report


def print_report(rep: dict) -> None:
    print(f"\n{'Market':<14}{'N':>8}{'base':>8}{'Brier':>9}{'vs base':>9}"
          f"{'skill':>8}{'logloss':>9}")
    print("-" * 64)
    for mkt in ALL_MARKETS:
        if mkt not in rep:
            continue
        r = rep[mkt]
        print(f"{mkt:<14}{r['n']:>8}{r['base_rate']:>8.3f}{r['brier']:>9.4f}"
              f"{r['brier_base']:>9.4f}{r['skill']:>8.1%}{r['logloss']:>9.4f}")
    print(f"\nHeadline Brier (match_winner): {rep['headline_brier']:.5f}")
    if "match_winner" in rep:
        print("\nMatch-winner reliability  (pred → actual, n):")
        for pp, yy, nn in rep["match_winner"]["reliability"]:
            bar = "█" * int(yy * 30)
            print(f"  {pp:>5.2f} → {yy:>5.2f}  {bar}  ({nn})")


# ─────────────────────────────────────────────
# Outright (draw) backtest — bracket reconstruction + Monte-Carlo
# ─────────────────────────────────────────────

def reconstruct_bracket(matches: pd.DataFrame):
    """Rebuild a single-elimination tournament tree from its completed matches.

    Each shallower-round match's two players each won a match in the next deeper
    round; linking those recursively from the final rebuilds the full bracket
    (players who entered via a bye/first round become leaves). Returns the root
    node dict, or None for non-knockout / irregular events (round-robin, no
    single final, fewer than 8 entrants).
    """
    rounds = set(matches["round"].astype(str))
    if "RR" in rounds:
        return None
    finals = matches[matches["round"].astype(str) == "F"]
    if len(finals) != 1:
        return None

    won_at: dict[str, dict[str, pd.Series]] = {}
    for _, m in matches.iterrows():
        r = str(m["round"])
        if r in _ROUND_RANK:
            won_at.setdefault(r, {})[str(m["winner"])] = m

    def build(match: pd.Series):
        r = str(match["round"])
        di = _ROUND_RANK[r]
        deeper = ROUND_SEQ[di + 1] if di + 1 < len(ROUND_SEQ) else None
        kids = []
        for player in (str(match["winner"]), str(match["loser"])):
            child = won_at.get(deeper, {}).get(player) if deeper else None
            kids.append(build(child) if child is not None else player)
        return {"round": r, "a": kids[0], "b": kids[1]}

    root = build(finals.iloc[0])
    if len(S._bracket_leaves(root)) < 8:
        return None
    return root


def _outright_actuals(matches: pd.DataFrame) -> dict[str, dict]:
    """Per-player actual reach (win/final/sf/qf) for one tournament, from the
    shallowest knockout round each player appears in."""
    best: dict[str, int] = {}
    for _, m in matches.iterrows():
        r = str(m["round"])
        if r not in _ROUND_RANK:
            continue
        rank = _ROUND_RANK[r]
        for player in (str(m["winner"]), str(m["loser"])):
            if rank < best.get(player, 99):
                best[player] = rank
    champ = None
    finals = matches[matches["round"].astype(str) == "F"]
    if len(finals) == 1:
        champ = str(finals.iloc[0]["winner"])
    out = {}
    for player, rk in best.items():
        out[player] = {
            "win": int(player == champ),
            "final": int(rk <= _ROUND_RANK["F"]),
            "sf": int(rk <= _ROUND_RANK["SF"]),
            "qf": int(rk <= _ROUND_RANK["QF"]),
        }
    return out


def walk_forward_outright(df: pd.DataFrame, tour: str, since: str, sims: int = 10000,
                          config: dict | None = None, seed: int = 0,
                          verbose: bool = True) -> pd.DataFrame:
    sub = df[df["tour"].astype(str).str.lower() == tour]
    since_ts = pd.Timestamp(since)
    rng = np.random.default_rng(seed)
    rows = []
    n_events = 0
    for tid, ev in sub.groupby("tourney_id"):
        start = pd.Timestamp(ev["date"].min())
        if start < since_ts:
            continue
        root = reconstruct_bracket(ev)
        if root is None:
            continue
        try:
            params = M.fit(df, tour=tour, asof=start, config=config)
        except ValueError:
            continue
        surface = str(ev["surface"].mode().iloc[0]).lower()
        best_of = int(ev["best_of"].mode().iloc[0]) if not ev["best_of"].mode().empty else 3
        res = S.simulate_bracket(root, params, surface, best_of=best_of,
                                 n_sims=sims, rng=rng)
        actual = _outright_actuals(ev)
        for player, pr in res.items():
            a = actual.get(player)
            if a is None:
                continue
            rows.append({
                "tour": tour, "date": str(start.date()), "tourney_id": str(tid),
                "player_a": player, "player_b": "",
                "p_win": pr["win"], "y_win": a["win"],
                "p_final": pr["final"], "y_final": a["final"],
                "p_sf": pr["sf"], "y_sf": a["sf"],
                "p_qf": pr["qf"], "y_qf": a["qf"],
            })
        n_events += 1
        if verbose:
            print(f"  outright {tour} {str(start.date())} {tid}  "
                  f"{len(S._bracket_leaves(root))} draw")
    if verbose:
        print(f"  {tour}: {len(rows):,} player-event outright rows over {n_events} events")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Walk-forward tennis backtest + gate")
    ap.add_argument("--tour", default="both", choices=["atp", "wta", "both"])
    ap.add_argument("--since", default="2023-01-01",
                    help="Evaluate matches on/after this date (default %(default)s)")
    ap.add_argument("--retrain-days", type=int, default=28,
                    help="Refit cadence in days (default %(default)s)")
    ap.add_argument("--outright", action="store_true",
                    help="Also backtest outright markets (win/final/SF/QF) by "
                         "reconstructing each tournament bracket and simulating it")
    ap.add_argument("--sims", type=int, default=10000,
                    help="Sims per tournament for the outright backtest")
    ap.add_argument("--gate", action="store_true",
                    help="Exit non-zero if headline Brier regresses vs baseline")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    df = M.load_matches_df()
    tours = ["atp", "wta"] if args.tour == "both" else [args.tour]
    print(f"Walk-forward from {args.since}  (refit every {args.retrain_days}d)…")
    frames = [walk_forward(df, t, since=args.since, retrain_days=args.retrain_days,
                           verbose=not args.quiet) for t in tours]
    if args.outright:
        print("Outright (draw) backtest…")
        frames += [walk_forward_outright(df, t, since=args.since, sims=args.sims,
                                         verbose=not args.quiet) for t in tours]
    pred = pd.concat([f for f in frames if not f.empty], ignore_index=True) \
        if any(not f.empty for f in frames) else pd.DataFrame()
    if pred.empty:
        print("No evaluable matches — seed more history or move --since earlier.")
        sys.exit(1)
    pred.to_csv(PRED_CSV, index=False)
    print(f"\n{len(pred):,} match predictions → {PRED_CSV}")

    rep = summarize(pred)
    print_report(rep)

    head = rep["headline_brier"]
    if BASELINE_JSON.exists():
        baseline = json.loads(BASELINE_JSON.read_text())
        prev = baseline.get("headline_brier", head)
        delta = head - prev
        print(f"\nBaseline headline Brier {prev:.5f}  →  now {head:.5f}  "
              f"(Δ {delta:+.5f}, tol {GATE_TOL})")
        if args.gate and delta > GATE_TOL:
            print("GATE FAIL: model regressed beyond tolerance.")
            sys.exit(2)
        if delta < -GATE_TOL:
            BASELINE_JSON.write_text(json.dumps(
                {"headline_brier": head, "gate_tol": GATE_TOL,
                 "asof": pred["date"].max()}, indent=1))
            print("Improved — baseline updated.")
    else:
        BASELINE_JSON.write_text(json.dumps(
            {"headline_brier": head, "gate_tol": GATE_TOL,
             "asof": pred["date"].max()}, indent=1))
        print(f"\nBaseline written → {BASELINE_JSON}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Backtest the NHL model against completed games.

The required CSV columns are:
  date, home, away, home_goals, away_goals

Optional historical odds columns enable one-unit betting ROI:
  odds_home/odds_away              moneyline
  total_line, odds_over, odds_under
  home_spread_line, odds_home_spread, odds_away_spread

Run:
  python3 -m nhl.backtest
  python3 -m nhl.backtest --results nhl/data/results.csv --model blend --min-edge 0.03
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd

from . import edge as E
from . import model as M

RESULTS_CSV = Path(__file__).resolve().parent / "data" / "results.csv"
REQUIRED = {"date", "home", "away", "home_goals", "away_goals"}

ODDS_ALIASES = {
    "home_ml": ("odds_home", "home_odds", "home_ml_odds", "ml_home"),
    "away_ml": ("odds_away", "away_odds", "away_ml_odds", "ml_away"),
    "total_line": ("total_line", "ou_line"),
    "over": ("odds_over", "over_odds", "over_price"),
    "under": ("odds_under", "under_odds", "under_price"),
    "home_spread_line": ("home_spread_line", "spread_home_line", "puck_line_home"),
    "home_spread": ("odds_home_spread", "home_spread_odds", "puck_line_home_odds"),
    "away_spread": ("odds_away_spread", "away_spread_odds", "puck_line_away_odds"),
}


def _clip_prob(p: float) -> float:
    return min(max(float(p), 1e-6), 1.0 - 1e-6)


def _logloss(p: float, y: int) -> float:
    p = _clip_prob(p)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def _num(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _first_number(row: pd.Series, aliases: tuple[str, ...]) -> float | None:
    for name in aliases:
        if name in row.index:
            value = _num(row[name])
            if value is not None:
                return value
    return None


def load_results(path: str | Path = RESULTS_CSV) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NHL results file not found: {path}")
    df = pd.read_csv(path)
    missing = sorted(REQUIRED - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    for col in ("home_goals", "away_goals"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["home_goals", "away_goals"]).copy()
    if df.empty:
        raise ValueError(f"{path} has no completed NHL games")
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    return df.sort_values("date").reset_index(drop=True)


def _grade(market: str, side: str, line: float | None,
           home_goals: int, away_goals: int) -> str:
    margin = home_goals - away_goals
    total = home_goals + away_goals
    if market == "ml":
        won = margin > 0 if side == "home" else margin < 0
        return "won" if won else "lost"
    if market == "spread" and line is not None:
        adj = margin + line if side == "home" else -margin + line
        if adj == 0:
            return "push"
        return "won" if adj > 0 else "lost"
    if market == "total" and line is not None:
        if total == line:
            return "push"
        won = total > line if side == "over" else total < line
        return "won" if won else "lost"
    return "skip"


def _book_probs(odds: list[float]) -> list[float]:
    inv = [1.0 / o for o in odds if o and o > 1.0]
    s = sum(inv)
    return [x / s for x in inv] if s > 0 else []


def _add_bet(candidates: list[dict[str, Any]], *, game: pd.Series, pred: dict,
             market: str, side: str, line: float | None, odds: float,
             p_book: float, min_edge: float) -> None:
    if odds <= 1.0:
        return
    p_model, p_push = M.market_probs(pred, market, side, line)
    edge = p_model - p_book
    ev = p_model * odds + p_push - 1.0
    if edge < min_edge or ev <= 0:
        return
    status = _grade(market, side, line, int(game["home_goals"]), int(game["away_goals"]))
    if status == "skip":
        return
    pnl = odds - 1.0 if status == "won" else (0.0 if status == "push" else -1.0)
    candidates.append({
        "date": str(game["date"]),
        "home": str(game["home"]),
        "away": str(game["away"]),
        "market": market,
        "side": side,
        "line": "" if line is None else E._fmt_line(line, market),
        "odds": round(float(odds), 3),
        "p_model": round(float(p_model), 4),
        "p_book": round(float(p_book), 4),
        "edge": round(float(edge), 4),
        "ev_per_unit": round(float(ev), 4),
        "status": status,
        "pnl": round(float(pnl), 4),
    })


def _bet_candidates(game: pd.Series, pred: dict, min_edge: float) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    home_ml = _first_number(game, ODDS_ALIASES["home_ml"])
    away_ml = _first_number(game, ODDS_ALIASES["away_ml"])
    if home_ml and home_ml > 1.0 and away_ml and away_ml > 1.0:
        ph_book, pa_book = _book_probs([home_ml, away_ml])
        _add_bet(candidates, game=game, pred=pred, market="ml", side="home",
                 line=None, odds=home_ml, p_book=ph_book, min_edge=min_edge)
        _add_bet(candidates, game=game, pred=pred, market="ml", side="away",
                 line=None, odds=away_ml, p_book=pa_book, min_edge=min_edge)

    total_line = _first_number(game, ODDS_ALIASES["total_line"])
    over = _first_number(game, ODDS_ALIASES["over"])
    under = _first_number(game, ODDS_ALIASES["under"])
    if total_line is not None and over and over > 1.0 and under and under > 1.0:
        po_book, pu_book = _book_probs([over, under])
        _add_bet(candidates, game=game, pred=pred, market="total", side="over",
                 line=total_line, odds=over, p_book=po_book, min_edge=min_edge)
        _add_bet(candidates, game=game, pred=pred, market="total", side="under",
                 line=total_line, odds=under, p_book=pu_book, min_edge=min_edge)

    home_spread_line = _first_number(game, ODDS_ALIASES["home_spread_line"])
    home_spread = _first_number(game, ODDS_ALIASES["home_spread"])
    away_spread = _first_number(game, ODDS_ALIASES["away_spread"])
    if home_spread_line is not None and home_spread and home_spread > 1.0 \
            and away_spread and away_spread > 1.0:
        ps_home, ps_away = _book_probs([home_spread, away_spread])
        _add_bet(candidates, game=game, pred=pred, market="spread", side="home",
                 line=home_spread_line, odds=home_spread, p_book=ps_home,
                 min_edge=min_edge)
        _add_bet(candidates, game=game, pred=pred, market="spread", side="away",
                 line=-home_spread_line, odds=away_spread, p_book=ps_away,
                 min_edge=min_edge)

    best_by_market: dict[tuple[str, str], dict[str, Any]] = {}
    for bet in candidates:
        key = (bet["market"], str(bet["line"]))
        if key not in best_by_market or float(bet["edge"]) > float(best_by_market[key]["edge"]):
            best_by_market[key] = bet
    return sorted(best_by_market.values(), key=lambda b: -float(b["edge"]))


def run_backtest(results: pd.DataFrame | None = None, *, model: str = "blend",
                 min_edge: float = 0.03) -> dict[str, Any]:
    games = load_results() if results is None else results.copy()
    rows: list[dict[str, Any]] = []
    bets: list[dict[str, Any]] = []
    for game in games.itertuples(index=False):
        g = pd.Series(game._asdict())
        try:
            pred = M.predict_match(str(g["home"]), str(g["away"]), model=model)
        except ValueError:
            continue
        home_goals = int(g["home_goals"])
        away_goals = int(g["away_goals"])
        y = 1 if home_goals > away_goals else 0
        p_home = float(pred["p_home"])
        pred_margin = float(pred["lambda_home"] - pred["lambda_away"])
        actual_margin = home_goals - away_goals
        pred_total = float(pred["total"])
        actual_total = home_goals + away_goals
        rows.append({
            "date": str(g["date"]),
            "home": str(g["home"]),
            "away": str(g["away"]),
            "score": f"{home_goals}-{away_goals}",
            "p_home": round(p_home, 4),
            "pick": str(g["home"]) if p_home >= 0.5 else str(g["away"]),
            "correct": (p_home >= 0.5) == bool(y),
            "lambda_home": round(float(pred["lambda_home"]), 3),
            "lambda_away": round(float(pred["lambda_away"]), 3),
            "pred_margin": round(pred_margin, 3),
            "actual_margin": actual_margin,
            "pred_total": round(pred_total, 3),
            "actual_total": actual_total,
            "brier": round((p_home - y) ** 2, 5),
            "logloss": round(_logloss(p_home, y), 5),
            "abs_margin_error": round(abs(pred_margin - actual_margin), 3),
            "abs_total_error": round(abs(pred_total - actual_total), 3),
        })
        bets.extend(_bet_candidates(g, pred, min_edge))

    if not rows:
        raise ValueError("No backtestable games after filtering unknown teams")
    out = pd.DataFrame(rows)
    summary = {
        "games": int(len(out)),
        "model": model,
        "accuracy": round(float(out["correct"].mean()), 4),
        "home_win_rate": round(float((out["actual_margin"] > 0).mean()), 4),
        "avg_home_prob": round(float(out["p_home"].mean()), 4),
        "brier": round(float(out["brier"].mean()), 5),
        "logloss": round(float(out["logloss"].mean()), 5),
        "margin_mae": round(float(out["abs_margin_error"].mean()), 3),
        "total_mae": round(float(out["abs_total_error"].mean()), 3),
    }
    betting = _summarize_bets(bets)
    return {"summary": summary, "rows": rows, "bets": bets, "betting": betting}


def _summarize_bets(bets: list[dict[str, Any]]) -> dict[str, Any]:
    if not bets:
        return {"bets": 0, "staked": 0.0, "pnl": 0.0, "roi": None, "win_rate": None}
    df = pd.DataFrame(bets)
    staked = float((df["status"] != "push").sum())
    pnl = float(df["pnl"].sum())
    decisions = df[df["status"] != "push"]
    return {
        "bets": int(len(df)),
        "staked": round(staked, 2),
        "won": int((df["status"] == "won").sum()),
        "lost": int((df["status"] == "lost").sum()),
        "push": int((df["status"] == "push").sum()),
        "pnl": round(pnl, 4),
        "roi": round(pnl / staked, 4) if staked else None,
        "win_rate": round(float((decisions["status"] == "won").mean()), 4) if not decisions.empty else None,
    }


def _print_report(report: dict[str, Any], show_bets: int) -> None:
    s = report["summary"]
    print(f"NHL backtest ({s['model']}) · {s['games']} games")
    print(f"  accuracy {s['accuracy']:.1%} · Brier {s['brier']:.4f} · log-loss {s['logloss']:.4f}")
    print(f"  margin MAE {s['margin_mae']:.2f} · total MAE {s['total_mae']:.2f}")
    b = report["betting"]
    if b["bets"]:
        roi = "n/a" if b["roi"] is None else f"{b['roi']:.1%}"
        print(f"  betting: {b['bets']} bet(s), {b['won']}-{b['lost']}-{b['push']}, "
              f"PnL {b['pnl']:+.2f}u, ROI {roi}")
        if show_bets:
            print()
            print(pd.DataFrame(report["bets"]).head(show_bets).to_string(index=False))
    else:
        print("  betting: no historical odds columns or no bets cleared the threshold")


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest NHL predictions against completed games")
    ap.add_argument("--results", default=str(RESULTS_CSV),
                    help="CSV with date, home, away, home_goals, away_goals")
    ap.add_argument("--model", choices=["blend", "power", "form"], default="blend")
    ap.add_argument("--min-edge", type=float, default=0.03,
                    help="minimum de-vig edge for optional odds-backed bets")
    ap.add_argument("--show-bets", type=int, default=10,
                    help="print this many historical bets when odds are present")
    args = ap.parse_args()

    report = run_backtest(load_results(args.results), model=args.model, min_edge=args.min_edge)
    _print_report(report, args.show_bets)


if __name__ == "__main__":
    main()

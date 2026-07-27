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
  python3 -m nhl.backtest --results nhl/data/results_2025_26.csv --odds-history nhl/data/odds_history.csv
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd

from . import edge as E
from . import model as M
from . import odds_history as OH

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

DEFAULT_PRIOR_GAMES = 20.0
STAT_COLUMNS = [
    "games", "goals_for", "goals_against", "shots_for", "shots_against",
    "power_play_pct", "penalty_kill_pct", "save_pct", "point_pct",
]
LEAGUE_PRIOR = {
    "goals_for_pg": 3.05,
    "goals_against_pg": 3.05,
    "shots_for_pg": 30.0,
    "shots_against_pg": 30.0,
    "power_play_pct": 0.20,
    "penalty_kill_pct": 0.80,
    "save_pct": 0.900,
    "point_pct": 0.500,
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


def _canonical_name(team: Any) -> str:
    text = str(team)
    try:
        return M.canonical_team_name(text)
    except ValueError:
        return text


def _league_seed_values() -> dict[str, float]:
    return dict(LEAGUE_PRIOR)


def _season_teams(games: pd.DataFrame) -> list[str]:
    teams = set(M.team_names())
    for col in ("home", "away"):
        teams.update(_canonical_name(t) for t in games[col].dropna().astype(str))
    return sorted(teams)


def _initial_walkforward_stats(games: pd.DataFrame,
                               prior_games: float = DEFAULT_PRIOR_GAMES) -> pd.DataFrame:
    if prior_games <= 0:
        raise ValueError("walk-forward prior_games must be > 0")
    seed = _league_seed_values()
    rows = []
    for team in _season_teams(games):
        rows.append({
            "team": team,
            "games": float(prior_games),
            "goals_for": seed["goals_for_pg"] * prior_games,
            "goals_against": seed["goals_against_pg"] * prior_games,
            "shots_for": seed["shots_for_pg"] * prior_games,
            "shots_against": seed["shots_against_pg"] * prior_games,
            "power_play_pct": seed["power_play_pct"],
            "penalty_kill_pct": seed["penalty_kill_pct"],
            "save_pct": seed["save_pct"],
            "point_pct": seed["point_pct"],
        })
    return pd.DataFrame(rows).set_index("team")


def _ensure_team(stats: pd.DataFrame, team: str, seed: dict[str, float],
                 prior_games: float) -> None:
    if team in stats.index:
        return
    stats.loc[team, STAT_COLUMNS] = [
        prior_games,
        seed["goals_for_pg"] * prior_games,
        seed["goals_against_pg"] * prior_games,
        seed["shots_for_pg"] * prior_games,
        seed["shots_against_pg"] * prior_games,
        seed["power_play_pct"],
        seed["penalty_kill_pct"],
        seed["save_pct"],
        seed["point_pct"],
    ]


def _apply_game_to_stats(stats: pd.DataFrame, home: str, away: str,
                         home_goals: int, away_goals: int,
                         seed: dict[str, float],
                         prior_games: float) -> None:
    _ensure_team(stats, home, seed, prior_games)
    _ensure_team(stats, away, seed, prior_games)
    for team, gf, ga in ((home, home_goals, away_goals), (away, away_goals, home_goals)):
        games_before = float(stats.at[team, "games"])
        points_before = float(stats.at[team, "point_pct"]) * games_before * 2.0
        games_after = games_before + 1.0
        points_added = 2.0 if gf > ga else 0.0
        stats.at[team, "games"] = games_after
        stats.at[team, "goals_for"] = float(stats.at[team, "goals_for"]) + float(gf)
        stats.at[team, "goals_against"] = float(stats.at[team, "goals_against"]) + float(ga)
        stats.at[team, "shots_for"] = float(stats.at[team, "shots_for"]) + seed["shots_for_pg"]
        stats.at[team, "shots_against"] = (
            float(stats.at[team, "shots_against"]) + seed["shots_against_pg"]
        )
        stats.at[team, "point_pct"] = (points_before + points_added) / (games_after * 2.0)


def _has_odds_columns(games: pd.DataFrame) -> bool:
    aliases = [name for names in ODDS_ALIASES.values() for name in names]
    return any(name in games.columns for name in aliases)


def _betting_warnings(games: pd.DataFrame) -> list[str]:
    if not _has_odds_columns(games):
        return []
    warnings = []
    if not any(c in games.columns for c in ("captured_at", "odds_captured_at", "price_time")):
        warnings.append("Historical odds have no captured_at/price_time column; ROI is not decision-grade.")
    if not any(c in games.columns for c in ("bookmaker", "sportsbook", "source")):
        warnings.append("Historical odds have no bookmaker/source column; ROI may mix incomparable books.")
    if not any(c in games.columns for c in ("start_time", "commence_time", "game_time")):
        warnings.append("Historical odds have no start_time/commence_time column; cannot prove prices were pre-game.")
    return warnings


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
             p_book: float, min_edge: float,
             extra: dict[str, Any] | None = None) -> None:
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
    row = {
        "date": str(game["date"]),
        "event_id": OH.id_key(game.get("game_id") or game.get("event_id") or ""),
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
    }
    if extra:
        row.update(extra)
    candidates.append(row)


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


def _history_bet_candidates(game: pd.Series, pred: dict, quotes: pd.DataFrame,
                            min_edge: float) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for _group_key, group in OH.quote_groups(quotes):
        p_books = OH.devig_group(group)
        for idx, quote in group.iterrows():
            market = str(quote["market"])
            side = str(quote["side"])
            line = None if market == "ml" else float(quote["_line_value"])
            _add_bet(
                candidates,
                game=game,
                pred=pred,
                market=market,
                side=side,
                line=line,
                odds=float(quote["decimal_odds"]),
                p_book=float(p_books[int(idx)]),
                min_edge=min_edge,
                extra={
                    "bookmaker": str(quote["bookmaker"]),
                    "captured_at_utc": _iso_utc(quote["_captured_at_utc"]),
                    "start_time_utc": _iso_utc(quote["_start_time_utc"]),
                    "source": str(quote["source"]),
                    "odds_source": "odds_history",
                },
            )

    best: dict[tuple[str, str, str], dict[str, Any]] = {}
    for bet in candidates:
        key = (str(bet.get("bookmaker", "")), str(bet["market"]), str(bet["line"]))
        if key not in best or float(bet["edge"]) > float(best[key]["edge"]):
            best[key] = bet
    return sorted(best.values(), key=lambda b: -float(b["edge"]))


def _iso_utc(value: Any) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.isoformat().replace("+00:00", "Z")


def _prepare_odds_history(odds_history: pd.DataFrame | str | Path | None,
                          games: pd.DataFrame,
                          hours_before_start: float | None) -> tuple[pd.DataFrame | None,
                                                                      dict[str, pd.DataFrame]]:
    if odds_history is None:
        return None, {}
    if isinstance(odds_history, (str, Path)):
        history = OH.load_odds_history(odds_history, results_df=games)
    else:
        history = OH.validate_odds_history(odds_history, results_df=games)
    latest = OH.latest_pre_game_quotes(history, hours_before_start=hours_before_start)
    by_event = {
        str(event_id): group.copy()
        for event_id, group in latest.groupby("event_id", sort=False)
    }
    return latest, by_event


def run_backtest(results: pd.DataFrame | None = None, *, model: str = "blend",
                 min_edge: float = 0.03, walk_forward: bool = True,
                 prior_games: float = DEFAULT_PRIOR_GAMES,
                 strict: bool = True,
                 odds_history: pd.DataFrame | str | Path | None = None,
                 hours_before_start: float | None = None) -> dict[str, Any]:
    games = load_results() if results is None else results.copy()
    games = games.sort_values("date").reset_index(drop=True)
    if not games.empty:
        for col in ("home_goals", "away_goals"):
            games[col] = pd.to_numeric(games[col], errors="coerce")
        games = games.dropna(subset=["home_goals", "away_goals"]).copy()
        for col in ("home_goals", "away_goals"):
            games[col] = games[col].astype(int)
    rows: list[dict[str, Any]] = []
    bets: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    selected_history, history_by_event = _prepare_odds_history(
        odds_history,
        games,
        hours_before_start,
    )
    stats = _initial_walkforward_stats(games, prior_games) if walk_forward else None
    seed = _league_seed_values()

    for _date, day in games.groupby(games["date"].astype(str), sort=True):
        updates: list[tuple[str, str, int, int]] = []
        for game in day.itertuples(index=False):
            g = pd.Series(game._asdict())
            home = _canonical_name(g["home"])
            away = _canonical_name(g["away"])
            try:
                pred = M.predict_match(
                    home,
                    away,
                    model=model,
                    stats_df=stats.reset_index() if stats is not None else None,
                )
            except Exception as e:  # noqa: BLE001
                skipped.append({
                    "date": str(g.get("date", "")),
                    "home": str(g.get("home", "")),
                    "away": str(g.get("away", "")),
                    "error": str(e),
                })
                continue

            home_goals = int(g["home_goals"])
            away_goals = int(g["away_goals"])
            updates.append((home, away, home_goals, away_goals))
            y = 1 if home_goals > away_goals else 0
            p_home = float(pred["p_home"])
            pred_margin = float(pred["lambda_home"] - pred["lambda_away"])
            actual_margin = home_goals - away_goals
            pred_total = float(pred["total"])
            actual_total = home_goals + away_goals
            rows.append({
                "date": str(g["date"]),
                "game_id": str(g.get("game_id", "")),
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
            event_id = OH.id_key(g.get("game_id") or g.get("event_id") or "")
            if selected_history is not None:
                quotes = history_by_event.get(event_id)
                if quotes is not None and not quotes.empty:
                    bets.extend(_history_bet_candidates(g, pred, quotes, min_edge))
            else:
                bets.extend(_bet_candidates(g, pred, min_edge))

        if stats is not None:
            for home, away, home_goals, away_goals in updates:
                _apply_game_to_stats(stats, home, away, home_goals, away_goals,
                                     seed, prior_games)

    if not rows:
        raise ValueError("No backtestable games after filtering unknown teams")
    if skipped and strict:
        first = skipped[0]
        raise ValueError(
            f"Skipped {len(skipped)} NHL game(s); first skip "
            f"{first['date']} {first['away']} @ {first['home']}: {first['error']}"
        )
    out = pd.DataFrame(rows)
    actual_home = (out["actual_margin"] > 0).astype(float)
    home_rate = float(actual_home.mean())
    brier = float(out["brier"].mean())
    accuracy = float(out["correct"].mean())
    home_rate_brier = float(((home_rate - actual_home) ** 2).mean())
    summary = {
        "games": int(len(out)),
        "model": model,
        "mode": "walk_forward" if walk_forward else "static_final_snapshot",
        "prior_games": round(float(prior_games), 2) if walk_forward else None,
        "skipped": int(len(skipped)),
        "accuracy": round(accuracy, 4),
        "home_win_rate": round(home_rate, 4),
        "always_home_accuracy": round(home_rate, 4),
        "avg_home_prob": round(float(out["p_home"].mean()), 4),
        "brier": round(brier, 5),
        "constant_50_brier": 0.25,
        "home_rate_brier": round(home_rate_brier, 5),
        "logloss": round(float(out["logloss"].mean()), 5),
        "margin_mae": round(float(out["abs_margin_error"].mean()), 3),
        "total_mae": round(float(out["abs_total_error"].mean()), 3),
        "beats_trivial_baselines": bool(brier < min(0.25, home_rate_brier) and accuracy > home_rate),
    }
    betting = _summarize_bets(bets)
    if selected_history is not None:
        betting["decision_grade"] = not selected_history.empty
        betting["odds_source"] = "odds_history"
        betting["eligible_quote_rows"] = int(len(selected_history))
        betting["warnings"] = [] if not selected_history.empty else [
            "Odds history validated, but no rows were eligible before the selected cutoff."
        ]
    else:
        warnings = _betting_warnings(games)
        if warnings:
            betting["decision_grade"] = False
            betting["warnings"] = warnings
        elif _has_odds_columns(games):
            betting["decision_grade"] = True
            betting["warnings"] = []
            betting["odds_source"] = "embedded_wide_columns"
    if selected_history is None and _has_odds_columns(games) and not betting.get("decision_grade"):
        betting["decision_grade"] = False
    return {"summary": summary, "rows": rows, "bets": bets, "betting": betting,
            "skipped": skipped}


def _summarize_bets(bets: list[dict[str, Any]]) -> dict[str, Any]:
    if not bets:
        return {"bets": 0, "staked": 0.0, "pnl": 0.0, "roi": None, "win_rate": None}
    df = pd.DataFrame(bets)
    summary = _basic_bet_summary(df)
    if "market" in df.columns:
        summary["by_market"] = {
            str(market): _basic_bet_summary(group)
            for market, group in df.groupby("market", sort=True)
        }
    if "bookmaker" in df.columns:
        summary["by_bookmaker"] = {
            str(bookmaker): _basic_bet_summary(group)
            for bookmaker, group in df.groupby("bookmaker", sort=True)
        }
    return summary


def _basic_bet_summary(df: pd.DataFrame) -> dict[str, Any]:
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
    mode = s.get("mode", "unknown")
    prior = "" if s.get("prior_games") is None else f", prior {s['prior_games']} games"
    print(f"NHL backtest ({s['model']}, {mode}{prior}) · {s['games']} games")
    print(f"  accuracy {s['accuracy']:.1%} · Brier {s['brier']:.4f} · log-loss {s['logloss']:.4f}")
    print(f"  baselines: always-home accuracy {s['always_home_accuracy']:.1%} · "
          f"home-rate Brier {s['home_rate_brier']:.4f} · "
          f"constant Brier {s['constant_50_brier']:.4f}")
    print(f"  beats trivial baselines: {s['beats_trivial_baselines']}")
    print(f"  margin MAE {s['margin_mae']:.2f} · total MAE {s['total_mae']:.2f}")
    b = report["betting"]
    if "decision_grade" in b:
        source = b.get("odds_source", "embedded_wide_columns")
        quotes = b.get("eligible_quote_rows")
        suffix = "" if quotes is None else f", {quotes} eligible quote row(s)"
        print(f"  odds: {source}, decision_grade={b['decision_grade']}{suffix}")
    for warning in b.get("warnings", []):
        print(f"  warning: {warning}")
    if b["bets"]:
        roi = "n/a" if b["roi"] is None else f"{b['roi']:.1%}"
        print(f"  betting: {b['bets']} bet(s), {b['won']}-{b['lost']}-{b['push']}, "
              f"PnL {b['pnl']:+.2f}u, ROI {roi}")
        for market, stats in b.get("by_market", {}).items():
            market_roi = "n/a" if stats["roi"] is None else f"{stats['roi']:.1%}"
            print(f"    {market}: {stats['bets']} bet(s), PnL {stats['pnl']:+.2f}u, ROI {market_roi}")
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
    ap.add_argument("--static", action="store_true",
                    help="use final snapshot ratings (leaky; diagnostic only)")
    ap.add_argument("--prior-games", type=float, default=DEFAULT_PRIOR_GAMES,
                    help="league-average prior games per team for walk-forward mode")
    ap.add_argument("--no-strict", action="store_true",
                    help="report skipped games instead of failing")
    ap.add_argument("--odds-history",
                    help="strict event_id/bookmaker/timestamp historical odds CSV")
    ap.add_argument("--hours-before-start", type=float,
                    help="use the latest quote captured at least this many hours before start")
    args = ap.parse_args()

    report = run_backtest(load_results(args.results), model=args.model, min_edge=args.min_edge,
                          walk_forward=not args.static, prior_games=args.prior_games,
                          strict=not args.no_strict, odds_history=args.odds_history,
                          hours_before_start=args.hours_before_start)
    _print_report(report, args.show_bets)


if __name__ == "__main__":
    main()

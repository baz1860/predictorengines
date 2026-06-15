"""
golf/market.py  –  Market anchoring (de-vig + blend) and CLV tracking.

Golf outright books quote 100+ runners with a large overround and a strong
favourite-longshot bias, so a flat multiplicative de-vig over-prices longshots.
This module adds the power method (solve k so Σ pᵢ^k = 1) for many-runner
markets, a log-odds blend of model vs market, and closing-line-value tracking
mirroring the root clv.py.

  fair = devig(odds_list, method="power")     # list of fair probabilities
  p    = blend(p_model, p_market, w)           # w = weight on market
  snapshot_fair(...) / clv_report(...)         # → data/odds_history.csv
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import math
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
BLEND_JSON = DATA_DIR / "market_blend.json"
ODDS_HISTORY = DATA_DIR / "odds_history.csv"

# Default market-blend weights (weight ON the market price). Sharp longshot
# outrights lean to the market; lower-variance model-edge markets lean to model.
DEFAULT_BLEND_W = {
    "win": 0.60, "top5": 0.45, "top10": 0.40, "top20": 0.35,
    "cut": 0.25, "matchup": 0.20, "3ball": 0.20,
}


# ─────────────────────────────────────────────
# De-vig
# ─────────────────────────────────────────────

def _implied(odds_list: list[float]) -> list[float]:
    return [1.0 / o for o in odds_list if o and o > 1.0]


def devig_multiplicative(odds_list: list[float]) -> list[float]:
    imp = _implied(odds_list)
    s = sum(imp)
    return [p / s for p in imp] if s > 0 else imp


def devig_power(odds_list: list[float], tol: float = 1e-9) -> list[float]:
    """Power de-vig: find k with Σ (1/oᵢ)^k = 1, return pᵢ = (1/oᵢ)^k.

    k > 1 shrinks longshots more than favourites, correcting the favourite-
    longshot bias multiplicative de-vig leaves in big outright fields.
    """
    imp = _implied(odds_list)
    if not imp:
        return imp
    if abs(sum(imp) - 1.0) < tol:
        return imp
    lo, hi = 0.3, 5.0
    for _ in range(100):
        k = 0.5 * (lo + hi)
        s = sum(p ** k for p in imp)
        if s > 1.0:
            lo = k          # need larger k to shrink the sum
        else:
            hi = k
        if abs(s - 1.0) < tol:
            break
    k = 0.5 * (lo + hi)
    out = [p ** k for p in imp]
    s = sum(out)
    return [p / s for p in out]


# Typical per-line bookmaker margin baked into a single-sided place price.
LINE_MARGIN = {"top5": 1.12, "top10": 1.10, "top20": 1.08, "cut": 1.06}


def devig(odds_list: list[float], method: str = "power") -> list[float]:
    """Fair probabilities for one MUTUALLY-EXCLUSIVE market (outright winner,
    matchup, 3-ball) — the listed outcomes partition the space so they
    normalise to 1. Do NOT use for place lines (top-N / cut): those aren't
    mutually exclusive across players — use devig_line()."""
    if method == "multiplicative":
        return devig_multiplicative(odds_list)
    return devig_power(odds_list)


def devig_line(odds: float, market: str) -> float:
    """Fair probability for a single-sided place line (top-N / make-cut), where
    the book bakes a per-line margin into the 'yes' price and quotes no
    complement. fair_p = implied / margin."""
    if not odds or odds <= 1.0:
        return 0.0
    return (1.0 / odds) / LINE_MARGIN.get(market, 1.10)


def fair_prob_map(odds_by_name: dict[str, float], method: str = "power") -> dict[str, float]:
    """De-vig a whole mutually-exclusive market keyed by name → fair prob."""
    names = [n for n, o in odds_by_name.items() if o and o > 1.0]
    fair = devig([odds_by_name[n] for n in names], method=method)
    return dict(zip(names, fair))


OUTRIGHT_MARGIN = 1.30   # assumed win-market margin when only a partial board is seen


def devig_outright(odds_by_name: dict[str, float], complete_threshold: float = 1.10
                   ) -> dict[str, float]:
    """Fair win probabilities from an outright board, robust to partial boards.

    A complete board's implied probs sum to >1 (the overround) → power de-vig,
    normalising to 1 and correcting favourite-longshot bias. A partial board
    (we only pulled the top N names, implied sum <1) must NOT be normalised to
    1 — instead strip an assumed market margin per price."""
    names = [n for n, o in odds_by_name.items() if o and o > 1.0]
    imp_sum = sum(1.0 / odds_by_name[n] for n in names)
    if imp_sum >= complete_threshold:
        return fair_prob_map(odds_by_name, method="power")
    return {n: (1.0 / odds_by_name[n]) / OUTRIGHT_MARGIN for n in names}


# ─────────────────────────────────────────────
# Blend (log-odds)
# ─────────────────────────────────────────────

def _logit(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def blend(p_model: float, p_market: float, w: float) -> float:
    """Blend model and market probabilities in log-odds space. w ∈ [0,1] is the
    weight on the market; w=0 → pure model, w=1 → pure market."""
    if p_market is None:
        return p_model
    return _sigmoid((1 - w) * _logit(p_model) + w * _logit(p_market))


def blend_weights() -> dict:
    if BLEND_JSON.exists():
        try:
            return {**DEFAULT_BLEND_W, **json.loads(BLEND_JSON.read_text())}
        except (ValueError, OSError):
            pass
    return dict(DEFAULT_BLEND_W)


# ─────────────────────────────────────────────
# CLV tracking
# ─────────────────────────────────────────────

CLV_COLS = ["ts", "event", "player", "market", "odds", "fair_odds", "fair_prob"]


def snapshot_fair(odds_by_market: dict[str, dict[str, float]], event: str = "",
                  method: str = "power") -> int:
    """Append a timestamped de-vigged snapshot of the current board to
    odds_history.csv. odds_by_market: {market: {player: decimal_odds}}.
    Returns rows written. The latest snapshot before settlement is the close."""
    ts = _dt.datetime.now().isoformat(timespec="seconds")
    rows = []
    for market, board in odds_by_market.items():
        fair = fair_prob_map(board, method=method)
        for player, o in board.items():
            fp = fair.get(player)
            rows.append({
                "ts": ts, "event": event, "player": player, "market": market,
                "odds": round(float(o), 3),
                "fair_odds": round(1.0 / fp, 3) if fp else "",
                "fair_prob": round(fp, 5) if fp else "",
            })
    if not rows:
        return 0
    exists = ODDS_HISTORY.exists()
    with open(ODDS_HISTORY, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CLV_COLS)
        if not exists:
            w.writeheader()
        w.writerows(rows)
    return len(rows)


def closing_fair(player: str, market: str, event: str = "") -> float | None:
    """Most recent recorded fair probability for a player/market (the close)."""
    if not ODDS_HISTORY.exists():
        return None
    best_ts, best = "", None
    with open(ODDS_HISTORY) as f:
        for r in csv.DictReader(f):
            if r["player"] == player and r["market"] == market and \
               (not event or r["event"] == event) and r["fair_prob"]:
                if r["ts"] >= best_ts:
                    best_ts, best = r["ts"], float(r["fair_prob"])
    return best


def clv_pct(bet_odds: float, player: str, market: str, event: str = "") -> float | None:
    """CLV% = bet_odds / closing_fair_odds − 1. Positive = beat the close."""
    fp = closing_fair(player, market, event)
    if not fp or fp <= 0:
        return None
    closing_odds = 1.0 / fp
    return round(bet_odds / closing_odds - 1.0, 4)


if __name__ == "__main__":
    # demo: power vs multiplicative on a realistic full outright board
    # (favourites + a long tail; implied probs sum to ~1.5 = 50% overround).
    board = {"Scheffler": 5.0, "McIlroy": 9.0, "Rahm": 15.0, "Schauffele": 22.0,
             "Hovland": 34.0, "Morikawa": 41.0}
    tail = {f"Field{i}": o for i, o in enumerate([81, 101, 126, 151, 201,
            251, 301, 351, 401, 501, 751, 1001])}
    board.update(tail)
    # add enough of the long tail that implied probs sum to a real overround
    board.update({f"Tail{i}": float(o) for i, o in enumerate(
        [61, 71, 81, 91, 101, 126, 151, 151, 201, 201, 251, 251,
         301, 351, 401, 401, 501, 501, 751, 1001] * 3)})
    mult = dict(zip(board, devig(list(board.values()), "multiplicative")))
    powr = dict(zip(board, devig(list(board.values()), "power")))
    print(f"runners: {len(board)}  overround: {sum(1/o for o in board.values()):.3f}")
    print(f"{'player':<11}{'odds':>7}{'mult %':>9}{'power %':>9}  (power shrinks longshots)")
    for n, o in list(board.items())[:8]:
        print(f"{n:<11}{o:>7.0f}{mult[n]*100:>8.2f}%{powr[n]*100:>8.2f}%")
    print("\nplace-line de-vig (single-sided):")
    for mkt, o in (("top10", 4.5), ("cut", 1.5)):
        print(f"  {mkt:<6} odds {o} → fair {devig_line(o, mkt)*100:.1f}%")

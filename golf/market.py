"""
golf/market.py  –  Market anchoring (de-vig + blend) and CLV tracking.

Golf outright books quote 100+ runners with a large overround and a strong
favourite-longshot bias, so a flat multiplicative de-vig over-prices longshots.
This module adds the power method (solve k so Σ pᵢ^k = 1) for many-runner
markets, a log-odds blend of model vs market, and closing-line-value tracking
mirroring the root clv.py.

  fair = devig(odds_list, method="power")     # list of fair probabilities
  p    = blend(p_model, p_market, w)           # w = weight on market
  golf.economic                                # prospective CLV / ROI evidence
"""

from __future__ import annotations

import json
import math
import argparse
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
BLEND_JSON = DATA_DIR / "market_blend.json"

# Market blending is disabled until weights are estimated from timestamped,
# point-in-time odds/outcome history. Guessed weights silently rewrite the model.
DEFAULT_BLEND_W = {
    "win": 0.0, "top5": 0.0, "top10": 0.0, "top20": 0.0,
    "cut": 0.0, "matchup": 0.0, "3ball": 0.0,
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


LINE_MARKETS = {"top5", "top10", "top20", "cut"}


def devig(odds_list: list[float], method: str = "power") -> list[float]:
    """Fair probabilities for one MUTUALLY-EXCLUSIVE market (outright winner,
    matchup, 3-ball) — the listed outcomes partition the space so they
    normalise to 1. Do NOT use for place lines (top-N / cut): those aren't
    mutually exclusive across players — use devig_line()."""
    if method == "multiplicative":
        return devig_multiplicative(odds_list)
    return devig_power(odds_list)


def devig_line(odds: float, market: str) -> float:
    """Raw implied probability for a single-sided place line.

    A one-sided quote cannot be de-vigged without its complement. Do not invent
    a margin: this value is diagnostic only and blending is disabled by default.
    """
    if not odds or odds <= 1.0:
        return 0.0
    return 1.0 / odds


def fair_prob_map(odds_by_name: dict[str, float], method: str = "power") -> dict[str, float]:
    """De-vig a whole mutually-exclusive market keyed by name → fair prob."""
    names = [n for n, o in odds_by_name.items() if o and o > 1.0]
    fair = devig([odds_by_name[n] for n in names], method=method)
    return dict(zip(names, fair))


def devig_outright(odds_by_name: dict[str, float], complete_threshold: float = 1.10,
                   complete: bool | None = None,
                   ) -> dict[str, float]:
    """Fair win probabilities from an outright board, robust to partial boards.

    A complete board's implied probs sum to >1 (the overround) → power de-vig,
    normalising to 1 and correcting favourite-longshot bias. A partial board
    (we only pulled the top N names, implied sum <1) must NOT be normalised to
    1 — return raw implied probabilities because the missing overround cannot be
    identified honestly."""
    names = [n for n, o in odds_by_name.items() if o and o > 1.0]
    imp_sum = sum(1.0 / odds_by_name[n] for n in names)
    if complete is True or (complete is None and imp_sum >= complete_threshold):
        return fair_prob_map(odds_by_name, method="power")
    return {n: 1.0 / odds_by_name[n] for n in names}


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

def snapshot_fair(odds_by_market: dict[str, dict[str, float]], event: str = "",
                  method: str = "power") -> int:
    """Retired name-only snapshot API.

    Event names cannot safely identify consecutive boards and this call lacks
    bookmaker/group/phase provenance. Provider refresh now records quotes via
    :func:`golf.economic.record_odds_snapshot`.
    """
    raise RuntimeError(
        "snapshot_fair is retired; run golf.refresh so golf.economic can "
        "capture an event-ID/provider/phase-aware snapshot"
    )


def closing_fair(player: str, market: str, event: str = "") -> float | None:
    """Retired ambiguous event-name lookup; use ``golf.economic``."""
    return None


def clv_pct(bet_odds: float, player: str, market: str, event: str = "") -> float | None:
    """CLV% = bet_odds / closing_fair_odds − 1. Positive = beat the close."""
    fp = closing_fair(player, market, event)
    if not fp or fp <= 0:
        return None
    closing_odds = 1.0 / fp
    return round(bet_odds / closing_odds - 1.0, 4)


def clv_report(edge_path: Path | None = None, predictions_path: Path | None = None,
               event: str = "") -> dict:
    """Compatibility entry point for the prospective economic report."""
    from .economic import economic_report

    return economic_report()


def _demo() -> None:
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Golf market utilities")
    ap.add_argument("--clv-report", action="store_true")
    ap.add_argument("--event", default="")
    args = ap.parse_args()
    if args.clv_report:
        rep = clv_report(event=args.event)
        print(json.dumps(rep, indent=2))
    else:
        _demo()


if __name__ == "__main__":
    main()

"""tennis/market.py — two-way market anchoring (de-vig + blend).

Tennis match markets are two-way, so de-vigging is simpler than golf's many-
runner outright boards: a multiplicative de-vig of the two prices is exact and
unbiased. A log-odds blend pulls the model toward the sharper market price.

  pa, pb = devig_two_way(odds_a, odds_b)        # fair two-way probabilities
  p      = blend(p_model, p_market, w)          # w = weight on market
"""
from __future__ import annotations

import json
import math
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
BLEND_JSON = DATA_DIR / "market_blend.json"

# Default market-blend weights (weight ON the market price). Singles books are
# sharp → match markets lean to the market; outright futures lean a touch less.
DEFAULT_BLEND_W = {
    "match_winner": 0.50, "set_hcp": 0.40, "first_set": 0.40, "total_games": 0.40,
    "win": 0.55, "final": 0.50, "sf": 0.45, "qf": 0.40,
}


# ─────────────────────────────────────────────
# De-vig
# ─────────────────────────────────────────────

def devig_two_way(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Fair (p_a, p_b) for a two-way match market. Multiplicative de-vig is exact
    for a two-outcome book, so no power correction is needed."""
    if not (odds_a and odds_a > 1.0 and odds_b and odds_b > 1.0):
        return (0.5, 0.5)
    qa, qb = 1.0 / odds_a, 1.0 / odds_b
    tot = qa + qb
    return (qa / tot, qb / tot)

# ─────────────────────────────────────────────
# Blend (log-odds)
# ─────────────────────────────────────────────

def _logit(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def blend(p_model: float, p_market: float | None, w: float) -> float:
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


if __name__ == "__main__":
    # demo: two-way de-vig + log-odds blend on a typical match price
    oa, ob = 1.80, 2.10
    pa, pb = devig_two_way(oa, ob)
    print(f"odds {oa}/{ob} → fair {pa:.3f}/{pb:.3f} (sum {pa + pb:.3f}, "
          f"overround {1/oa + 1/ob - 1:+.1%})")
    for pm in (0.50, 0.60, 0.75):
        print(f"  model {pm:.2f} blended @w=0.5 → {blend(pm, pa, 0.5):.3f}")

"""tennis/market.py — market anchoring (de-vig + blend) and CLV tracking.

Tennis match markets are two-way, so de-vigging is simpler than golf's many-
runner outright boards: a multiplicative de-vig of the two prices is exact and
unbiased. Outright (win/final/SF/QF) boards are many-way and get the power
de-vig to correct favourite-longshot bias. A log-odds blend pulls the model
toward the sharper market price (match markets lean to the market more than
futures, since books are sharper on singles), and a CLV log records model/fair
price vs the close.

  pa, pb = devig_two_way(odds_a, odds_b)        # fair two-way probabilities
  p      = blend(p_model, p_market, w)          # w = weight on market
  snapshot_fair(...) / clv_pct(...)             # → data/odds_history.csv
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

# Default market-blend weights (weight ON the market price). Singles books are
# sharp → match markets lean to the market; outright futures lean a touch less.
DEFAULT_BLEND_W = {
    "match_winner": 0.50, "set_hcp": 0.40, "first_set": 0.40, "total_games": 0.40,
    "win": 0.55, "final": 0.50, "sf": 0.45, "qf": 0.40,
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


def devig_two_way(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Fair (p_a, p_b) for a two-way match market. Multiplicative de-vig is exact
    for a two-outcome book, so no power correction is needed."""
    if not (odds_a and odds_a > 1.0 and odds_b and odds_b > 1.0):
        return (0.5, 0.5)
    qa, qb = 1.0 / odds_a, 1.0 / odds_b
    tot = qa + qb
    return (qa / tot, qb / tot)


def devig_power(odds_list: list[float], tol: float = 1e-9) -> list[float]:
    """Power de-vig for a many-way outright board: find k with Σ(1/oᵢ)^k = 1.

    k > 1 shrinks longshots more than favourites, correcting the favourite-
    longshot bias multiplicative de-vig leaves in a big outright field."""
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
            lo = k
        else:
            hi = k
        if abs(s - 1.0) < tol:
            break
    k = 0.5 * (lo + hi)
    out = [p ** k for p in imp]
    s = sum(out)
    return [p / s for p in out]


def fair_outright_map(odds_by_name: dict[str, float]) -> dict[str, float]:
    """Power-de-vig a many-way outright board keyed by player → fair prob."""
    names = [n for n, o in odds_by_name.items() if o and o > 1.0]
    fair = devig_power([odds_by_name[n] for n in names])
    return dict(zip(names, fair))


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


# ─────────────────────────────────────────────
# CLV tracking
# ─────────────────────────────────────────────

CLV_COLS = ["ts", "event", "player", "opponent", "market", "odds",
            "fair_odds", "fair_prob"]


def snapshot_fair(rows: list[dict], event: str = "") -> int:
    """Append a timestamped de-vigged snapshot of a match board to
    odds_history.csv. Each row: {player_a, player_b, odds_a, odds_b}. The latest
    snapshot before settlement is the closing line. Returns rows written."""
    ts = _dt.datetime.now().isoformat(timespec="seconds")
    out = []
    for r in rows:
        a, b = str(r.get("player_a", "")), str(r.get("player_b", ""))
        try:
            oa, ob = float(r["odds_a"]), float(r["odds_b"])
        except (ValueError, KeyError, TypeError):
            continue
        pa, pb = devig_two_way(oa, ob)
        for player, opp, o, fp in ((a, b, oa, pa), (b, a, ob, pb)):
            out.append({"ts": ts, "event": event, "player": player,
                        "opponent": opp, "market": "match_winner",
                        "odds": round(o, 3),
                        "fair_odds": round(1.0 / fp, 3) if fp else "",
                        "fair_prob": round(fp, 5) if fp else ""})
    if not out:
        return 0
    exists = ODDS_HISTORY.exists()
    with open(ODDS_HISTORY, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CLV_COLS)
        if not exists:
            w.writeheader()
        w.writerows(out)
    return len(out)


def closing_fair(player: str, opponent: str = "", event: str = "") -> float | None:
    """Most recent recorded fair probability for a player's match (the close)."""
    if not ODDS_HISTORY.exists():
        return None
    best_ts, best = "", None
    with open(ODDS_HISTORY) as f:
        for r in csv.DictReader(f):
            if r["player"] != player or not r["fair_prob"]:
                continue
            if opponent and r.get("opponent") and r["opponent"] != opponent:
                continue
            if event and r["event"] != event:
                continue
            if r["ts"] >= best_ts:
                best_ts, best = r["ts"], float(r["fair_prob"])
    return best


def clv_pct(bet_odds: float, player: str, opponent: str = "",
            event: str = "") -> float | None:
    """CLV% = bet_odds / closing_fair_odds − 1. Positive = beat the close."""
    fp = closing_fair(player, opponent, event)
    if not fp or fp <= 0:
        return None
    return round(bet_odds * fp - 1.0, 4)


if __name__ == "__main__":
    # demo: two-way de-vig + log-odds blend on a typical match price
    oa, ob = 1.80, 2.10
    pa, pb = devig_two_way(oa, ob)
    print(f"odds {oa}/{ob} → fair {pa:.3f}/{pb:.3f} (sum {pa + pb:.3f}, "
          f"overround {1/oa + 1/ob - 1:+.1%})")
    for pm in (0.50, 0.60, 0.75):
        print(f"  model {pm:.2f} blended @w=0.5 → {blend(pm, pa, 0.5):.3f}")

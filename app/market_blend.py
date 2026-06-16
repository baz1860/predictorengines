"""Shared market-anchored blending for every priced engine (V3 M5).

The model beats noise but rarely beats the closing line. Anchoring the model's
probabilities toward the de-vigged market removes most *fake* edges. The World
Cup engine has shipped this for 1X2 since V2 (`market_blend.py` at the root);
this module generalises the same idea so Club Soccer and CFB can use it through
the adapter layer without touching their flat-module runners.

Design choices:

- Blending is done in **logit space**, per outcome, then renormalised. This is
  the same transform the World Cup 1X2 blend uses, so behaviour is consistent.
- `w` is the weight on the **model**; `1 - w` is the weight on the market.
  `w = 1.0` is pure model (no anchoring); `w = 0.0` is pure market.
- The helpers are dependency-light (pure Python + math) so they import cleanly
  inside the subprocess runners *or* the adapter, with no pandas/numpy needs.

IMPORTANT (M5 guardrail): a generalised blend stays **experimental / OFF by
default** for an engine until a held-out metric shows it beats *both* pure model
and pure market for that engine. Until then it is exposed only behind a flag and
is never used for recommendations. See `V3_NOTES.md`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence

EPS = 1e-6

# Per-engine experimental blend weights. These are conservative placeholders,
# NOT fitted held-out values — which is exactly why the blend is OFF by default
# for these engines (see DEFAULT_BLEND_ON). Flipping a default requires a
# validation table per the M5/M6 guardrail.
_WEIGHTS_FILE = Path(__file__).resolve().parents[1] / "data" / "market_blend_suite.json"
_FALLBACK_W = 0.5
# Engines whose blend is a validated DEFAULT (not just available behind a flag).
# World Cup ships its own 1X2 blend inside edge.py; golf ships market_blend in
# its runner. Club Soccer and CFB are experimental until validated.
DEFAULT_BLEND_ON = frozenset()


def _clip(p: float) -> float:
    return min(max(float(p), EPS), 1.0 - EPS)


def _logit(p: float) -> float:
    p = _clip(p)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def blend_probs(p_model: Sequence[float], p_market: Sequence[float],
                w: float) -> list[float]:
    """Logit-space blend of two probability vectors, renormalised to sum to 1.

    Works for any number of mutually-exclusive outcomes (2 for ML/totals/BTTS,
    3 for 1X2). Returns the model vector unchanged if the blend degenerates."""
    pm = list(p_model)
    pk = list(p_market)
    if len(pm) != len(pk) or not pm:
        return [float(x) for x in pm]
    w = float(w)
    z = [w * _logit(m) + (1.0 - w) * _logit(k) for m, k in zip(pm, pk)]
    p = [_sigmoid(v) for v in z]
    s = sum(p)
    if s <= 0:
        return [float(x) for x in pm]
    return [v / s for v in p]


def blend_two(p_model: float, p_market: float, w: float) -> float:
    """Blend a single probability against its complement (two-outcome market).

    Returns just the blended `p_model` side; the complement is `1 - result`."""
    p, _ = blend_probs([p_model, 1.0 - p_model], [p_market, 1.0 - p_market], w)
    return p


def devig(odds: Iterable[float]) -> list[float]:
    """Proportional de-vig: normalise inverse decimal odds to sum to 1."""
    inv = [1.0 / float(o) for o in odds if float(o) > 1.0]
    s = sum(inv)
    if s <= 0:
        return []
    return [x / s for x in inv]


def anchor_line(model_line: float, market_line: float, w: float) -> float:
    """Convex blend of a model line toward a market line (spread/total points).

    Used where the natural anchor is a *line* rather than a probability. Linear
    space is correct for lines: `w * model + (1 - w) * market`."""
    return w * float(model_line) + (1.0 - w) * float(market_line)


def weight_for(engine: str) -> float:
    """Stored experimental blend weight for an engine, or a safe fallback.

    Reads data/market_blend_suite.json `{engine: w}` if present. Absent file or
    key → `_FALLBACK_W`. Never raises."""
    try:
        if _WEIGHTS_FILE.exists():
            d = json.loads(_WEIGHTS_FILE.read_text())
            if isinstance(d, dict) and engine in d:
                return float(d[engine])
    except Exception:
        pass
    return _FALLBACK_W


def is_default_on(engine: str) -> bool:
    """Whether the generalised blend is a validated default for this engine."""
    return engine in DEFAULT_BLEND_ON


def apply_blend_to_rows(rows: list[dict], engine: str, bankroll: float,
                        kelly_fraction: float, w: float | None = None,
                        kelly_key: str = "kelly_frac") -> float:
    """Anchor each edge row's `p_model` toward its de-vigged market prob
    (`p_book`/`p_market`) and recompute edge / EV / Kelly / stake **in place**.

    This is the adapter-level contract boundary: it lets every priced engine
    share one blend without editing its flat-module runner. Rows missing a usable
    `p_model`, market prob, or odds are left untouched. Returns the weight used.

    NOTE: two-outcome (probability-space) anchoring is applied uniformly. For
    CFB this anchors the cover/over probability rather than the points line
    directly — equivalent in pulling the model toward the market, and far less
    invasive than re-plumbing the spread/total math. See V3_NOTES.md (M5)."""
    if w is None:
        w = weight_for(engine)
    for r in rows:
        try:
            pm = float(r.get("p_model"))
            pk = float(r.get("p_market", r.get("p_book")))
            odds = float(r.get("odds"))
        except (TypeError, ValueError):
            continue
        if not (0.0 < pm < 1.0 and 0.0 < pk < 1.0 and odds > 1.0):
            continue
        new_pm = blend_two(pm, pk, w)
        b = odds - 1.0
        kelly = max(0.0, (new_pm * odds - 1.0) / b) if b > 0 else 0.0
        kfrac = kelly_fraction * kelly
        r["p_model"] = round(new_pm, 4)
        r["edge"] = round(new_pm - pk, 4)
        r["ev_per_unit"] = round(new_pm * odds - 1.0, 4)
        r[kelly_key] = round(kfrac, 4)
        r["stake_gbp"] = round(kfrac * float(bankroll), 2)
        r["market_blend_w"] = round(float(w), 3)
    return float(w)

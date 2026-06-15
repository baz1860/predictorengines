"""Suite-level portfolio risk controls (V3 M4).

One pooled bankroll backs every engine, so staking discipline has to be enforced
at the suite boundary too — not only inside each engine. The per-engine portfolio
logic (edge.portfolio_size for World Cup, golf/portfolio.py for golf) still runs
first; this module is the shared backstop applied when bets are actually recorded
(bankroll_store.place_bets), so no single event, correlated group, or day can
blow past a suite cap regardless of which engine proposed the bet.

All caps are *pure risk controls*: they only ever reduce a stake. Defaults are
generous enough that ordinary small stakes are untouched.
"""
from __future__ import annotations

MIN_STAKE = 0.10           # never record below this (currency)
SINGLE_EVENT_CAP = 0.15    # max fraction of bankroll on one event_id
CORRELATED_CAP = 0.25      # max fraction of bankroll per engine (correlated book)
DAILY_CAP = 0.40           # max fraction of bankroll in new stakes per day

# Drawdown brake: scale all new stakes down as bankroll falls below its peak.
DD_FULL = 0.85             # bankroll/peak above this → no brake
DD_FLOOR = 0.70            # at/below this → maximum brake
BRAKE_MIN = 0.50           # staking multiplier at full drawdown brake


def drawdown_factor(bankroll: float, peak: float | None) -> float:
    """Multiplier in [BRAKE_MIN, 1.0] based on how far below peak we are."""
    peak = peak if peak and peak > 0 else bankroll
    if peak <= 0:
        return 1.0
    dd = bankroll / peak
    if dd >= DD_FULL:
        return 1.0
    if dd <= DD_FLOOR:
        return BRAKE_MIN
    return BRAKE_MIN + (1 - BRAKE_MIN) * (dd - DD_FLOOR) / (DD_FULL - DD_FLOOR)


def _scale_group(items: list[dict], key, cap: float, prior: dict | None = None):
    """Scale each group's stakes down so group total ≤ cap (incl. prior open)."""
    if cap <= 0:
        return
    groups: dict = {}
    for it in items:
        groups.setdefault(key(it), []).append(it)
    for gkey, rs in groups.items():
        already = float((prior or {}).get(gkey, 0.0))
        total = already + sum(r["stake"] for r in rs)
        if total > cap:
            room = max(cap - already, 0.0)
            batch = sum(r["stake"] for r in rs)
            f = (room / batch) if batch > 0 else 0.0
            for r in rs:
                r["stake"] = round(r["stake"] * f, 2)


def apply_caps(candidates: list[dict], *, bankroll: float,
               peak: float | None = None,
               prior_event_stake: dict | None = None,
               prior_engine_stake: dict | None = None,
               prior_day_stake: float = 0.0) -> list[dict]:
    """Clamp a batch of candidate bets against the suite caps.

    Each candidate is a dict with at least `stake` (currency), `event_id`, and
    `engine`. Returns the survivors (stake ≥ MIN_STAKE) with adjusted `stake` and
    a `stake_capped` flag. `prior_*` carry already-open exposure so caps consider
    the whole book, not just this batch.
    """
    if not candidates or bankroll <= 0:
        return []
    raw = {id(c): float(c.get("stake", 0.0)) for c in candidates}
    items = [dict(c, stake=round(float(c.get("stake", 0.0)), 2)) for c in candidates]

    # 1. drawdown brake (applies to the whole batch)
    brake = drawdown_factor(bankroll, peak)
    if brake < 1.0:
        for it in items:
            it["stake"] = round(it["stake"] * brake, 2)

    # 2. single-event cap
    _scale_group(items, lambda r: r.get("event_id", ""),
                 SINGLE_EVENT_CAP * bankroll, prior_event_stake)
    # 3. correlated cap per engine
    _scale_group(items, lambda r: r.get("engine", ""),
                 CORRELATED_CAP * bankroll, prior_engine_stake)
    # 4. daily cap across the whole batch
    cap_day = DAILY_CAP * bankroll
    batch_total = prior_day_stake + sum(r["stake"] for r in items)
    if batch_total > cap_day:
        room = max(cap_day - prior_day_stake, 0.0)
        cur = sum(r["stake"] for r in items)
        f = (room / cur) if cur > 0 else 0.0
        for it in items:
            it["stake"] = round(it["stake"] * f, 2)

    out = []
    for it, orig in zip(items, candidates):
        it["stake_capped"] = it["stake"] < round(raw[id(orig)], 2) - 1e-9
        if it["stake"] >= MIN_STAKE:
            out.append(it)
    return out

"""
golf/portfolio.py  –  Simultaneous-Kelly staking discipline.

A golf week throws up many bets at once that are strongly correlated: win,
top-5, top-10, top-20 and make-cut on the SAME player are nested, and matchups
add more of the same exposure. Independent quarter-Kelly on each over-bets that
player. This caps exposure per player and in aggregate, and brakes staking
during a drawdown. Mirrors the World Cup v2 M7 portfolio defaults.

  rows = apply_portfolio(rows, bankroll, peak)   # mutates/returns staked rows
"""

from __future__ import annotations

from collections import defaultdict

PER_PLAYER_CAP = 0.10   # max fraction of bankroll across one player's bets
TOTAL_CAP = 0.40        # max fraction of bankroll staked in one week
DD_FULL = 0.85          # bankroll/peak above this → no brake
DD_FLOOR = 0.70         # at/below this → maximum brake
BRAKE_MIN = 0.50        # staking multiplier at full drawdown brake
MIN_STAKE = 0.10
CERTAINTY = 0.99        # never stake an implied-certain outcome (sim artifact)


def drawdown_factor(bankroll: float, peak: float) -> float:
    """Linear staking brake between DD_FLOOR and DD_FULL of the peak."""
    if peak <= 0:
        return 1.0
    dd = bankroll / peak
    if dd >= DD_FULL:
        return 1.0
    if dd <= DD_FLOOR:
        return BRAKE_MIN
    return BRAKE_MIN + (1 - BRAKE_MIN) * (dd - DD_FLOOR) / (DD_FULL - DD_FLOOR)


def apply_portfolio(rows: list[dict], bankroll: float, peak: float | None = None,
                    per_player_cap: float = PER_PLAYER_CAP,
                    total_cap: float = TOTAL_CAP,
                    min_stake: float = MIN_STAKE) -> list[dict]:
    """Scale per-bet `stake_gbp` down for drawdown, per-player correlation, and
    a total weekly cap. Returns the surviving rows (stakes ≥ min_stake), each
    annotated with `stake_capped` when it was reduced by a cap."""
    if not rows:
        return rows
    # Defensive: drop implied-certain outcomes (p_model ≥ CERTAINTY). These are
    # simulation artifacts (e.g. make-cut when the cut doesn't bind) that would
    # otherwise read as +EV at any odds > 1 and attract the largest stakes.
    rows = [r for r in rows if float(r.get("p_model", 0.0)) < CERTAINTY]
    if not rows:
        return rows
    peak = peak if peak and peak > 0 else bankroll
    brake = drawdown_factor(bankroll, peak)

    for r in rows:
        r["_raw_stake"] = float(r.get("stake_gbp", 0.0))
        r["stake_gbp"] = round(r["_raw_stake"] * brake, 2)

    # per-player correlation cap (win/top-N/cut/matchup on one player are nested)
    cap_p = per_player_cap * bankroll
    by_player: dict[str, list] = defaultdict(list)
    for r in rows:
        by_player[r["player"]].append(r)
    for rs in by_player.values():
        s = sum(r["stake_gbp"] for r in rs)
        if s > cap_p > 0:
            f = cap_p / s
            for r in rs:
                r["stake_gbp"] = round(r["stake_gbp"] * f, 2)

    # total weekly exposure cap
    tot = sum(r["stake_gbp"] for r in rows)
    cap_t = total_cap * bankroll
    if tot > cap_t > 0:
        f = cap_t / tot
        for r in rows:
            r["stake_gbp"] = round(r["stake_gbp"] * f, 2)

    out = []
    for r in rows:
        r["stake_capped"] = r["stake_gbp"] < round(r["_raw_stake"], 2)
        r.pop("_raw_stake", None)
        if r["stake_gbp"] >= min_stake:
            out.append(r)
    return out


def summary(rows: list[dict], bankroll: float, peak: float | None = None) -> str:
    staked = sum(r.get("stake_gbp", 0.0) for r in rows)
    brake = drawdown_factor(bankroll, peak or bankroll)
    line = (f"{len(rows)} bets · staked £{staked:.2f} "
            f"({staked / bankroll * 100:.0f}% of £{bankroll:.0f})")
    if brake < 1.0:
        line += f" · drawdown brake ×{brake:.2f}"
    return line


if __name__ == "__main__":
    demo = [
        {"player": "Scottie Scheffler", "side": "win", "stake_gbp": 6.0},
        {"player": "Scottie Scheffler", "side": "top5", "stake_gbp": 5.0},
        {"player": "Scottie Scheffler", "side": "top10", "stake_gbp": 4.0},
        {"player": "Scottie Scheffler", "side": "cut", "stake_gbp": 8.0},
        {"player": "Tommy Fleetwood", "side": "top10", "stake_gbp": 3.0},
        {"player": "Ludvig Aberg", "side": "top20", "stake_gbp": 2.0},
    ]
    for peak in (100.0, 130.0):
        rows = [dict(d) for d in demo]
        rows = apply_portfolio(rows, bankroll=100.0, peak=peak)
        print(f"\npeak £{peak:.0f} → {summary(rows, 100.0, peak)}")
        for r in rows:
            flag = "  (capped)" if r.get("stake_capped") else ""
            print(f"  {r['player']:<20}{r['side']:<7} £{r['stake_gbp']:.2f}{flag}")

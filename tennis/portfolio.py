"""tennis/portfolio.py — simultaneous-Kelly staking discipline.

A tennis card throws up correlated exposure: backing one player to win a match,
to win the tournament, and across set/handicap markets all load the SAME
player, and a deep run means that exposure repeats round after round.
Independent fractional-Kelly on each over-bets that player. This caps exposure
per player and in aggregate, and brakes staking during a drawdown. Mirrors
golf/portfolio.py.

  rows = apply_portfolio(rows, bankroll, peak)   # mutates/returns staked rows
"""
from __future__ import annotations

from collections import defaultdict

PER_PLAYER_CAP = 0.10   # max fraction of bankroll across one player's bets
TOTAL_CAP = 0.40        # max fraction of bankroll staked on one card
DD_FULL = 0.85          # bankroll/peak above this → no brake
DD_FLOOR = 0.70         # at/below this → maximum brake
BRAKE_MIN = 0.50        # staking multiplier at full drawdown brake
MIN_STAKE = 0.10
CERTAINTY = 0.99        # never stake an implied-certain outcome


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
    """Scale per-bet `stake_gbp` down for drawdown, per-player correlation, and a
    total-card cap. Returns the surviving rows (stakes ≥ min_stake), each
    annotated with `stake_capped` when a cap reduced it."""
    if not rows:
        return rows
    rows = [r for r in rows if float(r.get("p_model", 0.0)) < CERTAINTY]
    if not rows:
        return rows
    peak = peak if peak and peak > 0 else bankroll
    brake = drawdown_factor(bankroll, peak)

    for r in rows:
        r["_raw_stake"] = float(r.get("stake_gbp", 0.0))
        r["stake_gbp"] = round(r["_raw_stake"] * brake, 2)

    # per-player correlation cap (win/match/set markets on one player are nested)
    cap_p = per_player_cap * bankroll
    by_player: dict[str, list] = defaultdict(list)
    for r in rows:
        by_player[r.get("player", "")].append(r)
    for rs in by_player.values():
        s = sum(r["stake_gbp"] for r in rs)
        if s > cap_p > 0:
            f = cap_p / s
            for r in rs:
                r["stake_gbp"] = round(r["stake_gbp"] * f, 2)

    # total-card exposure cap
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
        {"player": "Carlos Alcaraz", "market": "win", "p_model": 0.30, "stake_gbp": 7.0},
        {"player": "Carlos Alcaraz", "market": "match_winner", "p_model": 0.75, "stake_gbp": 6.0},
        {"player": "Carlos Alcaraz", "market": "set_hcp", "p_model": 0.45, "stake_gbp": 4.0},
        {"player": "Jannik Sinner", "market": "match_winner", "p_model": 0.65, "stake_gbp": 5.0},
        {"player": "Jack Draper", "market": "match_winner", "p_model": 0.55, "stake_gbp": 3.0},
    ]
    for peak in (100.0, 130.0):
        rows = [dict(d) for d in demo]
        rows = apply_portfolio(rows, bankroll=100.0, peak=peak)
        print(f"\npeak £{peak:.0f} → {summary(rows, 100.0, peak)}")
        for r in rows:
            flag = "  (capped)" if r.get("stake_capped") else ""
            print(f"  {r['player']:<18}{r['market']:<14} £{r['stake_gbp']:.2f}{flag}")

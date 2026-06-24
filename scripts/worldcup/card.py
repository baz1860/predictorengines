#!/usr/bin/env python3
"""Narrative best-bets card -> card.md (the plain-English companion to the
dashboard).

Where the dashboard (report.py) shows the numbers as tables and charts, this
writes the same week in words: the model's title favourite, every bet it backs
with its own number, the price, and why there's an edge, then its read on the
title race and the upcoming fixtures. It mirrors the golf engine's card.md so
both sports read the same way.

Reads only local files (the daily pipeline already produced them):
  * bet_queue.csv                 — the bets the edge step queued
  * tournament_odds.csv           — champion / reach-final %, the title outlook
  * predictions_worldcup_2026.csv — per-match probabilities, the fixtures read
  * data/ledger.csv               — bankroll and settled record

Re-run anytime:  python3 scripts/worldcup/card.py
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = DATA / "worldcup" / "card.md"

QUEUE = ROOT / "bet_queue.csv"
TITLE_ODDS = ROOT / "tournament_odds.csv"
PREDICTIONS = ROOT / "predictions_worldcup_2026.csv"
LEDGER = DATA / "ledger.csv"

START_BANKROLL = 100.0


# ── render ───────────────────────────────────────────────────────────────────

def render_card() -> str:
    queue = _read_csv(QUEUE)
    title = _read_csv(TITLE_ODDS)
    preds = _read_csv(PREDICTIONS)
    bankroll, record = _bankroll_summary()
    generated = time.strftime("%Y-%m-%d %H:%M")

    L = [
        "# World Cup 2026 — Best Bets",
        "",
        f"_Generated {generated} · blend model · fitted Elo+Poisson_",
        "",
        _lead(queue, title, bankroll),
        "",
        "## Match bets",
        "",
        _bets_section(queue),
        "",
        "## Title outlook",
        "",
        _title_section(title),
        "",
        "## Fixtures forecast",
        "",
        _fixtures_section(preds),
        "",
        "## Notes",
        "",
        _notes_section(queue, record, bankroll),
        "",
    ]
    return "\n".join(L)


def _lead(queue, title, bankroll) -> str:
    """A short plain-English summary of the week before the detail sections."""
    out = ["The model simulated the World Cup field and weighed every price it "
           "could find against its own probabilities."]
    if title:
        fav = max(title, key=lambda r: _num(r.get("champion")))
        out.append(f" It makes **{fav.get('team', '')}** the favourite at "
                   f"{_num(fav.get('champion')) * 100:.0f}% to lift the trophy.")
    n = len(queue)
    if n:
        total = sum(_num(r.get("stake")) for r in queue)
        bets = "bet" if n == 1 else "bets"
        out.append(f" This week it backs **{n} {bets}** (total stake £{total:.2f}) "
                   "— each explained below, with the model's number, the price, and "
                   "why there's an edge. Stakes are fractional-Kelly on a "
                   f"£{bankroll:.0f} bankroll.")
    else:
        out.append(" This week the prices looked efficient — nothing cleared the "
                   "edge threshold, so there are no bets.")
    return "".join(out)


def _bets_section(queue) -> str:
    if not queue:
        return ("The edge step priced every upcoming match, but no bet was "
                "mispriced enough to clear the threshold — no match bets this "
                "week. Re-run `python3 -m engines.worldcup.edge` once fresh odds "
                "are in.")
    rows = sorted(queue, key=lambda r: -_num(r.get("edge")))
    total = sum(_num(r.get("stake")) for r in rows)
    n = len(rows)
    bets = "bet" if n == 1 else "bets"
    out = [
        "Each bet backs the model against the bookmaker's price on an upcoming "
        f"match. It bets only when its own probability beats the price. "
        f"**{n} {bets} cleared the threshold** (total stake £{total:.2f}), "
        "strongest edge first:",
        "",
    ]
    for r in rows[:6]:
        out.append(
            f"- **{r.get('bet', '')}** ({r.get('match', '')}) — "
            f"{_num(r.get('odds')):.2f}. The model makes this "
            f"{_pct(r.get('p_model'))}, against {_pct(r.get('p_book'))} implied by "
            f"the price — a +{_num(r.get('edge')) * 100:.1f}pp edge. "
            f"Stake **£{_num(r.get('stake')):.2f}**.")
    rest = rows[6:]
    if rest:
        out += ["", f"Also backed, at smaller edges ({len(rest)}):", "",
                "| Bet | Match | Odds | Model | Edge | Stake |",
                "|---|---|--:|--:|--:|--:|"]
        out += [
            f"| {r.get('bet', '')} | {r.get('match', '')} "
            f"| {_num(r.get('odds')):.2f} | {_pct(r.get('p_model'))} "
            f"| +{_num(r.get('edge')) * 100:.1f}pp | £{_num(r.get('stake')):.2f} |"
            for r in rest
        ]
    return "\n".join(out)


def _title_section(title, top: int = 12) -> str:
    if not title:
        return ("_No tournament_odds.csv — run "
                "`python3 -m engines.worldcup.simulate` to refresh the title race._")
    rows = sorted(title, key=lambda r: -_num(r.get("champion")))[:top]
    intro = ("Not bets — just the model's own read on the title race, from the "
             "tournament simulation: each side's chance to lift the trophy and to "
             "reach the final.")
    head = "| Team | Grp | Champion | Reach final |\n|---|---|--:|--:|"
    body = [
        f"| {r.get('team', '')} | {r.get('group', '')} "
        f"| {_num(r.get('champion')) * 100:.1f}% "
        f"| {_num(r.get('reach_final')) * 100:.0f}% |"
        for r in rows
    ]
    return intro + "\n\n" + head + "\n" + "\n".join(body)


def _fixtures_section(preds, top: int = 12) -> str:
    if not preds:
        return ("_No predictions_worldcup_2026.csv — run "
                "`python3 -m engines.worldcup.predictor --worldcup`._")
    today = time.strftime("%Y-%m-%d")
    day = [r for r in preds if r.get("date") == today]
    if not day:
        future = sorted({r.get("date") for r in preds if r.get("date", "") > today})
        if future:
            nxt = future[0]
            day = [r for r in preds if r.get("date") == nxt]
    if not day:
        return "_No upcoming fixtures in the prediction file._"
    when = day[0].get("date", "")
    intro = ("Not bets — the model's read on the next matchday "
             f"({when}): each side's win/draw/loss chance and its single most "
             "likely scoreline.")
    head = ("| Match | Home | Draw | Away | BTTS | Likely |\n"
            "|---|--:|--:|--:|--:|---|")
    body = []
    for r in day[:top]:
        btts = r.get("p_btts")
        btts_cell = f"{_num(btts) * 100:.0f}%" if btts not in (None, "") else "—"
        body.append(
            f"| {r.get('home', '')} v {r.get('away', '')} "
            f"| {_num(r.get('p_home')) * 100:.0f}% "
            f"| {_num(r.get('p_draw')) * 100:.0f}% "
            f"| {_num(r.get('p_away')) * 100:.0f}% "
            f"| {btts_cell} | {r.get('likely_score', '')} |")
    return intro + "\n\n" + head + "\n" + "\n".join(body)


def _notes_section(queue, record, bankroll) -> str:
    lines = [f"- Bankroll £{bankroll:.2f}. {record}"]
    adjustments = sorted({r.get("adjustments", "") for r in queue
                          if r.get("adjustments")})
    if adjustments:
        lines.append(f"- Model adjustments active: {', '.join(adjustments)}.")
    lines.append("- Dashboard with charts: `dashboard.html` "
                 "(`python3 scripts/worldcup/report.py`).")
    return "\n".join(lines)


# ── helpers ──────────────────────────────────────────────────────────────────

def _bankroll_summary() -> tuple[float, str]:
    rows = _read_csv(LEDGER)
    settled = [r for r in rows if r.get("status") in ("won", "lost")]
    if not settled:
        return START_BANKROLL, "No settled bets yet."
    bankroll = _num(settled[-1].get("bankroll_after"), START_BANKROLL)
    wins = sum(1 for r in settled if r.get("status") == "won")
    pnl = sum(_num(r.get("pnl")) for r in settled)
    return bankroll, (f"Settled {len(settled)} bets ({wins} won), "
                      f"net £{pnl:+.2f} on a £{START_BANKROLL:.0f} start.")


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _num(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pct(value) -> str:
    return f"{_num(value) * 100:.1f}%"


def main() -> None:
    OUT.write_text(render_card())
    print(f"Wrote {OUT.name}")


if __name__ == "__main__":
    main()

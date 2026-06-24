#!/usr/bin/env python3
"""Narrative best-bets card -> data/worldcup/card.md.

Where the dashboard (report.py) shows the numbers as tables and charts, this
writes the same week in words and — crucially — explains *why* the model lands
where it does: the Elo gap between two sides, how that becomes an expected
scoreline, and how that scoreline turns into a price the bet is measured
against. It mirrors the golf engine's card.md so both sports read the same way.

Reads only local files (the daily pipeline already produced them):
  * bet_queue.csv                 — the bets the edge step queued
  * tournament_odds.csv           — champion / reach-final %, per-team Elo
  * predictions_worldcup_2026.csv — expected goals + per-match probabilities
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
HOSTS = {"United States", "Mexico", "Canada"}


# ── render ───────────────────────────────────────────────────────────────────

def render_card() -> str:
    queue = _read_csv(QUEUE)
    title = _read_csv(TITLE_ODDS)
    preds = _read_csv(PREDICTIONS)
    elo = {r["team"]: r for r in title}
    pred_by_match = {(r.get("home"), r.get("away")): r for r in preds}
    bankroll, record = _bankroll_summary()
    generated = time.strftime("%Y-%m-%d %H:%M")

    L = [
        "# World Cup 2026 — Best Bets",
        "",
        f"_Generated {generated} · blend model · fitted Elo + Poisson_",
        "",
        _lead(queue, title, bankroll),
        "",
        "## How the model thinks",
        "",
        _how_it_works(),
        "",
        "## Match bets",
        "",
        _bets_section(queue, pred_by_match, elo),
        "",
        "## Title outlook",
        "",
        _title_section(title),
        "",
        "## Fixtures forecast",
        "",
        _fixtures_section(preds, elo),
        "",
        "## Notes",
        "",
        _notes_section(queue, record, bankroll),
        "",
    ]
    return "\n".join(L)


def _lead(queue, title, bankroll) -> str:
    """A short plain-English summary of the week before the detail sections."""
    out = ["The model rates every nation on its full international history, turns "
           "each fixture into an expected scoreline, and then weighs its own "
           "probabilities against every price it can find."]
    if title:
        fav = max(title, key=lambda r: _num(r.get("champion")))
        out.append(f" It makes **{fav.get('team', '')}** the tournament favourite "
                   f"at {_num(fav.get('champion')) * 100:.0f}% to lift the trophy.")
    n = len(queue)
    if n:
        total = sum(_num(r.get("stake")) for r in queue)
        bets = "bet" if n == 1 else "bets"
        out.append(f" This week it backs **{n} {bets}** (total stake £{total:.2f}) "
                   "— each explained below, with the model's number, the price, and "
                   "exactly where the edge comes from. Stakes are fractional-Kelly "
                   f"on a £{bankroll:.0f} bankroll.")
    else:
        out.append(" This week the prices looked efficient — nothing cleared the "
                   "edge threshold, so there are no bets.")
    return "".join(out)


def _how_it_works() -> str:
    return (
        "Every number below comes from one pipeline, so it's worth knowing what "
        "drives it:\n\n"
        "1. **Strength (Elo).** Each nation carries an Elo rating built from every "
        "international it has played, with bigger swings for World Cups and big "
        "wins than for friendlies. The gap between two ratings is the model's core "
        "read on who is better, and by how much.\n"
        "2. **Expected goals (Poisson).** That Elo gap is fed through a goal model "
        "fitted on every match since 2010. It turns the gap into an *expected "
        "scoreline* — e.g. 1.8 goals for the stronger side, 0.8 for the weaker — "
        "with a home-field bump only for the host nations (USA, Mexico, Canada) "
        "playing at home.\n"
        "3. **The full grid (Dixon-Coles).** From those two goal expectations the "
        "model builds the probability of every scoreline, nudged to fit how often "
        "low-scoring draws really happen. Summing the grid gives win/draw/loss, "
        "both-teams-to-score and over/under numbers.\n"
        "4. **Edge.** A bet is only listed when the model's probability is enough "
        "above the bookmaker's implied probability to clear the threshold. The "
        "title %s come from running the whole tournament tens of thousands of "
        "times, so they fold in group draws and bracket luck, not just raw strength."
    )


def _bets_section(queue, pred_by_match, elo) -> str:
    if not queue:
        return ("The edge step priced every upcoming match, but nothing was "
                "mispriced enough to clear the threshold — no match bets this "
                "week. Re-run `python3 -m engines.worldcup.edge` once fresh odds "
                "are in.")
    rows = sorted(queue, key=lambda r: -_num(r.get("edge")))
    total = sum(_num(r.get("stake")) for r in rows)
    n = len(rows)
    bets = "bet" if n == 1 else "bets"
    out = [
        "Each bet pits the model's probability against the bookmaker's price on an "
        f"upcoming match; it only fires when its own number is the bigger one. "
        f"**{n} {bets} cleared the threshold** (total stake £{total:.2f}), "
        "strongest edge first.",
        "",
    ]
    for r in rows:
        out.append(_explain_bet(r, pred_by_match, elo))
        out.append("")
    return "\n".join(out).rstrip()


def _explain_bet(r, pred_by_match, elo) -> str:
    """A short paragraph per bet: the header line, then why the edge exists."""
    bet = r.get("bet", "")
    match = r.get("match", "")
    odds = _num(r.get("odds"))
    p_model = _num(r.get("p_model"))
    p_book = _num(r.get("p_book"))
    edge = _num(r.get("edge")) * 100
    stake = _num(r.get("stake"))

    header = (f"### {bet} — {match}\n"
              f"**{odds:.2f}** · model {p_model * 100:.1f}% vs market "
              f"{p_book * 100:.1f}% · **+{edge:.1f}pp edge** · stake "
              f"**£{stake:.2f}**")

    home, away = _split_match(match)
    xgh, xga = _lookup_xg(home, away, pred_by_match)
    why = _why_for_bet(bet, home, away, xgh, xga, elo, p_model, p_book)
    return header + "\n\n" + why


def _lookup_xg(home, away, pred_by_match) -> tuple[float | None, float | None]:
    """Expected goals oriented to the bet's own home/away. The edge step and the
    predictor sometimes order a fixture the opposite way, so try both and swap."""
    pred = pred_by_match.get((home, away))
    if pred is not None:
        return _num(pred.get("xg_home")), _num(pred.get("xg_away"))
    pred = pred_by_match.get((away, home))
    if pred is not None:  # reversed fixture — swap so xgh matches the bet's home
        return _num(pred.get("xg_away")), _num(pred.get("xg_home"))
    return None, None


def _why_for_bet(bet, home, away, xgh, xga, elo, p_model, p_book) -> str:
    """Plain-English reasoning, tracing Elo gap -> expected goals -> price."""
    if xgh is None or not home:
        return ("The model's probability sits above the price by enough to bet; "
                "the underlying match detail wasn't found in the prediction file.")
    total = xgh + xga
    elo_line = _elo_line(home, away, elo)
    bl = bet.lower()

    if bl.endswith("win"):
        team = bet[: -len("win")].strip()
        is_home = team == home
        opp = away if is_home else home
        tg, og = (xgh, xga) if is_home else (xga, xgh)
        host_note = (" — and the host-nation home bump on top"
                     if is_home and home in HOSTS else "")
        return (f"{elo_line} Run through the goal model that comes out as an "
                f"expected **{tg:.2f}–{og:.2f}** in {team}'s favour{host_note}, "
                f"and once every scoreline is added up {team} win it "
                f"**{p_model * 100:.0f}%** of the time. The {_implied(p_book)} "
                f"price baked into the odds is too generous for a side the model "
                f"likes this much over {opp}.")

    if "over" in bl:
        return (f"{elo_line} The two attacks project to **{xgh:.2f} + {xga:.2f} = "
                f"{total:.2f}** expected goals, {_total_word(total)} the 2.5 line. "
                f"That makes Over a **{p_model * 100:.0f}%** shot, where the price "
                f"only allows {_implied(p_book)} — the market is pricing a tighter "
                f"game than the model sees.")

    if "under" in bl:
        return (f"{elo_line} Between them the sides project to only **{xgh:.2f} + "
                f"{xga:.2f} = {total:.2f}** expected goals, {_total_word(total)} "
                f"the 2.5 line, so the model leans Under at **{p_model * 100:.0f}%** "
                f"against the {_implied(p_book)} the price implies — it expects a "
                f"cagier match than the bookmaker.")

    if "draw" in bl:
        return (f"{elo_line} With an expected **{xgh:.2f}–{xga:.2f}** the sides are "
                f"close enough that the model gives the draw **{p_model * 100:.0f}%**, "
                f"more than the {_implied(p_book)} in the price.")

    if "both teams" in bl or bl.startswith("btts"):
        return (f"{elo_line} Both attacks are live in the expected **{xgh:.2f}–"
                f"{xga:.2f}**, putting both-teams-to-score at **{p_model * 100:.0f}%** "
                f"versus the {_implied(p_book)} priced in.")

    return (f"{elo_line} The model makes this **{p_model * 100:.0f}%** against the "
            f"{_implied(p_book)} in the price, an edge worth backing.")


def _elo_line(home, away, elo) -> str:
    eh = _num((elo.get(home) or {}).get("elo"))
    ea = _num((elo.get(away) or {}).get("elo"))
    if not eh or not ea:
        return ""
    gap = abs(eh - ea)
    stronger, weaker = (home, away) if eh >= ea else (away, home)
    if gap < 25:
        return (f"The model rates {home} (Elo {eh:.0f}) and {away} (Elo {ea:.0f}) "
                f"as near-equals — a gap of just {gap:.0f} points.")
    return (f"The model rates {stronger} at Elo {max(eh, ea):.0f} against "
            f"{weaker}'s {min(eh, ea):.0f}, a {gap:.0f}-point edge to {stronger}.")


def _title_section(title, top: int = 12) -> str:
    if not title:
        return ("_No tournament_odds.csv — run "
                "`python3 -m engines.worldcup.simulate` to refresh the title race._")
    rows = sorted(title, key=lambda r: -_num(r.get("champion")))
    lead = _title_prose(rows)
    intro = ("These aren't bets — they're the model's read on the title race, "
             "straight from the tournament simulation. Each side's chance to lift "
             "the trophy already folds in its group draw and likely knockout path, "
             "which is why raw Elo order and these numbers don't match exactly.")
    head = "| Team | Grp | Champion | Reach final |\n|---|---|--:|--:|"
    body = [
        f"| {r.get('team', '')} | {r.get('group', '')} "
        f"| {_num(r.get('champion')) * 100:.1f}% "
        f"| {_num(r.get('reach_final')) * 100:.0f}% |"
        for r in rows[:top]
    ]
    return lead + "\n\n" + intro + "\n\n" + head + "\n" + "\n".join(body)


def _title_prose(rows) -> str:
    if not rows:
        return ""
    fav = rows[0]
    parts = [
        f"**{fav.get('team', '')}** head the field: highest Elo in the draw "
        f"({_num(fav.get('elo')):.0f}) and champions in "
        f"**{_num(fav.get('champion')) * 100:.0f}%** of simulated tournaments."]
    if len(rows) > 1:
        chal = rows[1]
        parts.append(
            f" {chal.get('team', '')} are the closest challenger at "
            f"{_num(chal.get('champion')) * 100:.0f}%")
        if len(rows) > 3:
            pack = ", ".join(r.get("team", "") for r in rows[2:4])
            parts.append(f", with {pack} heading the chasing pack")
        parts.append(".")
    return "".join(parts)


def _fixtures_section(preds, elo, top: int = 12) -> str:
    if not preds:
        return ("_No predictions_worldcup_2026.csv — run "
                "`python3 -m engines.worldcup.predictor --worldcup`._")
    today = time.strftime("%Y-%m-%d")
    day = [r for r in preds if r.get("date") == today]
    if not day:
        future = sorted({r.get("date") for r in preds if r.get("date", "") > today})
        if future:
            day = [r for r in preds if r.get("date") == future[0]]
    if not day:
        return "_No upcoming fixtures in the prediction file._"
    when = day[0].get("date", "")
    out = ["Not bets — the model's read on the next matchday "
           f"({when}). For each game: the expected scoreline that falls out of the "
           "Elo gap, and where the probability lands.", ""]
    for r in day[:top]:
        out.append(_fixture_line(r, elo))
    return "\n".join(out)


def _fixture_line(r, elo) -> str:
    home, away = r.get("home", ""), r.get("away", "")
    xgh, xga = _num(r.get("xg_home")), _num(r.get("xg_away"))
    ph, pd, pa = _num(r.get("p_home")), _num(r.get("p_draw")), _num(r.get("p_away"))
    likely = r.get("likely_score", "")
    pick, p = max(((home, ph), ("a draw", pd), (away, pa)), key=lambda t: t[1])
    eh = _num((elo.get(home) or {}).get("elo"))
    ea = _num((elo.get(away) or {}).get("elo"))
    elo_bit = f" (Elo {eh:.0f} v {ea:.0f})" if eh and ea else ""
    favour = ("an even game" if pick == "a draw"
              else f"{pick} favoured at {p * 100:.0f}%")
    return (f"- **{home} v {away}**{elo_bit}: expected **{xgh:.2f}–{xga:.2f}**, "
            f"most likely {likely} — {favour}.")


def _notes_section(queue, record, bankroll) -> str:
    lines = [f"- Bankroll £{bankroll:.2f}. {record}"]
    adjustments = sorted({r.get("adjustments", "") for r in queue
                          if r.get("adjustments")})
    if adjustments:
        lines.append(f"- Model adjustments active this run: {', '.join(adjustments)}.")
    lines.append("- Same numbers as charts: `dashboard.html` "
                 "(`python3 scripts/worldcup/report.py`).")
    return "\n".join(lines)


# ── helpers ──────────────────────────────────────────────────────────────────

def _split_match(match: str) -> tuple[str, str]:
    parts = match.split(" v ", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return match.strip(), ""


def _total_word(total: float) -> str:
    if total >= 3.0:
        return "comfortably above"
    if total >= 2.7:
        return "above"
    if total >= 2.5:
        return "just above"
    if total >= 2.3:
        return "just below"
    if total >= 2.0:
        return "below"
    return "well below"


def _implied(p_book: float) -> str:
    return f"{p_book * 100:.0f}%"


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


def main() -> None:
    OUT.write_text(render_card())
    print(f"Wrote {OUT.name}")


if __name__ == "__main__":
    main()

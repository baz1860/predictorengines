#!/usr/bin/env python3
"""The whole suite on one page: every engine's bets for the day -> daily_card.md.

Where each engine writes its own card (data/worldcup/card.md, golf/data/card.md,
club_soccer/data/card.md, tennis/data/card.md), this asks every registered
engine for its current priced board — the same read-only edge preview the app
and daily_summary.py use — keeps the recommended bets, and writes ONE narrative
document that explains each of them in plain English: what the bet is, what the
model believes, what the price implies, and where the gap comes from.

Nothing is recorded to the ledger here (previews only), and an engine that
can't price today (offseason, no odds filled in, no API key) degrades to a
one-line explanation instead of failing the card.

Run it directly anytime:

    python3 scripts/daily_card.py            # -> daily_card.md at the repo root
    python3 scripts/daily_card.py --out X.md

or as the last step of the one-command refresh:  ./daily_card.sh
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "daily_card.md"

# Presentation order (registry order is registration order, which is arbitrary).
DISPLAY_ORDER = ["worldcup", "club_soccer", "golf", "tennis", "nfl", "cfb", "nhl"]

# How far ahead a bet may be dated and still make today's card.
HORIZON_DAYS = 8

ENGINE_CARDS = {
    "worldcup": "data/worldcup/card.md",
    "club_soccer": "club_soccer/data/card.md",
    "golf": "golf/data/card.md",
    "tennis": "tennis/data/card.md",
}

# One-sentence standing intro per engine, in the suite's own voice.
ENGINE_INTROS = {
    "worldcup": (
        "The World Cup model rates every nation on its full international "
        "history, turns each fixture into an expected scoreline, and only "
        "speaks up when its number beats the price."),
    "club_soccer": (
        "The club model does the same job league by league — team strength "
        "from results, goals from a fitted attack/defence model, and a "
        "do-not-bet filter for markets it knows it prices badly."),
    "golf": (
        "The golf model simulates the whole tournament thousands of times "
        "from each player's fitted skill and variance, so a price on a win, "
        "a top-10 or a made cut can be checked against an honest probability."),
    "tennis": (
        "The tennis model rates players by surface and recent form, prices "
        "every match in the live draw, and backs a player only when the "
        "market underrates them."),
    "nfl": (
        "The NFL engine builds power ratings from play-level efficiency and "
        "quarterback form, and bets the spread when its projected margin "
        "disagrees with the line."),
    "cfb": (
        "The college football engine blends power ratings with market priors "
        "and looks for lines that lag what the ratings already know."),
    "nhl": (
        "The NHL model prices moneylines, puck lines and totals from team "
        "scoring rates."),
}


# ── collection ────────────────────────────────────────────────────────────────

def collect() -> dict:
    """Ask every engine for its board (read-only). Returns everything render()
    needs; every failure is captured as a per-engine reason, never raised."""
    from app import bankroll_store, model_audit
    from app.engines import registry

    today = date.today()
    horizon = (today + timedelta(days=HORIZON_DAYS)).isoformat()
    yesterday = (today - timedelta(days=1)).isoformat()

    engines = {e.id: e for e in registry.all()}
    order = [i for i in DISPLAY_ORDER if i in engines]
    order += [i for i in engines if i not in order]

    sections = []
    for eid in order:
        eng = engines[eid]
        sec = {"id": eid, "name": eng.name, "sport": getattr(eng, "sport", ""),
               "bets": [], "n_priced": 0, "note": "", "reason": ""}
        try:
            audit = model_audit.audit(eid)
            sec["gate"] = audit["validation"]["status"]
            sec["freshness"] = audit.get("freshness_warnings") or []
        except Exception:
            sec["gate"], sec["freshness"] = "unknown", []
        if "edge" not in getattr(eng, "capabilities", set()):
            sec["reason"] = "no odds-pricing capability"
            sections.append(sec)
            continue
        try:
            res = eng.edge({})  # no record -> preview only
            rows = res.get("rows") or []
            sec["n_priced"] = len(rows)
            sec["note"] = str(res.get("note") or "")
            picks = [r for r in rows if r.get("recommended")]
            picks = [r for r in picks
                     if not r.get("match_date")
                     or yesterday <= str(r["match_date"]) <= horizon]
            picks.sort(key=lambda r: (str(r.get("match_date") or ""),
                                      -float(r.get("edge") or 0)))
            sec["bets"] = picks
        except Exception as exc:
            sec["reason"] = str(exc).strip() or exc.__class__.__name__
        sections.append(sec)

    try:
        bk = bankroll_store.status_summary()
        bankroll = {"bankroll": bk["bankroll"],
                    "net_pnl": bk["totals"]["net_pnl"],
                    "open": bk["totals"]["open_count"]}
    except Exception:
        bankroll = {"bankroll": None, "net_pnl": None, "open": None}

    return {"date": today, "sections": sections, "bankroll": bankroll,
            "generated": time.strftime("%Y-%m-%d %H:%M")}


# ── narrative helpers ─────────────────────────────────────────────────────────

def _pct(x) -> str:
    v = float(x or 0) * 100
    return f"{v:.1f}%" if 0 < v < 10 else f"{v:.0f}%"


def _num(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _price_phrase(odds: float) -> str:
    if odds <= 1.4:
        return "a short price"
    if odds <= 1.9:
        return "an odds-on price"
    if odds <= 2.2:
        return "close to even money"
    if odds <= 3.5:
        return "a fair-sized price"
    return "a proper longshot price"

def _edge_phrase(edge: float, i: int = 0) -> str:
    pp = edge * 100
    if pp < 2:
        variants = ["a thin but real edge", "a small edge, but a real one",
                    "a modest edge the maths still likes"]
    elif pp < 4:
        variants = ["a healthy gap", "a solid bit of value",
                    "a gap worth taking"]
    elif pp < 8:
        variants = ["a serious disagreement with the market",
                    "a proper difference of opinion",
                    "a gap too wide to ignore"]
    else:
        variants = ["a genuinely large gap",
                    "as big a disagreement as this market usually offers",
                    "a rare double-digit-sized edge"]
    return variants[i % len(variants)]


def _when_phrase(match_date: str, today: date) -> str:
    if not match_date:
        return ""
    try:
        d = date.fromisoformat(str(match_date)[:10])
    except ValueError:
        return ""
    if d == today:
        return "today"
    if d == today + timedelta(days=1):
        return "tomorrow"
    return d.strftime("on %A")


def _market_key(r: dict) -> str:
    """Collapse each engine's market spelling to one prose key."""
    m = str(r.get("market") or "").strip().lower()
    side = str(r.get("side") or "").strip().lower()
    if m.startswith("matchup") or side.startswith("matchup"):
        return "matchup"
    if m.startswith("3ball") or side.startswith("3ball") or "3_ball" in m:
        return "3ball"
    if m in ("match_winner", "match"):
        return "match"
    if m.startswith("win"):
        return "win"
    flat = m.replace("_", "")
    for t in ("top5", "top10", "top20"):
        if flat.startswith(t):
            return t
    if "cut" in m:
        return "cut"
    return m  # 1x2 / ml / total / spread / btts arrive already normalized


def _matchup_opponent(r: dict) -> str:
    """Golf matchups carry both names in side ('matchup:A|B'); return the one
    that isn't the entrant. Falls back to the 'Matchup vs X' bet text."""
    home = str(r.get("home") or "").strip()
    side = str(r.get("side") or "")
    if ":" in side:
        names = [n.strip() for n in side.split(":", 1)[1].split("|") if n.strip()]
        others = [n for n in names if n.lower() != home.lower()]
        if others:
            return " and ".join(others)
    bet = str(r.get("bet") or "")
    if "vs " in bet:
        return bet.split("vs ", 1)[1].split("—")[0].split(" - ")[0].strip()
    return ""


def _bet_label(r: dict) -> str:
    """A headline a person would actually say, not a machine key."""
    bet = str(r.get("bet") or "").strip()
    home, away = str(r.get("home") or ""), str(r.get("away") or "")
    mk = _market_key(r)
    if mk == "match" and away:
        return f"{home} to beat {away}"
    if mk == "matchup":
        opp = _matchup_opponent(r)
        return f"{home} to beat {opp}" if opp else (bet or home)
    if mk == "3ball":
        return f"{home} to win the 3-ball"
    if mk == "win":
        return f"{home} to win outright"
    if mk in ("top5", "top10", "top20"):
        return f"{home} top-{mk[3:]} finish"
    if mk == "cut":
        return f"{home} to make the cut"
    label = bet or f"{r.get('market', '')} {r.get('side', '')}".strip()
    fixture = f"{home} v {away}" if away else home
    if fixture and fixture.lower() not in label.lower():
        label = f"{label} — {fixture}"
    return label


def _bet_header(r: dict) -> str:
    odds = _num(r.get("odds"))
    stat = (f"**{odds:.2f}** · model {_pct(r.get('p_model'))} vs market "
            f"{_pct(r.get('p_book'))} · **+{_num(r.get('edge')) * 100:.1f}pp**")
    stake = _num(r.get("stake_gbp"))
    if stake > 0:
        stat += f" · stake **£{stake:.2f}**"
    return f"### {_bet_label(r)}\n{stat}"


def _bet_prose(r: dict, sport: str, today: date, i: int = 0) -> str:
    """One communicative paragraph: what the model believes, what the price
    says, and why that difference is worth money. `i` rotates phrasing so a
    run of similar bets doesn't read like a mail merge."""
    mk = _market_key(r)
    side = str(r.get("side") or "").lower()
    home, away = str(r.get("home") or ""), str(r.get("away") or "")
    bet = str(r.get("bet") or "")
    line = str(r.get("line") or "").strip()
    odds = _num(r.get("odds"))
    p_model, p_book = _num(r.get("p_model")), _num(r.get("p_book"))
    edge = _num(r.get("edge"))
    when = _when_phrase(str(r.get("match_date") or ""), today)
    when_bit = f" {when}" if when else ""

    price = _price_phrase(odds)
    gap = _edge_phrase(edge, i)

    if mk == "total":
        lean = ("an open, chance-trading game" if side.startswith("over")
                else "a tighter, more careful game")
        s = (f"The model expects {lean} between {home} and {away}{when_bit}: it "
             f"makes the {side or 'total'}{' ' + line if line else ''} a "
             f"{_pct(p_model)} shot, while the price of {odds:.2f} only asks for "
             f"{_pct(p_book)}. That's {gap} — the market is pricing a different "
             f"kind of game than the model projects.")
    elif mk == "btts":
        s = (f"Both attacks look live to the model{when_bit}: it has both teams "
             f"scoring {_pct(p_model)} of the time against the {_pct(p_book)} "
             f"implied by {odds:.2f} — {gap}.")
    elif mk in ("1x2", "ml") and "draw" in (side + bet.lower()):
        s = (f"The model sees {home} and {away} as close enough{when_bit} that "
             f"the draw lands {_pct(p_model)} of the time — more often than the "
             f"{_pct(p_book)} the price allows. Draws are unfashionable bets, "
             f"which is usually exactly when they're worth taking.")
    elif mk in ("1x2", "ml"):
        team = home if side == "home" else away if side == "away" else (bet.split(" win")[0].strip() or home)
        opp = away if team == home else home
        s = (f"A straight call on {team} to beat {opp}{when_bit}. The model "
             f"wins this match for {team} {_pct(p_model)} of the time; at "
             f"{odds:.2f} the book only needs {_pct(p_book)} to break even, so "
             f"every point above that is value — and this is {gap}.")
    elif mk == "spread":
        team = home if side == "home" else away
        line_bit = f" {line}" if line else ""
        s = (f"A handicap bet: {team}{line_bit}{when_bit}. The model's projected "
             f"margin covers this number {_pct(p_model)} of the time against the "
             f"{_pct(p_book)} implied at {odds:.2f} — the line hasn't moved as "
             f"far as the ratings say it should, and the leftover is {gap}.")
    elif mk in ("win", "top5", "top10", "top20", "cut"):
        does = {"win": "wins the tournament", "top5": "finishes in the top 5",
                "top10": "finishes in the top 10",
                "top20": "finishes in the top 20",
                "cut": "makes the cut"}[mk]
        s = (f"Across thousands of simulated tournaments, {home} {does} "
             f"{_pct(p_model)} of the time; the book's {odds:.2f} implies only "
             f"{_pct(p_book)} — {gap}.")
        if mk == "win" and p_model < 0.15:
            s += (f" To be clear, nobody is calling {home} likely to win — the "
                  f"bet is simply that it happens more often than the price "
                  f"admits, and at {odds:.2f} it doesn't need to happen often "
                  f"to pay for itself.")
    elif mk == "matchup":
        opp = _matchup_opponent(r) or "the group"
        variants = [
            (f"A round matchup — {home} against {opp}, lowest score over the "
             f"round wins. The simulation has {home} in front {_pct(p_model)} "
             f"of the time, where {odds:.2f} implies {_pct(p_book)}: {gap}."),
            (f"Head-to-head against {opp}. Strip out the leaderboard and just "
             f"race these two over 18 holes: the model makes {home} the better "
             f"scorer {_pct(p_model)} of the time, and the price is still "
             f"paying as if it were {_pct(p_book)}."),
            (f"{home} versus {opp}, straight up. Matchup markets are priced "
             f"loosely — books lean on tour averages, the model leans on "
             f"current form and course fit — and here that difference is worth "
             f"{edge * 100:.1f} points of probability to {home}."),
        ]
        s = variants[i % len(variants)]
    elif mk == "3ball":
        s = (f"A 3-ball: {home} to post the low score of the group. The model "
             f"makes that a {_pct(p_model)} shot against the {_pct(p_book)} in "
             f"the price — {gap} in a market the books price off rough "
             f"averages.")
    elif mk == "match" or sport == "tennis" or (away and mk == ""):
        underdog = (f" {home} is still the underdog here — the bet isn't that "
                    f"they probably win, it's that they win more often than "
                    f"{odds:.2f} says." if p_model < 0.4 else "")
        variants = [
            (f"The model — surface and recent form baked in — gives {home} a "
             f"{_pct(p_model)} chance against {away}{when_bit}, while "
             f"{odds:.2f} implies only {_pct(p_book)}: {gap}.{underdog}"),
            (f"At {odds:.2f} the market says {home} beats {away} about "
             f"{_pct(p_book)} of the time{when_bit}. The model's rating says "
             f"{_pct(p_model)}. That disagreement is the whole bet — {gap}."
             f"{underdog}"),
            (f"The model likes {home} over {away}{when_bit} more than the "
             f"price does: {_pct(p_model)} against an implied {_pct(p_book)}. "
             f"{gap.capitalize()}, at {price}.{underdog}"),
        ]
        s = variants[i % len(variants)]
    else:
        s = (f"The model prices this at {_pct(p_model)} against the "
             f"{_pct(p_book)} implied by {odds:.2f} — {gap}, enough to clear "
             f"the betting threshold.")

    extra = _row_caveats(r)
    return s + (f" {extra}" if extra else "")


def _row_caveats(r: dict) -> str:
    """Engine-specific colour when the row carries it (e.g. World Cup lineups)."""
    bits = []
    lu = str(r.get("lineup_status") or "").strip().lower()
    if lu in ("confirmed", "official"):
        bits.append("Lineups are confirmed, so no squad surprises are hiding in this number.")
    elif lu in ("projected", "expected"):
        bits.append("Lineups are still projected, not confirmed — worth a glance before kick-off.")
    conf = r.get("availability_confidence")
    if conf not in (None, "") and _num(conf) and _num(conf) < 0.7:
        bits.append(f"Squad-availability confidence is only {_pct(conf)}.")
    return " ".join(bits)


def _friendly_reason(reason: str) -> str:
    rl = reason.lower()
    if "api key" in rl:
        return "needs an odds API key before it can price anything"
    if "odds.csv" in rl and ("not found" in rl or "template" in rl):
        return "is waiting for an odds file — write the template and fill in prices"
    if "no filled-in rows" in rl or "no odds" in rl:
        return "has no odds filled in yet, so there is nothing to price"
    if "draw" in rl and ("load" in rl or "not found" in rl or "empty" in rl):
        return "has no live draw loaded — likely between tournaments"
    if "urlopen" in rl or "network" in rl or "connection" in rl or "resolve" in rl:
        return "couldn't reach its odds feed (offline?)"
    return f"couldn't price a board today ({reason.rstrip('.')})"


# ── render ────────────────────────────────────────────────────────────────────

def render(card: dict) -> str:
    today: date = card["date"]
    sections = card["sections"]
    bk = card["bankroll"]
    active = [s for s in sections if s["bets"]]
    quiet = [s for s in sections if not s["bets"]]
    all_bets = [(s, b) for s in active for b in s["bets"]]

    L = [f"# The Daily Card — {today.strftime('%A %-d %B %Y')}", "",
         f"_Generated {card['generated']} · every engine refreshed and "
         f"re-priced · previews only, nothing recorded._", ""]

    L += [_lead(active, quiet, all_bets, bk), ""]

    if all_bets:
        L += ["## The day at a glance", "", _glance_table(all_bets), ""]

    for s in active:
        L += [f"## {s['name']}", ""]
        intro = ENGINE_INTROS.get(s["id"], "")
        n = len(s["bets"])
        count = ("One bet made the card today."
                 if n == 1 else f"{n} bets made the card today.")
        priced = (f" It priced {s['n_priced']} markets to find them."
                  if s["n_priced"] > n else "")
        L += [f"{intro} {count}{priced}".strip(), ""]
        for i, b in enumerate(s["bets"]):
            L += [_bet_header(b), "", _bet_prose(b, s["sport"], today, i), ""]
        deep = ENGINE_CARDS.get(s["id"])
        if deep:
            L += [f"_Full workings for this sport: `{deep}`._", ""]

    if quiet:
        L += ["## Quiet today", "",
              "No value doesn't mean no work — each of these engines refreshed "
              "and looked, and here's why nothing made the card:", ""]
        for s in quiet:
            L.append(f"- **{s['name']}** — {_quiet_line(s)}")
        L.append("")

    L += ["## Notes & housekeeping", ""]
    L += _notes(sections, bk)
    L.append("")
    return "\n".join(L)


def _lead(active, quiet, all_bets, bk) -> str:
    if not all_bets:
        return ("A quiet day: every engine refreshed its data and re-priced its "
                "markets, and nothing anywhere cleared the edge threshold. That "
                "is the system working — the discipline that makes the betting "
                "days pay is the willingness to sit out the days like this one.")
    n = len(all_bets)
    sports = []
    for s in active:
        k = len(s["bets"])
        sports.append(f"{k} from {s['name']}" if k > 1
                      else f"one from {s['name']}")
    listing = sports[0] if len(sports) == 1 else \
        ", ".join(sports[:-1]) + f" and {sports[-1]}"
    total = sum(_num(b.get("stake_gbp")) for _, b in all_bets)
    best_s, best_b = max(all_bets, key=lambda t: _num(t[1].get("edge")))
    out = [f"**{n} bet{'s' if n != 1 else ''} today** — {listing}."]
    if total > 0:
        out.append(f" Total stake £{total:.2f}, sized by fractional Kelly")
        if bk.get("bankroll") is not None:
            out.append(f" on a £{bk['bankroll']:.2f} bankroll")
        out.append(".")
    out.append(f" The strongest opinion on the board is **{_bet_label(best_b)}"
               f"** ({best_s['name']}), where the model and the market are "
               f"{_num(best_b.get('edge')) * 100:.1f} percentage points apart. "
               "Every bet is explained below — the model's number, the price, "
               "and where the gap comes from.")
    return "".join(out)


def _glance_table(all_bets) -> str:
    head = ("| Sport | When | Bet | Odds | Model | Market | Edge | Stake |\n"
            "|---|---|---|--:|--:|--:|--:|--:|")
    rows = []
    for s, b in all_bets:
        stake = _num(b.get("stake_gbp"))
        rows.append(
            f"| {s['name']} | {b.get('match_date') or '—'} | {_bet_label(b)} "
            f"| {_num(b.get('odds')):.2f} | {_pct(b.get('p_model'))} "
            f"| {_pct(b.get('p_book'))} | +{_num(b.get('edge')) * 100:.1f}pp "
            f"| {'£%.2f' % stake if stake > 0 else '—'} |")
    return head + "\n" + "\n".join(rows)


def _quiet_line(s: dict) -> str:
    if s["reason"]:
        return _friendly_reason(s["reason"]) + "."
    if s["n_priced"]:
        return (f"priced {s['n_priced']} markets and found the books' numbers "
                "close enough to its own that nothing was worth backing.")
    return "priced its board and found nothing to bet."


def _notes(sections, bk) -> list[str]:
    lines = []
    if bk.get("bankroll") is not None:
        lines.append(f"- Bankroll £{bk['bankroll']:.2f} · net P&L "
                     f"£{bk['net_pnl']:+.2f} · {bk['open']} open bet(s).")
    gates = [s for s in sections
             if str(s.get("gate", "")).strip().lower()
             not in ("ok", "pass", "passed", "green", "unknown", "")]
    for s in gates:
        lines.append(f"- ⚠ {s['name']} validation gate: **{s['gate']}** — "
                     "treat its bets with extra care until it's back to green.")
    stale = [(s["name"], w) for s in sections for w in s.get("freshness", [])]
    if stale:
        lines.append("- Data freshness flags: " +
                     "; ".join(f"{n}: {w}" for n, w in stale[:6]) +
                     (" …" if len(stale) > 6 else ""))
    lines.append("- Per-sport detail cards: " +
                 ", ".join(f"`{p}`" for p in ENGINE_CARDS.values()) + ".")
    lines.append("- Regenerate anytime: `./daily_card.sh` (full refresh) or "
                 "`python3 scripts/daily_card.py` (re-render only).")
    return lines


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Combined daily betting card")
    ap.add_argument("--out", default=str(OUT), help="output markdown path")
    args = ap.parse_args()
    card = collect()
    out = Path(args.out)
    out.write_text(render(card))
    n = sum(len(s["bets"]) for s in card["sections"])
    print(f"Wrote {out} — {n} bet(s) across "
          f"{sum(1 for s in card['sections'] if s['bets'])} engine(s).")


if __name__ == "__main__":
    main()

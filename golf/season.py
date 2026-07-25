"""golf/season.py — the one front door for the golf engine.

Same mental model as the World Cup engine: pull the season's tournament list,
take a tournament's field, let the fitted model price it, and print the best
bets for that event — round by round. Everything else in this package
(`refresh`, `simulate`, `edge`, `round_pricer`, …) is plumbing this drives; you
should not need to call those directly for a normal week.

    python -m golf.season                 # this week's card (refresh → price)
    python -m golf.season --schedule      # the season's tournament list
    python -m golf.season --no-refresh    # reprice from cached data only
    python -m golf.season --round 2       # also price this round's 3-balls
    python -m golf.season --season 2026   # schedule for a specific season

The card is intentionally lean: it shows the **bets the model actually backs**
(staked, +EV, above the edge threshold) grouped by round, plus a short field
forecast for context. Sides the model prices but does not recommend are left out
so the page is signal, not noise.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import time
from pathlib import Path

from . import edge as GEDGE
from . import engine as GENG
from . import refresh as GREF
from .providers.espn import EspnGolfProvider
from .io_utils import atomic_write_text

DATA_DIR = Path(__file__).parent / "data"
PREDICTIONS_CSV = DATA_DIR / "predictions.csv"
EDGE_CSV = DATA_DIR / "edge_report.csv"
ROUND_3BALL_CSV = DATA_DIR / "round_edges.csv"
MANIFEST_JSON = DATA_DIR / "free_source_manifest.json"
CARD_MD = DATA_DIR / "card.md"
LIVE_STATE_JSON = DATA_DIR / "live_state.json"

TOTAL_ROUNDS = 4

# Markets that settle on the 72-hole tournament (priced pre-tournament).
_TOURNAMENT_MARKETS = ("win", "top", "cut", "make cut", "matchup")
_MAJOR_HINTS = ("masters", "pga championship", "u.s. open", "us open",
                "the open", "open championship")


# ── schedule ──────────────────────────────────────────────────────────────────

def schedule(season: int | None = None, *, use_cache: bool = True) -> list:
    """The season's PGA tournament list (ESPN), earliest first."""
    return EspnGolfProvider().schedule(season=season, use_cache=use_cache)


def print_schedule(season: int | None = None) -> None:
    season = season or dt.date.today().year
    events = schedule(season)
    print(f"PGA schedule {season} — {len(events)} events\n")
    if not events:
        print("  (no events returned — ESPN may be offline)")
        return
    today = dt.date.today().isoformat()
    for ev in events:
        marker = "→" if ev.start_date >= today else " "
        cname = getattr(ev, "course_name", "") or ""
        course = f" · {cname}" if cname and cname != ev.name else ""
        print(f"  {marker} {ev.start_date}  {ev.name}{course}")
    print("\n→ marks upcoming events. Price the current one with: python -m golf.season")


# ── in-play conditioning ─────────────────────────────────────────────────────
# The engine already ascertains the current event and its completed rounds:
# refresh reads the live leaderboard (espn.completed_round_scores) and records
# live_state.json + scores_live.csv, then engine.cmd_simulate / cmd_edge
# auto-route to the in-play simulator conditioned on that state. The front door
# just sequences those and mirrors the in-play projection into the canonical
# predictions.csv so every downstream reader sees the live-conditioned numbers.

def _mirror_inplay_predictions(sim_out: dict, path: Path = PREDICTIONS_CSV) -> None:
    """Write predictions.csv from an in-play cmd_simulate result.

    Keeps the pre-tournament schema (so existing readers keep working) and adds a
    `score_thru` column. cmd_simulate writes predictions_inplay.csv itself; this
    makes predictions.csv the same live-conditioned board rather than a stale
    pre-event one.
    """
    rows = sim_out.get("rows") or []
    cols = ["rank", "name", "rating", "sigma", "owgr", "win_pct", "top5_pct",
            "top10_pct", "top20_pct", "cut_pct", "avg_finish", "score_thru"]
    from .io_utils import atomic_write_csv
    output_rows = []
    for rank, r in enumerate(rows, 1):
        output_rows.append({
                "rank": rank, "name": r.get("name", ""),
                "rating": f"{_num(r.get('rating')):+.3f}",
                "sigma": "", "owgr": "",
                "win_pct": f"{_num(r.get('win')) * 100:.2f}",
                "top5_pct": f"{_num(r.get('top5')) * 100:.1f}",
                "top10_pct": f"{_num(r.get('top10')) * 100:.1f}",
                "top20_pct": f"{_num(r.get('top20')) * 100:.1f}",
                "cut_pct": "100.0",
                "avg_finish": f"{_num(r.get('avg_finish')):.1f}",
                "score_thru": r.get("score", ""),
        })
    atomic_write_csv(path, cols, output_rows)


def _is_inplay(sim_out: dict) -> bool:
    """cmd_simulate exposes a live-only 'score' (thru) column when in-play."""
    return any(c.get("key") == "score" for c in sim_out.get("columns") or [])



# ── card ────────────────────────────────────────────────────────────────────

def build_card(
    *,
    season: int | None = None,
    event_id: str = "",
    round_no: int | None = None,
    sims: int = 50_000,
    refresh: bool = True,
    stats: bool = False,
    weather: bool = False,
    fit: bool = False,
    major: bool | None = None,
    course: str = "",
    min_edge: float = 1.0,
    seed: int = 7,
    output: Path = CARD_MD,
) -> dict:
    """Run the weekly pipeline and write a round-by-round best-bets card.

    Returns a summary dict (event, counts, output path). The heavy lifting is
    delegated to the existing engine commands so the model/calibration/portfolio
    behaviour is identical to running them by hand — this just sequences them and
    curates the output.
    """
    notes: list[str] = []
    event: dict = {}

    # Refresh ascertains the current event (ESPN) and records the live state —
    # which rounds are complete — into live_state.json. cmd_simulate / cmd_edge
    # then auto-route to the in-play simulator off that state.
    refresh_round = None
    if refresh:
        # The round is only *finalised* by the live-leaderboard read inside the
        # refresh, but the refresh also needs the round up front to load the
        # right per-round manual board (threeballs_r{round}_raw.txt). Pre-detect
        # it from the last known live state so a --round-less run still reads the
        # current round's paste; any change is reconciled below.
        prelim = _read_json(LIVE_STATE_JSON)
        prelim_done = int(prelim.get("rounds_done") or 0)
        refresh_round = (round_no if round_no is not None
                         else (prelim_done + 1 if prelim_done else 1))
        manifest = GREF.run_refresh(season=season, event=event_id, stats=stats,
                                    weather=weather, fit=fit, round_no=refresh_round)
        qa_errors = ((manifest.get("qa") or {}).get("errors") or [])
        if qa_errors:
            detail = "; ".join(
                str(row.get("message") or row) for row in qa_errors
            )
            raise RuntimeError(
                f"current-event refresh failed QA; card not priced: {detail}"
            )
        event = manifest.get("event") or {}
        field_rows = _read_csv(GREF.store.FIELD_CSV)
        field_event_ids = {
            str(row.get("event_id") or "").strip()
            for row in field_rows if str(row.get("event_id") or "").strip()
        }
        current_event_id = str(event.get("event_id") or "").strip()
        if (
            not field_rows
            or not current_event_id
            or field_event_ids != {current_event_id}
        ):
            raise RuntimeError(
                "current field/event snapshot is incomplete or incoherent; "
                "card not priced"
            )
        notes.append("refresh: " + (event.get("name") or "current event"))

    # Read the live state the refresh just recorded to report the round and, when
    # the caller didn't force --round, target this round's 3-ball board.
    live_state = _read_json(LIVE_STATE_JSON)
    rounds_done = int(live_state.get("rounds_done") or 0)
    if round_no is None:
        round_no = rounds_done + 1 if rounds_done else 1

    # If the finalised round moved past what the refresh parsed the manual board
    # for (a round completed during this run), parse the target round's paste now
    # so the round pricer finds threeballs_r{round}.csv.
    if refresh and refresh_round != round_no:
        parsed = GREF.ensure_round_board(
            round_no, event_name=event.get("name") or "",
            event_id=event.get("event_id") or event_id)
        if parsed:
            notes.append(f"round {round_no} board: {parsed} group(s) parsed from paste")

    event_name = (event.get("name") or live_state.get("event_name")
                  or _field_event())
    if major is None:
        major = any(h in (event_name or "").lower() for h in _MAJOR_HINTS)
    course = course or event.get("course_name", "")

    base = {
        "sims": sims,
        "course": course,
        "major": major,
        "seed": seed,
        "weather": weather,
    }

    # Field projection. cmd_simulate auto-routes to the in-play sim once a round
    # is complete; mirror that live-conditioned board into predictions.csv so the
    # card and downstream readers see it (cmd_simulate itself writes the pre-event
    # predictions.csv, or predictions_inplay.csv when in-play).
    sim_out = GENG.cmd_simulate(dict(base))
    if _is_inplay(sim_out):
        _mirror_inplay_predictions(sim_out)
        notes.append(
            f"in-play after R{rounds_done} ({TOTAL_ROUNDS - rounds_done} to play): "
            f"predictions conditioned on the live leaderboard "
            f"({len(sim_out.get('rows') or [])} survivors)")
    else:
        notes.append("pre-tournament projection (no completed rounds)")
    try:
        # Full board — outrights, places and tournament matchups/3-balls, all
        # auto-conditioned on the live leaderboard in-play. cmd_edge computes but
        # doesn't persist, so write the board here for the card/summary to read.
        edge_out = GENG.cmd_edge(dict(base, min_edge=0.0))
        GEDGE.write_edge_report(edge_out.get("rows") or [], path=EDGE_CSV)
    except ValueError as exc:
        notes.append(f"edge skipped: {exc}")
    try:
        GENG.cmd_round_3balls(dict(base, round_no=round_no))  # → round_edges.csv
    except ValueError as exc:
        # Pricing didn't run, so any existing round_edges.csv is from a previous
        # round/run. Clear it so the card can't report phantom round bets.
        try:
            ROUND_3BALL_CSV.unlink(missing_ok=True)
        except OSError:
            pass
        notes.append(f"round 3-balls skipped: {exc}")

    predictions = _read_csv(PREDICTIONS_CSV)
    edge_rows = _read_csv(EDGE_CSV)
    threeball_rows = _read_csv(ROUND_3BALL_CSV)
    manifest = _read_json(MANIFEST_JSON)

    text = _render_card(event_name or "PGA event", predictions, edge_rows,
                        threeball_rows, manifest, sims, major, course,
                        round_no, min_edge, notes)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, text)

    staked = [r for r in edge_rows if _truthy(r.get("recommended"))]
    picks_3b = _recommended_3balls(threeball_rows, min_edge)
    return {
        "event": event_name,
        "round": round_no,
        "output": str(output),
        "tournament_bets": len(staked),
        "round_3ball_bets": len(picks_3b),
        "notes": notes,
    }


def _render_card(event_name, predictions, edge_rows, threeball_rows, manifest,
                 sims, major, course, round_no, min_edge, notes) -> str:
    generated = time.strftime("%Y-%m-%d %H:%M")
    tag = " · major" if major else ""
    tag += f" · {course}" if course else ""
    market_label = _round_market_label(threeball_rows)
    L = [
        f"# {event_name} — Best Bets",
        "",
        f"_Generated {generated} · fitted model · {sims:,} sims{tag}_",
        "",
        _lead(event_name, predictions, edge_rows, threeball_rows, min_edge, market_label),
        "",
        "## Tournament bets",
        "",
        _tournament_section(edge_rows),
        "",
        f"## Round {round_no} {market_label}",
        "",
        _threeball_section(threeball_rows, min_edge, round_no),
        "",
        "## Field forecast",
        "",
        _forecast_table(predictions),
        "",
        "## Notes",
        "",
        _notes_section(manifest, notes),
        "",
    ]
    return "\n".join(L)


def _lead(event_name, predictions, edge_rows, threeball_rows, min_edge,
          market_label) -> str:
    """A short plain-English summary of the week before the detail sections."""
    picks = _recommended_3balls(threeball_rows, min_edge)
    staked = [r for r in edge_rows if _truthy(r.get("recommended"))]
    n = len(picks) + len(staked)
    total = (sum(_num(r.get("kelly_stake")) for r in picks)
             + sum(_num(r.get("stake_gbp")) for r in staked))
    out = [f"The model simulated the {event_name} field and weighed every price it "
           "could find against its own probabilities."]
    if predictions:
        fav = predictions[0]
        out.append(f" It makes **{fav.get('name', '')}** the favourite at "
                   f"{_num(fav.get('win_pct')):.0f}% to win.")
    if n:
        bets = "bet" if n == 1 else "bets"
        out.append(f" This week it backs **{n} {bets}** (total stake £{total:.2f}) — "
                   "each explained below, with the model's number, the price, and why "
                   "there's an edge. Stakes are fractional-Kelly on a £100 bankroll.")
    else:
        out.append(" This week the prices looked efficient — nothing cleared the edge "
                   "threshold, so there are no bets.")
    return "".join(out)


def _tournament_section(edge_rows: list[dict]) -> str:
    staked = [r for r in edge_rows if _truthy(r.get("recommended"))]
    if not staked:
        return ("Outright winner, placement (top-5/10/20) and make-cut prices were all "
                "checked against the model. None offered enough edge over the market to "
                "bet this week, so there's nothing staked on the tournament outcome.")
    staked.sort(key=lambda r: -_num(r.get("stake_gbp")))
    lines = ["The model backs these tournament-long bets:", ""]
    for r in staked:
        lines.append(
            f"- **{r.get('player', '')}** — {r.get('market', '')} at "
            f"{_num(r.get('odds')):.2f}. The model gives this {_pct(r.get('p_model'))} "
            f"against the {_pct(r.get('p_market'))} the market implies "
            f"(+{_num(r.get('ev_per_unit')) * 100:.0f}% edge). Stake "
            f"**£{_num(r.get('stake_gbp')):.2f}**.")
    return "\n".join(lines)


_MARKET_NAMES = {"2ball": "2-balls", "3ball": "3-balls"}


def _round_market_label(rows: list[dict]) -> str:
    """Card section title reflecting the actual round market (twosomes vs
    threesomes), falling back to a neutral label when nothing is priced."""
    markets = sorted({r.get("market") for r in rows if r.get("market")})
    if not markets:
        return "round matchups"
    return " / ".join(_MARKET_NAMES.get(m, m) for m in markets)


def _opponents(group_id: str, rows: list[dict], player: str) -> list[str]:
    """The other player(s) sharing a pairing/group, for naming in the prose."""
    if not group_id:
        return []
    names = []
    for r in rows:
        if r.get("group_id") == group_id and r.get("player") != player:
            nm = r.get("player")
            if nm and nm not in names:
                names.append(nm)
    return names


def _threeball_section(rows: list[dict], min_edge: float, round_no: int) -> str:
    picks = _recommended_3balls(rows, min_edge)
    if not rows:
        return ("_No round board loaded for this round. Bovada's board is pulled "
                "automatically on refresh; to override, paste one into "
                "`golf/data/threeballs_r{n}_raw.txt` and rerun with `--round {n}`._"
                .format(n=round_no))
    if not picks:
        return ("The round board priced cleanly, but no pairing was mispriced enough "
                "to clear the edge threshold — no round bets this week.")
    market = picks[0].get("market", "2ball")
    unit = "group" if market == "3ball" else "pairing"
    total = sum(_num(p.get("kelly_stake")) for p in picks)
    out = [
        f"A first-round {unit} bet backs one player to post the lower opening round "
        f"within their tee {unit} (a tie splits the stake). The model simulates each "
        f"{unit} and bets only when its win probability beats the price. "
        f"**{len(picks)} cleared the threshold** this round (total stake £{total:.2f}), "
        "strongest edge first:",
        "",
    ]
    top = picks[:6]
    for p in top:
        opp = _opponents(p.get("group_id", ""), rows, p.get("player", ""))
        vs = " / ".join(opp) if opp else "the field"
        out.append(
            f"- **{p.get('player', '')}** over {vs} — {_num(p.get('odds')):.2f}. The "
            f"model has him {_pct(p.get('p_dead_heat_equiv'))} to take the {unit}, "
            f"against {_pct(p.get('p_market'))} implied by the price — a "
            f"+{_num(p.get('ev_pct')):.0f}% edge. Stake **£{_num(p.get('kelly_stake')):.2f}**.")
    rest = picks[6:]
    if rest:
        out += ["", f"Also backed, at smaller edges ({len(rest)}):", "",
                "| Player | Odds | Model | Edge | Stake |", "|---|--:|--:|--:|--:|"]
        out += [
            f"| {p.get('player', '')} | {_num(p.get('odds')):.2f} "
            f"| {_pct(p.get('p_dead_heat_equiv'))} | +{_num(p.get('ev_pct')):.0f}% "
            f"| £{_num(p.get('kelly_stake')):.2f} |"
            for p in rest
        ]
    return "\n".join(out)


def _forecast_table(predictions: list[dict], top: int = 10) -> str:
    if not predictions:
        return "_No field forecast — run a refresh or seed rounds.csv._"
    intro = ("Not bets — just the model's own read on the field, for context: each "
             "player's chance to win and to finish top-10.")
    head = "| Player | Win | Top 10 |\n|---|--:|--:|"
    rows = [
        f"| {r.get('name', '')} | {_num(r.get('win_pct')):.1f}% "
        f"| {_num(r.get('top10_pct')):.0f}% |"
        for r in predictions[:top]
    ]
    return intro + "\n\n" + head + "\n" + "\n".join(rows)


def _notes_section(manifest: dict, notes: list[str]) -> str:
    qa = manifest.get("qa") or {}
    warnings = qa.get("warnings") or []
    errors = qa.get("errors") or []
    lines = [f"- {n}" for n in notes]
    if errors:
        lines.append(f"- ⚠ {len(errors)} data error(s) — see free_source_manifest.json")
    if warnings:
        # High-signal warnings carry an actionable message (a fix command, or a
        # data-integrity red flag like a merged/implausible field), so show them
        # verbatim. Others stay summarised.
        def _high_signal(w: dict) -> bool:
            src = str(w.get("source", ""))
            return src.startswith("freshness.") or src == "espn.field"
        shown = [w for w in warnings if _high_signal(w)]
        other = [w for w in warnings if w not in shown]
        for w in shown:
            lines.append(f"- ⚠ {w.get('message')}")
        if other:
            lines.append(f"- {len(other)} other data warning(s) — see free_source_manifest.json")
    if not lines:
        lines.append("- Clean run, no data warnings.")
    return "\n".join(lines)


def _recommended_3balls(rows: list[dict], min_edge: float) -> list[dict]:
    """Same recommendation rule round_pricer uses: above edge, real stake,
    enough sample behind the player."""
    out = [r for r in rows
           if _num(r.get("ev_pct")) >= min_edge
           and _num(r.get("kelly_stake")) >= 0.5
           and not _truthy(r.get("thin_sample"))]
    out.sort(key=lambda r: -_num(r.get("ev_pct")))
    return out


# ── small helpers ──────────────────────────────────────────────────────────

def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict:
    import json
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _field_event() -> str:
    """Event name from the current field.csv, when no refresh was run."""
    from .store import FIELD_CSV
    rows = _read_csv(FIELD_CSV)
    return rows[0].get("event") or rows[0].get("event_name", "") if rows else ""


def _num(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _pct(value) -> str:
    return f"{_num(value) * 100:.1f}%"


def _truthy(value) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "y")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Golf engine — season schedule and round-by-round best-bets card")
    ap.add_argument("--schedule", action="store_true",
                    help="print the season's tournament list and exit")
    ap.add_argument("--season", type=int, default=None, help="season/year")
    ap.add_argument("--event", default="",
                    help="ESPN event id or name to force (default: auto-detect current)")
    ap.add_argument("--round", type=int, default=None, dest="round_no",
                    help="force the round to predict (default: auto-detect from the "
                         "live leaderboard)")
    ap.add_argument("--sims", type=int, default=50_000)
    ap.add_argument("--no-refresh", action="store_true",
                    help="reprice from cached data; skip the provider refresh")
    ap.add_argument("--stats", action="store_true", help="also pull PGA stat pages")
    ap.add_argument("--weather", action="store_true", help="also pull course weather")
    ap.add_argument("--fit", action="store_true", help="refit the model after refresh")
    ap.add_argument("--major", action="store_true", help="force major treatment")
    ap.add_argument("--course", default="", help="course name for course-fit")
    ap.add_argument("--min-edge", type=float, default=1.0,
                    help="min 3-ball EV%% to recommend (default 1.0)")
    args = ap.parse_args()

    if args.schedule:
        print_schedule(args.season)
        return

    summary = build_card(
        season=args.season, event_id=args.event, round_no=args.round_no,
        sims=args.sims, refresh=not args.no_refresh, stats=args.stats,
        weather=args.weather, fit=args.fit,
        major=True if args.major else None, course=args.course,
        min_edge=args.min_edge,
    )
    print(f"{summary['event']} — {summary['tournament_bets']} tournament bet(s), "
          f"{summary['round_3ball_bets']} round-{summary['round']} 3-ball bet(s)")
    for n in summary["notes"]:
        print(f"  · {n}")
    print(f"Card → {summary['output']}")


if __name__ == "__main__":
    main()

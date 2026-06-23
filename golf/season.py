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
import time
from pathlib import Path

from . import engine as GENG
from . import refresh as GREF
from .providers.espn import EspnGolfProvider

DATA_DIR = Path(__file__).parent / "data"
PREDICTIONS_CSV = DATA_DIR / "predictions.csv"
EDGE_CSV = DATA_DIR / "edge_report.csv"
ROUND_3BALL_CSV = DATA_DIR / "round_3ball_edges.csv"
MANIFEST_JSON = DATA_DIR / "free_source_manifest.json"
CARD_MD = DATA_DIR / "card.md"

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


# ── card ────────────────────────────────────────────────────────────────────

def build_card(
    *,
    season: int | None = None,
    event_id: str = "",
    round_no: int = 1,
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

    if refresh:
        manifest = GREF.run_refresh(season=season, event=event_id, stats=stats,
                                    weather=weather, fit=fit, round_no=round_no)
        event = manifest.get("event") or {}
        notes.append("refresh: " + (event.get("name") or "current event"))

    event_name = event.get("name") or _field_event()
    if major is None:
        major = any(h in (event_name or "").lower() for h in _MAJOR_HINTS)
    course = course or event.get("course_name", "")

    base = {"sims": sims, "course": course, "major": major, "seed": seed}

    GENG.cmd_simulate(dict(base))           # → predictions.csv
    try:
        GENG.cmd_edge(dict(base, min_edge=0.0))   # → edge_report.csv (full board)
    except ValueError as exc:
        notes.append(f"edge skipped: {exc}")
    try:
        GENG.cmd_round_3balls(dict(base, round_no=round_no))  # → round_3ball_edges.csv
    except ValueError as exc:
        notes.append(f"round 3-balls skipped: {exc}")

    predictions = _read_csv(PREDICTIONS_CSV)
    edge_rows = _read_csv(EDGE_CSV)
    threeball_rows = _read_csv(ROUND_3BALL_CSV)
    manifest = _read_json(MANIFEST_JSON)

    text = _render_card(event_name or "PGA event", predictions, edge_rows,
                        threeball_rows, manifest, sims, major, course,
                        round_no, min_edge, notes)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)

    staked = [r for r in edge_rows if _truthy(r.get("recommended"))]
    picks_3b = _recommended_3balls(threeball_rows, min_edge)
    return {
        "event": event_name,
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
    L = [
        f"# {event_name} — Best Bets",
        "",
        f"_Generated {generated} · fitted model · {sims:,} sims{tag}_",
        "",
        "## Tournament card (pre-tournament)",
        "",
        "Outrights, placement (top-5/10/20), make-cut and matchups — staked at "
        "calibrated, market-blended, portfolio-sized stakes. Only bets the model "
        "backs are listed.",
        "",
        _tournament_section(edge_rows),
        "",
        f"## Round {round_no} 3-balls",
        "",
        _threeball_section(threeball_rows, min_edge, round_no),
        "",
        "## Field forecast (model context)",
        "",
        _forecast_table(predictions),
        "",
        "## Notes",
        "",
        _notes_section(manifest, notes),
        "",
    ]
    return "\n".join(L)


def _tournament_section(edge_rows: list[dict]) -> str:
    staked = [r for r in edge_rows if _truthy(r.get("recommended"))]
    if not staked:
        return "_No tournament bet cleared the edge/stake threshold this week._"
    staked.sort(key=lambda r: -_num(r.get("stake_gbp")))
    head = ("| Selection | Market | Odds | Model | Market | Edge | Stake |\n"
            "|---|---|--:|--:|--:|--:|--:|")
    rows = [
        f"| {r.get('player','')} | {r.get('market','')} | {_num(r.get('odds')):.2f} "
        f"| {_pct(r.get('p_model'))} | {_pct(r.get('p_market'))} "
        f"| {_num(r.get('ev_per_unit'))*100:+.1f}% | £{_num(r.get('stake_gbp')):.2f} |"
        for r in staked
    ]
    return head + "\n" + "\n".join(rows)


def _threeball_section(rows: list[dict], min_edge: float, round_no: int) -> str:
    picks = _recommended_3balls(rows, min_edge)
    if not rows:
        return ("_No 3-ball board loaded for this round. Paste a bookmaker board "
                "into `golf/data/threeballs_r{n}_raw.txt` and rerun with "
                "`--round {n}`._".format(n=round_no))
    if not picks:
        return "_3-balls priced, but none cleared the edge/stake threshold._"
    head = ("| Round | Player | Odds | Model | Market | EV | Stake |\n"
            "|--:|---|--:|--:|--:|--:|--:|")
    body = [
        f"| {r.get('round','')} | {r.get('player','')} | {_num(r.get('odds')):.2f} "
        f"| {_pct(r.get('p_dead_heat_equiv'))} | {_pct(r.get('p_market'))} "
        f"| {_num(r.get('ev_pct')):+.1f}% | £{_num(r.get('kelly_stake')):.2f} |"
        for r in picks
    ]
    return head + "\n" + "\n".join(body)


def _forecast_table(predictions: list[dict], top: int = 10) -> str:
    if not predictions:
        return "_No field forecast — run a refresh or seed rounds.csv._"
    head = ("| # | Player | Win | Top 5 | Top 10 | Make cut | Avg fin |\n"
            "|--:|---|--:|--:|--:|--:|--:|")
    rows = []
    for i, r in enumerate(predictions[:top], 1):
        rows.append(
            f"| {i} | {r.get('name','')} | {_num(r.get('win_pct')):.1f}% "
            f"| {_num(r.get('top5_pct')):.0f}% | {_num(r.get('top10_pct')):.0f}% "
            f"| {_num(r.get('cut_pct')):.0f}% | {_num(r.get('avg_finish')):.1f} |")
    return head + "\n" + "\n".join(rows)


def _notes_section(manifest: dict, notes: list[str]) -> str:
    qa = manifest.get("qa") or {}
    warnings = qa.get("warnings") or []
    errors = qa.get("errors") or []
    lines = [f"- {n}" for n in notes]
    if errors:
        lines.append(f"- ⚠ {len(errors)} data error(s) — see free_source_manifest.json")
    if warnings:
        lines.append(f"- {len(warnings)} data warning(s)")
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
    ap.add_argument("--event", default="", help="ESPN event id (default: current)")
    ap.add_argument("--round", type=int, default=1, dest="round_no",
                    help="round to price 3-balls for (default 1)")
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
          f"{summary['round_3ball_bets']} round-{args.round_no} 3-ball bet(s)")
    for n in summary["notes"]:
        print(f"  · {n}")
    print(f"Card → {summary['output']}")


if __name__ == "__main__":
    main()

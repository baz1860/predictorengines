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

from . import engine as GENG
from . import model
from . import refresh as GREF
from . import simulate_inplay as GIP
from .providers.espn import EspnGolfProvider

DATA_DIR = Path(__file__).parent / "data"
PREDICTIONS_CSV = DATA_DIR / "predictions.csv"
PREDICTIONS_INPLAY_CSV = DATA_DIR / "predictions_inplay.csv"
EDGE_CSV = DATA_DIR / "edge_report.csv"
ROUND_3BALL_CSV = DATA_DIR / "round_edges.csv"
MANIFEST_JSON = DATA_DIR / "free_source_manifest.json"
CARD_MD = DATA_DIR / "card.md"
LIVE_STATE_JSON = DATA_DIR / "live_state.json"
SCORES_LIVE_CSV = DATA_DIR / "scores_live.csv"

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
# Once a round is complete, the field projection should condition on the live
# leaderboard rather than re-run the pre-tournament simulation. We pull ESPN's
# per-round line scores, reconstruct each survivor's cumulative score through the
# last fully-completed round, and hand that to the in-play simulator.

def _topar_from_display(value) -> int | None:
    """ESPN per-round to-par string ('-5', '+2', 'E', '-') → int, or None."""
    s = str(value).strip()
    if s in ("", "-", "—"):
        return None
    if s.upper() == "E":
        return 0
    try:
        return int(s.replace("+", ""))
    except ValueError:
        return None


def _live_state(event_id: str, target_round: int) -> dict | None:
    """Reconstruct cumulative to-par through the last fully-completed round.

    Returns a state dict (event_name, event_id, status, rounds_done,
    current_scores name_lower→to-par, board) or None when there is no live event
    or no completed round to condition on.
    """
    from .providers.espn import _status_name, _safe_int  # local: internal helpers

    prov = EspnGolfProvider()
    payload = prov.current_event_payload(event_id or None, use_cache=False)
    evs = payload.get("events") or []
    if not evs:
        return None
    ev = evs[0]
    ev_name = ev.get("name") or ""
    eid = str(ev.get("id") or event_id or "")
    status = ((ev.get("status") or {}).get("type") or {}).get("name") or ""
    comp = (ev.get("competitions") or [{}])[0]

    per_player = []
    for c in comp.get("competitors") or []:
        ath = c.get("athlete") or {}
        name = (ath.get("displayName") or ath.get("fullName") or "").strip()
        if not name:
            continue
        rounds: dict[int, int] = {}
        for ls in c.get("linescores") or []:
            period = _safe_int(ls.get("period"))
            has_holes = bool(ls.get("linescores"))     # round actually played out
            tp = _topar_from_display(ls.get("displayValue"))
            if period and has_holes and tp is not None:
                rounds[period] = tp
        per_player.append({
            "name": name, "rounds": rounds, "status": _status_name(c),
            "position": c.get("order"), "cum_score": c.get("score"),
        })

    completed = [max(p["rounds"]) for p in per_player if p["rounds"]]
    if not completed:
        return None
    field_completed = max(completed)
    # Only ever condition on rounds that are actually finished.
    rounds_done = min(target_round - 1, field_completed)
    if rounds_done < 1:
        return None

    need = range(1, rounds_done + 1)
    current_scores: dict[str, float] = {}
    board = []
    for p in per_player:
        rs = p["rounds"]
        if not all(r in rs for r in need):
            continue   # missed cut / withdrew → not in the remaining field
        cum = sum(rs[r] for r in need)
        current_scores[p["name"].lower()] = float(cum)
        board.append({"name": p["name"], "score_thru": cum,
                      "rounds_completed": max(rs), "position": p["position"]})
    if not current_scores:
        return None
    return {"event_name": ev_name, "event_id": eid, "status": status,
            "rounds_done": rounds_done, "current_scores": current_scores,
            "board": board}


def _write_inplay_predictions_csv(survivors, results, path: Path = PREDICTIONS_CSV) -> Path:
    """Write predictions.csv from in-play results, schema-compatible with the
    pre-tournament file plus a `score_thru` column. Ranked by win probability."""
    cols = ["rank", "name", "rating", "sigma", "owgr", "win_pct", "top5_pct",
            "top10_pct", "top20_pct", "cut_pct", "avg_finish", "score_thru"]
    ranked = sorted(survivors, key=lambda p: results[p.name]["win"], reverse=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for rank, p in enumerate(ranked, 1):
            r = results[p.name]
            sc = int(round(r["current_score"]))
            thru = "E" if sc == 0 else f"{sc:+d}"
            w.writerow({
                "rank": rank, "name": p.name,
                "rating": f"{p.rating:+.3f}", "sigma": f"{p.sigma:.2f}",
                "owgr": getattr(p, "owgr", 999),
                "win_pct": f"{r['win'] * 100:.2f}",
                "top5_pct": f"{r['top5'] * 100:.1f}",
                "top10_pct": f"{r['top10'] * 100:.1f}",
                "top20_pct": f"{r['top20'] * 100:.1f}",
                "cut_pct": "100.0",
                "avg_finish": f"{r['avg_finish']:.1f}",
                "score_thru": thru,
            })
    return path


def _write_live_state(state: dict, n_survivors: int, rounds_done: int) -> None:
    updated = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    LIVE_STATE_JSON.write_text(json.dumps({
        "event": state["event_name"],
        "event_id": state["event_id"],
        "status": state["status"],
        "rounds_done": rounds_done,
        "round_today": rounds_done + 1,
        "survivors": n_survivors,
        "updated": updated,
    }, indent=2))
    board = sorted(state["board"], key=lambda b: b["score_thru"])
    with open(SCORES_LIVE_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "score_thru", "rounds_completed",
                                          "position", "made_cut"])
        w.writeheader()
        for b in board:
            sc = int(round(b["score_thru"]))
            w.writerow({"name": b["name"],
                        "score_thru": "E" if sc == 0 else f"{sc:+d}",
                        "rounds_completed": b["rounds_completed"],
                        "position": b["position"], "made_cut": 1})


def _route_inplay(*, event_id, round_no, base, sims, seed, notes) -> dict | None:
    """If a round is complete, condition on the live leaderboard and write the
    in-play field projection. Returns the live state, or None to fall back to the
    pre-tournament simulation."""
    if round_no < 2:
        return None
    try:
        state = _live_state(event_id, round_no)
    except Exception as exc:  # noqa: BLE001 — never let a flaky feed break the run
        notes.append(f"in-play skipped (leaderboard error): {exc}")
        return None
    if not state:
        return None

    rounds_done = state["rounds_done"]
    current_scores = state["current_scores"]
    rated, fitted = GENG._rated_field(base.get("course", ""), bool(base.get("major")))
    survivors = [p for p in rated if p.name.lower() in current_scores]
    if not survivors:
        notes.append("in-play skipped: no overlap between field and leaderboard")
        return None

    import numpy as np
    rng = np.random.default_rng(int(seed) or None)
    results = GIP.simulate_inplay(survivors, current_scores, rounds_done,
                                  n_sims=sims, rng=rng)
    GIP.write_predictions_inplay(survivors, results, rounds_done,
                                 path=PREDICTIONS_INPLAY_CSV)
    _write_inplay_predictions_csv(survivors, results)
    _write_live_state(state, len(survivors), rounds_done)
    if rounds_done < round_no - 1:
        notes.append(f"requested round {round_no} but only R1–{rounds_done} are "
                     f"complete — conditioned on completed rounds only")
    notes.append(f"in-play: conditioned on R1–{rounds_done} "
                 f"({len(survivors)} survivors), simulating round {rounds_done + 1}")
    state["survivors"] = len(survivors)
    state["fitted"] = fitted
    return state


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

    # Once a round is complete, condition the field projection on the live
    # leaderboard (in-play). Otherwise fall back to the pre-tournament simulation.
    live = _route_inplay(event_id=event_id, round_no=round_no, base=base,
                         sims=sims, seed=seed, notes=notes)
    if live:
        event_name = live.get("event_name") or event_name
    else:
        GENG.cmd_simulate(dict(base))       # → predictions.csv (pre-tournament)
    try:
        GENG.cmd_edge(dict(base, min_edge=0.0))   # → edge_report.csv (full board)
    except ValueError as exc:
        notes.append(f"edge skipped: {exc}")
    try:
        GENG.cmd_round_3balls(dict(base, round_no=round_no))  # → round_edges.csv
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

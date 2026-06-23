"""Narrative weekly report for the free-source golf model.

The report is deterministic: it explains the model from the local data it has
actually produced rather than inventing subjective commentary. It reads the
latest field, fitted parameters, predictions, market edges, 3-ball prices, and
provider manifest, then writes a Markdown report suitable for weekly review.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

from . import engine as GENG
from . import model as GMOD
from . import refresh as GREF

DATA_DIR = Path(__file__).parent / "data"
REPORTS_DIR = Path(__file__).parent / "reports"
PREDICTIONS_CSV = DATA_DIR / "predictions.csv"
EDGE_CSV = DATA_DIR / "edge_report.csv"
ROUND_3BALL_CSV = DATA_DIR / "round_edges.csv"
MANIFEST_JSON = DATA_DIR / "free_source_manifest.json"
PARAMS_JSON = DATA_DIR / "model_params.json"
OUT_MD = DATA_DIR / "weekly_report.md"


def generate_report(
    *,
    top: int = 20,
    edge_top: int = 12,
    threeball_top: int = 12,
    output: Path = OUT_MD,
    archive: bool = False,
    run_refresh: bool = False,
    stats: bool = False,
    weather: bool = False,
    fit: bool = False,
    use_cache: bool = False,
    simulate: bool = False,
    edges: bool = False,
    round_3balls: bool = False,
    sims: int = 100_000,
    round_no: int = 1,
    course: str = "",
    major: bool = False,
    seed: int = 7,
) -> dict:
    run_notes = _run_optional_steps(
        run_refresh=run_refresh,
        stats=stats,
        weather=weather,
        fit=fit,
        use_cache=use_cache,
        simulate=simulate,
        edges=edges,
        round_3balls=round_3balls,
        sims=sims,
        round_no=round_no,
        course=course,
        major=major,
        seed=seed,
    )

    manifest = _read_json(MANIFEST_JSON)
    params = _read_json(PARAMS_JSON)
    predictions = _read_csv(PREDICTIONS_CSV)
    edge_rows = _read_csv(EDGE_CSV)
    threeball_rows = _read_csv(ROUND_3BALL_CSV)

    report = _render_report(
        manifest=manifest,
        params=params,
        predictions=predictions,
        edge_rows=edge_rows,
        threeball_rows=threeball_rows,
        run_notes=run_notes,
        top=top,
        edge_top=edge_top,
        threeball_top=threeball_top,
        course=course,
        major=major,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report)
    archive_path = None
    if archive:
        archive_path = _archive_report(output, manifest)
    return {
        "output": str(output),
        "archive": str(archive_path) if archive_path else "",
        "predictions": len(predictions),
        "edges": len(edge_rows),
        "threeballs": len(threeball_rows),
        "run_notes": run_notes,
    }


def _run_optional_steps(**kwargs) -> list[str]:
    notes = []
    if kwargs["run_refresh"]:
        manifest = GREF.run_refresh(
            stats=kwargs["stats"],
            weather=kwargs["weather"],
            fit=kwargs["fit"],
            use_cache=kwargs["use_cache"],
            round_no=kwargs["round_no"],
        )
        event = manifest.get("event") or {}
        notes.append(
            "refresh: "
            + (event.get("name") or "current event")
            + f" ({(manifest.get('provider_rows') or {}).get('espn_field', 0)} field rows)"
        )
    if kwargs["simulate"]:
        res = GENG.cmd_simulate({
            "sims": kwargs["sims"],
            "course": kwargs["course"],
            "major": kwargs["major"],
            "seed": kwargs["seed"],
        })
        notes.append("simulate: " + str(res.get("note", "")).strip())
    if kwargs["edges"]:
        try:
            res = GENG.cmd_edge({
                "sims": kwargs["sims"],
                "course": kwargs["course"],
                "major": kwargs["major"],
                "seed": kwargs["seed"],
            })
            notes.append("edge: " + str(res.get("note", "")).strip())
        except ValueError as exc:
            notes.append(f"edge skipped: {exc}")
    if kwargs["round_3balls"]:
        try:
            res = GENG.cmd_round_3balls({
                "round_no": kwargs["round_no"],
                "sims": kwargs["sims"],
                "course": kwargs["course"],
                "major": kwargs["major"],
            })
            notes.append("round 3-balls: " + str(res.get("note", "")).strip())
        except ValueError as exc:
            notes.append(f"round 3-balls skipped: {exc}")
    return [n for n in notes if n]


def _render_report(
    *,
    manifest: dict,
    params: dict,
    predictions: list[dict],
    edge_rows: list[dict],
    threeball_rows: list[dict],
    run_notes: list[str],
    top: int,
    edge_top: int,
    threeball_top: int,
    course: str,
    major: bool,
) -> str:
    event = manifest.get("event") or {}
    event_name = event.get("name") or _field_event() or "PGA event"
    generated = time.strftime("%Y-%m-%d %H:%M")
    provider_rows = manifest.get("provider_rows") or {}
    qa = manifest.get("qa") or {}
    players = params.get("players") or {}

    lines = [
        f"# {event_name} Weekly Model Report",
        "",
        f"Generated: {generated}",
        "",
        "## Executive View",
        "",
        _executive_summary(predictions, edge_rows, threeball_rows, params),
        "",
        "## Data Status",
        "",
        _data_status(event, provider_rows, qa, params, course, major, run_notes),
        "",
        "## Winner And Placement Forecast",
        "",
        _prediction_table(predictions[:top], params),
        "",
        "## Contender Reasoning",
        "",
        _contender_notes(predictions[: min(8, top)], players, params),
        "",
        "## Market Edge Read",
        "",
        _edge_section(edge_rows, edge_top),
        "",
        "## Round 3-Ball Read",
        "",
        _threeball_section(threeball_rows, threeball_top),
        "",
        "## Caveats",
        "",
        _caveats(manifest, params, predictions, edge_rows, threeball_rows),
        "",
    ]
    return "\n".join(lines)


def _executive_summary(predictions: list[dict], edge_rows: list[dict],
                       threeball_rows: list[dict], params: dict) -> str:
    if not predictions:
        return "No prediction file is available yet. Run `python3 -m golf.simulate` first."
    top = predictions[0]
    second = predictions[1] if len(predictions) > 1 else None
    gap = _num(top.get("win_pct")) - _num(second.get("win_pct")) if second else 0.0
    rec_edges = [r for r in edge_rows if _truthy(r.get("recommended")) and _num(r.get("stake_gbp")) > 0]
    three_picks = [r for r in threeball_rows if _num(r.get("ev_pct")) >= 4 and not _truthy(r.get("thin_sample"))]
    parts = [
        f"The model's top winner is **{top['name']}** at {_pct_text(top.get('win_pct'))} win probability",
    ]
    if second:
        parts.append(f"{gap:.1f} percentage points clear of **{second['name']}**")
    parts.append(f"with a {_pct_text(top.get('top10_pct'))} top-10 profile")
    summary = " ".join(parts) + ". "
    summary += _driver_sentence(top["name"], params, _num(top.get("rating")))
    if rec_edges:
        summary += f" The market screen currently has {len(rec_edges)} staked recommendation(s)."
    elif edge_rows:
        summary += " The market screen has priced lines, but portfolio staking is currently selective or flat."
    else:
        summary += " No current outright/place/matchup edge report is available."
    if threeball_rows:
        summary += f" The round 3-ball file contains {len(threeball_rows)} priced side(s), with {len(three_picks)} above the 4% EV reference line."
    return summary


def _data_status(event: dict, provider_rows: dict, qa: dict, params: dict,
                 course: str, major: bool, run_notes: list[str]) -> str:
    rows = [
        f"- Event: **{event.get('name', 'unknown')}** ({event.get('event_id', 'no event id')})",
        f"- Field rows: {provider_rows.get('espn_field', 0)} from ESPN scoreboard",
        f"- Historical round rows: {provider_rows.get('rounds_csv', 0)}",
        f"- Manual odds rows: {provider_rows.get('manual_odds', 0)}",
        f"- Model fit: {params.get('fitted_rounds', 0):,} rounds through {params.get('asof', 'unknown')}",
        f"- Field sigma: {_num(params.get('sigma_field')):.2f}; major mode: {'on' if major else 'off'}",
    ]
    if course:
        rows.append(f"- Course argument: {course}")
    warnings = (qa.get("warnings") or []) + (qa.get("errors") or [])
    if warnings:
        rows.append("- Provider warnings: " + "; ".join(f"{w.get('source')}: {w.get('message')}" for w in warnings))
    else:
        rows.append("- Provider QA: no current warnings")
    if run_notes:
        rows.append("- Steps run before report: " + "; ".join(run_notes))
    return "\n".join(rows)


def _prediction_table(rows: list[dict], params: dict) -> str:
    if not rows:
        return "_No predictions available._"
    out = [
        "| # | Player | Win | Top 5 | Top 10 | Top 20 | Cut | Rating | Reason |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        out.append(
            "| {rank} | {name} | {win} | {top5} | {top10} | {top20} | {cut} | {rating} | {reason} |".format(
                rank=r.get("rank", ""),
                name=_md(r.get("name", "")),
                win=_pct_text(r.get("win_pct")),
                top5=_pct_text(r.get("top5_pct")),
                top10=_pct_text(r.get("top10_pct")),
                top20=_pct_text(r.get("top20_pct")),
                cut=_pct_text(r.get("cut_pct")),
                rating=r.get("rating", ""),
                reason=_md(_short_reason(r, params)),
            )
        )
    return "\n".join(out)


def _contender_notes(rows: list[dict], players: dict, params: dict) -> str:
    if not rows:
        return "_No contender notes available._"
    field_sigma = _num(params.get("sigma_field"), 2.85)
    notes = []
    for r in rows:
        name = r.get("name", "")
        p = _player_entry(name, params)
        sigma = _num(r.get("sigma"), _num(p.get("sigma"), field_sigma))
        n = int(_num(p.get("n_rounds"), 0))
        skill = _num(p.get("skill"))
        form = _num(p.get("form"))
        risk = "higher week-to-week volatility" if sigma > field_sigma * 1.08 else (
            "steadier scoring profile" if sigma < field_sigma * 0.94 else "field-average volatility"
        )
        sample = "large sample" if n >= 120 else ("moderate sample" if n >= 50 else "thin sample")
        notes.append(
            f"- **{name}**: {_pct_text(r.get('win_pct'))} to win, {_pct_text(r.get('top10_pct'))} top 10. "
            f"Underlying skill {skill:+.2f}, recent form {form:+.2f}, {sample} ({n} rounds), {risk}."
        )
    return "\n".join(notes)


def _edge_section(rows: list[dict], limit: int) -> str:
    if not rows:
        return "_No edge report is available. Run `python3 -m golf.edge --sims 100000 --major` after adding odds._"
    rows = sorted(rows, key=lambda r: _num(r.get("ev_per_unit")), reverse=True)
    recs = [r for r in rows if _truthy(r.get("recommended")) and _num(r.get("stake_gbp")) > 0]
    shown = recs[:limit] or rows[:limit]
    intro = f"{len(rows)} market side(s) priced. "
    intro += f"{len(recs)} have a positive portfolio stake." if recs else "No side currently receives a portfolio stake."
    table = [
        intro,
        "",
        "| Player | Market | Odds | Model | Market | EV | Stake | Read |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in shown:
        model = _prob_text(r.get("p_model"))
        market = _prob_text(r.get("p_market"))
        ev = _num(r.get("ev_per_unit")) * 100
        read = f"model is {(_num(r.get('p_model')) - _num(r.get('p_market'))) * 100:+.1f} pp above market"
        table.append(
            f"| {_md(r.get('player', ''))} | {_md(r.get('market', ''))} | {_num(r.get('odds')):.2f} | "
            f"{model} | {market} | {ev:+.1f}% | £{_num(r.get('stake_gbp')):.2f} | {_md(read)} |"
        )
    return "\n".join(table)


def _threeball_section(rows: list[dict], limit: int) -> str:
    if not rows:
        return "_No round 3-ball pricing file is available. Run `python3 -m golf.round_pricer --round 1 --major --min-edge 4` after pasting a 3-ball board._"
    rows = sorted(rows, key=lambda r: _num(r.get("ev_pct")), reverse=True)
    picks = [r for r in rows if _num(r.get("ev_pct")) >= 4 and not _truthy(r.get("thin_sample"))]
    shown = picks[:limit] or rows[:limit]
    table = [
        f"{len(rows)} 3-ball side(s) priced. {len(picks)} are above the 4% EV reference line without a thin-sample flag.",
        "",
        "| Round | Group | Player | Odds | Model | Market | EV | Stake | Read |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in shown:
        diff = (_num(r.get("p_dead_heat_equiv")) - _num(r.get("p_market"))) * 100
        read = f"dead-heat adjusted model {diff:+.1f} pp vs market"
        if _truthy(r.get("thin_sample")):
            read += "; thin sample"
        table.append(
            f"| {r.get('round', '')} | {_md(_short_group(r.get('group_id', '')))} | {_md(r.get('player', ''))} | "
            f"{_num(r.get('odds')):.2f} | {_prob_text(r.get('p_dead_heat_equiv'))} | "
            f"{_prob_text(r.get('p_market'))} | {_num(r.get('ev_pct')):+.1f}% | "
            f"£{_num(r.get('kelly_stake')):.2f} | {_md(read)} |"
        )
    return "\n".join(table)


def _caveats(manifest: dict, params: dict, predictions: list[dict],
             edge_rows: list[dict], threeball_rows: list[dict]) -> str:
    caveats = [
        "- This is a model report, not a guarantee; outright golf probabilities remain low even for the strongest players.",
        "- Free-source course fit is not ShotLink-quality unless stable shot-level data is later added.",
    ]
    if not (manifest.get("stats_csv") or params.get("public_stat_priors")):
        caveats.append("- Current report does not show a fresh PGA public-stat prior file; run with `--refresh --stats --fit` for that layer.")
    if predictions and _num(predictions[0].get("cut_pct")) > 99:
        caveats.append("- Make-cut probabilities look degenerate; check whether the event is no-cut or the cut rule exceeds field size.")
    if edge_rows:
        caveats.append("- Edge tables depend on manually pasted odds being current and correctly parsed.")
    if threeball_rows:
        caveats.append("- 3-ball probabilities use a discrete single-round score simulation and dead-heat adjustment.")
    return "\n".join(caveats)


def _short_reason(row: dict, params: dict) -> str:
    name = row.get("name", "")
    p = _player_entry(name, params)
    skill = _num(p.get("skill"))
    form = _num(p.get("form"))
    n = int(_num(p.get("n_rounds"), 0))
    bits = []
    if skill >= 1.5:
        bits.append("elite baseline")
    elif skill >= 0.75:
        bits.append("strong baseline")
    elif skill >= 0.25:
        bits.append("above-average baseline")
    else:
        bits.append("field-average baseline")
    if form >= 0.35:
        bits.append("positive recent form")
    elif form <= -0.35:
        bits.append("soft recent form")
    if n and n < 50:
        bits.append("sample risk")
    return ", ".join(bits)


def _driver_sentence(name: str, params: dict, rating: float) -> str:
    p = _player_entry(name, params)
    skill = _num(p.get("skill"))
    form = _num(p.get("form"))
    n = int(_num(p.get("n_rounds"), 0))
    if not p:
        return f"The rating is {rating:+.2f} strokes per round versus this field, with limited stored component detail."
    return (
        f"The main driver is a {skill:+.2f} long-run skill estimate with {form:+.2f} recent-form input "
        f"over {n} fitted rounds."
    )


def _player_entry(name: str, params: dict) -> dict:
    players = params.get("players") or {}
    if name in players:
        return players[name]
    resolved = GMOD.resolve_name(name, params)
    if resolved and resolved in players:
        return players[resolved]
    return {}


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _field_event() -> str:
    rows = _read_csv(DATA_DIR / "field.csv")
    for r in rows:
        if r.get("event"):
            return r["event"]
    return ""


def _archive_report(output: Path, manifest: dict) -> Path:
    event = ((manifest.get("event") or {}).get("name") or "golf").lower()
    slug = "".join(ch if ch.isalnum() else "-" for ch in event).strip("-")
    slug = "-".join(x for x in slug.split("-") if x)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = REPORTS_DIR / f"{time.strftime('%Y-%m-%d')}_{slug or 'golf'}_weekly_report.md"
    shutil.copyfile(output, dest)
    return dest


def _short_group(group_id: str) -> str:
    s = str(group_id or "")
    return s.split(":", 1)[-1] if ":" in s else s


def _num(value, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(str(value).replace("%", ""))
    except (TypeError, ValueError):
        return default


def _pct_text(value) -> str:
    return f"{_num(value):.1f}%"


def _prob_text(value) -> str:
    x = _num(value)
    if x <= 1:
        x *= 100
    return f"{x:.1f}%"


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _md(value) -> str:
    return str(value or "").replace("|", "\\|")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a narrative weekly golf report")
    ap.add_argument("--output", type=Path, default=OUT_MD)
    ap.add_argument("--archive", action="store_true", help="also copy to golf/reports/YYYY-MM-DD_event_weekly_report.md")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--edge-top", type=int, default=12)
    ap.add_argument("--threeball-top", type=int, default=12)
    ap.add_argument("--refresh", action="store_true", help="run golf.refresh first")
    ap.add_argument("--stats", action="store_true", help="with --refresh, pull PGA public stats")
    ap.add_argument("--weather", action="store_true", help="with --refresh, pull Open-Meteo weather")
    ap.add_argument("--fit", action="store_true", help="with --refresh, refit model")
    ap.add_argument("--use-cache", action="store_true")
    ap.add_argument("--simulate", action="store_true", help="run tournament simulation before reporting")
    ap.add_argument("--edge", action="store_true", help="run market edge pricing before reporting")
    ap.add_argument("--round-3balls", action="store_true", help="run round-specific 3-ball pricing before reporting")
    ap.add_argument("--round", type=int, default=1, dest="round_no")
    ap.add_argument("--sims", type=int, default=100_000)
    ap.add_argument("--course", default="")
    ap.add_argument("--major", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    result = generate_report(
        top=args.top,
        edge_top=args.edge_top,
        threeball_top=args.threeball_top,
        output=args.output,
        archive=args.archive,
        run_refresh=args.refresh,
        stats=args.stats,
        weather=args.weather,
        fit=args.fit,
        use_cache=args.use_cache,
        simulate=args.simulate,
        edges=args.edge,
        round_3balls=args.round_3balls,
        sims=args.sims,
        round_no=args.round_no,
        course=args.course,
        major=args.major,
        seed=args.seed,
    )
    print(f"Weekly report -> {result['output']}")
    if result["archive"]:
        print(f"Archive copy -> {result['archive']}")
    print(f"Rows: {result['predictions']} predictions, {result['edges']} market edges, {result['threeballs']} 3-ball sides")
    for note in result["run_notes"]:
        print(f"  {note}")


if __name__ == "__main__":
    main()

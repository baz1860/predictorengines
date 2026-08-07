#!/usr/bin/env python3
"""The one front door: `python3 -m club_soccer.season [--fast] [--no-network]`.

Runs every step of the daily pipeline, each wrapped so a single step's
failure never kills the run (matches fetch.py/model.py's own
offline-first discipline), then writes club_soccer/data/card.md — the one
human-readable output, matching the golf/tennis/worldcup front-door pattern.

Steps (in order):
  1. health.run_checks() — ABORTS only on the two hard checks (future-dated
     finished rows, duplicate fixture_ids); everything else just reports.
  2. Fetch results + upcoming fixtures (skipped: --no-network / no BSD key).
  3. Pull absences, refresh player cache from new bsd_cache events, rebuild
     squads + transfers (skipped: --no-network / no BSD key).
  4. Snapshot odds; refresh fd.co.uk market history if >= 6 days stale
     (skipped: --no-network / no BSD key). Understat is blocked by its
     robots.txt (see docs/model_improvements_changelog.md) — not attempted.
  5. Refit the model (skipped: --fast).
  6. Standings are computed lazily by the card writer (no separate step).
  7. Price the card: BSD odds only (manual odds.csv needs --allow-manual-odds,
     is age-limited, and only prices future fixtures), with availability
     adjustments and the do-not-bet filter.
  8. Build one structured forecast set; freeze it and render the card from the
     exact same rows.
  9. Settle prior card forecasts and refresh forward performance metrics.
 10. Mondays only: append the latest validate --gate summary to the card.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api_keys import get_key
from . import health as H
from . import fetch as F
from . import model as M
from . import edge as E
from . import standings as ST
from . import club_squads as CS
from . import player_features as PF
from . import snapshot_odds as SO
from . import market_model as MM
from . import fetch_fdcouk as FD
from . import cache_retention as CR
from . import forecast_ledger as FL

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RUNTIME = Path(os.environ.get("CLUB_SOCCER_RUNTIME_DIR", str(DATA)))
# Tests and one-off diagnostics can redirect the card without touching the
# live production artifact. Production keeps the historical default.
CARD = Path(os.environ.get("CLUB_SOCCER_CARD_PATH", str(DATA / "card.md")))
CARD_HORIZON_DAYS = 7
FDCOUK_STALE_DAYS = 6
DAILY_FIXTURE_HORIZON_DAYS = 90
DAILY_RESULT_LOOKBACK_DAYS = 14
# Manual odds.csv is opt-in (--allow-manual-odds); freshness/future-kickoff
# rules live in edge.validate_quotes — the one gate shared with the app.
MANUAL_ODDS_MAX_AGE_DAYS = E.MANUAL_ODDS_MAX_AGE_DAYS


# Required steps that failed this run. The card is still written (degraded is
# better than absent for a human reader) but the process exits nonzero so
# schedulers and update.sh see the failure instead of a false-green run.
_FAILED_REQUIRED: list[str] = []


def _step(title: str, fn: Callable, *args, required: bool = False, **kwargs):
    print(f"\n== {title} ==")
    started = time.monotonic()
    try:
        result = fn(*args, **kwargs)
    except Exception as exc:
        print(f"   {'FAILED' if required else 'skipped'} "
              f"({type(exc).__name__}: {exc})")
        if required:
            _FAILED_REQUIRED.append(f"{title}: {type(exc).__name__}: {exc}")
        return None
    finally:
        elapsed = time.monotonic() - started
        print(f"   elapsed {elapsed:.1f}s", flush=True)
    return result


def _events_within(events: list[dict] | None, days_ahead: int,
                   today=None) -> list[dict] | None:
    """Slice a capture payload to the operational pricing horizon."""
    if events is None:
        return None
    today = today or datetime.now(timezone.utc).date()
    end = today + timedelta(days=days_ahead)
    out = []
    for event in events:
        raw = F.event_date_utc(event)
        try:
            date = datetime.fromisoformat(str(raw).replace("Z", "+00:00")).date()
        except (TypeError, ValueError):
            continue
        if today <= date <= end:
            out.append(event)
    return out


def _fdcouk_is_stale() -> bool:
    if not FD.MARKET_HISTORY.exists():
        return True
    age_s = datetime.now(timezone.utc).timestamp() - FD.MARKET_HISTORY.stat().st_mtime
    return age_s / 86400.0 >= FDCOUK_STALE_DAYS


def _rebuild_squads_and_transfers(store: PF.PlayerFeatureStore) -> None:
    squads = CS.build_squads(store)
    transfers = CS.detect_transfers(store)
    DATA.mkdir(exist_ok=True)
    squads.to_csv(CS.SQUADS_CSV, index=False)
    transfers.to_csv(CS.TRANSFERS_DETECTED_CSV, index=False)
    if not CS.TRANSFERS_MANUAL_CSV.exists():
        CS.TRANSFERS_MANUAL_CSV.write_text("effective_date,player,from_team,to_team\n")
    print(f"   {len(squads)} squad rows, {len(transfers)} detected transfer(s)")


def run_network_steps(api_key: str) -> tuple[list[dict], dict[str, dict]]:
    """Run network capture once and share its responses between consumers."""
    today = datetime.now(timezone.utc).date()
    all_events = _step(
        "Fetch BSD daily event index", F._fetch_bsd_events, api_key,
        date_from=str(today - timedelta(days=DAILY_RESULT_LOOKBACK_DAYS)),
        date_to=str(today + timedelta(days=DAILY_FIXTURE_HORIZON_DAYS)),
        required=True,
    )
    if all_events is None:
        return [], {}

    _step(
        "Merge results + upcoming fixtures", F.fetch_fixtures,
        current=True, api_key=api_key, events=all_events, required=True,
    )

    upcoming_events = [
        ev for ev in all_events
        if F.schema.normalize_status(ev.get("status")) == "NOT"
    ]
    _step("Pull absences", PF.pull_absences, api_key, events=upcoming_events)
    store = PF.PlayerFeatureStore().load()
    n = _step("Refresh player cache from bsd_cache", store.refresh_from_cache)
    if n:
        _step("Rebuild squads + transfers", _rebuild_squads_and_transfers, store)

    odds_cache: dict[str, dict] = {}

    def _snapshot():
        rows = SO.build_snapshot_rows(
            api_key, events=upcoming_events, odds_cache=odds_cache
        )
        return SO.append_snapshots(rows)
    _step("Snapshot odds", _snapshot)

    # Results for the leagues BSD does not carry. Without this the P3
    # expansion is a one-off backfill: 8 competitions — Austria among them —
    # would freeze at their ingest date and go on being priced from ratings
    # that never update again. Eight HTTP requests, so it runs every time
    # rather than on a staleness heuristic.
    def _refresh_expansion():
        from . import seed_fdcouk_leagues as SFL
        result = SFL.refresh(verbose=True)
        failed = [
            comp for comp, count
            in result.get("per_competition", {}).items()
            if int(count) < 0
        ]
        if failed:
            raise RuntimeError(
                "fd.co.uk refresh failed for: " + ", ".join(sorted(failed))
            )
        return result
    _step(
        "Refresh BSD-less leagues (fd.co.uk)", _refresh_expansion,
        required=True,
    )

    if _fdcouk_is_stale():
        _step("Refresh fd.co.uk market history (weekly)", FD.refresh)
    else:
        print("\n== fd.co.uk market history: fresh (< 6 days old), skipped ==")
    return upcoming_events, odds_cache


# ── card sections ──────────────────────────────────────────────────────────
def _freshness_header(today: str, report: dict) -> list[str]:
    absences_today = 0
    if PF.ABSENCES_CSV.exists():
        try:
            adf = pd.read_csv(PF.ABSENCES_CSV)
            absences_today = int((adf["recorded_at"].astype(str).str[:10] == today).sum())
        except Exception:
            pass
    snap_age = "n/a"
    if SO.ODDS_HISTORY_CSV.exists():
        try:
            h = pd.read_csv(SO.ODDS_HISTORY_CSV)
            if not h.empty:
                latest = pd.to_datetime(h["snapshot_time"], utc=True, format="mixed",
                                        errors="coerce").max()
                if pd.notna(latest):
                    hrs = (pd.Timestamp.now(tz="UTC") - latest).total_seconds() / 3600.0
                    snap_age = f"{hrs:.1f}h"
        except Exception:
            pass
    lines = [
        f"# Club Soccer — {today}",
        "",
        f"- Days since last result: {report.get('days_since_last_result')}",
        f"- Upcoming fixtures: {report.get('upcoming_count')}",
        f"- Absence rows recorded today: {absences_today}",
        f"- Odds snapshot age: {snap_age}",
        f"- Stats (SoT) coverage: "
        f"{report.get('stats_coverage'):.1%}" if report.get("stats_coverage") is not None else
        "- Stats (SoT) coverage: n/a",
        "",
    ]
    return lines


class _TableCache:
    def __init__(self) -> None:
        self._cache: dict[tuple, pd.DataFrame] = {}

    def get(self, competition: str, season: int, date: str) -> pd.DataFrame:
        key = (competition, season, date)
        if key not in self._cache:
            self._cache[key] = ST.table_asof(competition, season, date)
        return self._cache[key]

    def position_note(self, competition: str, season: int, date: str,
                      home: str, away: str) -> str:
        table = self.get(competition, season, date)
        if table.empty:
            return ""
        pos = {r.team: r.position for r in table.itertuples(index=False)}
        ph, pa = pos.get(home), pos.get(away)
        if ph is None or pa is None:
            return ""
        return f"{_ordinal(ph)} v {_ordinal(pa)}"


def _ordinal(n: int) -> str:
    n = int(n)
    if 10 <= n % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def _upcoming_section(forecasts: list[dict]) -> list[str]:
    """Render the exact structured rows frozen by ``forecast_ledger``.

    Prediction used to happen inside this Markdown loop, which made it
    impossible to prove that a separately tracked probability was the one a
    user actually saw. Forecast construction now happens once; rendering is a
    pure consumer of those immutable rows.
    """
    if not forecasts:
        return ["## Next 7 days", "", "No upcoming fixtures found.", ""]
    up = pd.DataFrame(forecasts).sort_values(
        ["match_date", "competition", "home"]
    )
    tables = _TableCache()
    lines = ["## Next 7 days", ""]
    for day, day_grp in up.groupby("match_date"):
        lines.append(f"### {day}")
        for comp, comp_grp in day_grp.groupby("competition"):
            lines.append(f"**{comp}**")
            for r in comp_grp.itertuples(index=False):
                p = {"home": float(r.p_home), "draw": float(r.p_draw),
                     "away": float(r.p_away)}
                fair = {k: (round(1.0 / v, 2) if v > 0 else None) for k, v in p.items()}
                bits = [f"{r.home} vs {r.away} — "
                       f"H {p['home']:.0%} D {p['draw']:.0%} A {p['away']:.0%} "
                       f"(fair {fair['home']}/{fair['draw']}/{fair['away']})"]

                n_out = int(r.n_missing_home) + int(r.n_missing_away)
                if n_out:
                    arrow_h = "▼" if float(r.home_attack_mult) < 1.0 else "▲"
                    arrow_a = "▼" if float(r.away_attack_mult) < 1.0 else "▲"
                    bits.append(f"availability: {r.home} {arrow_h} "
                               f"{float(r.home_attack_mult):.2f} / {r.away} {arrow_a} "
                               f"{float(r.away_attack_mult):.2f} "
                               f"(confidence {float(r.lineup_confidence):.0%}, {n_out} out)")

                if r.type == "league" and str(r.season).strip():
                    pos_note = tables.position_note(
                        comp, int(float(r.season)), str(r.match_date), r.home, r.away
                    )
                    if pos_note:
                        bits.append(f"table: {pos_note}")

                lines.append("- " + "; ".join(bits))
            lines.append("")
    return lines


def _capped_stakes(backed: list[dict]) -> dict[int, tuple[float, bool]]:
    """Portfolio-capped stakes for DISPLAY, using the same pure cap function
    recording applies (app.portfolio.apply_caps) on the card's 100-unit
    bankroll. Without this the card shows stakes the recorder would later
    reduce — the operator would plan a portfolio that can't be placed.
    Returns {row_index: (capped_stake, was_capped)}; rows capped below the
    minimum stake map to (0.0, True)."""
    try:
        from app.portfolio import apply_caps
    except Exception:
        return {}
    cand = [{"stake": float(r.get("stake_gbp") or 0.0),
             "event_id": f"{r.get('home')}|{r.get('away')}|{r.get('date')}",
             "engine": "club_soccer", "_i": i} for i, r in enumerate(backed)]
    try:
        capped = apply_caps(cand, bankroll=100.0)
    except Exception:
        return {}
    out = {i: (0.0, True) for i in range(len(backed))}
    for c in capped:
        out[c["_i"]] = (float(c["stake"]), bool(c.get("stake_capped")))
    return out


def _evidence_cell(row: dict) -> str:
    """Compact evidence marker for the bet tables (P0, report-only)."""
    tier = str(row.get("evidence_tier", "unknown"))
    if tier == "full":
        return "ok"
    nh = int(row.get("n_matches_home", 0) or 0)
    na = int(row.get("n_matches_away", 0) or 0)
    return f"**{tier.upper()}** ({nh}/{na} matches)"


def _low_evidence_section(edge_rows: list[dict]) -> list[str]:
    """Flag priced fixtures whose ratings rest on little or no real data.

    Pricing deliberately continues for these (P0 is measurement, not
    suppression), so this section is the safeguard: any suggestion built on a
    team the model has barely observed is named here, with the reason, at the
    point the suggestion is made.
    """
    seen: dict[str, dict] = {}
    for r in edge_rows:
        if str(r.get("evidence_tier", "full")) == "full":
            continue
        key = f"{r.get('date','')}|{r.get('match','')}"
        if key not in seen:
            seen[key] = r
    if not seen:
        return []
    lines = ["## ⚠ Low-evidence fixtures", "",
             "The model has little or no real data on the teams below, so these "
             "prices are weakly identified. They are shown because pricing "
             "continues through P0 by design — treat the probabilities as "
             "provisional, not as a read.", "",
             "| Date | Match | Tier | Why |", "|---|---|---|---|"]
    for r in sorted(seen.values(), key=lambda x: str(x.get("date", ""))):
        note = str(r.get("evidence_note", "")) or "under-evidenced"
        lines.append(f"| {r.get('date','')} | {r.get('match','')} | "
                     f"{str(r.get('evidence_tier','')).upper()} | {note} |")
    lines.append("")
    return lines


# Card leads with likely WINNERS, not edge. The goal here is a steady stream of
# bets the model expects to come in, with value flagged as secondary context —
# not high-edge longshots. Confidence (p_model) is the primary sort; edge is a
# column. Full-evidence only, so early qualifiers and barely-seen clubs never
# lead the card.
LIKELY_MIN_P = 0.55          # "likely to land" threshold
SWEET_MIN_P = 0.53          # a real chance of winning …
SWEET_MIN_EDGE = 0.03        # … AND the price is generous


def _value_flag(edge: float) -> str:
    if edge > SWEET_MIN_EDGE:
        return "value"
    if edge < -SWEET_MIN_EDGE:
        return "short"
    return "fair"


def _bet_prob(row: dict) -> float:
    """Model probability that THIS bet lands (side-aware)."""
    try:
        return float(row.get("p_model", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _likely_winners_section(edge_rows: list[dict], today_str: str) -> list[str]:
    """The lead section: bets ranked by how likely the model thinks they are.

    Two tables — the most likely to land (confidence first), then the sweet
    spot where a likely pick is ALSO underpriced. Value is shown but never the
    sort key, because the aim is regular winners, not maximised edge.
    """
    # The card is a betting surface, so its lead table obeys the same evidence
    # gate as the backed-bets table. Showing odds-ranked picks above the gate
    # while calling them merely "informational" made the safety boundary
    # cosmetic: the most prominent content still looked backable.
    live = [r for r in edge_rows
            if str(r.get("date", ""))[:10] >= today_str
            and not r.get("suppressed_reason")
            and float(r.get("kelly_stake", 0) or 0) > 0
            and str(r.get("evidence_tier", "")) == "full"]
    likely = sorted((r for r in live if _bet_prob(r) >= LIKELY_MIN_P),
                    key=_bet_prob, reverse=True)
    sweet = sorted((r for r in live if _bet_prob(r) >= SWEET_MIN_P
                    and float(r.get("edge", 0) or 0) > SWEET_MIN_EDGE),
                   key=_bet_prob, reverse=True)

    lines = ["## Most likely to land", ""]
    lines.append("_Gate-approved, full-evidence picks ranked by the model's chance "
                 "the bet wins. "
                 "`value` = also underpriced, `short` = likely but the book has it "
                 "tight, `fair` = square._")
    lines.append("")
    if not likely:
        lines.append("_No gate-approved, full-evidence pick clears "
                     f"{LIKELY_MIN_P:.0%} on the current board — typically an "
                     "off-season week with only tight fixtures priced._")
        lines.append("")
    else:
        lines.append("| Date | Match | Bet | Odds | Model | Value |")
        lines.append("|---|---|---|---|---|---|")
        for r in likely[:20]:
            lines.append(f"| {str(r.get('date',''))[:10]} | {r.get('match','')} | "
                         f"{r.get('bet','')} | {float(r.get('odds',0)):.2f} | "
                         f"{_bet_prob(r):.0%} | {_value_flag(float(r.get('edge',0) or 0))} |")
        lines.append("")

    if sweet:
        lines.append("### Sweet spot — likely *and* underpriced")
        lines.append("")
        lines.append("| Date | Match | Bet | Odds | Model | Edge |")
        lines.append("|---|---|---|---|---|---|")
        for r in sweet[:12]:
            lines.append(f"| {str(r.get('date',''))[:10]} | {r.get('match','')} | "
                         f"{r.get('bet','')} | {float(r.get('odds',0)):.2f} | "
                         f"{_bet_prob(r):.0%} | {float(r.get('edge',0) or 0):+.1%} |")
        lines.append("")
    return lines


def _backed_bets_section(edge_rows: list[dict], today_str: str,
                         pricing_note: str | None = None) -> list[str]:
    # Defense in depth: even if a stale quote survives upstream filtering,
    # a fixture dated before today is never presented as a backable bet.
    backed = [r for r in edge_rows if r.get("ev_per_unit", 0) > 0
             and float(r.get("kelly_stake", 0) or 0) > 0 and not r.get("suppressed_reason")
             and str(r.get("date", ""))[:10] >= today_str]
    suppressed = [r for r in edge_rows if r.get("suppressed_reason")]
    lines = ["## Backed bets", ""]
    if pricing_note:
        lines.append(f"_{pricing_note}_")
        lines.append("")
    if not backed:
        lines.append("No positive-EV, unsuppressed bets on the current odds.")
    else:
        caps = _capped_stakes(backed)
        lines.append("| Match | Market | Bet | Odds | Edge | Stake | Stake (capped) | Evidence |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for i, r in enumerate(backed):
            cs, was_capped = caps.get(i, (float(r["stake_gbp"]), False))
            cap_cell = f"£{cs:.2f}" + ("*" if was_capped else "")
            lines.append(f"| {r['match']} | {r['market']} | {r['bet']} | {r['odds']} | "
                         f"{r['edge']:+.1%} | £{r['stake_gbp']:.2f} | {cap_cell} | "
                         f"{_evidence_cell(r)} |")
        if any(v[1] for v in caps.values()):
            lines.append("")
            lines.append("_\\* reduced by portfolio caps (event/correlated/daily/"
                         "drawdown) — recording places the capped amount._")
    lines.append("")
    gate_suppressed = [r for r in suppressed
                       if str(r.get("suppressed_reason", "")).startswith("evidence-gate")]
    other_suppressed = [r for r in suppressed if r not in gate_suppressed]
    if gate_suppressed:
        lines.append(f"_Evidence gate CLOSED: {len(gate_suppressed)} priced outcome(s) "
                     "shown for diagnostics only — the stored backtest shows no "
                     "demonstrated edge, so no stakes are recommended "
                     "(`python3 -m club_soccer.evidence_gate` for the criteria)._")
        lines.append("")
    if other_suppressed:
        lines.append(f"_{len(other_suppressed)} bet(s) suppressed by the market-model "
                     "do-not-bet filter._")
        lines.append("")
    return lines


def _transfers_absences_section() -> list[str]:
    lines = ["## Transfers & absences", ""]
    if CS.TRANSFERS_DETECTED_CSV.exists():
        try:
            t = pd.read_csv(CS.TRANSFERS_DETECTED_CSV)
        except Exception:
            t = pd.DataFrame()
        if not t.empty:
            lines.append("**Recently detected transfers:**")
            for r in t.head(10).itertuples(index=False):
                lines.append(f"- {r.player}: {r.from_team} -> {r.to_team} ({r.date})")
            lines.append("")
    if PF.ABSENCES_CSV.exists():
        try:
            a = pd.read_csv(PF.ABSENCES_CSV)
        except Exception:
            a = pd.DataFrame()
        if not a.empty:
            latest = (a.sort_values("recorded_at")
                     .drop_duplicates(subset=["match_date", "team", "player"], keep="last"))
            upcoming_cut = str(datetime.now(timezone.utc).date())
            notable = latest[latest["match_date"] >= upcoming_cut].head(15)
            if not notable.empty:
                lines.append("**Notable absences (upcoming matches):**")
                for r in notable.itertuples(index=False):
                    lines.append(f"- {r.team}: {r.player} ({r.status}, {r.reason}) — {r.match_date}")
                lines.append("")
    if len(lines) == 2:
        lines.append("Nothing new to report.")
        lines.append("")
    return lines


_VALIDATION_MAX_AGE_DAYS = 8


def _read_validation_latest() -> dict | None:
    """Metrics from the last `validate --gate`, if they are recent enough.

    Returns None for a stale or unreadable artifact rather than the numbers it
    contains. Displaying a month-old Brier on today's card as though it were
    current is worse than displaying nothing — the card is where a regression
    is supposed to become visible.
    """
    path = DATA / "validation_latest.json"
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text())
    except Exception:
        return None
    stamp = doc.get("generated_at_utc")
    if not stamp:
        return None
    try:
        age = (datetime.now(timezone.utc)
               - datetime.fromisoformat(stamp)).total_seconds() / 86400
    except ValueError:
        return None
    return doc if age <= _VALIDATION_MAX_AGE_DAYS else None


def _weekly_footer() -> list[str]:
    lines = ["## Weekly check (Mondays)", ""]
    # Read the metrics the validation gate already wrote rather than running a
    # second full walk-forward. update.sh invokes `validate --gate` on every
    # run, so recomputing here duplicated a ~2-minute pass every Monday for an
    # identical number. Folds are cached now, which would have made the repeat
    # cheap, but free work is still work worth not doing.
    m = _read_validation_latest()
    if m:
        stamp = f", {m['generated_at_utc'][:10]}" if m.get("generated_at_utc") else ""
        lines.append(f"- Walk-forward Brier: {m['brier']:.4f} (n={m['n']}{stamp})")
    else:
        lines.append("- Walk-forward Brier: unavailable "
                     "(run `python3 -m club_soccer.validate --gate`)")
    lines.append("")
    return lines


def write_card(edge_rows: list[dict], player_adj_map: dict | None,
               calib_maps: dict | None = None,
               pricing_note: str | None = None,
               health_report: dict | None = None,
               forecasts: list[dict] | None = None) -> None:
    today = datetime.now(timezone.utc)
    today_str = str(today.date())
    lines: list[str] = []
    lines += _freshness_header(today_str, health_report or {})
    # Lead with likely winners — the primary use is a steady stream of bets the
    # model expects to come in. Edge is context, not the headline.
    lines += _likely_winners_section(edge_rows, today_str)
    if forecasts is None:
        # Compatibility for direct callers. Production passes pre-built rows so
        # the renderer and append-only ledger consume the identical objects.
        forecasts = FL.build_forecasts(
            pd.Timestamp(today.date()), player_adj_map, calib_maps,
            run_id=f"unrecorded-{uuid.uuid4().hex}", run_mode="unrecorded",
            primary_eligible=False, forecast_ts=today,
        )
    lines += _upcoming_section(forecasts)
    lines += _backed_bets_section(edge_rows, today_str, pricing_note)
    lines += _low_evidence_section(edge_rows)
    lines += _transfers_absences_section()
    if today.weekday() == 0:   # Monday
        lines += _weekly_footer()
    CARD.parent.mkdir(parents=True, exist_ok=True)
    CARD.write_text("\n".join(lines))
    print(f"\nWrote {CARD}")


# ── main ──────────────────────────────────────────────────────────────────
def _file_age_days(path: Path) -> float | None:
    try:
        return round((datetime.now(timezone.utc).timestamp()
                      - path.stat().st_mtime) / 86400.0, 2)
    except (FileNotFoundError, OSError):
        return None


LAST_RUN = RUNTIME / "last_run.json"
LOCK_FILE = RUNTIME / "season.lock"


class _SeasonLock:
    """Exclusive, non-blocking inter-process lock for a season run.

    Two overlapping launchd/manual runs must never interleave their status
    writes. The second caller fails fast with SystemExit rather than racing
    the first run's last_run.json."""

    def __init__(self, path: Path = LOCK_FILE):
        self.path = path
        self._fh = None

    def __enter__(self) -> "_SeasonLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w")
        try:
            fcntl.flock(self._fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._fh.close()
            self._fh = None
            raise SystemExit(
                f"Another club_soccer.season run holds {self.path.name}; "
                "refusing to run concurrently (overlapping runs corrupt status).")
        self._fh.write(f"{os.getpid()} "
                       f"{datetime.now(timezone.utc).isoformat()}\n")
        self._fh.flush()
        return self

    def __exit__(self, *exc) -> None:
        if self._fh is not None:
            try:
                fcntl.flock(self._fh, fcntl.LOCK_UN)
            finally:
                self._fh.close()
                self._fh = None


def _publish_status(payload: dict, run_id: str, claim: bool = False) -> None:
    """Atomically publish last_run.json via a run-unique tmp file.

    `claim=True` (the starting marker) takes ownership unconditionally: the
    exclusive lock already guarantees we are the only live run, and a stale
    marker left by a killed run must NOT wedge every future run out of
    publishing status.

    `claim=False` (the finished record) publishes only if we still own the
    status, so a run that somehow lost the race cannot overwrite the live
    run's marker with its own stale result. Either way the tmp file is
    run-unique, so two runs never fight over one .tmp path."""
    try:
        LAST_RUN.parent.mkdir(parents=True, exist_ok=True)
        if not claim:
            try:
                cur_id = json.loads(LAST_RUN.read_text()).get("run_id")
            except (FileNotFoundError, ValueError, OSError, AttributeError):
                cur_id = None
            if cur_id is not None and cur_id != run_id:
                print(f"   last_run.json owned by run {cur_id} — not overwriting")
                return
        tmp = LAST_RUN.parent / f"last_run.json.{run_id}.tmp"
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(LAST_RUN)
    except Exception as exc:
        print(f"   last_run.json not written ({exc})")


def _write_last_run(edge_rows: list[dict], pricing_note: str | None,
                    fast: bool, no_network: bool, run_id: str,
                    crashed: str | None = None) -> None:
    """Atomic, written on EVERY exit path (success, required-step failure,
    health hard-fail, uncaught crash). A monitor polls this file — a stale
    green status after a fatal run would be a false all-clear. Carries the same
    run_id as the running marker so status races are detectable."""
    status = {
        "state": "finished",
        "run_id": run_id,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "ok": not _FAILED_REQUIRED and crashed is None,
        "crashed": crashed,
        "failed_required_steps": list(_FAILED_REQUIRED),
        "pricing_note": pricing_note,
        "n_edge_rows": len(edge_rows),
        "n_backed": sum(1 for r in edge_rows
                        if float(r.get("kelly_stake", 0) or 0) > 0
                        and not r.get("suppressed_reason")),
        # Optional-input freshness: "0 absences" from a swallowed network
        # failure looks identical to a real quiet day — the file ages tell
        # the monitor (and the operator) which one it was.
        "absences_age_days": _file_age_days(PF.ABSENCES_CSV),
        "squads_age_days": _file_age_days(CS.SQUADS_CSV),
        "odds_snapshot_age_days": _file_age_days(SO.ODDS_HISTORY_CSV),
        "fast": fast, "no_network": no_network,
    }
    try:
        forecast_status = FL.status()
        status.update({
            "forecast_rows": forecast_status["forecast_rows"],
            "forecast_fixtures": forecast_status["forecast_fixtures"],
            "forecast_settled_fixtures": forecast_status["settled_fixtures"],
        })
        if FL.PERFORMANCE.exists():
            performance = json.loads(FL.PERFORMANCE.read_text())
            status["forecast_t24_scored"] = int(
                ((performance.get("cohorts") or {}).get("t24") or {}).get("n", 0)
            )
    except Exception:
        # Status publication is fail-safe; forecast observability is useful but
        # must not hide the core run outcome if its artifact is unreadable.
        pass
    _publish_status(status, run_id)

    # Append to the run ledger (P6). last_run.json answers "did the last run
    # work?"; the ledger answers "has this been working?", which is the only
    # way to see the slow failures — a league quietly ceasing to update,
    # coverage eroding — where every individual run still reports green.
    # Failures are recorded too: a history of successes alone cannot measure a
    # streak. Best-effort, so observability can never fail the pipeline.
    try:
        from . import run_ledger as RL
        RL.append({**status, **RL.snapshot()})
    except Exception as exc:
        print(f"   run ledger not written ({exc})")


def _write_running_marker(run_id: str) -> None:
    """Written atomically BEFORE any work: a SIGKILL/power loss mid-run then
    leaves state="running" with a started_at_utc, not the previous green
    status — the monitor fails a running state older than its deadline. The
    UUID run_id is preserved through to the finished record."""
    marker = {
        "state": "running",
        "run_id": run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    # Claim ownership: a stale marker from a killed run must not block us.
    _publish_status(marker, run_id, claim=True)


def run(fast: bool = False, no_network: bool = False,
        allow_manual_odds: bool = False) -> None:
    """Wrapper: every exit path — including the health hard-fail and any
    uncaught crash — records a durable last_run.json before propagating.

    An exclusive lock is held for the whole run so overlapping launchd/manual
    invocations cannot interleave their status writes; the second caller exits
    immediately via SystemExit."""
    with _SeasonLock():
        _FAILED_REQUIRED.clear()   # per-run state: never inherit an old failure
        run_id = uuid.uuid4().hex
        _write_running_marker(run_id)
        edge_rows: list[dict] = []
        pricing_note: str | None = None
        crashed: str | None = None
        pending_exit: BaseException | None = None
        try:
            edge_rows, pricing_note = _run_steps(
                fast, no_network, allow_manual_odds, run_id=run_id
            )
        except SystemExit as exc:
            crashed = f"SystemExit: {exc}"
            pending_exit = exc
        except BaseException as exc:
            crashed = f"{type(exc).__name__}: {exc}"
            pending_exit = exc
        finally:
            _write_last_run(edge_rows, pricing_note, fast, no_network,
                            run_id, crashed)
        if pending_exit is not None:
            raise pending_exit

    if _FAILED_REQUIRED:
        print(f"\n{len(_FAILED_REQUIRED)} REQUIRED step(s) failed "
              "(card written in degraded mode):")
        for msg in _FAILED_REQUIRED:
            print(f"  - {msg}")
        sys.exit(3)


def _run_steps(fast: bool, no_network: bool,
               allow_manual_odds: bool, run_id: str) -> tuple[list[dict], str | None]:
    _step("Prune recoverable caches if due", CR.prune_all_if_due)
    # Source freshness used to download fd.co.uk here and then download it
    # again in run_network_steps. Daily health is deliberately offline; the
    # following refresh is the single authoritative source fetch.
    report = H.run_checks(network=False)
    if not report.get("ok", True):
        _FAILED_REQUIRED.append("Health hard checks failed "
                                "(see health output above)")
        sys.exit("Health check hard-failed (future-dated finished rows, "
                 "duplicate fixture_ids, or void rows with results) — run "
                 "`python3 -m club_soccer.fetch --repair` before continuing.")

    api_key = get_key("bsd", env="BSD_API_KEY")
    upcoming_events: list[dict] | None = None
    odds_cache: dict[str, dict] | None = None
    if no_network or not api_key:
        reason = "--no-network" if no_network else "no BSD key"
        print(f"\n== Network steps skipped ({reason}) ==")
    else:
        upcoming_events, odds_cache = run_network_steps(api_key)
    pricing_events = _events_within(upcoming_events, CARD_HORIZON_DAYS)
    if pricing_events is not None:
        print(
            f"\n== Pricing horizon: {len(pricing_events)} event(s) in the next "
            f"{CARD_HORIZON_DAYS} days =="
        )

    if not fast:
        _step("Fit model if training data changed", M.fit_if_changed, required=True)
    else:
        print("\n== Refit skipped (--fast) ==")

    # Staking evidence, in three steps, all best-effort (never break the card):
    #  1. record  — freeze today's in-window decisions into the append-only
    #     ledger (immutable; the trustworthy basis for the gate).
    #  2. settle  — append results/CLV for decisions whose fixtures finished.
    #  3. backtest — recompute the gate artifact FROM the frozen ledger.
    if not no_network and api_key:
        def _record():
            from . import decision_ledger as DL
            return DL.record(
                api_key, verbose=True, events=pricing_events,
                odds_cache=odds_cache,
            )
        _step("Record staking decisions (ledger)", _record)

        def _close():
            from . import decision_ledger as DL
            return DL.capture_closing(api_key, verbose=True)
        _step("Capture near-kickoff closing quotes (ledger)", _close)
    def _settle():
        from . import decision_ledger as DL
        return DL.settle(verbose=True)
    _step("Settle staking decisions (ledger)", _settle)
    _step("Settle published card forecasts", FL.settle)

    def _decision_time():
        from . import decision_time_backtest as DTB
        return DTB.run(verbose=False)
    _step("Decision-time backtest (staking evidence)", _decision_time)

    player_adj_map = None
    from .calibrate import load_active_maps
    calib_maps = load_active_maps()
    if not no_network and api_key:
        player_adj_map = _step(
            "Player availability adjustments", E.fetch_player_adjustments,
            api_key, events=pricing_events,
        )

    edge_rows: list[dict] = []
    pricing_note: str | None = None
    if not no_network and api_key:
        # required: a networked daily run whose whole point is pricing must
        # not report success when the odds fetch itself failed. ("Provider
        # has no quoted markets" raises inside fetch_bsd_odds too — that's a
        # visible failure by design, not silently zero rows.)
        odds = _step(
            "Fetch BSD odds", E.fetch_bsd_odds, api_key,
            events=pricing_events, required=True,
        )
        if odds is not None:
            odds, issues = E.validate_quotes(odds, source="live")
            for msg in issues:
                print(f"   quote-validation: {msg}")
        if odds is not None and not odds.empty:
            history_days = MM.history_age_days()
            edge_rows = _step("Price the card", E.rows_from_odds, odds, "ensemble", 100.0,
                             calib_maps, player_adj_map, history_days >= MM.WARMUP_DAYS,
                             required=True) or []
        if not edge_rows:
            pricing_note = "Live pricing unavailable or failed — no bets staked."

    # Manual odds.csv is never an automatic substitute for live pricing: it
    # is opt-in and gated by edge.validate_quotes (age limit + future-kickoff
    # filter). The failure mode this prevents: live pricing crashes, a
    # weeks-old manual file silently prices the card, settled matches staked.
    if not edge_rows and allow_manual_odds:
        try:
            odds = E.load_odds()
        except FileNotFoundError:
            odds = None
            pricing_note = "No odds available (live or manual) — card has no bets."
        if odds is not None:
            odds, issues = E.validate_quotes(odds, source="manual")
            if odds.empty:
                pricing_note = ("Manual odds.csv unusable: " + "; ".join(issues)
                                if issues else
                                "Manual odds.csv has no future-dated fixtures — nothing staked.")
            else:
                edge_rows = E.rows_from_odds(odds, "ensemble", 100.0,
                                             calib_maps=calib_maps)
                pricing_note = ("Priced from manual odds.csv."
                                + (" " + "; ".join(issues) if issues else ""))
    elif not edge_rows and pricing_note is None:
        pricing_note = ("No live odds — card has no bets "
                        "(manual odds.csv requires --allow-manual-odds).")
    if pricing_note:
        print(f"\n== {pricing_note} ==")

    run_mode = "production_network" if not no_network and api_key else "offline"
    forecasts = _step(
        "Build card forecasts", FL.build_forecasts,
        pd.Timestamp(datetime.now(timezone.utc).date()), player_adj_map, calib_maps,
        run_id=run_id, run_mode=run_mode,
        primary_eligible=(run_mode == "production_network" and not _FAILED_REQUIRED),
        horizon_days=CARD_HORIZON_DAYS, required=True,
    ) or []
    write_card(edge_rows, player_adj_map, calib_maps, pricing_note, report,
               forecasts=forecasts)
    if forecasts:
        # If forecast construction succeeded before a later required failure,
        # preserve what the card showed but keep degraded rows out of the
        # primary T-24 performance cohort.
        if _FAILED_REQUIRED:
            for forecast in forecasts:
                forecast["primary_eligible"] = 0
        _step("Freeze published card forecasts", FL.append_forecasts,
              forecasts, required=True)
    _step("Score published card forecasts", FL.write_performance_report)
    return edge_rows, pricing_note


def main() -> None:
    ap = argparse.ArgumentParser(description="Club Soccer daily front door")
    ap.add_argument("--fast", action="store_true", help="skip the model refit")
    ap.add_argument("--no-network", action="store_true",
                    help="skip every network step; build the card from cached data only")
    ap.add_argument("--allow-manual-odds", action="store_true",
                    help="permit pricing from manual data/odds.csv when live odds are "
                         f"unavailable (age-limited to {MANUAL_ODDS_MAX_AGE_DAYS:g} days, "
                         "future fixtures only)")
    args = ap.parse_args()
    run(fast=args.fast, no_network=args.no_network,
        allow_manual_odds=args.allow_manual_odds)


if __name__ == "__main__":
    main()

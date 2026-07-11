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
  7. Price the card: BSD odds if available else manual odds.csv, with
     availability report, context multipliers (whatever's promoted — none
     are, by design, until a future gate passes), do-not-bet filter.
  8. Write club_soccer/data/card.md.
  9. Mondays only: append a validate --gate + backtest_market summary to
     the card footer.
"""
from __future__ import annotations

import argparse
import sys
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
from . import context as CTX
from . import club_squads as CS
from . import player_features as PF
from . import snapshot_odds as SO
from . import market_model as MM
from . import fetch_fdcouk as FD

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CARD = DATA / "card.md"
CARD_HORIZON_DAYS = 7
FDCOUK_STALE_DAYS = 6


def _step(title: str, fn: Callable, *args, **kwargs):
    print(f"\n== {title} ==")
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        print(f"   skipped ({type(exc).__name__}: {exc})")
        return None


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


def run_network_steps(api_key: str) -> None:
    _step("Fetch results + upcoming fixtures", F.fetch_fixtures, current=True, api_key=api_key)

    _step("Pull absences", PF.pull_absences, api_key)
    store = PF.PlayerFeatureStore().load()
    n = _step("Refresh player cache from bsd_cache", store.refresh_from_cache)
    if n:
        _step("Rebuild squads + transfers", _rebuild_squads_and_transfers, store)

    def _snapshot():
        rows = SO.build_snapshot_rows(api_key)
        return SO.append_snapshots(rows)
    _step("Snapshot odds", _snapshot)

    if _fdcouk_is_stale():
        _step("Refresh fd.co.uk market history (weekly)", FD.build)
    else:
        print("\n== fd.co.uk market history: fresh (< 6 days old), skipped ==")


# ── card sections ──────────────────────────────────────────────────────────
def _freshness_header(today: str) -> list[str]:
    report = H.run_checks()
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


def _upcoming_section(today_ts: pd.Timestamp, player_adj_map: dict | None,
                      calib_maps: dict | None = None) -> list[str]:
    fx = M.load_fixtures()
    up = M.upcoming(fx)
    horizon = today_ts + pd.Timedelta(days=CARD_HORIZON_DAYS)
    up = up[(up["date"] >= today_ts) & (up["date"] <= horizon)].sort_values(
        ["date", "competition", "home"])
    if up.empty:
        return ["## Next 7 days", "", "No upcoming fixtures found.", ""]

    params = M.load_params()
    tables = _TableCache()
    lines = ["## Next 7 days", ""]
    for day, day_grp in up.groupby(up["date"].dt.strftime("%Y-%m-%d")):
        lines.append(f"### {day}")
        for comp, comp_grp in day_grp.groupby("competition"):
            lines.append(f"**{comp}**")
            for r in comp_grp.itertuples(index=False):
                if r.home not in set(params["teams"]) or r.away not in set(params["teams"]):
                    continue
                p_adj = None
                if player_adj_map:
                    p_adj = player_adj_map.get((str(r.home).lower(), str(r.away).lower(), comp))
                try:
                    pred = M.predict_match(
                        r.home, r.away, comp, str(r.date.date()), "ensemble",
                        bool(r.neutral), params=params, player_adj=p_adj,
                        fixture_id=getattr(r, "fixture_id", None),
                    )
                    if calib_maps is not None:
                        from .calibrate import apply as apply_calibration
                        ph, pdr, pa = apply_calibration(
                            pred["probs"]["home"], pred["probs"]["draw"],
                            pred["probs"]["away"], calib_maps)
                        pred["probs"].update({"home": ph, "draw": pdr, "away": pa})
                except ValueError:
                    continue
                p = pred["probs"]
                fair = {k: (round(1.0 / v, 2) if v > 0 else None) for k, v in p.items()}
                bits = [f"{r.home} vs {r.away} — "
                       f"H {p['home']:.0%} D {p['draw']:.0%} A {p['away']:.0%} "
                       f"(fair {fair['home']}/{fair['draw']}/{fair['away']})"]

                if p_adj:
                    h_a, a_a = p_adj.get("home", {}), p_adj.get("away", {})
                    conf = p_adj.get("lineup_confidence", 1.0)
                    if h_a or a_a:
                        n_out = int(h_a.get("n_missing", 0)) + int(a_a.get("n_missing", 0))
                        if n_out:
                            arrow_h = "▼" if h_a.get("attack_mult", 1.0) < 1.0 else "▲"
                            arrow_a = "▼" if a_a.get("attack_mult", 1.0) < 1.0 else "▲"
                            bits.append(f"availability: {r.home} {arrow_h} "
                                       f"{h_a.get('attack_mult', 1.0):.2f} / {r.away} {arrow_a} "
                                       f"{a_a.get('attack_mult', 1.0):.2f} (confidence {conf:.0%}, "
                                       f"{n_out} out)")

                is_domestic = r.type in ("league", "cup")
                ctx_feats = CTX.context_features_asof(
                    r.home, r.away, str(r.date.date()), is_domestic, comp,
                    int(r.season) if pd.notna(r.season) else None, r.type == "cup")
                if ctx_feats["rest_diff"] or ctx_feats["cong14_diff"]:
                    bits.append(f"rest/congestion: rest_diff={ctx_feats['rest_diff']:+.0f}d "
                               f"cong14_diff={ctx_feats['cong14_diff']:+.0f}")

                if r.type == "league" and pd.notna(r.season):
                    pos_note = tables.position_note(comp, int(r.season), str(r.date.date()),
                                                     r.home, r.away)
                    if pos_note:
                        bits.append(f"table: {pos_note}")

                lines.append("- " + "; ".join(bits))
            lines.append("")
    return lines


def _backed_bets_section(edge_rows: list[dict]) -> list[str]:
    backed = [r for r in edge_rows if r.get("ev_per_unit", 0) > 0
             and float(r.get("kelly_stake", 0) or 0) > 0 and not r.get("suppressed_reason")]
    suppressed = [r for r in edge_rows if r.get("suppressed_reason")]
    lines = ["## Backed bets", ""]
    if not backed:
        lines.append("No positive-EV, unsuppressed bets on the current odds.")
    else:
        lines.append("| Match | Market | Bet | Odds | Edge | Stake |")
        lines.append("|---|---|---|---|---|---|")
        for r in backed:
            lines.append(f"| {r['match']} | {r['market']} | {r['bet']} | {r['odds']} | "
                         f"{r['edge']:+.1%} | £{r['stake_gbp']:.2f} |")
    lines.append("")
    if suppressed:
        lines.append(f"_{len(suppressed)} bet(s) suppressed by the market-model do-not-bet filter._")
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


def _weekly_footer() -> list[str]:
    lines = ["## Weekly check (Mondays)", ""]
    from . import validate as V
    from . import backtest_market as BM
    _, m = _step("Weekly walk-forward validate", V.walk_forward, verbose=False)
    if m:
        lines.append(f"- Walk-forward Brier: {m['brier']:.4f} (n={m['n']})")
    bt = _step("Weekly backtest_market summary", BM.run, verbose=False)
    if bt and bt.get("n_matched"):
        lines.append(f"- backtest_market: {bt['n_matched']} matched rows, model 1X2 "
                     f"log-loss {bt.get('model_log_loss_1x2')} vs market "
                     f"{bt.get('market_log_loss_1x2_devigged_pinnacle_closing')}")
    lines.append("")
    return lines


def write_card(edge_rows: list[dict], player_adj_map: dict | None,
               calib_maps: dict | None = None) -> None:
    today = datetime.now(timezone.utc)
    today_str = str(today.date())
    lines: list[str] = []
    lines += _freshness_header(today_str)
    lines += _upcoming_section(pd.Timestamp(today.date()), player_adj_map, calib_maps)
    lines += _backed_bets_section(edge_rows)
    lines += _transfers_absences_section()
    if today.weekday() == 0:   # Monday
        lines += _weekly_footer()
    DATA.mkdir(exist_ok=True)
    CARD.write_text("\n".join(lines))
    print(f"\nWrote {CARD}")


# ── main ──────────────────────────────────────────────────────────────────
def run(fast: bool = False, no_network: bool = False) -> None:
    report = H.run_checks()
    if not report.get("ok", True):
        sys.exit("Health check hard-failed (future-dated finished rows or "
                 "duplicate fixture_ids present) — run `python3 -m club_soccer.fetch "
                 "--repair` before continuing.")

    api_key = get_key("bsd", env="BSD_API_KEY")
    if no_network or not api_key:
        reason = "--no-network" if no_network else "no BSD key"
        print(f"\n== Network steps skipped ({reason}) ==")
    else:
        run_network_steps(api_key)

    if not fast:
        _step("Refit model", lambda: M.save_params(M.fit()))
    else:
        print("\n== Refit skipped (--fast) ==")

    player_adj_map = None
    from .calibrate import load_active_maps
    calib_maps = load_active_maps()
    if not no_network and api_key:
        player_adj_map = _step("Player availability adjustments", E.fetch_player_adjustments, api_key)

    edge_rows: list[dict] = []
    if not no_network and api_key:
        odds = _step("Fetch BSD odds", E.fetch_bsd_odds, api_key)
        if odds is not None:
            history_days = MM.history_age_days()
            edge_rows = _step("Price the card", E.rows_from_odds, odds, "ensemble", 100.0,
                             calib_maps, player_adj_map, history_days >= MM.WARMUP_DAYS) or []
    if not edge_rows:
        try:
            odds = E.load_odds()
            edge_rows = E.rows_from_odds(odds, "ensemble", 100.0,
                                         calib_maps=calib_maps)
            print("\n== Priced from manual odds.csv (no live BSD odds available) ==")
        except FileNotFoundError:
            print("\n== No odds available (live or manual) — card has no bets ==")

    write_card(edge_rows, player_adj_map, calib_maps)


def main() -> None:
    ap = argparse.ArgumentParser(description="Club Soccer daily front door")
    ap.add_argument("--fast", action="store_true", help="skip the model refit")
    ap.add_argument("--no-network", action="store_true",
                    help="skip every network step; build the card from cached data only")
    args = ap.parse_args()
    run(fast=args.fast, no_network=args.no_network)


if __name__ == "__main__":
    main()

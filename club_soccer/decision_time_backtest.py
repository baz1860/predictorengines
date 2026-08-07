#!/usr/bin/env python3
"""Decision-time backtest — the evidence the staking gate is waiting for.

Why this exists
---------------
The retired closing-price backtest selected and executed at a price unavailable
until kick-off. Its evidence could never justify a live staking decision.

This backtest replays only what a bettor could actually have done:

  1. at a DECISION TIME >= 60 min before kick-off,
  2. price the match with the model trained ONLY on prior results,
  3. bet at the quote that existed at that decision time,
  4. settle on the result,
  5. score CLV against the de-vigged Pinnacle close.

It produces the `decision_time_v2` artifact the gate validates, and — per the
"regular winners" goal — reports BOTH:

  * confidence table: hit-rate and yield by model-probability bucket, so you can
    read how often the picks come in;
  * value subset: the same, restricted to picks the model also thinks are
    underpriced.

The critical honesty this buys: a confident-but-short pick shows a high hit-rate
AND a negative CLV/yield in the same table — winning often is not the same as
making money, and this is where that becomes visible.

The hard constraint
--------------------
A decision-time quote cannot be reconstructed after the fact; it had to be
recorded before kick-off (odds_history_club.csv). So this backtest can only ever
cover fixtures that were snapshotted while upcoming, and it ACCUMULATES forward.
Today that is a handful of settled fixtures — far below the gate's 1,000-bet
bar — so the gate stays correctly closed. The engine is complete; the evidence
is not, and only calendar time fixes that.

CLI:
  python3 -m club_soccer.decision_time_backtest            # run + write artifact
  python3 -m club_soccer.decision_time_backtest --report   # human summary
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RUNTIME = Path(os.environ.get("CLUB_SOCCER_RUNTIME_DIR", str(DATA)))
SNAPSHOTS = DATA / "odds_history_club.csv"
MARKET_HISTORY = DATA / "market_history.csv"
FIXTURES = DATA / "fixtures.csv"
ARTIFACT = RUNTIME / "backtest_market.json"    # the file the gate reads
LEDGER = RUNTIME / "decision_time_ledger.csv"  # settled-bet replay snapshot

MIN_LEAD_MIN = 60                # gate floor: decision >= 60 min pre-kickoff
MAX_LEAD_MIN = 7 * 24 * 60       # gate ceiling: within a week
THRESHOLDS = {"2%": 0.02, "4%": 0.04, "6%": 0.06}
# Confidence buckets for the hit-rate table (the "regular winners" view).
CONF_BUCKETS = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 1.01)]
KELLY_FRACTION = 0.25
MAX_REUSE_DAYS = 6


# ── frozen-ledger replay ─────────────────────────────────────────────────


def build_bets(verbose: bool = False) -> pd.DataFrame:
    """Settled bets from the FROZEN decision ledger — no reconstruction.

    Every field (identity, executable price, model probability) was frozen at
    decision time; settlement/CLV was appended after the result. Deleting the
    alias map or refitting the model cannot change a single row here, which is
    the property the reconstruction backtest could not offer. Returns the same
    shape the metrics functions expect.
    """
    from . import decision_ledger as DL
    rows = DL.settled_bets()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # One strategy cohort per artifact. The fitted-parameter hash remains
    # provenance but is NOT a cohort boundary: routine refits are the intended
    # operation of the same model and must not reset a 1,000-bet evidence clock.
    # Resolver/code changes do alter the strategy and start a fresh cohort.
    cohort_cols = [c for c in ("resolver_version", "code_hash")
                   if c in df.columns]
    if cohort_cols and "decision_ts" in df.columns and len(df):
        latest = df.sort_values("decision_ts").iloc[-1]
        mask = pd.Series(True, index=df.index)
        for c in cohort_cols:
            mask &= (df[c] == latest[c])
        df = df[mask]
    if "strategy_eligible" in df.columns:
        eligible = df["strategy_eligible"].astype(str).str.strip().str.lower()
        df = df[eligible.isin({"", "1", "true", "yes"})]
    df = df.rename(columns={"odds_executed": "odds", "p_book_devig": "p_book",
                            "kickoff_utc": "kickoff"})
    df["date"] = df["kickoff"].astype(str).str[:10]
    df["key"] = df["provider_fixture_id"].astype(str)
    df["lead_min"] = df["decision_lead_min"]
    keep = ["key", "date", "kickoff", "competition", "market", "side", "odds",
            "p_model", "p_book", "p_close", "edge", "won", "lead_min", "clv",
            "lineup_confidence", "resolver_version", "model_hash", "code_hash"]
    return df[[c for c in keep if c in df.columns]]


# ── metrics ───────────────────────────────────────────────────────────────

def _kelly(p: float, o: float) -> float:
    b = o - 1.0
    return max(0.0, (p * b - (1 - p)) / b) if b > 0 else 0.0


_ZERO_ROW = {"n_bets": 0, "n_clv": 0, "flat_roi": None, "kelly_roi": None,
             "clv_mean": None, "clv_frac_positive": None}


def _rows_for_market(mbets: pd.DataFrame) -> dict:
    """Per-edge-threshold metric rows for one market's bets.

    Emits `n_clv` (the count of bets that could actually be CLV-scored)
    alongside `n_bets` — the two diverge whenever a closing feed is missing, and
    conflating them lets a handful of CLV samples masquerade as full-sample
    confidence in the gate's Wilson bound.
    """
    rows: dict = {}
    for label, thr in THRESHOLDS.items():
        sel = mbets[mbets["edge"] >= thr]
        n = len(sel)
        if n == 0:
            rows[label] = dict(_ZERO_ROW)
            continue
        profit = np.where(sel["won"] == 1, sel["odds"] - 1.0, -1.0)
        flat_roi = float(profit.mean())
        confidence = (
            pd.to_numeric(sel["lineup_confidence"], errors="coerce").fillna(1.0)
            if "lineup_confidence" in sel.columns
            else pd.Series(1.0, index=sel.index)
        ).clip(0.0, 1.0).to_numpy()
        kstakes = np.array([_kelly(p, o) * KELLY_FRACTION
                            for p, o in zip(sel["p_model"], sel["odds"])]) * confidence
        kprofit = np.where(sel["won"] == 1, kstakes * (sel["odds"] - 1.0), -kstakes)
        kelly_roi = float(kprofit.sum() / kstakes.sum()) if kstakes.sum() > 0 else 0.0
        clv = sel["clv"].dropna() if "clv" in sel.columns else pd.Series([], dtype=float)
        flat_lb = _block_bootstrap_lb(profit, sel["date"])
        kelly_lb = _block_bootstrap_ratio_lb(kprofit, kstakes, sel["date"])
        rows[label] = {
            "n_bets": int(n),
            "n_clv": int(len(clv)),
            "flat_roi": round(flat_roi, 5),
            "kelly_roi": round(kelly_roi, 5),
            "flat_roi_lb95": round(flat_lb, 5) if flat_lb is not None else None,
            "kelly_roi_lb95": round(kelly_lb, 5) if kelly_lb is not None else None,
            "clv_mean": round(float(clv.mean()), 5) if len(clv) else None,
            "clv_frac_positive": round(float((clv > 0).mean()), 5) if len(clv) else None,
        }
    return rows


def _threshold_metrics(bets: pd.DataFrame) -> dict:
    """Gate-format metrics per market per edge threshold (pooled across leagues)."""
    key_map = {"1x2": "1x2", "total25": "total_over_under_2_5"}
    if bets.empty or "edge" not in bets.columns:
        return {gk: {lbl: dict(_ZERO_ROW) for lbl in THRESHOLDS}
                for gk in key_map.values()}
    return {gkey: _rows_for_market(bets[bets["market"] == market])
            for market, gkey in key_map.items()}


def _threshold_metrics_by_league(bets: pd.DataFrame) -> dict:
    """Same gate-format metrics, broken out per competition.

    The pooled market metrics can pass on European CLV evidence while silently
    covering leagues that have NO closing feed (non-UEFA totals). The gate reads
    this per-league view so a market-level pass can only ever stake a league
    that independently clears the bar — an EU-derived totals pass can never
    unlock Brazil/Mexico/Japan totals.
    """
    key_map = {"1x2": "1x2", "total25": "total_over_under_2_5"}
    if (bets.empty or "edge" not in bets.columns
            or "competition" not in bets.columns):
        return {}
    out: dict = {}
    for market, gkey in key_map.items():
        mbets = bets[bets["market"] == market]
        if mbets.empty:
            continue
        leagues = {str(comp): _rows_for_market(cbets)
                   for comp, cbets in mbets.groupby("competition")}
        if leagues:
            out[gkey] = leagues
    return out


def _confidence_table(bets: pd.DataFrame, value_only: bool = False) -> list[dict]:
    """Hit-rate and yield by model-probability bucket — the regular-winners view."""
    if bets.empty or "edge" not in bets.columns:
        return [{"bucket": f"{lo:.0%}-{hi:.0%}", "n": 0} for lo, hi in CONF_BUCKETS]
    sel = bets[bets["edge"] > 0] if value_only else bets
    rows = []
    for lo, hi in CONF_BUCKETS:
        b = sel[(sel["p_model"] >= lo) & (sel["p_model"] < hi)]
        n = len(b)
        if n == 0:
            rows.append({"bucket": f"{lo:.0%}-{hi:.0%}", "n": 0})
            continue
        profit = np.where(b["won"] == 1, b["odds"] - 1.0, -1.0)
        rows.append({
            "bucket": f"{lo:.0%}-{hi:.0%}", "n": int(n),
            "hit_rate": round(float(b["won"].mean()), 4),
            "mean_odds": round(float(b["odds"].mean()), 3),
            "yield": round(float(profit.mean()), 4),
            "mean_model_p": round(float(b["p_model"].mean()), 4),
        })
    return rows


def _block_bootstrap_lb(profit: np.ndarray, dates: pd.Series,
                        n_boot: int = 2000, z: float = 0.05) -> float | None:
    """One-sided 95% lower bound on mean profit, resampling by calendar week so
    correlated same-round bets don't inflate confidence. Preregistered in the
    gate; with a tiny sample it will simply be very negative, which is correct.

    Returns None (not NaN) when there are too few bets to bootstrap, so the
    artifact stays valid JSON — a NaN written by json.dumps is non-standard and
    a NaN metric compares False against every gate guard."""
    if len(profit) < 2:
        return None
    weeks = pd.to_datetime(dates).dt.isocalendar().week.to_numpy()
    blocks = [profit[weeks == w] for w in np.unique(weeks)]
    rng = np.random.default_rng(0)
    means = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(blocks), len(blocks))
        sample = np.concatenate([blocks[i] for i in pick])
        means.append(sample.mean())
    return float(np.quantile(means, z))


def _block_bootstrap_ratio_lb(numer: np.ndarray, denom: np.ndarray,
                              dates: pd.Series, n_boot: int = 2000,
                              z: float = 0.05) -> float | None:
    """One-sided 95% lower bound on sum(numer)/sum(denom), block-resampled by
    calendar week. This is the correct estimator for a stake-weighted (Kelly)
    ROI, where the denominator varies per bet — a plain mean would mis-weight
    it. Returns None when it cannot be computed (too few bets, no stake)."""
    numer = np.asarray(numer, dtype=float)
    denom = np.asarray(denom, dtype=float)
    if len(numer) < 2 or denom.sum() <= 0:
        return None
    weeks = pd.to_datetime(dates).dt.isocalendar().week.to_numpy()
    blocks = [(numer[weeks == w], denom[weeks == w]) for w in np.unique(weeks)]
    rng = np.random.default_rng(0)
    ratios = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(blocks), len(blocks))
        num = np.concatenate([blocks[i][0] for i in pick])
        den = np.concatenate([blocks[i][1] for i in pick])
        if den.sum() > 0:
            ratios.append(num.sum() / den.sum())
    return float(np.quantile(ratios, z)) if ratios else None


# ── artifact ──────────────────────────────────────────────────────────────

def _cohort(bets: pd.DataFrame) -> dict:
    """The single (resolver, model, code) cohort the pooled bets belong to —
    recorded so a reader can verify no cross-regime pooling happened."""
    out: dict = {}
    for col in ("resolver_version", "model_hash", "code_hash"):
        if col in bets.columns and not bets.empty:
            vals = sorted({str(v) for v in bets[col].dropna().unique()})
            out[col] = vals[0] if len(vals) == 1 else vals
    return out


def _provenance(bets: pd.DataFrame) -> dict:
    from . import decision_ledger as DL

    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else ""
    return {
        "fixtures_sha256": _sha(FIXTURES),
        "snapshots_sha256": _sha(SNAPSHOTS),
        "model_params_sha256": _sha(DATA / "model_params.json"),
        # Hash the EXACT ledgers the evidence was built from, so a later append
        # or edit is detectable — the "append-only" claim is now checkable.
        "decision_ledger_sha256": _sha(DL.DECISIONS),
        "settlement_ledger_sha256": _sha(DL.SETTLEMENTS),
        "version_cohort": _cohort(bets),
        "n_settled_fixtures": int(bets["key"].nunique()) if not bets.empty else 0,
        "n_candidate_bets": int(len(bets)),
    }


def _input_fingerprint() -> str:
    """Hash the frozen ledgers and code that determines the evidence artifact."""
    from . import decision_ledger as DL

    digest = hashlib.sha256()
    for path in (
        DL.DECISIONS, DL.SETTLEMENTS, Path(__file__),
        HERE / "decision_ledger.py", HERE / "market_settlement.py",
    ):
        digest.update(path.read_bytes() if path.exists() else b"")
    return digest.hexdigest()[:32]


def _reusable_artifact(fingerprint: str) -> dict | None:
    if not ARTIFACT.exists():
        return None
    try:
        artifact = json.loads(ARTIFACT.read_text())
        generated = datetime.fromisoformat(
            str(artifact["generated_at_utc"]).replace("Z", "+00:00")
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
    age_days = (datetime.now(timezone.utc) - generated).total_seconds() / 86400
    if (
        artifact.get("input_fingerprint") == fingerprint
        and 0 <= age_days < MAX_REUSE_DAYS
    ):
        return artifact
    return None


def run(verbose: bool = True, force: bool = False) -> dict:
    fingerprint = _input_fingerprint()
    if not force:
        cached = _reusable_artifact(fingerprint)
        if cached is not None:
            if verbose:
                print("Decision-time backtest unchanged — reused cached artifact")
            return cached

    bets = build_bets(verbose=verbose)

    if not bets.empty and (LEDGER.exists() or True):
        bets.to_csv(LEDGER, index=False)     # snapshot of the current replay

    lead = float(bets["lead_min"].median()) if not bets.empty else MIN_LEAD_MIN
    lead = min(max(lead, MIN_LEAD_MIN), MAX_LEAD_MIN)

    ll_model, ll_market = _log_losses(bets)
    artifact = {
        "backtest_version": "decision_time_v2",
        "selection_method": "latest_quote_at_or_before_decision_time",
        "execution_method": "same_decision_time_quote",
        "clv_reference": "captured_closing_devigged",
        "decision_lead_minutes": round(lead, 1),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_fingerprint": fingerprint,
        "simulated_betting": _threshold_metrics(bets),
        "simulated_betting_by_league": _threshold_metrics_by_league(bets),
        "model_log_loss_1x2": ll_model,
        "market_log_loss_1x2_devigged_pinnacle_closing": ll_market,
        # Extra (ignored by the gate) — the regular-winners view.
        "confidence_strategy": _confidence_table(bets, value_only=False),
        "value_subset": _confidence_table(bets, value_only=True),
        "provenance": _provenance(bets),
        "note": ("Accumulates forward — a decision-time quote cannot be "
                 "reconstructed retroactively. The evidence gate is per-market "
                 "and, where available, per-league: 1X2 can open independently "
                 "of OU2.5. Markets or leagues without a closing reference stay "
                 "closed; the current shared blocker is settled decision volume."),
    }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False: a NaN would be written as the non-standard token `NaN`,
    # producing an artifact that is not valid JSON and a metric that slips
    # through every finite-comparison guard. Metrics are None where undefined.
    ARTIFACT.write_text(json.dumps(artifact, indent=2, allow_nan=False))
    if verbose:
        _print(artifact)
    return artifact


def _log_losses(bets: pd.DataFrame) -> tuple[float | None, float | None]:
    """1x2 model vs de-vigged-Pinnacle-close log-loss, on EXACTLY the fixtures
    where both exist. One row per fixture (not per side).

    The closing probability is the `p_close` frozen into each settled bet at
    settlement time — NOT a value re-derived here from a match key. The previous
    version looked the closing map up by `key` (a provider fixture id) against a
    map keyed by `date|home|away`, so the join never hit and market log-loss was
    always None, which kept 1X2 permanently un-openable. Requiring `p_close` for
    inclusion also guarantees model and market are scored on the identical set
    of fixtures.
    """
    if bets.empty or "market" not in bets.columns:
        return None, None
    m = bets[bets["market"] == "1x2"]
    if m.empty or "p_close" not in m.columns:
        return None, None
    model_ll, market_ll = [], []
    for _key_id, grp in m.groupby("key"):
        won_side = grp[grp["won"] == 1]
        if won_side.empty:
            continue
        s = won_side["side"].iloc[0]
        pc = won_side["p_close"].iloc[0]
        if pc is None or (isinstance(pc, float) and math.isnan(pc)) or float(pc) <= 0:
            continue                       # no closing ref → drop from BOTH sums
        pm = float(grp[grp["side"] == s]["p_model"].iloc[0])
        model_ll.append(-math.log(max(1e-12, pm)))
        market_ll.append(-math.log(max(1e-12, float(pc))))
    ml = round(float(np.mean(model_ll)), 5) if model_ll else None
    mk = round(float(np.mean(market_ll)), 5) if market_ll else None
    return ml, mk


def _print(a: dict) -> None:
    prov = a["provenance"]
    print(f"Decision-time backtest ({a['decision_lead_minutes']:.0f} min lead)")
    print(f"  settled fixtures : {prov['n_settled_fixtures']}")
    print(f"  candidate bets   : {prov['n_candidate_bets']}")
    print()
    print("  Confidence strategy (all full picks) — hit-rate by model probability:")
    _print_conf(a["confidence_strategy"])
    print("\n  Value subset (edge > 0):")
    _print_conf(a["value_subset"])
    print("\n  Gate view (edge thresholds):")
    for mkt, thr in a["simulated_betting"].items():
        for label, row in thr.items():
            if row["n_bets"]:
                print(f"    {mkt:<22} @{label}: n={row['n_bets']} "
                      f"flat_roi={row['flat_roi']} clv={row['clv_mean']}")
    from .evidence_gate import staking_allowed
    ok, reasons = staking_allowed()
    print(f"\n  staking gate: {'OPEN' if ok else 'CLOSED'}")
    for r in reasons[:4]:
        print(f"    - {r[:88]}")


def _print_conf(table: list[dict]) -> None:
    print(f"    {'bucket':<12}{'n':>5}{'hit':>8}{'odds':>7}{'yield':>8}")
    for r in table:
        if r.get("n"):
            print(f"    {r['bucket']:<12}{r['n']:>5}{r['hit_rate']:>8.0%}"
                  f"{r['mean_odds']:>7.2f}{r['yield']:>+8.1%}")
        else:
            print(f"    {r['bucket']:<12}{'0':>5}  (no bets)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", action="store_true",
                    help="print the last artifact without recomputing")
    ap.add_argument("--force", action="store_true",
                    help="recompute even when ledgers and backtest code are unchanged")
    args = ap.parse_args()
    if args.report and ARTIFACT.exists():
        _print(json.loads(ARTIFACT.read_text()))
        return
    run(force=args.force)


if __name__ == "__main__":
    main()

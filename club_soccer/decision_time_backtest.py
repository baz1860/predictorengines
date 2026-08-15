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
  5. score fair-probability CLV from complete raw closing markets using a power
     de-vig, and raw price CLV against the same executing book where available.

It produces the `decision_time_v3` artifact the gate validates, and — per the
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
Until enough compatible, independently blocked evidence accumulates, the gate
stays correctly closed. Explicit strategy versions preserve compatible history
across harmless code and identity-map maintenance.

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
N_BOOTSTRAP = 10_000
MIN_INDEPENDENT_BLOCKS = 8
# Six inspected market/threshold rows share one family-wise error budget.
BOOTSTRAP_ALPHA = 0.05 / 6.0


# ── frozen-ledger replay ─────────────────────────────────────────────────


def _settled_frame() -> pd.DataFrame:
    """All frozen settled rows with explicit compatibility metadata.

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
    df = df.rename(columns={"odds_executed": "odds", "p_book_devig": "p_book",
                            "kickoff_utc": "kickoff"})
    df["date"] = df["kickoff"].astype(str).str[:10]
    df["key"] = df["provider_fixture_id"].astype(str)
    df["lead_min"] = df["decision_lead_min"]
    return df


def build_bets(verbose: bool = False,
               strategy_version: str | None = None) -> pd.DataFrame:
    """Gate-eligible rows from one explicit strategy version.

    Identity and code hashes remain provenance.  They are deliberately not
    global cohort boundaries: frozen decisions do not change when an alias or
    a comment changes.  Specific bad identities are excluded through the
    identity-exclusion sidecar.
    """
    df = _settled_frame()
    if df.empty:
        return df
    if strategy_version is None:
        latest = df.sort_values("decision_ts").iloc[-1]
        strategy_version = str(latest.get("strategy_version") or "")
    if "strategy_version" in df.columns:
        df = df[df["strategy_version"].astype(str) == strategy_version]
    if "strategy_eligible" in df.columns:
        eligible = df["strategy_eligible"].astype(str).str.strip().str.lower()
        df = df[eligible.isin({"", "1", "true", "yes"})]
    if "identity_excluded" in df.columns:
        df = df[~df["identity_excluded"].fillna(False).astype(bool)]
    keep = ["key", "date", "kickoff", "competition", "market", "side", "odds",
            "p_model", "p_book", "p_close", "edge", "won", "lead_min", "clv",
            "raw_price_clv", "legacy_clv", "legacy_p_close", "clv_method",
            "clv_schema_version", "close_source", "close_odds",
            "lineup_confidence", "resolver_version", "model_hash", "code_hash",
            "strategy_version", "strategy_manifest_hash"]
    return df[[c for c in keep if c in df.columns]]


def _diagnostic_views(current: pd.DataFrame) -> dict:
    """Transparent current/all-history views; only current feeds the gate."""
    all_rows = _settled_frame()
    if all_rows.empty:
        return {"current_compatible": {"n_rows": 0}, "all_history": {"n_rows": 0},
                "exclusions": {}}
    eligible_text = all_rows.get("strategy_eligible", pd.Series("", index=all_rows.index))
    strategy_ok = eligible_text.astype(str).str.strip().str.lower().isin(
        {"", "1", "true", "yes"}
    )
    identity_bad = all_rows.get(
        "identity_excluded", pd.Series(False, index=all_rows.index)
    ).fillna(False).astype(bool)
    diagnostic = all_rows[strategy_ok & ~identity_bad].copy()
    # Legacy CLV is retained for labeled diagnostics only; it can never feed
    # the v3 gate, which consumes ``current`` and raw-market-v2 CLV exclusively.
    if "legacy_clv" in diagnostic.columns:
        diagnostic["clv"] = diagnostic["legacy_clv"]
    return {
        "current_compatible": {
            "strategy_version": (_cohort(current).get("strategy_version")
                                 if not current.empty else None),
            "n_rows": int(len(current)),
            "n_fixtures": int(current["key"].nunique()) if not current.empty else 0,
            "metrics": _threshold_metrics(current),
        },
        "all_history": {
            "label": "diagnostic_only_mixed_strategies_legacy_clv",
            "n_rows": int(len(diagnostic)),
            "n_fixtures": int(diagnostic["key"].nunique()) if not diagnostic.empty else 0,
            "metrics": _threshold_metrics(diagnostic),
        },
        "exclusions": {
            "strategy_ineligible": int((~strategy_ok).sum()),
            "identity_excluded": int(identity_bad.sum()),
            "incompatible_strategy": int(
                len(diagnostic) - len(current)
            ),
        },
        "cohorts": _cohort_table(all_rows),
    }


def _cohort_table(frame: pd.DataFrame) -> list[dict]:
    if frame.empty:
        return []
    cols = [c for c in ("strategy_version", "resolver_version", "code_hash")
            if c in frame.columns]
    out = []
    for values, group in frame.groupby(cols, dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        row = dict(zip(cols, (str(v) for v in values)))
        row.update({
            "start": str(group["decision_ts"].min()),
            "end": str(group["decision_ts"].max()),
            "n_rows": int(len(group)),
            "n_fixtures": int(group["provider_fixture_id"].nunique()),
            "n_clv_v2": int(group["clv"].notna().sum()) if "clv" in group else 0,
            "n_legacy_clv": int(group["legacy_clv"].notna().sum())
            if "legacy_clv" in group else 0,
        })
        out.append(row)
    return sorted(out, key=lambda row: row["start"])


# ── metrics ───────────────────────────────────────────────────────────────

def _kelly(p: float, o: float) -> float:
    b = o - 1.0
    return max(0.0, (p * b - (1 - p)) / b) if b > 0 else 0.0


_ZERO_ROW = {
    "n_bets": 0, "n_clv": 0, "n_raw_price_clv": 0,
    "n_independent_blocks": 0, "flat_roi": None, "kelly_roi": None,
    "flat_roi_lb95_simultaneous": None, "kelly_roi_lb95_simultaneous": None,
    "clv_mean": None, "clv_lb95_simultaneous": None,
    "clv_frac_positive": None, "raw_price_clv_mean": None,
}


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
        raw_clv = (sel["raw_price_clv"].dropna()
                   if "raw_price_clv" in sel.columns else pd.Series([], dtype=float))
        n_blocks = _independent_block_count(sel["date"])
        flat_lb = _block_bootstrap_lb(profit, sel["date"])
        kelly_lb = _block_bootstrap_ratio_lb(kprofit, kstakes, sel["date"])
        clv_lb = (_block_bootstrap_lb(
            clv.to_numpy(float), sel.loc[clv.index, "date"]
        ) if len(clv) else None)
        rows[label] = {
            "n_bets": int(n),
            "n_clv": int(len(clv)),
            "n_raw_price_clv": int(len(raw_clv)),
            "n_independent_blocks": n_blocks,
            "flat_roi": round(flat_roi, 5),
            "kelly_roi": round(kelly_roi, 5),
            "flat_roi_lb95_simultaneous": (
                round(flat_lb, 5) if flat_lb is not None else None
            ),
            "kelly_roi_lb95_simultaneous": (
                round(kelly_lb, 5) if kelly_lb is not None else None
            ),
            # Transitional aliases remain for human consumers; v3 gate reads
            # only the explicitly simultaneous fields above.
            "flat_roi_lb95": round(flat_lb, 5) if flat_lb is not None else None,
            "kelly_roi_lb95": round(kelly_lb, 5) if kelly_lb is not None else None,
            "clv_mean": round(float(clv.mean()), 5) if len(clv) else None,
            "clv_lb95_simultaneous": (
                round(clv_lb, 5) if clv_lb is not None else None
            ),
            "clv_frac_positive": round(float((clv > 0).mean()), 5) if len(clv) else None,
            "raw_price_clv_mean": (
                round(float(raw_clv.mean()), 5) if len(raw_clv) else None
            ),
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


def _week_keys(dates: pd.Series) -> np.ndarray:
    iso = pd.to_datetime(dates).dt.isocalendar()
    return (iso["year"].astype(str) + "-" + iso["week"].astype(str)).to_numpy()


def _independent_block_count(dates: pd.Series) -> int:
    return int(len(np.unique(_week_keys(dates)))) if len(dates) else 0


def _block_bootstrap_lb(profit: np.ndarray, dates: pd.Series,
                        n_boot: int = N_BOOTSTRAP,
                        z: float = BOOTSTRAP_ALPHA) -> float | None:
    """One-sided 95% lower bound on mean profit, resampling by calendar week so
    correlated same-round bets don't inflate confidence. Preregistered in the
    gate; with a tiny sample it will simply be very negative, which is correct.

    Returns None (not NaN) when there are too few bets to bootstrap, so the
    artifact stays valid JSON — a NaN written by json.dumps is non-standard and
    a NaN metric compares False against every gate guard."""
    if len(profit) < 2:
        return None
    weeks = _week_keys(dates)
    blocks = [profit[weeks == w] for w in np.unique(weeks)]
    if len(blocks) < MIN_INDEPENDENT_BLOCKS:
        return None
    rng = np.random.default_rng(0)
    means = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(blocks), len(blocks))
        sample = np.concatenate([blocks[i] for i in pick])
        means.append(sample.mean())
    return float(np.quantile(means, z))


def _block_bootstrap_ratio_lb(numer: np.ndarray, denom: np.ndarray,
                              dates: pd.Series, n_boot: int = N_BOOTSTRAP,
                              z: float = BOOTSTRAP_ALPHA) -> float | None:
    """One-sided 95% lower bound on sum(numer)/sum(denom), block-resampled by
    calendar week. This is the correct estimator for a stake-weighted (Kelly)
    ROI, where the denominator varies per bet — a plain mean would mis-weight
    it. Returns None when it cannot be computed (too few bets, no stake)."""
    numer = np.asarray(numer, dtype=float)
    denom = np.asarray(denom, dtype=float)
    if len(numer) < 2 or denom.sum() <= 0:
        return None
    weeks = _week_keys(dates)
    blocks = [(numer[weeks == w], denom[weeks == w]) for w in np.unique(weeks)]
    if len(blocks) < MIN_INDEPENDENT_BLOCKS:
        return None
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
    """Compatibility and provenance values represented by the current view."""
    out: dict = {}
    for col in ("strategy_version", "strategy_manifest_hash", "resolver_version",
                "model_hash", "code_hash"):
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
        "decision_strategy_ledger_sha256": _sha(DL.DECISION_STRATEGIES),
        "identity_exclusions_sha256": _sha(DL.IDENTITY_EXCLUSIONS),
        "raw_closing_ledger_sha256": _sha(DL.RAW_CLOSING),
        "settlement_clv_v2_sha256": _sha(DL.SETTLEMENT_CLV_V2),
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
        DL.DECISION_STRATEGIES, DL.IDENTITY_EXCLUSIONS, DL.RAW_CLOSING,
        DL.SETTLEMENT_CLV_V2, HERE / "decision_ledger.py",
        HERE / "market_settlement.py", HERE / "strategy_contract.py",
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
        ledger_tmp = LEDGER.with_name(f"{LEDGER.name}.{os.getpid()}.tmp")
        bets.to_csv(ledger_tmp, index=False)
        ledger_tmp.replace(LEDGER)           # atomic current-replay snapshot

    lead = float(bets["lead_min"].median()) if not bets.empty else MIN_LEAD_MIN
    lead = min(max(lead, MIN_LEAD_MIN), MAX_LEAD_MIN)

    ll_report = _log_loss_report(bets)
    ll_model = ll_report["model_log_loss"]
    ll_market = ll_report["market_log_loss"]
    from . import decision_ledger as DL
    artifact = {
        "backtest_version": "decision_time_v3",
        "selection_method": "first_complete_market_quote_within_decision_window",
        "execution_method": "best_executable_complete_market_quote_at_decision_time",
        "clv_reference": "raw_complete_closing_market",
        "clv_method": DL.CLV_DEVIG_METHOD,
        "clv_schema_version": DL.CLV_SCHEMA_VERSION,
        "decision_lead_minutes": round(lead, 1),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_fingerprint": fingerprint,
        "simulated_betting": _threshold_metrics(bets),
        "simulated_betting_by_league": _threshold_metrics_by_league(bets),
        "model_log_loss_1x2": ll_model,
        "market_log_loss_1x2_devigged_closing": ll_market,
        "market_comparison_1x2": ll_report,
        # Extra (ignored by the gate) — the regular-winners view.
        "confidence_strategy": _confidence_table(bets, value_only=False),
        "value_subset": _confidence_table(bets, value_only=True),
        "provenance": _provenance(bets),
        "diagnostic_views": _diagnostic_views(bets),
        "uncertainty_contract": {
            "bootstrap_resamples": N_BOOTSTRAP,
            "block": "ISO_year_week",
            "minimum_independent_blocks": MIN_INDEPENDENT_BLOCKS,
            "familywise_alpha": 0.05,
            "threshold_alpha": BOOTSTRAP_ALPHA,
        },
        "note": ("Accumulates forward — a decision-time quote cannot be "
                 "reconstructed retroactively. The evidence gate is per-market "
                 "and, where available, per-league: 1X2 can open independently "
                 "of OU2.5. Markets or leagues without a closing reference stay "
                 "closed. Legacy proportional CLV remains diagnostic-only; "
                 "the gate requires raw-market-v2 power-consensus evidence."),
    }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False: a NaN would be written as the non-standard token `NaN`,
    # producing an artifact that is not valid JSON and a metric that slips
    # through every finite-comparison guard. Metrics are None where undefined.
    artifact_tmp = ARTIFACT.with_name(f"{ARTIFACT.name}.{os.getpid()}.tmp")
    artifact_tmp.write_text(json.dumps(artifact, indent=2, allow_nan=False))
    artifact_tmp.replace(ARTIFACT)
    if verbose:
        _print(artifact)
    return artifact


def _log_losses(bets: pd.DataFrame) -> tuple[float | None, float | None]:
    report = _log_loss_report(bets)
    return report["model_log_loss"], report["market_log_loss"]


def _block_bootstrap_ub(values: np.ndarray, dates: pd.Series,
                        n_boot: int = N_BOOTSTRAP,
                        alpha: float = 0.05) -> float | None:
    lower = _block_bootstrap_lb(-np.asarray(values, dtype=float), dates,
                                n_boot=n_boot, z=alpha)
    return -lower if lower is not None else None


def _log_loss_report(bets: pd.DataFrame) -> dict:
    """1x2 model vs power-de-vigged raw-market close, on EXACTLY the fixtures
    where both exist. One row per fixture (not per side).

    The closing probability is the `p_close` frozen into each settled bet at
    settlement time — NOT a value re-derived here from a match key. The previous
    version looked the closing map up by `key` (a provider fixture id) against a
    map keyed by `date|home|away`, so the join never hit and market log-loss was
    always None, which kept 1X2 permanently un-openable. Requiring `p_close` for
    inclusion also guarantees model and market are scored on the identical set
    of fixtures.
    """
    empty = {"n_fixtures": 0, "n_independent_blocks": 0,
             "model_log_loss": None, "market_log_loss": None,
             "paired_delta_model_minus_market": None,
             "paired_delta_ub95": None}
    if bets.empty or "market" not in bets.columns:
        return empty
    m = bets[bets["market"] == "1x2"]
    if m.empty or "p_close" not in m.columns:
        return empty
    model_ll, market_ll, dates = [], [], []
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
        dates.append(str(grp["date"].iloc[0]) if "date" in grp.columns
                     else "1970-01-01")
    if not model_ll:
        return empty
    delta = np.asarray(model_ll) - np.asarray(market_ll)
    date_series = pd.Series(dates)
    ub = _block_bootstrap_ub(delta, date_series)
    return {
        "n_fixtures": int(len(delta)),
        "n_independent_blocks": _independent_block_count(date_series),
        "model_log_loss": round(float(np.mean(model_ll)), 5),
        "market_log_loss": round(float(np.mean(market_ll)), 5),
        "paired_delta_model_minus_market": round(float(delta.mean()), 5),
        "paired_delta_ub95": round(float(ub), 5) if ub is not None else None,
    }


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

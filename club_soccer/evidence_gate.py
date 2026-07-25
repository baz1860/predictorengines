#!/usr/bin/env python3
"""Evidence gate: staking is OFF until the stored evidence says the model wins.

The market backtest currently shows negative ROI at every edge threshold,
negative mean CLV, and a market log-loss better than the model's — i.e. the
system's own evidence says it has no demonstrated betting edge. Until that
changes, every stake the pricing layer produces is zeroed (edges/EV stay
visible for diagnostics; nothing is presented as backable or recordable).

Preregistered criteria — ALL must pass, per market, before staking enables:

  0. The artifact declares the decision-time methodology (backtest_version
     "decision_time_v2", selection/execution at a pre-kickoff quote, CLV vs
     de-vigged Pinnacle close, >= 60 min decision lead). Evidence from the
     legacy closing-odds simulator can NEVER open the gate.
  1. backtest_market.json carries a generated_at_utc <= MAX_ARTIFACT_AGE_DAYS
     old (signed provenance, not file mtime — evidence must be recent).
  2. n_bets >= MIN_BETS at the lowest edge threshold (sample size).
  3. flat_roi > 0 AND kelly_roi > 0 at every reported threshold
     (an edge that vanishes at realistic thresholds isn't an edge).
  4. clv_mean > 0 AND clv_frac_positive >= MIN_CLV_FRAC_POSITIVE where CLV
     is reported (beating the close is the strongest single predictor of
     long-run profitability).
  5. For 1x2: model log-loss <= market log-loss (the model must out-predict
     the de-vigged close before its prices are trusted with money).

There is NO in-code override. If you believe the gate is wrong, fix the
evidence or change the criteria here — in a commit, visibly, not at runtime.

CLI: python3 -m club_soccer.evidence_gate     # prints the verdict + reasons
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RUNTIME = Path(os.environ.get("CLUB_SOCCER_RUNTIME_DIR", str(DATA)))
BACKTEST_JSON = RUNTIME / "backtest_market.json"

MAX_ARTIFACT_AGE_DAYS = 14.0
MIN_BETS = 1000
MIN_CLV_FRAC_POSITIVE = 0.5
# CLV must actually cover the bets it vouches for. A market with 1000 bets but
# 10 CLV-scored samples has 10 samples of closing evidence, not 1000 — the
# Wilson bound must run on that 10, and the market must carry real coverage.
MIN_CLV_BETS = 200
MIN_CLV_COVERAGE = 0.8
# Per-league activation: a league needs its own settled evidence before a
# market-level pass may stake it, so a pooled (mostly-European) pass can never
# unlock a league that has no closing feed of its own.
MIN_LEAGUE_BETS = 200

# backtest_market.json market keys -> human label
_MARKETS = {"1x2": "1X2", "total_over_under_2_5": "OU2.5"}

# ── methodology requirements ──────────────────────────────────────────────
# The gate must be impossible to open with closing-odds simulation. Only an
# artifact that explicitly declares the frozen decision-time methodology
# counts.
REQUIRED_VERSION = "decision_time_v2"
REQUIRED_METHODOLOGY = {
    "selection_method": "latest_quote_at_or_before_decision_time",
    "execution_method": "same_decision_time_quote",
    "clv_reference": "pinnacle_closing_devigged",
}
MIN_DECISION_LEAD_MINUTES = 60
MAX_DECISION_LEAD_MINUTES = 7 * 24 * 60      # a week; Infinity must not pass
REQUIRED_THRESHOLDS = {"2%", "4%", "6%"}

# GATE-TIGHTENING TODOs (preregistered before any decision_time_v2 artifact
# exists, so the bar cannot be lowered to fit a result):
#  - flat/kelly ROI: one-sided 95% lower bound > 0 from a 10k calendar-week
#    block bootstrap of sum(profit)/sum(stake), simultaneous across every
#    activated market/threshold — not the point estimate used today.
#  - CLV = log(odds_executed * p_pinnacle_close_devigged): block-bootstrap
#    95% lower bound > 0.
#  - positive-CLV fraction: Wilson one-sided 95% lower bound > 0.5
#    (z = 1.644854), not the raw fraction.
#  - model-vs-market log-loss: paired-bootstrap one-sided 95% upper bound of
#    (model - market) < 0.
#  - per-league activation: >= 200 bets in a league before that league can
#    stake; a pooled pass must never unlock an under-sampled league.
#  - provenance: data_sha256 / model_sha256 / code_commit /
#    snapshot_manifest_sha256 + row counts, schema-validated with
#    additionalProperties: false.


def _reject_duplicate_keys(pairs):
    """object_pairs_hook that rejects duplicate keys at any nesting depth.
    Standard json.loads silently keeps the last value for a repeated key, so a
    duplicate nested field could quietly flip the artifact's meaning."""
    obj: dict = {}
    for key, val in pairs:
        if key in obj:
            raise ValueError(f"duplicate key {key!r} in evidence JSON")
        obj[key] = val
    return obj


def _finite(value, lo: float, hi: float) -> bool:
    """True only for a real, finite number inside [lo, hi]. NaN/Infinity/str
    all fail — `nan <= 0` is False, so naive comparisons silently pass
    corrupt metrics."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and lo <= value <= hi


# One-sided 95% z. Wilson lower bound on a proportion, so noise at exactly the
# floor cannot open the gate — the preregistered tightening, now enforced.
_Z95 = 1.6448536269514722


def _wilson_lower_bound(frac, n) -> float:
    """One-sided 95% Wilson lower bound on a proportion. Returns -inf on bad
    input so the caller's `> threshold` test fails closed."""
    if not _finite(frac, 0.0, 1.0) or not _finite(n, 1.0, 1e9):
        return float("-inf")
    n = float(n)
    z2 = _Z95 * _Z95
    denom = 1.0 + z2 / n
    centre = frac + z2 / (2 * n)
    margin = _Z95 * math.sqrt((frac * (1 - frac) + z2 / (4 * n)) / n)
    return (centre - margin) / denom


def _methodology_failures(bt: dict, now: datetime) -> list[str]:
    """Schema/methodology validation. Any failure keeps the gate closed —
    metrics from a wrong-methodology backtest are meaningless."""
    fails: list[str] = []
    if bt.get("backtest_version") != REQUIRED_VERSION:
        fails.append(f"backtest_version is {bt.get('backtest_version')!r}, "
                     f"need {REQUIRED_VERSION!r} (legacy closing-odds evidence "
                     "can never open staking)")
    for field, want in REQUIRED_METHODOLOGY.items():
        if bt.get(field) != want:
            fails.append(f"{field} is {bt.get(field)!r}, need {want!r}")
    lead = bt.get("decision_lead_minutes")
    if not _finite(lead, MIN_DECISION_LEAD_MINUTES, MAX_DECISION_LEAD_MINUTES):
        fails.append(f"decision_lead_minutes is {lead!r}, need finite value in "
                     f"[{MIN_DECISION_LEAD_MINUTES}, {MAX_DECISION_LEAD_MINUTES}]")
    gen = bt.get("generated_at_utc")
    try:
        gen_dt = datetime.fromisoformat(str(gen))
        if gen_dt.tzinfo is None:
            # A naive timestamp is ambiguous provenance — reject rather than
            # silently assume UTC.
            raise ValueError("naive timestamp (no timezone)")
        if gen_dt.utcoffset() != timezone.utc.utcoffset(None):
            # "_utc" means UTC: an artifact stamped +05:30 has questionable
            # provenance even if the instant is technically unambiguous.
            raise ValueError(f"offset {gen_dt.utcoffset()} != UTC; use Z/+00:00")
        age = (now - gen_dt).total_seconds() / 86400.0
        if age < 0:
            fails.append(f"generated_at_utc {gen!r} is in the future")
        elif age > MAX_ARTIFACT_AGE_DAYS:
            fails.append(f"evidence is {age:.1f} days old "
                         f"(limit {MAX_ARTIFACT_AGE_DAYS:g})")
    except (TypeError, ValueError) as exc:
        fails.append(f"generated_at_utc invalid ({gen!r}: {exc}) — must be "
                     "timezone-aware ISO-8601; file mtime is not provenance")
    return fails


def evaluate(now: datetime | None = None) -> dict:
    """Evaluate the gate. Returns {"allowed": bool, "reasons": [str, ...]}.

    "allowed" is True only when every criterion passes for every market
    with evidence; "reasons" lists every failure (empty when allowed)."""
    now = now or datetime.now(timezone.utc)
    reasons: list[str] = []

    if not BACKTEST_JSON.exists():
        return {"allowed": False,
                "reasons": ["no backtest_market.json — no evidence, no stakes"]}
    try:
        # Reject duplicate keys at EVERY depth: plain json.loads is
        # last-value-wins, so a duplicate nested `n_bets` could silently
        # decide the artifact's meaning — unacceptable for signed provenance.
        bt = json.loads(BACKTEST_JSON.read_text(),
                        object_pairs_hook=_reject_duplicate_keys)
    except Exception as exc:
        return {"allowed": False,
                "reasons": [f"backtest_market.json unreadable ({exc})"]}

    # The evaluator contract is a JSON OBJECT. A top-level null/list/string/
    # bool/number is malformed evidence — reject it as a reason instead of
    # letting bt.get(...) raise AttributeError inside _methodology_failures.
    if not isinstance(bt, dict):
        return {"allowed": False,
                "reasons": [f"backtest_market.json top level is "
                            f"{type(bt).__name__}, not an object"]}

    # Methodology first: metrics from a non-decision-time backtest are
    # meaningless, so fail before looking at any number.
    meth = _methodology_failures(bt, now)
    if meth:
        return {"allowed": False, "reasons": meth}

    sim = bt.get("simulated_betting")
    if not isinstance(sim, dict):
        return {"allowed": False,
                "reasons": [f"simulated_betting is {type(sim).__name__}, "
                            "not an object"]}
    market_pass: dict[str, bool] = {}
    market_active: dict[str, bool] = {}
    for mkey, label in _MARKETS.items():
        m_reasons: list[str] = []
        thresholds = sim.get(mkey) or {}
        if not isinstance(thresholds, dict):
            reasons.append(f"{label}: thresholds are "
                           f"{type(thresholds).__name__}, not an object")
            market_pass[mkey] = False
            market_active[mkey] = False
            continue
        if set(thresholds) != REQUIRED_THRESHOLDS:
            reasons.append(f"{label}: thresholds {sorted(thresholds)} != required "
                           f"{sorted(REQUIRED_THRESHOLDS)}")
            market_pass[mkey] = False
            market_active[mkey] = False
            continue
        # A market is ACTIVE only if it has real bets somewhere. An inactive
        # market (all n_bets == 0) is neither open nor a veto — it just has no
        # evidence yet, so its rows stay unstaked without blocking other
        # markets. This is what stops the CLV-less OU2.5 permanently vetoing a
        # 1X2 book that has earned staking.
        market_active[mkey] = any(
            isinstance(thresholds.get(t), dict)
            and _finite(thresholds[t].get("n_bets"), 1, 1e9)
            for t in thresholds)
        for thr in sorted(thresholds):
            row = thresholds[thr]
            if not isinstance(row, dict):
                # A null/list threshold row must fail as a reason, not crash
                # evaluate() with an AttributeError.
                m_reasons.append(f"{label} @{thr}: threshold row is "
                                 f"{type(row).__name__}, not an object")
                continue
            n = row.get("n_bets")
            # A bet count is a count: 1000.5 bets is corrupt data, not a
            # rounding style.
            if not _finite(n, MIN_BETS, 1e9) or float(n) != int(n):
                m_reasons.append(f"{label} @{thr}: n_bets {n!r} invalid, "
                                 f"non-integer, or < {MIN_BETS} "
                                 "(required at every threshold)")
            flat, kelly = row.get("flat_roi"), row.get("kelly_roi")
            # Every metric must be a finite number inside a sane range —
            # NaN compares False against everything, so an unchecked NaN
            # walks straight through "flat <= 0" style guards.
            if not (_finite(flat, 1e-9, 10.0) and _finite(kelly, 1e-9, 10.0)):
                m_reasons.append(f"{label} @{thr}: ROI not finite-positive "
                                 f"(flat {flat!r}, kelly {kelly!r})")
            # LOWER BOUNDS, not point estimates. A point ROI of 1e-9 at exactly
            # the floor is noise; the gate must require the block-bootstrap
            # lower bound (written by the backtest) to clear zero, or epsilon
            # opens real staking. Absent bound => fail closed.
            flat_lb = row.get("flat_roi_lb95")
            if not _finite(flat_lb, 1e-9, 10.0):
                m_reasons.append(f"{label} @{thr}: flat_roi 95% lower bound "
                                 f"{flat_lb!r} not finite-positive (point ROI "
                                 "alone is insufficient — need the bootstrap LB)")
            # Kelly ROI needs its OWN lower bound, not just a positive point
            # estimate: a Kelly-staked strategy can show kelly_roi ~ 1e-8 on pure
            # noise that the flat lower bound would reject. Absent bound => fail.
            kelly_lb = row.get("kelly_roi_lb95")
            if not _finite(kelly_lb, 1e-9, 10.0):
                m_reasons.append(f"{label} @{thr}: kelly_roi 95% lower bound "
                                 f"{kelly_lb!r} not finite-positive (point Kelly "
                                 "ROI alone lets epsilon noise open staking)")
            clv, frac = row.get("clv_mean"), row.get("clv_frac_positive")
            # n_clv is the count of bets that were ACTUALLY CLV-scored — the true
            # sample behind clv_frac_positive. It is what the Wilson bound and
            # coverage tests must use; n_bets includes bets with no closing feed.
            n_clv = row.get("n_clv")
            if not (_finite(clv, 1e-9, 1.0)
                    and _finite(frac, MIN_CLV_FRAC_POSITIVE, 1.0)):
                m_reasons.append(f"{label} @{thr}: CLV not finite-positive "
                                 f"(mean {clv!r}, frac+ {frac!r}; absent or "
                                 "non-finite CLV can never pass)")
            elif not (_finite(n_clv, MIN_CLV_BETS, 1e9) and float(n_clv) == int(n_clv)):
                m_reasons.append(f"{label} @{thr}: CLV-scored count {n_clv!r} "
                                 f"invalid, non-integer, or < {MIN_CLV_BETS} "
                                 "(closing evidence must cover the bets)")
            elif _finite(n, 1.0, 1e9) and float(n_clv) / float(n) < MIN_CLV_COVERAGE:
                m_reasons.append(f"{label} @{thr}: CLV coverage {n_clv}/{n} "
                                 f"below {MIN_CLV_COVERAGE:.0%} — too many bets "
                                 "have no closing price to be scored against")
            # Wilson 95% lower bound on the positive-CLV fraction must clear
            # 0.5 — the raw fraction at exactly 0.5 is a coin flip — and it runs
            # on n_clv, the real CLV sample, not the inflated n_bets.
            elif _wilson_lower_bound(frac, n_clv) <= MIN_CLV_FRAC_POSITIVE:
                m_reasons.append(f"{label} @{thr}: positive-CLV fraction "
                                 f"{frac!r} (n_clv={n_clv}) fails the Wilson 95% "
                                 f"lower bound > {MIN_CLV_FRAC_POSITIVE}")
        if mkey == "1x2":
            ml = bt.get("model_log_loss_1x2")
            mk = bt.get("market_log_loss_1x2_devigged_pinnacle_closing")
            if (not _finite(ml, 0.0, 10.0) or not _finite(mk, 0.0, 10.0)
                    or ml > mk):
                m_reasons.append(f"{label}: model log-loss {ml!r} not finite "
                                 f"and <= market {mk!r}")
        market_pass[mkey] = not m_reasons
        # Only an ACTIVE market's failures are gate reasons. An inactive
        # market's threshold complaints (n_bets == 0 everywhere) are noise, not
        # a veto, so they must not appear as reasons the whole gate is closed.
        if market_active[mkey]:
            reasons.extend(m_reasons)

    # Per-market activation. A market is OPEN when it is active AND clears every
    # criterion; the gate as a whole is "allowed" when at least one market is
    # open. An OU2.5 that cannot be CLV-scored (inactive or failing) can no
    # longer block a 1X2 book that has earned staking, and vice versa.
    markets = {
        mkey: {"active": market_active.get(mkey, False),
               "open": market_active.get(mkey, False) and market_pass.get(mkey, False)}
        for mkey in _MARKETS
    }
    # decision_time_v2 promises per-league activation. Missing or malformed
    # league evidence cannot fall back to a pooled market pass: that would let
    # a well-sampled European market authorize an unmeasured league.
    by_league = bt.get("simulated_betting_by_league")
    if not isinstance(by_league, dict):
        by_league = {}
    for mkey, state in markets.items():
        if not state["open"]:
            continue
        leagues = by_league.get(mkey)
        has_open_league = (
            isinstance(leagues, dict)
            and any(
                isinstance(thresholds, dict)
                and set(thresholds) == REQUIRED_THRESHOLDS
                and all(_league_row_ok(thresholds.get(t))
                        for t in REQUIRED_THRESHOLDS)
                for thresholds in leagues.values()
            )
        )
        if not has_open_league:
            state["open"] = False
            reasons.append(
                f"{_MARKETS[mkey]}: pooled market passed but no league has "
                "complete independently passing evidence"
            )
    allowed = any(m["open"] for m in markets.values())
    if not allowed and not reasons:
        # Closed purely because no market has enough settled evidence yet (every
        # market inactive). Say so, rather than reporting CLOSED with no reason.
        reasons.append("no market has enough settled evidence to open the gate "
                       "yet — the ledger accumulates forward and is still below "
                       f"the {MIN_BETS}-bet bar")
    return {"allowed": allowed, "reasons": [] if allowed else reasons,
            "markets": markets}


def market_staking_allowed() -> dict[str, bool]:
    """Per-market staking gate: {gate_market_key: is_open}. Never raises — an
    unreadable gate reports every market closed (fail-closed)."""
    try:
        v = evaluate()
        return {mkey: bool(m.get("open")) for mkey, m in v.get("markets", {}).items()}
    except Exception:
        return {mkey: False for mkey in _MARKETS}


def _league_row_ok(row) -> bool:
    """League-level pass for one edge-threshold row. Same shape as the market
    criteria (ROI lower bound > 0, positive CLV with real coverage, Wilson on
    n_clv), but with a lower per-league bet floor — a league needs its OWN
    evidence, not a share of a pooled pass."""
    if not isinstance(row, dict):
        return False
    n = row.get("n_bets")
    if not (_finite(n, MIN_LEAGUE_BETS, 1e9) and float(n) == int(n)):
        return False
    flat, kelly = row.get("flat_roi"), row.get("kelly_roi")
    if not (_finite(flat, 1e-9, 10.0) and _finite(kelly, 1e-9, 10.0)):
        return False
    if not _finite(row.get("flat_roi_lb95"), 1e-9, 10.0):
        return False
    if not _finite(row.get("kelly_roi_lb95"), 1e-9, 10.0):
        return False
    clv, frac, n_clv = (row.get("clv_mean"), row.get("clv_frac_positive"),
                        row.get("n_clv"))
    if not (_finite(clv, 1e-9, 1.0) and _finite(frac, MIN_CLV_FRAC_POSITIVE, 1.0)):
        return False
    if not (_finite(n_clv, MIN_CLV_BETS, 1e9) and float(n_clv) == int(n_clv)):
        return False
    if _finite(n, 1.0, 1e9) and float(n_clv) / float(n) < MIN_CLV_COVERAGE:
        return False
    return _wilson_lower_bound(frac, n_clv) > MIN_CLV_FRAC_POSITIVE


def market_league_staking_allowed() -> dict[tuple[str, str], bool]:
    """Per (gate_market_key, competition) gate.

    Returns {} when the artifact carries no valid league section; callers must
    treat that as every league closed. Every (market, league) present is
    reported: it is open only if its
    MARKET is open AND the league independently clears every threshold, so a
    pooled European pass can never unlock a CLV-less league. Never raises."""
    try:
        bt = json.loads(BACKTEST_JSON.read_text(),
                        object_pairs_hook=_reject_duplicate_keys)
        if not isinstance(bt, dict):
            return {}
        by_league = bt.get("simulated_betting_by_league")
        if not isinstance(by_league, dict) or not by_league:
            return {}
        market_open = market_staking_allowed()
        out: dict[tuple[str, str], bool] = {}
        for mkey, leagues in by_league.items():
            if not isinstance(leagues, dict):
                continue
            for comp, thresholds in leagues.items():
                ok = (bool(market_open.get(mkey))
                      and isinstance(thresholds, dict)
                      and set(thresholds) == REQUIRED_THRESHOLDS
                      and all(_league_row_ok(thresholds.get(t))
                              for t in REQUIRED_THRESHOLDS))
                out[(mkey, str(comp))] = bool(ok)
        return out
    except Exception:
        return {}


def staking_allowed() -> tuple[bool, list[str]]:
    """(any-market-open, reasons). Never raises: an unreadable gate fails
    closed. For per-market decisions use market_staking_allowed()."""
    try:
        v = evaluate()
        return bool(v["allowed"]), list(v["reasons"])
    except Exception as exc:                       # fail CLOSED, always
        return False, [f"evidence gate error ({type(exc).__name__}: {exc})"]


def main() -> None:
    allowed, reasons = staking_allowed()
    print(f"Staking allowed: {allowed}")
    for r in reasons:
        print(f"  - {r}")
    raise SystemExit(0 if allowed else 1)


if __name__ == "__main__":
    main()

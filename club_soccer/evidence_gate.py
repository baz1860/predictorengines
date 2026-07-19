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
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
BACKTEST_JSON = DATA / "backtest_market.json"

MAX_ARTIFACT_AGE_DAYS = 14.0
MIN_BETS = 1000
MIN_CLV_FRAC_POSITIVE = 0.5

# backtest_market.json market keys -> human label
_MARKETS = {"1x2": "1X2", "total_over_under_2_5": "OU2.5"}

# ── methodology requirements ──────────────────────────────────────────────
# The gate must be impossible to open with evidence from the legacy
# closing-odds simulator (which selects on the close and executes at the
# close — not a deployable strategy). Only an artifact that explicitly
# declares the decision-time methodology counts. The current
# backtest_market.py does NOT produce this artifact; the gate therefore
# stays closed until the backtest is redesigned — which is correct.
REQUIRED_VERSION = "decision_time_v2"
REQUIRED_METHODOLOGY = {
    "selection_method": "latest_quote_at_or_before_decision_time",
    "execution_method": "same_decision_time_quote",
    "clv_reference": "pinnacle_closing_devigged",
}
MIN_DECISION_LEAD_MINUTES = 60
REQUIRED_THRESHOLDS = {"2%", "4%", "6%"}


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
    if not isinstance(lead, (int, float)) or lead < MIN_DECISION_LEAD_MINUTES:
        fails.append(f"decision_lead_minutes is {lead!r}, need >= "
                     f"{MIN_DECISION_LEAD_MINUTES}")
    gen = bt.get("generated_at_utc")
    try:
        gen_dt = datetime.fromisoformat(str(gen))
        if gen_dt.tzinfo is None:
            gen_dt = gen_dt.replace(tzinfo=timezone.utc)
        age = (now - gen_dt).total_seconds() / 86400.0
        if age < 0:
            fails.append(f"generated_at_utc {gen!r} is in the future")
        elif age > MAX_ARTIFACT_AGE_DAYS:
            fails.append(f"evidence is {age:.1f} days old "
                         f"(limit {MAX_ARTIFACT_AGE_DAYS:g})")
    except (TypeError, ValueError):
        fails.append(f"generated_at_utc missing/unparseable ({gen!r}) — "
                     "file mtime is not acceptable provenance")
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
        bt = json.loads(BACKTEST_JSON.read_text())
    except Exception as exc:
        return {"allowed": False,
                "reasons": [f"backtest_market.json unreadable ({exc})"]}

    # Methodology first: metrics from a non-decision-time backtest are
    # meaningless, so fail before looking at any number.
    meth = _methodology_failures(bt, now)
    if meth:
        return {"allowed": False, "reasons": meth}

    sim = bt.get("simulated_betting") or {}
    market_pass: dict[str, bool] = {}
    for mkey, label in _MARKETS.items():
        m_reasons: list[str] = []
        thresholds = sim.get(mkey) or {}
        if set(thresholds) != REQUIRED_THRESHOLDS:
            reasons.append(f"{label}: thresholds {sorted(thresholds)} != required "
                           f"{sorted(REQUIRED_THRESHOLDS)}")
            market_pass[mkey] = False
            continue
        for thr in sorted(thresholds):
            row = thresholds[thr]
            n = int(row.get("n_bets") or 0)
            if n < MIN_BETS:
                m_reasons.append(f"{label} @{thr}: only {n} backtest bets "
                                 f"(< {MIN_BETS} at every threshold)")
            flat, kelly = row.get("flat_roi"), row.get("kelly_roi")
            if flat is None or flat <= 0 or kelly is None or kelly <= 0:
                m_reasons.append(f"{label} @{thr}: ROI not positive "
                                 f"(flat {flat}, kelly {kelly})")
            clv, frac = row.get("clv_mean"), row.get("clv_frac_positive")
            if clv is None or frac is None:
                m_reasons.append(f"{label} @{thr}: CLV evidence missing "
                                 "(absent CLV can never pass)")
            elif clv <= 0 or frac < MIN_CLV_FRAC_POSITIVE:
                m_reasons.append(f"{label} @{thr}: CLV not positive "
                                 f"(mean {clv}, frac+ {frac})")
        if mkey == "1x2":
            ml = bt.get("model_log_loss_1x2")
            mk = bt.get("market_log_loss_1x2_devigged_pinnacle_closing")
            if ml is None or mk is None or ml > mk:
                m_reasons.append(f"{label}: model log-loss {ml} worse than "
                                 f"market {mk}")
        market_pass[mkey] = not m_reasons
        reasons.extend(m_reasons)

    # The stake-zeroing is global, so the gate only opens when EVERY market
    # with evidence passes — a winning OU2.5 must not unlock 1X2 stakes.
    # (Per-market gating is a possible refinement once anything passes.)
    allowed = (bool(market_pass) and all(market_pass.values())
               and not reasons)
    return {"allowed": allowed, "reasons": [] if allowed else reasons}


def staking_allowed() -> tuple[bool, list[str]]:
    """(allowed, reasons). Never raises: an unreadable gate fails closed."""
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

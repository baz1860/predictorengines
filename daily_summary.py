#!/usr/bin/env python3
"""Daily suite summary (V3 M9).

One offline read-out of suite health for the morning check or the end of
update.sh: per-engine validation status, data-freshness warnings, current
recommendation count, CLV status, and bankroll state. Reads local files only and
runs each engine's edge in preview (no ledger writes); any unavailable input
degrades to a clear local action instead of failing.

    python3 daily_summary.py        # human-readable summary
    python3 daily_summary.py --json # machine-readable
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from app import bankroll_store, model_audit
from app.engines import registry
from core import clv_suite


def _recommended_count(engine) -> int | None:
    """Recommended bets from a read-only edge preview, or None if unavailable."""
    if "edge" not in engine.capabilities:
        return None
    try:
        res = engine.edge({})  # no record → preview only
        return sum(1 for r in (res.get("rows") or []) if r.get("recommended"))
    except Exception:
        return None


def _clv_status() -> dict:
    hist = clv_suite._load_history()
    if hist is None:
        return {"status": "no snapshots",
                "action": "run `python3 clv_suite.py --snapshot` before events start"}
    ledger = clv_suite._load_ledger()
    have = clv_suite._clv_table(ledger, hist) if not ledger.empty else ledger
    if have is None or have.empty:
        return {"status": "no settled bets with closing snapshots yet",
                "snapshots": int(len(hist))}
    return {"status": "ok", "snapshots": int(len(hist)),
            "settled_with_closing": int(len(have)),
            "mean_clv_pct": round(float(have["clv"].mean()) * 100, 2)}


def build_summary() -> dict:
    bk = bankroll_store.status_summary()
    engines = {}
    for eng in registry.all():
        a = model_audit.audit(eng.id)
        engines[eng.id] = {
            "name": eng.name,
            "validation": a["validation"]["status"],
            "freshness_warnings": a["freshness_warnings"],
            "params_age_days": a["params_age_days"],
            "recommended": _recommended_count(eng),
        }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bankroll": {"bankroll": bk["bankroll"], "net_pnl": bk["totals"]["net_pnl"],
                     "open_bets": bk["totals"]["open_count"]},
        "engines": engines,
        "clv": _clv_status(),
    }


def _print(s: dict) -> None:
    print(f"Suite daily summary · {s['generated_at']}\n")
    bk = s["bankroll"]
    print(f"Bankroll £{bk['bankroll']:.2f}  ·  net P&L £{bk['net_pnl']:+.2f}  ·  "
          f"{bk['open_bets']} open bet(s)\n")
    print(f"{'engine':<14}{'gate':>8}{'recs':>6}  freshness")
    for eid, e in s["engines"].items():
        recs = "—" if e["recommended"] is None else str(e["recommended"])
        warn = "; ".join(e["freshness_warnings"]) if e["freshness_warnings"] else "fresh"
        print(f"{eid:<14}{e['validation']:>8}{recs:>6}  {warn}")
    clv = s["clv"]
    print("\nCLV: " + clv["status"]
          + (f" · {clv['settled_with_closing']} settled, mean {clv['mean_clv_pct']:+.2f}%"
             if clv.get("status") == "ok" else ""))
    if clv.get("action"):
        print(f"     → {clv['action']}")
    gates = [e["validation"] for e in s["engines"].values()]
    if any(g == "FAIL" for g in gates):
        print("\n⚠ A validation gate is FAILING — review before betting.")
    elif any(g == "unknown" for g in gates):
        print("\nℹ Some gates have not run — `python3 validate_all.py --gate`.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Daily suite summary (V3 M9)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()
    s = build_summary()
    print(json.dumps(s, indent=2)) if args.json else _print(s)


if __name__ == "__main__":
    main()

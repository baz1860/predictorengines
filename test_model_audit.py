#!/usr/bin/env python3
"""M8 model-audit + template-enrichment tests (server-side pieces).

Covers:
  * audit() assembles validation status, freshness, params age and active flags
    offline for every engine;
  * audit degrades cleanly when no validation run exists;
  * enrich_template_result() adds an absolute path + data-row count;
  * the audit payload exposes no API-key values (M8 export safety).

Run: python3 test_model_audit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import model_audit
from app.engines.contracts import enrich_template_result

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def test_audit_structure():
    for eng in ("worldcup", "club_soccer", "cfb", "golf"):
        a = model_audit.audit(eng)
        check(f"{eng}: has validation status",
              a["validation"]["status"] in ("PASS", "FAIL", "unknown"),
              str(a["validation"]))
        check(f"{eng}: freshness is a list", isinstance(a["freshness"], list))
        check(f"{eng}: flags is a list", isinstance(a["flags"], list))
        check(f"{eng}: every flag has label+active",
              all("label" in f and "active" in f for f in a["flags"]), str(a["flags"]))


def test_cfb_flags():
    a = model_audit.audit("cfb")
    labels = [f["label"] for f in a["flags"]]
    check("cfb exposes blend-weight flag", "Elo/power blend weight" in labels, str(labels))
    check("cfb exposes market-blend flag", "Market blend" in labels, str(labels))
    mb = next(f for f in a["flags"] if f["label"] == "Market blend")
    check("cfb market blend is off by default", mb["active"] is False, str(mb))


def test_validation_unknown_when_missing():
    saved = model_audit.VALIDATION_SUITE
    try:
        model_audit.VALIDATION_SUITE = ROOT / "no_such_validation_suite.json"
        a = model_audit.audit("cfb")
        check("missing suite → status unknown", a["validation"]["status"] == "unknown",
              str(a["validation"]))
        check("missing suite → still returns freshness", isinstance(a["freshness"], list))
    finally:
        model_audit.VALIDATION_SUITE = saved


def test_template_enrichment():
    r = enrich_template_result({"path": "cfb/odds.csv"})
    check("abs_path is absolute", Path(r.get("abs_path", "")).is_absolute(), str(r))
    check("abs_path ends with cfb/odds.csv", r.get("abs_path", "").endswith("cfb/odds.csv"))
    check("row count is a non-negative int",
          isinstance(r.get("rows"), int) and r["rows"] >= 0, str(r.get("rows")))
    empty = enrich_template_result({})
    check("no path → unchanged", "abs_path" not in empty)


def test_no_secret_values():
    import json
    blob = json.dumps([model_audit.audit(e) for e in
                       ("worldcup", "club_soccer", "cfb", "golf")])
    # the audit must not embed any odds-API key material
    from app import settings_store
    keys = settings_store.load().get("odds_api_keys", {}) or {}
    leaked = [k for k in keys.values() if k and str(k) in blob]
    check("audit payload leaks no API-key values", not leaked, str(leaked))


def main():
    print("M8 model-audit tests")
    test_audit_structure()
    test_cfb_flags()
    test_validation_unknown_when_missing()
    test_template_enrichment()
    test_no_secret_values()
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

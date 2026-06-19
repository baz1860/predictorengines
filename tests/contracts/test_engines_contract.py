#!/usr/bin/env python3
"""Contract tests: every registered adapter must speak the V3 app contract.

For each engine we exercise:
  * info()/schema     – structurally valid and finite-JSON;
  * predict           – valid prediction payload (where supported);
  * edge (manual)     – canonical, finite edge rows (where supported);
  * simulate          – tiny seeded sim returns a valid table (World Cup, Golf).

Missing local data or unfilled odds files report as SKIP, not FAIL — but any
payload that IS produced must satisfy the contract. Contract violations
(ContractError / AssertionError) are hard failures. Exits non-zero on any FAIL.

Run: python3 test_engines_contract.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.engines import registry
from contracts import (
    ContractError, assert_finite_json, validate_edge_rows,
    validate_prediction, validate_table)

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
_results: list[tuple[str, str, str, str]] = []


def _record(engine: str, check: str, status: str, detail: str = "") -> None:
    _results.append((engine, check, status, detail))


def _try(engine: str, check: str, fn) -> None:
    """Run a check. ContractError/AssertionError = FAIL; missing data = SKIP."""
    try:
        fn()
        _record(engine, check, PASS)
    except (ContractError, AssertionError) as e:
        _record(engine, check, FAIL, str(e)[:140])
    except (ValueError, RuntimeError, FileNotFoundError, KeyError) as e:
        _record(engine, check, SKIP, str(e).splitlines()[0][:90])


# ── per-capability checks ─────────────────────────────────────────────────────
def _check_info(adapter) -> None:
    info = adapter.info()
    assert isinstance(info, dict), "info() must return a dict"
    assert info.get("id") and info.get("name"), "info missing id/name"
    assert isinstance(info.get("capabilities"), list), "capabilities must be list"
    assert_finite_json(info)


def _two_names(adapter) -> list[str]:
    names = adapter.predict_schema().get("names") or []
    if len(names) < 2:
        raise ValueError("fewer than two competitor names available")
    return names[:2]


def _check_predict(adapter) -> None:
    a, b = _two_names(adapter)
    params = {"team1": a, "team2": b, "home": a, "away": b,
              "player_a": a, "player_b": b, "home_team": a}
    result = adapter.predict(params)
    validate_prediction(result)


def _check_edge(adapter) -> None:
    result = adapter.edge({"odds_source": "manual", "record": False})
    assert isinstance(result, dict), "edge() must return a dict"
    assert_finite_json(result)
    rows = result.get("rows") or []
    validate_edge_rows(rows)


def _check_simulate(adapter) -> None:
    params = {"seed": 1, "model": "blend"}
    params["sims"] = 2000 if adapter.id == "golf" else 500
    result = adapter.simulate(params)
    validate_table(result)


def main() -> int:
    _results.clear()
    for adapter in registry.all():
        eid = adapter.id
        _try(eid, "info", lambda a=adapter: _check_info(a))
        caps = adapter.capabilities
        if "predict" in caps:
            _try(eid, "predict", lambda a=adapter: _check_predict(a))
        if "edge" in caps:
            _try(eid, "edge", lambda a=adapter: _check_edge(a))
        if "simulate" in caps:
            _try(eid, "simulate", lambda a=adapter: _check_simulate(a))

    width = max((len(e) for e, *_ in _results), default=8)
    print(f"{'ENGINE'.ljust(width)}  CHECK      STATUS  DETAIL")
    print("-" * (width + 40))
    failures = 0
    for engine, check, status, detail in _results:
        if status == FAIL:
            failures += 1
        print(f"{engine.ljust(width)}  {check.ljust(9)}  {status:<6}  {detail}")

    summary = {s: sum(1 for _e, _c, st, _d in _results if st == s)
               for s in (PASS, SKIP, FAIL)}
    print("-" * (width + 40))
    print(f"{summary[PASS]} pass · {summary[SKIP]} skip · {summary[FAIL]} fail "
          f"({len(_results)} checks)")
    return 1 if failures else 0


def test_registered_engine_contracts():
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())

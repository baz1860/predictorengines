"""Root pytest configuration.

Most suites in this repo use a homegrown `check(name, cond)` helper that
prints PASS/FAIL and increments a module-level counter (`FAIL`) or appends to
a list (`_fails`) WITHOUT raising. Under pytest that made every such suite
false-green: checks could fail while the run reported success.

This autouse fixture closes the hole generically: after every test, any
growth in the module's FAIL counter or _fails list fails that test. Script
mode (`python3 test_x.py`) is unaffected — modules keep their own
collect-and-report behavior there.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _check_counters_fail_the_test(request):
    mod = request.module
    fail_before = getattr(mod, "FAIL", None)
    list_before = None
    if hasattr(mod, "_fails") and isinstance(getattr(mod, "_fails"), list):
        list_before = len(mod._fails)

    yield

    problems: list[str] = []
    if isinstance(fail_before, int):
        grown = getattr(mod, "FAIL", fail_before) - fail_before
        if grown > 0:
            problems.append(f"{grown} check() failure(s) recorded in "
                            f"{mod.__name__}.FAIL")
    if list_before is not None:
        new = mod._fails[list_before:]
        if new:
            problems.append(f"check failure(s): " + ", ".join(map(str, new)))
    assert not problems, "; ".join(problems)

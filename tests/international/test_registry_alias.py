"""Engine-alias mechanism (plan §8 step 1, §7.4).

Must exist BEFORE any engine is renamed. `app/bankroll_store.py` settles a bet by
calling `registry.get(row["engine"])`; as of August 2026 data/suite_ledger.csv holds
21 rows with engine="worldcup". Renaming the adapter without an alias orphans them.

These tests pin the behaviour a rename depends on:
  * a legacy id resolves to the renamed adapter;
  * `all()` lists it once, under the new id, so the UI does not show a ghost engine;
  * `resolve_id()` gives new writes the canonical id while old reads keep working;
  * every id collision raises AT REGISTRATION, not silently at settlement time.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contracts.registry import AliasConflict, EngineAdapter, Registry  # noqa: E402

FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global FAIL
    if cond:
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}{(' — ' + detail) if detail else ''}")


class Intl(EngineAdapter):
    id = "international"
    name = "International Football"
    sport = "soccer"
    capabilities = {"predict"}
    legacy_ids = frozenset({"worldcup"})

    def predict_schema(self):
        return {"kind": "match", "names": [], "models": []}


class Other(EngineAdapter):
    id = "golf"
    name = "Golf"
    sport = "golf"
    capabilities = {"predict"}

    def predict_schema(self):
        return {"kind": "field", "names": [], "models": []}


def test_resolution() -> None:
    print("\nalias resolution")
    r = Registry()
    a = Intl()
    r.register(a)
    r.register(Other())

    check("primary id resolves", r.get("international") is a)
    check("legacy id resolves to the same adapter", r.get("worldcup") is a)
    check("all() lists primaries only", sorted(e.id for e in r.all())
          == ["golf", "international"])
    check("all() does not duplicate the aliased engine",
          sum(1 for e in r.all() if e is a) == 1)

    check("resolve_id maps legacy -> primary", r.resolve_id("worldcup") == "international")
    check("resolve_id is identity for a primary",
          r.resolve_id("international") == "international")
    check("is_alias distinguishes the two",
          r.is_alias("worldcup") and not r.is_alias("international"))
    check("aliases() reports the mapping",
          r.aliases() == {"worldcup": "international"})
    check("known_ids covers both namespaces",
          r.known_ids() == {"international", "golf", "worldcup"})
    check("info() advertises legacy ids", a.info()["legacy_ids"] == ["worldcup"])

    try:
        r.get("nope")
        check("unknown id still raises KeyError", False)
    except KeyError:
        check("unknown id still raises KeyError", True)


def test_collisions() -> None:
    print("\ncollision detection")

    class Dup(EngineAdapter):
        id = "international"
        capabilities = {"predict"}

        def predict_schema(self):
            return {}

    r = Registry()
    r.register(Intl())
    try:
        r.register(Dup())
        check("duplicate primary id raises", False)
    except AliasConflict:
        check("duplicate primary id raises", True)

    # A new engine must not claim an id already used as someone's alias.
    class StealsAlias(EngineAdapter):
        id = "worldcup"
        capabilities = {"predict"}

        def predict_schema(self):
            return {}

    try:
        r.register(StealsAlias())
        check("primary id colliding with an existing alias raises", False)
    except AliasConflict:
        check("primary id colliding with an existing alias raises", True)

    # Two engines must not claim the same legacy id.
    class AlsoClaims(EngineAdapter):
        id = "other"
        legacy_ids = frozenset({"worldcup"})
        capabilities = {"predict"}

        def predict_schema(self):
            return {}

    try:
        r.register(AlsoClaims())
        check("two engines claiming one alias raises", False)
    except AliasConflict:
        check("two engines claiming one alias raises", True)

    # An alias must not shadow an already-registered primary id.
    class ShadowsPrimary(EngineAdapter):
        id = "newthing"
        legacy_ids = frozenset({"golf"})
        capabilities = {"predict"}

        def predict_schema(self):
            return {}

    r2 = Registry()
    r2.register(Other())
    try:
        r2.register(ShadowsPrimary())
        check("alias shadowing a primary id raises", False)
    except AliasConflict:
        check("alias shadowing a primary id raises", True)

    class SelfAlias(EngineAdapter):
        id = "x"
        legacy_ids = frozenset({"x"})
        capabilities = {"predict"}

        def predict_schema(self):
            return {}

    try:
        Registry().register(SelfAlias())
        check("self-referential alias raises", False)
    except AliasConflict:
        check("self-referential alias raises", True)


def test_live_registry_unchanged() -> None:
    """The real registry must behave exactly as before — nothing is renamed yet."""
    print("\nlive registry")
    from app.engines import registry

    ids = sorted(e.id for e in registry.all())
    check("worldcup still resolves", registry.get("worldcup").id == "worldcup")
    check("worldcup is not yet an alias", not registry.is_alias("worldcup"))
    check("no aliases declared yet", registry.aliases() == {}, str(registry.aliases()))
    check("all engines still listed", "worldcup" in ids and len(ids) >= 7, str(ids))

    # The ledger ids that must keep resolving after a future rename.
    import pandas as pd
    ledger = ROOT / "data" / "suite_ledger.csv"
    if ledger.exists():
        engines = set(pd.read_csv(ledger)["engine"].dropna().astype(str))
        unresolvable = sorted(e for e in engines if e not in registry.known_ids())
        check("every engine id in the suite ledger resolves",
              unresolvable == [], str(unresolvable))


def main() -> int:
    test_resolution()
    test_collisions()
    test_live_registry_unchanged()
    print(f"\n{'ALL PASS' if FAIL == 0 else str(FAIL) + ' FAILURE(S)'}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

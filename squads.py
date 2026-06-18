"""Compatibility shim — `squads` moved to engines/worldcup/ (refactor Phase 3a).

Re-exports the package module so existing top-level `import squads` / `from squads import …`
keeps working until importers migrate in Phase 3b, at which point this file is removed.
"""
import sys as _sys
from engines.worldcup import squads as _m

_sys.modules[__name__] = _m

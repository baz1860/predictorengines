"""Compatibility shim — `market_blend` moved to engines/worldcup/ (refactor Phase 3a).

Re-exports the package module so existing top-level `import market_blend` / `from market_blend import …`
keeps working until importers migrate in Phase 3b, at which point this file is removed.
"""
import sys as _sys
from engines.worldcup import market_blend as _m

_sys.modules[__name__] = _m

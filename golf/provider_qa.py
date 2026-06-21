"""Small QA helpers for free golf data providers.

Free sources are useful but brittle: ESPN and PGA Tour pages can change shape,
and manual odds paste can be malformed. These helpers keep provider modules
honest by returning structured checks that can be surfaced in refresh output,
tests, and the app provenance layer.
"""

from __future__ import annotations

import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(frozen=True)
class SourceCheck:
    source: str
    ok: bool
    severity: str
    message: str
    rows: int = 0

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "ok": self.ok,
            "severity": self.severity,
            "message": self.message,
            "rows": self.rows,
        }


def _fold_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.lower().replace(".", "").replace(",", "").split())


def require_columns(source: str, rows: Iterable[Mapping], columns: list[str]) -> SourceCheck:
    rows = list(rows)
    if not rows:
        return SourceCheck(source, False, "error", "no rows returned", 0)
    missing = sorted({c for c in columns if c not in rows[0]})
    if missing:
        return SourceCheck(
            source,
            False,
            "error",
            f"missing required column(s): {', '.join(missing)}",
            len(rows),
        )
    return SourceCheck(source, True, "info", "schema ok", len(rows))


def min_rows(source: str, rows: Iterable[Mapping], minimum: int) -> SourceCheck:
    n = len(list(rows))
    if n < minimum:
        return SourceCheck(source, False, "warning", f"only {n} row(s), expected >= {minimum}", n)
    return SourceCheck(source, True, "info", f"{n} row(s)", n)


def file_freshness(source: str, path: str | Path, max_age_hours: float) -> SourceCheck:
    p = Path(path)
    if not p.exists():
        return SourceCheck(source, False, "warning", f"{p} does not exist")
    age_hours = (time.time() - p.stat().st_mtime) / 3600.0
    if age_hours > max_age_hours:
        return SourceCheck(
            source,
            False,
            "warning",
            f"{p} is stale ({age_hours:.1f}h old, max {max_age_hours:.1f}h)",
        )
    return SourceCheck(source, True, "info", f"{p} fresh ({age_hours:.1f}h old)")


def player_match_rate(
    source: str,
    provider_names: Iterable[str],
    model_names: Iterable[str],
    minimum: float = 0.90,
) -> SourceCheck:
    provider = {_fold_name(n) for n in provider_names if str(n or "").strip()}
    model = {_fold_name(n) for n in model_names if str(n or "").strip()}
    if not provider:
        return SourceCheck(source, False, "error", "no provider player names", 0)
    matched = len(provider & model)
    rate = matched / len(provider)
    ok = rate >= minimum
    return SourceCheck(
        source,
        ok,
        "info" if ok else "warning",
        f"player match rate {rate:.1%} ({matched}/{len(provider)})",
        len(provider),
    )


def summarize(checks: Iterable[SourceCheck]) -> dict:
    checks = list(checks)
    return {
        "ok": all(c.ok or c.severity == "info" for c in checks),
        "errors": [c.as_dict() for c in checks if not c.ok and c.severity == "error"],
        "warnings": [c.as_dict() for c in checks if not c.ok and c.severity == "warning"],
        "checks": [c.as_dict() for c in checks],
    }

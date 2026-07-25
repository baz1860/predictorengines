"""Bound recoverable provider caches by age and size."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


@dataclass(frozen=True)
class Policy:
    path: Path
    max_age_days: int
    max_bytes: int


POLICIES = (
    Policy(DATA / "bsd_cache", 180, 512 * 1024 * 1024),
    Policy(DATA / "bsd_enrichment", 365, 256 * 1024 * 1024),
)


def prune(policy: Policy, now: float | None = None) -> dict[str, int]:
    """Delete expired cache files, then oldest files until under the quota."""
    now = time.time() if now is None else float(now)
    if not policy.path.exists():
        return {"files_removed": 0, "bytes_removed": 0, "bytes_remaining": 0}
    files = [
        path for path in policy.path.rglob("*")
        if path.is_file() and not path.is_symlink()
    ]
    records = sorted(
        ((path.stat().st_mtime, path.stat().st_size, path) for path in files),
        key=lambda row: row[0],
    )
    cutoff = now - policy.max_age_days * 86400
    total = sum(size for _mtime, size, _path in records)
    removed_files = removed_bytes = 0
    for mtime, size, path in records:
        if mtime >= cutoff and total <= policy.max_bytes:
            continue
        path.unlink()
        total -= size
        removed_files += 1
        removed_bytes += size
    return {
        "files_removed": removed_files,
        "bytes_removed": removed_bytes,
        "bytes_remaining": total,
    }


def prune_all() -> dict[str, dict[str, int]]:
    results = {}
    for policy in POLICIES:
        result = prune(policy)
        results[policy.path.name] = result
        if result["files_removed"]:
            print(
                f"  {policy.path.name}: removed {result['files_removed']} files "
                f"({result['bytes_removed'] / 1024 / 1024:.1f} MiB)"
            )
    return results

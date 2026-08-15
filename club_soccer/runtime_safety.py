"""Single-writer guard and non-destructive Syncthing conflict audit."""
from __future__ import annotations

import argparse
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def assert_writer_host() -> None:
    """Reject scheduled writes on a host other than the declared authority.

    Launchd deployments set ``CLUB_SOCCER_WRITER_HOST``.  Tests and explicitly
    isolated runtimes may omit it, but synchronized production schedules may
    not silently turn a receive-only machine into a second writer.
    """
    expected = os.environ.get("CLUB_SOCCER_WRITER_HOST", "").strip().casefold()
    if not expected:
        return
    actual = socket.gethostname().strip().casefold()
    if actual != expected:
        raise RuntimeError(
            f"club_soccer runtime is single-writer: expected host {expected!r}, "
            f"running on {actual!r}"
        )


def conflict_report(data_dir: Path = DATA) -> dict:
    conflicts = []
    for path in sorted(data_dir.glob("*sync-conflict*")):
        conflicts.append({
            "name": path.name,
            "bytes": path.stat().st_size,
            "modified_at_utc": datetime.fromtimestamp(
                path.stat().st_mtime, timezone.utc
            ).isoformat(),
        })
    return {
        "ok": not conflicts,
        "data_dir": str(data_dir),
        "conflict_count": len(conflicts),
        "conflicts": conflicts,
        "note": ("Conflict copies are never canonical inputs. Preserve and "
                 "reconcile them by immutable ledger key; regenerate derived "
                 "artifacts from the chosen authoritative ledgers."),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = conflict_report()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Syncthing conflicts: {report['conflict_count']}")
        for row in report["conflicts"]:
            print(f"  {row['name']} ({row['bytes']} bytes)")
    raise SystemExit(0 if report["ok"] else 2)


if __name__ == "__main__":
    main()

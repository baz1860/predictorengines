"""Atomic machine-readable status for the CFB refresh pipeline."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATUS_JSON = HERE / "data" / "update_status.json"


def write_status(status: str, step: str, message: str = "",
                 path: str | Path = STATUS_JSON) -> Path:
    if status not in {"running", "success", "failure"}:
        raise ValueError(status)
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "step": step,
        "message": message,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    fd, tmp = tempfile.mkstemp(prefix=f".{dest.name}.", dir=dest.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        os.replace(tmp, dest)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", required=True, choices=["running", "success", "failure"])
    ap.add_argument("--step", required=True)
    ap.add_argument("--message", default="")
    args = ap.parse_args()
    write_status(args.status, args.step, args.message)


if __name__ == "__main__":
    main()


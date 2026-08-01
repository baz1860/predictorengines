"""Content fingerprints and provenance for CFB validation inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MANIFEST = DATA / "validation_datasets.json"

ARTIFACTS = {
    "games": {
        "path": DATA / "games.csv",
        "source": "sportsdataverse/cfbfastR-data schedules",
        "decision_time": "completed result available before each walk-forward fit",
    },
    "closing_spreads": {
        "path": DATA / "closing_spreads.csv",
        "source": (
            "legacy mixed file: sportsdataverse betting mirror plus CFBD /lines "
            "imports; per-row source was not retained"
        ),
        "decision_time": (
            "consensus closing benchmark; CFBD imports have no quote timestamp "
            "and assume -110 when juice is absent"
        ),
    },
    "closing_totals": {
        "path": DATA / "closing_totals.csv",
        "source": (
            "legacy mixed file: sportsdataverse betting mirror plus CFBD /lines "
            "imports; per-row source was not retained"
        ),
        "decision_time": (
            "consensus closing benchmark; not evidence for an opener or "
            "decision-time strategy"
        ),
    },
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(path: str | Path, *, source: str = "",
                decision_time: str = "") -> dict:
    file_path = Path(path)
    frame = pd.read_csv(file_path)
    seasons: dict[str, int] = {}
    if "season" in frame:
        values = pd.to_numeric(frame["season"], errors="coerce").dropna().astype(int)
        seasons = {str(int(season)): int(count)
                   for season, count in values.value_counts().sort_index().items()}
    try:
        display_path = str(file_path.relative_to(HERE.parent))
    except ValueError:
        display_path = str(file_path)
    return {
        "path": display_path,
        "sha256": _sha256(file_path),
        "bytes": int(file_path.stat().st_size),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "season_counts": seasons,
        "source": source,
        "decision_time": decision_time,
    }


def snapshot() -> dict:
    artifacts = {
        name: fingerprint(meta["path"], source=meta["source"],
                          decision_time=meta["decision_time"])
        for name, meta in ARTIFACTS.items()
    }
    digest = hashlib.sha256(json.dumps(
        {name: value["sha256"] for name, value in artifacts.items()},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    line_digest = hashlib.sha256(json.dumps(
        {name: value["sha256"] for name, value in artifacts.items()
         if name.startswith("closing_")},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"schema_version": 1, "combined_sha256": digest,
            "line_sha256": line_digest,
            "artifacts": artifacts}


def compact_snapshot() -> dict:
    current = snapshot()
    return {
        "combined_sha256": current["combined_sha256"],
        "line_sha256": current["line_sha256"],
        "artifacts": {
            name: {"sha256": value["sha256"], "rows": value["rows"],
                   "season_counts": value["season_counts"]}
            for name, value in current["artifacts"].items()
        },
    }


def write_manifest(path: str | Path = MANIFEST) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = snapshot()
    fd, tmp = tempfile.mkstemp(prefix=f".{dest.name}.", dir=dest.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(tmp, dest)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    value = snapshot()
    if args.write:
        print(f"wrote {write_manifest()}")
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()

"""Manual tee-time parser for pasted PGA tee sheets.

The reliable free feeds do not always carry tee times. This parser accepts a
plain-text paste and writes the CSV contract consumed by refresh.py:
event_id,event,name,round,tee_time,start_hole.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
RAW_TXT = DATA_DIR / "tee_times_raw.txt"
TEE_TIMES_CSV = DATA_DIR / "tee_times.csv"

_TIME_RE = re.compile(r"\b(\d{1,2}:\d{2})\s*([AP]M)?\b", re.I)
_ROUND_RE = re.compile(r"\b(?:round|r)\s*([12])\b", re.I)
_HOLE_RE = re.compile(r"\b(?:hole|tee|start(?:ing)? hole)\s*(1|10)\b", re.I)


def _clean(line: str) -> str:
    return " ".join(str(line or "").replace("\t", " ").split())


def _looks_like_name(line: str) -> bool:
    if not line or any(ch.isdigit() for ch in line):
        return False
    low = line.lower()
    if any(tok in low for tok in ("round", "tee time", "starting", "group")):
        return False
    return len(line.split()) >= 2


def parse_tee_sheet_text(raw: str, event_id: str = "", event: str = "",
                         default_round: int = 1) -> list[dict]:
    """Parse common tee-sheet paste shapes.

    Supported examples:
      8:05 AM  Tee 1  Scottie Scheffler / Rory McIlroy / Xander Schauffele
      13:20 10 Justin Rose, Tommy Fleetwood
      Round 2
      7:45 AM
      Player One
      Player Two
    """
    rows: list[dict] = []
    current_round = int(default_round or 1)
    pending_time = ""
    pending_hole = ""
    for raw_line in raw.splitlines():
        line = _clean(raw_line)
        if not line:
            continue
        m_round = _ROUND_RE.search(line)
        if m_round:
            current_round = int(m_round.group(1))
        m_time = _TIME_RE.search(line)
        if m_time:
            pending_time = m_time.group(1)
            if m_time.group(2):
                pending_time += " " + m_time.group(2).upper()
        m_hole = _HOLE_RE.search(line)
        if m_hole:
            pending_hole = m_hole.group(1)
        elif m_time:
            after = line[m_time.end():].strip()
            first = after.split(" ", 1)[0] if after else ""
            if first in {"1", "10"}:
                pending_hole = first

        names_part = line
        if m_time:
            names_part = line[m_time.end():]
            names_part = re.sub(r"^\s*(?:tee|hole)?\s*(?:1|10)\b", "", names_part, flags=re.I)
        names_part = re.sub(r"\b(?:round|r)\s*[12]\b", "", names_part, flags=re.I)
        names_part = re.sub(r"\b(?:tee|hole|start(?:ing)? hole)\s*(?:1|10)\b", "", names_part, flags=re.I)
        parts = [
            _clean(p) for p in re.split(r"\s*/\s*|\s*,\s*|\s{2,}", names_part)
            if _looks_like_name(_clean(p))
        ]
        if not parts and _looks_like_name(names_part):
            parts = [_clean(names_part)]
        for name in parts:
            if not pending_time:
                continue
            rows.append({
                "event_id": event_id,
                "event": event,
                "name": name,
                "round": current_round,
                "tee_time": pending_time,
                "start_hole": pending_hole,
            })
    return rows


def write_tee_times_csv(rows: list[dict], path: Path | None = None) -> Path:
    path = path or TEE_TIMES_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["event_id", "event", "name", "round", "tee_time", "start_hole"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse pasted golf tee sheet")
    ap.add_argument("--raw", default=str(RAW_TXT))
    ap.add_argument("--out", default=str(TEE_TIMES_CSV))
    ap.add_argument("--event-id", default="")
    ap.add_argument("--event", default="")
    ap.add_argument("--round", type=int, default=1, dest="round_no")
    args = ap.parse_args()
    raw_path = Path(args.raw)
    if not raw_path.exists():
        raise SystemExit(f"No tee sheet paste at {raw_path}")
    rows = parse_tee_sheet_text(
        raw_path.read_text(errors="replace"),
        event_id=args.event_id,
        event=args.event,
        default_round=args.round_no,
    )
    out = write_tee_times_csv(rows, Path(args.out))
    print(f"Parsed {len(rows)} tee-time rows -> {out}")


if __name__ == "__main__":
    main()

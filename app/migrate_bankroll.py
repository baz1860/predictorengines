"""One-time migration: fold existing per-engine bankroll/ledger files into the
unified suite store (app/bankroll_store.py).

Safe to run repeatedly: it refuses to overwrite an existing suite store unless
--force is passed, and it backs up everything it touches first.

    python3 -m app.migrate_bankroll            # dry run (report only)
    python3 -m app.migrate_bankroll --apply    # perform the migration
    python3 -m app.migrate_bankroll --apply --force   # overwrite existing suite store
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from . import bankroll_store as store

ROOT = store.ROOT

# (engine_id, sport, bankroll.json path, ledger.csv path)
SOURCES = [
    ("worldcup", "soccer", ROOT / "data" / "bankroll.json", ROOT / "data" / "ledger.csv"),
    ("cfb", "cfb", ROOT / "cfb" / "data" / "bankroll.json", ROOT / "cfb" / "data" / "ledger.csv"),
    ("golf", "golf", ROOT / "golf" / "data" / "bankroll.json", ROOT / "golf" / "data" / "ledger.csv"),
]


def _read_ledger(path: Path, engine: str, sport: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=store.COLS)
    df = pd.read_csv(path)
    df["engine"] = engine
    df["sport"] = sport
    for c in store.COLS:
        if c not in df.columns:
            df[c] = ""
    return df[store.COLS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="perform the migration (default: dry run)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing suite store")
    args = ap.parse_args()

    if store.LEDGER.exists() and not args.force:
        print(f"Suite store already exists at {store.LEDGER.name}. "
              "Re-run with --force to overwrite (a backup is made first).")
        if not args.apply:
            pass
        else:
            return

    frames, bankroll, peak = [], None, None
    print("Sources found:")
    for engine, sport, bj, lj in SOURCES:
        led = _read_ledger(lj, engine, sport)
        n = len(led)
        has_bank = bj.exists()
        print(f"  {engine:9s} ledger:{n:>3d} rows   bankroll.json:{'yes' if has_bank else 'no'}")
        if n:
            frames.append(led)
        # Take the bankroll/peak from the first engine that has one (soccer is
        # the live pot today; others have none). With multiple, they'd need a
        # manual decision, but in practice only one is populated.
        if has_bank and bankroll is None:
            d = json.loads(bj.read_text())
            bankroll = d.get("bankroll", store.START_BANKROLL)
            peak = d.get("peak", bankroll)

    merged = (pd.concat(frames, ignore_index=True) if frames
              else pd.DataFrame(columns=store.COLS))
    bankroll = bankroll if bankroll is not None else store.START_BANKROLL
    peak = peak if peak is not None else bankroll

    print(f"\nWould create: {store.LEDGER.name} ({len(merged)} rows), "
          f"{store.STATE.name} (bankroll £{bankroll:.2f}, peak £{peak:.2f})")

    if not args.apply:
        print("\nDry run — no files written. Re-run with --apply to migrate.")
        return

    store.DATA.mkdir(exist_ok=True)
    if store.LEDGER.exists():
        shutil.copy(store.LEDGER, store.LEDGER.with_suffix(".csv.premigrate.bak"))
    if store.STATE.exists():
        shutil.copy(store.STATE, store.STATE.with_suffix(".json.premigrate.bak"))

    merged.to_csv(store.LEDGER, index=False)
    store._save_state(bankroll, peak=peak, start=store.START_BANKROLL)
    print(f"\nMigrated. Suite bankroll £{bankroll:.2f}, {len(merged)} ledger rows.")
    print("Original per-engine files are left untouched as a backup.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Merge a freshly-fetched upstream results.csv into the local one without
losing locally-entered data.

The daily update fetches martj42/international_results and used to copy it over
data/results.csv wholesale. That blind overwrite can drop two things:
  * future / unplayed fixtures that only exist locally (scores = NA), which the
    predictor and edge engine need; and
  * manually-entered scores the upstream feed has not published yet.

Merge rule, keyed on (date, home_team, away_team):
  * if upstream has the match WITH a score  -> upstream wins (authoritative,
    and corrects any provisional/manual local score);
  * otherwise keep the local row             -> preserves local-only fixtures
    and manually-entered scores;
  * upstream-only rows are added as normal.

Usage:
  python3 merge_results.py UPSTREAM_CSV [LOCAL_CSV]   # default LOCAL = data/results.csv
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
KEY = ["date", "home_team", "away_team"]
NA_TOKENS = {"", "NA", "nan", "NaN", "None"}


def _read(path):
    # keep everything as literal strings so "NA" placeholders survive round-trips
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])


def _has_score(row):
    h, a = str(row.get("home_score", "")).strip(), str(row.get("away_score", "")).strip()
    return h not in NA_TOKENS and a not in NA_TOKENS


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: merge_results.py UPSTREAM_CSV [LOCAL_CSV]")
    upstream_path = Path(sys.argv[1])
    local_path = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data" / "results.csv"

    upstream = _read(upstream_path)
    if not local_path.exists():
        upstream.to_csv(local_path, index=False)
        print(f"   no local results.csv; wrote upstream ({len(upstream)} rows)")
        return

    local = _read(local_path)
    cols = list(upstream.columns)  # canonical column order

    # Upstream is kept verbatim (never drop/reorder its rows). We only index it.
    up = upstream.copy()
    keys_any = set(zip(*(up[k] for k in KEY)))
    scored_idx = {}            # key -> first upstream row index that already has a score
    blank_idx = {}             # key -> first upstream row index that is scoreless
    for i, r in enumerate(up.itertuples(index=False)):
        d = r._asdict(); key = tuple(d[k] for k in KEY)
        if _has_score(d):
            scored_idx.setdefault(key, i)
        else:
            blank_idx.setdefault(key, i)

    appended, patched = [], 0
    for r in local.itertuples(index=False):
        d = r._asdict(); key = tuple(d[k] for k in KEY)
        if key not in keys_any:
            appended.append(d)                          # local-only fixture or score
        elif key not in scored_idx and _has_score(d) and key in blank_idx:
            for c in ("home_score", "away_score"):      # fill scoreless upstream row
                up.iat[blank_idx[key], up.columns.get_loc(c)] = d[c]
            patched += 1
        # else: upstream already has a score (authoritative) -> leave it

    out = up if not appended else pd.concat([up, pd.DataFrame(appended)], ignore_index=True)
    for c in out.columns:                                # union of columns, upstream first
        if c not in cols:
            cols.append(c)
    out = out.reindex(columns=cols).sort_values("date", kind="stable")
    out.to_csv(local_path, index=False)
    appended_scores = sum(1 for d in appended if _has_score(d))
    print(f"   results.csv merged ({len(out)} rows; kept {len(upstream)} upstream, "
          f"appended {len(appended) - appended_scores} local fixture(s) + {appended_scores} "
          f"local score(s), patched {patched} scoreless upstream row(s))")


if __name__ == "__main__":
    main()

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

DATE-BOUNDARY RECONCILIATION (added August 2026)
------------------------------------------------
The exact key above cannot reconcile a fixture that the two sides date
differently. That is not hypothetical: on 8 August 2026 results.csv held BOTH

    2026-07-06,Argentina,Egypt,,,FIFA World Cup,Atlanta,...      (local, blank)
    2026-07-07,Argentina,Egypt,3,2,FIFA World Cup,Atlanta,...    (upstream, scored)

for one match — a North American evening kick-off, local date 6 July, UTC date
7 July. The local row was "local-only" under the exact key, so it was appended and
both survived. The blank row then satisfied `load_matches()`'s test for an upcoming
fixture, so a finished match was presented as forthcoming for a month.

Before appending a local-only row we now ask whether upstream already has the SAME
fixture within one day (same teams, same competition). If it does, the local row is
dropped as a duplicate. See international/identity.py for the classification rules;
ambiguous cases are never merged silently.

Usage:
  python3 merge_results.py UPSTREAM_CSV [LOCAL_CSV]   # default LOCAL = data/results.csv
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from international.identity import DATE_TOLERANCE_DAYS, signature  # noqa: E402

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

    # Signature -> upstream dates, for date-boundary reconciliation. A signature is
    # teams + competition with NO date, so a fixture dated differently by the two
    # sources still collides here.
    up_by_sig: dict[str, list[pd.Timestamp]] = {}
    for r in up.itertuples(index=False):
        d = r._asdict()
        sig = signature(d.get("home_team"), d.get("away_team"), d.get("tournament"))
        ts = pd.to_datetime(d.get("date"), errors="coerce")
        if pd.notna(ts):
            up_by_sig.setdefault(sig, []).append(ts)

    def _upstream_has_near(d: dict) -> pd.Timestamp | None:
        """Upstream date for the same fixture within the tolerance, if any."""
        sig = signature(d.get("home_team"), d.get("away_team"), d.get("tournament"))
        ts = pd.to_datetime(d.get("date"), errors="coerce")
        if pd.isna(ts):
            return None
        for other in up_by_sig.get(sig, ()):
            if other != ts and abs((other - ts).days) <= DATE_TOLERANCE_DAYS:
                return other
        return None

    appended, patched, reconciled = [], 0, []
    for r in local.itertuples(index=False):
        d = r._asdict(); key = tuple(d[k] for k in KEY)
        if key not in keys_any:
            near = _upstream_has_near(d)
            if near is not None and not _has_score(d):
                # Same fixture, different date, and OUR row has no score to lose.
                # Upstream's dated-and-scored row is authoritative; drop ours.
                reconciled.append((d.get("home_team"), d.get("away_team"),
                                   d.get("date"), str(near.date())))
                continue
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
    if reconciled:
        print(f"   reconciled {len(reconciled)} local fixture(s) against an upstream row "
              f"dated within {DATE_TOLERANCE_DAYS} day(s) (UTC/local date split):")
        for home, away, local_date, up_date in reconciled[:10]:
            print(f"     {home} v {away}: dropped local {local_date}, kept upstream {up_date}")


if __name__ == "__main__":
    main()

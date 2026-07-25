#!/usr/bin/env bash
# Backfill the non-UEFA club leagues (MLS, J1, K League, Brasileirão A/B,
# Liga MX, Saudi, CSL, Primera A, Botola, Libertadores, Sudamericana, CAF CL).
#
# Run from the repo root:   ./club_soccer/run_nonuefa_backfill.sh
#
# Safe to re-run. Fixture ids are deterministic, so a repeat merges rather
# than duplicates. Each slice is committed to fixtures.csv as it completes, so
# an interruption loses at most one slice.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 0. snapshot before we start =="
cp club_soccer/data/fixtures.csv "club_soccer/data/fixtures.csv.bak.pre_backfill.$(date +%Y%m%dT%H%M%S)"
python3 -c "
import pandas as pd
d=pd.read_csv('club_soccer/data/fixtures.csv',low_memory=False)
print(f'  starting from {len(d)} rows, {len(set(d[\"home\"].dropna())|set(d[\"away\"].dropna()))} identities')"

echo
echo "== 1. fetch in slices =="
# current=True is REQUIRED. current=False does not mean 'dry run' — it means
# 'do not merge, then overwrite the file with just this slice'. There is now a
# shrink guard that refuses that, but do not rely on it.
for range in \
  "2024-01-01:2024-06-30" \
  "2024-07-01:2024-12-31" \
  "2025-01-01:2025-06-30" \
  "2025-07-01:2025-12-31" \
  "2026-01-01:2026-07-31"
do
  from="${range%%:*}"; to="${range##*:}"
  echo "  -- $from .. $to"
  python3 -c "
from club_soccer import fetch as F
F.fetch_fixtures(current=True, date_from='$from', date_to='$to', enrich_stats=False)
" || echo "     slice failed — continuing (re-run the script to retry it)"
done

echo
echo "== 2. canonicalisation is enforced by the fixture write boundary =="

echo
echo "== 3. integrity check =="
python3 - <<'PY'
import collections, sys
import pandas as pd
sys.path.insert(0, '.')
from club_soccer import club_identity as CI
from club_soccer import fetch as F

df = pd.read_csv(CI.FIXTURES, low_memory=False)

# BSD occasionally emits home == away (corrupt rows like "Samsunspor v
# Samsunspor"); write_fixtures drops them, so route through it to clean any
# that predate the guard. This is known junk, not a merge error, so it is
# repaired rather than treated as a blocking failure.
selfm_before = int((df["home"] == df["away"]).sum())
if selfm_before:
    df = F.write_fixtures(df)
    df = pd.read_csv(CI.FIXTURES, low_memory=False)
    print(f"  cleaned {selfm_before} corrupt self-match row(s) from the provider")

names = set(df["home"].dropna()) | set(df["away"].dropna())
groups = collections.defaultdict(list)
for n in names:
    groups[CI._norm(n)].append(n)
dupes = {k: v for k, v in groups.items() if len(v) > 1}
selfm = int((df["home"] == df["away"]).sum())
key = (df["date"].astype(str).str[:10] + "|" + df["competition"].astype(str)
       + "|" + df["home"].astype(str) + "|" + df["away"].astype(str))

print(f"  rows                  : {len(df)}")
print(f"  identities            : {len(names)}")
print(f"  duplicate identities  : {dupes or 'none'}")
print(f"  self-matches          : {selfm}")
print(f"  duplicate match keys  : {int(key.duplicated().sum())}")

ab = df[(df["home"] == "Athletic Bilbao") | (df["away"] == "Athletic Bilbao")]
bad = [c for c in set(ab["competition"]) if "Brasileirao" in str(c)]
print(f"  Athletic Bilbao clean : {not bad}")

# dupes/contamination are merge errors and DO block; self-matches were just
# repaired above.
if dupes or selfm or bad:
    print("\n  PROBLEMS FOUND — do not refit. Restore the .bak.pre_backfill snapshot.")
    raise SystemExit(1)
PY

echo
echo "== 4. refit + validate =="
python3 -c "from club_soccer import model as M; M.save_params(M.fit())"
python3 -m club_soccer.validate --gate

echo
echo "== 5. review any new identity collisions =="
python3 -m club_soccer.identity_review
echo
echo "DONE."
echo "The report is read-only. Confirmed aliases must be reviewed and added"
echo "explicitly to club_soccer/data/club_alias_map.json."

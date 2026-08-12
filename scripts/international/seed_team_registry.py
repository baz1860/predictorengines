#!/usr/bin/env python3
"""Seed data/international/team_registry.csv (plan §3, Stage 1 deliverable).

Sources, in precedence order:
  1. NON_FIFA below — hand-classified, with a reason recorded for each entry.
  2. FIFA_EXTRA below — FIFA members absent from the confederation map in
     engines/worldcup/confederation_adj.py.
  3. The confederation map itself — 197 teams, all FIFA members.
  4. Everything else in results.csv -> status "unclassified" (dormant historical
     sides mostly; `international.registry.assert_scope_complete()` only requires
     ACTIVE teams to be classified).

This script is the audit trail. Re-running it regenerates the file deterministically;
edits should be made HERE, not in the CSV, so the reasoning stays reviewable.

Usage:
  python3 -m scripts.international.seed_team_registry           # dry run, prints summary
  python3 -m scripts.international.seed_team_registry --write   # write the CSV
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONF_SRC = ROOT / "engines" / "worldcup" / "confederation_adj.py"
RESULTS = ROOT / "data" / "results.csv"
OUT = ROOT / "data" / "international" / "team_registry.csv"

# FIFA members that the confederation map (built for World Cup fields) omits.
# Confederation assignments are current as of August 2026.
FIFA_EXTRA = {
    # CONCACAF — Caribbean members
    "Anguilla": "CONCACAF", "Antigua and Barbuda": "CONCACAF", "Aruba": "CONCACAF",
    "Bahamas": "CONCACAF", "British Virgin Islands": "CONCACAF",
    "Cayman Islands": "CONCACAF", "Dominica": "CONCACAF", "Grenada": "CONCACAF",
    "Guyana": "CONCACAF", "Montserrat": "CONCACAF", "Puerto Rico": "CONCACAF",
    "Saint Kitts and Nevis": "CONCACAF", "Saint Lucia": "CONCACAF",
    "Saint Vincent and the Grenadines": "CONCACAF",
    "Turks and Caicos Islands": "CONCACAF",
    "United States Virgin Islands": "CONCACAF",
    # AFC
    "Bhutan": "AFC", "Brunei": "AFC", "Laos": "AFC", "Macau": "AFC",
    "Taiwan": "AFC", "Timor-Leste": "AFC",
    # CAF
    "Djibouti": "CAF", "Liberia": "CAF",
    # UEFA
    "Israel": "UEFA",
}

# Not FIFA members. `reason` is carried into the CSV so the boundary is auditable.
NON_FIFA = {
    # Confederation associates — play in regional competition, no FIFA membership
    "Bonaire": ("CONCACAF", "CONCACAF associate, not a FIFA member"),
    "French Guiana": ("CONCACAF", "CONCACAF associate, not a FIFA member"),
    "Guadeloupe": ("CONCACAF", "CONCACAF associate, not a FIFA member"),
    "Martinique": ("CONCACAF", "CONCACAF associate, not a FIFA member"),
    "Saint Martin": ("CONCACAF", "CONCACAF associate, not a FIFA member"),
    "Sint Maarten": ("CONCACAF", "CONCACAF associate, not a FIFA member"),
    "Saint Barthélemy": ("CONCACAF", "not a FIFA member"),
    "Greenland": ("", "not a FIFA member"),
    "Northern Mariana Islands": ("AFC", "AFC member, not a FIFA member"),
    "Réunion": ("CAF", "CAF associate, not a FIFA member"),
    "Mayotte": ("", "French department, not a FIFA member"),
    "Zanzibar": ("CAF", "CAF associate, not a FIFA member"),
    # British Isles / European territories
    "Isle of Man": ("", "not a FIFA member"),
    "Jersey": ("", "not a FIFA member"),
    "Guernsey": ("", "not a FIFA member"),
    "Alderney": ("", "not a FIFA member"),
    "Gibraltar_dup": ("", "placeholder guard; real Gibraltar is a UEFA/FIFA member"),
    "Orkney": ("", "sub-national side"),
    "Shetland": ("", "sub-national side"),
    "Western Isles": ("", "sub-national side"),
    "Isle of Wight": ("", "sub-national side"),
    "Ynys Môn": ("", "sub-national side"),
    "Kernow": ("", "sub-national side"),
    "Åland Islands": ("", "sub-national side"),
    "Gozo": ("", "sub-national side"),
    "Menorca": ("", "sub-national side"),
    "Frøya": ("", "sub-national side"),
    "Hitra": ("", "sub-national side"),
    "Canton Ticino": ("", "sub-national side"),
    "Ticino": ("", "sub-national side"),
    "Raetia": ("", "sub-national side"),
    "Rouet-Provence": ("", "sub-national side"),
    "Padania": ("", "stateless / non-FIFA representative side"),
    "Two Sicilies": ("", "stateless / non-FIFA representative side"),
    "Galicia": ("", "sub-national side"),
    "Basque Country": ("", "sub-national side"),
    "Sápmi": ("", "stateless / non-FIFA representative side"),
    "Székely Land": ("", "stateless / non-FIFA representative side"),
    "Chameria": ("", "stateless / non-FIFA representative side"),
    "Northern Cyprus": ("", "not a FIFA member"),
    "Falkland Islands": ("", "not a FIFA member"),
    "Saint Helena": ("", "not a FIFA member"),
    # Asia / Pacific / other non-members
    "Tibet": ("", "stateless / non-FIFA representative side"),
    "East Turkestan": ("", "stateless / non-FIFA representative side"),
    "Tamil Eelam": ("", "stateless / non-FIFA representative side"),
    "West Papua": ("", "stateless / non-FIFA representative side"),
    "Hmong": ("", "stateless / non-FIFA representative side"),
    "Tuvalu": ("OFC", "OFC associate, not a FIFA member"),
    "Marshall Islands": ("", "not a FIFA member"),
    "Kiribati": ("OFC", "OFC associate, not a FIFA member"),
    "Niue": ("", "not a FIFA member"),
    "Micronesia": ("", "not a FIFA member"),
}
NON_FIFA.pop("Gibraltar_dup")  # guard entry, never emitted

# Defunct member associations. They WERE FIFA members, so a 1990 fixture is in
# scope; they are not members today, so they must not inflate the current count.
# This is the case effective-dating exists for.
MEMBER_TO = {
    "Czechoslovakia": "1993-12-31",   # last match Nov 1993; succeeded by CZE/SVK
    "Yugoslavia": "1992-12-31",       # SFR Yugoslavia; last match Mar 1992
}

# Confederation transfers. confederation_adj.py was built for World Cup fields and
# records the historical assignment. Only *current* confederation is stored here;
# effective-dating the confederation itself is deferred (it affects the
# confederation strength adjustment, not scope) and is recorded as a known gap in
# international/registry.py.
CONFEDERATION_OVERRIDE = {
    "Kazakhstan": ("UEFA", "transferred AFC -> UEFA in 2002"),
    "Australia": ("AFC", "transferred OFC -> AFC in 2006"),
}

# FIFA accession dates. Only associations that joined recently enough to affect
# fixtures in our modelling window (post-1990) are listed — for everyone else the
# open interval is correct, because they were members throughout.
#
# Why this matters: without a join date, a 1994 fixture involving a state that did
# not yet exist is silently judged against today's membership. Dating the joiners
# is the difference between a scope rule and a guess.
MEMBER_FROM = {
    # Post-Soviet and post-Yugoslav associations
    "Armenia": "1992-01-01", "Azerbaijan": "1994-01-01", "Belarus": "1992-01-01",
    "Estonia": "1992-01-01", "Georgia": "1992-01-01", "Kazakhstan": "1994-01-01",
    "Kyrgyzstan": "1994-01-01", "Latvia": "1992-01-01", "Lithuania": "1992-01-01",
    "Moldova": "1994-01-01", "Russia": "1992-01-01", "Tajikistan": "1994-01-01",
    "Turkmenistan": "1994-01-01", "Ukraine": "1992-01-01", "Uzbekistan": "1994-01-01",
    "Croatia": "1992-07-03", "Slovenia": "1992-07-03",
    "Bosnia and Herzegovina": "1996-07-01", "North Macedonia": "1994-06-01",
    "Serbia": "2006-06-01", "Montenegro": "2007-05-31", "Kosovo": "2016-05-13",
    "Czech Republic": "1994-01-01", "Slovakia": "1994-01-01",
    # Other post-1990 joiners
    "Eritrea": "1998-06-01", "Namibia": "1992-07-08", "South Africa": "1992-07-03",
    "Palestine": "1998-06-08", "Timor-Leste": "2005-09-12",
    "Gibraltar": "2016-05-13", "South Sudan": "2012-05-25",
    "Comoros": "2005-09-12", "Bhutan": "2000-01-01", "Macau": "1976-01-01",
    "Andorra": "1996-07-01", "Cook Islands": "1994-01-01",
    "Samoa": "1986-01-01", "Tonga": "1994-01-01", "Vanuatu": "1988-01-01",
    "American Samoa": "1998-06-01", "Anguilla": "1996-07-01",
    "Turks and Caicos Islands": "1998-06-01", "Montserrat": "1996-07-01",
    "Cape Verde": "1986-01-01", "São Tomé and Príncipe": "1986-01-01",
    "Equatorial Guinea": "1986-01-01", "Djibouti": "1994-01-01",
}

COLUMNS = ["team", "status", "confederation", "member_from", "member_to", "reason"]


def confederation_map() -> dict[str, str]:
    src = CONF_SRC.read_text()
    return dict(re.findall(
        r'"([^"]+)":\s*"(UEFA|CONMEBOL|CAF|AFC|CONCACAF|OFC)"', src))


def build() -> pd.DataFrame:
    df = pd.read_csv(RESULTS, usecols=["home_team", "away_team"])
    teams = sorted(set(df.home_team.dropna()) | set(df.away_team.dropna()))

    conf = confederation_map()
    rows = []
    for name in teams:
        if name in NON_FIFA:
            c, reason = NON_FIFA[name]
            rows.append((name, "non_fifa", c, "", "", reason))
        elif name in FIFA_EXTRA:
            rows.append((name, "fifa", FIFA_EXTRA[name],
                         MEMBER_FROM.get(name, ""), MEMBER_TO.get(name, ""),
                         "FIFA member; absent from confederation_adj map"))
        elif name in conf:
            reason = ("former FIFA member, membership ended"
                      if name in MEMBER_TO else
                      "FIFA member via confederation_adj map")
            c = conf[name]
            if name in CONFEDERATION_OVERRIDE:
                c, reason = CONFEDERATION_OVERRIDE[name]
            rows.append((name, "fifa", c, MEMBER_FROM.get(name, ""),
                         MEMBER_TO.get(name, ""), reason))
        else:
            rows.append((name, "unclassified", "", "", "",
                         "not yet reviewed — dormant unless listed as active"))
    return pd.DataFrame(rows, columns=COLUMNS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    reg = build()
    print(reg.status.value_counts().to_string())

    from international import registry as R
    active = R.active_teams()
    unclassified_active = sorted(
        reg[(reg.status == "unclassified") & (reg.team.isin(active))].team)
    print(f"\nactive teams: {len(active)}")
    print(f"unclassified AND active: {len(unclassified_active)}")
    for name in unclassified_active:
        print(f"  {name}")

    if a.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        reg.to_csv(OUT, index=False)
        print(f"\nwrote {OUT.relative_to(ROOT)} ({len(reg)} rows)")
    else:
        print("\n(dry run — pass --write to save)")


if __name__ == "__main__":
    main()

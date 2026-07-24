#!/usr/bin/env python3
"""Human review workflow for unresolved club identities.

Why this exists
---------------
The automatic matchers do the bulk of the work but leave a residue that no
heuristic can safely clear:

  * "Kuopion Palloseura" and "KuPS" are the same club with ZERO string
    similarity. No affinity threshold will ever find this.
  * "AEK Larnaca" (Cyprus) and "AEK" (Athens) look nearly identical and are
    different clubs.

Between those two cases sits a band where the only reliable signal is someone
who knows the football. Guessing there is expensive: a wrong merge welds two
clubs' rating histories together and nothing errors afterwards.

So: this module exports every unresolved Europe-only identity to a CSV, you
fill in a verdict column, and it reads the verdicts back and applies them.

Workflow
--------
    python3 -m club_soccer.identity_review --export
        -> writes data/identity_review.csv

    ... open it, fill in the VERDICT column ...

    python3 -m club_soccer.identity_review --apply
        -> applies your verdicts to fixtures.csv and records them permanently

VERDICT values:
    (blank)  undecided — ignored, safe to leave
    y        merge into the suggested_match
    n        NOT the same club — recorded so it is never re-proposed
    <name>   merge into this club instead (type the domestic spelling)

Your decisions persist in data/identity_verdicts.json, so a re-export never
asks twice and a re-ingest cannot undo them.

Why Europe-only clubs specifically
----------------------------------
A club that appears only in UEFA competitions is either (a) from a league we
cannot source — correct, nothing to do — or (b) the same club as one already in
a domestic league under a different spelling, in which case its rating is split
and both halves are wrong. Case (b) is what this file is for, and these clubs
are all actively priced, so the split matters.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import club_identity as CI
from .competitions import get as comp_get

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
REVIEW_CSV = DATA / "identity_review.csv"
VERDICTS = DATA / "identity_verdicts.json"

FIELDS = ["VERDICT", "claude_read", "europe_only_name", "suggested_match",
          "suggested_league", "confidence", "n_matches", "seasons", "why", "notes"]

# Countries with no sourceable domestic league (uefa_registry). A club from one
# of these SHOULD be Europe-only — there is nothing for it to merge into, and a
# suggested domestic match is almost certainly a different club in a different
# country. Used only to annotate my read, never to decide.
UNSOURCED_HINTS = (
    "praha", "plzen", "slavia", "sparta praha",          # Czechia
    "zagreb", "rijeka", "hajduk",                        # Croatia
    "crvena", "partizan", "beograd",                     # Serbia
    "kyiv", "kiev", "shakhtar", "donetsk", "dynamo kyiv",  # Ukraine
    "maccabi", "hapoel", "beer sheva",                   # Israel
    "larnaca", "nicosia", "apoel", "pafos", "omonia",    # Cyprus
    "ferencvaros", "puskas",                             # Hungary
    "slovan", "trnava", "bratislava",                    # Slovakia
    "qarabag", "neftci",                                 # Azerbaijan
    "ludogorets", "cska sofia",                          # Bulgaria
    "celje", "maribor", "olimpija",                      # Slovenia
    "astana", "kairat", "almaty",                        # Kazakhstan
    "minsk", "bate",                                     # Belarus
    "reykjavik", "vikingur", "breidablik",               # Iceland
    "lincoln red imps",                                  # Gibraltar
    "the new saints", "connah",                          # Wales
    "larne", "linfield", "crusaders",                    # Northern Ireland
    "hamrun", "birkirkara", "floriana",                  # Malta
    "escaldes", "santa coloma",                          # Andorra
    "noah", "pyunik", "ararat",                          # Armenia
    "iberia", "dinamo tbilisi",                          # Georgia
    "milsami", "sheriff", "petrocub",                     # Moldova
    "shkendija", "struga",                               # North Macedonia
    "borac", "zrinjski", "sarajevo",                     # Bosnia
    "riga", "rfs", "valmiera",                           # Latvia
    "zalgiris", "kaunas",                                # Lithuania
    "flora", "levadia", "paide",                         # Estonia
    "differdange", "dudelange",                          # Luxembourg
    "buducnost", "sutjeska",                             # Montenegro
    "egnatia", "tirana", "partizani",                    # Albania
    "klaksvik", "torshavn",                              # Faroes
    "drita", "ballkani", "prishtina",                    # Kosovo
    "virtus", "tre fiori", "folgore",                    # San Marino
)


def _looks_unsourced(name: str) -> bool:
    low = CI._norm(name)
    return any(hint in low for hint in UNSOURCED_HINTS)


def _load_verdicts() -> dict:
    if VERDICTS.exists():
        try:
            return json.loads(VERDICTS.read_text())
        except Exception:
            return {}
    return {}


def _save_verdicts(doc: dict) -> None:
    DATA.mkdir(exist_ok=True)
    VERDICTS.write_text(json.dumps(doc, indent=2, ensure_ascii=False))


def domestic_index(df: pd.DataFrame) -> dict[str, str]:
    """Domestic club -> its league."""
    out: dict[str, str] = {}
    for comp, home, away in zip(df["competition"], df["home"], df["away"]):
        c = comp_get(comp)
        if not c or c.kind != "league":
            continue
        out.setdefault(home, comp)
        out.setdefault(away, comp)
    return out


def suggest(name: str, domestic: dict[str, str],
            euro_dates: set[str], all_dates: dict[str, set[str]]
            ) -> tuple[str, float, str, int]:
    """Best domestic candidate for a Europe-only club.

    Same-day collisions are counted and shown, NOT used to exclude a candidate.
    Measured on known pairs, a small number of collisions indicates the two
    sources describe the same club and disagree about a date; genuinely
    different clubs show zero. Using it as a veto suppressed the correct
    suggestions and kept none of the wrong ones.
    """
    best_name, best_score, best_why, best_clash = "", 0.0, "", 0
    core = CI._core(name)
    for cand in domestic:
        ok, why = CI._affinity(name, cand)
        if not ok:
            continue
        score = difflib.SequenceMatcher(None, core, CI._core(cand)).ratio()
        if score > best_score:
            best_name, best_score, best_why = cand, score, why
            best_clash = len(euro_dates & all_dates.get(cand, set()))
    return best_name, round(best_score, 3), best_why, best_clash


def build_rows() -> list[dict]:
    df = pd.read_csv(CI.FIXTURES, low_memory=False)
    euro = CI.europe_only_teams(df)
    domestic = domestic_index(df)
    verdicts = _load_verdicts()

    dates: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    for date, home, away in zip(df["date"], df["home"], df["away"]):
        day = str(date)[:10]
        for team in (home, away):
            dates.setdefault(team, set()).add(day)
            counts[team] = counts.get(team, 0) + 1

    # Europe-only clubs that duplicate EACH OTHER. A domestic-only search
    # cannot see these: "Dynamo Kyiv"/"Dinamo Kiev" and "Viktoria Plzeň"/"FC
    # Viktoria Plzeň" are each one club split in two, and since neither
    # country has a sourceable league there is no domestic name to anchor on.
    euro_names = sorted(euro)
    h2h = CI._head_to_head(df)
    euro_pairs: dict[str, tuple[str, float]] = {}
    for i, a in enumerate(euro_names):
        for b in euro_names[i + 1:]:
            # Two clubs that have played each other are not one club.
            if frozenset((a, b)) in h2h:
                continue
            ok, _why = CI._affinity(a, b)
            if not ok:
                continue
            score = difflib.SequenceMatcher(None, CI._core(a), CI._core(b)).ratio()
            if score < 0.55:
                continue
            # Keep the better-evidenced spelling as the target.
            src, dst = (b, a) if counts.get(a, 0) >= counts.get(b, 0) else (a, b)
            prev = euro_pairs.get(src)
            if prev is None or score > prev[1]:
                euro_pairs[src] = (dst, round(score, 3))

    rows = []
    for name, seasons in euro.items():
        if name in verdicts:
            continue                      # already decided, never ask twice
        cand, score, why, clash = suggest(name, domestic, dates.get(name, set()), dates)
        league = domestic.get(cand, "") if cand else ""
        unsourced = _looks_unsourced(name)

        # Prefer a Europe-only twin over a weak domestic guess.
        twin = euro_pairs.get(name)
        if twin and (not cand or twin[1] > score):
            cand, score, league, why = twin[0], twin[1], "(Europe-only twin)", "euro-twin"

        # External reference: settles the country question for clubs our own
        # data cannot place. Where it speaks, it is more reliable than any
        # name heuristic — so it overrides my read rather than colouring it.
        registry_verdict = None
        if cand:
            try:
                from .club_registry import same_club_possible, country_of
                possible, why = same_club_possible(name, cand)
                if not possible:
                    registry_verdict = why
            except Exception:
                pass
        reg_country = None
        try:
            from .club_registry import country_of as _country_of
            reg_country = _country_of(name)
        except Exception:
            pass

        if registry_verdict:
            read = "NO — club registry says different clubs"
            note = (f"openfootball/clubs: {registry_verdict}. "
                    f"Overriding the name match — the reference is more "
                    f"reliable here than string similarity.")
            rows.append({
                "VERDICT": "", "claude_read": read,
                "europe_only_name": name, "suggested_match": cand,
                "suggested_league": league, "confidence": score,
                "n_matches": counts.get(name, 0),
                "seasons": ",".join(str(s) for s in sorted(seasons)),
                "why": why, "notes": note,
            })
            continue

        if cand and league == "(Europe-only twin)":
            # A shared CITY token is the trap here: "Slavia Praha"/"Sparta
            # Praha", "CSKA Sofia"/"Levski Sofia" and "Omonia Nikosia"/"APOEL
            # Nikosia" are all pairs of bitter rivals, not spellings of one
            # club. Above ~0.80 the match is carried by the club name itself;
            # below that it is often carried by the city alone.
            if score >= 0.80:
                read = "LIKELY YES — same club, two European spellings"
                note = ("neither has domestic data, so this is the only way to "
                        "join them")
            else:
                read = "CHECK — could be two clubs from the same city"
                note = ("matched partly on a shared city name. Same-city rivals "
                        "look identical to the matcher — please confirm")
        elif cand and score >= 0.90:
            read = "LIKELY YES"
            note = "near-exact name match"
        elif cand and unsourced:
            read = "LIKELY NO — different country"
            note = (f"this club is probably from an association we cannot source, "
                    f"so {cand!r} is likely a different club entirely")
        elif cand and score >= 0.65:
            read = "CHECK — plausible but not certain"
            note = "worth a look; the two names are similar but not conclusive"
        elif cand:
            read = "LIKELY NO"
            note = "weak name similarity — probably a coincidence"
        elif unsourced or reg_country:
            read = "EXPECTED — no domestic league available"
            note = ("its league is not sourceable, so Europe-only is correct. "
                    "Leave blank.")
            if reg_country:
                note = (f"club registry places this in {reg_country}, whose "
                        "league we do not carry — Europe-only is correct. "
                        "Leave blank.")
        else:
            read = "UNKNOWN — needs your eye"
            note = ("no candidate found. If it IS in one of our leagues under "
                    "another name, type that name in VERDICT.")
        if clash and cand:
            note += f" [{clash} same-day clash(es) — usually indicates the same club]"

        rows.append({
            "VERDICT": "",
            "claude_read": read,
            "europe_only_name": name,
            "suggested_match": cand,
            "suggested_league": league,
            "confidence": score if cand else "",
            "n_matches": counts.get(name, 0),
            "seasons": ",".join(str(s) for s in sorted(seasons)),
            "why": why,
            "notes": note,
        })
    # Most-priced clubs first — those are where a split rating costs most.
    rows.sort(key=lambda r: (-int(r["n_matches"]), r["europe_only_name"]))
    return rows


def export() -> Path:
    rows = build_rows()
    DATA.mkdir(exist_ok=True)
    with REVIEW_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    decided = len(_load_verdicts())
    with_sugg = sum(1 for r in rows if r["suggested_match"])
    print(f"wrote {REVIEW_CSV}")
    print(f"  {len(rows)} club(s) awaiting a verdict "
          f"({with_sugg} with a suggested match, {len(rows) - with_sugg} without)")
    if decided:
        print(f"  {decided} previously decided — not re-listed")
    print("\nFill in the VERDICT column:")
    print("  y       = yes, merge into suggested_match")
    print("  n       = no, different club")
    print("  <name>  = merge into this club instead")
    print(f"\nthen: python3 -m club_soccer.identity_review --apply")
    return REVIEW_CSV


def read_review() -> tuple[dict[str, str], list[str], list[str]]:
    """(merges, rejections, problems) from the edited CSV."""
    if not REVIEW_CSV.exists():
        raise SystemExit(f"{REVIEW_CSV.name} not found — run --export first.")
    df = pd.read_csv(CI.FIXTURES, low_memory=False)
    known = set(df["home"].dropna()) | set(df["away"].dropna())

    merges: dict[str, str] = {}
    rejections: list[str] = []
    problems: list[str] = []
    with REVIEW_CSV.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            verdict = (row.get("VERDICT") or "").strip()
            name = (row.get("europe_only_name") or "").strip()
            if not verdict or not name:
                continue
            low = verdict.lower()
            if low in ("n", "no"):
                rejections.append(name)
                continue
            if low in ("y", "yes"):
                target = (row.get("suggested_match") or "").strip()
                if not target:
                    problems.append(f"{name}: 'y' but no suggested_match to merge into")
                    continue
            else:
                target = verdict
            if target not in known:
                problems.append(
                    f"{name}: target {target!r} is not a club in fixtures.csv "
                    "(check the spelling against the suggested_league roster)")
                continue
            if target == name:
                problems.append(f"{name}: cannot merge into itself")
                continue
            merges[name] = target
    return merges, rejections, problems


def check_consistency() -> list[str]:
    """Verdicts, alias map and fixtures.csv must agree.

    Three stores can drift apart and nothing errors when they do:
      * identity_verdicts.json — what was decided
      * club_alias_map.json    — what future fetches will apply
      * fixtures.csv           — what is already merged

    A club marked `distinct` that still has an alias entry is the dangerous
    case: the reversal looks applied and is silently undone on the next fetch.
    """
    problems: list[str] = []
    verdicts = _load_verdicts()
    alias: dict[str, str] = {}
    if CI.ALIAS_MAP.exists():
        try:
            alias = json.loads(CI.ALIAS_MAP.read_text()).get("alias", {})
        except Exception:
            problems.append("club_alias_map.json is unreadable")

    for name, v in verdicts.items():
        if v["decision"] == "distinct" and name in alias:
            problems.append(
                f"{name!r} is marked distinct but the alias map still merges it "
                f"into {alias[name]!r} — the next fetch would undo the reversal")
        if v["decision"] == "merge" and alias.get(name) != v.get("target"):
            problems.append(
                f"{name!r} is marked merged into {v.get('target')!r} but the "
                f"alias map says {alias.get(name)!r}")

    if CI.FIXTURES.exists():
        df = pd.read_csv(CI.FIXTURES, low_memory=False)
        names = set(df["home"].dropna()) | set(df["away"].dropna())
        for name, v in verdicts.items():
            if v["decision"] == "merge" and name in names:
                problems.append(
                    f"{name!r} is marked merged but still present in fixtures.csv")
    return problems


def apply(dry_run: bool = False) -> None:
    from .identities import dedupe_fixtures

    merges, rejections, problems = read_review()
    if problems:
        print("PROBLEMS — nothing applied:")
        for p in problems:
            print(f"  - {p}")
        print("\nFix the CSV and re-run. Refusing to guess what you meant.")
        return
    if not merges and not rejections:
        print("no verdicts found in the CSV — nothing to do")
        return

    print(f"{len(merges)} merge(s), {len(rejections)} rejection(s)")
    for src, dst in sorted(merges.items()):
        print(f"  merge   {src!r} -> {dst!r}")
    for name in sorted(rejections):
        print(f"  reject  {name!r} (recorded, will not be re-proposed)")
    if dry_run:
        print("\nDRY RUN — nothing written")
        return

    verdicts = _load_verdicts()
    stamp = datetime.now(timezone.utc).isoformat()

    # A verdict flipped from merge -> distinct must UNDO its alias, not just
    # record the change of mind. Without this the alias map keeps the old merge
    # and every future fetch re-applies it, so the reversal appears to work and
    # silently does not. (Found when FK Žalgiris was correctly reversed to
    # 'distinct' and stayed merged anyway.)
    reversals = [n for n in rejections
                 if verdicts.get(n, {}).get("decision") == "merge"]
    if reversals:
        doc = json.loads(CI.ALIAS_MAP.read_text()) if CI.ALIAS_MAP.exists() else {"alias": {}}
        alias = doc.get("alias", {})
        for name in reversals:
            was = alias.pop(name, None)
            print(f"  REVERSED {name!r} (was merged into {was!r}) — alias removed")
        doc["alias"] = dict(sorted(alias.items()))
        CI.ALIAS_MAP.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
        CI.reload_resolver()
        print("\n  NOTE: fixtures.csv still holds the completed merge for the "
              "reversed club(s). The alias is gone so it will not recur, but the "
              "already-merged rows must be separated by hand or by re-ingesting.")

    for src, dst in merges.items():
        verdicts[src] = {"decision": "merge", "target": dst, "at": stamp}
    for name in rejections:
        verdicts[name] = {"decision": "distinct", "at": stamp}
    _save_verdicts(verdicts)

    if merges:
        df = pd.read_csv(CI.FIXTURES, low_memory=False)
        # Timestamped, never overwritten. A fixed name meant a second --apply
        # clobbered the pre-merge snapshot with an already-merged one, so the
        # only clean restore point was destroyed by the act of re-running.
        backup = CI.FIXTURES.with_suffix(
            f".csv.bak.pre_review.{datetime.now(timezone.utc):%Y%m%dT%H%M%S}")
        df.to_csv(backup, index=False)
        before = len(set(df["home"]) | set(df["away"]))
        merged = dedupe_fixtures(CI.apply_alias_map(df, merges))
        merged.to_csv(CI.FIXTURES, index=False)
        after = len(set(merged["home"]) | set(merged["away"]))
        print(f"\nbackup -> {backup.name}")
        print(f"rows {len(df)} -> {len(merged)};  identities {before} -> {after}")

        # Fold into the cumulative alias map so the DAILY FETCH keeps applying
        # these. Without this the next fetch re-creates every split you just
        # resolved, since canonical_name() reads that file.
        doc = json.loads(CI.ALIAS_MAP.read_text()) if CI.ALIAS_MAP.exists() else {"alias": {}}
        alias = doc.get("alias", {})
        alias.update(merges)
        doc["alias"] = dict(sorted(alias.items()))
        doc["human_verdicts_merged_at"] = stamp
        CI.ALIAS_MAP.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
        CI.reload_resolver()
        print(f"alias map now {len(alias)} entries (applied on every future fetch)")

    print("\nRefit to pick up the merged ratings:")
    print("  python3 -c \"from club_soccer import model as M; M.save_params(M.fit())\"")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", action="store_true", help="write the review CSV")
    ap.add_argument("--apply", action="store_true", help="apply your verdicts")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true", help="show decisions so far")
    args = ap.parse_args()

    if args.status:
        verdicts = _load_verdicts()
        merges = {k: v for k, v in verdicts.items() if v["decision"] == "merge"}
        distinct = {k: v for k, v in verdicts.items() if v["decision"] == "distinct"}
        print(f"decisions recorded: {len(verdicts)}")
        print(f"  merged   : {len(merges)}")
        print(f"  distinct : {len(distinct)}")
        for src, v in sorted(merges.items()):
            print(f"    {src!r} -> {v['target']!r}")

        problems = check_consistency()
        print(f"\nconsistency: {'OK' if not problems else 'PROBLEMS'}")
        for p in problems:
            print(f"  - {p}")
        return
    if args.apply:
        apply(dry_run=args.dry_run)
        return
    export()


if __name__ == "__main__":
    main()

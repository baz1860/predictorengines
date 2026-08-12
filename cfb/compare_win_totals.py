#!/usr/bin/env python3
"""Compare model projected win totals vs sportsbook lines.

Merges data/projected_win_totals_2026.csv (from win_totals.py) with
data/win_totals_lines_2026.csv (from fetch_win_total_lines.py) and shows:
  - Model projection vs book line
  - Model P(over) at the book's line
  - Recommended bet (if |gap| >= threshold)

Usage:
  python3 compare_win_totals.py              # threshold 0.5 wins
  python3 compare_win_totals.py --min 1.0    # bigger gap only
  python3 compare_win_totals.py --raw        # inspect raw JSON structure

Book names are resolved through the reviewed identity registry
(cfb/identity.py) — canonical CFBD names or explicitly reviewed aliases only.
Unresolved spellings are reported, never guessed onto the closest team.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import identity as IDENTITY

HERE = os.path.dirname(os.path.abspath(__file__))


def resolve_line_teams(lines, season):
    """Map provider team names to canonical CFBD names via the registry.

    The previous fuzzy matcher stripped the word "state", so 'Ohio State' and
    'Ohio' collapsed to the same key — exactly the silent misattribution the
    identity module exists to prevent. Unresolved names are returned for
    reporting instead of being attached to a plausible-looking team.
    """
    out = lines.copy()
    canonical, unresolved = [], []
    for name in out["team"]:
        match = IDENTITY.resolve(name, season, provider="the-odds-api")
        if match:
            canonical.append(match["canonical"])
        else:
            canonical.append(None)
            unresolved.append(name)
    out["team"] = canonical
    resolved = out[out["team"].notna()].copy()
    if resolved.empty:
        return resolved, unresolved
    # Several provider spellings (and several books) can resolve to the same
    # canonical team. Consolidate to ONE row per team — otherwise each
    # duplicate is priced and flagged as its own bet on the same market.
    agg = {"line": "median"}
    for col in ("over_odds", "under_odds"):
        if col in resolved:
            agg[col] = _median_american
    if "books" in resolved:
        agg["books"] = "sum"
    consolidated = (resolved.groupby("team", as_index=False)
                    .agg({k: v for k, v in agg.items() if k in resolved}))
    if "books" in consolidated:
        consolidated["books"] = consolidated["books"].round().astype(int)
    return consolidated, unresolved


def merge(model, lines):
    """Exact merge on canonical names (resolution happens upstream)."""
    return model.merge(lines, on="team", how="left")


def p_over_line(exp_wins, sd, line):
    """P(wins > line) using normal approximation (continuity correction)."""
    return float(1.0 - norm.cdf(line + 0.5, loc=exp_wins, scale=max(sd, 0.1)))


def american_to_implied(odds):
    """American odds -> implied probability (no vig removal).

    Values with |odds| < 100 are not valid American prices (they arise from
    naively averaging across the +/- boundary) and fall back to -110.
    """
    odds = float(odds)
    if abs(odds) < 100.0:
        odds = -110.0
    return -odds / (-odds + 100) if odds < 0 else 100 / (odds + 100)


def implied_to_american(p):
    """Implied probability -> American odds (inverse of american_to_implied)."""
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return -100.0 * p / (1.0 - p) if p > 0.5 else 100.0 * (1.0 - p) / p


def _median_american(values):
    """Median of American prices, taken in probability space.

    Medianing raw American odds is invalid: -110 and +100 average to -5, which
    is not a price at all and yields a nonsense edge.
    """
    probs = [american_to_implied(v) for v in values if pd.notna(v)]
    if not probs:
        return -110
    return int(round(implied_to_american(float(np.median(probs)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--min", type=float, default=0.5, help="Min gap (model-line) to flag a bet")
    ap.add_argument("--raw", action="store_true", help="Dump raw JSON structure and exit")
    args = ap.parse_args()

    if args.raw:
        raw = os.path.join(HERE, "data", f"win_totals_raw_{args.year}.json")
        with open(raw) as f:
            data = json.load(f)
        print(f"{len(data)} events in raw JSON\n")
        if data:
            print("First event keys:", list(data[0].keys()))
            if data[0].get("bookmakers"):
                bm = data[0]["bookmakers"][0]
                print("First bookmaker:", bm["title"])
                if bm.get("markets"):
                    mkt = bm["markets"][0]
                    print("First market key:", mkt["key"])
                    print("First 3 outcomes:", mkt["outcomes"][:3])
        return

    # win_totals.py writes beside the package, not under data/. This script
    # used to look only in data/, so it could never find the projections.
    candidates = [os.path.join(HERE, f"projected_win_totals_{args.year}.csv"),
                  os.path.join(HERE, "data", f"projected_win_totals_{args.year}.csv")]
    model_path = next((p for p in candidates if os.path.exists(p)), None)
    lines_path = os.path.join(HERE, "data", f"win_totals_lines_{args.year}.csv")

    if model_path is None:
        raise SystemExit(
            f"No projected_win_totals_{args.year}.csv in {HERE} or {HERE}/data "
            f"— run: python3 -m cfb.win_totals --year {args.year}")
    if not os.path.exists(lines_path):
        raise SystemExit(
            f"Missing {lines_path} — run: python3 -m cfb.fetch_win_total_lines "
            f"--year {args.year} (or --template to enter book lines by hand)")

    model = pd.read_csv(model_path)
    lines = pd.read_csv(lines_path)
    # A hand-filled template has blank lines for teams not yet entered.
    lines = lines[pd.to_numeric(lines["line"], errors="coerce").notna()].copy()
    if lines.empty:
        raise SystemExit(f"{lines_path} has no filled-in `line` values.")

    lines, unresolved = resolve_line_teams(lines, args.year)
    if unresolved:
        print(f"{len(unresolved)} book name(s) not in the reviewed identity "
              f"registry — excluded, not guessed:")
        print("  " + ", ".join(unresolved[:10])
              + ("..." if len(unresolved) > 10 else ""))
        print("  Add reviewed aliases to cfb/data/team_aliases.json to include them.")

    df = merge(model, lines)
    matched = df[df["line"].notna()].copy()
    unmatched = df[df["line"].isna()]["team"].tolist()
    if matched.empty:
        raise SystemExit("No model team matched a resolved book line.")

    # Recompute P(over book line) using normal approximation with our sd
    matched["p_over_book"] = matched.apply(
        lambda r: p_over_line(r["exp_wins"], r["sd"], r["line"]), axis=1)
    matched["gap"] = (matched["exp_wins"] - matched["line"]).round(2)

    # Implied probability from book over odds (for edge calc)
    matched["impl_over"] = matched["over_odds"].apply(american_to_implied)
    matched["impl_under"] = matched["under_odds"].apply(american_to_implied)
    matched["edge_over"] = (matched["p_over_book"] - matched["impl_over"]).round(3)
    matched["edge_under"] = ((1 - matched["p_over_book"]) - matched["impl_under"]).round(3)

    # Print full table
    pd.set_option("display.width", 160)
    cols = ["team", "conference", "exp_wins", "line", "gap", "p_over_book",
            "over_odds", "under_odds", "edge_over", "edge_under"]
    print(f"\n{'Model vs. Book Win Totals':=^90}")
    print(f"{'team':<28} {'conf':<14} {'model':>6} {'line':>5} {'gap':>5} "
          f"{'P(O)':>6} {'O odds':>7} {'U odds':>7} {'Oedge':>7} {'Uedge':>7}")
    print("-" * 90)
    for _, r in matched.sort_values("gap", ascending=False).iterrows():
        flag = ""
        if r["gap"] >= args.min and r["edge_over"] > 0:
            flag = " ← OVER"
        elif r["gap"] <= -args.min and r["edge_under"] > 0:
            flag = " ← UNDER"
        print(f"{r['team']:<28} {str(r.get('conference','')):<14} {r['exp_wins']:>6.1f} "
              f"{r['line']:>5.1f} {r['gap']:>+5.1f} {r['p_over_book']:>6.1%} "
              f"{int(r['over_odds']):>+7d} {int(r['under_odds']):>+7d} "
              f"{r['edge_over']:>+7.1%} {r['edge_under']:>+7.1%}{flag}")

    print(f"\n{matched[matched['gap'].abs() >= args.min]['team'].count()} teams with gap ≥ {args.min} wins")

    bets = matched[((matched["gap"] >= args.min) & (matched["edge_over"] > 0)) |
                   ((matched["gap"] <= -args.min) & (matched["edge_under"] > 0))]
    if not bets.empty:
        print(f"\n{'Flagged bets':=^60}")
        for _, r in bets.iterrows():
            if r["gap"] >= args.min and r["edge_over"] > 0:
                print(f"  OVER  {r['team']:<28}  model {r['exp_wins']:.1f} > line {r['line']:.1f}  "
                      f"P(O)={r['p_over_book']:.1%}  edge={r['edge_over']:+.1%}  odds={int(r['over_odds']):+d}")
            else:
                print(f"  UNDER {r['team']:<28}  model {r['exp_wins']:.1f} < line {r['line']:.1f}  "
                      f"P(U)={(1-r['p_over_book']):.1%}  edge={r['edge_under']:+.1%}  odds={int(r['under_odds']):+d}")

    # Save comparison CSV
    out_path = os.path.join(HERE, "data", f"win_total_comparison_{args.year}.csv")
    matched[cols + ["edge_over", "edge_under"]].to_csv(out_path, index=False)
    print(f"\nFull comparison -> {out_path}")

    if unmatched:
        print(f"\n{len(unmatched)} model teams with no book line (FCS foes, unmatched names):")
        print("  " + ", ".join(unmatched[:10]) + ("..." if len(unmatched) > 10 else ""))


if __name__ == "__main__":
    main()

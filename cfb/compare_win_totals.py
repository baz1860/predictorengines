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

Team name fuzzy matching handles minor name differences between the model
(CFBD names) and The Odds API (book names).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))


def _fuzzy_key(name):
    """Lowercase, drop common suffixes for matching."""
    import re
    name = name.lower()
    for pat in (r"\bstate\b", r"\buniversity\b", r"\bcollege\b", r"\bthe\b"):
        name = re.sub(pat, "", name)
    return re.sub(r"[^a-z]", "", name).strip()


def merge(model, lines):
    """Merge on exact name first, then fuzzy."""
    merged = model.merge(lines, on="team", how="left")
    unmatched_model = merged[merged["line"].isna()]["team"].tolist()
    unmatched_lines = set(lines["team"]) - set(model["team"])

    if unmatched_model and unmatched_lines:
        # Build fuzzy map
        fk_lines = {_fuzzy_key(t): t for t in unmatched_lines}
        name_map = {}
        for t in unmatched_model:
            fk = _fuzzy_key(t)
            if fk in fk_lines:
                name_map[t] = fk_lines[fk]
        if name_map:
            extra = lines.copy()
            extra["team"] = extra["team"].map(lambda t: {v: k for k, v in name_map.items()}.get(t, t))
            merged = model.merge(extra, on="team", how="left")

    return merged


def p_over_line(exp_wins, sd, line):
    """P(wins > line) using normal approximation (continuity correction)."""
    return float(1.0 - norm.cdf(line + 0.5, loc=exp_wins, scale=max(sd, 0.1)))


def american_to_implied(odds):
    """American odds -> implied probability (no vig removal)."""
    if odds < 0:
        return -odds / (-odds + 100)
    else:
        return 100 / (odds + 100)


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

    model_path = os.path.join(HERE, "data", f"projected_win_totals_{args.year}.csv")
    lines_path = os.path.join(HERE, "data", f"win_totals_lines_{args.year}.csv")

    if not os.path.exists(model_path):
        raise SystemExit(f"Missing {model_path} — run: python3 win_totals.py")
    if not os.path.exists(lines_path):
        raise SystemExit(f"Missing {lines_path} — run: python3 fetch_win_total_lines.py")

    model = pd.read_csv(model_path)
    lines = pd.read_csv(lines_path)

    df = merge(model, lines)
    matched = df[df["line"].notna()].copy()
    unmatched = df[df["line"].isna()]["team"].tolist()

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

"""NHL edge finder for moneyline, puck-line, and totals markets."""
from __future__ import annotations

import argparse
import csv
import math
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from . import model as M

DATA_DIR = Path(__file__).resolve().parent / "data"
ODDS_CSV = DATA_DIR / "odds.csv"
FIXTURES_CSV = DATA_DIR / "fixtures.csv"

HEADER = ["date", "home", "away", "market", "side", "line", "odds"]
KELLY_FRACTION = 0.25
DEFAULT_OVERROUND = 1.045
EDGE_THRESHOLD = 0.03


def _fmt_line(line: float | None, market: str) -> str:
    if line is None or (isinstance(line, float) and math.isnan(line)):
        return ""
    if market == "spread":
        return f"{float(line):+g}"
    return f"{float(line):g}"


def _read_odds(path: Path = ODDS_CSV) -> pd.DataFrame:
    if not path.exists():
        raise ValueError("No nhl/data/odds.csv. Use 'Write template' first, then fill in odds.")
    odds = pd.read_csv(path)
    missing = [c for c in HEADER if c not in odds.columns]
    if missing:
        raise ValueError(f"nhl/data/odds.csv missing columns: {missing}")
    odds = odds[odds["odds"].notna() & (odds["odds"].astype(str).str.strip() != "")].copy()
    if odds.empty:
        raise ValueError("nhl/data/odds.csv has no filled-in odds.")
    odds["odds"] = pd.to_numeric(odds["odds"], errors="coerce")
    odds["line"] = pd.to_numeric(odds["line"], errors="coerce")
    odds = odds[odds["odds"].notna() & (odds["odds"] > 1.0)].copy()
    if odds.empty:
        raise ValueError("nhl/data/odds.csv has no valid decimal odds.")
    odds["market"] = odds["market"].map(M.normalize_market)
    odds["side"] = odds["side"].astype(str).str.lower().str.strip()
    return odds


def _pair_key(row: pd.Series) -> tuple[Any, ...]:
    market = str(row["market"])
    if market == "ml":
        line_key = ""
    elif market == "spread":
        line_key = round(abs(float(row["line"])), 2) if pd.notna(row["line"]) else ""
    else:
        line_key = round(float(row["line"]), 2) if pd.notna(row["line"]) else ""
    return (str(row["date"]), str(row["home"]), str(row["away"]), market, line_key)


def _bet_label(market: str, side: str, line: float | None) -> str:
    line_str = _fmt_line(line, market)
    if market == "ml":
        return f"ML {side}"
    if market == "spread":
        return f"PUCK LINE {side}{(' ' + line_str) if line_str else ''}"
    return f"TOTAL {side}{(' ' + line_str) if line_str else ''}"


def edge_rows(odds: pd.DataFrame | None = None, *, bankroll: float = 100.0,
              model: str = "blend") -> list[dict[str, Any]]:
    odds = _read_odds() if odds is None else odds.copy()
    odds["pairkey"] = odds.apply(_pair_key, axis=1)
    inv_sum = odds.groupby("pairkey")["odds"].apply(lambda s: (1.0 / s).sum())
    sides_per_key = odds.groupby("pairkey")["odds"].size()

    rows: list[dict[str, Any]] = []
    for _, r in odds.iterrows():
        market = str(r["market"])
        line = None if pd.isna(r["line"]) else float(r["line"])
        if market != "ml" and line is None:
            continue
        try:
            pred = M.predict_match(str(r["home"]), str(r["away"]), model=model)
            p_model, p_push = M.market_probs(pred, market, str(r["side"]), line)
        except Exception:
            continue

        n_sides = int(sides_per_key[r["pairkey"]])
        overround = float(inv_sum[r["pairkey"]]) if n_sides >= 2 else DEFAULT_OVERROUND
        p_market = (1.0 / float(r["odds"])) / overround
        ev = p_model * float(r["odds"]) + p_push - 1.0
        b = float(r["odds"]) - 1.0
        kelly = max(0.0, ev / b) if b > 0 else 0.0
        stake = round(KELLY_FRACTION * kelly * float(bankroll), 2)
        line_str = _fmt_line(line, market)
        rows.append({
            "date": str(r["date"]),
            "match_date": str(r["date"]),
            "match": f"{r['away']} @ {r['home']}",
            "home": str(r["home"]),
            "away": str(r["away"]),
            "market": market,
            "side": str(r["side"]),
            "line": line_str,
            "bet": _bet_label(market, str(r["side"]), line),
            "odds": round(float(r["odds"]), 3),
            "p_model": round(float(p_model), 4),
            "p_push": round(float(p_push), 4),
            "p_book": round(float(p_market), 4),
            "p_market": round(float(p_market), 4),
            "edge": round(float(p_model - p_market), 4),
            "ev_per_unit": round(float(ev), 4),
            "kelly_frac": round(KELLY_FRACTION * kelly, 4),
            "stake_gbp": stake,
        })
    rows.sort(key=lambda x: (-float(x["edge"]), -float(x["ev_per_unit"])))
    return rows


def columns() -> list[dict[str, str]]:
    return [
        {"key": "date", "label": "Date", "fmt": "text"},
        {"key": "match", "label": "Match", "fmt": "text"},
        {"key": "bet", "label": "Bet", "fmt": "text"},
        {"key": "odds", "label": "Odds", "fmt": "num"},
        {"key": "p_model", "label": "Model", "fmt": "pct"},
        {"key": "p_book", "label": "Book", "fmt": "pct"},
        {"key": "edge", "label": "Edge", "fmt": "signed_pct"},
        {"key": "ev_per_unit", "label": "EV", "fmt": "num"},
        {"key": "stake_gbp", "label": "Stake", "fmt": "gbp"},
    ]


def build_report(*, bankroll: float = 100.0, model: str = "blend") -> dict[str, Any]:
    rows = edge_rows(bankroll=bankroll, model=model)
    return {
        "note": f"Manual odds for {len(rows)} NHL quote(s) (nhl/data/odds.csv)",
        "columns": columns(),
        "rows": rows,
    }


def write_template(path: Path = ODDS_CSV) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fixtures = []
    if FIXTURES_CSV.exists():
        df = pd.read_csv(FIXTURES_CSV)
        need = {"date", "home", "away"}
        if need.issubset(df.columns):
            fixtures = [
                (str(r.date), str(r.home), str(r.away))
                for r in df.itertuples(index=False)
                if str(r.home).strip() and str(r.away).strip()
            ][:10]
    if not fixtures:
        fixtures = [(str(date.today()), "Toronto Maple Leafs", "Boston Bruins")]

    rows = []
    for match_date, home, away in fixtures:
        base = [match_date, home, away]
        rows.extend([
            base + ["ml", "home", "", ""],
            base + ["ml", "away", "", ""],
            base + ["spread", "home", -1.5, ""],
            base + ["spread", "away", 1.5, ""],
            base + ["total", "over", 6.5, ""],
            base + ["total", "under", 6.5, ""],
        ])
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="NHL odds edge finder")
    ap.add_argument("--template", action="store_true", help="write nhl/data/odds.csv template")
    ap.add_argument("--bankroll", type=float, default=100.0)
    ap.add_argument("--model", choices=["blend", "power", "form"], default="blend")
    args = ap.parse_args()

    if args.template:
        path = write_template()
        print(f"wrote {path}")
        return
    report = build_report(bankroll=args.bankroll, model=args.model)
    df = pd.DataFrame(report["rows"])
    with pd.option_context("display.width", 200):
        print(df.to_string(index=False) if not df.empty else "no rows")


if __name__ == "__main__":
    main()

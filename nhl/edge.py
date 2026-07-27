"""NHL edge finder for moneyline, puck-line, and totals markets."""
from __future__ import annotations

import argparse
import csv
import math
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from . import gate as G
from . import model as M

DATA_DIR = Path(__file__).resolve().parent / "data"
ODDS_CSV = DATA_DIR / "odds.csv"
FIXTURES_CSV = DATA_DIR / "fixtures.csv"

HEADER = ["date", "home", "away", "market", "side", "line", "odds"]
TEMPLATE_HEADER = ["event_id", "bookmaker", "captured_at", *HEADER]
KELLY_FRACTION = 0.25
DEFAULT_OVERROUND = 1.045
EDGE_THRESHOLD = 0.03
VALID_SIDES = {
    "ml": {"home", "away"},
    "spread": {"home", "away"},
    "total": {"over", "under"},
}


def _id_key(value: Any) -> str:
    text = str(value or "").strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


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
    odds["_row_num"] = odds.index + 2
    if "event_id" not in odds.columns:
        odds["event_id"] = odds["game_id"] if "game_id" in odds.columns else ""
    if "bookmaker" not in odds.columns:
        odds["bookmaker"] = "manual"
    if "captured_at" not in odds.columns:
        odds["captured_at"] = ""
    odds["bookmaker"] = odds["bookmaker"].fillna("").astype(str).str.strip().replace("", "manual")
    odds["captured_at"] = odds["captured_at"].fillna("").astype(str).str.strip()
    odds["event_id"] = odds["event_id"].map(_id_key)
    odds = odds[odds["odds"].notna() & (odds["odds"].astype(str).str.strip() != "")].copy()
    if odds.empty:
        raise ValueError("nhl/data/odds.csv has no filled-in odds.")
    odds["odds"] = pd.to_numeric(odds["odds"], errors="coerce")
    odds["line"] = pd.to_numeric(odds["line"], errors="coerce")
    bad_odds = odds[odds["odds"].isna() | (odds["odds"] <= 1.0)]
    if not bad_odds.empty:
        rows = bad_odds["_row_num"].astype(int).tolist()
        raise ValueError(f"nhl/data/odds.csv has invalid decimal odds on row(s): {rows}")
    odds["market"] = odds["market"].apply(M.normalize_market)
    odds["side"] = odds["side"].astype(str).str.lower().str.strip()
    bad_sides = [
        int(r["_row_num"])
        for r in odds.to_dict("records")
        if str(r["side"]) not in VALID_SIDES[str(r["market"])]
    ]
    if bad_sides:
        raise ValueError(f"nhl/data/odds.csv has invalid side values on row(s): {bad_sides}")
    missing_lines = odds[(odds["market"] != "ml") & odds["line"].isna()]
    if not missing_lines.empty:
        rows = missing_lines["_row_num"].astype(int).tolist()
        raise ValueError(f"nhl/data/odds.csv missing line values on row(s): {rows}")
    return odds


def _pair_key(row: pd.Series) -> tuple[Any, ...]:
    market = str(row["market"])
    if market == "ml":
        line_key = ""
    elif market == "spread":
        line_key = round(abs(float(row["line"])), 2) if pd.notna(row["line"]) else ""
    else:
        line_key = round(float(row["line"]), 2) if pd.notna(row["line"]) else ""
    event_key = _id_key(row.get("event_id") or "")
    if not event_key:
        event_key = f"{row['date']}|{row['home']}|{row['away']}"
    return (
        event_key,
        str(row.get("bookmaker") or "manual"),
        str(row.get("captured_at") or ""),
        market,
        line_key,
    )


def _validate_quote_groups(odds: pd.DataFrame) -> None:
    for key, group in odds.groupby("pairkey", sort=False):
        market = str(group.iloc[0]["market"])
        expected = VALID_SIDES[market]
        side_counts = group["side"].value_counts()
        duplicates = sorted(side for side, count in side_counts.items() if int(count) > 1)
        if duplicates:
            rows = group[group["side"].isin(duplicates)]["_row_num"].astype(int).tolist()
            raise ValueError(f"Duplicate NHL {market} side quote(s) {duplicates} on row(s): {rows}")
        sides = set(group["side"].astype(str))
        if sides != expected:
            missing = sorted(expected - sides)
            extra = sorted(sides - expected)
            rows = group["_row_num"].astype(int).tolist()
            raise ValueError(
                f"Incomplete NHL {market} quote group {key}: rows {rows}, "
                f"missing={missing}, extra={extra}"
            )


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
    for col, default in (("event_id", ""), ("bookmaker", "manual"), ("captured_at", "")):
        if col not in odds.columns:
            odds[col] = default
    if "_row_num" not in odds.columns:
        odds["_row_num"] = odds.index + 2
    odds["event_id"] = odds["event_id"].map(_id_key)
    odds["bookmaker"] = odds["bookmaker"].fillna("").astype(str).str.strip().replace("", "manual")
    odds["captured_at"] = odds["captured_at"].fillna("").astype(str).str.strip()
    odds["market"] = odds["market"].apply(M.normalize_market)
    odds["side"] = odds["side"].astype(str).str.lower().str.strip()
    odds["odds"] = pd.to_numeric(odds["odds"], errors="coerce")
    odds["line"] = pd.to_numeric(odds["line"], errors="coerce")
    bad_sides = [
        int(r["_row_num"])
        for r in odds.to_dict("records")
        if str(r["side"]) not in VALID_SIDES[str(r["market"])]
    ]
    if bad_sides:
        raise ValueError(f"NHL odds contain invalid side values on row(s): {bad_sides}")
    bad_odds = odds[odds["odds"].isna() | (odds["odds"] <= 1.0)]
    if not bad_odds.empty:
        rows = bad_odds["_row_num"].astype(int).tolist()
        raise ValueError(f"NHL odds contain invalid decimal odds on row(s): {rows}")
    missing_lines = odds[(odds["market"] != "ml") & odds["line"].isna()]
    if not missing_lines.empty:
        rows = missing_lines["_row_num"].astype(int).tolist()
        raise ValueError(f"NHL odds missing line values on row(s): {rows}")
    odds["pairkey"] = odds.apply(_pair_key, axis=1)
    _validate_quote_groups(odds)
    inv_sum = odds.groupby("pairkey")["odds"].apply(lambda s: (1.0 / s).sum())
    sides_per_key = odds.groupby("pairkey")["odds"].size()

    rows: list[dict[str, Any]] = []
    for _, r in odds.iterrows():
        market = str(r["market"])
        line = None if pd.isna(r["line"]) else float(r["line"])
        pred = M.predict_match(str(r["home"]), str(r["away"]), model=model)
        p_model, p_push = M.market_probs(pred, market, str(r["side"]), line)

        n_sides = int(sides_per_key[r["pairkey"]])
        overround = float(inv_sum[r["pairkey"]]) if n_sides >= 2 else DEFAULT_OVERROUND
        p_market = (1.0 / float(r["odds"])) / overround
        ev = p_model * float(r["odds"]) + p_push - 1.0
        b = float(r["odds"]) - 1.0
        stake_risk = max(1e-9, 1.0 - p_push)
        kelly = max(0.0, ev / (b * stake_risk)) if b > 0 else 0.0
        stake = round(KELLY_FRACTION * kelly * float(bankroll), 2)
        line_str = _fmt_line(line, market)
        rows.append({
            "event_id": _id_key(r.get("event_id") or ""),
            "date": str(r["date"]),
            "match_date": str(r["date"]),
            "bookmaker": str(r.get("bookmaker") or "manual"),
            "captured_at": str(r.get("captured_at") or ""),
            "source": str(r.get("bookmaker") or "manual"),
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
    G.apply_staking_gate(rows)
    return rows


def columns() -> list[dict[str, str]]:
    return [
        {"key": "date", "label": "Date", "fmt": "text"},
        {"key": "bookmaker", "label": "Book", "fmt": "text"},
        {"key": "match", "label": "Match", "fmt": "text"},
        {"key": "bet", "label": "Bet", "fmt": "text"},
        {"key": "odds", "label": "Odds", "fmt": "num"},
        {"key": "p_model", "label": "Model", "fmt": "pct"},
        {"key": "p_book", "label": "Book Prob", "fmt": "pct"},
        {"key": "edge", "label": "Edge", "fmt": "signed_pct"},
        {"key": "ev_per_unit", "label": "EV", "fmt": "num"},
        {"key": "stake_gbp", "label": "Stake", "fmt": "gbp"},
    ]


def build_report(*, bankroll: float = 100.0, model: str = "blend") -> dict[str, Any]:
    rows = edge_rows(bankroll=bankroll, model=model)
    gate = G.load_gate()
    disabled = not G.staking_enabled(gate)
    note = f"Manual odds for {len(rows)} NHL quote(s) (nhl/data/odds.csv)"
    if disabled:
        note += f" · staking disabled: {gate.get('reason', 'validation gate failed')}"
    return {
        "note": note,
        "staking_gate": gate,
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
                (str(getattr(r, "game_id", "") or getattr(r, "event_id", "") or ""),
                 str(r.date), str(r.home), str(r.away))
                for r in df.itertuples(index=False)
                if str(r.home).strip() and str(r.away).strip()
            ][:10]
    if not fixtures:
        fixtures = [("", str(date.today()), "Toronto Maple Leafs", "Boston Bruins")]

    rows = []
    for event_id, match_date, home, away in fixtures:
        base = [event_id, "manual", "", match_date, home, away]
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
        w.writerow(TEMPLATE_HEADER)
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

#!/usr/bin/env python3
"""Isolated runner for the CFB engine.

Invoked as a subprocess by app/engines/cfb.py with cwd + PYTHONPATH set to the
cfb/ folder, so the engine's flat imports (`import elo`, `import power`,
`from predictor import ...`) resolve to the CFB modules without colliding with
the root engine's same-named modules. Reads a JSON params object on stdin and
writes a JSON result on stdout.

Commands: schema | predict | edge | edge_template
"""
import json
import sys

import pandas as pd

import elo as E
import power as P
from predictor import blend_predict
import edge as CE   # cfb/edge.py


def _params():
    raw = sys.stdin.read().strip()
    return json.loads(raw) if raw else {}


def cmd_schema():
    pp = P.load_params()
    return {"kind": "match", "names": sorted(pp["teams"]),
            "models": ["blend", "elo", "power"], "supports_home": False,
            "neutral_toggle": True, "team_label": "Team"}


def cmd_predict(p):
    t1 = (p.get("team1") or "").strip()
    t2 = (p.get("team2") or "").strip()
    if not t1 or not t2:
        raise ValueError("Pick two teams.")
    if t1 == t2:
        raise ValueError("Pick two different teams.")
    model = p.get("model", "blend")
    neutral = bool(p.get("neutral", False))
    eparams, pparams = E.build(), P.load_params()
    for t in (t1, t2):
        if t not in pparams["teams"]:
            raise ValueError(f"Unknown team: {t!r}")
    out = blend_predict(eparams, pparams, t1, t2, neutral, model)
    p1 = float(out["p1"]); margin = float(out["margin"]); total = float(out["total"])
    venue = "neutral site" if neutral else f"{t1} at home"
    return {
        "competitors": [{"name": t1, "sub": "home" if not neutral else "neutral"},
                        {"name": t2, "sub": ""}],
        "headline": f"Spread {t1} {-margin:+.1f} · Total {total:.1f} · {venue}",
        "outcomes": [{"label": f"{t1} win", "prob": round(p1, 4), "kind": "win"},
                     {"label": f"{t2} win", "prob": round(1 - p1, 4), "kind": "loss"}],
        "stats": [{"label": "Spread", "value": f"{t1} {-margin:+.1f}"},
                  {"label": "Total", "value": f"{total:.1f}"},
                  {"label": "Proj. margin", "value": f"{t1} {margin:+.1f}"}],
        "table": None}


def cmd_edge(p):
    import os
    if not os.path.exists(CE.ODDS_CSV):
        raise ValueError("No cfb/odds.csv. Use 'Write template' first, then fill in lines & odds.")
    odds = pd.read_csv(CE.ODDS_CSV)
    odds = odds[odds["odds"].notna() & (odds["odds"] != "")]
    if odds.empty:
        raise ValueError("cfb/odds.csv has no filled-in odds.")
    odds["odds"] = odds["odds"].astype(float)
    odds["line"] = pd.to_numeric(odds["line"], errors="coerce")
    bankroll = float(p.get("bankroll", 100.0))
    model = p.get("model", "blend")
    if model not in ("blend", "elo", "power"):
        raise ValueError(f"Unknown model: {model!r}")

    eparams, pparams = E.build(), P.load_params()

    def key(r):
        line_key = "" if r["market"] == "ml" else round(abs(r["line"]), 1)
        return (r["home"], r["away"], r["market"], line_key)

    odds["pairkey"] = odds.apply(key, axis=1)
    inv_sum = odds.groupby("pairkey")["odds"].apply(lambda s: (1.0 / s).sum())
    sides_per_key = odds.groupby("pairkey")["odds"].size()

    rows = []
    for r in odds.itertuples():
        try:
            pred = blend_predict(eparams, pparams, r.home, r.away,
                                 neutral=bool(r.neutral), model=model)
        except Exception:
            continue
        line = None if pd.isna(r.line) else float(r.line)
        if r.market != "ml" and line is None:
            continue
        p_model = CE.model_prob(pred, pparams, r.market, r.side, line)
        n_sides = int(sides_per_key[r.pairkey])
        over = float(inv_sum[r.pairkey]) if n_sides == 2 else CE.DEFAULT_OVERROUND
        p_imp = (1.0 / r.odds) / over
        edge = p_model - p_imp
        ev = p_model * r.odds - 1.0
        kelly = max(0.0, (p_model * r.odds - 1.0) / (r.odds - 1.0))
        stake = round(CE.KELLY_FRACTION * kelly * bankroll, 2)
        line_str = "" if line is None else f"{line:+g}"
        rows.append({
            "date": str(r.date), "match": f"{r.away} @ {r.home}",
            "home": r.home, "away": r.away,
            "bet": f"{r.market.upper()} {r.side}{(' ' + line_str) if line_str else ''}",
            "market": r.market, "side": r.side, "line": line_str, "odds": round(float(r.odds), 3),
            "p_model": round(float(p_model), 3), "p_book": round(float(p_imp), 3),
            "edge": round(float(edge), 3), "ev_per_unit": round(float(ev), 3),
            "kelly_frac": round(CE.KELLY_FRACTION * kelly, 4), "stake_gbp": stake})
    rows.sort(key=lambda x: -x["edge"])
    columns = [
        {"key": "date", "label": "Date", "fmt": "text"},
        {"key": "match", "label": "Match", "fmt": "text"},
        {"key": "bet", "label": "Bet", "fmt": "text"},
        {"key": "odds", "label": "Odds", "fmt": "num"},
        {"key": "p_model", "label": "Model", "fmt": "pct"},
        {"key": "p_book", "label": "Book", "fmt": "pct"},
        {"key": "edge", "label": "Edge", "fmt": "signed_pct"},
        {"key": "ev_per_unit", "label": "EV", "fmt": "num"},
        {"key": "stake_gbp", "label": "Stake", "fmt": "gbp"}]
    return {"note": f"Manual odds for {len(rows)} quote(s) (cfb/odds.csv)",
            "columns": columns, "rows": rows}


def cmd_edge_template(p):
    try:
        CE.write_template()
    except Exception:
        # cfb/edge.write_template() depends on a well-formed data/upcoming.csv
        # (home_div/away_div columns) that may be absent off-season — fall back
        # to a hand-editable sample template so the GUI button always works.
        import csv
        from datetime import date
        base = [str(date.today()), "Ohio State", "Michigan", 0]
        rows = [base + ["ml", "home", "", ""], base + ["ml", "away", "", ""],
                base + ["spread", "home", -6.5, ""], base + ["spread", "away", 6.5, ""],
                base + ["total", "over", 48.5, ""], base + ["total", "under", 48.5, ""]]
        with open(CE.ODDS_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(CE.HEADER)
            w.writerows(rows)
    return {"path": "cfb/odds.csv"}


COMMANDS = {"schema": lambda p: cmd_schema(), "predict": cmd_predict,
            "edge": cmd_edge, "edge_template": cmd_edge_template}


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "schema"
    try:
        result = COMMANDS[cmd](_params())
        print(json.dumps(result))
    except ValueError as e:
        print(json.dumps({"error": str(e)})); sys.exit(2)
    except Exception as e:  # noqa
        print(json.dumps({"error": f"{type(e).__name__}: {e}"})); sys.exit(1)


if __name__ == "__main__":
    main()

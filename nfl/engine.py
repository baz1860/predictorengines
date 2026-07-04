"""In-process command API for the NFL engine (mirrors cfb/engine.py).

Command logic called directly by app/engines/nfl.py (no subprocess). Predict
(win/spread/total via the margin PMF) and Edge (ml/spread/total from a manual
odds.csv). Settlement is done in-process with plain pandas against
nfl/data/games.csv in the adapter, so no engine import is needed there.
"""
from __future__ import annotations

import os

import pandas as pd

from . import edge as NE
from .predictor import Models, blend_predict
from .team_names import all_full_names

_MODELS: Models | None = None


def _models() -> Models:
    global _MODELS
    if _MODELS is None:
        _MODELS = Models.load()
    return _MODELS


def cmd_schema(_p: dict | None = None) -> dict:
    return {
        "kind": "match", "names": all_full_names(),
        "models": ["blend", "elo", "power", "epa"],
        "supports_home": False, "neutral_toggle": True, "team_label": "Team",
    }


def cmd_predict(p: dict) -> dict:
    t1 = (p.get("team1") or "").strip()
    t2 = (p.get("team2") or "").strip()
    if not t1 or not t2:
        raise ValueError("Pick two teams.")
    if t1 == t2:
        raise ValueError("Pick two different teams.")
    model = p.get("model", "blend")
    neutral = bool(p.get("neutral", False))
    season = p.get("season")
    week = p.get("week")
    models = _models()
    out = blend_predict(models, t1, t2, neutral=neutral, model=model,
                        season=season, week=week,
                        qb1=p.get("home_qb"), qb2=p.get("away_qb"))
    margin, total, p1 = out["margin"], out["total"], out["p1"]
    venue = "neutral site" if neutral else f"{out['team1']} at home"
    return {
        "competitors": [{"name": out["team1"], "sub": "home" if not neutral else "neutral"},
                        {"name": out["team2"], "sub": ""}],
        "headline": f"Spread {out['team1']} {-margin:+.1f} · Total {total:.1f} · {venue}",
        "outcomes": [{"label": f"{out['team1']} win", "prob": round(p1, 4), "kind": "win"},
                     {"label": f"{out['team2']} win", "prob": round(out['p2'], 4), "kind": "loss"}],
        "stats": [{"label": "Spread", "value": f"{out['team1']} {-margin:+.1f}"},
                  {"label": "Total", "value": f"{total:.1f}"},
                  {"label": "Proj. margin", "value": f"{out['team1']} {margin:+.1f}"},
                  {"label": "Model", "value": model}],
        "table": None,
    }


def cmd_edge(p: dict) -> dict:
    if not os.path.exists(NE.ODDS_CSV):
        raise ValueError("No nfl/odds.csv. Use 'Write template' first, then fill in lines & odds.")
    odds = pd.read_csv(NE.ODDS_CSV)
    odds = odds[odds["odds"].notna() & (odds["odds"] != "")]
    if odds.empty:
        raise ValueError("nfl/odds.csv has no filled-in odds.")
    odds["odds"] = odds["odds"].astype(float)
    odds["line"] = pd.to_numeric(odds["line"], errors="coerce")
    bankroll = float(p.get("bankroll", 100.0))
    model = p.get("model", "blend")
    if model not in ("blend", "elo", "power", "epa"):
        raise ValueError(f"Unknown model: {model!r}")

    models = _models()

    def key(r):
        line_key = "" if r["market"] == "ml" else round(abs(r["line"]), 1)
        return (r["home"], r["away"], r["market"], line_key)

    odds["pairkey"] = odds.apply(key, axis=1)
    inv_sum = odds.groupby("pairkey")["odds"].apply(lambda s: (1.0 / s).sum())
    sides_per_key = odds.groupby("pairkey")["odds"].size()

    rows = []
    for r in odds.itertuples():
        try:
            pred = blend_predict(models, r.home, r.away, neutral=bool(r.neutral), model=model,
                                 season=int(r.season), week=int(r.week))
        except Exception:
            continue
        line = None if pd.isna(r.line) else float(r.line)
        if r.market != "ml" and line is None:
            continue
        probs = NE.model_probs(pred, models.pmf, r.market, r.side, line)
        p_model = probs["p_win"]
        n_sides = int(sides_per_key[r.pairkey])
        over = float(inv_sum[r.pairkey]) if n_sides == 2 else NE.DEFAULT_OVERROUND
        p_imp = (1.0 / r.odds) / over
        edge = p_model - p_imp
        p_loss = max(0.0, 1.0 - p_model - probs["p_push"])
        ev = p_model * (r.odds - 1.0) - p_loss
        kelly = NE.kelly_trinomial(p_model, probs["p_push"], r.odds)
        stake = round(NE.KELLY_FRACTION * kelly * bankroll, 2)
        line_str = "" if line is None else f"{line:+g}"
        rows.append({
            "date": str(r.date), "match": f"{r.away} @ {r.home}",
            "home": r.home, "away": r.away,
            "bet": f"{r.market.upper()} {r.side}{(' ' + line_str) if line_str else ''}",
            "market": r.market, "side": r.side, "line": line_str, "odds": round(float(r.odds), 3),
            "p_model": round(float(p_model), 3), "p_book": round(float(p_imp), 3),
            "edge": round(float(edge), 3), "ev_per_unit": round(float(ev), 3),
            "kelly_frac": round(NE.KELLY_FRACTION * kelly, 4), "stake_gbp": stake})
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
    return {"note": f"Manual odds for {len(rows)} quote(s) (nfl/odds.csv)",
            "columns": columns, "rows": rows}


def cmd_edge_template(_p: dict | None = None) -> dict:
    try:
        NE.write_template()
    except Exception:
        import csv
        from datetime import date
        base = [date.today().year, 1, str(date.today()), "Kansas City Chiefs", "Buffalo Bills", 0]
        rows = [base + ["ml", "home", "", ""], base + ["ml", "away", "", ""],
                base + ["spread", "home", -2.5, ""], base + ["spread", "away", 2.5, ""],
                base + ["total", "over", 47.5, ""], base + ["total", "under", 47.5, ""]]
        with open(NE.ODDS_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(NE.HEADER)
            w.writerows(rows)
    return {"path": "nfl/odds.csv"}


COMMANDS = {"schema": lambda p: cmd_schema(), "predict": cmd_predict,
           "edge": cmd_edge, "edge_template": cmd_edge_template}

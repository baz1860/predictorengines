"""In-process command API for the NHL engine."""
from __future__ import annotations

from typing import Any

from . import edge as E
from . import model as M


def cmd_schema(_p: dict | None = None) -> dict[str, Any]:
    return {
        "kind": "match",
        "names": M.team_names(),
        "models": ["blend", "power", "form"],
        "supports_home": False,
        "neutral_toggle": True,
        "team_label": "Team",
    }


def cmd_predict(p: dict[str, Any]) -> dict[str, Any]:
    home = (p.get("team1") or p.get("home") or "").strip()
    away = (p.get("team2") or p.get("away") or "").strip()
    if not home or not away:
        raise ValueError("Pick two NHL teams.")
    model = str(p.get("model") or "blend").lower()
    neutral = bool(p.get("neutral", False))
    pred = M.predict_match(home, away, neutral=neutral, model=model)
    p_home = float(pred["p_home"])
    lam_h = float(pred["lambda_home"])
    lam_a = float(pred["lambda_away"])
    total = float(pred["total"])
    venue = "neutral site" if neutral else f"{home} at home"
    return {
        "competitors": [
            {"name": home, "sub": "home" if not neutral else "neutral"},
            {"name": away, "sub": ""},
        ],
        "headline": (
            f"{home} {p_home:.1%} · projected {lam_h:.2f}-{lam_a:.2f} "
            f"· total {total:.1f} · {venue}"
        ),
        "outcomes": [
            {"label": f"{home} win", "prob": round(p_home, 4), "kind": "win"},
            {"label": f"{away} win", "prob": round(1.0 - p_home, 4), "kind": "loss"},
        ],
        "stats": [
            {"label": "Projected goals", "value": f"{home} {lam_h:.2f} · {away} {lam_a:.2f}"},
            {"label": "Regulation tie", "value": f"{float(pred['p_reg_tie']):.1%}"},
            {"label": "Over 5.5", "value": f"{float(pred['p_over_5_5']):.1%}"},
            {"label": f"{home} -1.5", "value": f"{float(pred['p_home_minus_1_5']):.1%}"},
            {"label": "Model", "value": model},
        ],
        "table": None,
    }


def cmd_edge(p: dict[str, Any]) -> dict[str, Any]:
    bankroll = float(p.get("bankroll", 100.0))
    model = str(p.get("model") or "blend").lower()
    if model not in {"blend", "power", "form"}:
        raise ValueError(f"Unknown NHL model: {model!r}")
    return E.build_report(bankroll=bankroll, model=model)


def cmd_edge_template(_p: dict | None = None) -> dict[str, str]:
    E.write_template()
    return {"path": "nhl/data/odds.csv"}


COMMANDS = {
    "schema": lambda p: cmd_schema(p),
    "predict": cmd_predict,
    "edge": cmd_edge,
    "edge_template": cmd_edge_template,
}

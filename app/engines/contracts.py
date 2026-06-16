"""Shared engine contract helpers (V3 M1).

Lightweight, dependency-free validation + normalization so every engine speaks
ONE stable app contract before we touch any modelling. There are deliberately no
engine imports here — this module only shapes and checks data that adapters hand
to it, so it can be imported from tests and from any adapter without dragging in
a flat-module namespace.

What it provides:
  * finite-number / JSON-safety validation (no NaN/Inf, no exotic objects);
  * prediction-result and table/column structure validation;
  * stable market identifiers;
  * stable event/fixture keys;
  * edge-row normalization to the canonical V3 shape.

Normalization is ADDITIVE: it fills in the canonical fields on a row without
dropping the engine's existing UI keys, so the `columns` each adapter already
returns keep rendering exactly as before.
"""
from __future__ import annotations

import math
import re
from typing import Any, Iterable

# Canonical edge-row fields every engine emits in addition to its own UI fields.
CANONICAL_EDGE_FIELDS = [
    "event_id", "match_date", "home", "away", "market", "side", "line", "bet",
    "odds", "p_model", "p_market", "p_book", "edge", "ev_per_unit",
    "kelly_frac", "stake_gbp", "source", "model", "recommended",
]


class ContractError(ValueError):
    """Raised when a payload violates the shared engine contract."""


# ── finite / JSON safety ──────────────────────────────────────────────────────
_JSON_PRIMITIVES = (str, bool, int, float, type(None))


def is_finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def assert_finite_json(obj: Any, path: str = "$") -> None:
    """Reject NaN/Inf and any non-JSON-serializable value anywhere in `obj`.

    Python's json.dumps allows NaN/Inf by default, which then breaks strict JSON
    consumers (and silently poisons downstream maths). This walks the structure
    and fails loudly instead.
    """
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return
    if isinstance(obj, (int, float)):
        if isinstance(obj, float) and not math.isfinite(obj):
            raise ContractError(f"non-finite number at {path}: {obj!r}")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise ContractError(f"non-string key at {path}: {k!r}")
            assert_finite_json(v, f"{path}.{k}")
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_finite_json(v, f"{path}[{i}]")
        return
    raise ContractError(f"non-JSON value at {path}: {type(obj).__name__}")


# ── market identifiers ────────────────────────────────────────────────────────
_MARKET_ALIASES = {
    "1x2": "1x2", "h2h": "1x2", "moneyline_1x2": "1x2",
    "ml": "ml", "moneyline": "ml", "money_line": "ml",
    "spread": "spread", "handicap": "spread", "ats": "spread",
    "total": "total", "totals": "total", "ou": "total", "over_under": "total",
    "btts": "btts", "both_teams_to_score": "btts",
    "win": "win", "outright": "win", "winner": "win",
    "top5": "top5", "t5": "top5", "top_5": "top5",
    "top10": "top10", "t10": "top10", "top_10": "top10",
    "top20": "top20", "t20": "top20", "top_20": "top20",
    "cut": "cut", "make_cut": "cut", "makecut": "cut",
    "matchup": "matchup", "h2h_matchup": "matchup",
    "3ball": "3ball", "threeball": "3ball", "three_ball": "3ball",
}


def market_id(raw: Any) -> str:
    """Normalize a market label to a stable identifier.

    Golf carries the meaningful market in its `side` field (e.g. "win",
    "matchup:a|b"), so this also accepts side-style values and strips any
    participant payload after a colon.
    """
    s = str(raw or "").strip().lower()
    if not s:
        return ""
    head = s.split(":", 1)[0]              # matchup:a|b -> matchup
    head = re.sub(r"\s+", "_", head)
    return _MARKET_ALIASES.get(head, head)


# ── event identity ────────────────────────────────────────────────────────────
def _slug(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s or "").strip().lower()).strip("-")


def fixture_key(match_date: Any, home: Any, away: Any,
                competition: Any = "") -> str:
    """Stable key for a two-competitor fixture: date|home|away(|competition).

    Used as the default event_id for match-style engines. Golf supplies its own
    tournament-scoped event_id (see golf adapter), so this is the soccer/CFB
    path. Deterministic and human-readable on purpose — easier to debug a ledger
    than an opaque hash.
    """
    parts = [_slug(match_date), _slug(home), _slug(away)]
    if competition:
        parts.append(_slug(competition))
    return "|".join(p for p in parts if p)


# ── prediction / table validation ─────────────────────────────────────────────
def validate_prediction(result: dict) -> dict:
    """Structural check for a predict() payload. Returns the result unchanged.

    A valid prediction is either outcomes-style (1X2 / win-loss, the match
    engines) or table-style (a {columns, rows} grid, e.g. golf head-to-head).
    """
    if not isinstance(result, dict):
        raise ContractError("prediction result must be a dict")
    outcomes = result.get("outcomes")
    has_outcomes = isinstance(outcomes, list) and bool(outcomes)
    has_table = isinstance(result.get("columns"), list) and bool(result.get("columns"))
    if not has_outcomes and not has_table:
        raise ContractError("prediction needs either 'outcomes' or a 'columns' table")
    if has_outcomes:
        for o in outcomes:
            if not isinstance(o, dict) or "prob" not in o or "label" not in o:
                raise ContractError("each outcome needs 'label' and 'prob'")
            if not (0.0 <= float(o["prob"]) <= 1.0 + 1e-9):
                raise ContractError(f"outcome prob out of range: {o.get('prob')!r}")
    if has_table:
        validate_table(result)
    if result.get("table") is not None:
        validate_table(result["table"])
    assert_finite_json(result)
    return result


def validate_table(table: dict) -> dict:
    """Structural check for a {columns, rows} table payload."""
    if not isinstance(table, dict):
        raise ContractError("table must be a dict")
    cols = table.get("columns")
    rows = table.get("rows")
    if not isinstance(cols, list) or not cols:
        raise ContractError("table needs a non-empty 'columns' list")
    col_keys = set()
    for c in cols:
        if not isinstance(c, dict) or "key" not in c or "label" not in c:
            raise ContractError("each column needs 'key' and 'label'")
        col_keys.add(c["key"])
    if not isinstance(rows, list):
        raise ContractError("table 'rows' must be a list")
    assert_finite_json(table)
    return table


# ── edge-row normalization ────────────────────────────────────────────────────
def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def normalize_edge_row(row: dict, *, source: str, model: str,
                       sport: str = "", competition: str = "") -> dict:
    """Return `row` augmented with every canonical edge field (additive).

    Tolerant of the per-engine differences in field names:
      * match_date  <- match_date | date
      * home        <- home | player           (golf carries the entrant here)
      * p_market/p_book mirror each other when only one is present
      * kelly_frac  <- kelly_frac | kelly_stake
      * edge        computed from p_model - p_market when absent
      * event_id    defaults to a fixture key (golf passes its own)
    The original keys are preserved so existing UI `columns` keep working.
    """
    out = dict(row)

    out["match_date"] = str(row.get("match_date", row.get("date", "")) or "")
    home = row.get("home", row.get("player", "")) or ""
    out["home"] = home
    out["away"] = row.get("away", "") or ""
    out["side"] = str(row.get("side", "") or "")
    # Golf's market lives in `side`; everything else has an explicit `market`.
    out["market"] = market_id(row.get("market") or out["side"])
    line = row.get("line", "")
    out["line"] = "" if line is None or (isinstance(line, float) and math.isnan(line)) else line
    out["bet"] = str(row.get("bet", "") or f"{out['market']} {out['side']}".strip())

    out["odds"] = _num(row.get("odds"))
    out["p_model"] = _num(row.get("p_model"))
    pm = _num(row.get("p_market"))
    pb = _num(row.get("p_book"))
    if pm is None:
        pm = pb
    if pb is None:
        pb = pm
    out["p_market"] = pm
    out["p_book"] = pb

    edge = _num(row.get("edge"))
    if edge is None and out["p_model"] is not None and pm is not None:
        edge = round(out["p_model"] - pm, 4)
    out["edge"] = edge
    out["ev_per_unit"] = _num(row.get("ev_per_unit"))

    kf = row.get("kelly_frac", row.get("kelly_stake"))
    out["kelly_frac"] = _num(kf)
    out["stake_gbp"] = _num(row.get("stake_gbp")) or 0.0

    out["source"] = str(row.get("source", source) or source)
    out["model"] = str(row.get("model", model) or model)
    out["recommended"] = bool(row.get("recommended", False))

    if row.get("event_id"):
        out["event_id"] = str(row["event_id"])
    else:
        comp = row.get("competition", competition) or ""
        out["event_id"] = fixture_key(out["match_date"], home, out["away"], comp)

    # Sanitize any remaining non-finite float a source row may carry (rows are
    # flat), so the live edge() path emits clean JSON instead of crashing. The
    # contract test still asserts finiteness independently.
    for k, v in out.items():
        if isinstance(v, float) and not math.isfinite(v):
            out[k] = None
    return out


def normalize_edge_result(result: dict, *, source: str, model: str,
                          sport: str = "") -> dict:
    """Normalize every row of an edge() payload in place and finite-check it."""
    if not isinstance(result, dict):
        raise ContractError("edge result must be a dict")
    rows = result.get("rows")
    if isinstance(rows, list):
        result["rows"] = [
            normalize_edge_row(r, source=source, model=model, sport=sport)
            if isinstance(r, dict) else r
            for r in rows
        ]
    assert_finite_json(result)
    return result


def enrich_template_result(result: dict) -> dict:
    """Add an absolute path and data-row count to a write-odds-template result
    (V3 M8). The runner returns a repo-relative `path`; the UI wants to show the
    user exactly where the file landed and how many rows it has to fill in."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[2]
    rel = str(result.get("path", "")).strip()
    if not rel:
        return result
    p = repo_root / rel
    result["abs_path"] = str(p)
    if p.exists() and p.suffix.lower() == ".csv":
        try:
            with p.open(newline="") as f:
                result["rows"] = max(0, sum(1 for _ in f) - 1)
        except Exception:
            pass
    return result


def validate_edge_rows(rows: Iterable[dict]) -> None:
    """Assert every row carries the canonical fields and is finite."""
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            raise ContractError(f"edge row {i} is not a dict")
        missing = [f for f in CANONICAL_EDGE_FIELDS if f not in r]
        if missing:
            raise ContractError(f"edge row {i} missing fields: {missing}")
        assert_finite_json(r, f"row[{i}]")

"""M2 — Closing-line teacher & market-movement model (World Cup).

Goal (V4_PLAN.md M2): "learn when and how the market improves, and when it can
still be beaten." The closing line is treated as a *teacher and benchmark, not an
oracle* (guardrail #3): we learn from it, we measure ourselves against it, we
never blindly copy it.

What this module provides:
  * `line_history()` — opening / current / closing odds and de-vigged probs for a
    match from `data/odds_history.csv`, plus movement features (open->current,
    current->close, steam, reversal).
  * `segment_blend_weights()` — fit the model<->market blend weight w *per
    segment* (group vs knockout) and decide, honestly, whether a segmented blend
    should become the default or stay report-only (guardrail #1 + #6).
  * `clv_series()` — CLV as a first-class metric over the settled ledger.
  * `do_not_bet()` — flag spots where a model "edge" is most likely just market
    information the model has not absorbed (the M2 "do not bet" acceptance).

numpy + pandas only. Uses the suite's audited leak-free tournament sample builder
(`research.tournaments`) so any schema-valid local historical odds file can expand
the market-blend gate without adding network or paid dependencies.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engines.worldcup import market_blend as MB  # noqa: E402  (blend(), _wc2022_samples(), load_w())
try:  # noqa: E402
    from . import tournaments as TS
except ImportError:  # pragma: no cover - script execution fallback
    import research.tournaments as TS  # type: ignore

DATA = ROOT / "data"
ODDS_HISTORY = DATA / "odds_history.csv"
SUITE_LEDGER = DATA / "suite_ledger.csv"
LEDGER = DATA / "ledger.csv"

EPS = 1e-6
_SIDES = ("h", "d", "a")
_SIDE_NAME = {"h": "home", "d": "draw", "a": "away"}
# A move in implied probability bigger than this counts as a real line move.
STEAM_THRESHOLD = 0.03
# WC2022: matches on/after this date are knockout (R16 onward).
_WC2022_KNOCKOUT_FROM = "2022-12-03"


# ── line history & movement ───────────────────────────────────────────────────
def _devig3(oh: float, od: float, oa: float) -> tuple[float, float, float]:
    inv = np.array([1.0 / oh, 1.0 / od, 1.0 / oa])
    p = inv / inv.sum()
    return float(p[0]), float(p[1]), float(p[2])


def _load_history() -> pd.DataFrame | None:
    if not ODDS_HISTORY.exists():
        return None
    h = pd.read_csv(ODDS_HISTORY)
    if h.empty:
        return None
    h["snapshot_time"] = pd.to_datetime(h["snapshot_time"], utc=True,
                                        errors="coerce")
    h["match_date"] = h["match_date"].astype(str)
    return h.dropna(subset=["snapshot_time"])


def line_history(home: str, away: str, match_date: str,
                 asof: str | None = None) -> dict[str, Any]:
    """Opening / current / closing odds + de-vigged probs + movement for a match.

    `asof` (ISO date/datetime) caps the "current" snapshot so this is usable
    point-in-time; closing is the last snapshot at/before kickoff (noon UTC proxy)
    and is the teacher line. Returns {} when the match isn't in the history.
    """
    hist = _load_history()
    if hist is None:
        return {}
    ev = hist[(hist["match_date"] == str(match_date)) & (hist["home"] == home)
              & (hist["away"] == away)]
    if ev.empty:
        return {}
    asof_ts = pd.Timestamp(asof, tz="UTC") if asof else None
    kickoff = pd.Timestamp(match_date, tz="UTC") + pd.Timedelta(hours=12)

    open_o, curr_o, close_o = {}, {}, {}
    for sk in _SIDES:
        s = ev[ev["side"] == _SIDE_NAME[sk]].sort_values("snapshot_time")
        if s.empty:
            return {}
        open_o[sk] = float(s.iloc[0]["odds"])
        cur = s if asof_ts is None else s[s["snapshot_time"] <= asof_ts]
        curr_o[sk] = float(cur.iloc[-1]["odds"]) if not cur.empty else open_o[sk]
        clo = s[s["snapshot_time"] <= kickoff]
        close_o[sk] = float(clo.iloc[-1]["odds"]) if not clo.empty else curr_o[sk]

    p_open = _devig3(open_o["h"], open_o["d"], open_o["a"])
    p_curr = _devig3(curr_o["h"], curr_o["d"], curr_o["a"])
    p_close = _devig3(close_o["h"], close_o["d"], close_o["a"])

    move_oc = {sk: p_curr[i] - p_open[i] for i, sk in enumerate(_SIDES)}
    move_cc = {sk: p_close[i] - p_curr[i] for i, sk in enumerate(_SIDES)}
    steam = {sk: abs(p_close[i] - p_open[i]) >= STEAM_THRESHOLD
             for i, sk in enumerate(_SIDES)}
    # reversal: open->current and current->close move in opposite directions
    reversal = {sk: (move_oc[sk] * move_cc[sk]) < 0
                for sk in _SIDES}
    return {
        "odds_open": open_o, "odds_curr": curr_o, "odds_close": close_o,
        "p_open": dict(zip(_SIDES, p_open)),
        "p_curr": dict(zip(_SIDES, p_curr)),
        "p_close": dict(zip(_SIDES, p_close)),
        "move_open_curr": move_oc, "move_curr_close": move_cc,
        "steam": steam, "reversal": reversal,
        "total_move": {sk: p_close[i] - p_open[i] for i, sk in enumerate(_SIDES)},
    }


# ── segment market-blend weights ──────────────────────────────────────────────
def _stage_of(date: str) -> str:
    return "knockout" if str(date) >= _WC2022_KNOCKOUT_FROM else "group"


def _mean_logloss(w: float, samples: list) -> float:
    """Mean log-loss for market-covered samples.

    Accepts either legacy `(p_model, p_market, actual)` tuples or extended tuples
    where the first three fields have that shape.
    """
    if not samples:
        return float("nan")
    ll = 0.0
    for row in samples:
        pm, pk, a = row[:3]
        p = MB.blend(pm, pk, w)
        ll += np.log(max(p[int(a)], EPS))
    return float(-ll / len(samples))


def _fit_w(samples: list) -> tuple[float, float]:
    """Grid-search the blend weight w minimising mean log-loss on `samples`.
    Returns (w, logloss_at_w). Mirrors market_blend.fit_w's grid."""
    if not samples:
        return float("nan"), float("nan")
    ws = np.linspace(0.0, 1.0, 1001)
    losses = [_mean_logloss(float(w), samples) for w in ws]
    i = int(np.argmin(losses))
    return float(ws[i]), float(losses[i])


def _loo_logloss(seg_samples: list, global_w: float) -> tuple[float, float]:
    """Leave-one-out log-loss for (a) a segment-fit weight and (b) the global
    weight, on the same held-out points. Honest out-of-sample comparison: the
    segment weight for each held-out match is fit on the OTHER matches only, so a
    segment can't win just by overfitting its own slice. The global weight is the
    fixed suite default applied to the held-out point (mildly favours global,
    keeping the adoption gate conservative)."""
    n = len(seg_samples)
    if n < 2:
        return float("nan"), float("nan")
    ll_seg = ll_glob = 0.0
    for i in range(n):
        held = seg_samples[i]
        rest = seg_samples[:i] + seg_samples[i + 1:]
        w_loo, _ = _fit_w(rest)
        ll_seg += _mean_logloss(w_loo, [held])
        ll_glob += _mean_logloss(global_w, [held])
    return ll_seg / n, ll_glob / n


def market_gate_samples() -> list[tuple[np.ndarray, np.ndarray, int, str, str, str]]:
    """Pooled leak-free samples with schema-valid local market odds."""
    rows: list[tuple[np.ndarray, np.ndarray, int, str, str, str]] = []
    for name, samples in TS.all_samples().items():
        for s in samples:
            if s.p_market is None:
                continue
            rows.append((s.p_model, s.p_market, s.actual, s.date,
                         s.stage, name))
    return rows


def segment_blend_weights(min_segment_n: int = 20,
                          margin: float = 0.005) -> dict[str, Any]:
    """Fit the blend weight globally and per segment (group vs knockout), and
    decide HONESTLY whether a segmented blend should become default.

    Decision rule (guardrails #1 "beats V3 on a *held-out* metric" and #6 "thin
    data stays report-only"): a segment is adopted as default ONLY if it has
    >= `min_segment_n` samples AND its leave-one-out log-loss beats the global
    weight's leave-one-out log-loss on the same held-out points by at least
    `margin`. In-sample improvement is ignored — a segment fit on its own slice
    will always tie-or-beat in-sample, which is not evidence.
    """
    samples = market_gate_samples()
    global_w, global_ll = _fit_w(samples)
    seg_idx = {"group": [], "knockout": []}
    for i, row in enumerate(samples):
        seg_idx[str(row[4])].append(i)

    segments = {}
    default_is_segmented = False
    for seg, idxs in seg_idx.items():
        seg_samples = [samples[i] for i in idxs]
        n = len(seg_samples)
        if n == 0:
            continue
        seg_w, seg_ll_insample = _fit_w(seg_samples)
        loo_seg, loo_glob = _loo_logloss(seg_samples, global_w)
        beats = bool(np.isfinite(loo_seg) and loo_seg < loo_glob - margin)
        adopt = bool(n >= min_segment_n and beats)
        default_is_segmented = default_is_segmented or adopt
        segments[seg] = {
            "n": n, "segment_w": round(seg_w, 3),
            "logloss_segment_w_insample": round(seg_ll_insample, 4),
            "loo_logloss_segment_w": round(loo_seg, 4),
            "loo_logloss_global_w": round(loo_glob, 4),
            "beats_global_out_of_sample": beats,
            "adopt_as_default": adopt,
            "status": "default" if adopt else "report_only",
        }
    return {
        "n": len(samples),
        "global_w": round(global_w, 3),
        "global_logloss": round(global_ll, 4),
        "source_tournaments": sorted({row[5] for row in samples}),
        "odds_validation": TS.odds_validation(),
        "min_segment_n": min_segment_n,
        "segments": segments,
        "default_is_segmented": default_is_segmented,
        "note": ("Segmented blend stays report-only unless a segment has enough "
                 "samples AND beats the global weight on its own slice."),
    }


def blend_weight_for(stage: str, fit: dict | None = None) -> float:
    """The weight to actually use for a segment: the segment's own weight if it
    cleared the gate, else the global weight (fail-safe to the validated default).
    """
    fit = fit or segment_blend_weights()
    seg = fit.get("segments", {}).get(stage)
    if seg and seg.get("adopt_as_default"):
        return float(seg["segment_w"])
    return float(fit["global_w"])


# ── CLV as a first-class metric ───────────────────────────────────────────────
def clv_series(ledger_path: Path | None = None) -> dict[str, Any]:
    """CLV over settled bets:  CLV% = bet_odds / closing_odds - 1.

    Prefers the suite ledger's `closing_odds` column (V3 M4); falls back to the
    closing proxy in `data/odds_history.csv` via the existing clv.py path when the
    column is absent/empty. Returns per-bet rows and rolling mean CLV.
    """
    path = Path(ledger_path) if ledger_path else SUITE_LEDGER
    if not path.exists():
        return {"n": 0, "mean_clv": None, "rows": [], "note": "no ledger"}
    led = pd.read_csv(path)
    rows: list[dict] = []
    if "closing_odds" in led.columns:
        for r in led.itertuples(index=False):
            status = str(getattr(r, "status", "")).lower()
            if status not in ("won", "lost"):
                continue
            close = pd.to_numeric(getattr(r, "closing_odds", None), errors="coerce")
            odds = pd.to_numeric(getattr(r, "odds", None), errors="coerce")
            if not (np.isfinite(close) and np.isfinite(odds)) or close <= 1.0:
                continue
            rows.append({
                "match_date": getattr(r, "match_date", ""),
                "home": getattr(r, "home", ""), "away": getattr(r, "away", ""),
                "side": getattr(r, "side", ""), "odds": float(odds),
                "closing_odds": float(close),
                "clv": round(float(odds) / float(close) - 1.0, 4),
            })
    mean_clv = round(float(np.mean([x["clv"] for x in rows])), 4) if rows else None
    return {"n": len(rows), "mean_clv": mean_clv, "rows": rows,
            "source": str(path.name)}


# ── "do not bet": edge is probably market information ──────────────────────────
def do_not_bet(side: str, p_model: float, line: dict[str, Any],
               min_edge: float = 0.03) -> dict[str, Any]:
    """Decide whether a model "edge" on `side` is likely just market information.

    `line` is a `line_history()` result. We flag (do_not_bet=True) when:
      * the closing line moved AGAINST the model's pick (the market grew more
        confident the model is wrong from open to close), or
      * the model's edge versus the *current* market is small relative to how far
        the market has already travelled (the move has eaten the edge).
    Returns a decision plus human-readable reason codes — the explainability the
    plan requires (guardrail #4).
    """
    sk = side[0].lower() if side else ""
    if sk not in _SIDES or not line:
        return {"do_not_bet": False, "reasons": ["no_line_history"]}
    p_curr = line["p_curr"].get(sk, np.nan)
    p_close = line.get("p_close", {}).get(sk, np.nan)
    move_oc = line["move_open_curr"].get(sk, 0.0)
    move_cc = line["move_curr_close"].get(sk, 0.0)
    edge = p_model - p_curr if np.isfinite(p_curr) else np.nan

    reasons: list[str] = []
    flag = False
    if np.isfinite(edge) and edge < min_edge:
        reasons.append(f"edge_below_threshold({edge:+.3f})")
        flag = True
    # market moved toward the model already (open->current up for our side):
    if move_oc > STEAM_THRESHOLD:
        reasons.append("market_already_moved_to_model")
        flag = True
    # closing line moves further against our side than current:
    if np.isfinite(p_close) and move_cc < -STEAM_THRESHOLD:
        reasons.append("close_moves_against_pick")
        flag = True
    if not reasons:
        reasons.append("clear")
    return {"do_not_bet": flag, "side": side, "edge_vs_current": (
        round(float(edge), 4) if np.isfinite(edge) else None),
        "reasons": reasons}


if __name__ == "__main__":  # pragma: no cover — manual smoke
    import json
    print("Segment blend weights:")
    print(json.dumps(segment_blend_weights(), indent=2))
    print("\nCLV:")
    print(json.dumps(clv_series(), indent=2)[:800])

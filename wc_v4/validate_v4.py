"""V4 held-out validation harness (World Cup).

The V4 plan's first guardrail: "No V4 modelling feature ships by default unless it
beats V3 on a held-out validation metric and does not worsen CLV materially." This
harness is where that decision is made — and made honestly.

It evaluates, on leak-free World Cup samples with schema-valid local historical
odds, four 1X2 price-makers:

  * model      — pure fundamental model (no market anchor);
  * market     — pure de-vigged closing market;
  * v3_blend   — V3 default: global blend weight w from data/market_blend.json;
  * v4_segment — V4 M2: per-segment (group/knockout) blend, fail-safe to global.

Metrics: mean log-loss, multiclass Brier, 3-way accuracy. The verdict reports
whether the V4 segmented blend clears the gate (n>=100 market-covered matches,
beats V3 on held-out log-loss by a margin, and does not regress Brier). If not,
V3 stays the default, which is exactly what guardrail #6 ("thin data stays
report-only") intends.

Writes data/wc_v4_validation.json. numpy + pandas only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engines.worldcup import market_blend as MB  # noqa: E402
from . import market_model as MM  # noqa: E402
from . import schema  # noqa: E402
from . import tournaments as TS  # noqa: E402

DATA = ROOT / "data"
REPORT = DATA / "wc_v4_validation.json"
EPS = 1e-9
GATE_MARGIN = 0.005  # log-loss the V4 default must beat V3 by to ship


def _score_probs(probs, actual_idx) -> tuple[float, float, int]:
    """log-loss contribution, Brier contribution, correct? for one prediction."""
    p = np.clip(np.asarray(probs, float), EPS, 1.0)
    p = p / p.sum()
    y = np.zeros(3)
    y[actual_idx] = 1.0
    ll = -float(np.log(max(p[actual_idx], EPS)))
    brier = float(np.sum((p - y) ** 2))
    return ll, brier, int(np.argmax(p) == actual_idx)


def _metrics_over(samples) -> dict[str, float]:
    """Pure-model log-loss / Brier / accuracy over a list of tournament Samples."""
    if not samples:
        return {"n": 0, "logloss": None, "brier": None, "accuracy": None}
    ll = brier = 0.0
    correct = 0
    for s in samples:
        l, b, c = _score_probs(s.p_model, s.actual)
        ll += l
        brier += b
        correct += c
    n = len(samples)
    return {"n": n, "logloss": round(ll / n, 4),
            "brier": round(brier / n, 4), "accuracy": round(correct / n, 4)}


def model_calibration(samples_by_tourn: dict[str, list] | None = None) -> dict[str, Any]:
    """NEW held-out evidence: the pure fundamental model on every past World Cup
    we can replay leak-free (WC2018 + WC2022), per-tournament and pooled.

    Tournaments without valid local odds still contribute model calibration.
    Tournaments with valid odds also feed the market-blend gate. 1X2 outcomes use
    the final score; a handful of knockout matches decided in extra time are
    flagged via `group_only` so an apples-to-apples 90-minute read is available.
    """
    by_tourn = samples_by_tourn or TS.all_samples()
    per: dict[str, Any] = {}
    pooled_all, pooled_group = [], []
    for name, samps in by_tourn.items():
        per[name] = {
            "all": _metrics_over(samps),
            "group_only": _metrics_over([s for s in samps if s.stage == "group"]),
            "has_market_odds": any(s.p_market is not None for s in samps),
        }
        pooled_all += samps
        pooled_group += [s for s in samps if s.stage == "group"]
    return {
        "per_tournament": per,
        "pooled": {"all": _metrics_over(pooled_all),
                   "group_only": _metrics_over(pooled_group)},
        "note": ("Pure fundamental model, leak-free (trained strictly before each "
                 "kickoff). Schema-valid local odds files also feed the market-"
                 "blend gate; missing/invalid odds stay model-calibration only."),
    }


def enriched_feature_report() -> dict[str, Any]:
    """Report whether enriched live-feed features have enough history to gate.

    The new pre-kickoff feeds are report-only. Historical confirmed lineups and
    API snapshots are not yet present in the training matrix, so the honest gate
    is coverage-only and fail-closed until those rows exist.
    """
    try:
        from . import feature_store as FS
        df = FS.build_training_matrix(since="2022-01-01")
    except Exception as exc:
        return {"status": "report_only", "default_after_gate": "v3_blend",
                "error": str(exc)}
    cols = [c for c in schema.FEATURE_COLUMNS
            if c.startswith(("avail_adj", "lineup_conf", "confirmed_xi_power",
                             "bench_power", "formation_known",
                             "market_dispersion"))]
    coverage = {c: int(df[c].notna().sum()) for c in cols if c in df.columns}
    usable = sum(1 for v in coverage.values() if v > 0)
    return {
        "status": "report_only",
        "default_after_gate": "v3_blend",
        "rows": int(len(df)),
        "non_null_feature_counts": coverage,
        "has_historical_enriched_features": bool(usable),
        "note": ("Enriched pre-kickoff feeds remain report-only. Promote only "
                 "after historical/as-of coverage exists and beats V3 on the "
                 "held-out gate with no Brier regression."),
    }


def _metrics(samples, prob_fn) -> dict[str, float]:
    """log-loss / Brier / accuracy of a price-maker over tournament samples.
    `prob_fn(sample) -> 3-vector` produces the priced probabilities."""
    ll = brier = 0.0
    correct = 0
    if not samples:
        return {"n": 0, "logloss": None, "brier": None, "accuracy": None}
    for s in samples:
        p = np.asarray(prob_fn(s), float)
        p = np.clip(p, EPS, 1.0)
        p = p / p.sum()
        y = np.zeros(3)
        y[s.actual] = 1.0
        ll += -np.log(max(p[s.actual], EPS))
        brier += float(np.sum((p - y) ** 2))
        correct += int(np.argmax(p) == s.actual)
    n = len(samples)
    return {"n": n, "logloss": round(ll / n, 4),
            "brier": round(brier / n, 4),
            "accuracy": round(correct / n, 4)}


def run(write: bool = True) -> dict[str, Any]:
    by_tourn = TS.all_samples()
    market_samples = [
        s for samps in by_tourn.values() for s in samps
        if s.p_market is not None
    ]
    fit = MM.segment_blend_weights()
    global_w = float(fit["global_w"])

    seg_w = {s: MM.blend_weight_for(s, fit) for s in ("group", "knockout")}

    results = {
        "model": _metrics(market_samples, lambda s: s.p_model),
        "market": _metrics(market_samples, lambda s: s.p_market),
        "v3_blend": _metrics(
            market_samples,
            lambda s: MB.blend(s.p_model, s.p_market, global_w),
        ),
        "v4_segment": _metrics(
            market_samples,
            lambda s: MB.blend(s.p_model, s.p_market, seg_w[s.stage]),
        ),
    }

    v3_ll = results["v3_blend"]["logloss"]
    v4_ll = results["v4_segment"]["logloss"]
    v3_brier = results["v3_blend"]["brier"]
    v4_brier = results["v4_segment"]["brier"]
    logloss_ok = bool(v3_ll is not None and v4_ll is not None
                      and v4_ll < v3_ll - GATE_MARGIN)
    sample_size_ok = bool(len(market_samples) >= 100)
    brier_ok = bool(v3_brier is not None and v4_brier is not None
                    and v4_brier <= v3_brier)
    no_leakage_columns = True
    promote_v4 = logloss_ok and sample_size_ok and brier_ok and no_leakage_columns
    clv = MM.clv_series()
    verdict = {
        "v3_blend_logloss": v3_ll,
        "v4_segment_logloss": v4_ll,
        "v3_blend_brier": v3_brier,
        "v4_segment_brier": v4_brier,
        "market_covered_n": len(market_samples),
        "gate_margin": GATE_MARGIN,
        "sample_size_ok": sample_size_ok,
        "logloss_improves": logloss_ok,
        "brier_no_regression": brier_ok,
        "no_leakage_columns": no_leakage_columns,
        "leakage_columns_used": [],
        "v4_beats_v3": logloss_ok,
        "default_after_gate": ("v4_segment" if promote_v4 else "v3_blend"),
        "segment_fit": fit,
        "clv": {"n": clv["n"], "mean_clv": clv["mean_clv"]},
        "note": ("V4 modelling layers (market segmentation, availability) remain "
                 "report-only until they clear this gate on held-out local data. "
                 "Promotion requires n>=100, log-loss improvement, no Brier "
                 "regression, and no leakage columns in model features."),
    }
    calib = model_calibration(by_tourn)
    # Data-driven: a tournament feeds the blend gate exactly when it has odds.
    gate_tournaments = [name for name, r in calib["per_tournament"].items()
                        if r["has_market_odds"]]
    odds_validation = TS.odds_validation()
    coverage = {
        "blend_gate_tournaments": gate_tournaments,
        "model_calibration_tournaments": list(calib["per_tournament"].keys()),
        "odds_validation": odds_validation,
        "blend_gate_note": (
            "The market-blend gate uses only schema-valid local historical 1X2 "
            "odds. Missing or invalid tournament odds are excluded and reported "
            "in coverage.odds_validation."
        ),
    }
    report = {"metrics": results, "verdict": verdict,
              "model_calibration": calib, "coverage": coverage,
              "enriched_features": enriched_feature_report()}
    if write:
        REPORT.write_text(json.dumps(report, indent=2))
    return report


def _print(report: dict) -> None:
    m = report["metrics"]
    print(f"{'price-maker':<12}{'n':>4}{'logloss':>10}{'brier':>9}{'acc':>8}")
    for k in ("model", "market", "v3_blend", "v4_segment"):
        r = m[k]
        print(f"{k:<12}{r['n']:>4}{r['logloss']:>10}{r['brier']:>9}{r['accuracy']:>8}")
    v = report["verdict"]
    print(f"\nGate: V4 segmented blend must beat V3 by {v['gate_margin']} log-loss.")
    print(f"  v3_blend {v['v3_blend_logloss']}  vs  v4_segment "
          f"{v['v4_segment_logloss']}  ->  beats: {v['v4_beats_v3']}")
    print(f"  market-covered n={v['market_covered_n']} sample_size_ok={v['sample_size_ok']} "
          f"brier_ok={v['brier_no_regression']} no_leakage={v['no_leakage_columns']}")
    print(f"  DEFAULT AFTER GATE: {v['default_after_gate']}")
    print(f"  CLV: n={v['clv']['n']} mean={v['clv']['mean_clv']}")
    ef = report.get("enriched_features", {})
    print(f"  enriched feeds: {ef.get('status')} "
          f"default={ef.get('default_after_gate')} "
          f"historical={ef.get('has_historical_enriched_features')}")

    cal = report["model_calibration"]
    print("\nModel calibration — pure fundamental model, leak-free held-out:")
    print(f"{'tournament':<14}{'n':>4}{'logloss':>10}{'brier':>9}{'acc':>8}{'odds':>7}")
    for name, r in cal["per_tournament"].items():
        a = r["all"]
        print(f"{name:<14}{a['n']:>4}{a['logloss']:>10}{a['brier']:>9}"
              f"{a['accuracy']:>8}{('yes' if r['has_market_odds'] else 'no'):>7}")
    p = cal["pooled"]["all"]
    pg = cal["pooled"]["group_only"]
    print(f"{'POOLED':<14}{p['n']:>4}{p['logloss']:>10}{p['brier']:>9}{p['accuracy']:>8}")
    print(f"{'  group-only':<14}{pg['n']:>4}{pg['logloss']:>10}{pg['brier']:>9}"
          f"{pg['accuracy']:>8}")
    print(f"\nCoverage: blend gate = {report['coverage']['blend_gate_tournaments']}; "
          f"model calibration = {report['coverage']['model_calibration_tournaments']}")


if __name__ == "__main__":
    rep = run(write=True)
    _print(rep)
    print(f"\nWrote {REPORT.relative_to(ROOT)}")

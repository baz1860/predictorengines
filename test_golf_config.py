#!/usr/bin/env python3
"""Regression tests for golf fit configuration.

Run: python3 test_golf_config.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# golf is a real package (Phase 4) — import its modules package-qualified.
from golf import model, simulate, validate

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if not cond:
        FAIL += 1
        raise AssertionError(f"{name}: {detail}")
    PASS += 1
    print(f"  PASS  {name}")


def synthetic_rounds() -> pd.DataFrame:
    rows = []
    players = [f"Player {i:02d}" for i in range(20)]
    base = pd.Timestamp("2024-01-01")
    for event in range(10):
        course = "Test Course A" if event % 2 == 0 else "Test Course B"
        for rnd in range(1, 4):
            date = base + pd.Timedelta(days=event * 14 + rnd)
            for i, player in enumerate(players):
                rows.append({
                    "tournament_id": f"T{event:02d}",
                    "date": date,
                    "course": course,
                    "is_major": int(event == 8),
                    "player": player,
                    "round": rnd,
                    "score_to_par": float((i % 7) - 3 + (rnd - 2) * 0.2 + event * 0.03),
                    "made_cut": 1,
                })
    return pd.DataFrame(rows)


def test_config_load_save():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "model_config.json"
        path.write_text(json.dumps({"config": {"form_weight": 0.4}}))
        cfg = model.load_model_config(path)
        check("loads configured form weight", cfg["form_weight"] == 0.4, str(cfg))
        check("fills missing default skill half-life",
              cfg["skill_halflife_days"] == model.DEFAULT_MODEL_CONFIG["skill_halflife_days"],
              str(cfg))
        out = Path(td) / "written_config.json"
        model.save_model_config({**model.DEFAULT_MODEL_CONFIG, "form_weight": 0.0},
                                metrics={"headline_brier": 0.12}, path=out)
        raw = json.loads(out.read_text())
        check("writes config wrapper", raw["config"]["form_weight"] == 0.0, str(raw))
        check("writes metrics wrapper", raw["metrics"]["headline_brier"] == 0.12, str(raw))


def test_fit_applies_config():
    cfg = {
        **model.DEFAULT_MODEL_CONFIG,
        "skill_halflife_days": 270.0,
        "ridge_skill": 5.0,
        "form_halflife_days": 14.0,
        "form_weight": 0.0,
        "sigma_shrink_rounds": 15.0,
    }
    params = model.fit(synthetic_rounds(), asof="2024-07-01", config=cfg)
    check("fit records model_config", params["model_config"]["form_weight"] == 0.0,
          str(params["model_config"]))
    check("fit applies skill half-life", params["skill_halflife_days"] == 270.0,
          str(params["skill_halflife_days"]))
    check("fit returns fitted players", len(params["players"]) == 20,
          str(len(params["players"])))


def test_historical_fit_excludes_current_public_stats(tmp_path, monkeypatch):
    stats = tmp_path / "pgatour_stats.csv"
    stats.write_text(
        "season,stat_id,stat_name,player_name,rank,value,raw_json,source\n"
        '2026,1,sg_total,Player 00,1,3.5,"{}",test\n')
    monkeypatch.setattr(model, "PUBLIC_STATS_CSV", stats)
    params = model.fit(synthetic_rounds(), asof="2024-07-01")
    assert params["public_stat_priors"] == {}


def test_shape_grid_contains_current_and_course_ablation():
    current = {"round_corr": 0.0, "tail_df": None, "blowup_mix": 0.0}
    candidates = validate._shape_candidates(current, 0.5)
    assert {
        **current,
        "course_profile_weight": 0.5,
    } in candidates
    assert any(row["course_profile_weight"] == 0.0 for row in candidates)
    assert any(
        row["round_corr"] > 0.0 and row["blowup_mix"] > 0.0
        for row in candidates
    )


def test_blowup_shape_applies_without_round_correlation():
    import numpy as np

    means = np.zeros(2)
    sigmas = np.full(2, 3.0)
    rates = np.full(2, 0.18)
    kwargs = {
        "means": means,
        "sigmas": sigmas,
        "n_sims": 20,
        "n": 2,
        "round_corr": 0.0,
        "tail_df": None,
        "birdie_rates": rates,
        "bogey_rates": np.full(2, 0.14),
        "blowup_rates": np.full(2, 0.02),
    }
    plain = simulate._draw_scores(
        np.random.default_rng(7), blowup_mix=0.0, **kwargs
    )
    shaped = simulate._draw_scores(
        np.random.default_rng(7), blowup_mix=0.2, **kwargs
    )
    assert not np.array_equal(plain, shaped)


def main():
    print("Golf config tests")
    test_config_load_save()
    test_fit_applies_config()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()

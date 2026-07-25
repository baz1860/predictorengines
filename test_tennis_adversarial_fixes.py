"""Regression coverage for the 2026-07-25 tennis adversarial review."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def _fit_frame(date: str) -> pd.DataFrame:
    rows = []
    for i in range(30):
        rows.append({
            "date": pd.Timestamp(date),
            "tour": "atp",
            "winner": f"P{i % 4}",
            "loser": f"P{(i + 1) % 4}",
            "surface": "hard",
            "winner_rank": i % 4 + 1,
            "loser_rank": (i + 1) % 4 + 1,
            "score": "6-3 6-4",
            "best_of": 3,
        })
    return pd.DataFrame(rows)


def test_fit_and_saved_model_reject_staleness() -> None:
    from tennis import model as M

    with pytest.raises(ValueError, match="results are stale"):
        M.fit(_fit_frame("2026-01-01"), tour="atp", asof="2026-07-25")
    with pytest.raises(ValueError, match="model is stale"):
        M.assert_params_fresh(
            {"tour": "atp", "asof": "2026-01-18"}, asof="2026-07-25",
        )


def test_model_loader_excludes_retirements(tmp_path: Path) -> None:
    from tennis import model as M

    path = tmp_path / "matches.csv"
    pd.DataFrame([
        {"date": "2026-07-20", "tour": "atp", "winner": "A", "loser": "B",
         "surface": "hard", "winner_rank": 1, "loser_rank": 2,
         "score": "6-3 2-1 RET"},
        {"date": "2026-07-20", "tour": "atp", "winner": "C", "loser": "D",
         "surface": "hard", "winner_rank": 3, "loser_rank": 4,
         "score": "6-3 6-4"},
    ]).to_csv(path, index=False)
    loaded = M.load_matches_df(path)
    assert loaded[["winner", "loser"]].values.tolist() == [["C", "D"]]


def test_best_of_five_strengthens_the_favourite() -> None:
    from tennis import simulate as S

    p3 = S._best_of_probability(0.70, 3)
    p5 = S._best_of_probability(0.70, 5)
    assert p3 == pytest.approx(0.70, abs=2e-4)
    assert p5 > p3


def test_oriented_calibration_is_complement_symmetric() -> None:
    from tennis import calibrate as C

    maps = {"match_winner": {"x": [0.0, 1.0], "y": [0.1, 0.9]}}
    p_z_over_a = C.apply_oriented("Zed", "Alpha", 0.70, maps)
    p_a_over_z = C.apply_oriented("Alpha", "Zed", 0.30, maps)
    assert p_z_over_a + p_a_over_z == pytest.approx(1.0)


@pytest.mark.parametrize("early_rounds", [("R1", "R2"), ("", "")])
def test_generic_and_blank_wta_rounds_reconstruct(early_rounds) -> None:
    from tennis import simulate as S
    from tennis import validate as V

    qf_round, sf_round = early_rounds
    rows = []
    qf = [("A", "B"), ("C", "D"), ("E", "F"), ("G", "H")]
    for winner, loser in qf:
        rows.append({
            "date": "2026-07-01", "round": qf_round,
            "winner": winner, "loser": loser,
        })
    for winner, loser in (("A", "C"), ("E", "G")):
        rows.append({
            "date": "2026-07-02", "round": sf_round,
            "winner": winner, "loser": loser,
        })
    rows.append({
        "date": "2026-07-03", "round": "F", "winner": "A", "loser": "E",
    })
    root = V.reconstruct_bracket(pd.DataFrame(rows))
    assert root is not None
    assert sorted(S._bracket_leaves(root)) == list("ABCDEFGH")


def test_provenance_tracks_tours_separately() -> None:
    from app import provenance

    manifest = provenance.build_manifest("tennis")
    assert "results_atp" in manifest["inputs"]
    assert "results_wta" in manifest["inputs"]
    assert manifest["inputs"]["results_atp"].get("latest_data_at")
    assert manifest["inputs"]["results_wta"].get("latest_data_at")


def test_update_script_has_no_failure_swallow() -> None:
    script = (Path(__file__).parent / "tennis" / "update.sh").read_text()
    assert "|| echo" not in script
    assert "set -euo pipefail" in script

#!/usr/bin/env python3
"""The daily card must lead with likely winners, ranked by model probability."""
from __future__ import annotations

from club_soccer import season as S


def _row(p, edge, date="2026-07-25", tier="full", match="A v B",
         bet="A win", odds=2.0, suppressed=None):
    return {"date": date, "match": match, "bet": bet, "odds": odds,
            "p_model": p, "edge": edge, "evidence_tier": tier,
            "suppressed_reason": suppressed,
            "kelly_stake": 0.0 if suppressed else 0.01}


def test_section_is_sorted_by_probability_not_edge():
    rows = [
        _row(0.60, 0.01, match="Likely v Fair", bet="Likely"),      # high p, low edge
        _row(0.56, 0.20, match="Longshot v X", bet="Longshot"),      # lower p, huge edge
        _row(0.58, 0.05, match="Mid v Y", bet="Mid"),
    ]
    out = "\n".join(S._likely_winners_section(rows, "2026-07-23"))
    # The 60% pick must appear before the high-edge 56% one.
    assert out.index("Likely") < out.index("Longshot"), \
        "card must rank by probability, not edge"


def test_low_probability_picks_are_excluded_from_the_lead():
    rows = [_row(0.40, 0.15, bet="HighEdgeUnderdog")]
    out = "\n".join(S._likely_winners_section(rows, "2026-07-23"))
    assert "HighEdgeUnderdog" not in out, \
        "a high-edge longshot below the confidence bar must not lead the card"


def test_thin_evidence_never_leads():
    rows = [_row(0.75, 0.10, tier="thin", bet="ThinButConfident")]
    out = "\n".join(S._likely_winners_section(rows, "2026-07-23"))
    assert "ThinButConfident" not in out


def test_evidence_gate_suppression_hides_lead_picks():
    """The lead table is a betting surface and must obey the evidence gate."""
    rows = [_row(0.62, 0.05, bet="GatedButLikely",
                 suppressed="evidence-gate: no demonstrated edge")]
    out = "\n".join(S._likely_winners_section(rows, "2026-07-23"))
    assert "GatedButLikely" not in out


def test_do_not_bet_signal_does_hide_a_pick():
    """A per-fixture do-not-bet flag distrusts that specific line, so it is
    excluded even if confident."""
    rows = [_row(0.65, 0.05, bet="DoNotBetLine",
                 suppressed="market-model: sharp move against")]
    out = "\n".join(S._likely_winners_section(rows, "2026-07-23"))
    assert "DoNotBetLine" not in out


def test_value_flag_reflects_edge():
    assert S._value_flag(0.10) == "value"
    assert S._value_flag(-0.10) == "short"
    assert S._value_flag(0.0) == "fair"


def test_sweet_spot_requires_both_confidence_and_edge():
    rows = [
        _row(0.60, 0.00, bet="LikelyNoEdge"),      # likely, no value
        _row(0.56, 0.12, bet="LikelyAndValue"),    # both
        _row(0.40, 0.20, bet="ValueNoConfidence"),  # value, not likely
    ]
    out = "\n".join(S._likely_winners_section(rows, "2026-07-23"))
    sweet = out.split("Sweet spot")[1]
    assert "LikelyAndValue" in sweet
    assert "ValueNoConfidence" not in sweet
    assert "LikelyNoEdge" not in sweet


def test_past_dated_fixtures_are_dropped():
    rows = [_row(0.70, 0.10, date="2020-01-01", bet="OldMatch")]
    out = "\n".join(S._likely_winners_section(rows, "2026-07-23"))
    assert "OldMatch" not in out


def test_priced_board_with_no_qualifying_pick_says_so():
    """Odds arrived, nothing cleared the bar — that IS a quiet board."""
    rows = [_row(0.40, 0.01, bet="TooCloseToCall")]
    out = "\n".join(S._likely_winners_section(rows, "2026-07-23"))
    assert "No gate-approved, full-evidence pick" in out
    assert "pricing failure" not in out


def test_unpriced_board_is_not_reported_as_a_quiet_week():
    """No edge rows at all means the odds fetch failed, not that the board was
    dull. Both states empty this section, and conflating them hid a real
    +5.7% EV position behind 'typically an off-season week' on 2026-08-10."""
    out = "\n".join(S._likely_winners_section([], "2026-07-23"))
    assert "pricing failure, not a quiet board" in out
    assert "off-season week" not in out


def test_card_writer_puts_likely_winners_before_backed_bets():
    """Regression on ordering: the lead is likely winners, not the staking view."""
    import inspect
    src = inspect.getsource(S.write_card)
    assert src.index("_likely_winners_section") < src.index("_backed_bets_section")

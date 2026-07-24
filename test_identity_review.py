#!/usr/bin/env python3
"""Tests for the human identity-review workflow."""
from __future__ import annotations

import csv
import json

import pandas as pd
import pytest

from club_soccer import club_identity as CI
from club_soccer import identity_review as IR


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=IR.FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({f: r.get(f, "") for f in IR.FIELDS})


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(IR, "REVIEW_CSV", tmp_path / "identity_review.csv")
    monkeypatch.setattr(IR, "VERDICTS", tmp_path / "identity_verdicts.json")
    monkeypatch.setattr(IR, "DATA", tmp_path)
    return tmp_path


# ── reading verdicts ──────────────────────────────────────────────────────

def test_blank_verdicts_are_ignored(sandbox, monkeypatch):
    monkeypatch.setattr(CI, "FIXTURES", sandbox / "fx.csv")
    pd.DataFrame({"home": ["Twente"], "away": ["Ajax"]}).to_csv(CI.FIXTURES, index=False)
    _write_csv(IR.REVIEW_CSV, [{"europe_only_name": "FC Twente",
                                "suggested_match": "Twente", "VERDICT": ""}])
    merges, rejections, problems = IR.read_review()
    assert merges == {} and rejections == [] and problems == []


def test_y_merges_into_the_suggestion(sandbox, monkeypatch):
    monkeypatch.setattr(CI, "FIXTURES", sandbox / "fx.csv")
    pd.DataFrame({"home": ["Twente"], "away": ["Ajax"]}).to_csv(CI.FIXTURES, index=False)
    _write_csv(IR.REVIEW_CSV, [{"europe_only_name": "FC Twente",
                                "suggested_match": "Twente", "VERDICT": "y"}])
    merges, _r, problems = IR.read_review()
    assert merges == {"FC Twente": "Twente"}
    assert problems == []


def test_n_records_a_rejection(sandbox, monkeypatch):
    monkeypatch.setattr(CI, "FIXTURES", sandbox / "fx.csv")
    pd.DataFrame({"home": ["AEK"], "away": ["PAOK"]}).to_csv(CI.FIXTURES, index=False)
    _write_csv(IR.REVIEW_CSV, [{"europe_only_name": "AEK Larnaca",
                                "suggested_match": "AEK", "VERDICT": "n"}])
    merges, rejections, _p = IR.read_review()
    assert merges == {}
    assert rejections == ["AEK Larnaca"]


def test_freeform_name_overrides_the_suggestion(sandbox, monkeypatch):
    monkeypatch.setattr(CI, "FIXTURES", sandbox / "fx.csv")
    pd.DataFrame({"home": ["KuPS"], "away": ["HJK"]}).to_csv(CI.FIXTURES, index=False)
    _write_csv(IR.REVIEW_CSV, [{"europe_only_name": "Kuopion Palloseura",
                                "suggested_match": "", "VERDICT": "KuPS"}])
    merges, _r, problems = IR.read_review()
    assert merges == {"Kuopion Palloseura": "KuPS"}
    assert problems == []


# ── refusing to guess ─────────────────────────────────────────────────────

def test_unknown_target_is_a_problem_not_a_silent_skip(sandbox, monkeypatch):
    """A typo must stop the run, not quietly do nothing."""
    monkeypatch.setattr(CI, "FIXTURES", sandbox / "fx.csv")
    pd.DataFrame({"home": ["KuPS"], "away": ["HJK"]}).to_csv(CI.FIXTURES, index=False)
    _write_csv(IR.REVIEW_CSV, [{"europe_only_name": "Kuopion Palloseura",
                                "suggested_match": "", "VERDICT": "KuPSS"}])
    _m, _r, problems = IR.read_review()
    assert len(problems) == 1
    assert "not a club in fixtures.csv" in problems[0]


def test_y_without_a_suggestion_is_a_problem(sandbox, monkeypatch):
    monkeypatch.setattr(CI, "FIXTURES", sandbox / "fx.csv")
    pd.DataFrame({"home": ["A"], "away": ["B"]}).to_csv(CI.FIXTURES, index=False)
    _write_csv(IR.REVIEW_CSV, [{"europe_only_name": "Mystery FC",
                                "suggested_match": "", "VERDICT": "y"}])
    _m, _r, problems = IR.read_review()
    assert problems and "no suggested_match" in problems[0]


def test_self_merge_is_a_problem(sandbox, monkeypatch):
    monkeypatch.setattr(CI, "FIXTURES", sandbox / "fx.csv")
    pd.DataFrame({"home": ["Twente"], "away": ["Ajax"]}).to_csv(CI.FIXTURES, index=False)
    _write_csv(IR.REVIEW_CSV, [{"europe_only_name": "Twente",
                                "suggested_match": "", "VERDICT": "Twente"}])
    _m, _r, problems = IR.read_review()
    assert problems and "itself" in problems[0]


def test_apply_writes_nothing_when_there_are_problems(sandbox, monkeypatch, capsys):
    monkeypatch.setattr(CI, "FIXTURES", sandbox / "fx.csv")
    pd.DataFrame({"date": ["2025-01-01"], "competition": ["Eredivisie"],
                  "home": ["Twente"], "away": ["Ajax"]}).to_csv(CI.FIXTURES, index=False)
    _write_csv(IR.REVIEW_CSV, [{"europe_only_name": "X", "suggested_match": "",
                                "VERDICT": "NotAClub"}])
    IR.apply()
    assert "nothing applied" in capsys.readouterr().out
    assert not IR.VERDICTS.exists()


# ── persistence ───────────────────────────────────────────────────────────

def test_decided_clubs_are_not_re_listed(sandbox, monkeypatch):
    IR._save_verdicts({"FC Twente": {"decision": "merge", "target": "Twente"}})
    assert "FC Twente" in IR._load_verdicts()


def test_unsourced_hint_flags_a_club_from_an_unavailable_league():
    assert IR._looks_unsourced("AC Sparta Praha")
    assert IR._looks_unsourced("GNK Dinamo Zagreb")
    assert not IR._looks_unsourced("FC Twente")


# ── the live worklist ─────────────────────────────────────────────────────

def test_export_produces_a_usable_worklist():
    rows = IR.build_rows()
    assert rows, "there should be something to review"
    for r in rows:
        assert r["VERDICT"] == "", "verdict column must start empty"
        assert r["claude_read"], "every row needs a read to react to"
    # Most-priced clubs first: a split rating costs most where we price most.
    counts = [int(r["n_matches"]) for r in rows]
    assert counts == sorted(counts, reverse=True)


def test_high_confidence_suggestions_are_marked_likely_yes():
    rows = {r["europe_only_name"]: r for r in IR.build_rows()}
    twente = rows.get("FC Twente")
    if twente:
        assert twente["suggested_match"] == "Twente"
        assert twente["claude_read"].startswith("LIKELY YES")


def test_same_city_rivals_are_not_marked_likely_yes():
    """Slavia/Sparta Praha and CSKA/Levski Sofia are rivals, not spellings."""
    rows = {r["europe_only_name"]: r for r in IR.build_rows()}
    for name in ("SK Slavia Praha", "CSKA Sofia", "Omonia Nikosia"):
        r = rows.get(name)
        if r and r["suggested_match"]:
            assert not r["claude_read"].startswith("LIKELY YES"), \
                f"{name} -> {r['suggested_match']} must not be auto-endorsed"

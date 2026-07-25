#!/usr/bin/env python3
"""Regression: live-quote lifetime is fail-closed on provenance.

Only an allowlisted provider-timestamp source (provider_last_update) earns the
6-hour window. Missing, misspelled, unknown, or fetch_time_only provenance must
expire after 2 minutes — otherwise a quote with unknown staleness could be
staked on a provider timestamp we cannot trust.

Run: python3 -m pytest test_quote_provenance.py -q
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pandas as pd
import pytest

from club_soccer import edge

NOW = datetime(2026, 7, 19, 12, 0, 0, tzinfo=timezone.utc)
KO = (NOW + timedelta(hours=6)).isoformat()          # future kickoff, live-valid


def _row(source, age_minutes):
    return pd.DataFrame([{
        "home": "A", "away": "B",
        "date": "2026-07-19", "match_date": "2026-07-19", "kickoff_utc": KO,
        "market": "1x2", "selection": "home", "odds": 2.0,
        "quoted_at_utc": (NOW - timedelta(minutes=age_minutes)).isoformat(),
        "quote_time_source": source,
    }])


def _kept(source, age_minutes):
    out, _ = edge.validate_quotes(_row(source, age_minutes), source="live", now=NOW)
    return len(out)


@pytest.mark.parametrize("source", ["", "fetch_time_only", "weird_label",
                                    "provider_last_updat", "PROVIDER_LAST_UPDATE"])
def test_untrusted_source_expires_after_two_minutes(source):
    # 3 minutes old, no trusted provenance -> dropped (the review's repro).
    assert _kept(source, 3) == 0
    # 1 minute old -> still inside the fetch-time window.
    assert _kept(source, 1) == 1


def test_trusted_provider_source_keeps_six_hours():
    assert _kept("provider_last_update", 3) == 1
    assert _kept("provider_last_update", 300) == 1     # 5h < 6h -> kept
    assert _kept("provider_last_update", 420) == 0     # 7h > 6h -> dropped


def test_missing_source_column_defaults_to_two_minutes():
    # No quote_time_source column at all -> every live row is fetch-time (2 min).
    df3 = _row("provider_last_update", 3).drop(columns=["quote_time_source"])
    out3, _ = edge.validate_quotes(df3, source="live", now=NOW)
    assert len(out3) == 0                                # 3 min > 2 min -> dropped
    df1 = _row("provider_last_update", 1).drop(columns=["quote_time_source"])
    out1, _ = edge.validate_quotes(df1, source="live", now=NOW)
    assert len(out1) == 1                                # 1 min < 2 min -> kept

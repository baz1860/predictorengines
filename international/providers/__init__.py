"""Fixture providers.

Each provider module exposes the same two-layer shape:

  * **pure parsers** — provider payload in, canonical rows out. No network, no
    clock, no file access, so they can be tested against recorded payloads.
  * **fetch functions** — network calls that write a raw observation FIRST and
    then hand the payload to the parsers.

That split is what makes the evidence phase auditable: every canonical row can be
regenerated from a stored raw payload without re-querying, so a coverage claim can
be re-checked after the window has closed.

No provider is "primary". `scripts/international/fetch_fixtures.py` takes an
explicit `--provider` and records which one produced each row, because the plan's
one repeated mistake has been pre-committing to a provider before measuring it.
"""
from __future__ import annotations

__all__ = ["bsd"]

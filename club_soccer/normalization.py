"""Shared Unicode and club-name normalization primitives."""
from __future__ import annotations

import re
import unicodedata


def fold_accents(value) -> str:
    """Case-fold text and remove combining marks without changing punctuation."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    return "".join(
        char for char in text if not unicodedata.combining(char)
    ).casefold()


def normalise_spaces(value) -> str:
    return re.sub(r"\s+", " ", fold_accents(value)).strip()


def normalise_club_text(value) -> str:
    """Normalized lookup key shared by identity and registry resolution."""
    text = fold_accents(value).replace("&", " and ")
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

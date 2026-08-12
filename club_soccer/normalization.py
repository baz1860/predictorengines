"""Shared Unicode and club-name normalization primitives."""
from __future__ import annotations

import re
import unicodedata

# Latin letters that NFKD does NOT decompose into base + combining mark.
#
# `fold_accents` strips combining marks, which handles é -> e, ä -> a, ñ -> n.
# It does nothing for these, because they are atomic codepoints rather than a
# base letter with an accent hung off it. Downstream, `normalise_club_text`
# applies `[^a-z0-9 ]` and DELETES them, so "Brøndby" became "br ndby" and
# "Ħamrun Spartans" became "amrun spartans" — keys that can never match the
# same club spelled without the diacritic. That is the mechanism behind the
# Danish Superliga carrying two identities each for Brøndby, Sønderjyske and
# Nordsjælland (a 14-team table in a 12-team league).
#
# Transliterating before the strip keeps the letter's information instead of
# discarding it. Multi-character expansions (æ -> ae, ß -> ss) follow the
# conventional Latin-ASCII romanisation used by the fd.co.uk and openfootball
# sources we join against.
_TRANSLITERATIONS = {
    "ø": "o", "æ": "ae", "å": "a", "ð": "d", "þ": "th", "ß": "ss",
    "đ": "d", "ħ": "h", "ı": "i", "ł": "l", "ŀ": "l", "œ": "oe",
    "ŋ": "n", "ə": "e", "ɔ": "o", "ŧ": "t", "ĸ": "k",
}

_TRANSLITERATION_TABLE = str.maketrans(_TRANSLITERATIONS)


def transliterate(value) -> str:
    """Map non-decomposing Latin letters onto their ASCII equivalents.

    Applied after case-folding, so only the lowercase forms need entries.
    """
    return str(value or "").translate(_TRANSLITERATION_TABLE)


def fold_accents(value) -> str:
    """Case-fold text and remove combining marks without changing punctuation.

    Non-decomposing letters (ø, æ, ß, ł, ...) are transliterated rather than
    left to be deleted by a later ASCII filter — see _TRANSLITERATIONS.
    """
    text = unicodedata.normalize("NFKD", str(value or ""))
    stripped = "".join(
        char for char in text if not unicodedata.combining(char)
    ).casefold()
    return transliterate(stripped)


def normalise_spaces(value) -> str:
    return re.sub(r"\s+", " ", fold_accents(value)).strip()


def normalise_club_text(value) -> str:
    """Normalized lookup key shared by identity and registry resolution."""
    text = fold_accents(value).replace("&", " and ")
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

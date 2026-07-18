"""Feature eligibility registry (plan section 1).

Every model feature is registered with its source, event timestamp, cutoff
availability, lookback, missingness policy, leakage risk, and the scopes it is
allowed in (pure model, market blend, or both). Features that the plan calls
for but that the canonical schema cannot yet prove available at prediction
time are registered as ``blocked`` with the reason, so nobody quietly adds
them without provenance.

``verify_registry()`` cross-checks the registry against the live feature
schema in :mod:`horse_racing.features` and is executed by the offline test
suite: an unregistered feature is a build error, not a review comment.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from .features import FAMILIES, MISSING_INDICATOR, feature_names

PURE = "pure"
BLEND = "blend"
BOTH = "both"
BLOCKED = "blocked"


@dataclass(frozen=True)
class FeatureSpec:
    name: str                 # raw column name (pre race-relative transform)
    family: str
    source: str               # canonical table(s) the feature derives from
    event_timestamp: str      # which timestamp orders the underlying events
    available_at_cutoff: str  # why the value is known before prediction cutoff
    lookback: str             # decay horizon / window
    missingness: str          # policy when the raw value is absent
    leakage_risk: str         # low / medium / high + dominant failure mode
    allowed_in: str           # pure / blend / both / blocked


_DECL = "runners.source_updated_at <= cutoff (as-of snapshot)"
_RESULT = "results availability timestamp <= cutoff (chronological replay)"
_RACE = "races.source_updated_at <= cutoff (validated at build time)"
_FILL = "race-mean fill + explicit missingness indicator"
_FILL_PLAIN = "race-mean fill (z-score sends it to 0)"


def _spec(name, family, source, event_timestamp, available, lookback,
          missingness, risk, allowed) -> FeatureSpec:
    return FeatureSpec(name, family, source, event_timestamp, available,
                       lookback, missingness, risk, allowed)


REGISTRY: dict[str, FeatureSpec] = {spec.name: spec for spec in [
    # ---- core (V1 parity) ----------------------------------------------
    _spec("official_rating", "core", "runners", "source_updated_at", _DECL,
          "declaration snapshot", _FILL, "low", BOTH),
    _spec("weight", "core", "runners", "source_updated_at", _DECL,
          "declaration snapshot", _FILL, "low", BOTH),
    _spec("draw", "core", "runners", "source_updated_at", _DECL,
          "declaration snapshot", _FILL, "low", BOTH),
    _spec("age", "core", "runners", "source_updated_at", _DECL,
          "declaration snapshot", _FILL, "low", BOTH),
    _spec("horse_form", "core", "results+runners", "result availability", _RESULT,
          "90d half-life decayed finish performance", _FILL_PLAIN, "low", BOTH),
    _spec("horse_win", "core", "results", "result availability", _RESULT,
          "90d half-life decayed win rate, prior toward 1/field", _FILL_PLAIN,
          "low", BOTH),
    _spec("horse_starts", "core", "results", "result availability", _RESULT,
          "lifetime (log1p)", _FILL_PLAIN, "low", BOTH),
    _spec("days_since", "core", "results", "result availability", _RESULT,
          "time since last applied run", _FILL, "low", BOTH),
    _spec("trainer_win", "core", "results", "result availability", _RESULT,
          "365d half-life decayed win rate", _FILL_PLAIN, "low", BOTH),
    _spec("jockey_win", "core", "results", "result availability", _RESULT,
          "365d half-life decayed win rate", _FILL_PLAIN, "low", BOTH),
    _spec("surface_fit", "core", "results+races", "result availability", _RESULT,
          "365d half-life decayed perf on same surface", _FILL_PLAIN, "low", BOTH),
    _spec("distance_fit", "core", "results+races", "result availability", _RESULT,
          "365d half-life decayed perf in same 400m distance band", _FILL_PLAIN,
          "low", BOTH),
    _spec("course_fit", "core", "results+races", "result availability", _RESULT,
          "365d half-life decayed perf at same course", _FILL_PLAIN, "low", BOTH),
    # ---- form_multi ------------------------------------------------------
    _spec("form_short", "form_multi", "results", "result availability", _RESULT,
          "30d half-life decayed finish performance", _FILL_PLAIN, "low", BOTH),
    _spec("form_long", "form_multi", "results", "result availability", _RESULT,
          "365d half-life decayed finish performance", _FILL_PLAIN, "low", BOTH),
    _spec("form_n", "form_multi", "results", "result availability", _RESULT,
          "90d half-life decayed effective run count (log1p)", _FILL_PLAIN,
          "low", BOTH),
    _spec("last_perf", "form_multi", "results", "result availability", _RESULT,
          "most recent finished run", _FILL_PLAIN, "low", BOTH),
    _spec("top3_rate", "form_multi", "results", "result availability", _RESULT,
          "90d half-life decayed top-3 rate, prior toward 3/field", _FILL_PLAIN,
          "low", BOTH),
    _spec("tophalf_rate", "form_multi", "results", "result availability", _RESULT,
          "90d half-life decayed top-half rate, prior 0.5", _FILL_PLAIN, "low", BOTH),
    _spec("dnf_rate", "form_multi", "results", "result availability", _RESULT,
          "365d half-life decayed non-completion rate, prior 0.05", _FILL_PLAIN,
          "low", BOTH),
    _spec("trainer_form_short", "form_multi", "results", "result availability",
          _RESULT, "30d half-life decayed trainer performance", _FILL_PLAIN,
          "low", BOTH),
    _spec("jockey_form_short", "form_multi", "results", "result availability",
          _RESULT, "30d half-life decayed jockey performance", _FILL_PLAIN,
          "low", BOTH),
    _spec("pair_win", "form_multi", "results", "result availability", _RESULT,
          "365d half-life decayed trainer-jockey pair win rate", _FILL_PLAIN,
          "low", BOTH),
    # ---- class_struct ----------------------------------------------------
    _spec("class_move", "class_struct", "races+results", "result availability",
          _RACE + " and " + _RESULT,
          "current race_class minus 365d decayed mean class contested",
          _FILL_PLAIN, "low", BOTH),
    _spec("or_change", "class_struct", "runners+results", "result availability",
          _RESULT, "declaration OR minus OR at last applied run", _FILL,
          "medium: prior OR must come from replayed state, never from a "
          "retrospective profile", BOTH),
    _spec("weight_change", "class_struct", "runners+results",
          "result availability", _RESULT,
          "declared weight minus weight at last applied run", _FILL,
          "medium: same replay constraint as or_change", BOTH),
    _spec("or_x_handicap", "class_struct", "runners+races", "source_updated_at",
          _DECL + " and " + _RACE, "declaration snapshot x race handicap flag",
          _FILL_PLAIN, "low", BOTH),
    # ---- suitability -----------------------------------------------------
    _spec("going_fit", "suitability", "results+races", "result availability",
          _RESULT, "365d half-life decayed perf on same going group",
          _FILL_PLAIN,
          "medium: going may be updated close to off; the build validates "
          "races.source_updated_at <= cutoff", BOTH),
    _spec("course_dist_fit", "suitability", "results+races",
          "result availability", _RESULT,
          "365d half-life decayed perf at same course x distance band",
          _FILL_PLAIN, "low", BOTH),
    _spec("dist_delta_abs", "suitability", "races+results",
          "result availability", _RESULT,
          "abs(log distance today minus 365d decayed mean log distance raced)",
          _FILL_PLAIN, "low", BOTH),
    _spec("dist_delta_signed", "suitability", "races+results",
          "result availability", _RESULT,
          "signed log-distance move (step up vs drop back)", _FILL_PLAIN,
          "low", BOTH),
    # ---- draw_hier -------------------------------------------------------
    _spec("draw_effect", "draw_hier", "runners+results", "result availability",
          _RESULT,
          "730d half-life course x surface x distance-band draw slope, "
          "shrunk course->surface->zero (k=25/50)", _FILL_PLAIN, "low", BOTH),
    _spec("draw_norm", "draw_hier", "runners", "source_updated_at", _DECL,
          "declaration snapshot, normalized to [-1,1] within field",
          _FILL_PLAIN, "low", BOTH),
    # ---- weight_rating ---------------------------------------------------
    _spec("weight_x_dist", "weight_rating", "runners+races", "source_updated_at",
          _DECL + " and " + _RACE, "declaration snapshot x log race distance",
          _FILL_PLAIN, "low", BOTH),
    _spec("age_x_dist", "weight_rating", "runners+races", "source_updated_at",
          _DECL + " and " + _RACE, "declaration snapshot x log race distance",
          _FILL_PLAIN, "low", BOTH),
]}

# Features the plan names that CANNOT currently be proven available at
# prediction time or are absent from the canonical schema. Blocked means:
# do not add to any model until an adapter supplies them with point-in-time
# provenance (plan: "Defer For Now").
BLOCKED_FEATURES: dict[str, str] = {
    "beaten_distance": "results.csv carries finish_position only; rpscrape raw "
                       "has ovr_btn but it is not normalized into the canonical "
                       "schema with availability timestamps yet",
    "headgear": "not in canonical runners schema; needs declaration-time source",
    "headgear_change": "depends on headgear",
    "sex_restriction": "not in canonical races schema",
    "rating_band": "not in canonical races schema",
    "age_band": "not in canonical races schema",
    "race_type_detail": "canonical schema has code/handicap_flag only",
    "going_change_intraday": "no timestamped going revisions in canonical data",
    "prior_opposition_strength": "requires stable ability estimates first "
                                 "(dynamic-ability challenger, plan section 5)",
    "market_odds": "market data is blend-only by design (plan section 7); "
                   "never a pure-model feature",
    "starting_price": "post-race data; benchmark only, never a feature",
}


def verify_registry() -> None:
    """Raise if the live feature schema and the registry disagree."""
    raw_names = {col for family, cols in FAMILIES.items() for col in cols}
    registered = set(REGISTRY)
    unregistered = sorted(raw_names - registered)
    if unregistered:
        raise AssertionError(f"features missing from registry: {unregistered}")
    stale = sorted(registered - raw_names)
    if stale:
        raise AssertionError(f"registry entries without live features: {stale}")
    for name, spec in REGISTRY.items():
        family_cols = FAMILIES.get(spec.family, [])
        if name not in family_cols:
            raise AssertionError(f"{name} registered under wrong family {spec.family}")
        if spec.allowed_in not in {PURE, BLEND, BOTH}:
            raise AssertionError(f"{name} has invalid scope {spec.allowed_in}")
    blocked_overlap = sorted(set(BLOCKED_FEATURES) & raw_names)
    if blocked_overlap:
        raise AssertionError(f"blocked features present in live schema: {blocked_overlap}")
    # missingness indicators must exist exactly for features documenting them
    indicator = {name for name, spec in REGISTRY.items()
                 if "missingness indicator" in spec.missingness}
    if indicator != set(MISSING_INDICATOR):
        raise AssertionError("missingness-indicator registry entries do not match "
                             f"features.MISSING_INDICATOR: {sorted(indicator)} vs "
                             f"{sorted(MISSING_INDICATOR)}")
    # sanity: derived model schema is consistent
    feature_names()


def registry_report() -> dict:
    return {"features": {name: asdict(spec) for name, spec in sorted(REGISTRY.items())},
            "blocked": dict(sorted(BLOCKED_FEATURES.items()))}

#!/usr/bin/env python3
"""E2 — posterior-variance-aware staking, measured against closing prices.

Replays the decision ledger twice: once with the production flat quarter-Kelly,
once with the stake shrunk by how well-measured the two clubs are, and scores
both by expected log growth against the devigged closing price.

See `plans/club_soccer_uncertainty_experiment.md` §7.

Why expected log growth and not ROI
-----------------------------------
§7.1 of the plan: realised ROI on ~2,500 settled bets carries a standard error
of roughly 2 percentage points, so a staking refinement worth ~1pp is invisible
in it. Expected log growth replaces the realised 0/1 outcome with the devigged
closing probability — the market's best estimate of the truth — which removes
the binomial noise entirely and measures the quantity Kelly is actually trying
to maximise:

    g(f) = p_close * log(1 + f*(o-1)) + (1 - p_close) * log(1 - f)

Realised ROI is still reported, and is still NOT the promotion signal.

Why the comparison is run at matched total stake
------------------------------------------------
The shrink multiplier is clamped at 1.0, so the rule can only ever reduce a
stake. The decision-time book currently runs a negative flat ROI, so simply
staking less improves log growth for a reason that has nothing to do with
uncertainty. Scaling the candidate back up to the incumbent's total stake
removes that confound and leaves the question the experiment is actually
asking: given the same money at risk, is it better placed?

Point-in-time
-------------
Posterior SDs are recomputed by refitting at each decision's own
`train_cutoff`, never from today's model. A stake that could only have been
sized with hindsight is not evidence about a staking rule.

Usage
-----
    python3 -m club_soccer.posterior_kelly_ab
    python3 -m club_soccer.posterior_kelly_ab --write-evidence
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import edge as E
from . import model as M
from . import walkforward_cache as WFC

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DECISIONS = DATA / "decision_ledger.csv"
SETTLEMENTS = DATA / "settlement_ledger.csv"
EVIDENCE = DATA / "posterior_kelly_evidence.json"

JOIN_KEY = ["provider_fixture_id", "market", "side"]
# Day-level blocks: bets placed on the same day share a model fit, and often
# the same fixtures, so treating 1,244 decisions as 1,244 independent samples
# would overstate the evidence by an order of magnitude. The engine's own
# evidence gate takes the same view (`MIN_INDEPENDENT_BLOCKS = 8`).
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260816


def load_decisions() -> pd.DataFrame:
    """Eligible decisions that carry a devigged closing price."""
    dec = pd.read_csv(DECISIONS, low_memory=False)
    setl = pd.read_csv(SETTLEMENTS, low_memory=False)
    for col in JOIN_KEY:
        dec[col] = dec[col].astype(str)
        setl[col] = setl[col].astype(str)
    merged = dec.merge(
        setl[JOIN_KEY + ["won", "pinnacle_close_devig"]], on=JOIN_KEY,
        how="inner",
    )
    merged = merged[merged["strategy_eligible"] == 1]
    merged = merged.dropna(subset=["pinnacle_close_devig", "odds_executed",
                                   "p_model", "p_book_devig", "train_cutoff"])
    merged = merged[(merged["odds_executed"] > 1.0)
                    & merged["pinnacle_close_devig"].between(0.0, 1.0)]
    merged["decision_day"] = pd.to_datetime(
        merged["decision_ts"], format="mixed", utc=True
    ).dt.date.astype(str)
    return merged.reset_index(drop=True)


def posterior_sds(decisions: pd.DataFrame,
                  verbose: bool = True) -> tuple[dict[tuple, float],
                                                 dict[str, float]]:
    """Per-(cutoff, home, away) posterior SD, refitting at each cutoff.

    One fit per distinct `train_cutoff`, not per decision — the ledger records
    ~10 cutoffs across a few hundred fixtures.

    Also returns each cutoff's own reference SD. Using today's reference
    against a historical SD would compare a match to a scale derived from data
    that did not exist when the bet was placed, which is the same hindsight
    the refit exists to avoid.
    """
    frame = M.played(M.load_fixtures()).sort_values("date").reset_index(drop=True)
    out: dict[tuple, float] = {}
    refs: dict[str, float] = {}
    cutoffs = sorted(decisions["train_cutoff"].dropna().unique())
    for i, cutoff in enumerate(cutoffs, 1):
        train = frame[frame["date"] < str(cutoff)]
        if train.empty:
            continue
        try:
            params = M.fit(train, coef_as_of=str(cutoff)[:10], hierarchical=True)
        except Exception as exc:                      # pragma: no cover
            if verbose:
                print(f"  [{i}/{len(cutoffs)}] {cutoff}  fit failed: {exc}")
            continue
        ref = (params.get("pooled") or {}).get("eta_sd_reference")
        if ref is not None:
            refs[str(cutoff)] = float(ref)
        rows = decisions[decisions["train_cutoff"] == cutoff]
        hit = 0
        for home, away in set(zip(rows["club_home"], rows["club_away"])):
            sd = M.posterior_eta_sd(params, str(home), str(away))
            if sd is not None:
                out[(cutoff, str(home), str(away))] = sd
                hit += 1
        if verbose:
            shown = f"{ref:.4f}" if ref is not None else "n/a"
            print(f"  [{i}/{len(cutoffs)}] {cutoff}  {hit} fixtures priced "
                  f"(ref sd {shown})", flush=True)
    return out, refs


def _log_growth(fraction: float, odds: float, p_true: float) -> float:
    """Expected log bankroll growth from staking `fraction` at `odds`."""
    if fraction <= 0.0:
        return 0.0
    fraction = min(fraction, 0.999999)
    win = 1.0 + fraction * (odds - 1.0)
    lose = 1.0 - fraction
    if win <= 0.0 or lose <= 0.0:
        return float("-inf")
    return p_true * math.log(win) + (1.0 - p_true) * math.log(lose)


def build_arms(decisions: pd.DataFrame, sds: dict[tuple, float],
               references: dict[str, float]) -> pd.DataFrame:
    """Stake each decision both ways and score both against the close."""
    rows = []
    for r in decisions.itertuples(index=False):
        odds = float(r.odds_executed)
        p_model = float(r.p_model)
        p_book = float(r.p_book_devig)
        p_close = float(r.pinnacle_close_devig)
        conf = float(getattr(r, "lineup_confidence", 1.0) or 1.0)
        sd = sds.get((r.train_cutoff, str(r.club_home), str(r.club_away)))
        reference = references.get(str(r.train_cutoff))

        incumbent = E.posterior_kelly_stake(
            p_model, p_book, odds, None, None, active=False) * conf
        candidate = E.posterior_kelly_stake(
            p_model, p_book, odds, sd, reference, active=True) * conf
        rows.append({
            "day": r.decision_day, "market": r.market, "side": r.side,
            "competition": r.competition,
            "fixture": r.provider_fixture_id,
            "odds": odds, "p_model": p_model, "p_book": p_book,
            "p_close": p_close, "edge": p_model - p_book,
            "won": (None if pd.isna(r.won) else float(r.won)),
            "sd": sd, "has_sd": sd is not None and reference is not None,
            "keep": E.posterior_shrink(sd, reference),
            "stake_incumbent": incumbent,
            "stake_candidate": candidate,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Matched total stake — see the module docstring.
    total_i = out["stake_incumbent"].sum()
    total_c = out["stake_candidate"].sum()
    scale = (total_i / total_c) if total_c > 0 else 1.0
    out["stake_candidate_matched"] = out["stake_candidate"] * scale
    out.attrs["stake_scale"] = scale
    for arm in ("incumbent", "candidate", "candidate_matched"):
        out[f"growth_{arm}"] = [
            _log_growth(f, o, p)
            for f, o, p in zip(out[f"stake_{arm}"], out["odds"], out["p_close"])
        ]
    return out


def _block_bootstrap(arms: pd.DataFrame, column: str,
                     baseline: str = "growth_incumbent") -> dict:
    """Day-block bootstrap CI on the total log-growth difference.

    Resamples DAYS, not bets. With ~10 days of accumulated decision-time
    evidence this interval is wide, and that width is the finding rather than
    an inconvenience — see the plan's §7.1 power note.
    """
    days = sorted(arms["day"].unique())
    by_day = {d: g for d, g in arms.groupby("day")}
    observed = float(arms[column].sum() - arms[baseline].sum())
    if len(days) < 2:
        return {"observed": observed, "ci_low": None, "ci_high": None,
                "blocks": len(days), "note": "too few blocks to bootstrap"}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_DRAWS)
    for i in range(BOOTSTRAP_DRAWS):
        pick = rng.choice(len(days), size=len(days), replace=True)
        total = 0.0
        for j in pick:
            g = by_day[days[j]]
            total += float(g[column].sum() - g[baseline].sum())
        draws[i] = total
    return {
        "observed": observed,
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "blocks": len(days),
        "excludes_zero": bool(np.percentile(draws, 2.5) > 0.0
                              or np.percentile(draws, 97.5) < 0.0),
    }


def _realised_roi(arms: pd.DataFrame, stake_col: str) -> float | None:
    """Confirmatory only. Never the promotion signal — see §7.1."""
    settled = arms.dropna(subset=["won"])
    if settled.empty:
        return None
    staked = settled[stake_col].sum()
    if staked <= 0:
        return None
    ret = (settled[stake_col] * (settled["won"] * (settled["odds"] - 1.0)
                                 - (1.0 - settled["won"]))).sum()
    return float(ret / staked)


def run(write: bool = False, verbose: bool = True) -> dict:
    decisions = load_decisions()
    if decisions.empty:
        raise RuntimeError("no eligible decisions with a closing price")
    if verbose:
        print(f"[E2] {len(decisions)} eligible decisions, "
              f"{decisions['provider_fixture_id'].nunique()} fixtures, "
              f"{decisions['decision_day'].nunique()} days", flush=True)

    sds, references = posterior_sds(decisions, verbose=verbose)
    arms = build_arms(decisions, sds, references)
    covered = int(arms["has_sd"].sum())

    growth = {arm: float(arms[f"growth_{arm}"].sum())
              for arm in ("incumbent", "candidate", "candidate_matched")}
    matched = _block_bootstrap(arms, "growth_candidate_matched")
    raw = _block_bootstrap(arms, "growth_candidate")

    payload = {
        "experiment": "posterior_kelly",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": ("decision-ledger replay, expected log growth vs devigged "
                   "closing price, point-in-time posterior SDs refit at each "
                   "train_cutoff"),
        "n_decisions": int(len(arms)),
        "n_fixtures": int(arms["fixture"].nunique()),
        "n_days": int(arms["day"].nunique()),
        "n_with_posterior_sd": covered,
        "posterior_coverage": round(covered / max(1, len(arms)), 4),
        "eta_sd_reference_by_cutoff": references,
        "stake_match_scale": float(arms.attrs.get("stake_scale", 1.0)),
        "total_stake_incumbent": float(arms["stake_incumbent"].sum()),
        "total_stake_candidate_raw": float(arms["stake_candidate"].sum()),
        "log_growth": growth,
        "primary_matched_stake": {
            "delta": matched["observed"],
            "ci95": [matched.get("ci_low"), matched.get("ci_high")],
            "blocks": matched["blocks"],
            "excludes_zero": matched.get("excludes_zero"),
            "note": ("PRIMARY metric: same money at risk, placed differently. "
                     "CI is a day-block bootstrap."),
        },
        "secondary_raw_stake": {
            "delta": raw["observed"],
            "ci95": [raw.get("ci_low"), raw.get("ci_high")],
            "note": ("CONFOUNDED — the rule only reduces stakes and the book "
                     "runs a negative flat ROI, so this rewards betting less "
                     "regardless of whether uncertainty was used well."),
        },
        "realised_roi": {
            "incumbent": _realised_roi(arms, "stake_incumbent"),
            "candidate_matched": _realised_roi(arms, "stake_candidate_matched"),
            "note": ("CONFIRMATORY ONLY, never the promotion signal — §7.1: "
                     "ROI on this sample has a standard error far larger than "
                     "any effect a staking refinement could produce."),
        },
        "code_hash": WFC.code_fingerprint(),
        "command": "python3 -m club_soccer.posterior_kelly_ab --write-evidence",
    }
    payload["verdict"] = _verdict(payload)
    if write:
        EVIDENCE.write_text(json.dumps(payload, indent=2) + "\n")
        if verbose:
            print(f"Evidence -> {EVIDENCE.name}")
    return payload


def _verdict(payload: dict) -> dict:
    """Decide, or say plainly that the evidence cannot decide."""
    from .evidence_gate import MIN_INDEPENDENT_BLOCKS

    primary = payload["primary_matched_stake"]
    blocks = primary.get("blocks") or 0
    # The engine's own evidence gate will not stake a strategy on fewer than
    # MIN_INDEPENDENT_BLOCKS independent blocks. A refinement to how that
    # strategy sizes its stakes cannot reasonably be held to a weaker standard
    # than the strategy itself, so the threshold is read from there rather
    # than restated here, where the two could drift apart.
    if blocks < MIN_INDEPENDENT_BLOCKS:
        return {"decision": "undecidable",
                "reason": f"only {blocks} independent day-blocks of "
                          "decision-time evidence (evidence_gate requires "
                          f"{MIN_INDEPENDENT_BLOCKS} before staking at all)"}
    if primary.get("excludes_zero") and primary["delta"] > 0:
        return {"decision": "promote",
                "reason": "log growth at matched stake improves and the "
                          "day-block CI excludes zero"}
    if primary.get("excludes_zero") and primary["delta"] < 0:
        return {"decision": "retire",
                "reason": "log growth at matched stake is worse and the "
                          "day-block CI excludes zero"}
    return {"decision": "undecidable",
            "reason": "day-block CI spans zero — the sample cannot "
                      "distinguish this rule from the flat fraction"}


def _report(p: dict) -> None:
    print("\n== E2 posterior-variance staking ==")
    print(f"   {p['n_decisions']} decisions, {p['n_fixtures']} fixtures, "
          f"{p['n_days']} days, posterior SD on "
          f"{p['posterior_coverage']*100:.0f}% of them")
    refs = list((p.get("eta_sd_reference_by_cutoff") or {}).values())
    ref_txt = f"{min(refs):.4f}-{max(refs):.4f}" if refs else "n/a"
    print(f"   point-in-time reference eta SD {ref_txt}   "
          f"candidate staked {p['total_stake_candidate_raw']/max(1e-9,p['total_stake_incumbent'])*100:.1f}% "
          f"of incumbent before matching")
    pm = p["primary_matched_stake"]
    lo, hi = pm["ci95"]
    print(f"\n  PRIMARY  log-growth delta at matched stake: {pm['delta']:+.5f}")
    if lo is not None:
        print(f"           95% day-block CI [{lo:+.5f}, {hi:+.5f}] "
              f"over {pm['blocks']} blocks")
    sr = p["secondary_raw_stake"]
    print(f"  (raw, unmatched: {sr['delta']:+.5f} — confounded, see note)")
    roi = p["realised_roi"]
    if roi["incumbent"] is not None:
        print(f"  (realised ROI confirmatory: {roi['incumbent']:+.4f} -> "
              f"{roi['candidate_matched']:+.4f})")
    v = p["verdict"]
    print(f"\n  VERDICT: {v['decision'].upper()}\n  {v['reason']}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="E2 posterior-Kelly A/B")
    ap.add_argument("--write-evidence", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    payload = run(write=args.write_evidence, verbose=not args.quiet)
    _report(payload)


if __name__ == "__main__":
    main()

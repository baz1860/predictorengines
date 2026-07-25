# Scope — decision-time backtest & confidence-strategy evidence

**Status:** Scoping, for review before build
**Goal:** produce the `decision_time_v2` evidence artifact the staking gate is
waiting for, built around the confidence-ranked "likely winners" strategy —
measuring hit-rate and yield, not just edge-threshold ROI.

---

## 1. The one constraint that shapes everything

**You cannot backtest a decision-time strategy on closing odds, and closing
odds are almost all we have historically.**

- `market_history.csv` — **16,778 matches, 2022–2026**, but these are *closing*
  prices (B365, Pinnacle close `psc`). The retired `backtest_market.py` used
  them, which is why its evidence could never open the gate: selecting and
  executing at the close is not a strategy you can run — you don't know the
  close until kick-off.
- `odds_history_club.csv` — point-in-time snapshots captured *before* kick-off,
  which is the right data. But it is **17 days old, 623 rows, ~2 snapshots per
  fixture**, and mostly unsettled upcoming games.

A decision-time quote — the price 60+ minutes before kick-off — **cannot be
reconstructed retroactively**. It had to be recorded at the time. So the honest
position is:

> The backtest **engine** can be built now. It cannot **pass** now, because the
> settled point-in-time evidence does not yet exist. It accumulates forward from
> the day snapshotting becomes disciplined, and the gate opens when it clears
> the preregistered bar — realistically **months of calendar time**, not a code
> change.

Anyone who tells you the gate can open sooner is proposing to select on the
close. The gate exists precisely to refuse that.

---

## 2. What the gate already demands

`evidence_gate.py` preregisters the criteria (before any passing artifact
exists, so the bar can't be moved to fit a result). The artifact must declare:

- `backtest_version: "decision_time_v2"`
- selection = latest quote at/*before* the decision time
- execution = that same decision-time quote
- CLV reference = de-vigged Pinnacle close
- decision lead between 60 minutes and 7 days
- thresholds reported at 2%, 4%, 6%

And to actually open (all must hold, per market):

1. artifact ≤ 14 days old, signed provenance
2. ≥ **1,000 settled bets** at the lowest threshold
3. flat ROI > 0 **and** quarter-Kelly ROI > 0 at every threshold
4. CLV mean > 0 **and** positive-CLV fraction ≥ 0.5
5. for 1X2: model log-loss ≤ market log-loss

Plus the preregistered tightenings (block-bootstrap 95% lower bounds, per-league
≥200-bet activation, full provenance hashes). These are already written into the
gate as TODOs — the build implements them, it does not get to weaken them.

---

## 3. Reconciling the gate with "regular winners"

Your goal and the gate's criteria are not in tension — they measure two
different things, and the backtest must show both:

| Question | Metric | Whose concern |
|---|---|---|
| Does the pick come in often? | **hit-rate by confidence bucket** | yours — regular winners |
| Does betting it make money? | ROI + CLV lower bounds | the gate — don't bleed |

The critical case is the **confident-but-`short`** pick: high hit-rate, negative
CLV. It wins often and loses money long-term. The backtest must surface exactly
this so a high strike-rate never gets mistaken for a profitable one. The
`value ✓` picks are where hit-rate and yield agree — those are what eventually
stakes.

So the artifact reports, per confidence bucket (50–55, 55–60, 60–65, 65+) **and**
per edge threshold:

- n bets, hit-rate, mean odds
- flat & quarter-Kelly ROI (+ block-bootstrap 95% lower bound)
- CLV mean and positive fraction
- realised yield = profit / staked

That gives you a strike-rate table to read, and the gate its ROI/CLV bounds —
from one run.

---

## 4. Phases

### Phase A — build the engine *(buildable now, ~2–3 focused sessions)*

1. **`decision_time_backtest.py`** — replays each settled fixture from the
   point-in-time snapshot at its decision time (latest snapshot ≥60 min before
   kick-off), prices it with the *same* walk-forward model the card uses, bets
   at that quote, settles on the result, and scores CLV against the de-vigged
   Pinnacle close from `market_history.csv`.
2. **Confidence-strategy metrics** (§3) as the headline output.
3. **The `decision_time_v2` artifact** with the exact schema + provenance the
   gate validates (`additionalProperties: false`, duplicate-key rejection,
   data/model/code hashes).
4. **Block-bootstrap lower bounds** for ROI/CLV/log-loss per the preregistered
   tightenings.
5. Wire into `update.sh` so it runs and refreshes the artifact daily.

At the end of Phase A the gate still reads CLOSED — correctly — because
`n_bets < 1000`. But the machinery is done and the strike-rate table is live for
you to read from day one.

### Phase B — accumulate the evidence *(calendar time, mostly automatic)*

1. Make the daily snapshot **disciplined**: capture at a consistent decision
   lead (e.g. a snapshot in the 60–180 min pre-kick-off window per fixture),
   store kick-off time so lead is computable. Today `snapshot_odds.py` runs
   daily but coverage is thin and lead isn't pinned.
2. **Settle** snapshots against results into an append-only backtest ledger.
3. Monitor accumulation: bets-toward-1000 counter surfaced in the run ledger /
   readiness, so you can see the gate approaching rather than guessing.

Rough arithmetic: full-evidence fixtures across all leagues run into the
hundreds per week in season, so 1,000 *settled* qualifying bets is plausibly a
**single season's worth** of disciplined capture — call it months, faster once
the August restart fills the card with domestic fixtures.

### Phase C — open the gate *(automatic when earned)*

No new work — the gate already evaluates the artifact every run. When the
accumulated ledger clears every criterion, `staking_allowed()` flips to open and
the card's Backed-bets section starts recommending real stakes. The likely-
winners lead you have now is unaffected; it just gains a staking layer underneath
it that has actually been proven.

---

## 5. Risks & honest caveats

| Risk | Note |
|---|---|
| Impatience with the timeline | The wait is the point. There is no correct shortcut that isn't "select on the close." |
| Snapshot coverage gaps | A fixture with no pre-kick-off snapshot simply isn't a backtest bet — never a closing-price fallback. Coverage is a Phase B quality metric. |
| CLV needs a Pinnacle close | `market_history.csv` carries `psc` (Pinnacle close) for the European leagues; the **non-UEFA leagues have no closing feed**, so CLV — and therefore staking — can't activate there even once volume exists. Those stay information-only for longer. |
| Confidence buckets over-fit | Buckets are preregistered (50–55/55–60/…) and per-league activation needs ≥200 bets, so a lucky thin bucket can't unlock staking. |
| Model changes mid-accumulation | The ledger records the model hash per bet; a model change invalidates prior bets for gate purposes (same discipline as the walk-forward cache fingerprint). Worth knowing: promoting anything big resets the clock. |

---

## 6. Recommendation

Build **Phase A now** — it's self-contained, gives you the strike-rate table
immediately, and makes the accumulation visible. Treat Phase B as a background
process the daily run feeds. Don't touch the gate criteria; let it open itself.

One decision for you before I start Phase A: **should the backtested strategy be
"bet every full-evidence pick above a confidence threshold", or "bet only the
sweet-spot (confident + value) picks"?** The first gives more volume (reaches
1,000 sooner) and a fuller strike-rate picture; the second is closer to what you'd
actually stake. I'd build the engine to score **both from the same replay**, so
you see the strike-rate of the confidence strategy and the yield of the value
subset side by side — but tell me if you'd rather it lead with one.

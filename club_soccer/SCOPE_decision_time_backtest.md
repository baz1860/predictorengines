# Decision-time evidence contract

**Status:** implemented as `decision_time_v3`; staking remains fail-closed until
forward evidence earns it.

## Separate questions, separate ledgers

- `forecast_ledger.csv` measures calibration and accuracy of every published
  probability, including fixtures with no market quote.
- `decision_ledger.csv` freezes executable decisions and prices.
- `closing_market_ledger_v2.csv` preserves complete raw closing markets per
  bookmaker. It is never reconstructed from results or a median price.
- `settlement_clv_v2.csv` records power-de-vigged fair CLV and same-book raw
  price CLV. Legacy proportional CLV remains clearly labelled and diagnostic.

`market_diagnostics.py` compares first-published, T-24, and latest-pre-kickoff
forecasts with the latest complete pre-kickoff median snapshot. Its blend and
calibration results have no promotion authority and cannot open staking.

## Compatibility contract

Evidence compatibility is controlled by the explicit version and manifest in
`strategy_contract.py`. Pricing, selection, execution, eligibility, or staking
semantics require a deliberate version bump. Model refits, comments, logging,
and unrelated alias-map edits do not reset history.

Resolver and byte hashes remain provenance. If identity review finds a bad
fixture, `decision_ledger.review_identity()` appends an exclusion for only that
decision or fixture; a later audited reinstatement reverses it. Retired strategy
cohorts are reported beside current evidence but cannot feed the gate.

## Gate requirements

The `decision_time_v3` artifact declares:

- selection from the first complete quote inside the 60–120 minute window;
- execution at the best price among complete-market books at that instant;
- CLV from a frozen complete raw closing market;
- power-consensus fair probability plus same-executing-book raw price CLV;
- current strategy version, immutable-ledger hashes, and current/all-history
  diagnostic views.

Every 2%, 4%, and 6% threshold must independently clear all applicable bars:

1. at least 1,000 settled bets, with at least 200 CLV observations and 80%
   fair-CLV and raw-price-CLV coverage;
2. positive flat and quarter-Kelly ROI, with simultaneous week-block lower
   bounds above zero;
3. positive fair CLV, positive same-book raw price CLV, a positive simultaneous
   CLV bound, and a Wilson lower bound above 50% for positive fair CLV;
4. at least eight independent ISO-week blocks;
5. for 1X2, the paired model-minus-market log-loss upper bound is below zero;
6. a league separately clears its 200-bet evidence floor before it can stake.

The six market/threshold rows share a family-wise 5% error budget. Undefined
bounds, missing raw closes, stale artifacts, mixed strategy versions, or malformed
JSON all fail closed. There is no runtime override.

## Operational safety

The Mac mini is the only production writer. Launch jobs declare
`CLUB_SOCCER_WRITER_HOST`; record, close, settle, season, and transfer writes
abort on another host. CSV appends are locked and flushed, and derived reports
are replaced atomically.

Syncthing conflict copies are preserved for reconciliation and are not canonical
inputs. Run `python3 -m club_soccer.runtime_safety --json` to list them. Reconcile
append-only ledgers by their immutable keys, then regenerate derived artifacts;
never merge derived JSON or Markdown reports.

## Commands

```bash
python3 -m club_soccer.decision_ledger --status
python3 -m club_soccer.decision_ledger --exclude-fixture FIXTURE_ID --reason "identity mismatch"
python3 -m club_soccer.decision_ledger --reinstate-fixture FIXTURE_ID --reason "reviewed and confirmed"
python3 -m club_soccer.decision_time_backtest --report
python3 -m club_soccer.market_diagnostics
python3 -m club_soccer.runtime_safety --json
```

Run `market_diagnostics --write` only on the authoritative writer when a frozen
diagnostic artifact is wanted. Reading/printing it is the default.

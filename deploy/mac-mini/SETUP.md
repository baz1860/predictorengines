# Club-soccer on the Mac mini — LaunchDaemons + Syncthing

Goal: the whole `club_soccer` pipeline runs 24/7 on the **Mac mini** (which is
always on), and a read-only mirror of its runtime data lands on your **MacBook**
(and, optionally, the **NAS** as backup). Works on your LAN and over Tailscale
when you're remote.

The mini is the single source of truth. The MacBook only ever *reads* the synced
copy, so nothing races and it doesn't matter if the laptop is asleep.

Fill these in once and reuse them below:

| Placeholder            | Example                                        |
|------------------------|------------------------------------------------|
| `REPLACE_USER`         | `barrie` (the mini's login user)               |
| `REPLACE_PROJECT_DIR`  | `/Users/barrie/AI Models/Soccer Prediction`    |
| `REPLACE_PYDIR`        | `/Users/barrie/.pyenv/versions/3.12.7/bin`     |
| `REPLACE_PYBIN`        | `/Users/barrie/.pyenv/versions/3.12.7/bin/python3` |

---

## Part A — Prepare the Mac mini

1. **Get the code onto the mini.** Clone your repo (or copy the folder) to
   `REPLACE_PROJECT_DIR`. If it's a git repo, `git clone` is cleanest.

2. **Python 3.12 + dependencies.** Match what the laptop uses. With pyenv:

   ```bash
   brew install pyenv          # if not already installed
   pyenv install 3.12.7
   cd "REPLACE_PROJECT_DIR"
   pyenv local 3.12.7
   # Install the pipeline's runtime deps into THIS interpreter (the one the
   # daemons use). These are the four declared in pyproject.toml [project].
   # Use the interpreter's full path so it can't land in the wrong python.
   REPLACE_PYBIN -m pip install --upgrade pip
   REPLACE_PYBIN -m pip install numpy pandas scipy requests
   ```

   Confirm: `pyenv which python3` prints `REPLACE_PYBIN`, and
   `REPLACE_PYBIN -c "import numpy, pandas, scipy, requests; print('deps OK')"`
   succeeds. (The `install.sh` preflight checks this and will stop with
   "can't import pandas/numpy" if the deps are missing from that interpreter.)

3. **BSD API key.** Put it where the laptop has it — either `api_keys.py` in the
   project root, or export `BSD_API_KEY`. The daemon runs as `REPLACE_USER`, so a
   key in that user's `api_keys.py` is picked up automatically.

4. **Create the log folder** (the plists write here):

   ```bash
   mkdir -p "REPLACE_PROJECT_DIR/logs"
   ```

5. **Smoke-test both jobs by hand before daemonising.** If these don't work
   manually, the daemon won't either.

   ```bash
   cd "REPLACE_PROJECT_DIR"
   python3 -m club_soccer.decision_ledger --record --settle   # the capture job
   python3 -m club_soccer.decision_ledger --status            # should print counts
   bash club_soccer/update.sh                   # first cache migration may be slow
   ```

   The installed plists set `CLUB_SOCCER_WRITER_HOST=bingo.local`. Confirm
   `hostname` returns that value, or change the plist value to the mini's exact
   hostname before installing it. This guard prevents a synced MacBook copy
   from becoming a second ledger writer. Also run:

   ```bash
   python3 -m club_soccer.runtime_safety --json
   ```

   Any reported conflict copy must be reviewed; it is preserved and ignored as
   a canonical input. Reconcile append-only ledgers by immutable keys and
   regenerate derived JSON/Markdown artifacts on the mini.

---

## Part B — Install the two LaunchDaemons

The plists live in `deploy/mac-mini/`. They use `REPLACE_*` tokens; fill them in,
copy to `/Library/LaunchDaemons/`, then load.

1. **Fill in the placeholders** (run on the mini, from the project dir). This
   writes the edited copies to `/tmp` so the originals stay clean:

   ```bash
   cd "REPLACE_PROJECT_DIR/deploy/mac-mini"
   USER_="barrie"
   DIR_="/Users/barrie/AI Models/Soccer Prediction"
   PYDIR_="/Users/barrie/.pyenv/versions/3.12.7/bin"
   PYBIN_="$PYDIR_/python3"

   for f in com.barrie.sportspredictor.clubsoccer.capture.plist \
            com.barrie.sportspredictor.clubsoccer.season.plist; do
     sed -e "s#REPLACE_USER#$USER_#g" \
         -e "s#REPLACE_PROJECT_DIR#$DIR_#g" \
         -e "s#REPLACE_PYDIR#$PYDIR_#g" \
         -e "s#REPLACE_PYBIN#$PYBIN_#g" \
         "$f" > "/tmp/$f"
   done
   ```

   (The `#` delimiter in `sed` lets the paths contain `/`. Paths with spaces are
   fine inside the plist XML.)

2. **Install with root ownership and correct perms** (LaunchDaemons must be
   `root:wheel`, `644`):

   ```bash
   sudo cp /tmp/com.barrie.sportspredictor.clubsoccer.capture.plist /Library/LaunchDaemons/
   sudo cp /tmp/com.barrie.sportspredictor.clubsoccer.season.plist  /Library/LaunchDaemons/
   sudo chown root:wheel /Library/LaunchDaemons/com.barrie.sportspredictor.clubsoccer.*.plist
   sudo chmod 644        /Library/LaunchDaemons/com.barrie.sportspredictor.clubsoccer.*.plist
   ```

3. **Load them** (modern `bootstrap` syntax on macOS 11+):

   ```bash
   sudo launchctl bootstrap system /Library/LaunchDaemons/com.barrie.sportspredictor.clubsoccer.capture.plist
   sudo launchctl bootstrap system /Library/LaunchDaemons/com.barrie.sportspredictor.clubsoccer.season.plist
   ```

   The capture daemon has `RunAtLoad` = true, so it fires once immediately.

4. **Verify:**

   ```bash
   sudo launchctl list | grep clubsoccer          # both labels should appear
   tail -f "REPLACE_PROJECT_DIR/logs/club_soccer_capture.log"
   ```

   Within ~15 min you should see `decision_ledger: recorded N ... / settled N`.

**To change or remove a daemon** (you must `bootout` before re-bootstrapping an
edited plist):

```bash
sudo launchctl bootout system /Library/LaunchDaemons/com.barrie.sportspredictor.clubsoccer.capture.plist
# edit / re-copy, then bootstrap again
```

To force an immediate capture run without waiting:

```bash
sudo launchctl kickstart -k system/com.barrie.sportspredictor.clubsoccer.capture
```

> Note: keep the old **LaunchAgent** versions (in `deploy/`) OUT of the mini —
> you don't want the capture running twice. Use the LaunchDaemons here instead.

---

## Part C — Syncthing (mini → MacBook, + NAS backup)

Sync only the **runtime data folder**, not the code. Code stays in git on both
machines; Syncthing just mirrors what the pipeline writes. This avoids any
code-vs-sync conflict. The folder to share:

```
REPLACE_PROJECT_DIR/club_soccer/data
```

That holds the decision/settlement ledgers, the card forecast/scoring ledgers,
`model_params.json`, the gate artifact `backtest_market.json`, and the
coefficient files — everything the viewer needs. (If you also want the root-level `dashboard.html` /
`edge_report.csv` / `daily_card.md`, add a second shared folder for those later.)

### C1. Install Syncthing

- **Mac mini** and **MacBook**:
  ```bash
  brew install syncthing
  brew services start syncthing     # runs it as a background service, 24/7 on the mini
  ```
  Web UI: `http://127.0.0.1:8384`.

- **Synology NAS** (optional, backup only): Package Center → Settings → add the
  SynoCommunity source `https://packages.synocommunity.com/`, then install
  **Syncthing** from Community packages. Open its UI at `http://NAS-IP:8384`.
  (Alternatively run Syncthing as a container in Container Manager.)

### C2. Point the devices at each other over Tailscale

Each Syncthing instance has a **Device ID** (Actions → Show ID). With Tailscale
you get a stable IP for every machine, so add peers by their tailnet address and
it works identically at home or remote.

On the **mini**, add the MacBook (and NAS) as remote devices:

- Web UI → **Add Remote Device** → paste the MacBook's Device ID.
- Under **Advanced → Addresses**, replace `dynamic` with the MacBook's Tailscale
  address so it always connects over the tailnet:
  `tcp://100.x.y.z:22000`  (use the MacBook's `100.` Tailscale IP).
- Repeat for the NAS if you're using it.

Accept the pairing prompt that pops up on the MacBook/NAS.

### C3. Share the data folder — one-way, mini is master

On the **mini**:

- **Add Folder** → Folder Path = `REPLACE_PROJECT_DIR/club_soccer/data`,
  give it a Folder ID like `clubsoccer-data`.
- **Sharing** tab → tick the MacBook (and NAS).
- **Advanced** tab → **Folder Type = Send Only**. The mini only ever pushes.

On the **MacBook** (and NAS) accept the shared folder, choose where to store it,
and set:

- **Folder Type = Receive Only**.

Receive-Only means the laptop can never write back into that folder, so the mini
stays authoritative and there are no sync conflicts. (If you ever *do* edit a
file in it on the laptop, Syncthing flags it and offers a one-click "Revert local
changes" — that's the safety net working as intended.)

### C4. Sync ONLY the runtime files (this is what keeps laptop dev safe)

`club_soccer/data` holds two disjoint kinds of file:

- **git-tracked inputs**: `club_alias_map.json`, `club_registry.json`,
  `uefa_coefficients*.json`, calibration/evidence files, etc. These are managed by
  **git** and may be edited during development on the laptop.
- **git-ignored runtime outputs**: decision and forecast ledgers,
  `backtest_market.json`, `forecast_performance.json`, `model_params.json`, and
  `fixtures.csv`. These are produced by the pipeline on the mini.

You only want Syncthing to carry the **runtime outputs**. If it also synced the
tracked inputs, editing one on the laptop would fight the Receive-Only mirror.
Because the two sets are disjoint (ignored vs tracked), an allowlist keeps git
and Syncthing completely out of each other's way in the same folder.

In the shared folder's **Ignore Patterns**, use an allowlist — Syncthing is
first-match-wins, so the `!` includes win and the final `*` ignores everything
else:

```
!/decision_ledger.csv
!/settlement_ledger.csv
!/decision_strategy_ledger.csv
!/identity_exclusions.csv
!/closing_ledger.csv
!/closing_market_ledger_v2.csv
!/settlement_clv_v2.csv
!/decision_time_ledger.csv
!/backtest_market.json
!/market_diagnostics.json
!/forecast_ledger.csv
!/forecast_settlements.csv
!/forecast_performance.json
!/model_params.json
!/fixtures.csv
!/card.md
!/last_run.json
!/run_history.jsonl
!/validation_latest.json
!/odds_history_club.csv
!/absences_club.csv
!/transfers_bsd.sqlite3
!/market_history.csv
!/squads_club.csv
!/transfers_detected.csv
*
```

Set the **same** ignore patterns on every device sharing the folder. Now
Syncthing touches only the allow-listed generated files; git owns the rest.

### C5. Secure the remote GUI (if you'll admin it remotely)

The Syncthing GUI defaults to `127.0.0.1` only. To manage the mini's Syncthing
over Tailscale, either SSH-tunnel to it, or in **Settings → GUI** set the
listen address to the mini's Tailscale IP `100.x.y.z:8384` and set a
username/password. Don't bind it to `0.0.0.0` without a password.

---

## Part D — End-to-end verification

1. On the **mini**, force a capture and confirm the ledger grows:
   ```bash
   sudo launchctl kickstart -k system/com.barrie.sportspredictor.clubsoccer.capture
   python3 -m club_soccer.decision_ledger --status
   ```
2. On the **MacBook**, open the synced `data` folder and confirm
   `decision_ledger.csv` / `settlement_ledger.csv` appear and update within a
   minute or two of the mini writing them (Syncthing web UI shows "Up to Date").
3. Disconnect the MacBook from the LAN, connect via Tailscale only, and confirm
   syncing still resumes — the folder should catch up when it reconnects.

---

## Part E — Developing on the laptop (won't clash)

Code development stays on the laptop; the mini just runs what git gives it.

- **Code changes**: edit on the laptop → `git commit` → `git push`. Syncthing
  never touches code, so there's no interaction at all. To make the mini use the
  change, `git pull` on the mini (the daemon won't pull for you — either pull by
  hand, or add `git pull --ff-only` to the top of `update.sh`).
- **Editing a tracked input** (alias map, coefficients, registry): same git loop.
  It's a Syncthing-ignored file, so the mirror ignores it; the mini picks it up on
  the next `git pull`.
- **Running tests on the laptop**: fine. The suite is hermetic (temp dirs /
  monkeypatched paths), so it never writes into the synced `data` folder.
- **The one thing to avoid**: don't run the *production* pipeline
  (`update.sh` / `--record`) on the laptop against the synced folder — it would
  write the Receive-Only runtime files and Syncthing would revert them. If you
  need a full local run for debugging, do it in a separate clone outside the
  synced folder.

In short: git owns code and tracked inputs (edit anywhere, mini pulls);
Syncthing owns the five generated runtime files (mini writes, laptop mirrors).
They never touch the same file.

### Incremental-run behaviour

The 07:30 job retains the same fail-closed safety checks, but unchanged work is
reused. Expect one slow run after installing this version: it creates the
player-event manifest and invalidates validation/model fingerprints because
their code changed. Normal later runs ingest only new event files and refit or
revalidate only when the relevant inputs change. To deliberately bypass the
caches:

```bash
python3 -m club_soccer.model --fit
python3 -m club_soccer.validate --gate
python3 -m club_soccer.decision_time_backtest --force
```

## Troubleshooting quick hits

- **`launchctl list` shows the label but it never runs.** Check the log files in
  `REPLACE_PROJECT_DIR/logs/`. A non-zero exit or a Python traceback there tells
  you it's an env/path issue — usually `REPLACE_PYBIN` wrong or a missing pip
  package. Fix, `bootout`, `bootstrap` again.
- **"Load failed: 5: Input/output error".** The plist perms/owner are wrong —
  re-run the `chown root:wheel` / `chmod 644` step.
- **Capture records 0 every time.** Either no fixtures are currently 60–120 min
  from kickoff (normal off-peak), or the BSD key isn't visible to `REPLACE_USER`.
  Test manually: `python3 -m club_soccer.decision_ledger --record`.
- **Syncthing "Out of Sync" that won't clear.** Almost always a local edit on a
  Receive-Only side — use **Revert Local Changes** on the MacBook/NAS; the mini's
  copy wins.
- **Two captures running.** Make sure the old `deploy/*.capture.plist`
  LaunchAgent isn't also loaded on the mini (`launchctl list | grep clubsoccer`).

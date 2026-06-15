#!/usr/bin/env python3
"""M2 security & reliability hardening tests.

Covers the V3 M2 acceptance criteria:
  * settings never return raw keys;
  * a synthetic runner error containing a fake key is redacted;
  * unknown engine ids / bad slugs are rejected, never hit the filesystem;
  * oversized params are rejected and numeric params are clamped;
  * the runner environment is curated (no secret leakage), key env vars kept;
  * api_keys file is written owner-only on POSIX.

Run: python3 test_security.py
"""
from __future__ import annotations

import os
import stat
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PASS, FAIL = 0, 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


# ── redaction ─────────────────────────────────────────────────────────────────
def test_redaction():
    from app.security import redact
    fake = "sk_" + "live_" + ("A" * 24)
    msg = f"HTTP 401 for https://api.example.com?key={fake} denied"
    out = redact(msg, secrets=[fake])
    check("known key value redacted", fake not in out, out)
    # unknown long token also masked even without being in the secrets list
    out2 = redact(f"token leaked: {fake}")
    check("unknown key-looking token masked", fake not in out2, out2)
    check("redaction keeps readable context", "HTTP 401" in out, out)


# ── curated subprocess env ──────────────────────────────────────────────────--
def test_safe_env():
    from app.security import safe_runner_env
    os.environ["TOTALLY_SECRET_THING"] = "supersecret-value-leak"
    os.environ["THE_ODDS_API_KEY"] = "envkey123456789012345"
    try:
        env = safe_runner_env({"PYTHONPATH": "/x"})
    finally:
        os.environ.pop("TOTALLY_SECRET_THING", None)
    check("unrelated secret env dropped", "TOTALLY_SECRET_THING" not in env)
    check("key env var preserved", env.get("THE_ODDS_API_KEY") == "envkey123456789012345")
    check("PYTHONPATH injected", env.get("PYTHONPATH") == "/x")
    check("PATH preserved", "PATH" in env or os.name == "nt")


# ── hardened run_engine ───────────────────────────────────────────────────────
def test_run_engine_rejects_unknown_command():
    from app.engines._subprocess import run_engine
    try:
        run_engine(ROOT, ROOT / "preflight.py", "rm -rf /", {})
        check("unknown command rejected", False, "no error raised")
    except ValueError as e:
        check("unknown command rejected before launch", "Unknown runner command" in str(e))
    except Exception as e:  # noqa
        check("unknown command rejected before launch", False, repr(e))


def test_run_engine_redacts_runner_error():
    """A runner that emits an error carrying a fake key must surface redacted."""
    from app.engines._subprocess import run_engine
    fake = "leaked_KEY_" + ("b" * 24)
    with tempfile.TemporaryDirectory() as d:
        runner = Path(d) / "bad_runner.py"
        runner.write_text(
            "import json,sys\n"
            f"print(json.dumps({{'error': 'auth failed key={fake}'}}))\n"
            "sys.exit(2)\n")
        try:
            run_engine(Path(d), runner, "edge", {})
            check("runner error redacted", False, "no error raised")
        except ValueError as e:
            check("runner error redacted", fake not in str(e), str(e))
        except Exception as e:  # noqa
            check("runner error redacted", False, repr(e))


def test_run_engine_rejects_nonfinite():
    from app.engines._subprocess import run_engine
    with tempfile.TemporaryDirectory() as d:
        runner = Path(d) / "inf_runner.py"
        runner.write_text(
            "import sys\n"
            "print('{\"rows\": [{\"x\": Infinity}]}')\n")
        try:
            run_engine(Path(d), runner, "edge", {})
            check("non-finite JSON rejected", False, "no error raised")
        except RuntimeError as e:
            check("non-finite JSON rejected", "invalid JSON" in str(e), str(e))
        except Exception as e:  # noqa
            check("non-finite JSON rejected", False, repr(e))


# ── API request bounds ────────────────────────────────────────────────────────
def test_request_bounds():
    from app.server import EngineRequest
    from pydantic import ValidationError
    # bad slug (path traversal attempt) rejected
    for bad in ["../../etc/passwd", "a/b", "DROP TABLE", ""]:
        try:
            EngineRequest(engine=bad, params={})
            check(f"bad slug rejected: {bad!r}", False)
        except ValidationError:
            check(f"bad slug rejected: {bad!r}", True)
    # oversized params rejected
    try:
        EngineRequest(engine="worldcup", params={"junk": "x" * 60_000})
        check("oversized params rejected", False)
    except ValidationError:
        check("oversized params rejected", True)
    # numeric clamp applied
    req = EngineRequest(engine="golf", params={"sims": 9_999_999, "kelly": 5.0})
    check("sims clamped", req.params["sims"] == 200_000, str(req.params))
    check("kelly clamped", req.params["kelly"] == 1.0, str(req.params))
    # valid request untouched
    req2 = EngineRequest(engine="cfb", params={"model": "blend", "sims": 2000})
    check("valid params preserved", req2.params["sims"] == 2000)


# ── settings never leak raw keys ──────────────────────────────────────────────
def test_settings_masked():
    from app import settings_store
    view = settings_store.public_view()
    masked = view.get("odds_api_keys_masked", {})
    raw_ok = all("…" in v or v == "" for v in masked.values())
    check("public settings only expose masked keys", raw_ok, str(masked))
    check("public view has no raw odds_api_keys field",
          not any(isinstance(v, str) and len(v) > 12 for v in masked.values()))


# ── owner-only key file ───────────────────────────────────────────────────────
def test_key_file_perms():
    if os.name == "nt":
        check("key file perms (skipped on Windows)", True)
        return
    from api_keys import save_keys
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "keys.json"
        save_keys({"the-odds-api": "abc123"}, path=path)
        mode = stat.S_IMODE(os.stat(path).st_mode)
        check("key file is owner-only (0600)", mode == 0o600, oct(mode))


# ── preflight runs offline ────────────────────────────────────────────────────
def test_preflight():
    import preflight
    report = preflight.build_report()
    check("preflight reports all engines",
          set(report["engines"]) == {"worldcup", "club_soccer", "cfb", "golf"},
          str(set(report["engines"])))


def main():
    for fn in [test_redaction, test_safe_env,
               test_run_engine_rejects_unknown_command,
               test_run_engine_redacts_runner_error,
               test_run_engine_rejects_nonfinite,
               test_request_bounds, test_settings_masked,
               test_key_file_perms, test_preflight]:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

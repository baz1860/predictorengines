"""Generate the CFB README evidence section from frozen validation artifacts."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
README = HERE / "README.md"
START = "<!-- CFB_METRICS_START -->"
END = "<!-- CFB_METRICS_END -->"


def _load(name: str) -> dict:
    return json.loads((HERE / "data" / name).read_text())


def render() -> str:
    nested = _load("nested_validation_2025.json")
    market = _load("market_validation_2025.json")
    policy = _load("market_policy.json")
    weights = _load("blend_weight.json")
    selection, holdout = nested["selection"], nested["holdout"]
    n = selection["n_games"] + holdout["n_games"]

    def combined(metric: str) -> float:
        return ((selection[metric] * selection["n_games"]
                 + holdout[metric] * holdout["n_games"]) / n)

    ats, total = nested["markets"]["ats"], nested["markets"]["total"]
    spread_ch = market["markets"]["spread"]["scores"]["calibrated_discrete"]
    total_ch = market["markets"]["total"]["scores"]["calibrated_discrete"]
    line_hash = nested["data_fingerprint"]["line_sha256"][:16]
    lines = [
        START,
        "## Frozen validation evidence",
        "",
        f"Runtime blend: **{weights['w_elo']:.0%} Elo / {1-weights['w_elo']:.0%} power**, "
        f"selected on {nested['selection_window']} ({nested['selection_games']:,} games) "
        f"before the untouched {nested['holdout_season']} holdout "
        f"({nested['holdout_games']:,} games).",
        "",
        "| Window | Games | ML Brier | Accuracy | Margin MAE | Total MAE |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Selection {nested['selection_window']} | {selection['n_games']:,} | "
        f"{selection['ml_brier']:.5f} | {selection['ml_acc']:.1%} | "
        f"{selection['margin_mae']:.3f} | {selection['total_mae']:.3f} |",
        f"| Holdout {nested['holdout_season']} | {holdout['n_games']:,} | "
        f"{holdout['ml_brier']:.5f} | {holdout['ml_acc']:.1%} | "
        f"{holdout['margin_mae']:.3f} | {holdout['total_mae']:.3f} |",
        f"| Combined | {n:,} | {combined('ml_brier'):.5f} | "
        f"{combined('ml_acc'):.1%} | {combined('margin_mae'):.3f} | "
        f"{combined('total_mae'):.3f} |",
        "",
        "The frozen 2025 closing-line benchmark at a three-point disagreement:",
        "",
        "| Market | W-L-P | Bets | ROI | Week-block 95% CI | Policy |",
        "|---|---:|---:|---:|---:|---|",
        f"| Spread | {ats['won']}-{ats['lost']}-{ats['push']} | {ats['n']} | "
        f"{ats['roi']:+.1%} | [{ats['roi_ci_95'][0]:+.1%}, "
        f"{ats['roi_ci_95'][1]:+.1%}] | {policy['spread']} |",
        f"| Total | {total['won']}-{total['lost']}-{total['push']} | {total['n']} | "
        f"{total['roi']:+.1%} | [{total['roi_ci_95'][0]:+.1%}, "
        f"{total['roi_ci_95'][1]:+.1%}] | {policy['total']} |",
        "",
        "The push-aware calibrated-discrete challenger remains unpromoted. "
        f"Spread holdout Brier/ECE are {spread_ch['multiclass_brier']:.5f}/"
        f"{spread_ch['ece']:.5f}; totals are {total_ch['multiclass_brier']:.5f}/"
        f"{total_ch['ece']:.5f}. Neither market cleared the held-out betting gate.",
        "",
        f"Validation line fingerprint: `{line_hash}`. Regenerate with "
        "`python3 -m cfb.generate_docs --write`; CI-style drift check: "
        "`python3 -m cfb.generate_docs --check`.",
        END,
    ]
    return "\n".join(lines)


def replace(text: str, section: str) -> str:
    if START not in text or END not in text:
        raise ValueError("CFB README generated-section markers are missing")
    before, tail = text.split(START, 1)
    _, after = tail.split(END, 1)
    return before + section + after


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    current = README.read_text()
    expected = replace(current, render())
    if args.check:
        if current != expected:
            raise SystemExit("CFB README generated metrics are stale; run --write")
        print("CFB README generated metrics are current")
        return
    if args.write:
        _atomic_write(README, expected)
        print(f"updated {README}")
        return
    print(render())


if __name__ == "__main__":
    main()

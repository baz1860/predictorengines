"""Regularized race-level conditional-logit model and coherent calibration."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize, minimize_scalar

from .features import FEATURES, FEATURE_SCHEMA_VERSION, build_feature_frame
from .config import (ALLOWED_CODES, ALLOWED_JURISDICTIONS, DEFAULT_CUTOFF_MINUTES)
from .schema import (DATA_DIR, DataBundle, DataError, load_bundle, race_cutoff,
                     race_row, runner_snapshot)

ARTIFACT_PATH = DATA_DIR / "model_params.json"
MODEL_NAME = "regularized_conditional_logit_v2"


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_provenance(repo_root: str | Path | None = None) -> dict:
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"],
                                cwd=root,
                                capture_output=True, text=True, timeout=3, check=False)
        commit = result.stdout.strip() if result.returncode == 0 else ""
        status = subprocess.run(["git", "status", "--porcelain", "--", "horse_racing"],
                                cwd=root, capture_output=True, text=True,
                                timeout=3, check=False)
        dirty = status.returncode != 0 or bool(status.stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": "", "dirty": True}


def _code_sha256() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _data_provenance(data_dir: Path) -> dict | None:
    path = data_dir / "provider_manifest.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return {"provider": "unknown", "validation_grade": "invalid_manifest"}
    if not isinstance(payload, dict):
        return {"provider": "unknown", "validation_grade": "invalid_manifest"}
    return {key: payload.get(key) for key in (
        "provider", "validation_grade", "assumptions", "rpscrape_commit",
        "rpscrape_safety_patch_sha256", "raw_files") if key in payload}


def _valid_labelled(frame: pd.DataFrame) -> pd.DataFrame:
    valid = []
    for rid, group in frame.groupby("race_id", sort=False):
        if len(group) >= 2 and int(group["won"].sum()) == 1:
            valid.append(str(rid))
    return frame[frame["race_id"].astype(str).isin(valid)].copy()


def _matrix(frame: pd.DataFrame, means=None, scales=None, features=None):
    x = frame[list(features) if features is not None else FEATURES].astype(float).to_numpy()
    if means is None:
        means = np.nanmean(x, axis=0)
    if scales is None:
        scales = np.nanstd(x, axis=0)
        scales = np.where(scales > 1e-9, scales, 1.0)
    x = np.nan_to_num((x - means) / scales, nan=0.0, posinf=0.0, neginf=0.0)
    return x, np.asarray(means), np.asarray(scales)


def _groups(frame: pd.DataFrame) -> list[np.ndarray]:
    return [np.asarray(idx, dtype=int) for idx in frame.groupby("race_id", sort=False).indices.values()]


def _group_layout(frame: pd.DataFrame):
    """Contiguous group layout for vectorized grouped-softmax computation.

    Returns a row permutation making races contiguous, the group start
    offsets within that permutation, and each permuted row's group index.
    """
    groups = _groups(frame)
    order = np.concatenate(groups)
    lengths = np.asarray([len(idx) for idx in groups], dtype=int)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    row_group = np.repeat(np.arange(len(lengths)), lengths)
    return order, starts, row_group


def _grouped_softmax(z: np.ndarray, starts: np.ndarray,
                     row_group: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Probabilities and per-group log-sum-exp over contiguous groups."""
    gmax = np.maximum.reduceat(z, starts)
    expz = np.exp(z - gmax[row_group])
    sums = np.add.reduceat(expz, starts)
    return expz / sums[row_group], gmax + np.log(sums)


def _fit_coefficients(frame: pd.DataFrame, l2: float = 1.0, features=None):
    names = list(features) if features is not None else FEATURES
    frame = frame.reset_index(drop=True)
    x, means, scales = _matrix(frame, features=names)
    y = frame["won"].astype(float).to_numpy()
    order, starts, row_group = _group_layout(frame)
    x_o, y_o = x[order], y[order]
    winner_rows = np.flatnonzero(y_o == 1.0)

    def objective(beta):
        z = x_o @ beta
        probs, lse = _grouped_softmax(z, starts, row_group)
        loss = 0.5 * l2 * float(beta @ beta) + float(lse.sum() - z[winner_rows].sum())
        grad = l2 * beta + x_o.T @ (probs - y_o)
        return loss, grad

    result = minimize(lambda b: objective(b), np.zeros(len(names)), jac=True,
                      method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-11})
    if not result.success:
        raise RuntimeError(f"conditional-logit fit failed: {result.message}")
    return result.x, means, scales, float(result.fun)


def _probabilities(frame: pd.DataFrame, beta, means, scales,
                   temperature: float = 1.0, features=None) -> np.ndarray:
    frame = frame.reset_index(drop=True)
    x, _, _ = _matrix(frame, np.asarray(means), np.asarray(scales), features=features)
    logits = x @ np.asarray(beta)
    order, starts, row_group = _group_layout(frame)
    z = logits[order] / max(float(temperature), 0.05)
    probs, _ = _grouped_softmax(z, starts, row_group)
    out = np.zeros(len(frame), dtype=float)
    out[order] = probs
    return out


def _temperature(frame: pd.DataFrame, beta, means, scales, features=None) -> float:
    frame = frame.reset_index(drop=True)
    y = frame["won"].astype(float).to_numpy()
    x, _, _ = _matrix(frame, np.asarray(means), np.asarray(scales), features=features)
    logits = x @ np.asarray(beta)
    order, starts, row_group = _group_layout(frame)
    z_o, y_o = logits[order], y[order]
    winner_rows = np.flatnonzero(y_o == 1.0)
    n_groups = len(starts)

    def nll(log_t):
        t = float(np.exp(log_t))
        z = z_o / t
        _, lse = _grouped_softmax(z, starts, row_group)
        return float(lse.sum() - z[winner_rows].sum()) / max(n_groups, 1)

    result = minimize_scalar(nll, bounds=(np.log(0.25), np.log(4.0)), method="bounded")
    return float(np.exp(result.x)) if result.success else 1.0


def _race_metrics(frame: pd.DataFrame, p: np.ndarray) -> dict:
    work = frame.reset_index(drop=True).copy()
    work["p"] = p
    losses, briers = [], []
    for _rid, group in work.groupby("race_id", sort=False):
        winner = group[group["won"] == 1]
        if len(winner) != 1:
            continue
        losses.append(-np.log(max(float(winner.iloc[0]["p"]), 1e-12)))
        briers.append(float(np.sum((group["p"].to_numpy() - group["won"].to_numpy()) ** 2)))
    return {"races": len(losses), "log_loss": float(np.mean(losses)) if losses else None,
            "race_brier": float(np.mean(briers)) if briers else None}


def fit(bundle: DataBundle | None = None, data_dir: str | Path | None = None,
        min_races: int = 30, l2: float = 1.0,
        cutoff_minutes: int = DEFAULT_CUTOFF_MINUTES,
        half_life_scale: float = 1.0) -> dict:
    bundle = bundle or load_bundle(data_dir)
    frame = _valid_labelled(build_feature_frame(bundle, cutoff_minutes=cutoff_minutes,
                                                half_life_scale=half_life_scale))
    race_order = (frame[["race_id", "scheduled_off_utc"]].drop_duplicates()
                  .sort_values("scheduled_off_utc"))
    n_races = len(race_order)
    if n_races < min_races:
        raise DataError(f"need at least {min_races} completed valid races; found {n_races}")

    split = max(10, int(n_races * 0.8))
    split = min(split, n_races - max(5, int(n_races * 0.1)))
    train_ids = set(race_order.iloc[:split]["race_id"].astype(str))
    cal_ids = set(race_order.iloc[split:]["race_id"].astype(str))
    train = frame[frame["race_id"].astype(str).isin(train_ids)]
    cal = frame[frame["race_id"].astype(str).isin(cal_ids)]
    beta0, means0, scales0, _ = _fit_coefficients(train, l2=l2)
    temperature = _temperature(cal, beta0, means0, scales0) if len(cal_ids) >= 5 else 1.0

    # Refit the structural model on all available races. Temperature remains
    # learned from the strictly later held-out calibration slice.
    beta, means, scales, objective = _fit_coefficients(frame, l2=l2)
    p_all = _probabilities(frame, beta, means, scales, temperature)
    metrics = _race_metrics(frame, p_all)
    data_hash = hashlib.sha256(pd.util.hash_pandas_object(
        frame[["race_id", "runner_id", "won", *FEATURES]], index=False
    ).values.tobytes()).hexdigest()
    git = git_provenance()
    data_provenance = _data_provenance(bundle.data_dir)
    artifact = {
        "model": MODEL_NAME,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "features": FEATURES,
        "coefficients": [float(v) for v in beta],
        "means": [float(v) for v in means],
        "scales": [float(v) for v in scales],
        "temperature": float(temperature),
        "l2": float(l2), "cutoff_minutes": int(cutoff_minutes),
        "half_life_scale": float(half_life_scale),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trained_through": str(race_order.iloc[-1]["scheduled_off_utc"]),
        "n_races": int(n_races), "n_runners": int(len(frame)),
        "calibration_races": int(len(cal_ids)), "objective": float(objective),
        "training_metrics": metrics, "training_data_hash": data_hash,
        "scope": {"jurisdictions": sorted(ALLOWED_JURISDICTIONS),
                  "codes": sorted(ALLOWED_CODES), "market": "win",
                  "mode": "pre_race"},
        "calibration": {"method": "temperature_scaling",
                        "split": "chronological_last_20_percent",
                        "base_refit_on_all_data": True},
        "input_checksums": {name: _file_sha256(bundle.data_dir / f"{name}.csv")
                            for name in ("races", "runners", "results")},
        "git_commit": git["commit"], "git_dirty": bool(git["dirty"]),
        "data_provenance": data_provenance,
        "deployment_eligible": bool(
            data_provenance and data_provenance.get("validation_grade") == "point_in_time"),
        "code_sha256": _code_sha256(),
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__, "pandas": pd.__version__,
                        "scipy": scipy.__version__},
    }
    return artifact


def save_artifact(artifact: dict, path: str | Path | None = None) -> Path:
    target = Path(path) if path else ARTIFACT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact, indent=2) + "\n")
    return target


def artifact_path_for(data_dir: str | Path | None = None) -> Path:
    return Path(data_dir) / "model_params.json" if data_dir else ARTIFACT_PATH


def load_artifact(path: str | Path | None = None) -> dict:
    target = Path(path) if path else ARTIFACT_PATH
    if not target.exists():
        raise DataError(f"no fitted horse-racing model at {target}; run: python -m horse_racing fit")
    artifact = json.loads(target.read_text())
    if artifact.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise DataError("model feature schema does not match this code")
    if artifact.get("features") != FEATURES:
        raise DataError("model feature list does not match this code")
    code_hash = artifact.get("code_sha256")
    if not code_hash:
        raise DataError("model artifact has no code_sha256 provenance")
    if code_hash != _code_sha256():
        raise DataError("model artifact was fitted by different horse-racing code")
    return artifact


def predict_race(race_id: str, bundle: DataBundle | None = None,
                 artifact: dict | None = None, data_dir: str | Path | None = None,
                 artifact_path: str | Path | None = None) -> pd.DataFrame:
    bundle = bundle or load_bundle(data_dir)
    artifact = artifact or load_artifact(artifact_path or artifact_path_for(data_dir))
    cutoff_minutes = int(artifact.get("cutoff_minutes", 15))
    frame = build_feature_frame(bundle, [str(race_id)], cutoff_minutes=cutoff_minutes,
                                include_labels=False,
                                half_life_scale=float(artifact.get("half_life_scale", 1.0)))
    if frame.empty:
        raise DataError(f"race {race_id!r} has fewer than two eligible runners at cutoff")
    p = _probabilities(frame, artifact["coefficients"], artifact["means"],
                       artifact["scales"], artifact.get("temperature", 1.0))
    result = frame[["race_id", "runner_id", "horse_id", "horse_name", "cutoff"]].copy()
    result["p_model"] = p
    result["fair_odds"] = 1.0 / np.clip(p, 1e-12, 1.0)
    result["model"] = artifact["model"]
    result["model_version"] = artifact.get("created_at", "")
    race = race_row(bundle, race_id)
    runners = runner_snapshot(bundle, race_id, race_cutoff(
        race, int(artifact.get("cutoff_minutes", 15))))
    official_missing = pd.to_numeric(runners["official_rating"], errors="coerce").isna().mean()
    quality = "degraded" if float(official_missing) > 0.25 else "ok"
    result["data_quality"] = quality
    if not np.isclose(float(result["p_model"].sum()), 1.0, atol=1e-10):
        raise RuntimeError("race probabilities do not sum to one")
    return result.sort_values("p_model", ascending=False).reset_index(drop=True)


def probabilities_for_frame(frame: pd.DataFrame, artifact: dict) -> np.ndarray:
    """Public validation helper; frame must contain the artifact feature schema."""
    return _probabilities(frame, artifact["coefficients"], artifact["means"],
                          artifact["scales"], artifact.get("temperature", 1.0))

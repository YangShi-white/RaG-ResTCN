#!/usr/bin/env python3
"""Run Phase 08 Raman preprocessing and lightweight multimodal fusion analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fermnftp.data import load_json  # noqa: E402
from fermnftp.metrics import write_csv  # noqa: E402
from fermnftp.plot_style import apply_ai_conference_style  # noqa: E402

apply_ai_conference_style(plt)


PHASE = "Phase 08"
EXPERIMENT = "Raman preprocessing and multimodal fusion"
EPS = 1e-8


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def ensure_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "metrics": root / "metrics",
        "models": root / "models",
        "features": root / "features",
        "predictions": root / "predictions",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def ensure_asset_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "figures": root / "figures",
        "tables": root / "tables",
        "explanations": root / "explanations",
        "manifests": root / "manifests",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def save_explanation(
    path: Path,
    title: str,
    figure_id: str,
    purpose: str,
    data_source: str,
    design: str,
    results: str,
    interpretation: str,
    discussion: str,
    limitations: str,
    paper_use: str,
    command: str,
) -> None:
    text = f"""# {title}

## 

`{figure_id}`

## 

{purpose}

## 

{data_source}

## 

{design}

## 

{results}

## 

{interpretation}

## 

{discussion}

## 

{limitations}

## 

，；CSV ，。

## 

{paper_use}

## 

```bash
{command}
```
"""
    path.write_text(text, encoding="utf-8")


def split_map(split_csv: Path) -> dict[int, dict[str, Any]]:
    rows = read_csv_dicts(split_csv)
    out = {}
    for row in rows:
        batch_id = int(float(row["batch_id"]))
        out[batch_id] = {
            "split": row["split"],
            "fault_label": int(float(row["fault_label"])),
            "n_rows": int(float(row["n_rows"])),
            "duration_h": float(row["duration_h"]),
        }
    return out


def load_batch(processed_root: Path, batch_id: int) -> dict[str, Any]:
    data = np.load(processed_root / "batches" / f"batch_{batch_id:03d}.npz", allow_pickle=True)
    return {key: data[key] for key in data.files}


def split_batch_ids(split_info: dict[int, dict[str, Any]]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for batch_id, info in sorted(split_info.items()):
        out.setdefault(info["split"], []).append(batch_id)
    return out


def unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def select_dense_targets(variable_roles: dict[str, Any], scaler: dict[str, Any], missing_ratio_max: float) -> list[str]:
    selected = []
    for col in variable_roles["endogenous_states"]["columns"]:
        stats = scaler["process_columns"].get(col)
        if stats is None:
            continue
        if float(stats["missing_ratio"]) <= missing_ratio_max:
            selected.append(col)
    return selected


def column_stats(
    scaler: dict[str, Any], columns: list[str], scale_key: str = "robust_scale"
) -> tuple[np.ndarray, np.ndarray]:
    centers = []
    scales = []
    for col in columns:
        stats = scaler["process_columns"][col]
        center = float(stats["median"])
        scale = float(stats.get(scale_key, math.nan))
        if not np.isfinite(scale) or scale == 0:
            fallback = "std" if scale_key != "std" else "robust_scale"
            scale = float(stats.get(fallback, 1.0))
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        centers.append(center)
        scales.append(scale)
    return np.asarray(centers, dtype=np.float64), np.asarray(scales, dtype=np.float64)


def standardize(values: np.ndarray, centers: np.ndarray, scales: np.ndarray) -> np.ndarray:
    out = (values - centers) / scales
    return np.where(np.isfinite(out), out, 0.0)


def inverse_standardize(values: np.ndarray, centers: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return values * scales + centers


def row_median_fill(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    out = arr.copy()
    row_median = np.nanmedian(out, axis=1)
    row_median = np.where(np.isfinite(row_median), row_median, 0.0)
    missing = ~np.isfinite(out)
    if missing.any():
        rows, _ = np.where(missing)
        out[missing] = row_median[rows]
    return out


def row_snv(x: np.ndarray) -> np.ndarray:
    arr = row_median_fill(x)
    mean = np.mean(arr, axis=1, keepdims=True)
    std = np.std(arr, axis=1, keepdims=True)
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
    return (arr - mean) / std


def odd_window(requested: int, n: int) -> int:
    window = min(int(requested), n - 1 if n % 2 == 0 else n)
    if window % 2 == 0:
        window -= 1
    return max(window, 5)


def airpls_correct_matrix(x: np.ndarray, lam: float, n_iter: int) -> np.ndarray:
    arr = row_median_fill(x)
    n_features = arr.shape[1]
    diff = sparse.diags([1, -2, 1], [0, 1, 2], shape=(n_features - 2, n_features), format="csc")
    smooth = lam * (diff.T @ diff)
    corrected = np.empty_like(arr)
    for i, spectrum in enumerate(arr):
        weights = np.ones(n_features, dtype=np.float64)
        baseline = np.zeros(n_features, dtype=np.float64)
        norm_y = np.sum(np.abs(spectrum)) + EPS
        for iteration in range(1, int(n_iter) + 1):
            w_matrix = sparse.spdiags(weights, 0, n_features, n_features, format="csc")
            baseline = spsolve(w_matrix + smooth, weights * spectrum)
            residual = spectrum - baseline
            negative = residual[residual < 0]
            neg_sum = np.sum(np.abs(negative))
            if negative.size == 0 or neg_sum < 1e-3 * norm_y:
                break
            weights[residual >= 0] = 0.0
            adaptive = np.exp(np.minimum(50.0, iteration * np.abs(residual[residual < 0]) / (neg_sum + EPS)))
            weights[residual < 0] = adaptive
            edge_weight = float(np.max(adaptive)) if adaptive.size else 1.0
            weights[0] = edge_weight
            weights[-1] = edge_weight
        corrected[i] = spectrum - baseline
    return corrected


def preprocess_spectra(spectra: np.ndarray, method: str, cfg: dict[str, Any]) -> np.ndarray:
    spectra = row_median_fill(spectra)
    window = odd_window(int(cfg["savgol_window"]), spectra.shape[1])
    polyorder = min(int(cfg["savgol_polyorder"]), window - 2)
    if method == "raw":
        return spectra
    if method == "snv":
        return row_snv(spectra)
    if method == "sg1_snv":
        derivative = savgol_filter(spectra, window_length=window, polyorder=polyorder, deriv=1, axis=1)
        return row_snv(derivative)
    if method == "sg2_snv":
        derivative = savgol_filter(spectra, window_length=window, polyorder=polyorder, deriv=2, axis=1)
        return row_snv(derivative)
    if method == "airpls_snv":
        corrected = airpls_correct_matrix(
            spectra, lam=float(cfg["airpls_lambda"]), n_iter=int(cfg["airpls_iterations"])
        )
        return row_snv(corrected)
    raise ValueError(f"Unsupported Raman preprocessing method: {method}")


def fit_pca_scores(
    x: np.ndarray, train_mask: np.ndarray, n_components: int
) -> tuple[np.ndarray, dict[str, Any]]:
    train = x[train_mask]
    center = np.mean(train, axis=0)
    scale = np.std(train, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    train_scaled = (train - center) / scale
    _, singular_values, vt = np.linalg.svd(train_scaled, full_matrices=False)
    components = vt[:n_components]
    all_scaled = (x - center) / scale
    scores = all_scaled @ components.T
    eigenvalues = singular_values**2
    total = float(np.sum(eigenvalues))
    explained = eigenvalues[:n_components] / total if total > 0 else np.zeros(n_components)
    model = {
        "center": center.astype(np.float32),
        "scale": scale.astype(np.float32),
        "components": components.astype(np.float32),
        "explained_variance_ratio": explained.astype(np.float64),
        "n_train_samples": int(train.shape[0]),
    }
    return scores.astype(np.float32), model


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 3:
        return math.nan
    aa = a[finite]
    bb = b[finite]
    if np.std(aa) <= 0 or np.std(bb) <= 0:
        return math.nan
    return float(np.corrcoef(aa, bb)[0, 1])


def quality_metrics(
    method: str,
    x: np.ndarray,
    train_mask: np.ndarray,
    wavelengths: np.ndarray,
    sample_time_h: np.ndarray,
    pca_scores: np.ndarray,
    pca_model: dict[str, Any],
) -> list[dict[str, Any]]:
    train = x[train_mask]
    wave = wavelengths.astype(np.float64)
    wave_centered = wave - np.mean(wave)
    denom = float(np.sum(wave_centered**2)) if wave_centered.size else 1.0
    slopes = ((train - np.mean(train, axis=1, keepdims=True)) @ wave_centered) / max(denom, EPS)
    low_window = odd_window(401, train.shape[1])
    low_freq = savgol_filter(train, window_length=low_window, polyorder=3, axis=1)
    low_ratio = np.sum(low_freq**2, axis=1) / np.maximum(np.sum(train**2, axis=1), EPS)
    row_std = np.std(train, axis=1)
    pc1_corr = abs(safe_corr(pca_scores[train_mask, 0], sample_time_h[train_mask]))
    pc1_var = float(pca_model["explained_variance_ratio"][0])
    metrics = [
        ("median_low_frequency_energy_ratio", float(np.median(low_ratio)), float(np.percentile(low_ratio, 25)), float(np.percentile(low_ratio, 75)), "unitless", "lower"),
        ("median_abs_linear_slope", float(np.median(np.abs(slopes))), float(np.percentile(np.abs(slopes), 25)), float(np.percentile(np.abs(slopes), 75)), "intensity_per_cm-1", "lower"),
        ("median_row_standard_deviation", float(np.median(row_std)), float(np.percentile(row_std, 25)), float(np.percentile(row_std, 75)), "preprocessed_intensity", "context"),
        ("pc1_explained_variance_ratio", pc1_var, math.nan, math.nan, "unitless", "context"),
        ("abs_pc1_time_correlation", pc1_corr, math.nan, math.nan, "unitless", "context"),
    ]
    rows = []
    for name, value, q25, q75, unit, direction in metrics:
        rows.append(
            {
                "phase": PHASE,
                "experiment_name": EXPERIMENT,
                "split": "train_normal",
                "model": "raman_preprocessing",
                "method": method,
                "metric_name": name,
                "metric_value": value,
                "q25": q25,
                "q75": q75,
                "y_error_lower": value - q25 if np.isfinite(q25) else math.nan,
                "y_error_upper": q75 - value if np.isfinite(q75) else math.nan,
                "unit": unit,
                "preferred_direction": direction,
            }
        )
    return rows


def build_sample_metadata(
    processed_root: Path, split_info: dict[int, dict[str, Any]]
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    rows = read_csv_dicts(processed_root / "raman" / "raman_sample_rows.csv")
    metadata: list[dict[str, Any]] = []
    split_names = []
    batch_ids = []
    times = []
    max_time_alignment_error_h = 0.0
    missing_offline_at_raman: dict[str, int] = {}
    first_batch = load_batch(processed_root, 1)
    process_columns = [str(col) for col in first_batch["process_columns"]]
    batch_cache: dict[int, dict[str, Any]] = {}
    offline_cols = [
        "Offline Penicillin concentration(P_offline:P(g L^{-1}))",
        "Offline Biomass concentratio(X_offline:X(g L^{-1}))",
        "PAA concentration offline(PAA_offline:PAA (g L^{-1}))",
        "NH_3 concentration off-line(NH3_offline:NH3 (g L^{-1}))",
        "Viscosity(Viscosity_offline:centPoise)",
    ]
    for col in offline_cols:
        missing_offline_at_raman[col] = 0
    for row in rows:
        sample_index = int(float(row["sample_index"]))
        batch_id = int(float(row["batch_id"]))
        time_h = float(row["time_h"])
        if batch_id not in batch_cache:
            batch_cache[batch_id] = load_batch(processed_root, batch_id)
        batch = batch_cache[batch_id]
        batch_time = batch["time_h"].astype(np.float64)
        source_t = int(np.nanargmin(np.abs(batch_time - time_h)))
        alignment_error = float(abs(batch_time[source_t] - time_h))
        max_time_alignment_error_h = max(max_time_alignment_error_h, alignment_error)
        process = batch["process"].astype(np.float64)
        for col in offline_cols:
            value = process[source_t, process_columns.index(col)]
            if not np.isfinite(value):
                missing_offline_at_raman[col] += 1
        split = split_info[batch_id]["split"]
        metadata.append(
            {
                "sample_index": sample_index,
                "batch_id": batch_id,
                "split": split,
                "fault_label": split_info[batch_id]["fault_label"],
                "time_h": time_h,
                "source_t": source_t,
                "alignment_error_h": alignment_error,
            }
        )
        split_names.append(split)
        batch_ids.append(batch_id)
        times.append(time_h)
    meta_summary = {
        "n_raman_samples": len(metadata),
        "max_time_alignment_error_h": max_time_alignment_error_h,
        "offline_nonmissing_at_raman": {
            col: len(metadata) - missing for col, missing in missing_offline_at_raman.items()
        },
        "offline_missing_at_raman": missing_offline_at_raman,
    }
    return (
        metadata,
        np.asarray(split_names, dtype=object),
        np.asarray(batch_ids, dtype=np.int32),
        np.asarray(times, dtype=np.float64),
        meta_summary,
    )


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    x_aug = np.hstack([np.ones((x.shape[0], 1), dtype=np.float64), x])
    penalty = np.eye(x_aug.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    lhs = x_aug.T @ x_aug + alpha * penalty
    rhs = x_aug.T @ y
    return np.linalg.solve(lhs, rhs)


def predict_ridge(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    x_aug = np.hstack([np.ones((x.shape[0], 1), dtype=np.float64), x])
    return x_aug @ coef


def metric_values(y_true: np.ndarray, y_pred: np.ndarray, train_scales: np.ndarray) -> dict[str, np.ndarray]:
    err = y_pred - y_true
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err**2, axis=0))
    denom = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    r2 = 1.0 - np.sum(err**2, axis=0) / np.where(denom == 0, np.nan, denom)
    nrmse = rmse / train_scales
    return {"mae": mae, "rmse": rmse, "r2": r2, "nrmse_train_std": nrmse}


def mean_nrmse(y_true_scaled: np.ndarray, y_pred_scaled: np.ndarray) -> float:
    return float(np.mean(np.sqrt(np.mean((y_pred_scaled - y_true_scaled) ** 2, axis=0))))


def append_metric_rows(
    rows: list[dict[str, Any]],
    *,
    split: str,
    model: str,
    preprocessing_method: str,
    horizon_steps: int,
    horizon_h: float,
    alpha: float,
    pca_components: int,
    target_columns: list[str],
    metrics: dict[str, np.ndarray],
    n_samples: int,
) -> None:
    for idx, target in enumerate(target_columns):
        for metric_name, values in metrics.items():
            rows.append(
                {
                    "phase": PHASE,
                    "experiment_name": EXPERIMENT,
                    "split": split,
                    "model": model,
                    "method": preprocessing_method,
                    "horizon_steps": horizon_steps,
                    "horizon_h": horizon_h,
                    "target": target,
                    "metric_name": metric_name,
                    "metric_value": float(values[idx]),
                    "unit": "target_unit" if metric_name in {"mae", "rmse"} else "unitless",
                    "alpha": alpha,
                    "pca_components": pca_components,
                    "n_samples": n_samples,
                    "q25": math.nan,
                    "q75": math.nan,
                    "y_error_lower": math.nan,
                    "y_error_upper": math.nan,
                }
            )
    aggregate_metrics = {
        "mean_nrmse_train_std": float(np.mean(metrics["nrmse_train_std"])),
        "median_nrmse_train_std": float(np.median(metrics["nrmse_train_std"])),
        "mean_r2": float(np.nanmean(metrics["r2"])),
        "median_r2": float(np.nanmedian(metrics["r2"])),
    }
    nrmse_q25, nrmse_q75 = np.percentile(metrics["nrmse_train_std"], [25, 75])
    r2_q25, r2_q75 = np.nanpercentile(metrics["r2"], [25, 75])
    for metric_name, value in aggregate_metrics.items():
        q25 = nrmse_q25 if "nrmse" in metric_name else r2_q25
        q75 = nrmse_q75 if "nrmse" in metric_name else r2_q75
        rows.append(
            {
                "phase": PHASE,
                "experiment_name": EXPERIMENT,
                "split": split,
                "model": model,
                "method": preprocessing_method,
                "horizon_steps": horizon_steps,
                "horizon_h": horizon_h,
                "target": "ALL",
                "metric_name": metric_name,
                "metric_value": value,
                "unit": "unitless",
                "alpha": alpha,
                "pca_components": pca_components,
                "n_samples": n_samples,
                "q25": float(q25),
                "q75": float(q75),
                "y_error_lower": float(value - q25),
                "y_error_upper": float(q75 - value),
            }
        )


def build_raman_window_design(
    processed_root: Path,
    metadata: list[dict[str, Any]],
    split_info: dict[int, dict[str, Any]],
    cfg: dict[str, Any],
    variable_roles: dict[str, Any],
    scaler: dict[str, Any],
    pca_scores_by_method: dict[str, np.ndarray],
    process_columns: list[str],
    target_columns: list[str],
) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    exogenous_columns = variable_roles["exogenous_controls"]["columns"]
    history_columns = unique_in_order(exogenous_columns + target_columns)
    process_centers, process_scales = column_stats(
        scaler, process_columns, scale_key=cfg.get("input_scale_key", "robust_scale")
    )
    target_centers, target_scales = column_stats(
        scaler, target_columns, scale_key=cfg.get("target_scale_key", "std")
    )
    history_idx = [process_columns.index(col) for col in history_columns]
    exo_idx = [process_columns.index(col) for col in exogenous_columns]
    target_idx = [process_columns.index(col) for col in target_columns]
    batch_cache: dict[int, dict[str, Any]] = {}
    design_by_horizon: dict[int, dict[str, dict[str, list[Any]]]] = {}
    for horizon_steps in cfg["horizon_steps"]:
        design_by_horizon[int(horizon_steps)] = {
            split: {
                "process_x": [],
                "y_raw": [],
                "y_scaled": [],
                "batch_id": [],
                "source_t": [],
                "target_t": [],
                "target_time_h": [],
                "sample_index": [],
            }
            for split in ["train_normal", "val_normal", "test_normal", "test_fault"]
        }
        for method in pca_scores_by_method:
            for split in design_by_horizon[int(horizon_steps)]:
                design_by_horizon[int(horizon_steps)][split][f"raman_{method}"] = []

    for item in metadata:
        batch_id = int(item["batch_id"])
        split = split_info[batch_id]["split"]
        if split not in {"train_normal", "val_normal", "test_normal", "test_fault"}:
            continue
        if batch_id not in batch_cache:
            batch_cache[batch_id] = load_batch(processed_root, batch_id)
        batch = batch_cache[batch_id]
        process = batch["process"].astype(np.float64)
        time_h = batch["time_h"].astype(np.float64)
        process_scaled = standardize(process, process_centers, process_scales)
        source_t = int(item["source_t"])
        if source_t < int(cfg["history_steps"]) - 1:
            continue
        for horizon_steps in cfg["horizon_steps"]:
            horizon_steps = int(horizon_steps)
            target_t = source_t + horizon_steps
            if target_t >= process.shape[0]:
                continue
            target_raw = process[target_t, target_idx]
            if not np.isfinite(target_raw).all():
                continue
            history = process_scaled[source_t - int(cfg["history_steps"]) + 1 : source_t + 1, history_idx].reshape(-1)
            future_exo = process_scaled[target_t, exo_idx].reshape(-1)
            process_x = np.concatenate([history, future_exo])
            y_scaled = standardize(target_raw, target_centers, target_scales)
            split_store = design_by_horizon[horizon_steps][split]
            split_store["process_x"].append(process_x)
            split_store["y_raw"].append(target_raw)
            split_store["y_scaled"].append(y_scaled)
            split_store["batch_id"].append(batch_id)
            split_store["source_t"].append(source_t)
            split_store["target_t"].append(target_t)
            split_store["target_time_h"].append(float(time_h[target_t]))
            split_store["sample_index"].append(int(item["sample_index"]))
            for method, scores in pca_scores_by_method.items():
                split_store[f"raman_{method}"].append(scores[int(item["sample_index"])])

    out: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for horizon_steps, split_store in design_by_horizon.items():
        out[horizon_steps] = {}
        for split, values in split_store.items():
            out[horizon_steps][split] = {}
            for key, value in values.items():
                if key in {"batch_id", "source_t", "target_t", "sample_index"}:
                    out[horizon_steps][split][key] = np.asarray(value, dtype=np.int32)
                elif key == "target_time_h":
                    out[horizon_steps][split][key] = np.asarray(value, dtype=np.float64)
                else:
                    out[horizon_steps][split][key] = np.asarray(value, dtype=np.float64)
    return out


def run_fusion_experiments(
    design: dict[int, dict[str, dict[str, np.ndarray]]],
    cfg: dict[str, Any],
    target_columns: list[str],
    target_centers: np.ndarray,
    target_scales: np.ndarray,
    median_dt_h: float,
    pca_scores_by_method: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    metric_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    selected: dict[str, Any] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    selection_pool = cfg.get("recommended_chemometric_methods", list(pca_scores_by_method))
    for horizon_steps, split_data in sorted(design.items()):
        horizon_h = horizon_steps * median_dt_h
        train = split_data["train_normal"]
        val = split_data["val_normal"]
        configs = [("process_controlled_ridge", "none")]
        configs.extend(("process_raman_controlled_ridge", method) for method in selection_pool)
        best_for_model: dict[str, tuple[float, float, str]] = {}
        for model, method in configs:
            train_x = train["process_x"]
            val_x = val["process_x"]
            if model == "process_raman_controlled_ridge":
                train_x = np.hstack([train_x, train[f"raman_{method}"]])
                val_x = np.hstack([val_x, val[f"raman_{method}"]])
            for alpha in cfg["ridge_alpha_grid"]:
                coef = fit_ridge(train_x, train["y_scaled"], float(alpha))
                pred_val_scaled = predict_ridge(val_x, coef)
                score = mean_nrmse(val["y_scaled"], pred_val_scaled)
                selection_rows.append(
                    {
                        "phase": PHASE,
                        "experiment_name": EXPERIMENT,
                        "horizon_steps": horizon_steps,
                        "horizon_h": horizon_h,
                        "model": model,
                        "method": method,
                        "alpha": float(alpha),
                        "selection_split": cfg["selection_split"],
                        "selection_metric": cfg["selection_metric"],
                        "metric_value": score,
                        "pca_components": 0 if method == "none" else int(cfg["pca_components"]),
                        "test_fault_usage": "not used for selection",
                    }
                )
                current = best_for_model.get(model)
                if current is None or score < current[0]:
                    best_for_model[model] = (score, float(alpha), method)

        for model, (_, alpha, method) in best_for_model.items():
            train_x = train["process_x"]
            if model == "process_raman_controlled_ridge":
                train_x = np.hstack([train_x, train[f"raman_{method}"]])
            coef = fit_ridge(train_x, train["y_scaled"], alpha)
            selected[f"{model}_h{horizon_steps}"] = {
                "model": model,
                "method": method,
                "alpha": alpha,
                "selection_split": cfg["selection_split"],
                "selection_metric": cfg["selection_metric"],
                "selection_value": best_for_model[model][0],
                "horizon_steps": horizon_steps,
                "horizon_h": horizon_h,
            }
            for split, data in split_data.items():
                x = data["process_x"]
                if model == "process_raman_controlled_ridge":
                    x = np.hstack([x, data[f"raman_{method}"]])
                pred_scaled = predict_ridge(x, coef)
                pred_raw = inverse_standardize(pred_scaled, target_centers, target_scales)
                metrics = metric_values(data["y_raw"], pred_raw, target_scales)
                append_metric_rows(
                    metric_rows,
                    split=split,
                    model=model,
                    preprocessing_method=method,
                    horizon_steps=horizon_steps,
                    horizon_h=horizon_h,
                    alpha=alpha,
                    pca_components=0 if method == "none" else int(cfg["pca_components"]),
                    target_columns=target_columns,
                    metrics=metrics,
                    n_samples=int(data["y_raw"].shape[0]),
                )
                key = f"{model}_h{horizon_steps}_{split}"
                predictions[key] = {
                    "y_true": data["y_raw"].astype(np.float32),
                    "y_pred": pred_raw.astype(np.float32),
                    "batch_id": data["batch_id"],
                    "target_time_h": data["target_time_h"],
                    "sample_index": data["sample_index"],
                    "target_columns": np.asarray(target_columns, dtype=object),
                    "method": np.asarray([method], dtype=object),
                }
    return metric_rows, selection_rows, selected, predictions


def metric_lookup(rows: list[dict[str, Any]], split: str, model: str, horizon: int, metric: str, target: str = "ALL") -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["split"] == split
        and row["model"] == model
        and int(row["horizon_steps"]) == int(horizon)
        and row["metric_name"] == metric
        and row["target"] == target
    ]
    if not matches:
        raise KeyError((split, model, horizon, metric, target))
    return matches[0]


def figure_preprocessing_example(
    dirs: dict[str, Path],
    wavelengths: np.ndarray,
    spectra: np.ndarray,
    sample_metadata: list[dict[str, Any]],
    example_values: dict[str, np.ndarray],
    cfg: dict[str, Any],
) -> dict[str, str]:
    figure_id = "Fig16_raman_preprocessing_example"
    train_normal = [item for item in sample_metadata if item["split"] == "train_normal"]
    example = max(train_normal, key=lambda item: float(item["time_h"]))
    idx = int(example["sample_index"])
    rows = []
    corrected_raw = airpls_correct_matrix(
        spectra[idx : idx + 1], lam=float(cfg["airpls_lambda"]), n_iter=int(cfg["airpls_iterations"])
    )[0]
    estimated_baseline = spectra[idx] - corrected_raw
    methods = ["raw", "estimated_baseline", "snv", "sg1_snv", "sg2_snv", "airpls_snv"]
    labels = {
        "raw": "Raw",
        "estimated_baseline": "Estimated Baseline",
        "snv": "SNV",
        "sg1_snv": "SG First Derivative + SNV",
        "sg2_snv": "SG Second Derivative + SNV",
        "airpls_snv": "airPLS Corrected + SNV",
    }
    for method in methods:
        if method == "raw":
            values = spectra[idx]
        elif method == "estimated_baseline":
            values = estimated_baseline
        else:
            values = example_values[method][idx]
        for wave, value in zip(wavelengths, values):
            rows.append(
                {
                    "figure_id": figure_id,
                    "phase": PHASE,
                    "experiment_name": EXPERIMENT,
                    "split": example["split"],
                    "batch_id": example["batch_id"],
                    "time_h": example["time_h"],
                    "model": "raman_preprocessing",
                    "method": labels[method],
                    "x_value": float(wave),
                    "y_value": float(value),
                    "metric_name": "preprocessed_intensity",
                    "unit": "a.u.",
                }
            )
    csv_path = dirs["tables"] / f"{figure_id}_data.csv"
    write_csv(csv_path, rows)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True)
    axes[0].plot(wavelengths, spectra[idx], color="#595959", linewidth=1.0, label="Raw")
    axes[0].plot(wavelengths, estimated_baseline, color="#D55E00", linewidth=1.0, label="Estimated Baseline")
    axes[0].set_ylabel("Intensity")
    axes[0].set_title("Raw Spectrum and Baseline-Corrected Representation", loc="left")
    axes[0].legend(loc="best")
    for method, color in [("snv", "#0072B2"), ("sg1_snv", "#009E73"), ("sg2_snv", "#CC79A7"), ("airpls_snv", "#D55E00")]:
        axes[1].plot(wavelengths, example_values[method][idx], linewidth=0.9, label=labels[method], color=color)
    axes[1].set_xlabel("Raman Shift (cm$^{-1}$)")
    axes[1].set_ylabel("Normalized Intensity")
    axes[1].set_title("Chemometric Preprocessing Alternatives", loc="left")
    axes[1].legend(loc="best")
    for ax in axes:
        ax.invert_xaxis()
        ax.grid(alpha=0.25, linewidth=0.5)
    fig.suptitle("Raman Preprocessing Example", y=0.995)
    fig.tight_layout()
    fig_path = dirs["figures"] / f"{figure_id}.png"
    fig.savefig(fig_path, dpi=400)
    plt.close(fig)
    explanation_path = dirs["explanations"] / f"{figure_id}_explanation.md"
    save_explanation(
        explanation_path,
        "Raman Preprocessing Example ",
        figure_id,
        " Raman ， Phase 08  SNV。",
        " Phase 04  Raman spectra。 train-normal  Raman ， test-fault 。",
        " raw spectrum  airPLS  baseline； SNV、SG  + SNV、SG  + SNV、airPLS  + SNV。",
        "，。",
        "，Raman ；SG derivative  airPLS 。",
        "， PCA  ridge  train-normal 、。",
        "，； Fig17  Fig18。",
        " Raman preprocessing ，。",
        "python3 scripts/13_run_phase08_raman_multimodal.py",
    )
    return {"figure_id": figure_id, "figure_path": str(fig_path), "csv_path": str(csv_path), "explanation_path": str(explanation_path)}


def figure_preprocessing_quality(dirs: dict[str, Path], quality_rows: list[dict[str, Any]]) -> dict[str, str]:
    figure_id = "Fig17_raman_preprocessing_quality_metrics"
    metrics_to_plot = [
        "median_low_frequency_energy_ratio",
        "pc1_explained_variance_ratio",
        "abs_pc1_time_correlation",
    ]
    csv_rows = [{"figure_id": figure_id, **row, "x_value": row["method"], "y_value": row["metric_value"]} for row in quality_rows]
    csv_path = dirs["tables"] / f"{figure_id}_data.csv"
    write_csv(csv_path, csv_rows)
    methods = ["raw", "snv", "sg1_snv", "sg2_snv", "airpls_snv"]
    labels = ["Raw", "SNV", "SG1+SNV", "SG2+SNV", "airPLS+SNV"]
    colors = ["#595959", "#0072B2", "#009E73", "#CC79A7", "#D55E00"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=False)
    titles = ["Low-Frequency Energy", "PC1 Explained Variance", "|PC1-Time Correlation|"]
    for ax, metric_name, title in zip(axes, metrics_to_plot, titles):
        values = []
        for method in methods:
            row = next(item for item in csv_rows if item["method"] == method and item["metric_name"] == metric_name)
            values.append(float(row["metric_value"]))
        ax.bar(labels, values, color=colors)
        ax.set_title(title)
        ax.set_ylabel("Metric Value")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    fig.suptitle("Raman Preprocessing Quality Metrics", y=0.995)
    fig.tight_layout()
    fig_path = dirs["figures"] / f"{figure_id}.png"
    fig.savefig(fig_path, dpi=400)
    plt.close(fig)
    explanation_path = dirs["explanations"] / f"{figure_id}_explanation.md"
    best_low = min(
        (row for row in quality_rows if row["metric_name"] == "median_low_frequency_energy_ratio"),
        key=lambda row: float(row["metric_value"]),
    )
    save_explanation(
        explanation_path,
        "Raman Preprocessing Quality Metrics ",
        figure_id,
        " Raman 。",
        " train-normal Raman 。PCA 、 train-normal 。",
        "、、。，CSV  IQR 。",
        f" {best_low['method']}，。",
        " PC1 ，。",
        "PC1 ，； raw/SNV  PC1 ，。",
        "，； Fig18 。",
        " Raman preprocessing ， SG derivative  airPLS 。",
        "python3 scripts/13_run_phase08_raman_multimodal.py",
    )
    return {"figure_id": figure_id, "figure_path": str(fig_path), "csv_path": str(csv_path), "explanation_path": str(explanation_path)}


def figure_fusion_forecasting(
    dirs: dict[str, Path], metric_rows: list[dict[str, Any]], selected: dict[str, Any]
) -> dict[str, str]:
    figure_id = "Fig18_raman_fusion_forecasting_performance"
    models = ["process_controlled_ridge", "process_raman_controlled_ridge"]
    model_labels = ["Process", "Process + Raman"]
    horizons = sorted({int(row["horizon_steps"]) for row in metric_rows})
    csv_rows = []
    for horizon in horizons:
        for model in models:
            nrmse = metric_lookup(metric_rows, "test_normal", model, horizon, "median_nrmse_train_std")
            r2 = metric_lookup(metric_rows, "test_normal", model, horizon, "median_r2")
            csv_rows.append(
                {
                    "figure_id": figure_id,
                    "phase": PHASE,
                    "experiment_name": EXPERIMENT,
                    "split": "test_normal",
                    "model": model,
                    "method": nrmse["method"],
                    "horizon_steps": horizon,
                    "horizon_h": nrmse["horizon_h"],
                    "metric_name": "median_nrmse_and_r2",
                    "x_value": horizon,
                    "y_value": float(nrmse["metric_value"]),
                    "median_nrmse_train_std": float(nrmse["metric_value"]),
                    "nrmse_q25": nrmse.get("q25", math.nan),
                    "nrmse_q75": nrmse.get("q75", math.nan),
                    "y_error_lower": nrmse.get("y_error_lower", math.nan),
                    "y_error_upper": nrmse.get("y_error_upper", math.nan),
                    "median_r2": float(r2["metric_value"]),
                    "r2_q25": r2.get("q25", math.nan),
                    "r2_q75": r2.get("q75", math.nan),
                }
            )
    csv_path = dirs["tables"] / f"{figure_id}_data.csv"
    write_csv(csv_path, csv_rows)
    x = np.arange(len(horizons))
    width = 0.34
    colors = ["#595959", "#D55E00"]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    for i, model in enumerate(models):
        sub = [row for row in csv_rows if row["model"] == model]
        values = [float(row["median_nrmse_train_std"]) for row in sub]
        axes[0].bar(
            x + (i - 0.5) * width,
            values,
            width,
            label=model_labels[i],
            color=colors[i],
        )
        axes[1].bar(
            x + (i - 0.5) * width,
            [float(row["median_r2"]) for row in sub],
            width,
            label=model_labels[i],
            color=colors[i],
        )
    axes[0].set_xticks(x, [f"{h} steps" for h in horizons])
    axes[0].set_ylabel("Median nRMSE")
    axes[0].set_title("Test-Normal Forecast Error")
    axes[1].set_xticks(x, [f"{h} steps" for h in horizons])
    axes[1].set_ylabel("Median R$^2$")
    axes[1].set_title("Test-Normal Forecast R$^2$")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.legend(loc="best")
    fig.suptitle("Raman Fusion Controlled Forecasting Performance", y=0.995)
    fig.tight_layout()
    fig_path = dirs["figures"] / f"{figure_id}.png"
    fig.savefig(fig_path, dpi=400)
    plt.close(fig)
    h20_process = metric_lookup(metric_rows, "test_normal", "process_controlled_ridge", max(horizons), "median_nrmse_train_std")
    h20_fusion = metric_lookup(metric_rows, "test_normal", "process_raman_controlled_ridge", max(horizons), "median_nrmse_train_std")
    delta = (float(h20_process["metric_value"]) - float(h20_fusion["metric_value"])) / max(float(h20_process["metric_value"]), EPS) * 100.0
    if delta >= 0:
        result_sentence = f" horizon  Raman  median nRMSE  {delta:.2f}%。"
        interpretation_sentence = " PCA  Raman  horizon 。"
    else:
        result_sentence = f" horizon  Raman  median nRMSE  {abs(delta):.2f}%，。"
        interpretation_sentence = " Raman ， PCA ；， Raman  process baseline。"
    selected_methods = ", ".join(
        f"h{h}: {selected[f'process_raman_controlled_ridge_h{h}']['method']}" for h in horizons
    )
    explanation_path = dirs["explanations"] / f"{figure_id}_explanation.md"
    save_explanation(
        explanation_path,
        "Raman Fusion Controlled Forecasting Performance ",
        figure_id,
        " Raman ，process-only controlled ridge  process+Raman controlled ridge 。",
        " Phase 04 process/Raman 。Raman PCA  train-normal ；ridge alpha  Raman  val-normal 。",
        " test-normal  horizon  median nRMSE； median R2。，CSV  IQR 。",
        f"val-normal  Raman ：{selected_methods}。{result_sentence}",
        interpretation_sentence,
        " ridge ， FermNFTP； gated/attention multimodal fusion， Raman PCA 。",
        "Raman ， Phase05/06 。",
        " Raman fusion ，。",
        "python3 scripts/13_run_phase08_raman_multimodal.py",
    )
    return {"figure_id": figure_id, "figure_path": str(fig_path), "csv_path": str(csv_path), "explanation_path": str(explanation_path)}


def short_target_name(target: str) -> str:
    replacements = {
        "Substrate concentration(S:g/L)": "Substrate",
        "Dissolved oxygen concentration(DO2:mg/L)": "DO",
        "Penicillin concentration(P:g/L)": "Penicillin",
        "Vessel Volume(V:L)": "Volume",
        "Vessel Weight(Wt:Kg)": "Weight",
        "pH(pH:pH)": "pH",
        "Temperature(T:K)": "Temperature",
        "Generated heat(Q:kJ)": "Heat",
        "carbon dioxide percent in off-gas(CO2outgas:%)": "CO2 Off-Gas",
        "Oxygen Uptake Rate(OUR:(g min^{-1}))": "OUR",
        "Oxygen in percent in off-gas(O2:O2  (%))": "O2 Off-Gas",
        "Carbon evolution rate(CER:g/h)": "CER",
    }
    return replacements.get(target, target.split("(")[0].strip())


def figure_target_level_delta(
    dirs: dict[str, Path], metric_rows: list[dict[str, Any]], selected: dict[str, Any]
) -> dict[str, str]:
    figure_id = "Fig19_raman_fusion_target_level_delta"
    horizons = sorted({int(row["horizon_steps"]) for row in metric_rows})
    horizon = max(horizons)
    target_rows = []
    targets = sorted(
        {
            row["target"]
            for row in metric_rows
            if row["split"] == "test_normal" and row["target"] != "ALL" and row["metric_name"] == "nrmse_train_std" and int(row["horizon_steps"]) == horizon
        }
    )
    for target in targets:
        process = metric_lookup(metric_rows, "test_normal", "process_controlled_ridge", horizon, "nrmse_train_std", target)
        fusion = metric_lookup(metric_rows, "test_normal", "process_raman_controlled_ridge", horizon, "nrmse_train_std", target)
        process_r2 = metric_lookup(metric_rows, "test_normal", "process_controlled_ridge", horizon, "r2", target)
        fusion_r2 = metric_lookup(metric_rows, "test_normal", "process_raman_controlled_ridge", horizon, "r2", target)
        delta = (float(process["metric_value"]) - float(fusion["metric_value"])) / max(float(process["metric_value"]), EPS) * 100.0
        target_rows.append(
            {
                "figure_id": figure_id,
                "phase": PHASE,
                "experiment_name": EXPERIMENT,
                "split": "test_normal",
                "model": "process_raman_controlled_ridge",
                "method": fusion["method"],
                "horizon_steps": horizon,
                "horizon_h": fusion["horizon_h"],
                "target": target,
                "target_short": short_target_name(target),
                "metric_name": "nrmse_reduction_percent",
                "x_value": short_target_name(target),
                "y_value": delta,
                "process_nrmse": float(process["metric_value"]),
                "fusion_nrmse": float(fusion["metric_value"]),
                "process_r2": float(process_r2["metric_value"]),
                "fusion_r2": float(fusion_r2["metric_value"]),
                "r2_delta": float(fusion_r2["metric_value"]) - float(process_r2["metric_value"]),
            }
        )
    target_rows.sort(key=lambda row: row["y_value"])
    csv_path = dirs["tables"] / f"{figure_id}_data.csv"
    write_csv(csv_path, target_rows)
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    colors = ["#D55E00" if row["y_value"] < 0 else "#009E73" for row in target_rows]
    ax.barh([row["target_short"] for row in target_rows], [row["y_value"] for row in target_rows], color=colors)
    ax.axvline(0, color="#111111", linewidth=1.0)
    ax.set_xlabel("nRMSE Reduction from Raman Fusion (%)")
    ax.set_ylabel("Target")
    ax.set_title(f"Target-Level Raman Fusion Effect at {horizon} Steps")
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig_path = dirs["figures"] / f"{figure_id}.png"
    fig.savefig(fig_path, dpi=400)
    plt.close(fig)
    improved = sum(1 for row in target_rows if row["y_value"] > 0)
    method = selected[f"process_raman_controlled_ridge_h{horizon}"]["method"]
    explanation_path = dirs["explanations"] / f"{figure_id}_explanation.md"
    save_explanation(
        explanation_path,
        "Target-Level Raman Fusion Effect ",
        figure_id,
        " horizon ，Raman  nRMSE 。",
        " test-normal Raman 。 process-only controlled ridge  nRMSE reduction percentage。",
        " Raman  nRMSE， Raman  nRMSE。CSV  process/fusion nRMSE、R2  R2 delta。",
        f" {horizon} step horizon ，val-normal  Raman  {method}， {improved}/{len(target_rows)}  test-normal  nRMSE 。",
        " Raman ，。",
        "， Raman ；、。",
        " ridge ，。",
        "， Raman 。",
        "python3 scripts/13_run_phase08_raman_multimodal.py",
    )
    return {"figure_id": figure_id, "figure_path": str(fig_path), "csv_path": str(csv_path), "explanation_path": str(explanation_path)}


def update_loss_sources(project_root: Path, cfg: dict[str, Any], manifest_path: Path) -> None:
    path = project_root / "data" / "processed" / "stats" / "loss_weight_sources.json"
    sources = load_json(path) if path.exists() else {"schema_version": "phase08_v1"}
    sources["phase08_raman_multimodal"] = {
        "raman_preprocessing_selection_source": "val_normal selection among predefined chemometric methods",
        "pca_source": "train_normal Raman spectra only",
        "pca_center_scale_source": "train_normal Raman spectra only",
        "ridge_alpha_source": "val_normal grid selection",
        "target_scale_source": str(project_root / cfg["train_normal_scalers"]),
        "test_fault_usage": "evaluation only; not used for preprocessing, PCA, alpha, or threshold selection",
        "manifest": str(manifest_path),
    }
    path.write_text(json.dumps(sources, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 08 Raman preprocessing and fusion")
    parser.add_argument("--config", default="configs/model/phase08_raman_multimodal.json")
    parser.add_argument("--asset-root", default="outputs/paper_assets")
    args = parser.parse_args()

    project_root = Path.cwd()
    cfg = load_json(project_root / args.config)
    processed_root = project_root / cfg["processed_root"]
    output_root = project_root / cfg["output_root"]
    dirs = ensure_dirs(output_root)
    asset_dirs = ensure_asset_dirs(project_root / args.asset_root)
    split_info = split_map(project_root / cfg["split_csv"])
    variable_roles = load_json(project_root / cfg["variable_roles"])
    scaler = load_json(project_root / cfg["train_normal_scalers"])
    first_batch = load_batch(processed_root, 1)
    process_columns = [str(col) for col in first_batch["process_columns"]]
    target_columns = select_dense_targets(variable_roles, scaler, float(cfg["target_missing_ratio_max"]))
    target_centers, target_scales = column_stats(
        scaler, target_columns, scale_key=cfg.get("target_scale_key", "std")
    )
    time_h = first_batch["time_h"].astype(np.float64)
    median_dt_h = float(np.median(np.diff(time_h)[np.diff(time_h) > 0]))

    spectra = np.load(processed_root / "raman" / "raman_sample_spectra.npy").astype(np.float64)
    wavelengths = np.load(processed_root / "raman" / "raman_wavelengths.npy").astype(np.float64)
    sample_metadata, split_names, _batch_ids, sample_time_h, sample_summary = build_sample_metadata(
        processed_root, split_info
    )
    train_mask = split_names == "train_normal"

    pca_scores_by_method: dict[str, np.ndarray] = {}
    pca_models: dict[str, dict[str, Any]] = {}
    quality_rows: list[dict[str, Any]] = []
    example_values: dict[str, np.ndarray] = {}
    for method in cfg["raman_preprocessing_methods"]:
        preprocessed = preprocess_spectra(spectra, method, cfg)
        if method != "raw":
            example_values[method] = preprocessed.astype(np.float32)
        scores, model = fit_pca_scores(preprocessed, train_mask, int(cfg["pca_components"]))
        pca_scores_by_method[method] = scores
        pca_models[method] = model
        quality_rows.extend(
            quality_metrics(method, preprocessed, train_mask, wavelengths, sample_time_h, scores, model)
        )

    feature_path = dirs["features"] / "raman_pca_scores.npz"
    np.savez_compressed(feature_path, **{f"{method}_scores": scores for method, scores in pca_scores_by_method.items()})
    model_path = dirs["models"] / "raman_pca_models.npz"
    np.savez_compressed(
        model_path,
        **{
            f"{method}_{key}": value
            for method, model in pca_models.items()
            for key, value in model.items()
            if isinstance(value, np.ndarray)
        },
    )
    pca_meta_path = dirs["models"] / "raman_pca_model_metadata.json"
    pca_meta = {
        method: {
            "n_train_samples": int(model["n_train_samples"]),
            "pca_components": int(cfg["pca_components"]),
            "explained_variance_ratio": model["explained_variance_ratio"].tolist(),
        }
        for method, model in pca_models.items()
    }
    pca_meta_path.write_text(json.dumps(pca_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    quality_path = dirs["metrics"] / "raman_preprocessing_quality_metrics.csv"
    write_csv(quality_path, quality_rows)
    sample_meta_path = dirs["metrics"] / "raman_sample_alignment_summary.json"
    sample_meta_path.write_text(json.dumps(sample_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    design = build_raman_window_design(
        processed_root,
        sample_metadata,
        split_info,
        cfg,
        variable_roles,
        scaler,
        pca_scores_by_method,
        process_columns,
        target_columns,
    )
    metric_rows, selection_rows, selected, predictions = run_fusion_experiments(
        design,
        cfg,
        target_columns,
        target_centers,
        target_scales,
        median_dt_h,
        pca_scores_by_method,
    )
    metrics_path = dirs["metrics"] / "raman_fusion_forecasting_metrics.csv"
    selection_path = dirs["metrics"] / "raman_fusion_alpha_method_selection.csv"
    write_csv(metrics_path, metric_rows)
    write_csv(selection_path, selection_rows)
    prediction_path = dirs["predictions"] / "raman_fusion_predictions.npz"
    np.savez_compressed(
        prediction_path,
        **{
            f"{key}_{subkey}": value
            for key, pred in predictions.items()
            for subkey, value in pred.items()
            if isinstance(value, np.ndarray)
        },
    )

    assets = [
        figure_preprocessing_example(asset_dirs, wavelengths, spectra, sample_metadata, example_values, cfg),
        figure_preprocessing_quality(asset_dirs, quality_rows),
        figure_fusion_forecasting(asset_dirs, metric_rows, selected),
        figure_target_level_delta(asset_dirs, metric_rows, selected),
    ]

    manifest = {
        "phase": "08_raman_multimodal",
        "experiment_name": EXPERIMENT,
        "generation_script": "scripts/13_run_phase08_raman_multimodal.py",
        "config": str(project_root / args.config),
        "source_processed_root": str(processed_root),
        "source_split_csv": str(project_root / cfg["split_csv"]),
        "sample_alignment_summary": sample_summary,
        "target_columns": target_columns,
        "raman_preprocessing_methods": cfg["raman_preprocessing_methods"],
        "recommended_chemometric_methods": cfg["recommended_chemometric_methods"],
        "pca_components": int(cfg["pca_components"]),
        "pca_feature_npz": str(feature_path),
        "pca_model_npz": str(model_path),
        "pca_metadata_json": str(pca_meta_path),
        "quality_metrics_csv": str(quality_path),
        "fusion_metrics_csv": str(metrics_path),
        "selection_csv": str(selection_path),
        "prediction_npz": str(prediction_path),
        "selected_models": selected,
        "assets": assets,
        "test_fault_usage": "final evaluation only",
    }
    manifest_path = asset_dirs["manifests"] / "phase_08_paper_assets.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    update_loss_sources(project_root, cfg, manifest_path)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run Phase 16 strong forecasting and anomaly-detection baselines.

The script performs real model fitting and validation-based model selection.
It never synthesizes metrics or training curves. XGBoost is attempted when the
package is available; otherwise a HistGradientBoosting fallback is recorded
explicitly in the output tables and manifests.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fermnftp.data import (  # noqa: E402
    build_forecasting_metadata,
    inverse_standardize,
    load_batch_npz,
    load_json,
    standardize,
)
from fermnftp.metrics import metric_rows, regression_metrics, write_csv  # noqa: E402
from fermnftp.plot_style import apply_ai_conference_style, polish_axis  # noqa: E402

apply_ai_conference_style(plt)


PHASE = "Phase 16"
EXPERIMENT = "Additional strong baselines and anomaly detection"
SPLITS = ["train_normal", "val_normal", "test_normal", "test_fault"]


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_union_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    write_csv(path, rows, fieldnames=fieldnames)


def ensure_dirs(output_root: Path, paper_asset_root: Path) -> dict[str, Path]:
    dirs = {
        "output": output_root,
        "metrics": output_root / "metrics",
        "predictions": output_root / "predictions",
        "models": output_root / "models",
        "figures": paper_asset_root / "figures",
        "tables": paper_asset_root / "tables",
        "explanations": paper_asset_root / "explanations",
        "manifests": paper_asset_root / "manifests",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def save_explanation(
    path: Path,
    *,
    title: str,
    figure_id: str,
    purpose: str,
    data_source: str,
    design: str,
    results: str,
    interpretation: str,
    discussion: str,
    limitations: str,
    caption: str,
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

、p 、。、、， CSV 。

## 

{caption}

## 

```bash
{command}
```
"""
    path.write_text(text, encoding="utf-8")


def import_sklearn() -> dict[str, Any]:
    try:
        from sklearn.decomposition import PCA
        from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest, RandomForestRegressor
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.svm import OneClassSVM
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Phase16 requires scikit-learn for RF, GP, PCA, OCSVM, and IsolationForest baselines."
        ) from exc
    return {
        "PCA": PCA,
        "RandomForestRegressor": RandomForestRegressor,
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
        "GaussianProcessRegressor": GaussianProcessRegressor,
        "ConstantKernel": ConstantKernel,
        "RBF": RBF,
        "WhiteKernel": WhiteKernel,
        "MultiOutputRegressor": MultiOutputRegressor,
        "OneClassSVM": OneClassSVM,
        "IsolationForest": IsolationForest,
    }


def import_torch() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except ModuleNotFoundError as exc:
        raise SystemExit("Phase16 PatchTST and AE baselines require PyTorch.") from exc
    return torch, nn, DataLoader, Dataset


def set_seed(seed: int, torch: Any | None = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def median_dt_h(metadata: dict[str, Any]) -> float:
    first_batch = metadata["splits"]["train_normal"][0]
    time_h = load_batch_npz(metadata["processed_root"], first_batch)["time_h"].astype(np.float64)
    diffs = np.diff(time_h)
    finite = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(np.median(finite)) if finite.size else 0.2


def deterministic_cap_indices(n: int, cap: int | None) -> np.ndarray:
    idx = np.arange(n, dtype=np.int64)
    if cap is None or n <= cap:
        return idx
    step = max(1, n // cap)
    return idx[::step][:cap]


def mean_nrmse_scaled(y_true_scaled: np.ndarray, y_pred_scaled: np.ndarray) -> float:
    return float(np.mean(np.sqrt(np.mean((y_pred_scaled - y_true_scaled) ** 2, axis=0))))


def finite_or_raise(array: np.ndarray, context: str) -> None:
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{context}: non-finite values detected")


def build_window_design(
    *,
    metadata: dict[str, Any],
    split: str,
    history_steps: int,
    horizon_steps: int,
    max_windows_per_batch: int | None,
) -> dict[str, np.ndarray]:
    process_columns = metadata["process_columns"]
    history_idx = [process_columns.index(col) for col in metadata["history_columns"]]
    exo_idx = [process_columns.index(col) for col in metadata["exogenous_columns"]]
    target_idx = [process_columns.index(col) for col in metadata["target_columns"]]
    history_values: list[np.ndarray] = []
    future_exogenous_values: list[np.ndarray] = []
    x_values: list[np.ndarray] = []
    y_scaled_values: list[np.ndarray] = []
    y_raw_values: list[np.ndarray] = []
    batch_values: list[int] = []
    source_t_values: list[int] = []
    target_t_values: list[int] = []
    target_time_values: list[float] = []
    for batch_id in metadata["splits"][split]:
        batch = load_batch_npz(metadata["processed_root"], batch_id)
        process = batch["process"].astype(np.float32)
        process_scaled = standardize(process, metadata["process_centers"], metadata["process_scales"])
        time_h = batch["time_h"].astype(np.float32)
        candidates: list[int] = []
        for t in range(history_steps - 1, process.shape[0] - horizon_steps):
            target_t = t + horizon_steps
            y_raw = process[target_t, target_idx]
            if np.isfinite(y_raw).all():
                candidates.append(t)
        capped = deterministic_cap_indices(len(candidates), max_windows_per_batch)
        for local_idx in capped:
            t = candidates[int(local_idx)]
            target_t = t + horizon_steps
            history = process_scaled[t - history_steps + 1 : t + 1, history_idx]
            future_exogenous = process_scaled[target_t, exo_idx]
            y_raw = process[target_t, target_idx]
            y_scaled = standardize(y_raw, metadata["target_centers"], metadata["target_scales"])
            x = np.concatenate([history.reshape(-1), future_exogenous.reshape(-1)]).astype(np.float32)
            history_values.append(history.astype(np.float32))
            future_exogenous_values.append(future_exogenous.astype(np.float32))
            x_values.append(x)
            y_scaled_values.append(y_scaled.astype(np.float32))
            y_raw_values.append(y_raw.astype(np.float32))
            batch_values.append(int(batch_id))
            source_t_values.append(int(t))
            target_t_values.append(int(target_t))
            target_time_values.append(float(time_h[target_t]))
    if not x_values:
        raise RuntimeError(f"Empty design for split={split}, history={history_steps}, horizon={horizon_steps}")
    return {
        "history": np.asarray(history_values, dtype=np.float32),
        "future_exogenous": np.asarray(future_exogenous_values, dtype=np.float32),
        "x": np.asarray(x_values, dtype=np.float32),
        "y_scaled": np.asarray(y_scaled_values, dtype=np.float32),
        "y_raw": np.asarray(y_raw_values, dtype=np.float32),
        "batch_id": np.asarray(batch_values, dtype=np.int64),
        "source_t": np.asarray(source_t_values, dtype=np.int64),
        "target_t": np.asarray(target_t_values, dtype=np.int64),
        "target_time_h": np.asarray(target_time_values, dtype=np.float32),
    }


def get_design_bundle(
    *,
    cache: dict[tuple[int, int, str, str], dict[str, np.ndarray]],
    metadata: dict[str, Any],
    history_steps: int,
    horizon_steps: int,
    train_cap: int | None,
    eval_cap: int | None,
) -> dict[str, dict[str, np.ndarray]]:
    bundle = {}
    for split in SPLITS:
        cap = train_cap if split == "train_normal" else eval_cap
        key = (history_steps, horizon_steps, split, str(cap))
        if key not in cache:
            cache[key] = build_window_design(
                metadata=metadata,
                split=split,
                history_steps=history_steps,
                horizon_steps=horizon_steps,
                max_windows_per_batch=cap,
            )
        bundle[split] = cache[key]
    return bundle


def add_common_metric_fields(rows: list[dict[str, Any]], **fields: Any) -> list[dict[str, Any]]:
    for row in rows:
        row.update(fields)
    return rows


def evaluate_scaled_predictions(
    *,
    metadata: dict[str, Any],
    split_data: dict[str, np.ndarray],
    pred_scaled: np.ndarray,
    model_name: str,
    model_family: str,
    forecast_mode: str,
    horizon_steps: int,
    horizon_h: float,
    seed: int | str,
    grid_id: str,
    stage: str,
    actual_backend: str,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], float]:
    finite_or_raise(pred_scaled, f"{model_name}:pred_scaled")
    pred_raw = inverse_standardize(pred_scaled, metadata["target_centers"], metadata["target_scales"])
    metrics = regression_metrics(split_data["y_raw"], pred_raw, metadata["target_scales"])
    rows = metric_rows(
        phase=PHASE,
        experiment_name=EXPERIMENT,
        split=stage,
        model=model_name,
        forecast_mode=forecast_mode,
        horizon_steps=horizon_steps,
        horizon_h=horizon_h,
        target_columns=metadata["target_columns"],
        metrics=metrics,
    )
    add_common_metric_fields(
        rows,
        model_family=model_family,
        actual_backend=actual_backend,
        seed=seed,
        grid_id=grid_id,
        selection_policy="val_normal hyperparameter selection before test evaluation",
        history_steps=forecast_mode.split("history")[-1] if "history" in forecast_mode else "",
    )
    payload = {
        "y_true": split_data["y_raw"].astype(np.float32),
        "y_pred": pred_raw.astype(np.float32),
        "y_pred_scaled": pred_scaled.astype(np.float32),
        "batch_id": split_data["batch_id"].astype(np.int64),
        "source_t": split_data["source_t"].astype(np.int64),
        "target_t": split_data["target_t"].astype(np.int64),
        "target_time_h": split_data["target_time_h"].astype(np.float32),
    }
    score = mean_nrmse_scaled(split_data["y_scaled"], pred_scaled)
    return rows, payload, score


def fit_feature_reducer(
    sk: dict[str, Any], x_train: np.ndarray, x_eval: dict[str, np.ndarray], n_components: int | None, seed: int
) -> tuple[np.ndarray, dict[str, np.ndarray], str]:
    if n_components is None:
        return x_train, x_eval, "none"
    n_components = min(int(n_components), x_train.shape[0] - 1, x_train.shape[1])
    if n_components < 1:
        return x_train, x_eval, "none"
    pca = sk["PCA"](n_components=n_components, svd_solver="randomized", random_state=seed)
    x_train_red = pca.fit_transform(x_train)
    x_eval_red = {name: pca.transform(values) for name, values in x_eval.items()}
    return x_train_red.astype(np.float32), {k: v.astype(np.float32) for k, v in x_eval_red.items()}, f"pca{n_components}"


def build_classical_model(sk: dict[str, Any], model_family: str, grid: dict[str, Any], seed: int, n_jobs: int) -> tuple[Any, str]:
    if model_family == "random_forest":
        model = sk["RandomForestRegressor"](
            n_estimators=int(grid["n_estimators"]),
            max_depth=None if grid.get("max_depth") is None else int(grid["max_depth"]),
            min_samples_leaf=int(grid["min_samples_leaf"]),
            max_features=grid.get("max_features", 1.0),
            random_state=seed,
            n_jobs=n_jobs,
        )
        return model, "sklearn_random_forest"
    if model_family == "xgboost":
        if importlib.util.find_spec("xgboost") is not None:
            from xgboost import XGBRegressor

            base = XGBRegressor(
                n_estimators=int(grid["n_estimators"]),
                max_depth=int(grid["max_depth"]),
                learning_rate=float(grid["learning_rate"]),
                subsample=float(grid["subsample"]),
                colsample_bytree=float(grid["colsample_bytree"]),
                reg_lambda=float(grid["reg_lambda"]),
                objective="reg:squarederror",
                tree_method="hist",
                random_state=seed,
                n_jobs=n_jobs,
                verbosity=1,
            )
            return sk["MultiOutputRegressor"](base, n_jobs=1), "xgboost"
        base = sk["HistGradientBoostingRegressor"](
            max_iter=int(grid["n_estimators"]),
            max_leaf_nodes=31,
            learning_rate=float(grid["learning_rate"]),
            l2_regularization=float(grid["reg_lambda"]),
            random_state=seed,
        )
        return sk["MultiOutputRegressor"](base, n_jobs=1), "sklearn_hist_gradient_boosting_fallback"
    if model_family == "gaussian_process":
        kernel = (
            sk["ConstantKernel"](1.0, constant_value_bounds="fixed")
            * sk["RBF"](length_scale=float(grid["length_scale"]), length_scale_bounds="fixed")
            + sk["WhiteKernel"](noise_level=float(grid["alpha"]), noise_level_bounds="fixed")
        )
        base = sk["GaussianProcessRegressor"](
            kernel=kernel,
            alpha=float(grid["alpha"]),
            normalize_y=True,
            random_state=seed,
            optimizer=None,
        )
        return sk["MultiOutputRegressor"](base, n_jobs=1), "sklearn_gaussian_process"
    raise ValueError(f"Unknown classical model family: {model_family}")


def model_complexity(model: Any, model_family: str) -> dict[str, Any]:
    out: dict[str, Any] = {"complexity_unit": "not_available", "complexity_value": math.nan}
    if model_family == "random_forest" and hasattr(model, "estimators_"):
        nodes = sum(int(est.tree_.node_count) for est in model.estimators_)
        out.update({"complexity_unit": "tree_nodes", "complexity_value": nodes})
    elif hasattr(model, "estimators_"):
        count = len(getattr(model, "estimators_", []))
        out.update({"complexity_unit": "wrapped_estimators", "complexity_value": count})
    return out


def run_classical_forecasting(
    *,
    cfg: dict[str, Any],
    dirs: dict[str, Path],
    metadata: dict[str, Any],
    sk: dict[str, Any],
    n_jobs: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    design_cache: dict[tuple[int, int, str, str], dict[str, np.ndarray]] = {}
    dt_h = median_dt_h(metadata)
    all_metric_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    complexity_rows: list[dict[str, Any]] = []
    for horizon in [int(h) for h in cfg["horizon_steps"]]:
        for model_family, grids in cfg["classical_baselines"].items():
            candidates: list[dict[str, Any]] = []
            log(f"Classical tuning: model={model_family} horizon={horizon}")
            for grid in grids:
                history_steps = int(grid["history_steps"])
                bundle = get_design_bundle(
                    cache=design_cache,
                    metadata=metadata,
                    history_steps=history_steps,
                    horizon_steps=horizon,
                    train_cap=cfg.get("classical_train_max_windows_per_batch"),
                    eval_cap=cfg.get("classical_eval_max_windows_per_batch"),
                )
                train = bundle["train_normal"]
                val = bundle["val_normal"]
                x_eval = {"val_normal": val["x"]}
                x_train = train["x"]
                if model_family == "gaussian_process":
                    seed = int(cfg["tuning_seed"])
                    cap_idx = deterministic_cap_indices(
                        train["x"].shape[0], int(cfg.get("gp_train_max_samples", 700))
                    )
                    x_train = train["x"][cap_idx]
                    y_train = train["y_scaled"][cap_idx]
                else:
                    seed = int(cfg["tuning_seed"])
                    y_train = train["y_scaled"]
                x_train_fit, x_eval_fit, reducer = fit_feature_reducer(
                    sk, x_train, x_eval, grid.get("feature_pca_components"), seed
                )
                model, backend = build_classical_model(sk, model_family, grid, seed, n_jobs)
                start = time.time()
                model.fit(x_train_fit, y_train)
                elapsed = time.time() - start
                pred_val = model.predict(x_eval_fit["val_normal"]).astype(np.float32)
                score = mean_nrmse_scaled(val["y_scaled"], pred_val)
                finite_or_raise(pred_val, f"{model_family}:{grid['grid_id']}:validation_prediction")
                candidates.append(
                    {
                        "model_family": model_family,
                        "actual_backend": backend,
                        "horizon_steps": horizon,
                        "history_steps": history_steps,
                        "grid": grid,
                        "grid_id": grid["grid_id"],
                        "feature_reducer": reducer,
                        "validation_mean_nrmse_train_std": score,
                        "tuning_seconds": elapsed,
                    }
                )
                selection_rows.append(
                    {
                        "phase": PHASE,
                        "model_family": model_family,
                        "actual_backend": backend,
                        "horizon_steps": horizon,
                        "history_steps": history_steps,
                        "grid_id": grid["grid_id"],
                        "feature_reducer": reducer,
                        "validation_mean_nrmse_train_std": score,
                        "tuning_seconds": elapsed,
                        "selection_split": "val_normal",
                        "selected": 0,
                        "grid_json": json.dumps(grid, sort_keys=True),
                    }
                )
                log(
                    f"Tuned {model_family} h{horizon} grid={grid['grid_id']} "
                    f"backend={backend} val_nrmse={score:.6f} seconds={elapsed:.1f}"
                )
            best = min(candidates, key=lambda item: float(item["validation_mean_nrmse_train_std"]))
            for row in selection_rows:
                if (
                    row["model_family"] == model_family
                    and int(row["horizon_steps"]) == horizon
                    and row["grid_id"] == best["grid_id"]
                ):
                    row["selected"] = 1
            for seed in [int(s) for s in cfg["final_seeds"]]:
                grid = dict(best["grid"])
                history_steps = int(grid["history_steps"])
                bundle = get_design_bundle(
                    cache=design_cache,
                    metadata=metadata,
                    history_steps=history_steps,
                    horizon_steps=horizon,
                    train_cap=cfg.get("classical_train_max_windows_per_batch"),
                    eval_cap=cfg.get("classical_eval_max_windows_per_batch"),
                )
                train = bundle["train_normal"]
                x_train = train["x"]
                y_train = train["y_scaled"]
                if model_family == "gaussian_process":
                    cap_idx = deterministic_cap_indices(x_train.shape[0], int(cfg.get("gp_train_max_samples", 700)))
                    x_train = x_train[cap_idx]
                    y_train = y_train[cap_idx]
                x_eval = {split: bundle[split]["x"] for split in SPLITS}
                x_train_fit, x_eval_fit, reducer = fit_feature_reducer(
                    sk, x_train, x_eval, grid.get("feature_pca_components"), seed
                )
                model, backend = build_classical_model(sk, model_family, grid, seed, n_jobs)
                start = time.time()
                model.fit(x_train_fit, y_train)
                training_seconds = time.time() - start
                model_name = model_family if backend == "xgboost" else (
                    "xgboost_fallback_hgb" if model_family == "xgboost" else model_family
                )
                prediction_manifest: dict[str, str] = {}
                split_scores: dict[str, float] = {}
                latency_ms = math.nan
                for split in SPLITS:
                    start_pred = time.time()
                    pred_scaled = model.predict(x_eval_fit[split]).astype(np.float32)
                    pred_seconds = time.time() - start_pred
                    if split == "test_normal" and pred_scaled.shape[0] > 0:
                        latency_ms = 1000.0 * pred_seconds / pred_scaled.shape[0]
                    rows, payload, score = evaluate_scaled_predictions(
                        metadata=metadata,
                        split_data=bundle[split],
                        pred_scaled=pred_scaled,
                        model_name=model_name,
                        model_family=model_family,
                        forecast_mode=f"controlled_history{history_steps}_{reducer}",
                        horizon_steps=horizon,
                        horizon_h=horizon * dt_h,
                        seed=seed,
                        grid_id=grid["grid_id"],
                        stage=split,
                        actual_backend=backend,
                    )
                    all_metric_rows.extend(rows)
                    split_scores[split] = score
                    pred_path = (
                        dirs["predictions"]
                        / f"{model_name}_h{horizon}_{grid['grid_id']}_seed{seed}_{split}.npz"
                    )
                    np.savez_compressed(pred_path, **payload)
                    prediction_manifest[split] = str(pred_path)
                complexity = model_complexity(model, model_family)
                complexity_row = {
                    "phase": PHASE,
                    "model": model_name,
                    "model_family": model_family,
                    "actual_backend": backend,
                    "horizon_steps": horizon,
                    "history_steps": history_steps,
                    "seed": seed,
                    "grid_id": grid["grid_id"],
                    "feature_reducer": reducer,
                    "training_seconds": training_seconds,
                    "test_normal_latency_ms_per_sample": latency_ms,
                    "train_windows": int(train["x"].shape[0]),
                    "val_nrmse_train_std": split_scores.get("val_normal", math.nan),
                    "test_normal_nrmse_train_std": split_scores.get("test_normal", math.nan),
                    "test_fault_nrmse_train_std": split_scores.get("test_fault", math.nan),
                    **complexity,
                }
                complexity_rows.append(complexity_row)
                manifest = {
                    "phase": "16_strong_baselines",
                    "run_type": "classical_forecasting",
                    "model": model_name,
                    "model_family": model_family,
                    "actual_backend": backend,
                    "horizon_steps": horizon,
                    "horizon_h": horizon * dt_h,
                    "history_steps": history_steps,
                    "seed": seed,
                    "grid": grid,
                    "feature_reducer": reducer,
                    "selection_metric": "val_normal mean_nrmse_train_std",
                    "selected_validation_score": best["validation_mean_nrmse_train_std"],
                    "training_seconds": training_seconds,
                    "prediction_files": prediction_manifest,
                    "complexity": complexity_row,
                }
                manifest_path = (
                    dirs["models"] / f"{model_name}_h{horizon}_{grid['grid_id']}_seed{seed}_manifest.json"
                )
                manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
                log(
                    f"Final {model_name} h{horizon} seed={seed} "
                    f"val={split_scores.get('val_normal', math.nan):.6f} "
                    f"test_normal={split_scores.get('test_normal', math.nan):.6f} "
                    f"test_fault={split_scores.get('test_fault', math.nan):.6f}"
                )
    return all_metric_rows, selection_rows, complexity_rows


def build_patchtst_model_class(torch: Any, nn: Any) -> Any:
    class PatchTSTForecaster(nn.Module):
        def __init__(
            self,
            input_dim: int,
            exogenous_dim: int,
            output_dim: int,
            history_steps: int,
            patch_len: int,
            stride: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.patch_len = int(patch_len)
            self.stride = int(stride)
            n_patches = 1 + max(0, (int(history_steps) - int(patch_len)) // int(stride))
            self.patch_proj = nn.Linear(input_dim * int(patch_len), d_model)
            self.position = nn.Parameter(torch.zeros(1, max(1, n_patches), d_model))
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.head = nn.Sequential(
                nn.LayerNorm(d_model + exogenous_dim),
                nn.Linear(d_model + exogenous_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, output_dim),
            )

        def forward(self, history: Any, future_exogenous: Any) -> Any:
            if history.shape[1] < self.patch_len:
                pad = self.patch_len - history.shape[1]
                history = torch.nn.functional.pad(history, (0, 0, pad, 0))
            patches = history.unfold(dimension=1, size=self.patch_len, step=self.stride)
            patches = patches.permute(0, 1, 3, 2).contiguous().flatten(start_dim=2)
            tokens = self.patch_proj(patches)
            tokens = tokens + self.position[:, : tokens.shape[1], :]
            encoded = self.encoder(tokens).mean(dim=1)
            return self.head(torch.cat([encoded, future_exogenous], dim=-1))

    return PatchTSTForecaster


def build_array_dataset_class(Dataset: Any) -> Any:
    class ArrayWindowDataset(Dataset):
        def __init__(self, data: dict[str, np.ndarray]) -> None:
            self.data = data

        def __len__(self) -> int:
            return int(self.data["history"].shape[0])

        def __getitem__(self, idx: int) -> dict[str, Any]:
            return {
                "history": self.data["history"][idx].astype(np.float32),
                "future_exogenous": self.data["future_exogenous"][idx].astype(np.float32),
                "target_scaled": self.data["y_scaled"][idx].astype(np.float32),
                "target_raw": self.data["y_raw"][idx].astype(np.float32),
                "batch_id": np.int64(self.data["batch_id"][idx]),
                "source_t": np.int64(self.data["source_t"][idx]),
                "target_t": np.int64(self.data["target_t"][idx]),
                "target_time_h": np.float32(self.data["target_time_h"][idx]),
            }

    return ArrayWindowDataset


def load_torch_checkpoint(path: Path, device: Any, torch: Any) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def run_patchtst_one(
    *,
    cfg: dict[str, Any],
    dirs: dict[str, Path],
    metadata: dict[str, Any],
    bundle: dict[str, dict[str, np.ndarray]],
    grid: dict[str, Any],
    horizon: int,
    seed: int,
    stage: str,
    epochs: int,
    device_arg: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    torch, nn, DataLoader, Dataset = import_torch()
    set_seed(seed, torch)
    device = torch.device("cuda" if device_arg == "auto" and torch.cuda.is_available() else ("cpu" if device_arg == "auto" else device_arg))
    ArrayWindowDataset = build_array_dataset_class(Dataset)
    PatchTSTForecaster = build_patchtst_model_class(torch, nn)
    training_cfg = cfg["patchtst"]["training"]
    run_name = f"{stage}_patchtst_h{horizon}_{grid['grid_id']}_seed{seed}"
    run_dir = dirs["output"] / "patchtst_runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = ArrayWindowDataset(bundle["train_normal"])
    val_dataset = ArrayWindowDataset(bundle["val_normal"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(training_cfg["num_workers"]),
        pin_memory=bool(training_cfg["pin_memory"]),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(training_cfg["num_workers"]),
        pin_memory=bool(training_cfg["pin_memory"]),
    )
    model = PatchTSTForecaster(
        input_dim=len(metadata["history_columns"]),
        exogenous_dim=len(metadata["exogenous_columns"]),
        output_dim=len(metadata["target_columns"]),
        history_steps=int(grid["history_steps"]),
        patch_len=int(grid["patch_len"]),
        stride=int(grid["stride"]),
        d_model=int(grid["d_model"]),
        num_layers=int(grid["num_layers"]),
        num_heads=int(grid["num_heads"]),
        dropout=float(grid["dropout"]),
    ).to(device)
    param_count = sum(param.numel() for param in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(grid["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    loss_fn = torch.nn.MSELoss()
    best_val = math.inf
    best_epoch = -1
    min_delta = float(training_cfg["min_delta"])
    patience_left = int(training_cfg["patience"])
    stop_reason = "max_epochs"
    checkpoint_path = run_dir / "best_model.pt"
    history_rows: list[dict[str, Any]] = []
    start = time.time()
    log(f"PatchTST start {run_name} device={device} parameters={param_count}")
    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            history = batch["history"].to(device=device, dtype=torch.float32)
            future_exogenous = batch["future_exogenous"].to(device=device, dtype=torch.float32)
            target = batch["target_scaled"].to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            pred = model(history, future_exogenous)
            loss = loss_fn(pred, target)
            if not bool(torch.isfinite(loss).detach().cpu()):
                raise FloatingPointError(f"{run_name}: non-finite train loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_cfg["grad_clip_norm"]))
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                history = batch["history"].to(device=device, dtype=torch.float32)
                future_exogenous = batch["future_exogenous"].to(device=device, dtype=torch.float32)
                target = batch["target_scaled"].to(device=device, dtype=torch.float32)
                pred = model(history, future_exogenous)
                loss = loss_fn(pred, target)
                if not bool(torch.isfinite(loss).detach().cpu()):
                    raise FloatingPointError(f"{run_name}: non-finite val loss")
                val_losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        improvement = best_val - val_loss
        is_best = improvement > min_delta
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss_mse_scaled": train_loss,
                "val_loss_mse_scaled": val_loss,
                "best_val_before_update": best_val,
                "improvement_vs_best": improvement if math.isfinite(best_val) else math.nan,
                "min_delta": min_delta,
                "is_best": int(is_best),
                "patience_left_before_update": patience_left,
                "model": "patchtst",
                "horizon_steps": horizon,
                "seed": seed,
                "grid_id": grid["grid_id"],
                "stage": stage,
            }
        )
        log(
            f"{run_name}: epoch={epoch} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} best={best_val:.6f} improvement={improvement:.10f}"
        )
        if is_best:
            best_val = val_loss
            best_epoch = epoch
            patience_left = int(training_cfg["patience"])
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "grid": grid,
                    "seed": seed,
                    "horizon_steps": horizon,
                    "best_val_loss_mse_scaled": best_val,
                    "best_epoch": best_epoch,
                },
                checkpoint_path,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                stop_reason = "early_stop_min_delta_patience"
                break
    if best_epoch < 0:
        raise RuntimeError(f"{run_name}: no valid checkpoint")
    write_union_csv(run_dir / "training_history.csv", history_rows)
    checkpoint = load_torch_checkpoint(checkpoint_path, device, torch)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dt_h = median_dt_h(metadata)
    all_rows: list[dict[str, Any]] = []
    prediction_manifest: dict[str, str] = {}
    split_scores: dict[str, float] = {}
    latency_ms = math.nan
    eval_splits = ["val_normal"] if stage == "tuning" else SPLITS
    with torch.no_grad():
        for split in eval_splits:
            dataset = ArrayWindowDataset(bundle[split])
            loader = DataLoader(
                dataset,
                batch_size=int(training_cfg["batch_size"]),
                shuffle=False,
                num_workers=int(training_cfg["num_workers"]),
                pin_memory=bool(training_cfg["pin_memory"]),
            )
            pred_blocks = []
            start_pred = time.time()
            for batch in loader:
                history = batch["history"].to(device=device, dtype=torch.float32)
                future_exogenous = batch["future_exogenous"].to(device=device, dtype=torch.float32)
                pred = model(history, future_exogenous)
                if not bool(torch.isfinite(pred).all().detach().cpu()):
                    raise FloatingPointError(f"{run_name}: non-finite prediction")
                pred_blocks.append(pred.detach().cpu().numpy().astype(np.float32))
            pred_scaled = np.vstack(pred_blocks)
            pred_seconds = time.time() - start_pred
            if split == "test_normal" and pred_scaled.shape[0] > 0:
                latency_ms = 1000.0 * pred_seconds / pred_scaled.shape[0]
            rows, payload, score = evaluate_scaled_predictions(
                metadata=metadata,
                split_data=bundle[split],
                pred_scaled=pred_scaled,
                model_name="patchtst",
                model_family="patchtst",
                forecast_mode=f"controlled_history{grid['history_steps']}_patch{grid['patch_len']}",
                horizon_steps=horizon,
                horizon_h=horizon * dt_h,
                seed=seed,
                grid_id=grid["grid_id"],
                stage=split,
                actual_backend="pytorch_patchtst",
            )
            all_rows.extend(rows)
            split_scores[split] = score
            pred_path = run_dir / f"predictions_{split}.npz"
            np.savez_compressed(pred_path, **payload)
            prediction_manifest[split] = str(pred_path)
    manifest = {
        "phase": "16_strong_baselines",
        "run_type": "patchtst_forecasting",
        "stage": stage,
        "model": "patchtst",
        "actual_backend": "pytorch_patchtst",
        "horizon_steps": horizon,
        "horizon_h": horizon * dt_h,
        "history_steps": int(grid["history_steps"]),
        "seed": seed,
        "grid": grid,
        "best_epoch": best_epoch,
        "best_val_loss_mse_scaled": best_val,
        "stop_reason": stop_reason,
        "elapsed_seconds": time.time() - start,
        "parameter_count": param_count,
        "test_normal_latency_ms_per_sample": latency_ms,
        "prediction_files": prediction_manifest,
        "checkpoint": str(checkpoint_path),
        "training_history_csv": str(run_dir / "training_history.csv"),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    complexity_rows = [
        {
            "phase": PHASE,
            "model": "patchtst",
            "model_family": "patchtst",
            "actual_backend": "pytorch_patchtst",
            "horizon_steps": horizon,
            "history_steps": int(grid["history_steps"]),
            "seed": seed,
            "grid_id": grid["grid_id"],
            "feature_reducer": "patch_embedding",
            "training_seconds": manifest["elapsed_seconds"],
            "test_normal_latency_ms_per_sample": latency_ms,
            "train_windows": int(bundle["train_normal"]["x"].shape[0]),
            "val_nrmse_train_std": split_scores.get("val_normal", math.nan),
            "test_normal_nrmse_train_std": split_scores.get("test_normal", math.nan),
            "test_fault_nrmse_train_std": split_scores.get("test_fault", math.nan),
            "complexity_unit": "trainable_parameters",
            "complexity_value": param_count,
        }
    ]
    return manifest, all_rows, complexity_rows


def run_patchtst_forecasting(
    *,
    cfg: dict[str, Any],
    dirs: dict[str, Path],
    metadata: dict[str, Any],
    device_arg: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not bool(cfg["patchtst"].get("enabled", True)):
        return [], [], []
    design_cache: dict[tuple[int, int, str, str], dict[str, np.ndarray]] = {}
    metric_rows_all: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    complexity_rows_all: list[dict[str, Any]] = []
    final_manifests: list[dict[str, Any]] = []
    for horizon in [int(h) for h in cfg["horizon_steps"]]:
        tuning_manifests = []
        for grid in cfg["patchtst"]["tuning_grids"]:
            history_steps = int(grid["history_steps"])
            bundle = get_design_bundle(
                cache=design_cache,
                metadata=metadata,
                history_steps=history_steps,
                horizon_steps=horizon,
                train_cap=cfg.get("patchtst_train_max_windows_per_batch"),
                eval_cap=cfg.get("patchtst_eval_max_windows_per_batch"),
            )
            manifest, rows, complexity_rows = run_patchtst_one(
                cfg=cfg,
                dirs=dirs,
                metadata=metadata,
                bundle=bundle,
                grid=grid,
                horizon=horizon,
                seed=int(cfg["tuning_seed"]),
                stage="tuning",
                epochs=int(cfg["patchtst"]["training"]["tuning_epochs"]),
                device_arg=device_arg,
            )
            tuning_manifests.append(manifest)
            metric_rows_all.extend(rows)
            complexity_rows_all.extend(complexity_rows)
            selection_rows.append(
                {
                    "phase": PHASE,
                    "model_family": "patchtst",
                    "actual_backend": "pytorch_patchtst",
                    "horizon_steps": horizon,
                    "history_steps": int(grid["history_steps"]),
                    "grid_id": grid["grid_id"],
                    "feature_reducer": "patch_embedding",
                    "validation_mean_nrmse_train_std": math.nan,
                    "validation_loss_mse_scaled": manifest["best_val_loss_mse_scaled"],
                    "tuning_seconds": manifest["elapsed_seconds"],
                    "selection_split": "val_normal",
                    "selected": 0,
                    "grid_json": json.dumps(grid, sort_keys=True),
                }
            )
        best = min(tuning_manifests, key=lambda item: float(item["best_val_loss_mse_scaled"]))
        best_grid = best["grid"]
        for row in selection_rows:
            if (
                row["model_family"] == "patchtst"
                and int(row["horizon_steps"]) == horizon
                and row["grid_id"] == best_grid["grid_id"]
            ):
                row["selected"] = 1
        for seed in [int(s) for s in cfg["final_seeds"]]:
            history_steps = int(best_grid["history_steps"])
            bundle = get_design_bundle(
                cache=design_cache,
                metadata=metadata,
                history_steps=history_steps,
                horizon_steps=horizon,
                train_cap=cfg.get("patchtst_train_max_windows_per_batch"),
                eval_cap=cfg.get("patchtst_eval_max_windows_per_batch"),
            )
            manifest, rows, complexity_rows = run_patchtst_one(
                cfg=cfg,
                dirs=dirs,
                metadata=metadata,
                bundle=bundle,
                grid=best_grid,
                horizon=horizon,
                seed=seed,
                stage="final",
                epochs=int(cfg["patchtst"]["training"]["final_epochs"]),
                device_arg=device_arg,
            )
            final_manifests.append(manifest)
            metric_rows_all.extend(rows)
            complexity_rows_all.extend(complexity_rows)
    (dirs["output"] / "patchtst_final_manifests.json").write_text(
        json.dumps(final_manifests, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metric_rows_all, selection_rows, complexity_rows_all


def build_anomaly_design(
    *,
    metadata: dict[str, Any],
    split: str,
    history_steps: int,
    max_windows_per_batch: int | None,
    eval_stride: int,
) -> dict[str, np.ndarray]:
    current_values: list[np.ndarray] = []
    dynamic_values: list[np.ndarray] = []
    label_values: list[int] = []
    batch_values: list[int] = []
    time_values: list[float] = []
    for batch_id in metadata["splits"][split]:
        batch = load_batch_npz(metadata["processed_root"], batch_id)
        process = batch["process"].astype(np.float32)
        process_scaled = standardize(process, metadata["process_centers"], metadata["process_scales"])
        time_h = batch["time_h"].astype(np.float32)
        fault_label = int(batch["fault_label"])
        candidates = list(range(history_steps - 1, process.shape[0], max(1, eval_stride)))
        capped = deterministic_cap_indices(len(candidates), max_windows_per_batch)
        for local_idx in capped:
            t = candidates[int(local_idx)]
            current = process_scaled[t]
            dynamic = process_scaled[t - history_steps + 1 : t + 1].reshape(-1)
            current_values.append(current.astype(np.float32))
            dynamic_values.append(dynamic.astype(np.float32))
            label_values.append(fault_label)
            batch_values.append(int(batch_id))
            time_values.append(float(time_h[t]))
    return {
        "current_x": np.asarray(current_values, dtype=np.float32),
        "dynamic_x": np.asarray(dynamic_values, dtype=np.float32),
        "label": np.asarray(label_values, dtype=np.int64),
        "batch_id": np.asarray(batch_values, dtype=np.int64),
        "time_h": np.asarray(time_values, dtype=np.float32),
    }


def auroc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return math.nan
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    rank_sum_pos = float(np.sum(ranks[labels == 1]))
    auc = (rank_sum_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size)
    return float(auc)


def auprc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(int)
    positives = int(np.sum(labels == 1))
    if positives == 0:
        return math.nan
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    labels = labels.astype(int)
    pred = scores >= threshold
    normal = labels == 0
    fault = labels == 1
    tp = float(np.sum(pred & fault))
    fp = float(np.sum(pred & normal))
    fn = float(np.sum((~pred) & fault))
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "false_alarm_rate": float(np.mean(pred[normal])) if np.any(normal) else math.nan,
        "fault_detection_rate": float(np.mean(pred[fault])) if np.any(fault) else math.nan,
        "missed_detection_rate": float(np.mean(~pred[fault])) if np.any(fault) else math.nan,
        "sample_f1": float(f1),
    }


def batch_delay_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float, batch_ids: np.ndarray, time_h: np.ndarray
) -> dict[str, float]:
    batch_true = []
    batch_pred = []
    delays = []
    for batch_id in sorted(set(int(v) for v in batch_ids.tolist())):
        mask = batch_ids == batch_id
        label = int(np.max(labels[mask]))
        detected = bool(np.any(scores[mask] >= threshold))
        batch_true.append(label)
        batch_pred.append(1 if detected else 0)
        if label == 1:
            times = time_h[mask]
            flagged = times[scores[mask] >= threshold]
            delays.append(float(np.min(flagged) - np.min(times)) if flagged.size else math.nan)
    batch_true_arr = np.asarray(batch_true, dtype=int)
    batch_pred_arr = np.asarray(batch_pred, dtype=int)
    tp = float(np.sum((batch_true_arr == 1) & (batch_pred_arr == 1)))
    fp = float(np.sum((batch_true_arr == 0) & (batch_pred_arr == 1)))
    fn = float(np.sum((batch_true_arr == 1) & (batch_pred_arr == 0)))
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    finite_delays = [d for d in delays if math.isfinite(d)]
    return {
        "batch_level_f1": float(f1),
        "mean_detection_delay_h": float(np.mean(finite_delays)) if finite_delays else math.nan,
        "median_detection_delay_h": float(np.median(finite_delays)) if finite_delays else math.nan,
    }


def anomaly_rows(
    *,
    method: str,
    backend: str,
    grid_id: str,
    threshold: float,
    train_scores: np.ndarray,
    normal_payload: dict[str, np.ndarray],
    fault_payload: dict[str, np.ndarray],
    extra: dict[str, Any],
) -> list[dict[str, Any]]:
    labels = np.concatenate([
        np.zeros_like(normal_payload["label"], dtype=np.int64),
        np.ones_like(fault_payload["label"], dtype=np.int64),
    ])
    scores = np.concatenate([normal_payload["score"], fault_payload["score"]]).astype(np.float64)
    batch_ids = np.concatenate([normal_payload["batch_id"], fault_payload["batch_id"]]).astype(np.int64)
    time_h = np.concatenate([normal_payload["time_h"], fault_payload["time_h"]]).astype(np.float64)
    metrics = {
        "auroc": auroc_score(labels, scores),
        "auprc": auprc_score(labels, scores),
        "threshold": float(threshold),
        "train_score_mean": float(np.mean(train_scores)),
        "train_score_std": float(np.std(train_scores)),
        **threshold_metrics(labels, scores, threshold),
        **batch_delay_metrics(labels, scores, threshold, batch_ids, time_h),
    }
    rows = []
    for metric_name, value in metrics.items():
        rows.append(
            {
                "phase": PHASE,
                "experiment_name": EXPERIMENT,
                "method": method,
                "actual_backend": backend,
                "grid_id": grid_id,
                "metric_name": metric_name,
                "metric_value": value,
                "threshold_source": "train_normal_quantile",
                **extra,
            }
        )
    return rows


def run_anomaly_baselines(
    *,
    cfg: dict[str, Any],
    dirs: dict[str, Path],
    metadata: dict[str, Any],
    sk: dict[str, Any],
    device_arg: str,
    n_jobs: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not bool(cfg["anomaly_baselines"].get("enabled", True)):
        return [], []
    acfg = cfg["anomaly_baselines"]
    history_steps = int(acfg["history_steps"])
    log("Anomaly baselines: building train_normal design")
    train = build_anomaly_design(
        metadata=metadata,
        split="train_normal",
        history_steps=history_steps,
        max_windows_per_batch=int(acfg["train_max_windows_per_batch"]),
        eval_stride=1,
    )
    log("Anomaly baselines: building val_normal/test_normal/test_fault designs")
    val = build_anomaly_design(
        metadata=metadata,
        split="val_normal",
        history_steps=history_steps,
        max_windows_per_batch=None,
        eval_stride=int(acfg["eval_stride"]),
    )
    test_normal = build_anomaly_design(
        metadata=metadata,
        split="test_normal",
        history_steps=history_steps,
        max_windows_per_batch=None,
        eval_stride=int(acfg["eval_stride"]),
    )
    test_fault = build_anomaly_design(
        metadata=metadata,
        split="test_fault",
        history_steps=history_steps,
        max_windows_per_batch=None,
        eval_stride=int(acfg["eval_stride"]),
    )
    threshold_q = float(acfg["threshold_quantile"])
    rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    log(
        "Anomaly baselines: design sizes "
        f"train={train['dynamic_x'].shape} val={val['dynamic_x'].shape} "
        f"test_normal={test_normal['dynamic_x'].shape} test_fault={test_fault['dynamic_x'].shape}"
    )

    def save_scores(method: str, payloads: dict[str, dict[str, np.ndarray]]) -> None:
        for split, payload in payloads.items():
            np.savez_compressed(dirs["predictions"] / f"anomaly_{method}_{split}.npz", **payload)

    pca_candidates = []
    log("Anomaly baselines: fitting PCA-Q candidates")
    for n_comp in [int(v) for v in acfg["pca_components_grid"]]:
        n_current = min(n_comp, train["current_x"].shape[1], train["current_x"].shape[0] - 1)
        pca = sk["PCA"](n_components=n_current, svd_solver="randomized", random_state=int(cfg["tuning_seed"]))
        score_train = pca.fit_transform(train["current_x"])
        recon_train = pca.inverse_transform(score_train)
        train_q = np.mean((train["current_x"] - recon_train) ** 2, axis=1)
        val_score = pca.transform(val["current_x"])
        val_q = np.mean((val["current_x"] - pca.inverse_transform(val_score)) ** 2, axis=1)
        thr = float(np.quantile(train_q, threshold_q))
        val_false_alarm = float(np.mean(val_q >= thr))
        pca_candidates.append((val_false_alarm, float(np.mean(val_q)), n_current, pca, train_q, thr))
        selection_rows.append(
            {
                "phase": PHASE,
                "method": "pca_q",
                "grid_id": f"pca_q_k{n_current}",
                "validation_false_alarm_rate": val_false_alarm,
                "validation_mean_score": float(np.mean(val_q)),
                "selected": 0,
            }
        )
    pca_best = min(pca_candidates, key=lambda item: (item[0], item[1]))
    _, _, n_current, pca_q, train_q, thr_q = pca_best
    for row in selection_rows:
        if row["method"] == "pca_q" and row["grid_id"] == f"pca_q_k{n_current}":
            row["selected"] = 1
    payloads = {}
    for split_name, data in [("test_normal", test_normal), ("test_fault", test_fault)]:
        z = pca_q.transform(data["current_x"])
        q = np.mean((data["current_x"] - pca_q.inverse_transform(z)) ** 2, axis=1)
        payloads[split_name] = {"score": q, "label": data["label"], "batch_id": data["batch_id"], "time_h": data["time_h"]}
    rows.extend(
        anomaly_rows(
            method="PCA-Q",
            backend="sklearn_pca_reconstruction",
            grid_id=f"pca_q_k{n_current}",
            threshold=thr_q,
            train_scores=train_q,
            normal_payload=payloads["test_normal"],
            fault_payload=payloads["test_fault"],
            extra={"history_steps": 1},
        )
    )
    save_scores("pca_q", payloads)
    log("Anomaly baselines: PCA-Q complete")

    z_train = pca_q.transform(train["current_x"])
    train_std = np.std(z_train, axis=0)
    train_std = np.where(train_std > 1e-8, train_std, 1.0)
    train_t2 = np.sum((z_train / train_std) ** 2, axis=1)
    thr_t2 = float(np.quantile(train_t2, threshold_q))
    payloads = {}
    for split_name, data in [("test_normal", test_normal), ("test_fault", test_fault)]:
        z = pca_q.transform(data["current_x"])
        t2 = np.sum((z / train_std) ** 2, axis=1)
        payloads[split_name] = {"score": t2, "label": data["label"], "batch_id": data["batch_id"], "time_h": data["time_h"]}
    rows.extend(
        anomaly_rows(
            method="PCA-T2",
            backend="sklearn_pca_hotelling",
            grid_id=f"pca_t2_k{n_current}",
            threshold=thr_t2,
            train_scores=train_t2,
            normal_payload=payloads["test_normal"],
            fault_payload=payloads["test_fault"],
            extra={"history_steps": 1},
        )
    )
    save_scores("pca_t2", payloads)
    log("Anomaly baselines: PCA-T2 complete")

    dpca_candidates = []
    log("Anomaly baselines: fitting DPCA-Q candidates")
    for n_comp in [int(v) for v in acfg["pca_components_grid"]]:
        n_dynamic = min(n_comp, train["dynamic_x"].shape[1], train["dynamic_x"].shape[0] - 1)
        dpca = sk["PCA"](n_components=n_dynamic, svd_solver="randomized", random_state=int(cfg["tuning_seed"]))
        z_train = dpca.fit_transform(train["dynamic_x"])
        train_score = np.mean((train["dynamic_x"] - dpca.inverse_transform(z_train)) ** 2, axis=1)
        z_val = dpca.transform(val["dynamic_x"])
        val_score = np.mean((val["dynamic_x"] - dpca.inverse_transform(z_val)) ** 2, axis=1)
        thr = float(np.quantile(train_score, threshold_q))
        val_false_alarm = float(np.mean(val_score >= thr))
        dpca_candidates.append((val_false_alarm, float(np.mean(val_score)), n_dynamic, dpca, train_score, thr))
        selection_rows.append(
            {
                "phase": PHASE,
                "method": "dpca_q",
                "grid_id": f"dpca_q_k{n_dynamic}",
                "validation_false_alarm_rate": val_false_alarm,
                "validation_mean_score": float(np.mean(val_score)),
                "selected": 0,
            }
        )
    dpca_best = min(dpca_candidates, key=lambda item: (item[0], item[1]))
    _, _, n_dynamic, dpca, train_dpca, thr_dpca = dpca_best
    for row in selection_rows:
        if row["method"] == "dpca_q" and row["grid_id"] == f"dpca_q_k{n_dynamic}":
            row["selected"] = 1
    payloads = {}
    for split_name, data in [("test_normal", test_normal), ("test_fault", test_fault)]:
        z = dpca.transform(data["dynamic_x"])
        score = np.mean((data["dynamic_x"] - dpca.inverse_transform(z)) ** 2, axis=1)
        payloads[split_name] = {"score": score, "label": data["label"], "batch_id": data["batch_id"], "time_h": data["time_h"]}
    rows.extend(
        anomaly_rows(
            method="DPCA-Q",
            backend="sklearn_dynamic_pca",
            grid_id=f"dpca_q_k{n_dynamic}",
            threshold=thr_dpca,
            train_scores=train_dpca,
            normal_payload=payloads["test_normal"],
            fault_payload=payloads["test_fault"],
            extra={"history_steps": history_steps},
        )
    )
    save_scores("dpca_q", payloads)
    log("Anomaly baselines: DPCA-Q complete")

    for method, grids, estimator_builder in [
        (
            "OCSVM",
            acfg["one_class_svm_grid"],
            lambda grid, seed: sk["OneClassSVM"](nu=float(grid["nu"]), gamma=grid["gamma"]),
        ),
        (
            "IsolationForest",
            acfg["isolation_forest_grid"],
            lambda grid, seed: sk["IsolationForest"](
                n_estimators=int(grid["n_estimators"]),
                contamination=float(grid["contamination"]),
                max_samples=float(grid["max_samples"]),
                random_state=seed,
                n_jobs=n_jobs,
            ),
        ),
    ]:
        candidates = []
        fit_idx = deterministic_cap_indices(
            train["dynamic_x"].shape[0], int(acfg.get("kernel_train_max_samples", train["dynamic_x"].shape[0]))
        )
        fit_x = train["dynamic_x"][fit_idx]
        log(f"Anomaly baselines: fitting {method} candidates on {fit_x.shape[0]} train-normal windows")
        for grid in grids:
            model = estimator_builder(grid, int(cfg["tuning_seed"]))
            start = time.time()
            model.fit(fit_x)
            fit_seconds = time.time() - start
            train_score = -model.score_samples(fit_x)
            val_score = -model.score_samples(val["dynamic_x"])
            thr = float(np.quantile(train_score, threshold_q))
            val_false_alarm = float(np.mean(val_score >= thr))
            candidates.append((val_false_alarm, float(np.mean(val_score)), grid, model, train_score, thr))
            selection_rows.append(
                {
                    "phase": PHASE,
                    "method": method,
                    "grid_id": grid["grid_id"],
                    "validation_false_alarm_rate": val_false_alarm,
                    "validation_mean_score": float(np.mean(val_score)),
                    "fit_seconds": fit_seconds,
                    "fit_window_count": int(fit_x.shape[0]),
                    "selected": 0,
                }
            )
            log(
                f"Anomaly baselines: {method} grid={grid['grid_id']} "
                f"fit_seconds={fit_seconds:.1f} val_false_alarm={val_false_alarm:.4f}"
            )
        _, _, best_grid, best_model, train_score, thr = min(candidates, key=lambda item: (item[0], item[1]))
        for row in selection_rows:
            if row["method"] == method and row["grid_id"] == best_grid["grid_id"]:
                row["selected"] = 1
        payloads = {}
        for split_name, data in [("test_normal", test_normal), ("test_fault", test_fault)]:
            score = -best_model.score_samples(data["dynamic_x"])
            payloads[split_name] = {"score": score, "label": data["label"], "batch_id": data["batch_id"], "time_h": data["time_h"]}
        rows.extend(
            anomaly_rows(
                method=method,
                backend=f"sklearn_{method.lower()}",
                grid_id=best_grid["grid_id"],
                threshold=thr,
                train_scores=train_score,
                normal_payload=payloads["test_normal"],
                fault_payload=payloads["test_fault"],
                extra={
                    "history_steps": history_steps,
                    "fit_window_count": int(fit_x.shape[0]),
                    "threshold_source": "train_normal_fit_subset_quantile",
                },
            )
        )
        save_scores(method.lower(), payloads)
        log(f"Anomaly baselines: {method} complete")

    ae_rows, ae_selection = run_autoencoder_anomaly(
        cfg=cfg,
        dirs=dirs,
        train=train,
        val=val,
        test_normal=test_normal,
        test_fault=test_fault,
        device_arg=device_arg,
    )
    rows.extend(ae_rows)
    selection_rows.extend(ae_selection)
    return rows, selection_rows


def run_autoencoder_anomaly(
    *,
    cfg: dict[str, Any],
    dirs: dict[str, Path],
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    test_normal: dict[str, np.ndarray],
    test_fault: dict[str, np.ndarray],
    device_arg: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    torch, nn, DataLoader, Dataset = import_torch()
    set_seed(int(cfg["tuning_seed"]), torch)
    acfg = cfg["anomaly_baselines"]
    tcfg = acfg["autoencoder_training"]
    threshold_q = float(acfg["threshold_quantile"])
    ae_idx = deterministic_cap_indices(
        train["dynamic_x"].shape[0], int(acfg.get("autoencoder_train_max_samples", train["dynamic_x"].shape[0]))
    )
    ae_train_x = train["dynamic_x"][ae_idx]
    log(f"Anomaly baselines: Autoencoder train subset size={ae_train_x.shape[0]}")

    class MatrixDataset(Dataset):
        def __init__(self, x: np.ndarray) -> None:
            self.x = x.astype(np.float32)

        def __len__(self) -> int:
            return int(self.x.shape[0])

        def __getitem__(self, idx: int) -> np.ndarray:
            return self.x[idx]

    class AE(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def forward(self, x: Any) -> Any:
            return self.net(x)

    def train_one(grid: dict[str, Any], stage: str, epochs: int) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
        set_seed(int(cfg["tuning_seed"]), torch)
        device = torch.device("cuda" if device_arg == "auto" and torch.cuda.is_available() else ("cpu" if device_arg == "auto" else device_arg))
        model = AE(train["dynamic_x"].shape[1], int(grid["hidden_dim"]), int(grid["latent_dim"])).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(grid["learning_rate"]),
            weight_decay=float(tcfg["weight_decay"]),
        )
        loss_fn = torch.nn.MSELoss()
        train_loader = DataLoader(
            MatrixDataset(ae_train_x),
            batch_size=int(tcfg["batch_size"]),
            shuffle=True,
            num_workers=int(tcfg["num_workers"]),
            pin_memory=bool(tcfg["pin_memory"]),
        )
        val_loader = DataLoader(
            MatrixDataset(val["dynamic_x"]),
            batch_size=int(tcfg["batch_size"]),
            shuffle=False,
            num_workers=int(tcfg["num_workers"]),
            pin_memory=bool(tcfg["pin_memory"]),
        )
        best_val = math.inf
        best_state = None
        best_epoch = -1
        patience_left = int(tcfg["patience"])
        min_delta = float(tcfg["min_delta"])
        stop_reason = "max_epochs"
        history_rows = []
        for epoch in range(1, int(epochs) + 1):
            model.train()
            train_losses = []
            for x in train_loader:
                x = x.to(device=device, dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                recon = model(x)
                loss = loss_fn(recon, x)
                if not bool(torch.isfinite(loss).detach().cpu()):
                    raise FloatingPointError(f"AE {grid['grid_id']}: non-finite train loss")
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))
            model.eval()
            val_losses = []
            with torch.no_grad():
                for x in val_loader:
                    x = x.to(device=device, dtype=torch.float32)
                    recon = model(x)
                    loss = loss_fn(recon, x)
                    if not bool(torch.isfinite(loss).detach().cpu()):
                        raise FloatingPointError(f"AE {grid['grid_id']}: non-finite val loss")
                    val_losses.append(float(loss.detach().cpu()))
            train_loss = float(np.mean(train_losses))
            val_loss = float(np.mean(val_losses))
            improvement = best_val - val_loss
            is_best = improvement > min_delta
            history_rows.append(
                {
                    "epoch": epoch,
                    "train_loss_mse": train_loss,
                    "val_loss_mse": val_loss,
                    "best_val_before_update": best_val,
                    "improvement_vs_best": improvement if math.isfinite(best_val) else math.nan,
                    "is_best": int(is_best),
                    "grid_id": grid["grid_id"],
                    "stage": stage,
                }
            )
            if is_best:
                best_val = val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_left = int(tcfg["patience"])
            else:
                patience_left -= 1
                if patience_left <= 0:
                    stop_reason = "early_stop_min_delta_patience"
                    break
        if best_state is None:
            raise RuntimeError(f"AE {grid['grid_id']}: no valid state")
        model.load_state_dict(best_state)
        manifest = {
            "grid": grid,
            "stage": stage,
            "best_val_loss_mse": best_val,
            "best_epoch": best_epoch,
            "stop_reason": stop_reason,
            "parameter_count": sum(param.numel() for param in model.parameters()),
        }
        write_union_csv(dirs["output"] / "anomaly_ae" / f"{stage}_{grid['grid_id']}_history.csv", history_rows)
        return model, manifest, history_rows

    def score_model(model: Any, x: np.ndarray) -> np.ndarray:
        device = next(model.parameters()).device
        loader = DataLoader(
            MatrixDataset(x),
            batch_size=int(tcfg["batch_size"]),
            shuffle=False,
            num_workers=int(tcfg["num_workers"]),
            pin_memory=bool(tcfg["pin_memory"]),
        )
        scores = []
        model.eval()
        with torch.no_grad():
            for block in loader:
                block = block.to(device=device, dtype=torch.float32)
                recon = model(block)
                score = torch.mean((recon - block) ** 2, dim=1)
                scores.append(score.detach().cpu().numpy())
        return np.concatenate(scores).astype(np.float64)

    selection_rows = []
    candidates = []
    (dirs["output"] / "anomaly_ae").mkdir(parents=True, exist_ok=True)
    for grid in acfg["autoencoder_grid"]:
        log(f"Anomaly baselines: fitting Autoencoder grid={grid['grid_id']}")
        model, manifest, _ = train_one(grid, "tuning", int(tcfg["epochs"]))
        train_score = score_model(model, ae_train_x)
        val_score = score_model(model, val["dynamic_x"])
        thr = float(np.quantile(train_score, threshold_q))
        val_false_alarm = float(np.mean(val_score >= thr))
        candidates.append((val_false_alarm, manifest["best_val_loss_mse"], grid, model, train_score, thr, manifest))
        selection_rows.append(
            {
                "phase": PHASE,
                "method": "Autoencoder",
                "grid_id": grid["grid_id"],
                "validation_false_alarm_rate": val_false_alarm,
                "validation_mean_score": float(np.mean(val_score)),
                "validation_loss_mse": manifest["best_val_loss_mse"],
                "fit_window_count": int(ae_train_x.shape[0]),
                "selected": 0,
            }
        )
    _, _, best_grid, best_model, train_score, thr, manifest = min(candidates, key=lambda item: (item[0], item[1]))
    for row in selection_rows:
        if row["method"] == "Autoencoder" and row["grid_id"] == best_grid["grid_id"]:
            row["selected"] = 1
    payloads = {}
    for split_name, data in [("test_normal", test_normal), ("test_fault", test_fault)]:
        score = score_model(best_model, data["dynamic_x"])
        payloads[split_name] = {"score": score, "label": data["label"], "batch_id": data["batch_id"], "time_h": data["time_h"]}
        np.savez_compressed(dirs["predictions"] / f"anomaly_autoencoder_{split_name}.npz", **payloads[split_name])
    rows = anomaly_rows(
        method="Autoencoder",
        backend="pytorch_mlp_autoencoder",
        grid_id=best_grid["grid_id"],
        threshold=thr,
        train_scores=train_score,
        normal_payload=payloads["test_normal"],
        fault_payload=payloads["test_fault"],
        extra={
            "history_steps": int(cfg["anomaly_baselines"]["history_steps"]),
            "parameter_count": manifest["parameter_count"],
            "fit_window_count": int(ae_train_x.shape[0]),
            "threshold_source": "train_normal_fit_subset_quantile",
        },
    )
    log("Anomaly baselines: Autoencoder complete")
    return rows, selection_rows


def summarize_forecasting(metrics: pd.DataFrame) -> pd.DataFrame:
    df = metrics.copy()
    df["metric_value"] = pd.to_numeric(df["metric_value"], errors="coerce")
    df["horizon_steps"] = df["horizon_steps"].astype(int)
    sub = df[(df["target"] == "ALL") & (df["metric_name"].isin(["mean_nrmse_train_std", "mean_r2", "median_r2"]))]
    wide = sub.pivot_table(
        index=["split", "model", "model_family", "actual_backend", "horizon_steps", "seed", "grid_id"],
        columns="metric_name",
        values="metric_value",
        aggfunc="first",
    ).reset_index()
    grouped = (
        wide.groupby(["split", "model", "model_family", "actual_backend", "horizon_steps"], dropna=False)
        .agg(
            mean_nrmse=("mean_nrmse_train_std", "mean"),
            std_nrmse=("mean_nrmse_train_std", "std"),
            mean_r2=("mean_r2", "mean"),
            median_r2=("median_r2", "mean"),
            seed_count=("seed", "nunique"),
        )
        .reset_index()
    )
    return grouped


def make_phase16_assets(
    *,
    cfg: dict[str, Any],
    dirs: dict[str, Path],
    metric_rows_all: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    complexity_rows: list[dict[str, Any]],
    anomaly_metric_rows: list[dict[str, Any]],
    anomaly_selection_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    command = "python -u scripts/26_run_phase16_strong_baselines.py --config configs/model/phase16_strong_baselines.json --device auto"
    metric_df = pd.DataFrame(metric_rows_all)
    summary = summarize_forecasting(metric_df)
    fig52_csv = dirs["tables"] / "Fig52_phase16_strong_baseline_prediction_data.csv"
    summary.to_csv(fig52_csv, index=False)
    fig52_png = dirs["figures"] / "Fig52_phase16_strong_baseline_prediction.png"
    plot_df = summary[summary["split"].isin(["test_normal", "test_fault"])].copy()
    models = [m for m in ["random_forest", "xgboost", "xgboost_fallback_hgb", "gaussian_process", "patchtst"] if m in set(plot_df["model"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=False)
    for ax, split in zip(axes, ["test_normal", "test_fault"]):
        part = plot_df[plot_df["split"] == split]
        for model in models:
            mdf = part[part["model"] == model].sort_values("horizon_steps")
            if not mdf.empty:
                ax.plot(mdf["horizon_steps"], mdf["mean_nrmse"], marker="o", label=model.replace("_", " ").title())
        ax.set_title(split.replace("_", " ").title())
        ax.set_xlabel("Forecast Horizon (steps)")
        ax.set_ylabel("Mean nRMSE")
        polish_axis(ax)
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(fig52_png)
    plt.close(fig)
    save_explanation(
        dirs["explanations"] / "Fig52_phase16_strong_baseline_prediction_explanation.md",
        title="Phase16 ",
        figure_id="Fig52_phase16_strong_baseline_prediction",
        purpose=" RF、XGBoost/、GP  PatchTST 。",
        data_source="Phase16 ， val_normal ，test_normal/test_fault 。",
        design="， mean nRMSE，。",
        results="CSV 、、split 、、R2  seed 。",
        interpretation="。。",
        discussion=" XGBoost ， xgboost fallback hgb ， XGBoost。",
        limitations="； seed  CSV 。",
        caption="Strong forecasting baselines across normal and fault test splits.",
        command=command,
    )

    fig53_csv = dirs["tables"] / "Fig53_phase16_hyperparameter_selection_data.csv"
    pd.DataFrame(selection_rows).to_csv(fig53_csv, index=False)
    fig53_png = dirs["figures"] / "Fig53_phase16_hyperparameter_selection.png"
    selected = pd.DataFrame(selection_rows)
    if not selected.empty:
        selected_plot = selected[selected["selected"].astype(int) == 1].copy()
        nrmse_score = pd.to_numeric(selected_plot.get("validation_mean_nrmse_train_std"), errors="coerce")
        loss_score = pd.to_numeric(selected_plot.get("validation_loss_mse_scaled"), errors="coerce")
        selected_plot["score"] = nrmse_score.fillna(loss_score)
        selected_plot["label"] = selected_plot["model_family"].astype(str) + "-h" + selected_plot["horizon_steps"].astype(str)
        fig, ax = plt.subplots(figsize=(7.4, 3.0))
        ax.bar(selected_plot["label"], selected_plot["score"], color="#0072B2")
        ax.set_ylabel("Validation Selection Score")
        ax.set_xlabel("Selected Model-Horizon")
        ax.set_title("Validation-Selected Strong Baseline Configurations")
        ax.tick_params(axis="x", rotation=70)
        polish_axis(ax)
        fig.tight_layout()
        fig.savefig(fig53_png)
        plt.close(fig)
    save_explanation(
        dirs["explanations"] / "Fig53_phase16_hyperparameter_selection_explanation.md",
        title="Phase16 ",
        figure_id="Fig53_phase16_hyperparameter_selection",
        purpose="， val_normal 。",
        data_source="Phase16 tuning  validation score  selected 。",
        design="。",
        results="、 CSV。",
        interpretation="。",
        discussion="PatchTST  MSE， nRMSE；。",
        limitations="，。",
        caption="Validation-based hyperparameter selection for added strong baselines.",
        command=command,
    )

    anomaly_df = pd.DataFrame(anomaly_metric_rows)
    fig54_csv = dirs["tables"] / "Fig54_phase16_anomaly_baseline_detection_data.csv"
    anomaly_df.to_csv(fig54_csv, index=False)
    fig54_png = dirs["figures"] / "Fig54_phase16_anomaly_baseline_detection.png"
    if not anomaly_df.empty:
        pivot = anomaly_df[anomaly_df["metric_name"].isin(["auroc", "auprc", "false_alarm_rate", "fault_detection_rate"])].copy()
        pivot["metric_value"] = pd.to_numeric(pivot["metric_value"], errors="coerce")
        methods = list(dict.fromkeys(pivot["method"].tolist()))
        metrics = ["auroc", "auprc", "false_alarm_rate", "fault_detection_rate"]
        x = np.arange(len(methods))
        width = 0.18
        fig, ax = plt.subplots(figsize=(7.4, 3.0))
        for i, metric in enumerate(metrics):
            vals = []
            for method in methods:
                row = pivot[(pivot["method"] == method) & (pivot["metric_name"] == metric)]
                vals.append(float(row["metric_value"].iloc[0]) if not row.empty else math.nan)
            ax.bar(x + (i - 1.5) * width, vals, width=width, label=metric.upper())
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=35, ha="right")
        ax.set_ylabel("Detection Metric")
        ax.set_title("Unsupervised Anomaly Baseline Detection")
        ax.set_ylim(0, 1.05)
        ax.legend(ncol=2)
        polish_axis(ax)
        fig.tight_layout()
        fig.savefig(fig54_png)
        plt.close(fig)
    save_explanation(
        dirs["explanations"] / "Fig54_phase16_anomaly_baseline_detection_explanation.md",
        title="Phase16 ",
        figure_id="Fig54_phase16_anomaly_baseline_detection",
        purpose=" PCA-T2/Q、DPCA、Autoencoder、OCSVM、Isolation Forest 。",
        data_source=" train_normal ， train_normal ， test_normal/test_fault 。",
        design=" AUROC、AUPRC、false alarm rate  fault detection rate。",
        results="CSV  missed detection、batch-level F1  detection delay。",
        interpretation="AUROC/AUPRC ，false alarm ，fault detection rate 。",
        discussion="。",
        limitations="，，。",
        caption="Unsupervised anomaly detection baselines trained only on normal batches.",
        command=command,
    )

    fig55_csv = dirs["tables"] / "Fig55_phase16_baseline_complexity_data.csv"
    pd.DataFrame(complexity_rows).to_csv(fig55_csv, index=False)
    fig55_png = dirs["figures"] / "Fig55_phase16_baseline_complexity.png"
    comp = pd.DataFrame(complexity_rows)
    if not comp.empty:
        comp["training_seconds"] = pd.to_numeric(comp["training_seconds"], errors="coerce")
        comp_plot = comp.groupby("model", dropna=False)["training_seconds"].median().reset_index()
        fig, ax = plt.subplots(figsize=(6.6, 2.8))
        ax.bar(comp_plot["model"], comp_plot["training_seconds"], color="#009E73")
        ax.set_yscale("log")
        ax.set_ylabel("Median Training Time (s, log)")
        ax.set_xlabel("Model")
        ax.set_title("Strong Baseline Computational Cost")
        ax.tick_params(axis="x", rotation=35)
        polish_axis(ax)
        fig.tight_layout()
        fig.savefig(fig55_png)
        plt.close(fig)
    save_explanation(
        dirs["explanations"] / "Fig55_phase16_baseline_complexity_explanation.md",
        title="Phase16 ",
        figure_id="Fig55_phase16_baseline_complexity",
        purpose="、，。",
        data_source="Phase16 。",
        design="， log scale。",
        results=" CSV 、、。",
        interpretation="。",
        discussion="GP ， PatchTST ，。",
        limitations=" wall time ，。",
        caption="Computational cost of additional strong baselines.",
        command=command,
    )

    fig56_csv = dirs["tables"] / "Fig56_phase16_proposed_vs_strong_baselines_data.csv"
    comparison_rows = build_proposed_comparison(cfg, summary)
    pd.DataFrame(comparison_rows).to_csv(fig56_csv, index=False)
    fig56_png = dirs["figures"] / "Fig56_phase16_proposed_vs_strong_baselines.png"
    comp_df = pd.DataFrame(comparison_rows)
    if not comp_df.empty and "proposed_best_nrmse" in comp_df:
        fig, ax = plt.subplots(figsize=(6.8, 2.8))
        width = 0.36
        x = np.arange(comp_df.shape[0])
        ax.bar(x - width / 2, comp_df["strong_baseline_best_nrmse"], width=width, label="Best Strong Baseline")
        ax.bar(x + width / 2, comp_df["proposed_best_nrmse"], width=width, label="Best Phase15 Proposed")
        ax.set_xticks(x)
        ax.set_xticklabels(comp_df["split"].astype(str) + "-h" + comp_df["horizon_steps"].astype(str), rotation=55, ha="right")
        ax.set_ylabel("Mean nRMSE")
        ax.set_title("Proposed Method Versus Added Strong Baselines")
        ax.legend()
        polish_axis(ax)
        fig.tight_layout()
        fig.savefig(fig56_png)
        plt.close(fig)
    save_explanation(
        dirs["explanations"] / "Fig56_phase16_proposed_vs_strong_baselines_explanation.md",
        title="Phase16 ",
        figure_id="Fig56_phase16_proposed_vs_strong_baselines",
        purpose=" Phase15 ，。",
        data_source="Phase16  Phase15 Fig43 。",
        design=" split/horizon  nRMSE  Phase15  nRMSE。",
        results="CSV  horizon  best strong baseline、best Phase15 model 。",
        interpretation=" Phase15 ，；，。",
        discussion="。",
        limitations="Phase15  Phase16 ，。",
        caption="Comparison between added strong baselines and the best Phase15 proposed result.",
        command=command,
    )

    anomaly_selection_csv = dirs["tables"] / "Fig54_phase16_anomaly_selection_data.csv"
    pd.DataFrame(anomaly_selection_rows).to_csv(anomaly_selection_csv, index=False)
    asset_manifest = {
        "phase": PHASE,
        "experiment": EXPERIMENT,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tables": [str(fig52_csv), str(fig53_csv), str(fig54_csv), str(fig55_csv), str(fig56_csv), str(anomaly_selection_csv)],
        "figures": [str(fig52_png), str(fig53_png), str(fig54_png), str(fig55_png), str(fig56_png)],
        "explanations": [
            str(dirs["explanations"] / "Fig52_phase16_strong_baseline_prediction_explanation.md"),
            str(dirs["explanations"] / "Fig53_phase16_hyperparameter_selection_explanation.md"),
            str(dirs["explanations"] / "Fig54_phase16_anomaly_baseline_detection_explanation.md"),
            str(dirs["explanations"] / "Fig55_phase16_baseline_complexity_explanation.md"),
            str(dirs["explanations"] / "Fig56_phase16_proposed_vs_strong_baselines_explanation.md"),
        ],
        "no_error_bars_or_significance_marks": True,
    }
    manifest_path = dirs["manifests"] / "phase_16_strong_baselines_assets.json"
    manifest_path.write_text(json.dumps(asset_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return asset_manifest


def build_proposed_comparison(cfg: dict[str, Any], phase16_summary: pd.DataFrame) -> list[dict[str, Any]]:
    project_root = Path.cwd()
    phase15_path = project_root / cfg["reference_phase15_table"]
    if not phase15_path.exists():
        return [
            {
                "status": "phase15_reference_missing",
                "reference_phase15_table": str(phase15_path),
                "note": "Comparison was not generated because the Phase15 table is absent in this run package.",
            }
        ]
    phase15 = pd.read_csv(phase15_path)
    rows = []
    for split in ["test_normal", "test_fault"]:
        for horizon in [int(h) for h in cfg["horizon_steps"]]:
            strong = phase16_summary[
                (phase16_summary["split"] == split) & (phase16_summary["horizon_steps"] == horizon)
            ].copy()
            p15 = phase15[(phase15["split"] == split) & (phase15["horizon_steps"] == horizon)].copy()
            if strong.empty or p15.empty:
                continue
            strong_best = strong.sort_values("mean_nrmse").iloc[0]
            p15_best = p15.sort_values("mean_nrmse").iloc[0]
            rows.append(
                {
                    "phase": PHASE,
                    "split": split,
                    "horizon_steps": horizon,
                    "strong_baseline_best_model": strong_best["model"],
                    "strong_baseline_best_backend": strong_best["actual_backend"],
                    "strong_baseline_best_nrmse": float(strong_best["mean_nrmse"]),
                    "proposed_best_model": p15_best.get("model_label", p15_best.get("model", "")),
                    "proposed_best_nrmse": float(p15_best["mean_nrmse"]),
                    "relative_change_proposed_vs_strong_pct": 100.0
                    * (float(p15_best["mean_nrmse"]) - float(strong_best["mean_nrmse"]))
                    / max(float(strong_best["mean_nrmse"]), 1e-12),
                    "interpretation": "negative means Phase15 proposed is lower nRMSE than the added strong baseline",
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase16 additional strong baselines")
    parser.add_argument("--config", default="configs/model/phase16_strong_baselines.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--skip-patchtst", action="store_true")
    parser.add_argument("--skip-anomaly", action="store_true")
    args = parser.parse_args()

    project_root = Path.cwd()
    cfg = load_json(project_root / args.config)
    if args.skip_patchtst:
        cfg["patchtst"]["enabled"] = False
    if args.skip_anomaly:
        cfg["anomaly_baselines"]["enabled"] = False
    dirs = ensure_dirs(project_root / cfg["output_root"], project_root / cfg["paper_asset_root"])
    metadata = build_forecasting_metadata(cfg, project_root)
    sk = import_sklearn()

    all_metric_rows: list[dict[str, Any]] = []
    all_selection_rows: list[dict[str, Any]] = []
    all_complexity_rows: list[dict[str, Any]] = []
    anomaly_metric_rows: list[dict[str, Any]] = []
    anomaly_selection_rows: list[dict[str, Any]] = []

    classical_metrics, classical_selection, classical_complexity = run_classical_forecasting(
        cfg=cfg,
        dirs=dirs,
        metadata=metadata,
        sk=sk,
        n_jobs=int(args.n_jobs),
    )
    all_metric_rows.extend(classical_metrics)
    all_selection_rows.extend(classical_selection)
    all_complexity_rows.extend(classical_complexity)

    patch_metrics, patch_selection, patch_complexity = run_patchtst_forecasting(
        cfg=cfg,
        dirs=dirs,
        metadata=metadata,
        device_arg=args.device,
    )
    all_metric_rows.extend(patch_metrics)
    all_selection_rows.extend(patch_selection)
    all_complexity_rows.extend(patch_complexity)

    anomaly_metric_rows, anomaly_selection_rows = run_anomaly_baselines(
        cfg=cfg,
        dirs=dirs,
        metadata=metadata,
        sk=sk,
        device_arg=args.device,
        n_jobs=int(args.n_jobs),
    )

    metrics_csv = dirs["metrics"] / "phase16_forecasting_metrics.csv"
    selection_csv = dirs["metrics"] / "phase16_forecasting_selection.csv"
    complexity_csv = dirs["metrics"] / "phase16_complexity.csv"
    anomaly_csv = dirs["metrics"] / "phase16_anomaly_metrics.csv"
    anomaly_selection_csv = dirs["metrics"] / "phase16_anomaly_selection.csv"
    write_union_csv(metrics_csv, all_metric_rows)
    write_union_csv(selection_csv, all_selection_rows)
    write_union_csv(complexity_csv, all_complexity_rows)
    write_union_csv(anomaly_csv, anomaly_metric_rows)
    write_union_csv(anomaly_selection_csv, anomaly_selection_rows)

    assets = make_phase16_assets(
        cfg=cfg,
        dirs=dirs,
        metric_rows_all=all_metric_rows,
        selection_rows=all_selection_rows,
        complexity_rows=all_complexity_rows,
        anomaly_metric_rows=anomaly_metric_rows,
        anomaly_selection_rows=anomaly_selection_rows,
    )
    aggregate = {
        "phase": "16_strong_baselines",
        "config": str(project_root / args.config),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "forecast_metric_rows": len(all_metric_rows),
        "forecast_selection_rows": len(all_selection_rows),
        "complexity_rows": len(all_complexity_rows),
        "anomaly_metric_rows": len(anomaly_metric_rows),
        "anomaly_selection_rows": len(anomaly_selection_rows),
        "metrics_csv": str(metrics_csv),
        "selection_csv": str(selection_csv),
        "complexity_csv": str(complexity_csv),
        "anomaly_csv": str(anomaly_csv),
        "anomaly_selection_csv": str(anomaly_selection_csv),
        "asset_manifest": assets,
        "xgboost_available": importlib.util.find_spec("xgboost") is not None,
        "notes": cfg.get("notes", []),
    }
    (dirs["output"] / "aggregate_run_manifest.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

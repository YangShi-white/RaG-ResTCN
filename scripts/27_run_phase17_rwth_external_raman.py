#!/usr/bin/env python3
"""Run Phase 17 external RWTH Raman soft-sensing validation.

This script performs real cross-batch model fitting on the RWTH malic-acid
fermentation Raman dataset. It uses only F1-F3 for hyperparameter selection
and final training, then evaluates F4, F5, and inline spectra as external
tests. It does not synthesize results or loss curves.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import struct
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.signal import savgol_filter
from scipy.sparse.linalg import spsolve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fermnftp.metrics import write_csv  # noqa: E402
from fermnftp.plot_style import apply_ai_conference_style, polish_axis  # noqa: E402

apply_ai_conference_style(plt)


PHASE = "Phase 17"
EXPERIMENT = "RWTH external Raman soft-sensing validation"
EPS = 1e-12


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
        "cache": output_root / "cache",
        "models": output_root / "models",
        "predictions": output_root / "predictions",
        "metrics": output_root / "metrics",
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

、p 、。、、 seed  CSV ，。

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
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.neural_network import MLPRegressor
        from sklearn.svm import SVR
    except ModuleNotFoundError as exc:
        raise SystemExit("Phase17 requires scikit-learn in the sy520 environment.") from exc
    return {
        "PLSRegression": PLSRegression,
        "RandomForestRegressor": RandomForestRegressor,
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
        "SVR": SVR,
        "MLPRegressor": MLPRegressor,
        "ConvergenceWarning": ConvergenceWarning,
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def target_short_name(target: str) -> str:
    return (
        target.replace(" [g/L]", "")
        .replace("Malic acid", "Malic acid")
        .replace("Succinic acid", "Succinic acid")
    )


def split_short_name(split: str) -> str:
    return (
        split.replace("Fermentation ", "")
        .replace(" Validation", "")
        .replace(" Control", "")
        .replace(" with inline measurement", "Inline")
    )


def read_spc(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    if len(data) < 544:
        raise ValueError(f"SPC file too small: {path}")
    npts = struct.unpack("<i", data[4:8])[0]
    xfirst = struct.unpack("<d", data[8:16])[0]
    xlast = struct.unpack("<d", data[16:24])[0]
    nsub = struct.unpack("<i", data[24:28])[0]
    if npts <= 0 or nsub != 1:
        raise ValueError(f"Unsupported SPC header in {path}: npts={npts}, nsub={nsub}")
    needed = 544 + npts * 4
    if len(data) < needed:
        raise ValueError(f"SPC file has incomplete float32 block: {path}")
    x = np.linspace(xfirst, xlast, npts, dtype=np.float64)
    y = np.frombuffer(data, dtype="<f4", count=npts, offset=544).astype(np.float64)
    return x, y


def make_grid(cfg: dict[str, Any]) -> np.ndarray:
    grid_cfg = cfg["raman_grid"]
    start = float(grid_cfg["start_cm"])
    end = float(grid_cfg["end_cm"])
    step = float(grid_cfg["step_cm"])
    n = int(round((end - start) / step)) + 1
    return start + np.arange(n, dtype=np.float64) * step


def load_metadata(data_root: Path, target_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for table_path in sorted(data_root.rglob("DataTable*.xlsx")):
        if "Pure component" in str(table_path):
            continue
        df = pd.read_excel(table_path)
        file_col = df.columns[0]
        time_cols = [c for c in df.columns if str(c).startswith("Time")]
        if not time_cols:
            continue
        time_col = time_cols[0]
        spc_files = {p.name for p in table_path.parent.glob("*.spc")}
        for _, row in df.iterrows():
            filename = str(row[file_col]).split("#")[0]
            item: dict[str, Any] = {
                "batch": table_path.parent.name,
                "file": filename,
                "time_h": float(row[time_col]) if pd.notna(row[time_col]) else math.nan,
                "spc_path": str(table_path.parent / filename),
                "spc_exists": filename in spc_files,
            }
            for target in target_columns:
                item[target] = float(row[target]) if target in df.columns and pd.notna(row[target]) else math.nan
            rows.append(item)
    if not rows:
        raise RuntimeError(f"No RWTH DataTable files were parsed under {data_root}")
    return pd.DataFrame(rows)


def audit_dataset(meta: pd.DataFrame, data_root: Path, target_columns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch, sub in meta.groupby("batch", sort=True):
        item: dict[str, Any] = {
            "phase": PHASE,
            "experiment_name": EXPERIMENT,
            "dataset_root": str(data_root),
            "batch": batch,
            "spectra_rows": int(len(sub)),
            "spc_exists_rows": int(sub["spc_exists"].sum()),
            "time_min_h": float(np.nanmin(sub["time_h"])),
            "time_max_h": float(np.nanmax(sub["time_h"])),
        }
        for target in target_columns:
            item[f"{target}_valid"] = int(sub[target].notna().sum())
        rows.append(item)
    total: dict[str, Any] = {
        "phase": PHASE,
        "experiment_name": EXPERIMENT,
        "dataset_root": str(data_root),
        "batch": "ALL",
        "spectra_rows": int(len(meta)),
        "spc_exists_rows": int(meta["spc_exists"].sum()),
        "time_min_h": float(np.nanmin(meta["time_h"])),
        "time_max_h": float(np.nanmax(meta["time_h"])),
    }
    for target in target_columns:
        total[f"{target}_valid"] = int(meta[target].notna().sum())
    rows.append(total)
    return rows


def load_spectra_for_indices(meta: pd.DataFrame, indices: np.ndarray, grid: np.ndarray) -> np.ndarray:
    x_out = np.empty((len(indices), len(grid)), dtype=np.float32)
    for out_i, row_idx in enumerate(indices):
        row = meta.iloc[int(row_idx)]
        x, y = read_spc(Path(row["spc_path"]))
        x_out[out_i] = np.interp(grid, x, y).astype(np.float32)
        if (out_i + 1) % 1000 == 0:
            log(f"Loaded {out_i + 1}/{len(indices)} spectra")
    return x_out


def row_median_fill(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if np.isfinite(arr).all():
        return arr
    out = arr.copy()
    med = np.nanmedian(out, axis=1)
    med = np.where(np.isfinite(med), med, 0.0)
    bad = ~np.isfinite(out)
    out[bad] = np.take(med, np.where(bad)[0])
    return out


def row_snv(x: np.ndarray) -> np.ndarray:
    arr = row_median_fill(x)
    mean = np.mean(arr, axis=1, keepdims=True)
    std = np.std(arr, axis=1, keepdims=True)
    std = np.where(std > EPS, std, 1.0)
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
    if method == "snv":
        return row_snv(spectra).astype(np.float32)
    if method == "sg1_snv":
        derivative = savgol_filter(spectra, window_length=window, polyorder=polyorder, deriv=1, axis=1)
        return row_snv(derivative).astype(np.float32)
    if method == "sg2_snv":
        derivative = savgol_filter(spectra, window_length=window, polyorder=polyorder, deriv=2, axis=1)
        return row_snv(derivative).astype(np.float32)
    if method == "airpls_snv":
        corrected = airpls_correct_matrix(spectra, lam=float(cfg["airpls_lambda"]), n_iter=int(cfg["airpls_iterations"]))
        return row_snv(corrected).astype(np.float32)
    raise ValueError(f"Unsupported preprocessing method: {method}")


@dataclass
class PCATransformer:
    center: np.ndarray
    scale: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, n_components: int) -> "PCATransformer":
        center = np.mean(x, axis=0)
        scale = np.std(x, axis=0)
        scale = np.where(scale > EPS, scale, 1.0)
        z = (x - center) / scale
        _, singular_values, vt = np.linalg.svd(z, full_matrices=False)
        k = min(int(n_components), vt.shape[0])
        eigen = singular_values**2
        total = float(np.sum(eigen))
        ratio = eigen[:k] / total if total > EPS else np.zeros(k, dtype=np.float64)
        return cls(center=center, scale=scale, components=vt[:k], explained_variance_ratio=ratio)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.center) / self.scale) @ self.components.T


class PCRRidge:
    def __init__(self, alpha: float, n_components: int) -> None:
        self.alpha = float(alpha)
        self.n_components = int(n_components)
        self.pca: PCATransformer | None = None
        self.y_mean = 0.0
        self.beta: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PCRRidge":
        self.pca = PCATransformer.fit(x, self.n_components)
        t = self.pca.transform(x)
        self.y_mean = float(np.mean(y))
        yc = y - self.y_mean
        a = t.T @ t + self.alpha * np.eye(t.shape[1])
        self.beta = np.linalg.solve(a, t.T @ yc)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.pca is None or self.beta is None:
            raise RuntimeError("PCRRidge has not been fitted")
        return self.pca.transform(x) @ self.beta + self.y_mean


class PCAEstimator:
    def __init__(self, estimator: Any, n_components: int) -> None:
        self.estimator = estimator
        self.n_components = int(n_components)
        self.pca: PCATransformer | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PCAEstimator":
        self.pca = PCATransformer.fit(x, self.n_components)
        self.estimator.fit(self.pca.transform(x), y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("PCAEstimator has not been fitted")
        pred = self.estimator.predict(self.pca.transform(x))
        return np.asarray(pred, dtype=np.float64).reshape(-1)


class DirectEstimator:
    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    def fit(self, x: np.ndarray, y: np.ndarray) -> "DirectEstimator":
        self.estimator.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        pred = self.estimator.predict(x)
        return np.asarray(pred, dtype=np.float64).reshape(-1)


def xgboost_available() -> bool:
    return importlib.util.find_spec("xgboost") is not None


def make_model(
    family: str,
    params: dict[str, Any],
    seed: int,
    n_jobs: int,
    sk: dict[str, Any],
) -> tuple[Any, str]:
    if family == "PCR_Ridge":
        return PCRRidge(alpha=float(params["alpha"]), n_components=int(params["n_components"])), "numpy_pcr_ridge"
    if family == "PLS":
        estimator = sk["PLSRegression"](n_components=int(params["n_components"]), scale=True)
        return DirectEstimator(estimator), "sklearn_pls"
    if family == "SVR_RBF":
        estimator = sk["SVR"](C=float(params["C"]), epsilon=float(params["epsilon"]), gamma="scale", kernel="rbf")
        return PCAEstimator(estimator, int(params["n_components"])), "sklearn_svr"
    if family == "RandomForest":
        estimator = sk["RandomForestRegressor"](
            n_estimators=int(params["n_estimators"]),
            max_depth=params["max_depth"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            random_state=int(seed),
            n_jobs=int(n_jobs),
        )
        return PCAEstimator(estimator, int(params["n_components"])), "sklearn_random_forest"
    if family == "XGBoostOrHistGB":
        if xgboost_available():
            from xgboost import XGBRegressor

            estimator = XGBRegressor(
                objective="reg:squarederror",
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]),
                learning_rate=float(params["learning_rate"]),
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=int(seed),
                n_jobs=int(n_jobs),
                tree_method="hist",
                verbosity=0,
            )
            return PCAEstimator(estimator, int(params["n_components"])), "xgboost"
        estimator = sk["HistGradientBoostingRegressor"](
            max_iter=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            random_state=int(seed),
            early_stopping=True,
            validation_fraction=0.2,
        )
        return PCAEstimator(estimator, int(params["n_components"])), "sklearn_hist_gradient_boosting"
    if family == "MLP":
        estimator = sk["MLPRegressor"](
            hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
            alpha=float(params["alpha"]),
            max_iter=1200,
            early_stopping=True,
            validation_fraction=0.2,
            learning_rate_init=0.001,
            random_state=int(seed),
        )
        return PCAEstimator(estimator, int(params["n_components"])), "sklearn_mlp"
    raise ValueError(f"Unsupported model family: {family}")


def candidate_grid(family: str, spec: dict[str, Any]) -> list[dict[str, Any]]:
    if not spec.get("enabled", True):
        return []
    rows: list[dict[str, Any]] = []
    if family == "PCR_Ridge":
        for n in spec["n_components"]:
            for alpha in spec["alpha"]:
                rows.append({"n_components": int(n), "alpha": float(alpha)})
    elif family == "PLS":
        for n in spec["n_components"]:
            rows.append({"n_components": int(n)})
    elif family == "SVR_RBF":
        for n in spec["n_components"]:
            for c_value in spec["C"]:
                for epsilon in spec["epsilon"]:
                    rows.append({"n_components": int(n), "C": float(c_value), "epsilon": float(epsilon)})
    elif family == "RandomForest":
        for n in spec["n_components"]:
            for n_estimators in spec["n_estimators"]:
                for max_depth in spec["max_depth"]:
                    for leaf in spec["min_samples_leaf"]:
                        rows.append(
                            {
                                "n_components": int(n),
                                "n_estimators": int(n_estimators),
                                "max_depth": None if max_depth is None else int(max_depth),
                                "min_samples_leaf": int(leaf),
                            }
                        )
    elif family == "XGBoostOrHistGB":
        for n in spec["n_components"]:
            for n_estimators in spec["n_estimators"]:
                for max_depth in spec["max_depth"]:
                    for learning_rate in spec["learning_rate"]:
                        rows.append(
                            {
                                "n_components": int(n),
                                "n_estimators": int(n_estimators),
                                "max_depth": int(max_depth),
                                "learning_rate": float(learning_rate),
                            }
                        )
    elif family == "MLP":
        for n in spec["n_components"]:
            for hidden in spec["hidden_layer_sizes"]:
                for alpha in spec["alpha"]:
                    rows.append(
                        {
                            "n_components": int(n),
                            "hidden_layer_sizes": list(hidden),
                            "alpha": float(alpha),
                        }
                    )
    else:
        raise ValueError(f"Unsupported model family: {family}")
    return rows


def finite_or_raise(array: np.ndarray, context: str) -> None:
    if not np.isfinite(array).all():
        bad = int((~np.isfinite(array)).sum())
        raise RuntimeError(f"Non-finite values in {context}: {bad}")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, train_y: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(err**2) / denom) if denom > EPS else math.nan
    pearson = (
        float(np.corrcoef(y_true, y_pred)[0, 1])
        if len(y_true) >= 3 and np.std(y_true) > EPS and np.std(y_pred) > EPS
        else math.nan
    )
    train_std = float(np.std(train_y, ddof=1)) if len(train_y) > 1 else math.nan
    train_range = float(np.max(train_y) - np.min(train_y)) if len(train_y) > 1 else math.nan
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson_r": pearson,
        "nrmse_train_std": rmse / train_std if train_std > EPS else math.nan,
        "nrmse_train_range": rmse / train_range if train_range > EPS else math.nan,
        "train_target_std": train_std,
        "train_target_range": train_range,
    }


def evaluation_splits(cfg: dict[str, Any]) -> dict[str, list[str]]:
    splits: dict[str, list[str]] = {
        "Train F1-F3": list(cfg["train_batches"]),
    }
    for batch in cfg["external_batches"]:
        splits[batch] = [batch]
    splits["External F4+F5"] = list(cfg["external_batches"])
    for batch in cfg["domain_shift_batches"]:
        splits[batch] = [batch]
    return splits


def run_model_selection(
    *,
    cfg: dict[str, Any],
    meta_labeled: pd.DataFrame,
    x_by_method: dict[str, np.ndarray],
    seeds: list[int],
    n_jobs: int,
    sk: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    selection_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    selected_records: dict[tuple[str, str], dict[str, Any]] = {}
    batches = meta_labeled["batch"].to_numpy()
    train_batches = set(cfg["train_batches"])
    split_map = evaluation_splits(cfg)

    for target in cfg["target_columns"]:
        available = np.isfinite(meta_labeled[target].to_numpy(dtype=np.float64))
        train_mask = available & np.isin(batches, list(train_batches))
        if int(train_mask.sum()) < 10:
            log(f"Skipping target {target}: not enough training labels")
            continue
        y_all = meta_labeled[target].to_numpy(dtype=np.float64)
        y_train = y_all[train_mask]
        groups_train = batches[train_mask]
        fold_batches = sorted(set(groups_train))
        log(f"Target={target}: train labels={int(train_mask.sum())}, CV folds={fold_batches}")
        for seed in seeds:
            set_seed(seed)
            for family, spec in cfg["model_families"].items():
                candidates = candidate_grid(family, spec)
                if not candidates:
                    continue
                best: dict[str, Any] | None = None
                for method in cfg["preprocessing_methods"]:
                    z_all = x_by_method[method]
                    z_train = z_all[train_mask]
                    for params in candidates:
                        fold_scores: list[float] = []
                        fold_rmses: list[float] = []
                        fold_failures: list[str] = []
                        backend = ""
                        for held_batch in fold_batches:
                            fold_train = groups_train != held_batch
                            fold_val = groups_train == held_batch
                            if int(fold_train.sum()) < 5 or int(fold_val.sum()) < 2:
                                continue
                            x_tr = z_train[fold_train]
                            y_tr = y_train[fold_train]
                            x_va = z_train[fold_val]
                            y_va = y_train[fold_val]
                            try:
                                model, backend = make_model(family, params, seed, n_jobs, sk)
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore", sk["ConvergenceWarning"])
                                    model.fit(x_tr, y_tr)
                                pred = model.predict(x_va)
                                finite_or_raise(pred, f"{family}/{method}/{target}/fold={held_batch}")
                                mets = regression_metrics(y_va, pred, y_tr)
                                fold_scores.append(float(mets["nrmse_train_std"]))
                                fold_rmses.append(float(mets["rmse"]))
                            except Exception as exc:  # Records real failures instead of hiding them.
                                fold_failures.append(f"{held_batch}: {type(exc).__name__}: {exc}")
                                break
                        if fold_failures or not fold_scores:
                            row = {
                                "phase": PHASE,
                                "experiment_name": EXPERIMENT,
                                "target": target,
                                "seed": seed,
                                "model_family": family,
                                "backend": backend,
                                "preprocessing": method,
                                "params_json": json.dumps(params, sort_keys=True),
                                "cv_status": "failed",
                                "failure": " | ".join(fold_failures),
                            }
                            selection_rows.append(row)
                            continue
                        row = {
                            "phase": PHASE,
                            "experiment_name": EXPERIMENT,
                            "target": target,
                            "seed": seed,
                            "model_family": family,
                            "backend": backend,
                            "preprocessing": method,
                            "params_json": json.dumps(params, sort_keys=True),
                            "cv_status": "ok",
                            "cv_n_folds": len(fold_scores),
                            "cv_nrmse_train_std_mean": float(np.mean(fold_scores)),
                            "cv_nrmse_train_std_std": float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0,
                            "cv_rmse_mean": float(np.mean(fold_rmses)),
                            "cv_rmse_std": float(np.std(fold_rmses, ddof=1)) if len(fold_rmses) > 1 else 0.0,
                        }
                        row.update({f"param_{k}": v for k, v in params.items()})
                        selection_rows.append(row)
                        if best is None or row["cv_nrmse_train_std_mean"] < best["cv_nrmse_train_std_mean"]:
                            best = row
                if best is None:
                    log(f"No valid candidate for target={target}, family={family}, seed={seed}")
                    continue
                params = json.loads(best["params_json"])
                method = str(best["preprocessing"])
                z_all = x_by_method[method]
                model, backend = make_model(family, params, seed, n_jobs, sk)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", sk["ConvergenceWarning"])
                    model.fit(z_all[train_mask], y_train)
                for split, split_batches in split_map.items():
                    eval_mask = available & np.isin(batches, split_batches)
                    if int(eval_mask.sum()) < 3:
                        continue
                    pred = model.predict(z_all[eval_mask])
                    mets = regression_metrics(y_all[eval_mask], pred, y_train)
                    metric_row = {
                        "phase": PHASE,
                        "experiment_name": EXPERIMENT,
                        "target": target,
                        "seed": seed,
                        "model_family": family,
                        "backend": backend,
                        "preprocessing": method,
                        "params_json": best["params_json"],
                        "split": split,
                        "n_samples": int(eval_mask.sum()),
                        "cv_nrmse_train_std_mean": best["cv_nrmse_train_std_mean"],
                        "cv_rmse_mean": best["cv_rmse_mean"],
                    }
                    metric_row.update(mets)
                    metric_rows.append(metric_row)
                    eval_meta = meta_labeled.loc[eval_mask, ["batch", "file", "time_h"]].reset_index(drop=True)
                    for i, p in enumerate(pred):
                        prediction_rows.append(
                            {
                                "phase": PHASE,
                                "experiment_name": EXPERIMENT,
                                "target": target,
                                "seed": seed,
                                "model_family": family,
                                "backend": backend,
                                "preprocessing": method,
                                "split": split,
                                "batch": eval_meta.loc[i, "batch"],
                                "file": eval_meta.loc[i, "file"],
                                "time_h": float(eval_meta.loc[i, "time_h"]),
                                "y_true": float(y_all[eval_mask][i]),
                                "y_pred": float(p),
                            }
                        )
                key = (target, family)
                previous = selected_records.get(key)
                if previous is None or best["cv_nrmse_train_std_mean"] < previous["cv_nrmse_train_std_mean"]:
                    selected_records[key] = dict(best)
    return selection_rows, metric_rows, prediction_rows, selected_records


def aggregate_metrics(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(metric_rows)
    if df.empty:
        return []
    agg_rows: list[dict[str, Any]] = []
    group_cols = ["target", "model_family", "backend", "preprocessing", "split"]
    for keys, sub in df.groupby(group_cols, dropna=False, sort=True):
        row = {
            "phase": PHASE,
            "experiment_name": EXPERIMENT,
        }
        row.update(dict(zip(group_cols, keys)))
        row["n_seeds"] = int(sub["seed"].nunique())
        row["n_samples_per_seed_min"] = int(sub["n_samples"].min())
        row["n_samples_per_seed_max"] = int(sub["n_samples"].max())
        for metric in ["rmse", "mae", "r2", "pearson_r", "nrmse_train_std", "nrmse_train_range"]:
            values = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=np.float64)
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_std"] = float(np.nanstd(values, ddof=1)) if np.isfinite(values).sum() > 1 else 0.0
        agg_rows.append(row)
    return agg_rows


def best_family_by_target(selection_rows: list[dict[str, Any]]) -> dict[str, str]:
    df = pd.DataFrame([r for r in selection_rows if r.get("cv_status") == "ok"])
    if df.empty:
        return {}
    grouped = (
        df.groupby(["target", "model_family"], as_index=False)["cv_nrmse_train_std_mean"]
        .mean()
        .sort_values(["target", "cv_nrmse_train_std_mean"])
    )
    out: dict[str, str] = {}
    for target, sub in grouped.groupby("target", sort=True):
        out[str(target)] = str(sub.iloc[0]["model_family"])
    return out


def selected_best_record_for_target(
    target: str,
    family: str,
    selected_records: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any] | None:
    return selected_records.get((target, family))


def export_trajectory_predictions(
    *,
    cfg: dict[str, Any],
    meta_all: pd.DataFrame,
    x_train_labeled_by_method: dict[str, np.ndarray],
    meta_labeled: pd.DataFrame,
    selected_records: dict[tuple[str, str], dict[str, Any]],
    best_family: dict[str, str],
    grid: np.ndarray,
    n_jobs: int,
    sk: dict[str, Any],
) -> list[dict[str, Any]]:
    if not cfg.get("export_full_trajectory_predictions", True):
        return []
    rows: list[dict[str, Any]] = []
    batches = meta_labeled["batch"].to_numpy()
    train_mask_by_target: dict[str, np.ndarray] = {}
    for target in cfg["target_columns"]:
        train_mask_by_target[target] = np.isfinite(meta_labeled[target].to_numpy(dtype=np.float64)) & np.isin(
            batches, cfg["train_batches"]
        )

    batch_cache: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    for target in cfg["primary_targets_for_paper"]:
        family = best_family.get(target)
        if family is None:
            continue
        record = selected_best_record_for_target(target, family, selected_records)
        if record is None:
            continue
        params = json.loads(record["params_json"])
        method = str(record["preprocessing"])
        seed = int(record["seed"])
        train_mask = train_mask_by_target[target]
        y_train = meta_labeled.loc[train_mask, target].to_numpy(dtype=np.float64)
        model, backend = make_model(family, params, seed, n_jobs, sk)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", sk["ConvergenceWarning"])
            model.fit(x_train_labeled_by_method[method][train_mask], y_train)
        for batch in cfg["full_trajectory_batches"]:
            if batch not in batch_cache:
                batch_meta = meta_all[(meta_all["batch"] == batch) & meta_all["spc_exists"]].copy().reset_index(drop=True)
                log(f"Loading full trajectory spectra for {batch}: {len(batch_meta)} spectra")
                x_batch = load_spectra_for_indices(batch_meta, np.arange(len(batch_meta)), grid)
                batch_cache[batch] = (batch_meta, x_batch)
            batch_meta, x_batch_raw = batch_cache[batch]
            log(f"Preprocessing full trajectory batch={batch}, target={target}, method={method}")
            x_batch = preprocess_spectra(x_batch_raw, method, cfg)
            pred = model.predict(x_batch)
            for i, value in enumerate(pred):
                ref_value = batch_meta.loc[i, target] if target in batch_meta.columns else math.nan
                rows.append(
                    {
                        "phase": PHASE,
                        "experiment_name": EXPERIMENT,
                        "target": target,
                        "model_family": family,
                        "backend": backend,
                        "preprocessing": method,
                        "seed": seed,
                        "batch": batch,
                        "file": batch_meta.loc[i, "file"],
                        "time_h": float(batch_meta.loc[i, "time_h"]),
                        "y_pred": float(value),
                        "y_true": float(ref_value) if pd.notna(ref_value) else math.nan,
                    }
                )
    return rows


def plot_dataset_audit(rows: list[dict[str, Any]], cfg: dict[str, Any], figure_path: Path) -> None:
    df = pd.DataFrame([r for r in rows if r["batch"] != "ALL"])
    targets = cfg["target_columns"]
    x = np.arange(len(df))
    width = 0.18
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for i, target in enumerate(targets):
        ax.bar(x + (i - 1.5) * width, df[f"{target}_valid"], width=width, label=target_short_name(target))
    ax.set_xticks(x)
    ax.set_xticklabels([split_short_name(v) for v in df["batch"]], rotation=25, ha="right")
    ax.set_ylabel("Reference Samples")
    ax.set_title("RWTH Raman Dataset Reference Labels")
    ax.legend(ncol=2, loc="upper right")
    polish_axis(ax)
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)


def plot_model_selection(selection_rows: list[dict[str, Any]], cfg: dict[str, Any], figure_path: Path) -> None:
    df = pd.DataFrame([r for r in selection_rows if r.get("cv_status") == "ok"])
    if df.empty:
        return
    table = (
        df.groupby(["target", "model_family"], as_index=False)["cv_nrmse_train_std_mean"]
        .mean()
        .pivot(index="target", columns="model_family", values="cv_nrmse_train_std_mean")
    )
    table = table.reindex(cfg["target_columns"])
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    im = ax.imshow(table.to_numpy(dtype=np.float64), cmap="viridis_r", aspect="auto")
    ax.set_xticks(np.arange(len(table.columns)))
    ax.set_xticklabels(table.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(table.index)))
    ax.set_yticklabels([target_short_name(t) for t in table.index])
    ax.set_title("Training-Batch CV nRMSE")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Mean nRMSE")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = table.to_numpy()[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value > np.nanmedian(table.to_numpy()) else "black", fontsize=7)
    polish_axis(ax, grid_axis="none")
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)


def plot_external_performance(metric_rows: list[dict[str, Any]], best_family: dict[str, str], cfg: dict[str, Any], figure_path: Path) -> None:
    df = pd.DataFrame(metric_rows)
    if df.empty:
        return
    keep_rows = []
    splits = list(cfg["external_batches"]) + ["External F4+F5"] + list(cfg["domain_shift_batches"])
    for target, family in best_family.items():
        sub = df[(df["target"] == target) & (df["model_family"] == family) & (df["split"].isin(splits))]
        keep_rows.append(sub)
    keep = pd.concat(keep_rows, ignore_index=True) if keep_rows else pd.DataFrame()
    if keep.empty:
        return
    table = keep.groupby(["target", "split"], as_index=False)["r2"].mean().pivot(index="target", columns="split", values="r2")
    ordered_cols = [c for c in splits if c in table.columns]
    table = table.reindex(cfg["target_columns"])[ordered_cols]
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    im = ax.imshow(table.to_numpy(dtype=np.float64), cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(table.columns)))
    ax.set_xticklabels([split_short_name(c) for c in table.columns], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(table.index)))
    ax.set_yticklabels([target_short_name(t) for t in table.index])
    ax.set_title("External Batch R2 of Selected Models")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("R2")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = table.to_numpy()[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if abs(value) > 0.55 else "black", fontsize=7)
    polish_axis(ax, grid_axis="none")
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)


def plot_observed_predicted(prediction_rows: list[dict[str, Any]], best_family: dict[str, str], cfg: dict[str, Any], figure_path: Path) -> pd.DataFrame:
    df = pd.DataFrame(prediction_rows)
    if df.empty:
        return pd.DataFrame()
    kept = []
    for target in cfg["primary_targets_for_paper"]:
        family = best_family.get(target)
        if family:
            kept.append(df[(df["target"] == target) & (df["model_family"] == family) & (df["split"].isin(["External F4+F5"] + list(cfg["domain_shift_batches"])))])
    data = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()
    if data.empty:
        return pd.DataFrame()
    group_cols = ["target", "split", "batch", "file", "time_h", "y_true"]
    mean_pred = data.groupby(group_cols, as_index=False)["y_pred"].mean()
    targets = list(dict.fromkeys(mean_pred["target"]))
    fig, axes = plt.subplots(1, len(targets), figsize=(3.0 * len(targets), 3.0), squeeze=False)
    for ax, target in zip(axes[0], targets):
        sub = mean_pred[mean_pred["target"] == target]
        for split, ssub in sub.groupby("split", sort=True):
            ax.scatter(ssub["y_true"], ssub["y_pred"], s=20, alpha=0.8, label=split_short_name(split))
        lo = float(np.nanmin([sub["y_true"].min(), sub["y_pred"].min()]))
        hi = float(np.nanmax([sub["y_true"].max(), sub["y_pred"].max()]))
        ax.plot([lo, hi], [lo, hi], color="#303030", linewidth=1.0)
        ax.set_title(target_short_name(target))
        ax.set_xlabel("Observed (g/L)")
        ax.set_ylabel("Predicted (g/L)")
        polish_axis(ax)
    axes[0][-1].legend(loc="best")
    fig.suptitle("Observed vs Predicted External Raman Soft-Sensing", y=1.02)
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)
    return mean_pred


def plot_trajectory_predictions(trajectory_rows: list[dict[str, Any]], cfg: dict[str, Any], figure_path: Path) -> None:
    df = pd.DataFrame(trajectory_rows)
    if df.empty:
        return
    targets = [t for t in cfg["primary_targets_for_paper"] if t in set(df["target"])]
    batches = [b for b in cfg["full_trajectory_batches"] if b in set(df["batch"])]
    fig, axes = plt.subplots(len(targets), len(batches), figsize=(3.1 * len(batches), 2.35 * len(targets)), squeeze=False, sharex=False)
    for i, target in enumerate(targets):
        for j, batch in enumerate(batches):
            ax = axes[i][j]
            sub = df[(df["target"] == target) & (df["batch"] == batch)].sort_values("time_h")
            if sub.empty:
                ax.axis("off")
                continue
            ax.plot(sub["time_h"], sub["y_pred"], color="#0072B2", linewidth=1.2, label="Prediction")
            labeled = sub[np.isfinite(sub["y_true"])]
            if not labeled.empty:
                ax.scatter(labeled["time_h"], labeled["y_true"], color="#D55E00", s=14, label="Reference")
            if i == 0:
                ax.set_title(split_short_name(batch))
            if j == 0:
                ax.set_ylabel(f"{target_short_name(target)} (g/L)")
            ax.set_xlabel("Time (h)")
            polish_axis(ax)
    axes[0][-1].legend(loc="best")
    fig.suptitle("External Raman Soft-Sensing Trajectories", y=1.01)
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)


def plot_preprocessing_example(
    x_raw: np.ndarray,
    grid: np.ndarray,
    cfg: dict[str, Any],
    figure_path: Path,
) -> list[dict[str, Any]]:
    methods = ["snv", "sg1_snv", "sg2_snv", "airpls_snv"]
    rows: list[dict[str, Any]] = []
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.4), sharex=True)
    axes[0].plot(grid, x_raw, color="#303030", linewidth=0.9, label="Raw")
    axes[0].set_ylabel("Intensity")
    axes[0].set_title("Raw Raman Spectrum")
    axes[0].legend(loc="best")
    polish_axis(axes[0])
    for method in methods:
        y = preprocess_spectra(x_raw[None, :], method, cfg)[0]
        axes[1].plot(grid, y, linewidth=1.0, label=method.replace("_", "+"))
        for x_val, y_val in zip(grid, y):
            rows.append({"phase": PHASE, "experiment_name": EXPERIMENT, "raman_shift_cm": float(x_val), "method": method, "value": float(y_val)})
    axes[1].set_xlabel("Raman Shift (cm$^{-1}$)")
    axes[1].set_ylabel("Processed Value")
    axes[1].set_title("Chemometric Preprocessing")
    axes[1].legend(ncol=2, loc="best")
    polish_axis(axes[1])
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)
    return rows


def build_manifest(
    *,
    cfg: dict[str, Any],
    data_root: Path,
    dirs: dict[str, Path],
    command: str,
    assets: dict[str, str],
    audit_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    best_family: dict[str, str],
) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "experiment_name": EXPERIMENT,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_root": str(data_root),
        "command": command,
        "output_root": str(dirs["output"]),
        "paper_asset_root": str(dirs["figures"].parents[0]),
        "config": cfg,
        "assets": assets,
        "audit_total": next((r for r in audit_rows if r["batch"] == "ALL"), {}),
        "n_metric_rows": len(metric_rows),
        "best_family_by_target": best_family,
        "truthfulness_policy": [
            "No random or synthetic metrics are generated.",
            "F4, F5, and inline spectra are excluded from model selection.",
            "This phase is external Raman soft-sensing validation, not full FermNFTP multistep forecasting."
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase17 RWTH external Raman validation")
    parser.add_argument("--config", default="configs/model/phase17_rwth_external_raman.json")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--n-jobs", type=int, default=max(1, os.cpu_count() or 1))
    args = parser.parse_args()

    cfg_path = (PROJECT_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_json(cfg_path)
    data_root = Path(args.data_root) if args.data_root else PROJECT_ROOT / cfg["default_data_root"]
    data_root = data_root.resolve()
    if not data_root.exists():
        raise SystemExit(f"RWTH data root does not exist: {data_root}")

    dirs = ensure_dirs(PROJECT_ROOT / cfg["output_root"], PROJECT_ROOT / cfg["paper_asset_root"])
    command = " ".join(sys.argv)
    sk = import_sklearn()
    log(f"Using data root: {data_root}")
    log(f"Using n_jobs={args.n_jobs}")
    log(f"XGBoost available: {xgboost_available()}")

    grid = make_grid(cfg)
    meta_all = load_metadata(data_root, cfg["target_columns"])
    audit_rows = audit_dataset(meta_all, data_root, cfg["target_columns"])
    audit_csv = dirs["tables"] / "Fig57_phase17_rwth_dataset_audit_data.csv"
    write_union_csv(audit_csv, audit_rows)
    write_union_csv(dirs["output"] / "rwth_dataset_audit.csv", audit_rows)
    plot_dataset_audit(audit_rows, cfg, dirs["figures"] / "Fig57_phase17_rwth_dataset_audit.png")

    labeled_mask = np.zeros(len(meta_all), dtype=bool)
    for target in cfg["target_columns"]:
        labeled_mask |= np.isfinite(meta_all[target].to_numpy(dtype=np.float64))
    labeled_mask &= meta_all["spc_exists"].to_numpy(dtype=bool)
    meta_labeled = meta_all.loc[labeled_mask].copy().reset_index(drop=True)
    log(f"Labeled spectra with at least one reference target: {len(meta_labeled)}")
    x_labeled_raw = load_spectra_for_indices(meta_labeled, np.arange(len(meta_labeled)), grid)
    finite_or_raise(x_labeled_raw, "labeled raw spectra")

    log("Computing Raman preprocessing matrices for labeled spectra")
    x_by_method: dict[str, np.ndarray] = {}
    for method in cfg["preprocessing_methods"]:
        log(f"Preprocessing labeled spectra: {method}")
        x_by_method[method] = preprocess_spectra(x_labeled_raw, method, cfg)
        finite_or_raise(x_by_method[method], f"preprocessed labeled spectra {method}")

    example_rows = plot_preprocessing_example(
        x_labeled_raw[0].astype(np.float64),
        grid,
        cfg,
        dirs["figures"] / "Fig62_phase17_rwth_preprocessing_example.png",
    )
    preprocessing_example_csv = dirs["tables"] / "Fig62_phase17_rwth_preprocessing_example_data.csv"
    write_union_csv(preprocessing_example_csv, example_rows)

    selection_rows, metric_rows, prediction_rows, selected_records = run_model_selection(
        cfg=cfg,
        meta_labeled=meta_labeled,
        x_by_method=x_by_method,
        seeds=[int(s) for s in cfg["seeds"]],
        n_jobs=int(args.n_jobs),
        sk=sk,
    )
    selection_csv = dirs["tables"] / "Fig58_phase17_rwth_model_selection_data.csv"
    metric_csv = dirs["tables"] / "Fig59_phase17_rwth_external_performance_data.csv"
    prediction_csv = dirs["tables"] / "Fig60_phase17_rwth_observed_predicted_data.csv"
    write_union_csv(selection_csv, selection_rows)
    write_union_csv(metric_csv, metric_rows)
    write_union_csv(prediction_csv, prediction_rows)
    write_union_csv(dirs["output"] / "model_selection_all_candidates.csv", selection_rows)
    write_union_csv(dirs["output"] / "external_performance_seed_metrics.csv", metric_rows)
    write_union_csv(dirs["output"] / "external_predictions_labeled.csv", prediction_rows)

    aggregate_rows = aggregate_metrics(metric_rows)
    write_union_csv(dirs["output"] / "external_performance_aggregated.csv", aggregate_rows)

    best_family = best_family_by_target(selection_rows)
    log(f"Best family by target from F1-F3 CV: {best_family}")

    plot_model_selection(selection_rows, cfg, dirs["figures"] / "Fig58_phase17_rwth_model_selection.png")
    plot_external_performance(metric_rows, best_family, cfg, dirs["figures"] / "Fig59_phase17_rwth_external_performance.png")
    observed_predicted_mean = plot_observed_predicted(
        prediction_rows,
        best_family,
        cfg,
        dirs["figures"] / "Fig60_phase17_rwth_observed_predicted.png",
    )
    if not observed_predicted_mean.empty:
        write_union_csv(dirs["output"] / "external_predictions_mean_by_seed.csv", observed_predicted_mean.to_dict("records"))

    trajectory_rows = export_trajectory_predictions(
        cfg=cfg,
        meta_all=meta_all,
        x_train_labeled_by_method=x_by_method,
        meta_labeled=meta_labeled,
        selected_records=selected_records,
        best_family=best_family,
        grid=grid,
        n_jobs=int(args.n_jobs),
        sk=sk,
    )
    trajectory_csv = dirs["tables"] / "Fig61_phase17_rwth_trajectory_predictions_data.csv"
    write_union_csv(trajectory_csv, trajectory_rows)
    write_union_csv(dirs["output"] / "full_trajectory_predictions.csv", trajectory_rows)
    plot_trajectory_predictions(trajectory_rows, cfg, dirs["figures"] / "Fig61_phase17_rwth_trajectory_predictions.png")

    explanation_specs = [
        (
            "Fig57_phase17_rwth_dataset_audit_explanation.md",
            "Phase17 RWTH dataset audit ",
            "Fig57",
            " RWTH  Raman 、。",
            "RWTH Raman spectral files and reference concentration tables； DataTable Excel， .spc 。",
            " Glucose、Malic acid、Succinic acid  BTM ；，。",
            "， inline  Succinic acid。",
            " F4/F5 ，；，。",
            "，。",
            "，；。",
            "。",
        ),
        (
            "Fig58_phase17_rwth_model_selection_explanation.md",
            "Phase17 RWTH model selection ",
            "Fig58",
            " Raman  F1-F3 。",
            " F1、F2、F3  Raman ；F4、F5、inline 。",
            " CV nRMSE；。",
            " PCR/PLS/SVR/RF/XGBoost/MLP 。",
            "； CV 。",
            " strong baseline tuning 。",
            "CV  F1-F3 ，，。",
            " Raman 。",
        ),
        (
            "Fig59_phase17_rwth_external_performance_explanation.md",
            "Phase17 RWTH external performance ",
            "Fig59",
            " F1-F3  Raman  F4、F5  inline 。",
            " F1-F3； F4、F5  inline process spectra；CSV  seed RMSE、MAE、R2、Pearson r  nRMSE。",
            " R2； 1 ， 0 。",
            " Glucose、Malic acid、Succinic acid  Raman ， BTM  inline 。",
            " R2  Raman ；inline 。",
            " Raman soft-sensing external validation， FermNFTP 。",
            "RWTH ， FermNFTP 。",
            "。",
        ),
        (
            "Fig60_phase17_rwth_observed_predicted_explanation.md",
            "Phase17 RWTH observed-predicted ",
            "Fig60",
            "。",
            " F4/F5  inline ；CSV  observed  predicted。",
            "，；。",
            "， Raman ；。",
            " R2 。",
            "， inline 。",
            "； seed  CSV 。",
            "。",
        ),
        (
            "Fig61_phase17_rwth_trajectory_predictions_explanation.md",
            "Phase17 RWTH trajectory predictions ",
            "Fig61",
            " Raman ，。",
            " F1-F3 ； F4、F5  inline  Raman ；。",
            " Raman ，。",
            "，，。",
            " Raman ：。",
            " Raman soft-sensing，；。",
            "，。",
            " Raman 。",
        ),
        (
            "Fig62_phase17_rwth_preprocessing_example_explanation.md",
            "Phase17 RWTH preprocessing example ",
            "Fig62",
            " RWTH Raman 。",
            " RWTH  Raman ； SNV、SG 、SG  airPLS。",
            "，。",
            "SG derivative  airPLS ，SNV 。",
            " Raman ，。",
            " Fig58 ：。",
            "，； Fig58  CV 。",
            " Raman 。",
        ),
    ]
    for filename, title, fig_id, purpose, data_source, design, results, interpretation, discussion, limitations, caption in explanation_specs:
        save_explanation(
            dirs["explanations"] / filename,
            title=title,
            figure_id=fig_id,
            purpose=purpose,
            data_source=data_source,
            design=design,
            results=results,
            interpretation=interpretation,
            discussion=discussion,
            limitations=limitations,
            caption=caption,
            command=command,
        )

    assets = {
        "dataset_audit_csv": str(audit_csv),
        "model_selection_csv": str(selection_csv),
        "external_performance_csv": str(metric_csv),
        "observed_predicted_csv": str(prediction_csv),
        "trajectory_predictions_csv": str(trajectory_csv),
        "preprocessing_example_csv": str(preprocessing_example_csv),
    }
    manifest = build_manifest(
        cfg=cfg,
        data_root=data_root,
        dirs=dirs,
        command=command,
        assets=assets,
        audit_rows=audit_rows,
        metric_rows=metric_rows,
        best_family=best_family,
    )
    manifest_path = dirs["manifests"] / "phase17_rwth_external_raman_assets.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Phase17 completed. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

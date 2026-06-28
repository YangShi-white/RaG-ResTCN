"""Data loading utilities for FermNFTP forecasting experiments."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def split_batch_ids(split_csv: Path) -> dict[str, list[int]]:
    rows = read_csv_dicts(split_csv)
    splits: dict[str, list[int]] = {}
    for row in rows:
        splits.setdefault(row["split"], []).append(int(float(row["batch_id"])))
    return splits


def select_dense_targets(
    variable_roles: dict[str, Any], scaler: dict[str, Any], missing_ratio_max: float
) -> list[str]:
    selected = []
    for col in variable_roles["endogenous_states"]["columns"]:
        stats = scaler["process_columns"].get(col)
        if stats is None:
            continue
        if float(stats["missing_ratio"]) <= missing_ratio_max:
            selected.append(col)
    return selected


def column_center_scale(
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
    return np.asarray(centers, dtype=np.float32), np.asarray(scales, dtype=np.float32)


def standardize(values: np.ndarray, centers: np.ndarray, scales: np.ndarray) -> np.ndarray:
    out = (values - centers) / scales
    return np.where(np.isfinite(out), out, 0.0).astype(np.float32)


def inverse_standardize(values: np.ndarray, centers: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return (values * scales + centers).astype(np.float32)


def load_batch_npz(processed_root: Path, batch_id: int) -> dict[str, Any]:
    path = processed_root / "batches" / f"batch_{batch_id:03d}.npz"
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def build_forecasting_metadata(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    processed_root = project_root / config["processed_root"]
    variable_roles = load_json(project_root / config["variable_roles"])
    scaler = load_json(project_root / config["train_normal_scalers"])
    splits = split_batch_ids(project_root / config["split_csv"])
    first_batch = load_batch_npz(processed_root, splits["train_normal"][0])
    process_columns = [str(col) for col in first_batch["process_columns"]]
    exogenous_columns = variable_roles["exogenous_controls"]["columns"]
    target_columns = select_dense_targets(
        variable_roles, scaler, float(config["target_missing_ratio_max"])
    )
    history_columns = unique_in_order(exogenous_columns + target_columns)
    process_centers, process_scales = column_center_scale(
        scaler, process_columns, scale_key=config.get("input_scale_key", "robust_scale")
    )
    target_centers, target_scales = column_center_scale(
        scaler, target_columns, scale_key=config.get("target_scale_key", "std")
    )
    return {
        "processed_root": processed_root,
        "splits": splits,
        "process_columns": process_columns,
        "exogenous_columns": exogenous_columns,
        "target_columns": target_columns,
        "history_columns": history_columns,
        "process_centers": process_centers,
        "process_scales": process_scales,
        "target_centers": target_centers,
        "target_scales": target_scales,
    }


class ForecastWindowDataset:
    """Torch-compatible map-style dataset for process forecasting.

    The class avoids importing torch so the module can be syntax-checked in
    lightweight local environments. It returns NumPy arrays; PyTorch DataLoader
    converts them to tensors through the default collate function.
    """

    def __init__(
        self,
        processed_root: Path,
        batch_ids: list[int],
        history_steps: int,
        horizon_steps: int,
        process_columns: list[str],
        history_columns: list[str],
        exogenous_columns: list[str],
        target_columns: list[str],
        process_centers: np.ndarray,
        process_scales: np.ndarray,
        target_centers: np.ndarray,
        target_scales: np.ndarray,
        max_windows_per_batch: int | None = None,
    ) -> None:
        self.processed_root = processed_root
        self.batch_ids = batch_ids
        self.history_steps = history_steps
        self.horizon_steps = horizon_steps
        self.process_columns = process_columns
        self.history_columns = history_columns
        self.exogenous_columns = exogenous_columns
        self.target_columns = target_columns
        self.process_centers = process_centers
        self.process_scales = process_scales
        self.target_centers = target_centers
        self.target_scales = target_scales
        self.history_idx = [process_columns.index(col) for col in history_columns]
        self.exogenous_idx = [process_columns.index(col) for col in exogenous_columns]
        self.target_idx = [process_columns.index(col) for col in target_columns]
        self._batches: dict[int, dict[str, Any]] = {}
        self.index: list[tuple[int, int]] = []
        for batch_id in batch_ids:
            batch = self._load_batch(batch_id)
            process = batch["process"]
            candidates = []
            for t in range(history_steps - 1, process.shape[0] - horizon_steps):
                target_t = t + horizon_steps
                y = process[target_t, self.target_idx]
                y_now = process[t, self.target_idx]
                if np.isfinite(y).all() and np.isfinite(y_now).all():
                    candidates.append((batch_id, t))
            if max_windows_per_batch and len(candidates) > max_windows_per_batch:
                step = max(1, len(candidates) // max_windows_per_batch)
                candidates = candidates[::step][:max_windows_per_batch]
            self.index.extend(candidates)

    def _load_batch(self, batch_id: int) -> dict[str, Any]:
        if batch_id not in self._batches:
            batch = load_batch_npz(self.processed_root, batch_id)
            process = batch["process"].astype(np.float32)
            batch["process_float32"] = process
            batch["process_scaled"] = standardize(
                process, self.process_centers, self.process_scales
            )
            batch["time_h_float32"] = batch["time_h"].astype(np.float32)
            self._batches[batch_id] = batch
        return self._batches[batch_id]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, Any]:
        batch_id, t = self.index[item]
        batch = self._load_batch(batch_id)
        process = batch["process_float32"]
        process_scaled = batch["process_scaled"]
        time_h = batch["time_h_float32"]
        target_t = t + self.horizon_steps
        history = process_scaled[t - self.history_steps + 1 : t + 1, self.history_idx]
        future_exogenous = process_scaled[target_t, self.exogenous_idx]
        y_raw = process[target_t, self.target_idx]
        y_scaled = standardize(y_raw, self.target_centers, self.target_scales)
        y_current_raw = process[t, self.target_idx]
        return {
            "history": history.astype(np.float32),
            "future_exogenous": future_exogenous.astype(np.float32),
            "target_scaled": y_scaled.astype(np.float32),
            "target_raw": y_raw.astype(np.float32),
            "current_target_raw": y_current_raw.astype(np.float32),
            "batch_id": np.int64(batch_id),
            "source_t": np.int64(t),
            "target_t": np.int64(target_t),
            "target_time_h": np.float32(time_h[target_t]),
        }

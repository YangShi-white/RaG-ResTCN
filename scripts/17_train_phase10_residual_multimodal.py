#!/usr/bin/env python3
"""Train Phase 10 residual target-adaptive multimodal FermNFTP models."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fermnftp.data import inverse_standardize, load_json  # noqa: E402
from fermnftp.metrics import metric_rows, regression_metrics, write_csv  # noqa: E402


PHASE = "Phase 10"
EXPERIMENT = "Residual target-adaptive multimodal FermNFTP"


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def import_torch() -> tuple[Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is required. Run Phase 10 on the HPC environment.") from exc
    return torch, nn, DataLoader, Dataset


def set_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def split_map(path: Path) -> dict[int, dict[str, Any]]:
    out = {}
    for row in read_csv_dicts(path):
        batch_id = int(float(row["batch_id"]))
        out[batch_id] = {
            "split": row["split"],
            "fault_label": int(float(row["fault_label"])),
            "n_rows": int(float(row["n_rows"])),
            "duration_h": float(row["duration_h"]),
        }
    return out


def split_batch_ids(split_info: dict[int, dict[str, Any]]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for batch_id, info in sorted(split_info.items()):
        out.setdefault(info["split"], []).append(batch_id)
    return out


def load_batch(processed_root: Path, batch_id: int) -> dict[str, Any]:
    data = np.load(processed_root / "batches" / f"batch_{batch_id:03d}.npz", allow_pickle=True)
    return {key: data[key] for key in data.files}


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
        if stats is not None and float(stats["missing_ratio"]) <= missing_ratio_max:
            selected.append(col)
    return selected


def column_stats(scaler: dict[str, Any], columns: list[str], scale_key: str) -> tuple[np.ndarray, np.ndarray]:
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


def median_dt_h(processed_root: Path, batch_id: int) -> float:
    time_h = load_batch(processed_root, batch_id)["time_h"].astype(np.float64)
    diffs = np.diff(time_h)
    finite = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(np.median(finite)) if finite.size else 0.2


def build_metadata(cfg: dict[str, Any], project_root: Path) -> dict[str, Any]:
    processed_root = project_root / cfg["processed_root"]
    split_info = split_map(project_root / cfg["split_csv"])
    splits = split_batch_ids(split_info)
    variable_roles = load_json(project_root / cfg["variable_roles"])
    scaler = load_json(project_root / cfg["train_normal_scalers"])
    first_batch = load_batch(processed_root, splits["train_normal"][0])
    process_columns = [str(col) for col in first_batch["process_columns"]]
    exogenous_columns = variable_roles["exogenous_controls"]["columns"]
    target_columns = select_dense_targets(variable_roles, scaler, float(cfg["target_missing_ratio_max"]))
    history_columns = unique_in_order(exogenous_columns + target_columns)
    process_centers, process_scales = column_stats(
        scaler, process_columns, cfg.get("input_scale_key", "robust_scale")
    )
    target_centers, target_scales = column_stats(
        scaler, target_columns, cfg.get("target_scale_key", "std")
    )
    return {
        "processed_root": processed_root,
        "split_info": split_info,
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


def build_raman_sample_metadata(processed_root: Path, split_info: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = read_csv_dicts(processed_root / "raman" / "raman_sample_rows.csv")
    batch_cache: dict[int, dict[str, Any]] = {}
    out = []
    for row in rows:
        batch_id = int(float(row["batch_id"]))
        if batch_id not in batch_cache:
            batch_cache[batch_id] = load_batch(processed_root, batch_id)
        time_h = float(row["time_h"])
        batch_time = batch_cache[batch_id]["time_h"].astype(np.float64)
        source_t = int(np.nanargmin(np.abs(batch_time - time_h)))
        out.append(
            {
                "sample_index": int(float(row["sample_index"])),
                "batch_id": batch_id,
                "split": split_info[batch_id]["split"],
                "fault_label": split_info[batch_id]["fault_label"],
                "time_h": time_h,
                "source_t": source_t,
            }
        )
    return out


def stack_or_empty(values: list[Any], shape_tail: tuple[int, ...], dtype: Any) -> np.ndarray:
    if values:
        return np.asarray(values, dtype=dtype)
    return np.empty((0, *shape_tail), dtype=dtype)


def build_design(
    *,
    cfg: dict[str, Any],
    metadata: dict[str, Any],
    sample_metadata: list[dict[str, Any]],
    raman_scores_by_method: dict[str, np.ndarray],
) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    history_idx = [metadata["process_columns"].index(col) for col in metadata["history_columns"]]
    exo_idx = [metadata["process_columns"].index(col) for col in metadata["exogenous_columns"]]
    target_idx = [metadata["process_columns"].index(col) for col in metadata["target_columns"]]
    history_steps = int(cfg["history_steps"])
    split_names = ["train_normal", "val_normal", "test_normal", "test_fault"]
    design: dict[int, dict[str, dict[str, list[Any]]]] = {
        int(h): {
            split: {
                "history": [],
                "future_exogenous": [],
                "process_x": [],
                "y_scaled": [],
                "y_raw": [],
                "batch_id": [],
                "sample_index": [],
                "source_t": [],
                "target_t": [],
                "target_time_h": [],
            }
            for split in split_names
        }
        for h in cfg["horizon_steps"]
    }
    for horizon in design:
        for split in split_names:
            for method in raman_scores_by_method:
                design[horizon][split][f"raman_{method}"] = []

    batch_cache: dict[int, dict[str, Any]] = {}
    for item in sample_metadata:
        split = item["split"]
        if split not in split_names:
            continue
        batch_id = int(item["batch_id"])
        if batch_id not in batch_cache:
            batch = load_batch(metadata["processed_root"], batch_id)
            process = batch["process"].astype(np.float32)
            batch["process_float32"] = process
            batch["process_scaled"] = standardize(
                process, metadata["process_centers"], metadata["process_scales"]
            )
            batch["time_h_float32"] = batch["time_h"].astype(np.float32)
            batch_cache[batch_id] = batch
        batch = batch_cache[batch_id]
        process = batch["process_float32"]
        process_scaled = batch["process_scaled"]
        source_t = int(item["source_t"])
        if source_t < history_steps - 1:
            continue
        for horizon in design:
            target_t = source_t + horizon
            if target_t >= process.shape[0]:
                continue
            y_raw = process[target_t, target_idx]
            if not np.isfinite(y_raw).all():
                continue
            history = process_scaled[source_t - history_steps + 1 : source_t + 1, history_idx]
            future_exogenous = process_scaled[target_t, exo_idx]
            process_x = np.concatenate([history.reshape(-1), future_exogenous.reshape(-1)]).astype(np.float32)
            y_scaled = standardize(y_raw, metadata["target_centers"], metadata["target_scales"])
            store = design[horizon][split]
            store["history"].append(history.astype(np.float32))
            store["future_exogenous"].append(future_exogenous.astype(np.float32))
            store["process_x"].append(process_x)
            store["y_scaled"].append(y_scaled.astype(np.float32))
            store["y_raw"].append(y_raw.astype(np.float32))
            store["batch_id"].append(batch_id)
            store["sample_index"].append(int(item["sample_index"]))
            store["source_t"].append(source_t)
            store["target_t"].append(target_t)
            store["target_time_h"].append(float(batch["time_h_float32"][target_t]))
            for method, scores in raman_scores_by_method.items():
                store[f"raman_{method}"].append(scores[int(item["sample_index"])].astype(np.float32))

    out: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    history_dim = len(metadata["history_columns"])
    exo_dim = len(metadata["exogenous_columns"])
    target_dim = len(metadata["target_columns"])
    process_x_dim = history_steps * history_dim + exo_dim
    raman_dim = next(iter(raman_scores_by_method.values())).shape[1]
    for horizon, splits in design.items():
        out[horizon] = {}
        for split, values in splits.items():
            out[horizon][split] = {
                "history": stack_or_empty(values["history"], (history_steps, history_dim), np.float32),
                "future_exogenous": stack_or_empty(values["future_exogenous"], (exo_dim,), np.float32),
                "process_x": stack_or_empty(values["process_x"], (process_x_dim,), np.float32),
                "y_scaled": stack_or_empty(values["y_scaled"], (target_dim,), np.float32),
                "y_raw": stack_or_empty(values["y_raw"], (target_dim,), np.float32),
                "batch_id": np.asarray(values["batch_id"], dtype=np.int64),
                "sample_index": np.asarray(values["sample_index"], dtype=np.int64),
                "source_t": np.asarray(values["source_t"], dtype=np.int64),
                "target_t": np.asarray(values["target_t"], dtype=np.int64),
                "target_time_h": np.asarray(values["target_time_h"], dtype=np.float32),
            }
            for method in raman_scores_by_method:
                out[horizon][split][f"raman_{method}"] = stack_or_empty(
                    values[f"raman_{method}"], (raman_dim,), np.float32
                )
    return out


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    x_aug = np.hstack([np.ones((x.shape[0], 1), dtype=np.float64), x.astype(np.float64)])
    penalty = np.eye(x_aug.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    lhs = x_aug.T @ x_aug + float(alpha) * penalty
    rhs = x_aug.T @ y.astype(np.float64)
    return np.linalg.solve(lhs, rhs).astype(np.float32)


def predict_ridge(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    x_aug = np.hstack([np.ones((x.shape[0], 1), dtype=np.float64), x.astype(np.float64)])
    return (x_aug @ coef.astype(np.float64)).astype(np.float32)


def mean_scaled_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0))))


def add_common_fields(rows: list[dict[str, Any]], **fields: Any) -> list[dict[str, Any]]:
    for row in rows:
        row.update(fields)
    return rows


def prepare_ridge_baselines(
    *,
    cfg: dict[str, Any],
    project_root: Path,
    metadata: dict[str, Any],
    design: dict[int, dict[str, dict[str, np.ndarray]]],
) -> dict[int, dict[str, Any]]:
    output_root = project_root / cfg["output_root"]
    baseline_dir = output_root / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    all_metric_rows: list[dict[str, Any]] = []
    baseline_by_horizon: dict[int, dict[str, Any]] = {}
    dt_h = median_dt_h(metadata["processed_root"], metadata["splits"]["train_normal"][0])
    for horizon in [int(h) for h in cfg["horizon_steps"]]:
        train = design[horizon]["train_normal"]
        val = design[horizon]["val_normal"]
        best_alpha = None
        best_val = math.inf
        best_coef = None
        for alpha in cfg["ridge_alpha_grid"]:
            coef = fit_ridge(train["process_x"], train["y_scaled"], float(alpha))
            val_pred = predict_ridge(val["process_x"], coef)
            score = mean_scaled_rmse(val["y_scaled"], val_pred)
            if score < best_val:
                best_val = score
                best_alpha = float(alpha)
                best_coef = coef
        assert best_alpha is not None and best_coef is not None
        train_pred = predict_ridge(train["process_x"], best_coef)
        residual_scale = np.std(train["y_scaled"] - train_pred, axis=0)
        residual_scale = np.where(np.isfinite(residual_scale) & (residual_scale > 1e-6), residual_scale, 1.0).astype(np.float32)
        horizon_payload: dict[str, Any] = {
            "alpha": best_alpha,
            "val_mean_scaled_rmse": best_val,
            "coef": best_coef,
            "residual_scale": residual_scale,
            "splits": {},
        }
        horizon_h = horizon * dt_h
        for split in ["train_normal", "val_normal", "test_normal", "test_fault"]:
            split_data = design[horizon][split]
            pred_scaled = predict_ridge(split_data["process_x"], best_coef)
            pred_raw = inverse_standardize(pred_scaled, metadata["target_centers"], metadata["target_scales"])
            metrics = regression_metrics(split_data["y_raw"], pred_raw, metadata["target_scales"])
            rows = metric_rows(
                phase=PHASE,
                experiment_name=EXPERIMENT,
                split=split,
                model="ridge_only",
                forecast_mode="controlled_ridge",
                horizon_steps=horizon,
                horizon_h=horizon_h,
                target_columns=metadata["target_columns"],
                metrics=metrics,
            )
            add_common_fields(
                rows,
                raman_method="none",
                residual_model="ridge_only",
                baseline_alpha=best_alpha,
                residual_scale_source="train_normal_residual_std",
                seed="not_applicable",
            )
            all_metric_rows.extend(rows)
            horizon_payload["splits"][split] = {
                "pred_scaled": pred_scaled.astype(np.float32),
                "pred_raw": pred_raw.astype(np.float32),
            }
        np.savez_compressed(
            baseline_dir / f"ridge_baseline_h{horizon}.npz",
            coef=best_coef.astype(np.float32),
            residual_scale=residual_scale,
            alpha=np.asarray([best_alpha], dtype=np.float32),
            **{
                f"{split}_pred_scaled": horizon_payload["splits"][split]["pred_scaled"]
                for split in horizon_payload["splits"]
            },
        )
        baseline_by_horizon[horizon] = horizon_payload
        log(f"Prepared ridge baseline h{horizon}: alpha={best_alpha} val_mean_scaled_rmse={best_val:.6f}")
    write_csv(baseline_dir / "ridge_baseline_metrics.csv", all_metric_rows)
    manifest = {
        "phase": "10_residual_multimodal_baselines",
        "baseline_model": "controlled_ridge",
        "alpha_grid": cfg["ridge_alpha_grid"],
        "selection_split": "val_normal",
        "selection_metric": "mean_scaled_rmse",
        "residual_scale_source": "train_normal residual standard deviation",
        "horizons": {
            str(h): {
                "alpha": baseline_by_horizon[h]["alpha"],
                "val_mean_scaled_rmse": baseline_by_horizon[h]["val_mean_scaled_rmse"],
                "residual_scale": baseline_by_horizon[h]["residual_scale"].tolist(),
            }
            for h in baseline_by_horizon
        },
        "metrics_csv": str(baseline_dir / "ridge_baseline_metrics.csv"),
    }
    (baseline_dir / "ridge_baseline_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return baseline_by_horizon


def build_dataset_class(Dataset: Any) -> Any:
    class ResidualWindowDataset(Dataset):
        def __init__(
            self,
            *,
            split_data: dict[str, np.ndarray],
            baseline_pred_scaled: np.ndarray,
            residual_scale: np.ndarray,
            raman_method: str,
            max_samples: int | None = None,
        ) -> None:
            self.split_data = split_data
            self.baseline_pred_scaled = baseline_pred_scaled.astype(np.float32)
            self.residual_scale = residual_scale.astype(np.float32)
            self.raman_key = f"raman_{raman_method}" if raman_method != "none" else None
            self.indices = np.arange(split_data["history"].shape[0], dtype=np.int64)
            if max_samples is not None and self.indices.size > max_samples:
                step = max(1, self.indices.size // max_samples)
                self.indices = self.indices[::step][:max_samples]

        def __len__(self) -> int:
            return int(self.indices.size)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            i = int(self.indices[idx])
            y_scaled = self.split_data["y_scaled"][i]
            baseline_scaled = self.baseline_pred_scaled[i]
            residual_z = (y_scaled - baseline_scaled) / self.residual_scale
            if self.raman_key is None:
                raman = np.zeros((1,), dtype=np.float32)
            else:
                raman = self.split_data[self.raman_key][i].astype(np.float32)
            return {
                "history": self.split_data["history"][i].astype(np.float32),
                "future_exogenous": self.split_data["future_exogenous"][i].astype(np.float32),
                "raman": raman.astype(np.float32),
                "baseline_scaled": baseline_scaled.astype(np.float32),
                "target_residual_z": residual_z.astype(np.float32),
                "target_raw": self.split_data["y_raw"][i].astype(np.float32),
                "target_scaled": y_scaled.astype(np.float32),
                "batch_id": np.int64(self.split_data["batch_id"][i]),
                "sample_index": np.int64(self.split_data["sample_index"][i]),
                "target_time_h": np.float32(self.split_data["target_time_h"][i]),
            }

    return ResidualWindowDataset


def build_model_class(torch: Any, nn: Any) -> Any:
    class CausalConvBlock(nn.Module):
        def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
            super().__init__()
            self.padding = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(channels, channels, kernel_size, padding=self.padding, dilation=dilation)
            self.norm = nn.BatchNorm1d(channels)
            self.act = nn.GELU()
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Any) -> Any:
            y = self.conv(x)
            if self.padding:
                y = y[..., : -self.padding]
            return x + self.dropout(self.act(self.norm(y)))

    class ResidualTargetAdaptiveTCN(nn.Module):
        def __init__(
            self,
            residual_model: str,
            input_dim: int,
            exogenous_dim: int,
            raman_dim: int,
            output_dim: int,
            hidden_dim: int,
            num_layers: int,
            kernel_size: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.residual_model = residual_model
            self.output_dim = output_dim
            self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
            self.blocks = nn.Sequential(
                *[
                    CausalConvBlock(hidden_dim, kernel_size, 2**layer, dropout)
                    for layer in range(num_layers)
                ]
            )
            self.raman_embed = nn.Sequential(
                nn.LayerNorm(raman_dim),
                nn.Linear(raman_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            base_dim = hidden_dim + exogenous_dim + output_dim
            self.process_head = nn.Sequential(
                nn.LayerNorm(base_dim),
                nn.Linear(base_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
            self.naive_head = nn.Sequential(
                nn.LayerNorm(base_dim + hidden_dim),
                nn.Linear(base_dim + hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
            self.raman_residual_head = nn.Sequential(
                nn.LayerNorm(base_dim + hidden_dim),
                nn.Linear(base_dim + hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
            self.global_gate = nn.Sequential(
                nn.LayerNorm(base_dim + hidden_dim),
                nn.Linear(base_dim + hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
            self.target_gate = nn.Sequential(
                nn.LayerNorm(base_dim + hidden_dim),
                nn.Linear(base_dim + hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
                nn.Sigmoid(),
            )
            self.raman_token_proj = nn.Linear(1, hidden_dim)
            self.raman_token_position = nn.Parameter(torch.zeros(1, max(128, raman_dim + 1), hidden_dim))
            self.attention_query = nn.Linear(base_dim, hidden_dim)
            self.raman_attention = nn.MultiheadAttention(
                hidden_dim,
                num_heads=4 if hidden_dim % 4 == 0 else 1,
                dropout=dropout,
                batch_first=True,
            )
            self.attention_head = nn.Sequential(
                nn.LayerNorm(base_dim + hidden_dim),
                nn.Linear(base_dim + hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(
            self,
            history: Any,
            future_exogenous: Any,
            raman: Any,
            baseline_scaled: Any,
            return_aux: bool = False,
        ) -> Any:
            x = self.input_proj(history.transpose(1, 2))
            encoded = self.blocks(x)[..., -1]
            base = torch.cat([encoded, future_exogenous, baseline_scaled], dim=-1)
            aux: dict[str, Any] = {}
            if self.residual_model == "residual_process_tcn":
                residual_z = self.process_head(base)
            else:
                raman_embed = self.raman_embed(raman)
                fused_input = torch.cat([base, raman_embed], dim=-1)
                if self.residual_model == "residual_naive_raman_tcn":
                    residual_z = self.naive_head(fused_input)
                elif self.residual_model == "residual_global_gate_tcn":
                    process_residual = self.process_head(base)
                    raman_residual = self.raman_residual_head(fused_input)
                    gate = self.global_gate(fused_input)
                    residual_z = process_residual + gate * raman_residual
                    aux["global_gate"] = gate
                    aux["gate_mean"] = gate.reshape(-1)
                elif self.residual_model == "residual_attention_tcn":
                    token_count = int(raman.shape[1])
                    raman_tokens = self.raman_token_proj(raman.unsqueeze(-1))
                    raman_tokens = raman_tokens + self.raman_token_position[:, :token_count, :]
                    query = self.attention_query(base).unsqueeze(1)
                    attended, weights = self.raman_attention(
                        query, raman_tokens, raman_tokens, need_weights=True
                    )
                    attended = attended.squeeze(1)
                    residual_z = self.attention_head(torch.cat([base, attended], dim=-1))
                    weights = weights.squeeze(1).clamp_min(1e-12)
                    entropy = -(weights * weights.log()).sum(dim=-1) / math.log(max(token_count, 2))
                    aux["attention_entropy"] = entropy
                elif self.residual_model == "residual_target_gate_tcn":
                    process_residual = self.process_head(base)
                    raman_residual = self.raman_residual_head(fused_input)
                    gate = self.target_gate(fused_input)
                    residual_z = process_residual + gate * raman_residual
                    aux["target_gate"] = gate
                    aux["gate_mean"] = gate.mean(dim=-1)
                else:
                    raise ValueError(f"Unknown residual_model: {self.residual_model}")
            if return_aux:
                return residual_z, aux
            return residual_z

    return ResidualTargetAdaptiveTCN


def tensor_to_numpy(tensor: Any) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def evaluate(
    *,
    model: Any,
    loader: Any,
    device: Any,
    torch: Any,
    metadata: dict[str, Any],
    residual_model: str,
    raman_method: str,
    horizon_steps: int,
    horizon_h: float,
    residual_scale: np.ndarray,
    split: str,
    baseline_alpha: float,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, float]]:
    model.eval()
    y_true_blocks = []
    y_pred_blocks = []
    baseline_blocks = []
    batch_ids = []
    sample_indices = []
    target_times = []
    aux_blocks: dict[str, list[np.ndarray]] = {}
    residual_scale_np = residual_scale.astype(np.float32)
    with torch.no_grad():
        for batch in loader:
            history = batch["history"].to(device=device, dtype=torch.float32)
            future_exogenous = batch["future_exogenous"].to(device=device, dtype=torch.float32)
            raman = batch["raman"].to(device=device, dtype=torch.float32)
            baseline_scaled = batch["baseline_scaled"].to(device=device, dtype=torch.float32)
            residual_z, aux = model(history, future_exogenous, raman, baseline_scaled, return_aux=True)
            pred_scaled = tensor_to_numpy(baseline_scaled) + tensor_to_numpy(residual_z) * residual_scale_np
            pred_raw = inverse_standardize(pred_scaled, metadata["target_centers"], metadata["target_scales"])
            baseline_raw = inverse_standardize(
                tensor_to_numpy(baseline_scaled), metadata["target_centers"], metadata["target_scales"]
            )
            y_pred_blocks.append(pred_raw)
            baseline_blocks.append(baseline_raw)
            y_true_blocks.append(tensor_to_numpy(batch["target_raw"]).astype(np.float32))
            batch_ids.append(tensor_to_numpy(batch["batch_id"]).astype(np.int64))
            sample_indices.append(tensor_to_numpy(batch["sample_index"]).astype(np.int64))
            target_times.append(tensor_to_numpy(batch["target_time_h"]).astype(np.float32))
            for key, value in aux.items():
                aux_blocks.setdefault(key, []).append(tensor_to_numpy(value).astype(np.float32))
    y_true = np.vstack(y_true_blocks)
    y_pred = np.vstack(y_pred_blocks)
    baseline_pred = np.vstack(baseline_blocks)
    metrics = regression_metrics(y_true, y_pred, metadata["target_scales"])
    rows = metric_rows(
        phase=PHASE,
        experiment_name=EXPERIMENT,
        split=split,
        model=residual_model,
        forecast_mode=f"controlled_ridge_residual_{raman_method}",
        horizon_steps=horizon_steps,
        horizon_h=horizon_h,
        target_columns=metadata["target_columns"],
        metrics=metrics,
    )
    add_common_fields(
        rows,
        raman_method=raman_method,
        residual_model=residual_model,
        baseline_alpha=baseline_alpha,
        residual_scale_source="train_normal_residual_std",
    )
    payload = {
        "y_true": y_true.astype(np.float32),
        "y_pred": y_pred.astype(np.float32),
        "baseline_pred": baseline_pred.astype(np.float32),
        "batch_id": np.concatenate(batch_ids),
        "sample_index": np.concatenate(sample_indices),
        "target_time_h": np.concatenate(target_times),
    }
    aux_summary = {}
    for key, blocks in aux_blocks.items():
        values = np.concatenate(blocks, axis=0)
        payload[key] = values
        aux_summary[f"{split}_{key}_mean"] = float(np.mean(values))
        aux_summary[f"{split}_{key}_std"] = float(np.std(values))
        if key == "target_gate":
            for idx, target in enumerate(metadata["target_columns"]):
                aux_summary[f"{split}_target_gate_mean__{target}"] = float(np.mean(values[:, idx]))
    return rows, payload, aux_summary


def train_one_run(
    *,
    cfg: dict[str, Any],
    project_root: Path,
    metadata: dict[str, Any],
    design: dict[int, dict[str, dict[str, np.ndarray]]],
    baseline_by_horizon: dict[int, dict[str, Any]],
    residual_model: str,
    raman_method: str,
    horizon_steps: int,
    seed: int,
    device_arg: str,
    max_samples_per_split: int | None,
) -> dict[str, Any]:
    torch, nn, DataLoader, Dataset = import_torch()
    set_seed(seed, torch)
    device = torch.device("cuda" if device_arg == "auto" and torch.cuda.is_available() else ("cpu" if device_arg == "auto" else device_arg))
    ResidualWindowDataset = build_dataset_class(Dataset)
    ResidualModel = build_model_class(torch, nn)
    baseline = baseline_by_horizon[horizon_steps]
    baseline_alpha = float(baseline["alpha"])
    residual_scale = baseline["residual_scale"].astype(np.float32)
    run_name = f"{residual_model}_{raman_method}_h{horizon_steps}_seed{seed}"
    output_root = project_root / cfg["output_root"]
    run_dir = output_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log(f"Starting run {run_name} device={device} baseline_alpha={baseline_alpha}")
    train_dataset = ResidualWindowDataset(
        split_data=design[horizon_steps]["train_normal"],
        baseline_pred_scaled=baseline["splits"]["train_normal"]["pred_scaled"],
        residual_scale=residual_scale,
        raman_method=raman_method,
        max_samples=max_samples_per_split,
    )
    val_dataset = ResidualWindowDataset(
        split_data=design[horizon_steps]["val_normal"],
        baseline_pred_scaled=baseline["splits"]["val_normal"]["pred_scaled"],
        residual_scale=residual_scale,
        raman_method=raman_method,
        max_samples=max_samples_per_split,
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError(f"{run_name}: empty train or validation dataset")
    log(
        f"{run_name}: train_samples={len(train_dataset)} val_samples={len(val_dataset)} "
        f"history_dim={len(metadata['history_columns'])} target_dim={len(metadata['target_columns'])}"
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["training"]["num_workers"]),
        pin_memory=bool(cfg["training"]["pin_memory"]),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"]["num_workers"]),
        pin_memory=bool(cfg["training"]["pin_memory"]),
    )
    raman_dim = 1 if raman_method == "none" else design[horizon_steps]["train_normal"][f"raman_{raman_method}"].shape[1]
    hp = cfg["hyperparameters"]
    model = ResidualModel(
        residual_model=residual_model,
        input_dim=len(metadata["history_columns"]),
        exogenous_dim=len(metadata["exogenous_columns"]),
        raman_dim=raman_dim,
        output_dim=len(metadata["target_columns"]),
        hidden_dim=int(hp["hidden_dim"]),
        num_layers=int(hp["num_layers"]),
        kernel_size=int(hp["kernel_size"]),
        dropout=float(hp["dropout"]),
    ).to(device)
    parameter_count = int(sum(p.numel() for p in model.parameters()))
    log(f"{run_name}: model_parameters={parameter_count}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    loss_fn = torch.nn.MSELoss()
    best_val = math.inf
    best_epoch = -1
    patience_left = int(cfg["training"]["patience"])
    min_delta = float(cfg["training"].get("min_delta", 0.0))
    stop_reason = "max_epochs"
    history_rows: list[dict[str, Any]] = []
    checkpoint_path = run_dir / "best_model.pt"
    start = time.time()
    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        train_losses = []
        log(f"{run_name}: epoch={epoch} train_start batches={len(train_loader)}")
        for batch_idx, batch in enumerate(train_loader, start=1):
            history = batch["history"].to(device=device, dtype=torch.float32)
            future_exogenous = batch["future_exogenous"].to(device=device, dtype=torch.float32)
            raman = batch["raman"].to(device=device, dtype=torch.float32)
            baseline_scaled = batch["baseline_scaled"].to(device=device, dtype=torch.float32)
            target = batch["target_residual_z"].to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            pred = model(history, future_exogenous, raman, baseline_scaled)
            loss = loss_fn(pred, target)
            if not bool(torch.isfinite(loss).detach().cpu()):
                raise FloatingPointError(f"{run_name}: epoch={epoch} batch={batch_idx} non-finite train loss")
            loss_value = float(loss.detach().cpu())
            if not math.isfinite(loss_value):
                raise FloatingPointError(f"{run_name}: epoch={epoch} batch={batch_idx} non-finite train loss value")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["grad_clip_norm"]))
            optimizer.step()
            train_losses.append(loss_value)
            if batch_idx == 1 or batch_idx == len(train_loader):
                log(f"{run_name}: epoch={epoch} train_batch={batch_idx}/{len(train_loader)} loss={train_losses[-1]:.6f}")
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                history = batch["history"].to(device=device, dtype=torch.float32)
                future_exogenous = batch["future_exogenous"].to(device=device, dtype=torch.float32)
                raman = batch["raman"].to(device=device, dtype=torch.float32)
                baseline_scaled = batch["baseline_scaled"].to(device=device, dtype=torch.float32)
                target = batch["target_residual_z"].to(device=device, dtype=torch.float32)
                pred = model(history, future_exogenous, raman, baseline_scaled)
                val_loss_tensor = loss_fn(pred, target)
                if not bool(torch.isfinite(val_loss_tensor).detach().cpu()):
                    raise FloatingPointError(f"{run_name}: epoch={epoch} non-finite validation loss")
                val_loss_value = float(val_loss_tensor.detach().cpu())
                if not math.isfinite(val_loss_value):
                    raise FloatingPointError(f"{run_name}: epoch={epoch} non-finite validation loss value")
                val_losses.append(val_loss_value)
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        if not math.isfinite(train_loss) or not math.isfinite(val_loss):
            raise FloatingPointError(f"{run_name}: epoch={epoch} non-finite epoch loss")
        improvement = best_val - val_loss
        is_best = improvement > min_delta
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss_mse_residual_z": train_loss,
                "val_loss_mse_residual_z": val_loss,
                "best_val_before_update": best_val,
                "improvement_vs_best": improvement if math.isfinite(best_val) else math.nan,
                "min_delta": min_delta,
                "is_best": int(is_best),
                "patience_left_before_update": patience_left,
                "residual_model": residual_model,
                "raman_method": raman_method,
                "horizon_steps": horizon_steps,
                "seed": seed,
                "baseline_alpha": baseline_alpha,
            }
        )
        log(
            f"{run_name}: epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"best_val={best_val:.6f} improvement={improvement:.10f} "
            f"patience_left={patience_left} elapsed_min={(time.time()-start)/60:.2f}"
        )
        if is_best:
            best_val = val_loss
            best_epoch = epoch
            patience_left = int(cfg["training"]["patience"])
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "residual_model": residual_model,
                    "raman_method": raman_method,
                    "horizon_steps": horizon_steps,
                    "seed": seed,
                    "target_columns": metadata["target_columns"],
                    "history_columns": metadata["history_columns"],
                    "exogenous_columns": metadata["exogenous_columns"],
                    "baseline_alpha": baseline_alpha,
                    "residual_scale": residual_scale,
                    "best_val_loss": best_val,
                    "best_epoch": best_epoch,
                },
                checkpoint_path,
            )
            log(f"{run_name}: epoch={epoch} new_best_val={best_val:.6f}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                stop_reason = "early_stop_min_delta_patience"
                log(f"{run_name}: early_stop epoch={epoch}")
                break
    write_csv(run_dir / "training_history.csv", history_rows)
    if best_epoch < 0:
        raise RuntimeError(f"{run_name}: no valid checkpoint was created")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    all_metric_rows: list[dict[str, Any]] = []
    prediction_manifest: dict[str, str] = {}
    inference_summary: dict[str, dict[str, float]] = {}
    aux_summary: dict[str, float] = {}
    horizon_h = horizon_steps * median_dt_h(metadata["processed_root"], metadata["splits"]["train_normal"][0])
    for split in ["train_normal", "val_normal", "test_normal", "test_fault"]:
        dataset = ResidualWindowDataset(
            split_data=design[horizon_steps][split],
            baseline_pred_scaled=baseline["splits"][split]["pred_scaled"],
            residual_scale=residual_scale,
            raman_method=raman_method,
            max_samples=None,
        )
        loader = DataLoader(
            dataset,
            batch_size=int(cfg["training"]["batch_size"]),
            shuffle=False,
            num_workers=int(cfg["training"]["num_workers"]),
            pin_memory=bool(cfg["training"]["pin_memory"]),
        )
        log(f"{run_name}: evaluation_start split={split} samples={len(dataset)}")
        inference_start = time.time()
        rows, payload, split_aux = evaluate(
            model=model,
            loader=loader,
            device=device,
            torch=torch,
            metadata=metadata,
            residual_model=residual_model,
            raman_method=raman_method,
            horizon_steps=horizon_steps,
            horizon_h=horizon_h,
            residual_scale=residual_scale,
            split=split,
            baseline_alpha=baseline_alpha,
        )
        inference_elapsed = max(time.time() - inference_start, 1e-12)
        inference_summary[split] = {
            "samples": int(len(dataset)),
            "elapsed_seconds": round(inference_elapsed, 6),
            "latency_ms_per_sample": round(1000.0 * inference_elapsed / max(len(dataset), 1), 6),
            "samples_per_second": round(len(dataset) / inference_elapsed, 6),
        }
        all_metric_rows.extend(rows)
        aux_summary.update(split_aux)
        pred_path = run_dir / f"predictions_{split}.npz"
        np.savez_compressed(pred_path, **payload)
        prediction_manifest[split] = str(pred_path)
        log(f"{run_name}: evaluation_done split={split}")
    metrics_path = run_dir / "metrics.csv"
    write_csv(metrics_path, all_metric_rows)
    run_manifest = {
        "run_name": run_name,
        "phase": "10_residual_multimodal",
        "residual_model": residual_model,
        "raman_method": raman_method,
        "forecast_mode": cfg["forecast_mode"],
        "horizon_steps": horizon_steps,
        "horizon_h": horizon_h,
        "seed": seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_loss_mse_residual_z": best_val,
        "min_delta": min_delta,
        "stop_reason": stop_reason,
        "baseline_alpha": baseline_alpha,
        "residual_scale_source": "train_normal_residual_std",
        "elapsed_seconds": round(time.time() - start, 2),
        "parameter_count": parameter_count,
        "inference_summary": inference_summary,
        "metrics_csv": str(metrics_path),
        "training_history_csv": str(run_dir / "training_history.csv"),
        "checkpoint": str(checkpoint_path),
        "prediction_files": prediction_manifest,
        "target_columns": metadata["target_columns"],
        "history_columns": metadata["history_columns"],
        "exogenous_columns": metadata["exogenous_columns"],
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "aux_summary": aux_summary,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"{run_name}: run_complete elapsed_seconds={run_manifest['elapsed_seconds']}")
    return run_manifest


def resolve_arg(value: str, available: list[Any], cast=str) -> list[Any]:
    if value == "all":
        return available
    return [cast(value)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase 10 residual multimodal FermNFTP")
    parser.add_argument("--config", default="configs/model/phase10_residual_multimodal.json")
    parser.add_argument("--residual-model", default="all")
    parser.add_argument("--raman-method", default="all")
    parser.add_argument("--horizon-steps", default="all")
    parser.add_argument("--seed", default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--prepare-baseline-only", action="store_true")
    args = parser.parse_args()

    project_root = Path.cwd()
    cfg = load_json(project_root / args.config)
    output_root = project_root / cfg["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    metadata = build_metadata(cfg, project_root)
    sample_metadata = build_raman_sample_metadata(metadata["processed_root"], metadata["split_info"])
    scores_npz = np.load(project_root / cfg["phase08_pca_scores"])
    raman_scores_by_method: dict[str, np.ndarray] = {}
    for method in cfg["raman_methods"]:
        key = f"{method}_scores"
        if key not in scores_npz:
            raise ValueError(f"Missing Raman score key {key} in {project_root / cfg['phase08_pca_scores']}")
        raman_scores_by_method[method] = scores_npz[key].astype(np.float32)
    log("Building Phase 10 Raman-aligned design arrays")
    design = build_design(
        cfg=cfg,
        metadata=metadata,
        sample_metadata=sample_metadata,
        raman_scores_by_method=raman_scores_by_method,
    )
    baseline_by_horizon = prepare_ridge_baselines(
        cfg=cfg,
        project_root=project_root,
        metadata=metadata,
        design=design,
    )
    if args.prepare_baseline_only:
        log("Prepared ridge baselines only; exiting before residual training")
        return
    residual_models = resolve_arg(args.residual_model, cfg["residual_models"], str)
    horizons = resolve_arg(args.horizon_steps, cfg["horizon_steps"], int)
    seeds = [int(seed) for seed in cfg["seeds"]] if args.seed == "all" else [int(args.seed)]
    manifests = []
    for seed in seeds:
        for horizon in horizons:
            for residual_model in residual_models:
                methods = ["none"] if residual_model == "residual_process_tcn" else resolve_arg(args.raman_method, cfg["raman_methods"], str)
                for raman_method in methods:
                    manifests.append(
                        train_one_run(
                            cfg=cfg,
                            project_root=project_root,
                            metadata=metadata,
                            design=design,
                            baseline_by_horizon=baseline_by_horizon,
                            residual_model=residual_model,
                            raman_method=raman_method,
                            horizon_steps=int(horizon),
                            seed=int(seed),
                            device_arg=args.device,
                            max_samples_per_split=args.max_samples_per_split,
                        )
                    )
    aggregate = {
        "phase": "10_residual_multimodal",
        "config": str(project_root / args.config),
        "run_count": len(manifests),
        "baseline_manifest": str(output_root / "baselines" / "ridge_baseline_manifest.json"),
        "runs": manifests,
    }
    (output_root / "aggregate_run_manifest.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    log("Phase 10 all requested runs complete")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

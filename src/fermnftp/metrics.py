"""Metrics and CSV helpers for FermNFTP experiments."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, target_scales: np.ndarray
) -> dict[str, np.ndarray]:
    err = y_pred - y_true
    mae = np.mean(np.abs(err), axis=0)
    rmse = np.sqrt(np.mean(err**2, axis=0))
    denom = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    r2 = 1.0 - np.sum(err**2, axis=0) / np.where(denom == 0, np.nan, denom)
    nrmse = rmse / target_scales
    return {"mae": mae, "rmse": rmse, "r2": r2, "nrmse_train_std": nrmse}


def metric_rows(
    *,
    phase: str,
    experiment_name: str,
    split: str,
    model: str,
    forecast_mode: str,
    horizon_steps: int,
    horizon_h: float,
    target_columns: list[str],
    metrics: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, target in enumerate(target_columns):
        for metric_name, values in metrics.items():
            rows.append(
                {
                    "phase": phase,
                    "experiment_name": experiment_name,
                    "split": split,
                    "model": model,
                    "forecast_mode": forecast_mode,
                    "horizon_steps": horizon_steps,
                    "horizon_h": horizon_h,
                    "target": target,
                    "metric_name": metric_name,
                    "metric_value": float(values[idx]),
                    "unit": "target_unit" if metric_name in {"mae", "rmse"} else "unitless",
                }
            )
    rows.append(
        {
            "phase": phase,
            "experiment_name": experiment_name,
            "split": split,
            "model": model,
            "forecast_mode": forecast_mode,
            "horizon_steps": horizon_steps,
            "horizon_h": horizon_h,
            "target": "ALL",
            "metric_name": "mean_nrmse_train_std",
            "metric_value": float(np.mean(metrics["nrmse_train_std"])),
            "unit": "unitless",
        }
    )
    rows.append(
        {
            "phase": phase,
            "experiment_name": experiment_name,
            "split": split,
            "model": model,
            "forecast_mode": forecast_mode,
            "horizon_steps": horizon_steps,
            "horizon_h": horizon_h,
            "target": "ALL",
            "metric_name": "median_nrmse_train_std",
            "metric_value": float(np.median(metrics["nrmse_train_std"])),
            "unit": "unitless",
        }
    )
    rows.append(
        {
            "phase": phase,
            "experiment_name": experiment_name,
            "split": split,
            "model": model,
            "forecast_mode": forecast_mode,
            "horizon_steps": horizon_steps,
            "horizon_h": horizon_h,
            "target": "ALL",
            "metric_name": "mean_r2",
            "metric_value": float(np.nanmean(metrics["r2"])),
            "unit": "unitless",
        }
    )
    rows.append(
        {
            "phase": phase,
            "experiment_name": experiment_name,
            "split": split,
            "model": model,
            "forecast_mode": forecast_mode,
            "horizon_steps": horizon_steps,
            "horizon_h": horizon_h,
            "target": "ALL",
            "metric_name": "median_r2",
            "metric_value": float(np.nanmedian(metrics["r2"])),
            "unit": "unitless",
        }
    )
    return rows

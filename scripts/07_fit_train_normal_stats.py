#!/usr/bin/env python3
"""Fit train-normal scaling statistics and residual scale sources."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def finite_stats(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "count": 0,
            "missing": int(values.size),
            "missing_ratio": 1.0,
            "mean": math.nan,
            "std": math.nan,
            "median": math.nan,
            "mad": math.nan,
            "robust_scale": math.nan,
            "min": math.nan,
            "max": math.nan,
        }
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    robust_scale = 1.4826 * mad
    if not np.isfinite(robust_scale) or robust_scale == 0:
        robust_scale = float(np.std(finite))
    if not np.isfinite(robust_scale) or robust_scale == 0:
        robust_scale = 1.0
    return {
        "count": int(finite.size),
        "missing": int(values.size - finite.size),
        "missing_ratio": float((values.size - finite.size) / values.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "median": median,
        "mad": mad,
        "robust_scale": robust_scale,
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def gradient_by_time(values: np.ndarray, time_h: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(values) & np.isfinite(time_h)
    if finite.sum() < 2:
        return out
    idx = np.where(finite)[0]
    out[idx] = np.gradient(values[idx], time_h[idx])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Phase 04 train-normal statistics")
    parser.add_argument("--processed-root", default="data/processed/phase04")
    parser.add_argument("--split-csv", default="data/processed/phase04/splits/batch_splits.csv")
    parser.add_argument("--output-dir", default="data/processed/phase04/stats")
    args = parser.parse_args()

    project_root = Path.cwd()
    processed_root = project_root / args.processed_root
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    split_rows = read_csv_dicts(project_root / args.split_csv)
    train_rows = [row for row in split_rows if row["split"] == "train_normal"]
    if not train_rows:
        raise RuntimeError("No train_normal batches found.")

    process_blocks: list[np.ndarray] = []
    derived_blocks: list[np.ndarray] = []
    derivative_blocks: dict[str, list[np.ndarray]] = {
        "d_substrate_mass_g_per_h": [],
        "d_penicillin_mass_g_per_h": [],
        "d_offline_biomass_mass_g_per_h": [],
    }
    process_columns: list[str] | None = None
    derived_columns: list[str] | None = None

    for row in train_rows:
        batch_id = int(float(row["batch_id"]))
        path = processed_root / "batches" / f"batch_{batch_id:03d}.npz"
        data = np.load(path, allow_pickle=True)
        process = data["process"].astype(np.float64)
        derived = data["derived"].astype(np.float64)
        time_h = data["time_h"].astype(np.float64)
        process_blocks.append(process)
        derived_blocks.append(derived)
        process_columns = [str(col) for col in data["process_columns"]]
        derived_columns = [str(col) for col in data["derived_columns"]]
        derived_lookup = {name: idx for idx, name in enumerate(derived_columns)}
        derivative_blocks["d_substrate_mass_g_per_h"].append(
            gradient_by_time(derived[:, derived_lookup["substrate_mass_g"]], time_h)
        )
        derivative_blocks["d_penicillin_mass_g_per_h"].append(
            gradient_by_time(derived[:, derived_lookup["penicillin_mass_g"]], time_h)
        )
        derivative_blocks["d_offline_biomass_mass_g_per_h"].append(
            gradient_by_time(derived[:, derived_lookup["offline_biomass_mass_g"]], time_h)
        )

    if process_columns is None or derived_columns is None:
        raise RuntimeError("No data were loaded.")
    process_all = np.vstack(process_blocks)
    derived_all = np.vstack(derived_blocks)

    process_stats = {
        column: finite_stats(process_all[:, idx]) for idx, column in enumerate(process_columns)
    }
    derived_stats = {
        column: finite_stats(derived_all[:, idx]) for idx, column in enumerate(derived_columns)
    }
    derivative_stats = {
        name: finite_stats(np.concatenate(values)) for name, values in derivative_blocks.items()
    }

    scaler = {
        "schema_version": "phase04_v1",
        "source_split": "train_normal",
        "source_batches": [int(float(row["batch_id"])) for row in train_rows],
        "n_batches": len(train_rows),
        "n_rows": int(process_all.shape[0]),
        "method": "median/MAD robust statistics plus mean/std/min/max diagnostics",
        "process_columns": process_stats,
        "derived_columns": derived_stats,
    }
    scaler_path = output_dir / "train_normal_scalers.json"
    scaler_path.write_text(json.dumps(scaler, indent=2, ensure_ascii=False), encoding="utf-8")

    residual_scales = {
        "schema_version": "phase04_v1",
        "source_split": "train_normal",
        "source_batches": [int(float(row["batch_id"])) for row in train_rows],
        "scale_rule": "Each residual channel is divided by its train-normal robust scale, estimated as 1.4826 * MAD. If MAD is zero, standard deviation is used; if that is also zero, scale is set to 1.",
        "biokinetic_state_scales": {
            "substrate_mass_g": derived_stats["substrate_mass_g"],
            "penicillin_mass_g": derived_stats["penicillin_mass_g"],
            "offline_biomass_mass_g": derived_stats["offline_biomass_mass_g"],
            "rq_proxy": derived_stats["rq_proxy"],
            "cumulative_sugar_feed_L": derived_stats["cumulative_sugar_feed_L"],
        },
        "biokinetic_derivative_scales": derivative_stats,
        "weight_policy": {
            "residual_channel_weights": "equal after train-normal robust-scale normalization",
            "global_loss_weights": "not selected in Phase 04; must be selected by a predefined validation-normal grid in model-training phases",
            "test_fault_usage": "forbidden for scale estimation or weight selection",
        },
    }
    residual_scale_path = output_dir / "biokinetic_residual_scale_sources.json"
    residual_scale_path.write_text(
        json.dumps(residual_scales, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    common_stats_dir = project_root / "data" / "processed" / "stats"
    common_stats_dir.mkdir(parents=True, exist_ok=True)
    loss_weight_sources = {
        "schema_version": "phase04_v1",
        "status": "prepared_scale_sources_only_no_model_loss_weights_selected",
        "source_split": "train_normal",
        "train_normal_scaler_json": str(scaler_path),
        "biokinetic_residual_scale_sources_json": str(residual_scale_path),
        "allowed_weight_sources": [
            "train-normal robust statistics",
            "train-normal parameter fitting",
            "predefined validation-normal grid selection",
            "uncertainty modeling trained without test-fault data"
        ],
        "forbidden": [
            "manual empirical weights without recorded source",
            "using test-fault data for loss-weight or anomaly-threshold selection",
            "changing weights after inspecting final test-fault metrics"
        ],
    }
    common_loss_path = common_stats_dir / "loss_weight_sources.json"
    common_loss_path.write_text(
        json.dumps(loss_weight_sources, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "train_normal_scalers": str(scaler_path),
                "biokinetic_residual_scale_sources": str(residual_scale_path),
                "loss_weight_sources": str(common_loss_path),
                "n_train_batches": len(train_rows),
                "n_train_rows": int(process_all.shape[0]),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

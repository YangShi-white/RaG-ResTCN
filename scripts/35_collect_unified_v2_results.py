#!/usr/bin/env python3
"""Collect unified-v2 outputs into paper-ready metric CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fermnftp.data import build_forecasting_metadata, inverse_standardize, load_json  # noqa: E402


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def regression(y_true: np.ndarray, y_pred: np.ndarray, scales: np.ndarray) -> dict[str, np.ndarray | float]:
    err = y_pred - y_true
    rmse = np.sqrt(np.mean(err**2, axis=0))
    mae = np.mean(np.abs(err), axis=0)
    nrmse = rmse / np.maximum(scales, 1e-8)
    ss_res = np.sum(err**2, axis=0)
    centered = y_true - np.mean(y_true, axis=0, keepdims=True)
    ss_tot = np.sum(centered**2, axis=0)
    r2 = np.where(ss_tot > 1e-12, 1.0 - ss_res / ss_tot, np.nan)
    return {
        "nrmse": nrmse,
        "mae": mae,
        "r2": r2,
        "mean_nrmse": float(np.nanmean(nrmse)),
        "mean_mae": float(np.nanmean(mae)),
        "mean_r2": float(np.nanmean(r2)),
    }


def model_label(model: str, raman_method: str = "") -> str:
    mapping = {
        "direct_tcn": "Direct TCN",
        "linear_mlp": "Linear-MLP",
        "dlinear": "Linear-MLP",
        "dlinear_control": "DLinear-Control",
        "itransformer_lite": "iTransformer-Lite",
        "tsmixer": "TSMixer",
        "controlled_ridge": "Controlled Ridge",
        "residual_process_tcn": "Process-only ResTCN",
        "residual_naive_raman_tcn": "Naive Raman ResTCN",
        "residual_global_gate_tcn": "Global-gate Raman ResTCN",
        "residual_attention_tcn": "Attention Raman ResTCN",
        "residual_target_gate_tcn": "RaG-ResTCN",
        "residual_conservative_target_gate_tcn": "Conservative RaG-ResTCN",
    }
    label = mapping.get(model, model)
    if model == "residual_target_gate_tcn" and raman_method.endswith("_zero"):
        label = "RaG-ResTCN zero Raman"
    if model == "residual_target_gate_tcn" and raman_method.endswith("_shuffled"):
        label = "RaG-ResTCN shuffled Raman"
    if model == "residual_conservative_target_gate_tcn" and raman_method.endswith("_zero"):
        label = "Conservative RaG-ResTCN zero Raman"
    if model == "residual_conservative_target_gate_tcn" and raman_method.endswith("_shuffled"):
        label = "Conservative RaG-ResTCN shuffled Raman"
    return label


def append_per_batch(
    rows: list[dict[str, Any]],
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    batch_id: np.ndarray,
    scales: np.ndarray,
    targets: list[str],
    fields: dict[str, Any],
) -> None:
    for bid in sorted(np.unique(batch_id).tolist()):
        mask = batch_id == bid
        m = regression(y_true[mask], y_pred[mask], scales)
        rows.append({**fields, "batch_id": int(bid), "target": "__target_average__", "nrmse": m["mean_nrmse"], "mae": m["mean_mae"], "r2": m["mean_r2"], "n_windows": int(mask.sum())})
        for j, target in enumerate(targets):
            rows.append({**fields, "batch_id": int(bid), "target": target, "nrmse": float(m["nrmse"][j]), "mae": float(m["mae"][j]), "r2": float(m["r2"][j]) if np.isfinite(m["r2"][j]) else "", "n_windows": int(mask.sum())})


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    keys = ["model_label", "horizon_steps", "split", "target"]
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    out = []
    for key, vals in sorted(groups.items()):
        rec = dict(zip(keys, key))
        for metric in ["nrmse", "mae", "r2"]:
            arr = np.asarray([float(v[metric]) for v in vals if str(v.get(metric, "")) != ""], dtype=float)
            rec[f"{metric}_mean"] = float(np.nanmean(arr)) if arr.size else ""
            rec[f"{metric}_std"] = float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0
            rec[f"{metric}_n"] = int(np.isfinite(arr).sum()) if arr.size else 0
        out.append(rec)
    return out


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return math.nan
    rx = rankdata(x[mask])
    ry = rankdata(y[mask])
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model/unified_v2_rag_restcn.json")
    args = parser.parse_args()
    cfg = load_json(PROJECT_ROOT / args.config)
    metadata = build_forecasting_metadata(cfg, PROJECT_ROOT)
    out_root = PROJECT_ROOT / cfg["output_root"]
    paper_dir = out_root / "paper_results"
    target_scales = metadata["target_scales"].astype(np.float32)
    targets = metadata["target_columns"]

    per_batch_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    seen_ridge: set[tuple[Any, ...]] = set()

    modern_batch = out_root / "modern_deep_baselines" / "metrics" / "per_batch_metrics.csv"
    if modern_batch.exists():
        for row in read_rows(modern_batch):
            row = dict(row)
            row["model_label"] = model_label(row["model"])
            per_batch_rows.append(row)

    for manifest_path in sorted((out_root / "rag_restcn").glob("split_seed_*/**/runs/*/run_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        split_seed = int(manifest["seed"])
        model = str(manifest["residual_model"])
        raman_method = str(manifest["raman_method"])
        horizon = int(manifest["horizon_steps"])
        label = model_label(model, raman_method)
        for split, pred_path_str in manifest["prediction_files"].items():
            pred_path = Path(pred_path_str)
            if not pred_path.is_absolute():
                pred_path = PROJECT_ROOT / pred_path
            data = np.load(pred_path, allow_pickle=True)
            y_true = data["y_true"].astype(np.float32)
            y_pred = data["y_pred"].astype(np.float32)
            batch_id = data["batch_id"].astype(np.int64)
            fields = {
                "seed": split_seed,
                "model": model,
                "model_label": label,
                "control_mode": "future_controls",
                "raman_method": raman_method,
                "horizon_steps": horizon,
                "split": split,
            }
            append_per_batch(per_batch_rows, y_true=y_true, y_pred=y_pred, batch_id=batch_id, scales=target_scales, targets=targets, fields=fields)

            ridge_key = (split_seed, horizon, split)
            if ridge_key not in seen_ridge and "baseline_pred" in data:
                append_per_batch(
                    per_batch_rows,
                    y_true=y_true,
                    y_pred=data["baseline_pred"].astype(np.float32),
                    batch_id=batch_id,
                    scales=target_scales,
                    targets=targets,
                    fields={**fields, "model": "controlled_ridge", "model_label": "Controlled Ridge", "raman_method": "none"},
                )
                seen_ridge.add(ridge_key)

            if split == "test_normal" and "target_gate" in data:
                gates = data["target_gate"].astype(np.float32)
                for j, target in enumerate(targets):
                    gate_rows.append(
                        {
                            "seed": split_seed,
                            "horizon_steps": horizon,
                            "target": target,
                            "model_label": label,
                            "raman_method": raman_method,
                            "gate_mean": float(np.mean(gates[:, j])),
                            "gate_std": float(np.std(gates[:, j])),
                        }
                    )

    write_rows(paper_dir / "unified_v2_per_batch_metrics.csv", per_batch_rows)
    summary_rows = summarize([row for row in per_batch_rows if str(row.get("target")) == "__target_average__"])
    write_rows(paper_dir / "unified_v2_main_summary_mean_std.csv", summary_rows)
    write_rows(paper_dir / "unified_v2_gate_summary.csv", gate_rows)

    process_groups: dict[tuple[int, int, str], list[float]] = {}
    for r in per_batch_rows:
        if r.get("model_label") == "Process-only ResTCN" and r.get("split") == "test_normal" and r.get("target") != "__target_average__":
            process_groups.setdefault((int(r["seed"]), int(r["horizon_steps"]), str(r["target"])), []).append(float(r["nrmse"]))
    process_lookup = {key: float(np.mean(vals)) for key, vals in process_groups.items()}
    rag_groups: dict[tuple[int, int, str], list[float]] = {}
    rag_template: dict[tuple[int, int, str], dict[str, Any]] = {}
    gain_model_labels = {"RaG-ResTCN", "Conservative RaG-ResTCN"}
    for r in per_batch_rows:
        if r.get("model_label") in gain_model_labels and r.get("split") == "test_normal" and r.get("target") != "__target_average__":
            key = (int(r["seed"]), int(r["horizon_steps"]), str(r["target"]))
            rag_groups.setdefault(key, []).append(float(r["nrmse"]))
            rag_template.setdefault(key, dict(r))
    gain_rows = []
    for key, vals in rag_groups.items():
        if key in process_lookup:
            row = rag_template[key]
            gain_rows.append({**row, "nrmse": float(np.mean(vals)), "raman_gain_nrmse": process_lookup[key] - float(np.mean(vals))})
    write_rows(paper_dir / "unified_v2_raman_gain_by_target.csv", gain_rows)

    gate_gain_rows = []
    gate_lookup = {(int(r["seed"]), int(r["horizon_steps"]), str(r["target"])): float(r["gate_mean"]) for r in gate_rows if r["model_label"] in gain_model_labels}
    for row in gain_rows:
        key = (int(row["seed"]), int(row["horizon_steps"]), str(row["target"]))
        if key in gate_lookup:
            gate_gain_rows.append({**row, "gate_mean": gate_lookup[key]})
    rho = spearman(
        np.asarray([r["gate_mean"] for r in gate_gain_rows], dtype=float),
        np.asarray([r["raman_gain_nrmse"] for r in gate_gain_rows], dtype=float),
    ) if gate_gain_rows else math.nan
    (paper_dir / "unified_v2_gate_gain_correlation.json").write_text(
        json.dumps({"spearman_rho": rho, "n": len(gate_gain_rows)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"per_batch_rows": len(per_batch_rows), "summary_rows": len(summary_rows), "gate_rows": len(gate_rows), "gate_gain_spearman": rho}, indent=2))


if __name__ == "__main__":
    main()

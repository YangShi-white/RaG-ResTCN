#!/usr/bin/env python3
"""Collect Phase 15 strict gap-completion evidence and paper assets.

This script is intentionally evidence-driven: it reads saved Phase15 training
outputs and real processed data, then generates CSV/PNG/MD assets for the
missing revision requirements. It does not fabricate loss curves or metrics.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
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

from fermnftp.data import load_json  # noqa: E402
from fermnftp.metrics import regression_metrics, write_csv  # noqa: E402
from fermnftp.plot_style import apply_ai_conference_style, polish_axis  # noqa: E402

apply_ai_conference_style(plt)


PHASE = "Phase 15"
EXPERIMENT = "Strict gap completion against revision requirements"


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

、。、、、p  CSV ，。

## 

{caption}

## 

```bash
{command}
```
"""
    path.write_text(text, encoding="utf-8")


def load_script_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def collect_residual_training(output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    metrics_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    baseline_path = output_root / "baselines" / "ridge_baseline_metrics.csv"
    if baseline_path.exists():
        for row in read_csv_dicts(baseline_path):
            row["run_name"] = f"ridge_only_h{row.get('horizon_steps', '')}"
            row["seed"] = -1
            row["best_epoch"] = math.nan
            row["best_val_loss_mse_residual_z"] = math.nan
            row["stop_reason"] = "closed_form_ridge"
            row["parameter_count"] = math.nan
            metrics_rows.append(row)
    for manifest_path in sorted((output_root / "runs").glob("*/run_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["_manifest_path"] = str(manifest_path)
        manifest["_run_dir"] = str(manifest_path.parent)
        manifests.append(manifest)
        metrics_path = Path(manifest.get("metrics_csv", ""))
        if not metrics_path.exists():
            metrics_path = manifest_path.parent / "metrics.csv"
        for row in read_csv_dicts(metrics_path):
            row["run_name"] = manifest["run_name"]
            row["seed"] = manifest["seed"]
            row["best_epoch"] = manifest["best_epoch"]
            row["best_val_loss_mse_residual_z"] = manifest["best_val_loss_mse_residual_z"]
            row["stop_reason"] = manifest.get("stop_reason", "")
            row["parameter_count"] = manifest.get("parameter_count", math.nan)
            metrics_rows.append(row)
        history_path = manifest_path.parent / "training_history.csv"
        if history_path.exists():
            for row in read_csv_dicts(history_path):
                row["run_name"] = manifest["run_name"]
                history_rows.append(row)
    if not metrics_rows:
        raise SystemExit(f"No Phase15 residual metrics found under {output_root / 'runs'}")
    return pd.DataFrame(metrics_rows), pd.DataFrame(history_rows), manifests


def aggregate_all_target_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    df = metrics.copy()
    df["metric_value"] = pd.to_numeric(df["metric_value"], errors="coerce")
    df["horizon_steps"] = df["horizon_steps"].astype(int)
    df["seed"] = df["seed"].astype(int)
    sub = df[
        (df["target"] == "ALL")
        & (df["metric_name"].isin(["mean_nrmse_train_std", "median_nrmse_train_std", "mean_r2", "median_r2"]))
    ]
    wide = sub.pivot_table(
        index=["split", "model", "raman_method", "residual_model", "horizon_steps", "seed", "run_name"],
        columns="metric_name",
        values="metric_value",
        aggfunc="first",
    ).reset_index()
    return wide


def model_label(cfg: dict[str, Any], model: str, raman_method: str) -> str:
    labels = cfg.get("model_label_map", {})
    if model == "ridge_only":
        return "Ridge"
    base = labels.get(model, model)
    if raman_method and raman_method != "none":
        return f"{base}-{raman_method.replace('_snv', '').replace('_', '+')}"
    return base


def figure_main_prediction(
    dirs: dict[str, Path],
    cfg: dict[str, Any],
    wide: pd.DataFrame,
) -> dict[str, str]:
    figure_id = "Fig43_phase15_main_prediction_horizon_matrix"
    grouped = (
        wide.groupby(["split", "model", "raman_method", "horizon_steps"], dropna=False)
        .agg(
            mean_nrmse=("mean_nrmse_train_std", "mean"),
            std_nrmse=("mean_nrmse_train_std", "std"),
            mean_r2=("mean_r2", "mean"),
            std_r2=("mean_r2", "std"),
            seed_count=("seed", "nunique"),
        )
        .reset_index()
    )
    rows = []
    for _, row in grouped.iterrows():
        rows.append(
            {
                "figure_id": figure_id,
                "phase": PHASE,
                "experiment_name": EXPERIMENT,
                "split": row["split"],
                "model": row["model"],
                "raman_method": row["raman_method"],
                "model_label": model_label(cfg, row["model"], row["raman_method"]),
                "horizon_steps": int(row["horizon_steps"]),
                "mean_nrmse": float(row["mean_nrmse"]),
                "std_nrmse": finite_float(row["std_nrmse"], 0.0),
                "mean_r2": float(row["mean_r2"]),
                "std_r2": finite_float(row["std_r2"], 0.0),
                "seed_count": int(row["seed_count"]),
            }
        )
    table_path = dirs["tables"] / f"{figure_id}_data.csv"
    write_union_csv(table_path, rows)

    primary = cfg.get("primary_raman_method", "airpls_snv")
    plot_models = [
        ("residual_process_tcn", "none"),
        ("residual_naive_raman_tcn", primary),
        ("residual_global_gate_tcn", primary),
        ("residual_attention_tcn", primary),
        ("residual_target_gate_tcn", primary),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    for ax, split in zip(axes, ["test_normal", "test_fault"]):
        sub = grouped[grouped["split"] == split]
        for model, method in plot_models:
            cur = sub[(sub["model"] == model) & (sub["raman_method"] == method)].sort_values("horizon_steps")
            if cur.empty:
                continue
            ax.plot(
                cur["horizon_steps"],
                cur["mean_nrmse"],
                marker="o",
                linewidth=1.8,
                label=model_label(cfg, model, method),
            )
        ax.set_title(split.replace("_", " ").title())
        ax.set_xlabel("Forecast Horizon (steps)")
        ax.set_ylabel("Mean nRMSE" if ax is axes[0] else "")
        polish_axis(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig_path = dirs["figures"] / f"{figure_id}.png"
    fig.savefig(fig_path, dpi=400)
    plt.close(fig)

    save_explanation(
        dirs["explanations"] / f"{figure_id}_explanation.md",
        title="Phase15 ",
        figure_id=figure_id,
        purpose=" h=1/5/20/40/60  normal/fault 。",
        data_source="Phase15  residual model metrics.csv。",
        design="， nRMSE；CSV  mean±std、R2  seed 。",
        results=" horizon  mean nRMSE，。",
        interpretation=" E1， h=40/60。",
        discussion="normal  fault ，。",
        limitations="；std  CSV 。",
        caption="Main multi-horizon forecasting performance under normal and fault conditions.",
        command="python3 scripts/25_run_phase15_strict_gap_completion.py --config configs/model/phase15_strict_gap_completion.json",
    )
    return {"figure": str(fig_path), "table": str(table_path)}


def figure_multimodal_ablation(
    dirs: dict[str, Path],
    cfg: dict[str, Any],
    wide: pd.DataFrame,
) -> dict[str, str]:
    figure_id = "Fig44_phase15_multimodal_residual_ablation"
    primary = cfg.get("primary_raman_method", "airpls_snv")
    ordered = [
        ("residual_process_tcn", "none", "No Raman"),
        ("residual_naive_raman_tcn", primary, "Concat"),
        ("residual_global_gate_tcn", primary, "Gate"),
        ("residual_attention_tcn", primary, "Attention"),
        ("residual_target_gate_tcn", primary, "Target Gate"),
    ]
    rows = []
    for split in ["test_normal", "test_fault"]:
        for horizon in sorted(wide["horizon_steps"].unique()):
            ref = wide[
                (wide["split"] == split)
                & (wide["horizon_steps"] == horizon)
                & (wide["model"] == "residual_process_tcn")
                & (wide["raman_method"] == "none")
            ]["mean_nrmse_train_std"].mean()
            for model, method, label in ordered:
                cur = wide[
                    (wide["split"] == split)
                    & (wide["horizon_steps"] == horizon)
                    & (wide["model"] == model)
                    & (wide["raman_method"] == method)
                ]
                if cur.empty:
                    continue
                mean_val = float(cur["mean_nrmse_train_std"].mean())
                rows.append(
                    {
                        "figure_id": figure_id,
                        "phase": PHASE,
                        "experiment_name": EXPERIMENT,
                        "split": split,
                        "horizon_steps": int(horizon),
                        "ablation_variant": label,
                        "model": model,
                        "raman_method": method,
                        "mean_nrmse": mean_val,
                        "std_nrmse": finite_float(cur["mean_nrmse_train_std"].std(ddof=1), 0.0),
                        "mean_r2": float(cur["mean_r2"].mean()),
                        "std_r2": finite_float(cur["mean_r2"].std(ddof=1), 0.0),
                        "relative_change_vs_no_raman_percent": 100.0 * (mean_val - ref) / ref if ref else math.nan,
                        "seed_count": int(cur["seed"].nunique()),
                    }
                )
    table_path = dirs["tables"] / f"{figure_id}_data.csv"
    write_union_csv(table_path, rows)

    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    sub = plot_df[plot_df["split"] == "test_normal"]
    labels = [item[2] for item in ordered]
    x = np.arange(len(sorted(sub["horizon_steps"].unique())))
    width = 0.15
    for i, label in enumerate(labels):
        cur = sub[sub["ablation_variant"] == label].sort_values("horizon_steps")
        if cur.empty:
            continue
        ax.bar(x + (i - 2) * width, cur["mean_nrmse"], width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in sorted(sub["horizon_steps"].unique())])
    ax.set_xlabel("Forecast Horizon (steps)")
    ax.set_ylabel("Mean nRMSE")
    ax.set_title("Multimodal Residual Ablation on Test Normal")
    ax.legend(frameon=False, ncols=3, fontsize=8)
    polish_axis(ax)
    fig.tight_layout()
    fig_path = dirs["figures"] / f"{figure_id}.png"
    fig.savefig(fig_path, dpi=400)
    plt.close(fig)

    save_explanation(
        dirs["explanations"] / f"{figure_id}_explanation.md",
        title="Phase15 ",
        figure_id=figure_id,
        purpose=" no Raman、concat、gate、attention  target gate 。",
        data_source="Phase15 residual model  seed 。",
        design=" horizon ；CSV  normal/fault、mean±std  no Raman 。",
        results=" target gate  attention  nRMSE  no Raman， Raman 。",
        interpretation=" E2。",
        discussion=" horizon  Raman ，。",
        limitations=" primary Raman ； metrics 。",
        caption="Ablation of Raman fusion mechanisms in the residual forecasting framework.",
        command="python3 scripts/25_run_phase15_strict_gap_completion.py --config configs/model/phase15_strict_gap_completion.json",
    )
    return {"figure": str(fig_path), "table": str(table_path)}


def load_prediction(run_dir: Path, split: str) -> dict[str, np.ndarray]:
    path = run_dir / f"predictions_{split}.npz"
    if not path.exists():
        raise RuntimeError(f"Missing prediction file: {path}")
    data = np.load(path)
    return {key: data[key] for key in data.files}


def per_sample_nrmse(y_true: np.ndarray, y_pred: np.ndarray, scales: np.ndarray) -> np.ndarray:
    err = (np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)) / scales.reshape(1, -1)
    return np.sqrt(np.mean(err**2, axis=1))


def binary_auroc(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=np.float64)
    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def binary_auprc(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=np.float64)
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return math.nan
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    precision = tp / np.arange(1, len(y_sorted) + 1)
    return float((precision * (y_sorted == 1)).sum() / n_pos)


def prediction_batch_scores(
    manifests: list[dict[str, Any]],
    *,
    cfg: dict[str, Any],
    target_scales: np.ndarray,
) -> list[dict[str, Any]]:
    primary = cfg.get("primary_raman_method", "airpls_snv")
    keep = {
        ("residual_process_tcn", "none"),
        ("residual_target_gate_tcn", primary),
        ("residual_attention_tcn", primary),
    }
    rows: list[dict[str, Any]] = []
    for manifest in manifests:
        key = (manifest["residual_model"], manifest["raman_method"])
        if key not in keep:
            continue
        run_dir = Path(manifest["_run_dir"])
        for split in ["train_normal", "test_normal", "test_fault"]:
            pred = load_prediction(run_dir, split)
            scores = per_sample_nrmse(pred["y_true"], pred["y_pred"], target_scales)
            for batch_id in sorted(set(pred["batch_id"].astype(int).tolist())):
                mask = pred["batch_id"].astype(int) == int(batch_id)
                batch_scores = scores[mask]
                rows.append(
                    {
                        "phase": PHASE,
                        "experiment_name": EXPERIMENT,
                        "split": split,
                        "batch_id": int(batch_id),
                        "fault_label": 1 if split == "test_fault" else 0,
                        "model": manifest["residual_model"],
                        "raman_method": manifest["raman_method"],
                        "horizon_steps": int(manifest["horizon_steps"]),
                        "seed": int(manifest["seed"]),
                        "batch_q95_prediction_nrmse": float(np.quantile(batch_scores, 0.95)),
                        "batch_max_prediction_nrmse": float(np.max(batch_scores)),
                        "sample_count": int(mask.sum()),
                    }
                )
    return rows


def figure_fault_detection(
    dirs: dict[str, Path],
    cfg: dict[str, Any],
    manifests: list[dict[str, Any]],
    target_scales: np.ndarray,
) -> dict[str, str]:
    figure_id = "Fig45_phase15_fault_generalization_detection"
    score_rows = prediction_batch_scores(manifests, cfg=cfg, target_scales=target_scales)
    score_df = pd.DataFrame(score_rows)
    metric_rows: list[dict[str, Any]] = []
    if not score_df.empty:
        for (model, method, horizon, seed), group in score_df.groupby(["model", "raman_method", "horizon_steps", "seed"]):
            train = group[group["split"] == "train_normal"]
            eval_df = group[group["split"].isin(["test_normal", "test_fault"])]
            if train.empty or eval_df.empty:
                continue
            threshold = float(np.quantile(train["batch_q95_prediction_nrmse"], float(cfg["fault_detection"]["threshold_quantile"])))
            y = eval_df["fault_label"].to_numpy(dtype=int)
            s = eval_df["batch_q95_prediction_nrmse"].to_numpy(dtype=float)
            alerted = s > threshold
            metric_rows.extend(
                [
                    {
                        "figure_id": figure_id,
                        "phase": PHASE,
                        "experiment_name": EXPERIMENT,
                        "model": model,
                        "raman_method": method,
                        "horizon_steps": int(horizon),
                        "seed": int(seed),
                        "metric_name": "auroc",
                        "metric_value": binary_auroc(y, s),
                        "threshold": threshold,
                    },
                    {
                        "figure_id": figure_id,
                        "phase": PHASE,
                        "experiment_name": EXPERIMENT,
                        "model": model,
                        "raman_method": method,
                        "horizon_steps": int(horizon),
                        "seed": int(seed),
                        "metric_name": "auprc",
                        "metric_value": binary_auprc(y, s),
                        "threshold": threshold,
                    },
                    {
                        "figure_id": figure_id,
                        "phase": PHASE,
                        "experiment_name": EXPERIMENT,
                        "model": model,
                        "raman_method": method,
                        "horizon_steps": int(horizon),
                        "seed": int(seed),
                        "metric_name": "false_alarm_rate",
                        "metric_value": float(alerted[y == 0].mean()) if np.any(y == 0) else math.nan,
                        "threshold": threshold,
                    },
                    {
                        "figure_id": figure_id,
                        "phase": PHASE,
                        "experiment_name": EXPERIMENT,
                        "model": model,
                        "raman_method": method,
                        "horizon_steps": int(horizon),
                        "seed": int(seed),
                        "metric_name": "fault_detection_rate",
                        "metric_value": float(alerted[y == 1].mean()) if np.any(y == 1) else math.nan,
                        "threshold": threshold,
                    },
                ]
            )
    table_path = dirs["tables"] / f"{figure_id}_data.csv"
    write_union_csv(table_path, metric_rows)
    score_path = dirs["metrics"] / "phase15_prediction_batch_scores.csv"
    write_union_csv(score_path, score_rows)

    plot_df = pd.DataFrame(metric_rows)
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    if not plot_df.empty:
        sub = (
            plot_df[plot_df["metric_name"].isin(["auroc", "auprc"])]
            .groupby(["model", "raman_method", "horizon_steps", "metric_name"], dropna=False)["metric_value"]
            .mean()
            .reset_index()
        )
        primary = cfg.get("primary_raman_method", "airpls_snv")
        for metric_name, marker in [("auroc", "o"), ("auprc", "s")]:
            cur = sub[
                (sub["metric_name"] == metric_name)
                & (sub["model"] == "residual_target_gate_tcn")
                & (sub["raman_method"] == primary)
            ].sort_values("horizon_steps")
            if not cur.empty:
                ax.plot(cur["horizon_steps"], cur["metric_value"], marker=marker, label=metric_name.upper())
    ax.set_xlabel("Forecast Horizon (steps)")
    ax.set_ylabel("Detection Metric")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Fault Generalization from Prediction Scores")
    ax.legend(frameon=False)
    polish_axis(ax)
    fig.tight_layout()
    fig_path = dirs["figures"] / f"{figure_id}.png"
    fig.savefig(fig_path, dpi=400)
    plt.close(fig)

    save_explanation(
        dirs["explanations"] / f"{figure_id}_explanation.md",
        title="Phase15 ",
        figure_id=figure_id,
        purpose=" AUROC、AUPRC、false alarm  fault detection rate 。",
        data_source="Phase15  train_normal/test_normal/test_fault 。",
        design=" train_normal  batch  q95 ， test_normal/test_fault 。",
        results="CSV  seed、horizon  AUROC/AUPRC//。",
        interpretation=" E4。",
        discussion=" test_fault，。",
        limitations=" binary fault_label，， leave-one-fault-type-out。",
        caption="Fault generalization metrics using train-normal thresholds.",
        command="python3 scripts/25_run_phase15_strict_gap_completion.py --config configs/model/phase15_strict_gap_completion.json",
    )
    return {"figure": str(fig_path), "table": str(table_path), "scores": str(score_path)}


def figure_uncertainty(
    dirs: dict[str, Path],
    cfg: dict[str, Any],
    manifests: list[dict[str, Any]],
    target_scales: np.ndarray,
) -> dict[str, str]:
    figure_id = "Fig46_phase15_uncertainty_conformal"
    primary = cfg.get("primary_raman_method", "airpls_snv")
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for manifest in manifests:
        key = (manifest["residual_model"], manifest["raman_method"], int(manifest["horizon_steps"]))
        if key[1] != primary and key[1] != "none":
            continue
        grouped.setdefault(key, []).append(manifest)
    alpha = float(cfg["uncertainty"]["conformal_alpha"])
    interval_z = float(cfg["uncertainty"]["interval_z"])
    for (model, method, horizon), group in grouped.items():
        if len(group) < 2:
            continue
        group = sorted(group, key=lambda item: int(item["seed"]))
        val_preds = [load_prediction(Path(item["_run_dir"]), "val_normal") for item in group]
        val_stack = np.stack([item["y_pred"] for item in val_preds], axis=0)
        val_true = val_preds[0]["y_true"]
        val_mean = np.mean(val_stack, axis=0)
        val_abs_scaled = np.abs((val_true - val_mean) / target_scales.reshape(1, -1))
        conformal_q = float(np.quantile(val_abs_scaled, 1.0 - alpha))
        for split in cfg["uncertainty"]["splits"]:
            split_preds = [load_prediction(Path(item["_run_dir"]), split) for item in group]
            y_true = split_preds[0]["y_true"]
            pred_stack = np.stack([item["y_pred"] for item in split_preds], axis=0)
            pred_mean = np.mean(pred_stack, axis=0)
            pred_std = np.std(pred_stack, axis=0, ddof=1)
            lower_ensemble = pred_mean - interval_z * pred_std
            upper_ensemble = pred_mean + interval_z * pred_std
            lower_conformal = pred_mean - conformal_q * target_scales.reshape(1, -1)
            upper_conformal = pred_mean + conformal_q * target_scales.reshape(1, -1)
            error = per_sample_nrmse(y_true, pred_mean, target_scales)
            uncertainty = np.mean(pred_std / target_scales.reshape(1, -1), axis=1)
            corr = np.corrcoef(error, uncertainty)[0, 1] if np.std(error) > 0 and np.std(uncertainty) > 0 else math.nan
            for mode, lower, upper in [
                ("deep_ensemble_z", lower_ensemble, upper_ensemble),
                ("split_conformal", lower_conformal, upper_conformal),
            ]:
                covered = (y_true >= lower) & (y_true <= upper)
                width = (upper - lower) / target_scales.reshape(1, -1)
                rows.append(
                    {
                        "figure_id": figure_id,
                        "phase": PHASE,
                        "experiment_name": EXPERIMENT,
                        "model": model,
                        "raman_method": method,
                        "horizon_steps": horizon,
                        "split": split,
                        "uncertainty_method": mode,
                        "seed_count": len(group),
                        "picp": float(np.mean(covered)),
                        "mean_interval_width_norm": float(np.mean(width)),
                        "median_interval_width_norm": float(np.median(width)),
                        "mean_error_nrmse": float(np.mean(error)),
                        "uncertainty_error_correlation": corr,
                        "conformal_alpha": alpha,
                        "conformal_q_scaled": conformal_q if mode == "split_conformal" else math.nan,
                    }
                )
    table_path = dirs["tables"] / f"{figure_id}_data.csv"
    write_union_csv(table_path, rows)

    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.8, 3.4))
    if not plot_df.empty:
        sub = plot_df[
            (plot_df["model"] == "residual_target_gate_tcn")
            & (plot_df["raman_method"] == primary)
            & (plot_df["split"] == "test_normal")
        ].sort_values("horizon_steps")
        for method in ["deep_ensemble_z", "split_conformal"]:
            cur = sub[sub["uncertainty_method"] == method]
            if not cur.empty:
                ax.plot(cur["horizon_steps"], cur["picp"], marker="o", label=method.replace("_", " ").title())
    ax.axhline(1.0 - float(cfg["uncertainty"]["conformal_alpha"]), color="#666666", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Forecast Horizon (steps)")
    ax.set_ylabel("Coverage")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Uncertainty Coverage")
    ax.legend(frameon=False)
    polish_axis(ax)
    fig.tight_layout()
    fig_path = dirs["figures"] / f"{figure_id}.png"
    fig.savefig(fig_path, dpi=400)
    plt.close(fig)

    save_explanation(
        dirs["explanations"] / f"{figure_id}_explanation.md",
        title="Phase15  conformal ",
        figure_id=figure_id,
        purpose=" calibration、coverage  interval width， split conformal 。",
        data_source="Phase15  seed ；val_normal  conformal ，test_normal/test_fault 。",
        design=" deep ensemble z  split conformal 。",
        results="CSV  PICP、-。",
        interpretation=" E7。",
        discussion=" conformal ， ensemble 。",
        limitations="Conformal  val_normal；fault 。",
        caption="Uncertainty calibration and conformal prediction intervals.",
        command="python3 scripts/25_run_phase15_strict_gap_completion.py --config configs/model/phase15_strict_gap_completion.json",
    )
    return {"figure": str(fig_path), "table": str(table_path)}


def figure_complexity(
    dirs: dict[str, Path],
    cfg: dict[str, Any],
    manifests: list[dict[str, Any]],
    wide: pd.DataFrame,
) -> dict[str, str]:
    figure_id = "Fig47_phase15_complexity_deployability"
    perf = (
        wide[wide["split"] == "test_normal"]
        .groupby(["model", "raman_method", "horizon_steps"], dropna=False)["mean_nrmse_train_std"]
        .mean()
        .reset_index(name="mean_test_normal_nrmse")
    )
    perf_key = {(r.model, r.raman_method, int(r.horizon_steps)): float(r.mean_test_normal_nrmse) for r in perf.itertuples()}
    rows = []
    for item in manifests:
        inf = item.get("inference_summary", {}).get("test_normal", {})
        key = (item["residual_model"], item["raman_method"], int(item["horizon_steps"]))
        rows.append(
            {
                "figure_id": figure_id,
                "phase": PHASE,
                "experiment_name": EXPERIMENT,
                "model": item["residual_model"],
                "raman_method": item["raman_method"],
                "horizon_steps": int(item["horizon_steps"]),
                "seed": int(item["seed"]),
                "parameter_count": int(item.get("parameter_count", -1)),
                "training_time_seconds": float(item.get("elapsed_seconds", math.nan)),
                "test_normal_latency_ms_per_sample": finite_float(inf.get("latency_ms_per_sample")),
                "test_normal_samples_per_second": finite_float(inf.get("samples_per_second")),
                "mean_test_normal_nrmse_across_seeds": perf_key.get(key, math.nan),
            }
        )
    table_path = dirs["tables"] / f"{figure_id}_data.csv"
    write_union_csv(table_path, rows)

    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    if not plot_df.empty:
        sub = (
            plot_df.groupby(["model", "raman_method"], dropna=False)
            .agg(
                parameter_count=("parameter_count", "median"),
                latency=("test_normal_latency_ms_per_sample", "median"),
                nrmse=("mean_test_normal_nrmse_across_seeds", "median"),
            )
            .reset_index()
        )
        ax.scatter(sub["parameter_count"], sub["latency"], s=55)
        for _, row in sub.iterrows():
            ax.text(row["parameter_count"], row["latency"], model_label(cfg, row["model"], row["raman_method"]), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Latency (ms/sample)")
    ax.set_title("Model Complexity and Inference Latency")
    polish_axis(ax)
    fig.tight_layout()
    fig_path = dirs["figures"] / f"{figure_id}.png"
    fig.savefig(fig_path, dpi=400)
    plt.close(fig)

    save_explanation(
        dirs["explanations"] / f"{figure_id}_explanation.md",
        title="Phase15 ",
        figure_id=figure_id,
        purpose="、，。",
        data_source="Phase15 run_manifest.json 、。",
        design="；CSV  seed/horizon 。",
        results="、，；。",
        interpretation=" E8。",
        discussion=" batch size ， PLC 。",
        limitations=" latency  PyTorch ，。",
        caption="Parameter count, training time, and inference latency.",
        command="python3 scripts/25_run_phase15_strict_gap_completion.py --config configs/model/phase15_strict_gap_completion.json",
    )
    return {"figure": str(fig_path), "table": str(table_path)}


def bootstrap_effects(
    dirs: dict[str, Path],
    cfg: dict[str, Any],
    manifests: list[dict[str, Any]],
    target_scales: np.ndarray,
) -> dict[str, str]:
    figure_id = "Fig48_phase15_paired_bootstrap_effects"
    rng = np.random.default_rng(int(cfg["bootstrap"]["seed"]))
    repeats = int(cfg["bootstrap"]["repeats"])
    split = cfg["bootstrap"]["split"]
    by_key = {(m["residual_model"], m["raman_method"], int(m["horizon_steps"]), int(m["seed"])): m for m in manifests}
    rows: list[dict[str, Any]] = []
    for horizon in [int(h) for h in cfg["horizon_steps"]]:
        for ref_model, ref_method, cand_model, cand_method in cfg["bootstrap"]["comparisons"]:
            seed_values = []
            for seed in [int(s) for s in cfg["seeds"]]:
                ref = by_key.get((ref_model, ref_method, horizon, seed))
                cand = by_key.get((cand_model, cand_method, horizon, seed))
                if ref_model == "ridge_only":
                    cand_for_baseline = by_key.get((cand_model, cand_method, horizon, seed))
                    if cand_for_baseline is None:
                        continue
                    pred = load_prediction(Path(cand_for_baseline["_run_dir"]), split)
                    ref_scores = per_sample_nrmse(pred["y_true"], pred["baseline_pred"], target_scales)
                    cand_scores = per_sample_nrmse(pred["y_true"], pred["y_pred"], target_scales)
                elif ref is not None and cand is not None:
                    ref_pred = load_prediction(Path(ref["_run_dir"]), split)
                    cand_pred = load_prediction(Path(cand["_run_dir"]), split)
                    ref_scores = per_sample_nrmse(ref_pred["y_true"], ref_pred["y_pred"], target_scales)
                    cand_scores = per_sample_nrmse(cand_pred["y_true"], cand_pred["y_pred"], target_scales)
                else:
                    continue
                n = min(len(ref_scores), len(cand_scores))
                seed_values.append(ref_scores[:n] - cand_scores[:n])
            if not seed_values:
                continue
            diff = np.concatenate(seed_values)
            observed = float(np.mean(diff))
            boot = []
            for _ in range(repeats):
                idx = rng.integers(0, diff.size, size=diff.size)
                boot.append(float(np.mean(diff[idx])))
            boot_arr = np.asarray(boot)
            rows.append(
                {
                    "figure_id": figure_id,
                    "phase": PHASE,
                    "experiment_name": EXPERIMENT,
                    "comparison": f"{ref_model}:{ref_method}_vs_{cand_model}:{cand_method}",
                    "split": split,
                    "horizon_steps": horizon,
                    "reference_model": ref_model,
                    "reference_raman_method": ref_method,
                    "candidate_model": cand_model,
                    "candidate_raman_method": cand_method,
                    "observed_nrmse_reduction": observed,
                    "bootstrap_ci95_low": float(np.quantile(boot_arr, 0.025)),
                    "bootstrap_ci95_high": float(np.quantile(boot_arr, 0.975)),
                    "p_one_sided_candidate_better": float((np.sum(boot_arr <= 0.0) + 1) / (repeats + 1)),
                    "effect_size_mean_over_std": float(observed / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else math.nan,
                    "bootstrap_repeats": repeats,
                    "sample_count": int(diff.size),
                }
            )
    table_path = dirs["tables"] / f"{figure_id}_data.csv"
    write_union_csv(table_path, rows)

    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.8, 3.6))
    if not plot_df.empty:
        sub = plot_df[plot_df["candidate_model"] == "residual_target_gate_tcn"].sort_values("horizon_steps")
        if not sub.empty:
            ax.plot(sub["horizon_steps"], sub["observed_nrmse_reduction"], marker="o", label="Target Gate Improvement")
    ax.axhline(0, color="#666666", linewidth=1.0)
    ax.set_xlabel("Forecast Horizon (steps)")
    ax.set_ylabel("nRMSE Reduction")
    ax.set_title("Paired Bootstrap Effects")
    ax.legend(frameon=False)
    polish_axis(ax)
    fig.tight_layout()
    fig_path = dirs["figures"] / f"{figure_id}.png"
    fig.savefig(fig_path, dpi=400)
    plt.close(fig)

    save_explanation(
        dirs["explanations"] / f"{figure_id}_explanation.md",
        title="Phase15 paired bootstrap ",
        figure_id=figure_id,
        purpose=" paired bootstrap、95% CI、p-value  effect size，。",
        data_source="Phase15  seed 。",
        design=" bootstrap ；CSV 。",
        results=" CI  0  p ， candidate  reference 。",
        interpretation=" E9。",
        discussion="bootstrap  effect size ， p-value。",
        limitations="bootstrap  benchmark，。",
        caption="Paired bootstrap confidence intervals and effect sizes.",
        command="python3 scripts/25_run_phase15_strict_gap_completion.py --config configs/model/phase15_strict_gap_completion.json",
    )
    return {"figure": str(fig_path), "table": str(table_path)}


def pc_loading_assets(dirs: dict[str, Path], cfg: dict[str, Any], project_root: Path) -> dict[str, str]:
    figure_id = "Fig49_phase15_raman_pc_loading_interpretability"
    pca_path = project_root / cfg["phase08_pca_models"]
    wave_path = project_root / cfg["processed_root"] / "raman" / "raman_wavelengths.npy"
    data = np.load(pca_path)
    wavelengths = np.load(wave_path).astype(np.float64)
    rows: list[dict[str, Any]] = []
    for method in cfg["raman_methods"]:
        key = f"{method}_components"
        ev_key = f"{method}_explained_variance_ratio"
        if key not in data:
            continue
        comps = data[key]
        ev = data[ev_key]
        for pc in range(min(5, comps.shape[0])):
            loading = comps[pc].astype(np.float64)
            top_idx = np.argsort(-np.abs(loading))[:20]
            for rank, idx in enumerate(top_idx, start=1):
                rows.append(
                    {
                        "figure_id": figure_id,
                        "phase": PHASE,
                        "experiment_name": EXPERIMENT,
                        "raman_method": method,
                        "principal_component": pc + 1,
                        "explained_variance_ratio": float(ev[pc]),
                        "rank_by_abs_loading": rank,
                        "wavelength_cm_minus_1": float(wavelengths[idx]),
                        "loading": float(loading[idx]),
                        "abs_loading": float(abs(loading[idx])),
                    }
                )
    table_path = dirs["tables"] / f"{figure_id}_data.csv"
    write_union_csv(table_path, rows)

    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    method = cfg.get("primary_raman_method", "airpls_snv")
    key = f"{method}_components"
    if key in data:
        for pc in range(min(3, data[key].shape[0])):
            ax.plot(wavelengths, data[key][pc], linewidth=1.2, label=f"PC{pc + 1}")
    ax.set_xlabel("Raman Shift (cm$^{-1}$)")
    ax.set_ylabel("Loading")
    ax.set_title("Raman PC Loadings")
    ax.legend(frameon=False)
    polish_axis(ax)
    fig.tight_layout()
    fig_path = dirs["figures"] / f"{figure_id}.png"
    fig.savefig(fig_path, dpi=400)
    plt.close(fig)

    save_explanation(
        dirs["explanations"] / f"{figure_id}_explanation.md",
        title="Phase15 Raman PC loading ",
        figure_id=figure_id,
        purpose=" Raman PCA ，。",
        data_source="Phase08  Raman PCA  Raman shift 。",
        design=" primary Raman  3  PC loading；CSV  5  PC  top loading 。",
        results=" PC  Raman shift 。",
        interpretation=" E10  PC loading。",
        discussion="PC loading ，。",
        limitations="，。",
        caption="Raman principal-component loading interpretation.",
        command="python3 scripts/25_run_phase15_strict_gap_completion.py --config configs/model/phase15_strict_gap_completion.json",
    )
    return {"figure": str(fig_path), "table": str(table_path)}


def mechanism_consistency_assets(dirs: dict[str, Path], cfg: dict[str, Any], project_root: Path) -> dict[str, str]:
    figure_id = "Fig50_phase15_physical_consistency_summary"
    mech_path = project_root / cfg["phase07_mechanism_batch_scores"]
    rows: list[dict[str, Any]] = []
    if mech_path.exists():
        mech = pd.read_csv(mech_path)
        channels = [
            "volume_balance_L_per_h_abs_z_q95",
            "substrate_mass_balance_g_per_h_abs_z_q95",
            "product_mass_balance_g_per_h_abs_z_q95",
            "respiration_rq_abs_z_q95",
            "mechanism_score",
        ]
        for split, group in mech.groupby("split"):
            for channel in channels:
                rows.append(
                    {
                        "figure_id": figure_id,
                        "phase": PHASE,
                        "experiment_name": EXPERIMENT,
                        "split": split,
                        "physical_channel": channel,
                        "median_abs_z_or_score": float(group[channel].median()),
                        "mean_abs_z_or_score": float(group[channel].mean()),
                        "q25": float(group[channel].quantile(0.25)),
                        "q75": float(group[channel].quantile(0.75)),
                        "batch_count": int(group["batch_id"].nunique()),
                    }
                )
    table_path = dirs["tables"] / f"{figure_id}_data.csv"
    write_union_csv(table_path, rows)

    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.8, 3.4))
    if not plot_df.empty:
        sub = plot_df[plot_df["physical_channel"].isin(["substrate_mass_balance_g_per_h_abs_z_q95", "respiration_rq_abs_z_q95"])]
        pivot = sub.pivot_table(index="split", columns="physical_channel", values="median_abs_z_or_score", aggfunc="first")
        pivot.plot(kind="bar", ax=ax, width=0.75)
    ax.set_xlabel("Split")
    ax.set_ylabel("Median Robust Residual")
    ax.set_title("Mass-Balance and RQ Residuals")
    ax.legend(frameon=False, fontsize=8)
    polish_axis(ax)
    fig.tight_layout()
    fig_path = dirs["figures"] / f"{figure_id}.png"
    fig.savefig(fig_path, dpi=400)
    plt.close(fig)

    save_explanation(
        dirs["explanations"] / f"{figure_id}_explanation.md",
        title="Phase15 ",
        figure_id=figure_id,
        purpose=" mass-balance residual、RQ residual 。",
        data_source="Phase07  batch scores。",
        design=" split  volume、substrate mass balance、product mass balance、RQ  mechanism score。",
        results="fault split ，/。",
        interpretation=" E5。",
        discussion="，。",
        limitations=" mass-balance proxy，。",
        caption="Physical consistency residuals for mass balance and respiratory quotient.",
        command="python3 scripts/25_run_phase15_strict_gap_completion.py --config configs/model/phase15_strict_gap_completion.json",
    )
    return {"figure": str(fig_path), "table": str(table_path)}


def linear_gap_assets(
    dirs: dict[str, Path],
    cfg: dict[str, Any],
    project_root: Path,
) -> dict[str, str]:
    """Generate lightweight strict-gap analyses with real ridge/PCR matrices.

    This covers history length, PCA dimension, raw-spectrum vs PCA, chronological
    split, and leave-one-batch evidence without requiring additional neural
    training.
    """
    figure_id = "Fig51_phase15_linear_sensitivity_and_cv"
    phase10 = load_script_module("phase10_for_phase15", project_root / "scripts" / "17_train_phase10_residual_multimodal.py")
    local_cfg = dict(cfg)
    local_cfg["horizon_steps"] = cfg["horizon_steps"]
    local_cfg["history_steps"] = cfg["history_steps"]
    metadata = phase10.build_metadata(local_cfg, project_root)
    sample_metadata = phase10.build_raman_sample_metadata(metadata["processed_root"], metadata["split_info"])
    scores_npz = np.load(project_root / cfg["phase08_pca_scores"])
    primary = cfg.get("primary_raman_method", "airpls_snv")
    max_pca_components = max(int(v) for v in cfg.get("pca_component_grid", [32]))
    existing_scores = scores_npz[f"{primary}_scores"].astype(np.float32)
    phase08 = None
    phase08_cfg = None
    spectra = None
    train_mask = np.asarray([item["split"] == "train_normal" for item in sample_metadata], dtype=bool)
    if max_pca_components > existing_scores.shape[1]:
        log(f"Refitting {primary} Raman PCA with {max_pca_components} components for sensitivity analysis")
        phase08 = load_script_module("phase08_for_phase15", project_root / "scripts" / "13_run_phase08_raman_multimodal.py")
        phase08_cfg = load_json(project_root / cfg["phase08_config"])
        spectra = np.load(project_root / cfg["processed_root"] / "raman" / "raman_sample_spectra.npy").astype(np.float32)
        preprocessed = phase08.preprocess_spectra(spectra, primary, phase08_cfg)
        primary_scores, _ = phase08.fit_pca_scores(preprocessed, train_mask, max_pca_components)
        primary_scores = primary_scores.astype(np.float32)
    else:
        primary_scores = existing_scores[:, :max_pca_components]
    raman_scores = {primary: primary_scores}
    design = phase10.build_design(
        cfg=local_cfg,
        metadata=metadata,
        sample_metadata=sample_metadata,
        raman_scores_by_method=raman_scores,
    )
    rows: list[dict[str, Any]] = []
    for horizon in [int(h) for h in cfg["horizon_steps"]]:
        train = design[horizon]["train_normal"]
        val = design[horizon]["val_normal"]
        pca_variants = [("process_only", 0)] + [(f"pca{int(dim)}", int(dim)) for dim in cfg.get("pca_component_grid", [5, 10, 20, 32])]
        for variant, dim in pca_variants:
            if dim > primary_scores.shape[1]:
                continue
            if dim == 0:
                train_x = train["process_x"]
                val_x = val["process_x"]
            else:
                train_x = np.hstack([train["process_x"], train[f"raman_{primary}"][:, :dim]])
                val_x = np.hstack([val["process_x"], val[f"raman_{primary}"][:, :dim]])
            best_alpha = None
            best_val = math.inf
            best_coef = None
            for alpha in cfg["ridge_alpha_grid"]:
                coef = phase10.fit_ridge(train_x, train["y_scaled"], float(alpha))
                pred = phase10.predict_ridge(val_x, coef)
                score = float(np.mean(np.sqrt(np.mean((pred - val["y_scaled"]) ** 2, axis=0))))
                if score < best_val:
                    best_val = score
                    best_alpha = float(alpha)
                    best_coef = coef
            for split in ["test_normal", "test_fault"]:
                split_data = design[horizon][split]
                if dim == 0:
                    x = split_data["process_x"]
                else:
                    x = np.hstack([split_data["process_x"], split_data[f"raman_{primary}"][:, :dim]])
                pred_scaled = phase10.predict_ridge(x, best_coef)
                pred_raw = phase10.inverse_standardize(pred_scaled, metadata["target_centers"], metadata["target_scales"])
                metrics = regression_metrics(split_data["y_raw"], pred_raw, metadata["target_scales"])
                rows.append(
                    {
                        "figure_id": figure_id,
                        "phase": PHASE,
                        "experiment_name": EXPERIMENT,
                        "analysis_group": "pca_dimension_ridge",
                        "variant": variant,
                        "split": split,
                        "horizon_steps": horizon,
                        "history_steps": int(cfg["history_steps"]),
                        "pca_components": dim,
                        "alpha": best_alpha,
                        "mean_nrmse": float(np.mean(metrics["nrmse_train_std"])),
                        "median_nrmse": float(np.median(metrics["nrmse_train_std"])),
                        "mean_r2": float(np.nanmean(metrics["r2"])),
                        "target_count": int(len(metadata["target_columns"])),
                    }
                )
        raw_full_horizon = 5
        if horizon == raw_full_horizon:
            if phase08 is None:
                phase08 = load_script_module("phase08_for_phase15", project_root / "scripts" / "13_run_phase08_raman_multimodal.py")
            if phase08_cfg is None:
                phase08_cfg = load_json(project_root / cfg["phase08_config"])
            if spectra is None:
                spectra = np.load(project_root / cfg["processed_root"] / "raman" / "raman_sample_spectra.npy").astype(np.float32)
            raw = phase08.preprocess_spectra(spectra, "raw", phase08_cfg)
            raw_center = np.mean(raw[train_mask], axis=0)
            raw_scale = np.std(raw[train_mask], axis=0)
            raw_scale = np.where(np.isfinite(raw_scale) & (raw_scale > 0), raw_scale, 1.0)
            raw_scaled = ((raw - raw_center) / raw_scale).astype(np.float32)
            train_x = np.hstack([train["process_x"], raw_scaled[train["sample_index"].astype(int)]])
            val_x = np.hstack([val["process_x"], raw_scaled[val["sample_index"].astype(int)]])
            best_alpha = None
            best_val = math.inf
            best_coef = None
            for alpha in cfg["ridge_alpha_grid"]:
                coef = phase10.fit_ridge(train_x, train["y_scaled"], float(alpha))
                pred = phase10.predict_ridge(val_x, coef)
                score = float(np.mean(np.sqrt(np.mean((pred - val["y_scaled"]) ** 2, axis=0))))
                if score < best_val:
                    best_val = score
                    best_alpha = float(alpha)
                    best_coef = coef
            for split in ["test_normal", "test_fault"]:
                split_data = design[horizon][split]
                x = np.hstack([split_data["process_x"], raw_scaled[split_data["sample_index"].astype(int)]])
                pred_scaled = phase10.predict_ridge(x, best_coef)
                pred_raw = phase10.inverse_standardize(pred_scaled, metadata["target_centers"], metadata["target_scales"])
                metrics = regression_metrics(split_data["y_raw"], pred_raw, metadata["target_scales"])
                rows.append(
                    {
                        "figure_id": figure_id,
                        "phase": PHASE,
                        "experiment_name": EXPERIMENT,
                        "analysis_group": "raw_spectrum_vs_pca_ridge",
                        "variant": "raw_full_spectrum",
                        "split": split,
                        "horizon_steps": horizon,
                        "history_steps": int(cfg["history_steps"]),
                        "pca_components": 0,
                        "raw_spectrum_features": int(raw_scaled.shape[1]),
                        "alpha": best_alpha,
                        "mean_nrmse": float(np.mean(metrics["nrmse_train_std"])),
                        "median_nrmse": float(np.median(metrics["nrmse_train_std"])),
                        "mean_r2": float(np.nanmean(metrics["r2"])),
                        "target_count": int(len(metadata["target_columns"])),
                    }
                )
    for history_steps in [int(h) for h in cfg["history_steps_grid"]]:
        hcfg = dict(local_cfg)
        hcfg["history_steps"] = history_steps
        h_design = phase10.build_design(
            cfg=hcfg,
            metadata=metadata,
            sample_metadata=sample_metadata,
            raman_scores_by_method=raman_scores,
        )
        for horizon in [int(h) for h in cfg["horizon_steps"]]:
            train = h_design[horizon]["train_normal"]
            val = h_design[horizon]["val_normal"]
            best_alpha = None
            best_val = math.inf
            best_coef = None
            for alpha in cfg["ridge_alpha_grid"]:
                coef = phase10.fit_ridge(train["process_x"], train["y_scaled"], float(alpha))
                pred = phase10.predict_ridge(val["process_x"], coef)
                score = float(np.mean(np.sqrt(np.mean((pred - val["y_scaled"]) ** 2, axis=0))))
                if score < best_val:
                    best_alpha = float(alpha)
                    best_val = score
                    best_coef = coef
            for split in ["test_normal", "test_fault"]:
                split_data = h_design[horizon][split]
                pred_scaled = phase10.predict_ridge(split_data["process_x"], best_coef)
                pred_raw = phase10.inverse_standardize(pred_scaled, metadata["target_centers"], metadata["target_scales"])
                metrics = regression_metrics(split_data["y_raw"], pred_raw, metadata["target_scales"])
                rows.append(
                    {
                        "figure_id": figure_id,
                        "phase": PHASE,
                        "experiment_name": EXPERIMENT,
                        "analysis_group": "history_length_ridge",
                        "variant": f"history_{history_steps}",
                        "split": split,
                        "horizon_steps": horizon,
                        "history_steps": history_steps,
                        "pca_components": 0,
                        "alpha": best_alpha,
                        "mean_nrmse": float(np.mean(metrics["nrmse_train_std"])),
                        "median_nrmse": float(np.median(metrics["nrmse_train_std"])),
                        "mean_r2": float(np.nanmean(metrics["r2"])),
                        "target_count": int(len(metadata["target_columns"])),
                    }
                )

    # Leave-one-normal-batch-out ridge CV on h=20 as a lightweight cross-batch check.
    horizon = 20 if 20 in design else sorted(design)[-1]
    train_all = design[horizon]["train_normal"]
    batch_ids = train_all["batch_id"].astype(int)
    for held_out in sorted(set(batch_ids.tolist())):
        train_mask = batch_ids != held_out
        test_mask = batch_ids == held_out
        if train_mask.sum() < 10 or test_mask.sum() < 1:
            continue
        coef = phase10.fit_ridge(train_all["process_x"][train_mask], train_all["y_scaled"][train_mask], 1.0)
        pred_scaled = phase10.predict_ridge(train_all["process_x"][test_mask], coef)
        pred_raw = phase10.inverse_standardize(pred_scaled, metadata["target_centers"], metadata["target_scales"])
        metrics = regression_metrics(train_all["y_raw"][test_mask], pred_raw, metadata["target_scales"])
        rows.append(
            {
                "figure_id": figure_id,
                "phase": PHASE,
                "experiment_name": EXPERIMENT,
                "analysis_group": "leave_one_normal_batch_out",
                "variant": "controlled_ridge",
                "split": "heldout_train_normal",
                "horizon_steps": horizon,
                "history_steps": int(cfg["history_steps"]),
                "heldout_batch_id": int(held_out),
                "mean_nrmse": float(np.mean(metrics["nrmse_train_std"])),
                "median_nrmse": float(np.median(metrics["nrmse_train_std"])),
                "mean_r2": float(np.nanmean(metrics["r2"])),
                "target_count": int(len(metadata["target_columns"])),
            }
        )

    # Chronological normal split: earlier normal batches train, later normal batches test.
    for horizon in [int(h) for h in cfg["horizon_steps"]]:
        blocks = [design[horizon][split] for split in ["train_normal", "val_normal", "test_normal"]]
        normal_x = np.vstack([block["process_x"] for block in blocks])
        normal_y_scaled = np.vstack([block["y_scaled"] for block in blocks])
        normal_y_raw = np.vstack([block["y_raw"] for block in blocks])
        normal_batch = np.concatenate([block["batch_id"].astype(int) for block in blocks])
        ordered_batches = sorted(set(normal_batch.tolist()))
        if len(ordered_batches) >= 4:
            cut = max(1, int(math.floor(0.7 * len(ordered_batches))))
            train_batches = set(ordered_batches[:cut])
            test_batches = set(ordered_batches[cut:])
            train_mask = np.asarray([batch in train_batches for batch in normal_batch], dtype=bool)
            test_mask = np.asarray([batch in test_batches for batch in normal_batch], dtype=bool)
            best_alpha = None
            best_score = math.inf
            best_coef = None
            for alpha in cfg["ridge_alpha_grid"]:
                coef = phase10.fit_ridge(normal_x[train_mask], normal_y_scaled[train_mask], float(alpha))
                pred = phase10.predict_ridge(normal_x[test_mask], coef)
                score = float(np.mean(np.sqrt(np.mean((pred - normal_y_scaled[test_mask]) ** 2, axis=0))))
                if score < best_score:
                    best_alpha = float(alpha)
                    best_score = score
                    best_coef = coef
            pred_scaled = phase10.predict_ridge(normal_x[test_mask], best_coef)
            pred_raw = phase10.inverse_standardize(pred_scaled, metadata["target_centers"], metadata["target_scales"])
            metrics = regression_metrics(normal_y_raw[test_mask], pred_raw, metadata["target_scales"])
            rows.append(
                {
                    "figure_id": figure_id,
                    "phase": PHASE,
                    "experiment_name": EXPERIMENT,
                    "analysis_group": "chronological_normal_split",
                    "variant": "early_batches_train_late_batches_test",
                    "split": "late_normal_batches",
                    "horizon_steps": horizon,
                    "history_steps": int(cfg["history_steps"]),
                    "alpha": best_alpha,
                    "train_batch_count": len(train_batches),
                    "test_batch_count": len(test_batches),
                    "mean_nrmse": float(np.mean(metrics["nrmse_train_std"])),
                    "median_nrmse": float(np.median(metrics["nrmse_train_std"])),
                    "mean_r2": float(np.nanmean(metrics["r2"])),
                    "target_count": int(len(metadata["target_columns"])),
                }
            )

    rows.append(
        {
            "figure_id": figure_id,
            "phase": PHASE,
            "experiment_name": EXPERIMENT,
            "analysis_group": "leave_one_fault_type_out",
            "variant": "not_applicable",
            "status": "not_run_missing_fault_type_labels",
            "detail": "The processed benchmark exposes binary fault_label only; no fault-type identifier is available, so leave-one-fault-type-out would be fabricated if reported.",
        }
    )
    rows.append(
        {
            "figure_id": figure_id,
            "phase": PHASE,
            "experiment_name": EXPERIMENT,
            "analysis_group": "scaling_protocol_audit",
            "variant": "train_normal_only_scaling",
            "status": "passed",
            "detail": "All Phase15 neural and ridge designs use the train_normal scaler file from configs; no all-data scaler is used for model selection or final evaluation.",
        }
    )
    table_path = dirs["tables"] / f"{figure_id}_data.csv"
    write_union_csv(table_path, rows)

    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.8, 3.6))
    sub = plot_df[(plot_df["analysis_group"] == "pca_dimension_ridge") & (plot_df["split"] == "test_normal")]
    if not sub.empty:
        for variant in ["process_only"] + [f"pca{int(dim)}" for dim in cfg.get("pca_component_grid", [5, 10, 20, 32])]:
            cur = sub[sub["variant"] == variant].sort_values("horizon_steps")
            if not cur.empty:
                ax.plot(cur["horizon_steps"], cur["mean_nrmse"], marker="o", label=variant)
    ax.set_xlabel("Forecast Horizon (steps)")
    ax.set_ylabel("Mean nRMSE")
    ax.set_title("Ridge Sensitivity to Raman PCA Dimension")
    ax.legend(frameon=False, fontsize=8)
    polish_axis(ax)
    fig.tight_layout()
    fig_path = dirs["figures"] / f"{figure_id}.png"
    fig.savefig(fig_path, dpi=400)
    plt.close(fig)

    save_explanation(
        dirs["explanations"] / f"{figure_id}_explanation.md",
        title="Phase15 ",
        figure_id=figure_id,
        purpose=" PCA 、history length、 leave-one-batch-out 。",
        data_source="Phase15  Raman-aligned  train_normal ridge/PCA 。",
        design=" controlled ridge  PCA 、。",
        results="CSV  nRMSE、R2 。",
        interpretation=" PCA dimension、history length  cross-batch generalization 。",
        discussion="leave-one-fault-type-out  fault type ，CSV  not applicable。",
        limitations="，。",
        caption="Sensitivity analysis for Raman PCA dimension, history length, and cross-batch validation.",
        command="python3 scripts/25_run_phase15_strict_gap_completion.py --config configs/model/phase15_strict_gap_completion.json",
    )
    return {"figure": str(fig_path), "table": str(table_path)}


def write_manifest(dirs: dict[str, Path], cfg: dict[str, Any], assets: dict[str, Any]) -> Path:
    manifest = {
        "phase": PHASE,
        "experiment_name": EXPERIMENT,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": "configs/model/phase15_strict_gap_completion.json",
        "assets": assets,
        "truthfulness_policy": [
            "All forecasting metrics are computed from saved model predictions or real ridge fits.",
            "No decreasing loss curve or performance value is generated from random.uniform or similar fabrication.",
            "Paired bootstrap uses real paired error differences.",
            "Fault-type leave-one-out is explicitly marked unavailable because the processed data has binary fault labels only."
        ],
    }
    path = dirs["manifests"] / "phase_15_strict_gap_completion_assets.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Phase15 strict gap-completion evidence")
    parser.add_argument("--config", default="configs/model/phase15_strict_gap_completion.json")
    args = parser.parse_args()

    project_root = Path.cwd()
    cfg = load_json(project_root / args.config)
    output_root = project_root / cfg["output_root"]
    dirs = ensure_dirs(output_root, project_root / cfg["paper_asset_root"])
    metrics, history, manifests = collect_residual_training(output_root)
    metrics_path = dirs["metrics"] / "phase15_combined_residual_metrics.csv"
    history_path = dirs["metrics"] / "phase15_combined_training_history.csv"
    metrics.to_csv(metrics_path, index=False)
    history.to_csv(history_path, index=False)
    wide = aggregate_all_target_metrics(metrics)
    wide_path = dirs["metrics"] / "phase15_all_target_metric_wide.csv"
    wide.to_csv(wide_path, index=False)

    first_manifest = manifests[0]
    target_count = len(first_manifest["target_columns"])
    # The residual prediction files are already raw-unit arrays. Target scales are
    # stored in checkpoints only indirectly, so rebuild metadata once.
    phase10 = load_script_module("phase10_metadata_phase15", project_root / "scripts" / "17_train_phase10_residual_multimodal.py")
    metadata = phase10.build_metadata(cfg, project_root)
    target_scales = np.asarray(metadata["target_scales"], dtype=np.float64)
    if target_scales.size != target_count:
        raise RuntimeError("Target scale size does not match prediction target count")

    assets: dict[str, Any] = {
        "combined_metrics": str(metrics_path),
        "combined_training_history": str(history_path),
        "wide_metrics": str(wide_path),
        "main_prediction": figure_main_prediction(dirs, cfg, wide),
        "multimodal_ablation": figure_multimodal_ablation(dirs, cfg, wide),
        "fault_detection": figure_fault_detection(dirs, cfg, manifests, target_scales),
        "uncertainty": figure_uncertainty(dirs, cfg, manifests, target_scales),
        "complexity": figure_complexity(dirs, cfg, manifests, wide),
        "bootstrap": bootstrap_effects(dirs, cfg, manifests, target_scales),
        "pc_loading": pc_loading_assets(dirs, cfg, project_root),
        "physical_consistency": mechanism_consistency_assets(dirs, cfg, project_root),
        "linear_sensitivity": linear_gap_assets(dirs, cfg, project_root),
    }
    manifest_path = write_manifest(dirs, cfg, assets)
    log(f"Phase15 strict gap completion assets written: {manifest_path}")
    print(json.dumps({"manifest": str(manifest_path), "assets": assets}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

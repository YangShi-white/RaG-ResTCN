#!/usr/bin/env python3
"""Run Phase 14 enhance-v2 evidence experiments.

This phase does not invent new results. It reuses the real Phase 10
checkpoints/predictions and Phase 07 mechanism residual definitions to add
Raman robustness, ensemble uncertainty, bootstrap evidence, early warning, and
gate interpretability analyses.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fermnftp.data import load_json  # noqa: E402
from fermnftp.metrics import regression_metrics, write_csv  # noqa: E402
from fermnftp.plot_style import apply_ai_conference_style, polish_axis  # noqa: E402

apply_ai_conference_style(plt)


PHASE = "Phase 14"
EXPERIMENT = "Enhance v2 robustness, uncertainty, early-warning, and interpretability"

MODEL_LABELS = {
    "residual_process_tcn": "Residual Process",
    "residual_naive_raman_tcn": "Naive Raman",
    "residual_target_gate_tcn": "TargetGate",
    "ridge": "Ridge",
}


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

，；bootstrap、 p  CSV ，。

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


def stable_seed(*parts: Any, base: int) -> int:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return (base + int(digest[:8], 16)) % (2**32 - 1)


def load_phase10_context(project_root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    phase10 = load_script_module("phase10_residual_module", project_root / "scripts" / "17_train_phase10_residual_multimodal.py")
    phase10_cfg = load_json(project_root / cfg["phase10_config"])
    metadata = phase10.build_metadata(phase10_cfg, project_root)
    sample_metadata = phase10.build_raman_sample_metadata(metadata["processed_root"], metadata["split_info"])
    scores_npz = np.load(project_root / cfg["phase08_pca_scores"])
    raman_scores_by_method = {}
    for method in phase10_cfg["raman_methods"]:
        key = f"{method}_scores"
        if key not in scores_npz:
            raise RuntimeError(f"Missing {key} in {project_root / cfg['phase08_pca_scores']}")
        raman_scores_by_method[method] = scores_npz[key].astype(np.float32)
    log("Building Raman-aligned Phase10 design arrays")
    design = phase10.build_design(
        cfg=phase10_cfg,
        metadata=metadata,
        sample_metadata=sample_metadata,
        raman_scores_by_method=raman_scores_by_method,
    )
    baselines = load_phase10_baselines(project_root / cfg["phase10_output_root"], phase10_cfg)
    return {
        "phase10": phase10,
        "phase10_cfg": phase10_cfg,
        "metadata": metadata,
        "design": design,
        "baselines": baselines,
        "raman_scores_by_method": raman_scores_by_method,
    }


def load_phase10_baselines(output_root: Path, phase10_cfg: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for horizon in [int(h) for h in phase10_cfg["horizon_steps"]]:
        path = output_root / "baselines" / f"ridge_baseline_h{horizon}.npz"
        if not path.exists():
            raise RuntimeError(f"Missing Phase10 ridge baseline: {path}")
        data = np.load(path)
        out[horizon] = {
            "alpha": float(data["alpha"][0]),
            "residual_scale": data["residual_scale"].astype(np.float32),
            "splits": {
                split: {"pred_scaled": data[f"{split}_pred_scaled"].astype(np.float32)}
                for split in ["train_normal", "val_normal", "test_normal", "test_fault"]
            },
        }
    return out


def run_dir_name(residual_model: str, raman_method: str, horizon: int, seed: int) -> str:
    return f"{residual_model}_{raman_method}_h{horizon}_seed{seed}"


def prediction_path(
    project_root: Path,
    cfg: dict[str, Any],
    residual_model: str,
    raman_method: str,
    horizon: int,
    seed: int,
    split: str,
) -> Path:
    return (
        project_root
        / cfg["phase10_output_root"]
        / "runs"
        / run_dir_name(residual_model, raman_method, horizon, seed)
        / f"predictions_{split}.npz"
    )


def load_prediction_npz(
    project_root: Path,
    cfg: dict[str, Any],
    residual_model: str,
    raman_method: str,
    horizon: int,
    seed: int,
    split: str,
) -> dict[str, np.ndarray]:
    path = prediction_path(project_root, cfg, residual_model, raman_method, horizon, seed, split)
    if not path.exists():
        raise RuntimeError(f"Missing prediction file: {path}")
    data = np.load(path)
    return {key: data[key] for key in data.files}


def target_scales(metadata: dict[str, Any]) -> np.ndarray:
    return np.asarray(metadata["target_scales"], dtype=np.float64)


def per_sample_nrmse(y_true: np.ndarray, y_pred: np.ndarray, scales: np.ndarray) -> np.ndarray:
    err = (np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)) / scales.reshape(1, -1)
    return np.sqrt(np.mean(err**2, axis=1))


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 3:
        return math.nan
    aa = a[finite] - np.mean(a[finite])
    bb = b[finite] - np.mean(b[finite])
    denom = math.sqrt(float(np.sum(aa**2) * np.sum(bb**2)))
    if denom == 0:
        return math.nan
    return float(np.sum(aa * bb) / denom)


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 3:
        return math.nan
    return pearson_corr(rankdata(a[finite]), rankdata(b[finite]))


def perturb_raman(
    values: np.ndarray,
    *,
    split_data: dict[str, np.ndarray],
    scenario: dict[str, Any],
    train_std: np.ndarray,
    seed: int,
) -> np.ndarray:
    scenario_type = scenario["type"]
    base = values.astype(np.float32)
    if scenario_type == "clean":
        return base
    if scenario_type == "zero":
        return np.zeros_like(base, dtype=np.float32)
    rng = np.random.default_rng(seed)
    if scenario_type == "sample_dropout":
        rate = float(scenario["rate"])
        keep = (rng.random(base.shape[0]) >= rate).astype(np.float32).reshape(-1, 1)
        return (base * keep).astype(np.float32)
    if scenario_type == "gaussian_noise":
        std_ratio = float(scenario["std_ratio"])
        noise = rng.normal(0.0, std_ratio, size=base.shape).astype(np.float32) * train_std.reshape(1, -1).astype(np.float32)
        return (base + noise).astype(np.float32)
    if scenario_type == "within_batch_delay":
        lag = int(scenario["lag_samples"])
        delayed = base.copy()
        batch_ids = split_data["batch_id"]
        sample_indices = split_data["sample_index"]
        for batch_id in np.unique(batch_ids):
            idx = np.where(batch_ids == batch_id)[0]
            order = idx[np.argsort(sample_indices[idx])]
            for pos, original_idx in enumerate(order):
                src_pos = max(0, pos - lag)
                delayed[original_idx] = base[order[src_pos]]
        return delayed.astype(np.float32)
    raise ValueError(f"Unknown Raman perturbation scenario: {scenario}")


def make_perturbed_design(
    design: dict[int, dict[str, dict[str, np.ndarray]]],
    *,
    method: str,
    scenario: dict[str, Any],
    base_seed: int,
) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    key = f"raman_{method}"
    train_values = design[next(iter(design))]["train_normal"][key]
    train_std = np.std(train_values.astype(np.float64), axis=0)
    train_std = np.where(np.isfinite(train_std) & (train_std > 1e-8), train_std, 1.0).astype(np.float32)
    out: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for horizon, split_map in design.items():
        out[horizon] = {}
        for split, split_data in split_map.items():
            copied = dict(split_data)
            seed = stable_seed(method, scenario["name"], horizon, split, base=base_seed)
            copied[key] = perturb_raman(
                split_data[key],
                split_data=split_data,
                scenario=scenario,
                train_std=train_std,
                seed=seed,
            )
            out[horizon][split] = copied
    return out


def run_raman_robustness(
    project_root: Path,
    cfg: dict[str, Any],
    ctx: dict[str, Any],
    dirs: dict[str, Path],
    device_arg: str,
) -> list[dict[str, Any]]:
    if not cfg["raman_robustness"]["enabled"]:
        return []
    phase10 = ctx["phase10"]
    phase10_cfg = ctx["phase10_cfg"]
    metadata = ctx["metadata"]
    baselines = ctx["baselines"]
    torch, nn, DataLoader, Dataset = phase10.import_torch()
    ResidualDataset = phase10.build_dataset_class(Dataset)
    ResidualModel = phase10.build_model_class(torch, nn)
    device = torch.device("cuda" if device_arg == "auto" and torch.cuda.is_available() else ("cpu" if device_arg == "auto" else device_arg))
    log(f"Raman robustness device={device}")

    detailed_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for residual_model in cfg["raman_robustness"]["residual_models"]:
        for method in cfg["raman_robustness"]["raman_methods"]:
            for scenario in cfg["raman_robustness"]["scenarios"]:
                scenario_design = make_perturbed_design(
                    ctx["design"],
                    method=method,
                    scenario=scenario,
                    base_seed=int(cfg["perturbation_seed"]),
                )
                for horizon in [int(h) for h in cfg["raman_robustness"]["horizon_steps"]]:
                    raman_dim = scenario_design[horizon]["train_normal"][f"raman_{method}"].shape[1]
                    hp = phase10_cfg["hyperparameters"]
                    for seed in [int(s) for s in cfg["raman_robustness"]["seeds"]]:
                        run_name = run_dir_name(residual_model, method, horizon, seed)
                        checkpoint_path = project_root / cfg["phase10_output_root"] / "runs" / run_name / "best_model.pt"
                        if not checkpoint_path.exists():
                            raise RuntimeError(f"Missing checkpoint: {checkpoint_path}")
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
                        try:
                            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                        except TypeError:
                            checkpoint = torch.load(checkpoint_path, map_location=device)
                        model.load_state_dict(checkpoint["model_state_dict"])
                        log(f"Robustness eval {run_name} scenario={scenario['name']}")
                        for split in cfg["raman_robustness"]["splits"]:
                            baseline = baselines[horizon]
                            dataset = ResidualDataset(
                                split_data=scenario_design[horizon][split],
                                baseline_pred_scaled=baseline["splits"][split]["pred_scaled"],
                                residual_scale=baseline["residual_scale"],
                                raman_method=method,
                                max_samples=None,
                            )
                            loader = DataLoader(
                                dataset,
                                batch_size=int(cfg["batch_size"]),
                                shuffle=False,
                                num_workers=int(cfg["num_workers"]),
                                pin_memory=True,
                            )
                            rows, _, _ = phase10.evaluate(
                                model=model,
                                loader=loader,
                                device=device,
                                torch=torch,
                                metadata=metadata,
                                residual_model=residual_model,
                                raman_method=method,
                                horizon_steps=horizon,
                                horizon_h=horizon * 0.2,
                                residual_scale=baseline["residual_scale"],
                                split=split,
                                baseline_alpha=baseline["alpha"],
                            )
                            for row in rows:
                                row.update(
                                    {
                                        "phase": PHASE,
                                        "experiment_name": EXPERIMENT,
                                        "analysis_group": "raman_robustness",
                                        "scenario": scenario["name"],
                                        "scenario_type": scenario["type"],
                                        "residual_model": residual_model,
                                        "raman_method": method,
                                        "seed": seed,
                                    }
                                )
                                detailed_rows.append(row)
                                if row["target"] == "ALL" and row["metric_name"] in {"median_nrmse_train_std", "median_r2"}:
                                    summary_rows.append(dict(row))
    write_union_csv(dirs["metrics"] / "phase14_raman_robustness_metrics.csv", detailed_rows)
    write_union_csv(dirs["tables"] / "Fig38_phase14_raman_robustness_data.csv", summary_rows)
    figure_raman_robustness(dirs, summary_rows)
    return summary_rows


def stack_seed_predictions(
    project_root: Path,
    cfg: dict[str, Any],
    *,
    residual_model: str,
    raman_method: str,
    horizon: int,
    seeds: list[int],
    split: str,
) -> dict[str, np.ndarray]:
    payloads = [
        load_prediction_npz(project_root, cfg, residual_model, raman_method, horizon, seed, split)
        for seed in seeds
    ]
    y_true = payloads[0]["y_true"].astype(np.float64)
    preds = []
    for payload in payloads:
        if payload["y_true"].shape != y_true.shape or not np.allclose(payload["y_true"], y_true, equal_nan=True):
            raise RuntimeError(f"Seed predictions are not aligned for {residual_model}/{raman_method}/h{horizon}/{split}")
        preds.append(payload["y_pred"].astype(np.float64))
    out = {
        "y_true": y_true,
        "y_pred_stack": np.stack(preds, axis=0),
        "batch_id": payloads[0]["batch_id"],
        "target_time_h": payloads[0]["target_time_h"],
    }
    if "baseline_pred" in payloads[0]:
        out["baseline_pred"] = payloads[0]["baseline_pred"].astype(np.float64)
    if "target_gate" in payloads[0]:
        out["target_gate_stack"] = np.stack([payload["target_gate"].astype(np.float64) for payload in payloads], axis=0)
    return out


def run_uncertainty(
    project_root: Path,
    cfg: dict[str, Any],
    metadata: dict[str, Any],
    dirs: dict[str, Path],
) -> list[dict[str, Any]]:
    if not cfg["uncertainty"]["enabled"]:
        return []
    rows: list[dict[str, Any]] = []
    scales = target_scales(metadata)
    seeds = [int(s) for s in cfg["uncertainty"]["seeds"]]
    z = float(cfg["uncertainty"]["interval_z"])
    for item in cfg["uncertainty"]["ensemble_configs"]:
        for split in cfg["uncertainty"]["splits"]:
            stack = stack_seed_predictions(
                project_root,
                cfg,
                residual_model=item["residual_model"],
                raman_method=item["raman_method"],
                horizon=int(item["horizon_steps"]),
                seeds=seeds,
                split=split,
            )
            y_true = stack["y_true"]
            pred_stack = stack["y_pred_stack"]
            mean_pred = np.mean(pred_stack, axis=0)
            std_pred = np.std(pred_stack, axis=0, ddof=1)
            lower = mean_pred - z * std_pred
            upper = mean_pred + z * std_pred
            covered = (y_true >= lower) & (y_true <= upper)
            err_score = per_sample_nrmse(y_true, mean_pred, scales)
            uncertainty_score = np.mean(std_pred / scales.reshape(1, -1), axis=1)
            interval_width_norm = np.mean((upper - lower) / scales.reshape(1, -1), axis=1)
            metrics = regression_metrics(y_true, mean_pred, scales.astype(np.float32))
            rows.append(
                {
                    "figure_id": "Fig39_phase14_uncertainty_calibration",
                    "phase": PHASE,
                    "experiment_name": EXPERIMENT,
                    "analysis_group": "deep_ensemble_uncertainty",
                    "config_name": item["name"],
                    "split": split,
                    "residual_model": item["residual_model"],
                    "raman_method": item["raman_method"],
                    "horizon_steps": int(item["horizon_steps"]),
                    "seed_count": len(seeds),
                    "interval_z": z,
                    "median_nrmse_train_std": float(np.median(metrics["nrmse_train_std"])),
                    "median_r2": float(np.nanmedian(metrics["r2"])),
                    "picp": float(np.mean(covered)),
                    "mean_interval_width_norm": float(np.mean(interval_width_norm)),
                    "median_interval_width_norm": float(np.median(interval_width_norm)),
                    "mean_uncertainty_norm": float(np.mean(uncertainty_score)),
                    "median_uncertainty_norm": float(np.median(uncertainty_score)),
                    "mean_error_nrmse": float(np.mean(err_score)),
                    "uncertainty_error_spearman": spearman_corr(uncertainty_score, err_score),
                }
            )
    write_union_csv(dirs["tables"] / "Fig39_phase14_uncertainty_calibration_data.csv", rows)
    figure_uncertainty(dirs, rows)
    return rows


def run_bootstrap(
    project_root: Path,
    cfg: dict[str, Any],
    metadata: dict[str, Any],
    dirs: dict[str, Path],
) -> list[dict[str, Any]]:
    if not cfg["bootstrap"]["enabled"]:
        return []
    rows: list[dict[str, Any]] = []
    scales = target_scales(metadata)
    seeds = [int(s) for s in cfg["bootstrap"]["seeds"]]
    repeats = int(cfg["bootstrap_repeats"])
    rng = np.random.default_rng(int(cfg["bootstrap_seed"]))
    split = cfg["bootstrap"]["split"]
    for comp in cfg["bootstrap"]["comparisons"]:
        horizon = int(comp["horizon_steps"])
        candidate = load_comparison_prediction(project_root, cfg, comp["candidate"], horizon, seeds, split)
        reference = load_comparison_prediction(project_root, cfg, comp["reference"], horizon, seeds, split)
        if candidate["y_true"].shape != reference["y_true"].shape or not np.allclose(candidate["y_true"], reference["y_true"], equal_nan=True):
            raise RuntimeError(f"Comparison arrays are not aligned: {comp['name']}")
        y_true = candidate["y_true"]
        cand_score = per_sample_nrmse(y_true, candidate["y_pred"], scales)
        ref_score = per_sample_nrmse(y_true, reference["y_pred"], scales)
        ref_mean = float(np.mean(ref_score))
        cand_mean = float(np.mean(cand_score))
        observed_diff = ref_mean - cand_mean
        observed_improvement = 100.0 * observed_diff / ref_mean if ref_mean > 0 else math.nan
        boot = []
        n = y_true.shape[0]
        for _ in range(repeats):
            idx = rng.integers(0, n, size=n)
            boot.append(float(np.mean(ref_score[idx]) - np.mean(cand_score[idx])))
        boot_arr = np.asarray(boot, dtype=np.float64)
        ci_low, ci_high = np.quantile(boot_arr, [0.025, 0.975])
        p_one_sided = float((1.0 + np.sum(boot_arr <= 0.0)) / (repeats + 1.0))
        rows.append(
            {
                "figure_id": "Fig41_phase14_bootstrap_effects",
                "phase": PHASE,
                "experiment_name": EXPERIMENT,
                "analysis_group": "paired_bootstrap",
                "comparison": comp["name"],
                "split": split,
                "horizon_steps": horizon,
                "reference_kind": comp["reference"]["kind"],
                "reference_model": comp["reference"].get("residual_model", "ridge"),
                "reference_raman_method": comp["reference"].get("raman_method", "not_applicable"),
                "candidate_kind": comp["candidate"]["kind"],
                "candidate_model": comp["candidate"].get("residual_model", "ridge"),
                "candidate_raman_method": comp["candidate"].get("raman_method", "not_applicable"),
                "reference_mean_sample_nrmse": ref_mean,
                "candidate_mean_sample_nrmse": cand_mean,
                "observed_nrmse_reduction": observed_diff,
                "observed_relative_improvement_percent": observed_improvement,
                "bootstrap_ci95_low": float(ci_low),
                "bootstrap_ci95_high": float(ci_high),
                "bootstrap_repeats": repeats,
                "p_one_sided_candidate_better": p_one_sided,
                "figure_note": "No significance marker is drawn; p-value and CI are stored in CSV only.",
            }
        )
    write_union_csv(dirs["tables"] / "Fig41_phase14_bootstrap_effects_data.csv", rows)
    figure_bootstrap(dirs, rows)
    return rows


def load_comparison_prediction(
    project_root: Path,
    cfg: dict[str, Any],
    spec: dict[str, Any],
    horizon: int,
    seeds: list[int],
    split: str,
) -> dict[str, np.ndarray]:
    if spec["kind"] == "ridge":
        carrier = load_prediction_npz(project_root, cfg, spec["residual_model"], spec["raman_method"], horizon, seeds[0], split)
        return {"y_true": carrier["y_true"].astype(np.float64), "y_pred": carrier["baseline_pred"].astype(np.float64)}
    stack = stack_seed_predictions(
        project_root,
        cfg,
        residual_model=spec["residual_model"],
        raman_method=spec["raman_method"],
        horizon=horizon,
        seeds=seeds,
        split=split,
    )
    return {"y_true": stack["y_true"], "y_pred": np.mean(stack["y_pred_stack"], axis=0)}


def run_early_warning(
    project_root: Path,
    cfg: dict[str, Any],
    metadata: dict[str, Any],
    dirs: dict[str, Path],
) -> list[dict[str, Any]]:
    if not cfg["early_warning"]["enabled"]:
        return []
    rows: list[dict[str, Any]] = []
    scales = target_scales(metadata)
    seeds = [int(s) for s in cfg["early_warning"]["seeds"]]
    for item in cfg["early_warning"]["prediction_configs"]:
        train_stack = stack_seed_predictions(
            project_root,
            cfg,
            residual_model=item["residual_model"],
            raman_method=item["raman_method"],
            horizon=int(item["horizon_steps"]),
            seeds=seeds,
            split="train_normal",
        )
        train_pred = np.mean(train_stack["y_pred_stack"], axis=0)
        train_score = per_sample_nrmse(train_stack["y_true"], train_pred, scales)
        threshold = float(np.quantile(train_score[np.isfinite(train_score)], float(cfg["early_warning"]["prediction_threshold_quantile"])))
        for split in ["test_normal", "test_fault"]:
            stack = stack_seed_predictions(
                project_root,
                cfg,
                residual_model=item["residual_model"],
                raman_method=item["raman_method"],
                horizon=int(item["horizon_steps"]),
                seeds=seeds,
                split=split,
            )
            pred = np.mean(stack["y_pred_stack"], axis=0)
            score = per_sample_nrmse(stack["y_true"], pred, scales)
            rows.extend(
                first_alert_rows(
                    score=score,
                    batch_id=stack["batch_id"],
                    target_time_h=stack["target_time_h"],
                    threshold=threshold,
                    split=split,
                    score_name=f"prediction_error_{item['name']}",
                    horizon=int(item["horizon_steps"]),
                )
            )
    rows.extend(run_mechanism_early_warning(project_root, cfg))
    write_union_csv(dirs["tables"] / "Fig40_phase14_early_warning_data.csv", rows)
    figure_early_warning(dirs, rows)
    return rows


def first_alert_rows(
    *,
    score: np.ndarray,
    batch_id: np.ndarray,
    target_time_h: np.ndarray,
    threshold: float,
    split: str,
    score_name: str,
    horizon: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for bid in sorted(np.unique(batch_id)):
        idx = np.where(batch_id == bid)[0]
        order = idx[np.argsort(target_time_h[idx])]
        score_sorted = score[order]
        time_sorted = target_time_h[order]
        alert_mask = score_sorted > threshold
        alerted = bool(np.any(alert_mask))
        first_time = float(time_sorted[np.argmax(alert_mask)]) if alerted else math.nan
        out.append(
            {
                "figure_id": "Fig40_phase14_early_warning",
                "phase": PHASE,
                "experiment_name": EXPERIMENT,
                "analysis_group": "early_warning",
                "score_name": score_name,
                "split": split,
                "batch_id": int(bid),
                "fault_label": 1 if split == "test_fault" else 0,
                "horizon_steps": horizon,
                "threshold_source": "train_normal_timewise_quantile",
                "threshold": threshold,
                "alerted": int(alerted),
                "time_to_first_alert_h": first_time,
                "batch_max_score": float(np.nanmax(score_sorted)),
                "batch_q95_score": float(np.nanquantile(score_sorted, 0.95)),
            }
        )
    return out


def run_mechanism_early_warning(project_root: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    phase07 = load_script_module("phase07_mechanism_module", project_root / "scripts" / "12_run_phase07_mechanism_diagnostics.py")
    phase07_cfg = load_json(project_root / cfg["phase07_config"])
    processed_root = project_root / phase07_cfg["processed_root"]
    variable_roles = load_json(project_root / phase07_cfg["variable_roles"])
    split_info = phase07.split_map(project_root / phase07_cfg["split_csv"])
    model_path = project_root / phase07_cfg["output_root"] / "models" / "mechanism_proxy_parameters.json"
    scale_path = project_root / phase07_cfg["output_root"] / "models" / "mechanism_residual_scales.json"
    if model_path.exists() and scale_path.exists():
        models = load_json(model_path)
        scales = load_json(scale_path)
    else:
        models = phase07.fit_mechanism_models(processed_root, phase07_cfg, variable_roles, split_info)
        scales = phase07.fit_residual_scales(processed_root, phase07_cfg, variable_roles, split_info, models)
    score_channels = phase07_cfg.get("composite_residual_channels", phase07_cfg["residual_channels"])
    train_scores = []
    ts_by_batch: dict[int, dict[str, np.ndarray]] = {}
    for batch_id, info in sorted(split_info.items()):
        residual = phase07.residuals_for_batch(processed_root, batch_id, phase07_cfg, variable_roles, models)
        z = phase07.standardized_residuals(residual, scales, phase07_cfg["residual_channels"], score_channels)
        ts_by_batch[batch_id] = z
        if info["split"] == "train_normal":
            train_scores.append(z["mechanism_score"])
    threshold = float(np.nanquantile(np.concatenate(train_scores), float(cfg["early_warning"]["mechanism_threshold_quantile"])))
    rows: list[dict[str, Any]] = []
    for batch_id, info in sorted(split_info.items()):
        if info["split"] not in {"test_normal", "test_fault"}:
            continue
        z = ts_by_batch[batch_id]
        rows.extend(
            first_alert_rows(
                score=z["mechanism_score"],
                batch_id=np.full(z["mechanism_score"].shape, batch_id),
                target_time_h=z["time_h"],
                threshold=threshold,
                split=info["split"],
                score_name="mechanism_residual_score",
                horizon=0,
            )
        )
    return rows


def run_gate_interpretability(
    project_root: Path,
    cfg: dict[str, Any],
    metadata: dict[str, Any],
    dirs: dict[str, Path],
) -> list[dict[str, Any]]:
    if not cfg["gate_interpretability"]["enabled"]:
        return []
    split_info = metadata["split_info"]
    duration_by_batch = {batch_id: float(info["duration_h"]) for batch_id, info in split_info.items()}
    rows: list[dict[str, Any]] = []
    target_columns = metadata["target_columns"]
    seeds = [int(s) for s in cfg["gate_interpretability"]["seeds"]]
    for method in cfg["gate_interpretability"]["raman_methods"]:
        for horizon in [int(h) for h in cfg["gate_interpretability"]["horizon_steps"]]:
            for split in cfg["gate_interpretability"]["splits"]:
                stacks = []
                batch_ids = None
                times = None
                for seed in seeds:
                    payload = load_prediction_npz(project_root, cfg, "residual_target_gate_tcn", method, horizon, seed, split)
                    if "target_gate" not in payload:
                        continue
                    stacks.append(payload["target_gate"].astype(np.float64))
                    batch_ids = payload["batch_id"]
                    times = payload["target_time_h"]
                if not stacks or batch_ids is None or times is None:
                    continue
                gate = np.mean(np.stack(stacks, axis=0), axis=0)
                for target_idx, target in enumerate(target_columns):
                    rows.append(
                        {
                            "figure_id": "Fig42_phase14_gate_interpretability",
                            "phase": PHASE,
                            "experiment_name": EXPERIMENT,
                            "analysis_group": "target_gate_interpretability",
                            "split": split,
                            "raman_method": method,
                            "horizon_steps": horizon,
                            "target": target,
                            "phase_bin": "all",
                            "mean_gate": float(np.mean(gate[:, target_idx])),
                            "median_gate": float(np.median(gate[:, target_idx])),
                            "sample_count": int(gate.shape[0]),
                            "seed_count": len(stacks),
                        }
                    )
                frac = np.asarray([float(t) / max(duration_by_batch.get(int(b), float(t)), 1e-8) for b, t in zip(batch_ids, times)])
                for phase_bin in cfg["gate_interpretability"]["phase_bins"]:
                    mask = (frac >= float(phase_bin["min_fraction"])) & (frac < float(phase_bin["max_fraction"]))
                    if not np.any(mask):
                        continue
                    for target_idx, target in enumerate(target_columns):
                        rows.append(
                            {
                                "figure_id": "Fig42_phase14_gate_interpretability",
                                "phase": PHASE,
                                "experiment_name": EXPERIMENT,
                                "analysis_group": "target_gate_interpretability",
                                "split": split,
                                "raman_method": method,
                                "horizon_steps": horizon,
                                "target": target,
                                "phase_bin": phase_bin["name"],
                                "mean_gate": float(np.mean(gate[mask, target_idx])),
                                "median_gate": float(np.median(gate[mask, target_idx])),
                                "sample_count": int(mask.sum()),
                                "seed_count": len(stacks),
                            }
                        )
    write_union_csv(dirs["tables"] / "Fig42_phase14_gate_interpretability_data.csv", rows)
    figure_gate_interpretability(dirs, rows)
    return rows


def figure_raman_robustness(dirs: dict[str, Path], rows: list[dict[str, Any]]) -> None:
    figure_id = "Fig38_phase14_raman_robustness"
    sub = [
        row for row in rows
        if row["split"] == "test_normal"
        and row["residual_model"] == "residual_target_gate_tcn"
        and row["raman_method"] == "airpls_snv"
        and row["metric_name"] == "median_nrmse_train_std"
    ]
    if not sub:
        return
    scenario_order = []
    for row in sub:
        if row["scenario"] not in scenario_order:
            scenario_order.append(row["scenario"])
    horizons = sorted({int(row["horizon_steps"]) for row in sub})
    means = {}
    for horizon in horizons:
        vals = []
        for scenario in scenario_order:
            v = [float(row["metric_value"]) for row in sub if int(row["horizon_steps"]) == horizon and row["scenario"] == scenario]
            vals.append(float(np.mean(v)) if v else math.nan)
        means[horizon] = vals
    x = np.arange(len(scenario_order))
    plt.figure(figsize=(12.8, 4.8))
    for horizon in horizons:
        plt.plot(x, means[horizon], marker="o", label=f"H{horizon}")
    plt.xticks(x, scenario_order, rotation=35, ha="right")
    plt.ylabel("Median nRMSE")
    plt.xlabel("Raman Perturbation Scenario")
    plt.title("Raman Robustness of TargetGate-airPLS")
    plt.legend(title="Horizon")
    polish_axis(plt.gca(), grid_axis="y")
    plt.tight_layout()
    fig_path = dirs["figures"] / f"{figure_id}.png"
    plt.savefig(fig_path, dpi=400)
    plt.close()
    save_explanation(
        dirs["explanations"] / f"{figure_id}_explanation.md",
        title="Phase14 Raman Robustness ",
        figure_id=figure_id,
        purpose=" Raman 、，TargetGate-airPLS residual 。",
        data_source="Phase10  checkpoint、Phase08 train-normal PCA Raman scores、Phase04 Raman-aligned test windows。",
        design=" Raman ， test-normal  seed  median nRMSE；。",
        results=" horizon  Raman ； seed  CSV。",
        interpretation=" nRMSE ， Raman  horizon ；， gate  Raman。",
        discussion="， Raman 、、。",
        limitations=" Raman PCA ，。",
        caption="Raman robustness under missing, noisy, delayed, and zeroed spectral inputs.",
        command="python3 scripts/24_run_phase14_enhance_v2.py --config configs/model/phase14_enhance_v2.json",
    )


def figure_uncertainty(dirs: dict[str, Path], rows: list[dict[str, Any]]) -> None:
    figure_id = "Fig39_phase14_uncertainty_calibration"
    configs = list(dict.fromkeys(row["config_name"] for row in rows))
    normal = {row["config_name"]: float(row["mean_uncertainty_norm"]) for row in rows if row["split"] == "test_normal"}
    fault = {row["config_name"]: float(row["mean_uncertainty_norm"]) for row in rows if row["split"] == "test_fault"}
    x = np.arange(len(configs))
    width = 0.36
    plt.figure(figsize=(10.8, 4.7))
    plt.bar(x - width / 2, [normal.get(c, math.nan) for c in configs], width=width, label="Test Normal")
    plt.bar(x + width / 2, [fault.get(c, math.nan) for c in configs], width=width, label="Test Fault")
    plt.xticks(x, configs, rotation=25, ha="right")
    plt.ylabel("Mean Ensemble Std / Train Std")
    plt.xlabel("Ensemble Configuration")
    plt.title("Deep-Ensemble Uncertainty on Normal and Fault Batches")
    plt.legend()
    polish_axis(plt.gca(), grid_axis="y")
    plt.tight_layout()
    plt.savefig(dirs["figures"] / f"{figure_id}.png", dpi=400)
    plt.close()
    save_explanation(
        dirs["explanations"] / f"{figure_id}_explanation.md",
        title="Phase14 Uncertainty Calibration ",
        figure_id=figure_id,
        purpose=" seed  deep ensemble 。",
        data_source="Phase10  seed 。",
        design=" test-normal  test-fault  ensemble ， train-normal target std 。",
        results="CSV  PICP、MPIW、nRMSE、R2、uncertainty-error Spearman 。",
        interpretation=" fault batch ，； PICP ， seed ensemble 。",
        discussion="“”。",
        limitations=" seed ensemble ， seed  conformal prediction。",
        caption="Deep-ensemble uncertainty comparison between normal and fault batches.",
        command="python3 scripts/24_run_phase14_enhance_v2.py --config configs/model/phase14_enhance_v2.json",
    )


def figure_bootstrap(dirs: dict[str, Path], rows: list[dict[str, Any]]) -> None:
    figure_id = "Fig41_phase14_bootstrap_effects"
    labels = [row["comparison"] for row in rows]
    values = [float(row["observed_relative_improvement_percent"]) for row in rows]
    x = np.arange(len(labels))
    plt.figure(figsize=(11.4, 4.7))
    colors = ["#009E73" if v >= 0 else "#D55E00" for v in values]
    plt.bar(x, values, color=colors)
    plt.axhline(0, color="#222222", linewidth=0.8)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Relative nRMSE Reduction (%)")
    plt.xlabel("Paired Comparison")
    plt.title("Paired Bootstrap Effect Summary")
    polish_axis(plt.gca(), grid_axis="y")
    plt.tight_layout()
    plt.savefig(dirs["figures"] / f"{figure_id}.png", dpi=400)
    plt.close()
    save_explanation(
        dirs["explanations"] / f"{figure_id}_explanation.md",
        title="Phase14 Bootstrap Effect ",
        figure_id=figure_id,
        purpose=" paired bootstrap ， nRMSE 。",
        data_source="Phase10  seed  Ridge baseline prediction。",
        design=" candidate  reference  nRMSE ； p  CSV，。",
        results="CSV  observed effect、95% bootstrap CI、one-sided p-value。",
        interpretation=" candidate  reference ； CSV  CI  0，。",
        discussion="，。",
        limitations="bootstrap  Raman-aligned windows， bootstrap。",
        caption="Paired bootstrap effect sizes for key Phase10 comparisons.",
        command="python3 scripts/24_run_phase14_enhance_v2.py --config configs/model/phase14_enhance_v2.json",
    )


def figure_early_warning(dirs: dict[str, Path], rows: list[dict[str, Any]]) -> None:
    figure_id = "Fig40_phase14_early_warning"
    score_names = list(dict.fromkeys(row["score_name"] for row in rows))
    vals = []
    for score_name in score_names:
        fault_rows = [row for row in rows if row["score_name"] == score_name and row["split"] == "test_fault"]
        vals.append(float(np.mean([int(row["alerted"]) for row in fault_rows])) if fault_rows else math.nan)
    x = np.arange(len(score_names))
    plt.figure(figsize=(9.6, 4.4))
    plt.bar(x, vals, color="#0072B2")
    plt.ylim(0, 1.05)
    plt.xticks(x, score_names, rotation=25, ha="right")
    plt.ylabel("Fault Batch Alert Rate")
    plt.xlabel("Score")
    plt.title("Early-Warning Alert Rate on Fault Batches")
    polish_axis(plt.gca(), grid_axis="y")
    plt.tight_layout()
    plt.savefig(dirs["figures"] / f"{figure_id}.png", dpi=400)
    plt.close()
    save_explanation(
        dirs["explanations"] / f"{figure_id}_explanation.md",
        title="Phase14 Early Warning ",
        figure_id=figure_id,
        purpose=" fault batch 。",
        data_source="Phase10 、Phase07 、Phase04 。",
        design=" test-fault  train-normal q99 ；CSV  batch 。",
        results=" fault alert rate、test-normal false alert、time-to-first-alert。",
        interpretation=" test-normal ，。",
        discussion="。",
        limitations="， batch 。",
        caption="Early-warning alert rates using prediction-error and mechanism-residual scores.",
        command="python3 scripts/24_run_phase14_enhance_v2.py --config configs/model/phase14_enhance_v2.json",
    )


def figure_gate_interpretability(dirs: dict[str, Path], rows: list[dict[str, Any]]) -> None:
    figure_id = "Fig42_phase14_gate_interpretability"
    sub = [
        row for row in rows
        if row["split"] == "test_normal"
        and row["raman_method"] == "airpls_snv"
        and int(row["horizon_steps"]) == 5
        and row["phase_bin"] == "all"
    ]
    if not sub:
        return
    targets = [row["target"] for row in sub]
    values = [float(row["mean_gate"]) for row in sub]
    short_targets = [t.split("(")[0].strip()[:28] for t in targets]
    y = np.arange(len(short_targets))
    plt.figure(figsize=(7.4, 6.2))
    plt.barh(y, values, color="#009E73")
    plt.yticks(y, short_targets)
    plt.xlabel("Mean Target Gate")
    plt.ylabel("Target")
    plt.title("Target-wise Raman Gate Dependence")
    plt.xlim(0, 1)
    polish_axis(plt.gca(), grid_axis="x")
    plt.tight_layout()
    plt.savefig(dirs["figures"] / f"{figure_id}.png", dpi=400)
    plt.close()
    save_explanation(
        dirs["explanations"] / f"{figure_id}_explanation.md",
        title="Phase14 Gate Interpretability ",
        figure_id=figure_id,
        purpose=" target-adaptive gate  Raman 。",
        data_source="Phase10 TargetGate  target_gate 。",
        design=" test-normal、H5、airPLS-SNV  gate 。",
        results="CSV  early/middle/late fermentation phase  gate ， Raman 。",
        interpretation="gate ， Raman residual 。",
        discussion="， gate 。",
        limitations="gate ，。",
        caption="Target-wise Raman gate dependence for the H5 TargetGate-airPLS residual model.",
        command="python3 scripts/24_run_phase14_enhance_v2.py --config configs/model/phase14_enhance_v2.json",
    )


def write_manifest(dirs: dict[str, Path], cfg: dict[str, Any], row_counts: dict[str, int]) -> None:
    manifest = {
        "phase": PHASE,
        "experiment_name": EXPERIMENT,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": "configs/model/phase14_enhance_v2.json",
        "row_counts": row_counts,
        "outputs": {
            "metrics_dir": str(dirs["metrics"]),
            "paper_tables_dir": str(dirs["tables"]),
            "paper_figures_dir": str(dirs["figures"]),
            "paper_explanations_dir": str(dirs["explanations"]),
        },
        "truthfulness_policy": cfg["notes"],
    }
    (dirs["output"] / "phase14_run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (dirs["manifests"] / "phase_14_enhance_v2_assets.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase14 enhance-v2 evidence experiments")
    parser.add_argument("--config", default="configs/model/phase14_enhance_v2.json")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    project_root = Path.cwd()
    cfg = load_json(project_root / args.config)
    device_arg = args.device or cfg.get("device", "auto")
    dirs = ensure_dirs(project_root / cfg["output_root"], project_root / cfg["paper_asset_root"])
    log("Starting Phase14 enhance_v2")
    ctx = load_phase10_context(project_root, cfg)
    metadata = ctx["metadata"]

    robustness_rows = run_raman_robustness(project_root, cfg, ctx, dirs, device_arg)
    uncertainty_rows = run_uncertainty(project_root, cfg, metadata, dirs)
    bootstrap_rows = run_bootstrap(project_root, cfg, metadata, dirs)
    early_rows = run_early_warning(project_root, cfg, metadata, dirs)
    gate_rows = run_gate_interpretability(project_root, cfg, metadata, dirs)

    write_manifest(
        dirs,
        cfg,
        {
            "raman_robustness_summary_rows": len(robustness_rows),
            "uncertainty_rows": len(uncertainty_rows),
            "bootstrap_rows": len(bootstrap_rows),
            "early_warning_rows": len(early_rows),
            "gate_interpretability_rows": len(gate_rows),
        },
    )
    log("Phase14 enhance_v2 complete")


if __name__ == "__main__":
    main()

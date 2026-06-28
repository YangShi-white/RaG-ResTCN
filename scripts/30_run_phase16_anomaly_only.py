#!/usr/bin/env python3
"""Run only the unfinished Phase16 anomaly baselines.

This wrapper reuses the real anomaly code from `26_run_phase16_strong_baselines.py`
and deliberately skips all forecasting baselines.
"""

from __future__ import annotations

import argparse
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


def load_phase16_module() -> Any:
    path = PROJECT_ROOT / "scripts" / "26_run_phase16_strong_baselines.py"
    spec = importlib.util.spec_from_file_location("phase16_strong_baselines", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import Phase16 module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


phase16 = load_phase16_module()


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def save_anomaly_assets(
    *,
    cfg: dict[str, Any],
    dirs: dict[str, Path],
    anomaly_metric_rows: list[dict[str, Any]],
    anomaly_selection_rows: list[dict[str, Any]],
    command: str,
) -> dict[str, Any]:
    anomaly_df = pd.DataFrame(anomaly_metric_rows)
    selection_df = pd.DataFrame(anomaly_selection_rows)

    fig54_csv = dirs["tables"] / "Fig54_phase16_anomaly_only_detection_data.csv"
    fig54_selection_csv = dirs["tables"] / "Fig54_phase16_anomaly_only_selection_data.csv"
    anomaly_df.to_csv(fig54_csv, index=False)
    selection_df.to_csv(fig54_selection_csv, index=False)

    fig54_png = dirs["figures"] / "Fig54_phase16_anomaly_only_detection.png"
    if not anomaly_df.empty:
        plot_df = anomaly_df[
            anomaly_df["metric_name"].isin(["auroc", "auprc", "false_alarm_rate", "fault_detection_rate"])
        ].copy()
        plot_df["metric_value"] = pd.to_numeric(plot_df["metric_value"], errors="coerce")
        methods = list(dict.fromkeys(plot_df["method"].tolist()))
        metrics = ["auroc", "auprc", "false_alarm_rate", "fault_detection_rate"]
        x = np.arange(len(methods))
        width = 0.18
        fig, ax = plt.subplots(figsize=(7.4, 3.0))
        for i, metric in enumerate(metrics):
            vals = []
            for method in methods:
                row = plot_df[(plot_df["method"] == method) & (plot_df["metric_name"] == metric)]
                vals.append(float(row["metric_value"].iloc[0]) if not row.empty else math.nan)
            ax.bar(x + (i - 1.5) * width, vals, width=width, label=metric.upper())
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=35, ha="right")
        ax.set_ylabel("Detection Metric")
        ax.set_title("Unsupervised Anomaly Baseline Detection")
        ax.set_ylim(0, 1.05)
        ax.legend(ncol=2)
        phase16.polish_axis(ax)
        fig.tight_layout()
        fig.savefig(fig54_png)
        plt.close(fig)

    explanation = dirs["explanations"] / "Fig54_phase16_anomaly_only_detection_explanation.md"
    phase16.save_explanation(
        explanation,
        title="Phase16 anomaly-only ",
        figure_id="Fig54_phase16_anomaly_only_detection",
        purpose=" GPU ，，。",
        data_source="Phase04  IndPenSim process windows； train_normal 。",
        design=" AUROC、AUPRC、false alarm rate  fault detection rate；CSV 、。",
        results="， PCA-Q、PCA-T2、DPCA-Q、OCSVM、IsolationForest  Autoencoder 。",
        interpretation="AUROC/AUPRC ，false alarm rate ，fault detection rate 。detection delay  batch-level F1  CSV 。",
        discussion=" Phase16  anomaly baselines ， returned forecasting-baseline summary 。",
        limitations="， fault detection，。",
        caption="Unsupervised anomaly baselines trained only on normal batches.",
        command=command,
    )

    manifest = {
        "phase": "Phase 16 Anomaly Only",
        "experiment_name": "Phase16 anomaly baselines only",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_root": str(dirs["output"]),
        "metric_rows": len(anomaly_metric_rows),
        "selection_rows": len(anomaly_selection_rows),
        "tables": [str(fig54_csv), str(fig54_selection_csv)],
        "figures": [str(fig54_png)],
        "explanations": [str(explanation)],
        "forecasting_baselines_rerun": False,
        "anomaly_methods_expected": ["PCA-Q", "PCA-T2", "DPCA-Q", "OCSVM", "IsolationForest", "Autoencoder"],
        "truthfulness_policy": [
            "No synthetic anomaly scores are generated.",
            "All thresholds are derived from train_normal scores.",
            "test_normal and test_fault are used only for final evaluation."
        ],
    }
    manifest_path = dirs["manifests"] / "phase16_anomaly_only_assets.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase16 anomaly baselines only")
    parser.add_argument("--config", default="configs/model/phase16_anomaly_only.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-jobs", type=int, default=max(1, os.cpu_count() or 1))
    args = parser.parse_args()

    project_root = Path.cwd()
    cfg = phase16.load_json(project_root / args.config)
    cfg["anomaly_baselines"]["enabled"] = True
    cfg["patchtst"]["enabled"] = False
    cfg["classical_baselines"] = {}
    dirs = phase16.ensure_dirs(project_root / cfg["output_root"], project_root / cfg["paper_asset_root"])
    metadata = phase16.build_forecasting_metadata(cfg, project_root)
    sk = phase16.import_sklearn()

    log("Starting Phase16 anomaly-only run")
    log(f"device={args.device} n_jobs={args.n_jobs}")
    anomaly_metric_rows, anomaly_selection_rows = phase16.run_anomaly_baselines(
        cfg=cfg,
        dirs=dirs,
        metadata=metadata,
        sk=sk,
        device_arg=args.device,
        n_jobs=int(args.n_jobs),
    )

    metrics_csv = dirs["metrics"] / "phase16_anomaly_only_metrics.csv"
    selection_csv = dirs["metrics"] / "phase16_anomaly_only_selection.csv"
    phase16.write_union_csv(metrics_csv, anomaly_metric_rows)
    phase16.write_union_csv(selection_csv, anomaly_selection_rows)

    manifest = save_anomaly_assets(
        cfg=cfg,
        dirs=dirs,
        anomaly_metric_rows=anomaly_metric_rows,
        anomaly_selection_rows=anomaly_selection_rows,
        command=" ".join(sys.argv),
    )
    aggregate = {
        "phase": "Phase 16 Anomaly Only",
        "anomaly_metric_rows": len(anomaly_metric_rows),
        "anomaly_selection_rows": len(anomaly_selection_rows),
        "metrics_csv": str(metrics_csv),
        "selection_csv": str(selection_csv),
        "asset_manifest": str(dirs["manifests"] / "phase16_anomaly_only_assets.json"),
        "manifest": manifest,
    }
    aggregate_path = dirs["output"] / "phase16_anomaly_only_run_manifest.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Phase16 anomaly-only run finished. Manifest: {aggregate_path}")


if __name__ == "__main__":
    main()

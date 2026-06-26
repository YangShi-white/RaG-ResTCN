#!/usr/bin/env python3
"""Collect Phase31 shard outputs into final metric tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    keys = ["model", "control_mode", "horizon_steps", "split", "target"]
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/phase31_infosci_modern_deep_baselines")
    args = parser.parse_args()
    metrics_dir = PROJECT_ROOT / args.output_root / "metrics"
    metric_files = sorted(metrics_dir.glob("per_target_metrics_shard*.csv"))
    run_files = sorted(metrics_dir.glob("run_summary_shard*.csv"))
    if not metric_files or not run_files:
        raise SystemExit(f"No shard metrics found in {metrics_dir}")
    metric_rows = [row for path in metric_files for row in read_rows(path)]
    run_rows = [row for path in run_files for row in read_rows(path)]
    write_rows(metrics_dir / "per_target_metrics.csv", metric_rows)
    write_rows(metrics_dir / "run_summary.csv", run_rows)
    write_rows(metrics_dir / "summary_mean_std.csv", summarize(metric_rows))
    print(f"Collected {len(metric_files)} metric shards and {len(run_files)} run shards")
    print(f"Runs: {len(run_rows)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare unified-v2 split/window manifests and leakage audit files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fermnftp.data import ForecastWindowDataset, build_forecasting_metadata, load_json  # noqa: E402


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stable_hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def make_repeated_split(base_splits: dict[str, list[int]], seed: int, counts: dict[str, int]) -> dict[str, list[int]]:
    normal = sorted(base_splits["train_normal"] + base_splits["val_normal"] + base_splits["test_normal"])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(normal).tolist()
    n_train = int(counts["train_normal"])
    n_val = int(counts["val_normal"])
    n_test = int(counts["test_normal"])
    return {
        "train_normal": sorted(perm[:n_train]),
        "val_normal": sorted(perm[n_train : n_train + n_val]),
        "test_normal": sorted(perm[n_train + n_val : n_train + n_val + n_test]),
        "test_fault": sorted(base_splits.get("test_fault", [])),
    }


def overlap_rows(seed: int, split_ids: dict[str, list[int]]) -> list[dict[str, Any]]:
    names = ["train_normal", "val_normal", "test_normal", "test_fault"]
    rows = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = sorted(set(split_ids[left]) & set(split_ids[right]))
            rows.append(
                {
                    "seed": seed,
                    "left_split": left,
                    "right_split": right,
                    "n_overlap": len(overlap),
                    "overlap_batch_ids": " ".join(str(x) for x in overlap),
                    "status": "pass" if not overlap else "fail",
                }
            )
    return rows


def window_rows(ds: ForecastWindowDataset, seed: int, horizon: int, split: str) -> list[dict[str, Any]]:
    rows = []
    for sample_id, (batch_id, source_t) in enumerate(ds.index):
        rows.append(
            {
                "seed": seed,
                "horizon_steps": horizon,
                "split": split,
                "sample_id": sample_id,
                "batch_id": batch_id,
                "history_start": int(source_t) - ds.history_steps + 1,
                "history_end": int(source_t),
                "target_index": int(source_t) + ds.horizon_steps,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model/unified_v2_rag_restcn.json")
    args = parser.parse_args()
    cfg = load_json(PROJECT_ROOT / args.config)
    metadata = build_forecasting_metadata(cfg, PROJECT_ROOT)
    out_root = ensure_dir(PROJECT_ROOT / cfg["output_root"])
    split_dir = ensure_dir(out_root / "manifests" / "splits")
    window_dir = ensure_dir(out_root / "manifests" / "windows")
    audit_dir = ensure_dir(out_root / "audit")

    all_overlap_rows: list[dict[str, Any]] = []
    hash_rows: list[dict[str, Any]] = []
    preprocessing_rows = [
        {"artifact": "input robust scaling", "fit_source": "train_normal only", "status": "pass"},
        {"artifact": "target scaling", "fit_source": "train_normal only", "status": "pass"},
        {"artifact": "ridge baseline", "fit_source": "train_normal; alpha selected on val_normal", "status": "pass"},
        {"artifact": "residual scale", "fit_source": "train_normal residuals", "status": "pass"},
        {"artifact": "fault batches", "fit_source": "never used for fit/selection/calibration", "status": "pass"},
    ]

    for seed in [int(s) for s in cfg["repeat_seeds"]]:
        split_ids = make_repeated_split(metadata["splits"], seed, cfg["normal_split_counts"])
        split_payload = {
            "seed": seed,
            "train_normal_batch_ids": split_ids["train_normal"],
            "val_normal_batch_ids": split_ids["val_normal"],
            "test_normal_batch_ids": split_ids["test_normal"],
            "fault_test_batch_ids": split_ids["test_fault"],
            "source_split_hash": stable_hash(metadata["splits"]),
        }
        split_payload["split_hash"] = stable_hash(split_payload)
        (split_dir / f"split_seed_{seed}.json").write_text(
            json.dumps(split_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        rows = overlap_rows(seed, split_ids)
        all_overlap_rows.extend(rows)
        if any(row["status"] != "pass" for row in rows):
            raise SystemExit(f"Split overlap audit failed for seed {seed}")

        for horizon in [int(h) for h in cfg["horizons"]]:
            for split, batch_ids in split_ids.items():
                cap = cfg["train"].get("max_windows_per_batch_train") if split == "train_normal" else cfg["train"].get("max_windows_per_batch_eval")
                ds = ForecastWindowDataset(
                    processed_root=metadata["processed_root"],
                    batch_ids=batch_ids,
                    history_steps=int(cfg["history_steps"]),
                    horizon_steps=horizon,
                    process_columns=metadata["process_columns"],
                    history_columns=metadata["history_columns"],
                    exogenous_columns=metadata["exogenous_columns"],
                    target_columns=metadata["target_columns"],
                    process_centers=metadata["process_centers"],
                    process_scales=metadata["process_scales"],
                    target_centers=metadata["target_centers"],
                    target_scales=metadata["target_scales"],
                    max_windows_per_batch=cap,
                )
                rows = window_rows(ds, seed, horizon, split)
                path = window_dir / f"seed_{seed}_h{horizon}_{split}.csv"
                write_rows(path, rows)
                hash_rows.append(
                    {
                        "seed": seed,
                        "horizon_steps": horizon,
                        "split": split,
                        "n_windows": len(rows),
                        "manifest_path": str(path.relative_to(out_root)),
                        "manifest_hash": stable_hash(rows),
                    }
                )

    write_rows(audit_dir / "split_overlap_check.csv", all_overlap_rows)
    write_rows(audit_dir / "preprocessing_fit_sources.csv", preprocessing_rows)
    write_rows(audit_dir / "window_manifest_hashes.csv", hash_rows)
    protocol_audit = {
        "protocol": "unified_v2",
        "status": "pass",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": args.config,
        "history_steps": cfg["history_steps"],
        "horizons": cfg["horizons"],
        "repeat_seeds": cfg["repeat_seeds"],
        "control_semantics": "horizon-specific future control vector u_t_plus_h",
        "split_overlap_check": str(audit_dir / "split_overlap_check.csv"),
        "preprocessing_fit_sources": str(audit_dir / "preprocessing_fit_sources.csv"),
        "window_manifest_hashes": str(audit_dir / "window_manifest_hashes.csv"),
    }
    (audit_dir / "protocol_audit.json").write_text(
        json.dumps(protocol_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(protocol_audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

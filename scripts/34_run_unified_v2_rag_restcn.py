#!/usr/bin/env python3
"""Run unified-v2 controlled residual and Raman fusion experiments.

This script reuses the validated Phase 10 residual learner implementation, but
materializes the Phase 31/unified-v2 repeated split protocol before each run.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fermnftp.data import load_json  # noqa: E402


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_phase10_module() -> Any:
    path = PROJECT_ROOT / "scripts" / "17_train_phase10_residual_multimodal.py"
    spec = importlib.util.spec_from_file_location("phase10_residual_multimodal", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["phase10_residual_multimodal"] = module
    spec.loader.exec_module(module)
    return module


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


def assert_no_overlap(split_ids: dict[str, list[int]], seed: int) -> None:
    names = ["train_normal", "val_normal", "test_normal", "test_fault"]
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = sorted(set(split_ids[left]) & set(split_ids[right]))
            if overlap:
                raise RuntimeError(f"Seed {seed} split overlap {left}/{right}: {overlap}")


def remap_split_info(base_split_info: dict[int, dict[str, Any]], split_ids: dict[str, list[int]]) -> dict[int, dict[str, Any]]:
    out = copy.deepcopy(base_split_info)
    for split, ids in split_ids.items():
        for batch_id in ids:
            out[int(batch_id)]["split"] = split
    return out


def build_raman_scores(
    *,
    cfg: dict[str, Any],
    phase10: Any,
    sample_metadata: list[dict[str, Any]],
    seed: int,
) -> dict[str, np.ndarray]:
    scores_npz = np.load(PROJECT_ROOT / cfg["phase08_pca_scores"])
    scores: dict[str, np.ndarray] = {}
    for method in cfg.get("base_raman_methods", cfg["raman_methods"]):
        key = f"{method}_scores"
        if key not in scores_npz:
            raise ValueError(f"Missing Raman score key {key} in {PROJECT_ROOT / cfg['phase08_pca_scores']}")
        scores[method] = scores_npz[key].astype(np.float32)

    if "sg1_snv_zero" in cfg["raman_methods"]:
        base = scores["sg1_snv"]
        scores["sg1_snv_zero"] = np.zeros_like(base, dtype=np.float32)
    if "sg1_snv_shuffled" in cfg["raman_methods"]:
        base = scores["sg1_snv"]
        shuffled = base.copy()
        rng = np.random.default_rng(seed + 91031)
        by_batch: dict[int, list[int]] = {}
        for item in sample_metadata:
            by_batch.setdefault(int(item["batch_id"]), []).append(int(item["sample_index"]))
        for indices in by_batch.values():
            if len(indices) > 1:
                perm = rng.permutation(indices)
                shuffled[np.asarray(indices, dtype=int)] = base[np.asarray(perm, dtype=int)]
        scores["sg1_snv_shuffled"] = shuffled.astype(np.float32)
    return {method: scores[method] for method in cfg["raman_methods"] if method in scores}


def get_shard() -> tuple[int, int]:
    count = int(os.environ.get("UNIFIED_V2_SHARD_COUNT", os.environ.get("PHASE31_SHARD_COUNT", "1")))
    index = int(os.environ.get("UNIFIED_V2_SHARD_INDEX", "0"))
    if count < 1 or not 0 <= index < count:
        raise ValueError("Invalid unified-v2 shard index/count")
    return index, count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model/unified_v2_rag_restcn.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--prepare-baseline-only", action="store_true")
    args = parser.parse_args()

    cfg = load_json(PROJECT_ROOT / args.config)
    phase10 = load_phase10_module()
    base_metadata = phase10.build_metadata(cfg, PROJECT_ROOT)
    shard_index, shard_count = get_shard()

    tasks: list[dict[str, Any]] = []
    for repeat_id, split_seed in enumerate([int(s) for s in cfg["repeat_seeds"]]):
        for horizon in [int(h) for h in cfg["horizons"]]:
            for task in cfg["unified_tasks"]:
                tasks.append(
                    {
                        "repeat_id": repeat_id,
                        "split_seed": split_seed,
                        "horizon": horizon,
                        "residual_model": task["residual_model"],
                        "raman_method": task["raman_method"],
                    }
                )
    selected = [task for task_id, task in enumerate(tasks) if task_id % shard_count == shard_index]
    log(f"Unified-v2 shard {shard_index}/{shard_count}: selected {len(selected)} of {len(tasks)} residual tasks")

    manifests = []
    for split_seed in [int(s) for s in cfg["repeat_seeds"]]:
        split_ids = make_repeated_split(base_metadata["splits"], split_seed, cfg["normal_split_counts"])
        assert_no_overlap(split_ids, split_seed)
        metadata = copy.deepcopy(base_metadata)
        metadata["splits"] = split_ids
        metadata["split_info"] = remap_split_info(base_metadata["split_info"], split_ids)
        sample_metadata = phase10.build_raman_sample_metadata(metadata["processed_root"], metadata["split_info"])
        raman_scores = build_raman_scores(cfg=cfg, phase10=phase10, sample_metadata=sample_metadata, seed=split_seed)
        repeat_cfg = copy.deepcopy(cfg)
        repeat_cfg["seeds"] = [split_seed]
        repeat_cfg["horizon_steps"] = cfg["horizons"]
        repeat_cfg["output_root"] = str(Path(cfg["output_root"]) / "rag_restcn" / f"split_seed_{split_seed}" / f"shard_{shard_index:02d}")
        out_root = PROJECT_ROOT / repeat_cfg["output_root"]
        out_root.mkdir(parents=True, exist_ok=True)
        log(f"Building design arrays for split_seed={split_seed}")
        design = phase10.build_design(
            cfg=repeat_cfg,
            metadata=metadata,
            sample_metadata=sample_metadata,
            raman_scores_by_method=raman_scores,
        )
        baseline_by_horizon = phase10.prepare_ridge_baselines(
            cfg=repeat_cfg,
            project_root=PROJECT_ROOT,
            metadata=metadata,
            design=design,
        )
        if args.prepare_baseline_only:
            continue
        for task in selected:
            if int(task["split_seed"]) != split_seed:
                continue
            manifest = phase10.train_one_run(
                cfg=repeat_cfg,
                project_root=PROJECT_ROOT,
                metadata=metadata,
                design=design,
                baseline_by_horizon=baseline_by_horizon,
                residual_model=str(task["residual_model"]),
                raman_method=str(task["raman_method"]),
                horizon_steps=int(task["horizon"]),
                seed=int(split_seed),
                device_arg=args.device,
                max_samples_per_split=args.max_samples_per_split,
            )
            manifest["unified_v2_split_seed"] = split_seed
            manifest["unified_v2_repeat_id"] = task["repeat_id"]
            manifests.append(manifest)

    aggregate = {
        "phase": "unified_v2_rag_restcn",
        "config": args.config,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "run_count": len(manifests),
        "runs": manifests,
    }
    suffix = f"_shard{shard_index:02d}" if shard_count > 1 else ""
    aggregate_path = PROJECT_ROOT / cfg["output_root"] / f"rag_restcn_aggregate{suffix}.json"
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Unified-v2 residual runs complete: {aggregate_path}")


if __name__ == "__main__":
    main()

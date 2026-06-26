#!/usr/bin/env python3
"""Phase 31: modern deep baselines and future-control ablations.

This runner trains real forecasting models on batch-level splits. It does not
synthesize predictions, metrics, or convergence traces. All reported metrics are
computed from saved model predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fermnftp.data import build_forecasting_metadata, ForecastWindowDataset, load_json  # noqa: E402


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed: int, torch: Any | None = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def import_torch() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is required for Phase31 deep baselines.") from exc
    return torch, nn, DataLoader, torch.nn.functional


def finite_or_raise(array: np.ndarray, context: str) -> None:
    if not np.isfinite(array).all():
        bad = int((~np.isfinite(array)).sum())
        raise FloatingPointError(f"Non-finite values in {context}: {bad}")


def make_repeated_split(base_splits: dict[str, list[int]], seed: int, counts: dict[str, int]) -> dict[str, list[int]]:
    normal = sorted(base_splits["train_normal"] + base_splits["val_normal"] + base_splits["test_normal"])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(normal).tolist()
    n_train = int(counts["train_normal"])
    n_val = int(counts["val_normal"])
    n_test = int(counts["test_normal"])
    return {
        "train_normal": sorted(perm[:n_train]),
        "val_normal": sorted(perm[n_train:n_train + n_val]),
        "test_normal": sorted(perm[n_train + n_val:n_train + n_val + n_test]),
        "test_fault": sorted(base_splits.get("test_fault", [])),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_scales: np.ndarray) -> dict[str, np.ndarray | float]:
    err = y_pred - y_true
    mse = np.mean(err ** 2, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(err), axis=0)
    denom = np.maximum(np.asarray(target_scales, dtype=np.float64), 1e-8)
    nrmse = rmse / denom
    ss_res = np.sum(err ** 2, axis=0)
    centered = y_true - np.mean(y_true, axis=0, keepdims=True)
    ss_tot = np.sum(centered ** 2, axis=0)
    r2 = np.where(ss_tot > 1e-12, 1.0 - ss_res / ss_tot, np.nan)
    return {
        "nrmse": nrmse,
        "mae": mae,
        "r2": r2,
        "mean_nrmse": float(np.nanmean(nrmse)),
        "mean_mae": float(np.nanmean(mae)),
        "mean_r2": float(np.nanmean(r2)),
    }


class ControlWrapper:
    def __init__(self, base: Any, control_mode: str, n_exog: int, seed: int) -> None:
        self.base = base
        self.control_mode = control_mode
        self.n_exog = n_exog
        self.seed = seed
        self.perm = np.random.default_rng(seed).permutation(len(base)) if control_mode == "shuffled_future_controls" else None

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = dict(self.base[idx])
        if self.control_mode == "future_controls":
            return item
        if self.control_mode == "no_future_controls":
            item["future_exogenous"] = np.zeros_like(item["future_exogenous"], dtype=np.float32)
            return item
        if self.control_mode == "last_control_carry":
            item["future_exogenous"] = item["history"][-1, : self.n_exog].astype(np.float32)
            return item
        if self.control_mode == "shuffled_future_controls":
            other = self.base[int(self.perm[idx])]
            item["future_exogenous"] = np.asarray(other["future_exogenous"], dtype=np.float32)
            return item
        raise ValueError(f"Unknown control mode: {self.control_mode}")


@dataclass
class BatchPack:
    history: Any
    future: Any
    y_scaled: Any
    y_raw: Any
    batch_id: Any


def collate_numpy(batch: list[dict[str, Any]], torch: Any, device: Any) -> BatchPack:
    history = torch.as_tensor(np.nan_to_num(np.stack([b["history"] for b in batch]), nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32, device=device)
    future = torch.as_tensor(np.nan_to_num(np.stack([b["future_exogenous"] for b in batch]), nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32, device=device)
    y_scaled = torch.as_tensor(np.nan_to_num(np.stack([b["target_scaled"] for b in batch]), nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32, device=device)
    y_raw = torch.as_tensor(np.nan_to_num(np.stack([b["target_raw"] for b in batch]), nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32, device=device)
    batch_id = torch.as_tensor(np.asarray([b["batch_id"] for b in batch]), dtype=torch.long, device=device)
    return BatchPack(history, future, y_scaled, y_raw, batch_id)


def build_model(model_name: str, params: dict[str, Any], history_steps: int, n_hist: int, n_future: int, n_targets: int, nn: Any, torch: Any) -> Any:
    if model_name == "direct_tcn":
        return DirectTCN(history_steps, n_hist, n_future, n_targets, nn, **params)
    if model_name == "dlinear":
        return DLinearBaseline(history_steps, n_hist, n_future, n_targets, nn, **params)
    if model_name == "itransformer_lite":
        return ITransformerLite(history_steps, n_hist, n_future, n_targets, nn, **params)
    if model_name == "tsmixer":
        return TSMixer(history_steps, n_hist, n_future, n_targets, nn, **params)
    raise ValueError(f"Unknown model: {model_name}")


class DirectTCN:
    def __new__(cls, history_steps: int, n_hist: int, n_future: int, n_targets: int, nn: Any, hidden: int = 96, levels: int = 4, dropout: float = 0.1):
        torch_mod = __import__("torch")
        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                layers = []
                in_ch = n_hist
                for i in range(levels):
                    dilation = 2 ** i
                    pad = dilation * 2
                    layers += [
                        nn.Conv1d(in_ch, hidden, kernel_size=3, padding=pad, dilation=dilation),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    ]
                    in_ch = hidden
                self.net = nn.Sequential(*layers)
                self.future = nn.Sequential(nn.Linear(n_future, hidden), nn.GELU())
                self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_targets))
            def forward(self, history, future):
                x = history.transpose(1, 2)
                z = self.net(x)[..., -history_steps:]
                z = z[..., -1]
                f = self.future(future)
                return self.head(torch_mod.cat([z, f], dim=-1))
        return _Model()


class DLinearBaseline:
    def __new__(cls, history_steps: int, n_hist: int, n_future: int, n_targets: int, nn: Any, hidden: int = 128, dropout: float = 0.05):
        torch_mod = __import__("torch")
        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(history_steps * n_hist + n_future, n_targets)
                self.refine = nn.Sequential(nn.Linear(n_targets, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_targets))
            def forward(self, history, future):
                flat = history.reshape(history.shape[0], -1)
                base = self.linear(torch_mod.cat([flat, future], dim=-1))
                return base + 0.1 * self.refine(base)
        return _Model()


class ITransformerLite:
    def __new__(cls, history_steps: int, n_hist: int, n_future: int, n_targets: int, nn: Any, d_model: int = 96, nhead: int = 4, layers: int = 2, dropout: float = 0.1):
        torch_mod = __import__("torch")
        class _Model(nn.Module):
            recommended_learning_rate = 0.0002
            def __init__(self) -> None:
                super().__init__()
                self.var_embed = nn.Linear(history_steps, d_model)
                enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 3, dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
                self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
                self.future = nn.Sequential(nn.Linear(n_future, d_model), nn.GELU())
                self.head = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, n_targets))
            def forward(self, history, future):
                tokens = history.transpose(1, 2)
                z = self.encoder(self.var_embed(tokens)).mean(dim=1)
                f = self.future(future)
                return self.head(torch_mod.cat([z, f], dim=-1))
        return _Model()


class TSMixer:
    def __new__(cls, history_steps: int, n_hist: int, n_future: int, n_targets: int, nn: Any, hidden: int = 128, blocks: int = 3, dropout: float = 0.1):
        torch_mod = __import__("torch")
        class MixerBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.time_norm = nn.LayerNorm(n_hist)
                self.time_mlp = nn.Sequential(nn.Linear(history_steps, history_steps), nn.GELU(), nn.Dropout(dropout), nn.Linear(history_steps, history_steps))
                self.chan_norm = nn.LayerNorm(n_hist)
                self.chan_mlp = nn.Sequential(nn.Linear(n_hist, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_hist))
            def forward(self, x):
                y = self.time_norm(x).transpose(1, 2)
                x = x + self.time_mlp(y).transpose(1, 2)
                x = x + self.chan_mlp(self.chan_norm(x))
                return x
        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = nn.ModuleList([MixerBlock() for _ in range(blocks)])
                self.future = nn.Sequential(nn.Linear(n_future, hidden), nn.GELU())
                self.head = nn.Sequential(nn.Linear(history_steps * n_hist + hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_targets))
            def forward(self, history, future):
                x = history
                for block in self.blocks:
                    x = block(x)
                flat = x.reshape(x.shape[0], -1)
                return self.head(torch_mod.cat([flat, self.future(future)], dim=-1))
        return _Model()


def train_one(model: Any, loaders: dict[str, Any], torch: Any, nn: Any, cfg: dict[str, Any], device: Any) -> dict[str, Any]:
    lr = float(getattr(model, "recommended_learning_rate", cfg["learning_rate"]))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(cfg["weight_decay"]))
    loss_fn = nn.MSELoss()
    best_state = None
    best_val = math.inf
    best_epoch = -1
    wait = 0
    history = []
    min_delta = float(cfg.get("min_delta", 1e-6))
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        train_losses = []
        for batch in loaders["train_normal"]:
            opt.zero_grad(set_to_none=True)
            pred = model(batch.history, batch.future)
            loss = loss_fn(pred, batch.y_scaled)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("grad_clip_norm", 1.0)))
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
        val_loss = evaluate_loss(model, loaders["val_normal"], loss_fn, torch)
        train_loss = float(np.mean(train_losses)) if train_losses else math.nan
        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss})
        if best_val - val_loss > min_delta:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= int(cfg["patience"]):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_mse": best_val, "best_epoch": best_epoch, "epochs_run": len(history), "history": history}


def evaluate_loss(model: Any, loader: Any, loss_fn: Any, torch: Any) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch.history, batch.future)
            loss = loss_fn(pred, batch.y_scaled)
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else math.inf


def predict(model: Any, loader: Any, torch: Any, target_centers: np.ndarray, target_scales: np.ndarray) -> dict[str, np.ndarray]:
    model.eval()
    pred_scaled, true_scaled, true_raw, batch_ids = [], [], [], []
    centers = torch.as_tensor(target_centers, dtype=torch.float32, device=next(model.parameters()).device)
    scales = torch.as_tensor(target_scales, dtype=torch.float32, device=next(model.parameters()).device)
    with torch.no_grad():
        for batch in loader:
            p_scaled = model(batch.history, batch.future)
            p_raw = p_scaled * scales + centers
            pred_scaled.append(p_scaled.detach().cpu().numpy())
            true_scaled.append(batch.y_scaled.detach().cpu().numpy())
            true_raw.append(batch.y_raw.detach().cpu().numpy())
            batch_ids.append(batch.batch_id.detach().cpu().numpy())
    out = {
        "pred_scaled": np.concatenate(pred_scaled, axis=0),
        "true_scaled": np.concatenate(true_scaled, axis=0),
        "pred_raw": np.concatenate([x for x in []], axis=0) if False else None,
        "true_raw": np.concatenate(true_raw, axis=0),
        "batch_id": np.concatenate(batch_ids, axis=0),
    }
    out["pred_raw"] = out["pred_scaled"] * target_scales.reshape(1, -1) + target_centers.reshape(1, -1)
    return out


def make_loaders(metadata: dict[str, Any], split_ids: dict[str, list[int]], history_steps: int, horizon: int, train_cfg: dict[str, Any], control_mode: str, seed: int, torch: Any, DataLoader: Any, device: Any) -> dict[str, Any]:
    loaders = {}
    n_exog = len(metadata["exogenous_columns"])
    for split_name, batch_ids in split_ids.items():
        cap = train_cfg.get("max_windows_per_batch_train") if split_name == "train_normal" else train_cfg.get("max_windows_per_batch_eval")
        ds = ForecastWindowDataset(
            processed_root=metadata["processed_root"],
            batch_ids=batch_ids,
            history_steps=history_steps,
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
        ds = ControlWrapper(ds, control_mode, n_exog, seed + horizon)
        loaders[split_name] = DataLoader(
            ds,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=(split_name == "train_normal"),
            num_workers=int(train_cfg.get("num_workers", 1)),
            collate_fn=lambda batch, torch=torch, device=device: collate_numpy(batch, torch, device),
            drop_last=False,
        )
    return loaders


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    keys = ["model", "control_mode", "horizon_steps", "split", "target"]
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    out = []
    for key, vals in sorted(groups.items()):
        rec = dict(zip(keys, key))
        for metric in ["nrmse", "mae", "r2"]:
            arr = np.asarray([float(v[metric]) for v in vals], dtype=float)
            rec[f"{metric}_mean"] = float(np.nanmean(arr))
            rec[f"{metric}_std"] = float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0
            rec[f"{metric}_n"] = int(np.isfinite(arr).sum())
        out.append(rec)
    return out


def get_shard() -> tuple[int, int]:
    shard_count = int(os.environ.get("PHASE31_SHARD_COUNT", "1"))
    shard_index = int(os.environ.get("PHASE31_SHARD_INDEX", os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    if shard_count < 1:
        raise ValueError("PHASE31_SHARD_COUNT must be >= 1")
    if not 0 <= shard_index < shard_count:
        raise ValueError("PHASE31_SHARD_INDEX must satisfy 0 <= index < count")
    return shard_index, shard_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model/phase31_infosci_modern_deep_baselines.json")
    args = parser.parse_args()
    cfg = load_json(PROJECT_ROOT / args.config)
    torch, nn, DataLoader, _F = import_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")
    metadata = build_forecasting_metadata(cfg, PROJECT_ROOT)
    shard_index, shard_count = get_shard()
    output_root = ensure_dir(PROJECT_ROOT / cfg["output_root"])
    metrics_dir = ensure_dir(output_root / "metrics")
    pred_dir = ensure_dir(output_root / "predictions")
    hist_dir = ensure_dir(output_root / "training_history")
    manifest = {
        "config": cfg,
        "target_columns": metadata["target_columns"],
        "exogenous_columns": metadata["exogenous_columns"],
        "history_columns": metadata["history_columns"],
        "shard_index": shard_index,
        "shard_count": shard_count,
        "started_at": time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    shard_suffix = f"_shard{shard_index:02d}" if shard_count > 1 else ""
    (output_root / f"run_manifest_start{shard_suffix}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    metric_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    tasks: list[tuple[int, int, int, str, str]] = []
    for repeat_id, seed in enumerate(cfg["repeat_seeds"]):
        for horizon in cfg["horizons"]:
            for control_mode in cfg["control_modes"]:
                for model_name in cfg["models"]:
                    tasks.append((repeat_id, int(seed), int(horizon), str(control_mode), str(model_name)))
    selected_tasks = [task for task_id, task in enumerate(tasks) if task_id % shard_count == shard_index]
    log(f"Shard {shard_index}/{shard_count}: selected {len(selected_tasks)} of {len(tasks)} tasks")
    split_cache: dict[int, dict[str, list[int]]] = {}
    for repeat_id, seed in enumerate(cfg["repeat_seeds"]):
        split_ids = split_cache.setdefault(int(seed), make_repeated_split(metadata["splits"], int(seed), cfg["normal_split_counts"]))
        for horizon in cfg["horizons"]:
            for control_mode in cfg["control_modes"]:
                loaders = make_loaders(metadata, split_ids, int(cfg["history_steps"]), int(horizon), cfg["train"], control_mode, int(seed), torch, DataLoader, device)
                for model_name in cfg["models"]:
                    if (repeat_id, int(seed), int(horizon), str(control_mode), str(model_name)) not in selected_tasks:
                        continue
                    set_seed(int(seed), torch)
                    model = build_model(
                        model_name,
                        cfg.get("model_params", {}).get(model_name, {}),
                        int(cfg["history_steps"]),
                        len(metadata["history_columns"]),
                        len(metadata["exogenous_columns"]),
                        len(metadata["target_columns"]),
                        nn,
                        torch,
                    ).to(device)
                    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    run_name = f"r{repeat_id:02d}_s{seed}_h{horizon}_{control_mode}_{model_name}"
                    log(f"Train {run_name} params={n_params}")
                    train_info = train_one(model, loaders, torch, nn, cfg["train"], device)
                    write_rows(hist_dir / f"{run_name}_history.csv", [dict(run_name=run_name, **r) for r in train_info["history"]])
                    for split_name in ["val_normal", "test_normal", "test_fault"]:
                        pred = predict(model, loaders[split_name], torch, metadata["target_centers"], metadata["target_scales"])
                        finite_or_raise(pred["pred_raw"], f"predictions {run_name} {split_name}")
                        np.savez_compressed(
                            pred_dir / f"{run_name}_{split_name}.npz",
                            pred_raw=pred["pred_raw"],
                            true_raw=pred["true_raw"],
                            pred_scaled=pred["pred_scaled"],
                            true_scaled=pred["true_scaled"],
                            batch_id=pred["batch_id"],
                            target_columns=np.asarray(metadata["target_columns"], dtype=object),
                        )
                        m = regression_metrics(pred["true_raw"], pred["pred_raw"], metadata["target_scales"])
                        for j, target in enumerate(metadata["target_columns"]):
                            metric_rows.append({
                                "repeat_id": repeat_id,
                                "seed": seed,
                                "model": model_name,
                                "control_mode": control_mode,
                                "horizon_steps": horizon,
                                "split": split_name,
                                "target": target,
                                "nrmse": float(m["nrmse"][j]),
                                "mae": float(m["mae"][j]),
                                "r2": float(m["r2"][j]) if np.isfinite(m["r2"][j]) else "",
                                "n_windows": int(pred["true_raw"].shape[0]),
                            })
                    run_rows.append({
                        "run_name": run_name,
                        "repeat_id": repeat_id,
                        "seed": seed,
                        "model": model_name,
                        "control_mode": control_mode,
                        "horizon_steps": horizon,
                        "best_epoch": train_info["best_epoch"],
                        "epochs_run": train_info["epochs_run"],
                        "best_val_mse": train_info["best_val_mse"],
                        "n_params": n_params,
                    })
                    write_rows(metrics_dir / f"per_target_metrics_partial{shard_suffix}.csv", metric_rows)
                    write_rows(metrics_dir / f"run_summary_partial{shard_suffix}.csv", run_rows)
    summary = summarize(metric_rows)
    write_rows(metrics_dir / f"per_target_metrics{shard_suffix}.csv", metric_rows)
    write_rows(metrics_dir / f"summary_mean_std{shard_suffix}.csv", summary)
    write_rows(metrics_dir / f"run_summary{shard_suffix}.csv", run_rows)
    manifest["finished_at"] = time.strftime('%Y-%m-%d %H:%M:%S')
    manifest["outputs"] = {
        "per_target_metrics": str(metrics_dir / f"per_target_metrics{shard_suffix}.csv"),
        "summary_mean_std": str(metrics_dir / f"summary_mean_std{shard_suffix}.csv"),
        "run_summary": str(metrics_dir / f"run_summary{shard_suffix}.csv"),
        "predictions": str(pred_dir),
    }
    (output_root / f"run_manifest{shard_suffix}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log("Phase31 completed")


if __name__ == "__main__":
    main()

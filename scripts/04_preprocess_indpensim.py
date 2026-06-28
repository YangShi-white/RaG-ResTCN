#!/usr/bin/env python3
"""Phase 04 preprocessing for the IndPenSim restart project.

This script is intentionally conservative about the known raw CSV alignment
issue. It uses the first 37 raw row fields as process columns, infers batch IDs
from Time (h) resets, reads fault labels from the statistics CSV, and treats raw
row positions 37:2237 as the 2200 Raman wavelengths 2400..201.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


MISSING_VALUES = {"", "nan", "NaN", "NAN", "null", "None"}
PROCESS_COLUMN_COUNT = 37
RAMAN_START_INDEX = 37
RAMAN_COLUMN_COUNT = 2200


@dataclass
class BatchBuffer:
    batch_id: int
    rows: list[list[float]]
    source_row_start: int
    source_row_end: int = 0
    previous_time: float | None = None
    last_raman_sample_time: float = -math.inf


def parse_float(value: str | None) -> float:
    if value is None:
        return math.nan
    stripped = value.strip()
    if stripped in MISSING_VALUES:
        return math.nan
    try:
        return float(stripped)
    except ValueError:
        return math.nan


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_fault_statistics(path: Path) -> dict[int, dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        stats: dict[int, dict[str, Any]] = {}
        for row in reader:
            batch_id = int(float(row["Batch ref"]))
            fault_key = "Fault ref(0-NoFault 1-Fault)"
            stats[batch_id] = {
                "batch_id": batch_id,
                "fault_label": int(float(row[fault_key])),
                "penicillin_harvested_during_batch_kg": parse_float(
                    row.get("Penicllin_harvested_during_batch(kg)")
                ),
                "penicillin_harvested_end_of_batch_kg": parse_float(
                    row.get("Penicllin_harvested_end_of_batch (kg)")
                ),
                "penicillin_yield_total_kg": parse_float(
                    row.get("Penicllin_yield_total (kg)")
                ),
            }
    return stats


def infer_wavelengths() -> np.ndarray:
    return np.arange(2400, 200, -1, dtype=np.float32)


def safe_column_index(columns: list[str], name: str) -> int:
    try:
        return columns.index(name)
    except ValueError as exc:
        raise ValueError(f"Required column is missing: {name}") from exc


def finite_median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    return float(np.median(finite))


def gradient_by_time(values: np.ndarray, time_h: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(values) & np.isfinite(time_h)
    if finite.sum() < 2:
        return out
    idx = np.where(finite)[0]
    out[idx] = np.gradient(values[idx].astype(np.float64), time_h[idx].astype(np.float64)).astype(
        np.float32
    )
    return out


def cumulative_trapezoid_rate(rate: np.ndarray, time_h: np.ndarray) -> np.ndarray:
    out = np.zeros(rate.shape, dtype=np.float32)
    finite = np.isfinite(rate) & np.isfinite(time_h)
    if finite.sum() < 2:
        out[:] = np.nan
        return out
    clean_rate = rate.astype(np.float64).copy()
    clean_time = time_h.astype(np.float64).copy()
    valid = np.isfinite(clean_rate) & np.isfinite(clean_time)
    clean_rate[~valid] = np.nan
    clean_rate = np.where(np.isfinite(clean_rate), clean_rate, 0.0)
    for i in range(1, len(rate)):
        dt = clean_time[i] - clean_time[i - 1]
        if not np.isfinite(dt) or dt < 0:
            out[i] = out[i - 1]
            continue
        out[i] = out[i - 1] + 0.5 * (clean_rate[i] + clean_rate[i - 1]) * dt
    return out


def make_derived(process: np.ndarray, columns: list[str]) -> tuple[np.ndarray, list[str]]:
    time_h = process[:, safe_column_index(columns, "Time (h)")].astype(np.float32)
    substrate = process[:, safe_column_index(columns, "Substrate concentration(S:g/L)")].astype(
        np.float32
    )
    penicillin = process[:, safe_column_index(columns, "Penicillin concentration(P:g/L)")].astype(
        np.float32
    )
    volume = process[:, safe_column_index(columns, "Vessel Volume(V:L)")].astype(np.float32)
    biomass_offline = process[
        :, safe_column_index(columns, "Offline Biomass concentratio(X_offline:X(g L^{-1}))")
    ].astype(np.float32)
    cer = process[:, safe_column_index(columns, "Carbon evolution rate(CER:g/h)")].astype(
        np.float32
    )
    our = process[:, safe_column_index(columns, "Oxygen Uptake Rate(OUR:(g min^{-1}))")].astype(
        np.float32
    )
    sugar_feed = process[:, safe_column_index(columns, "Sugar feed rate(Fs:L/h)")].astype(
        np.float32
    )
    do = process[
        :, safe_column_index(columns, "Dissolved oxygen concentration(DO2:mg/L)")
    ].astype(np.float32)

    eps = np.float32(1e-6)
    derived_columns = [
        "substrate_mass_g",
        "penicillin_mass_g",
        "offline_biomass_mass_g",
        "rq_proxy",
        "cumulative_sugar_feed_L",
        "sugar_feed_rate_derivative_L_per_h2",
        "do_slope_mg_per_L_per_h",
    ]
    substrate_mass = volume * substrate
    penicillin_mass = volume * penicillin
    biomass_mass = volume * biomass_offline
    rq_proxy = cer / (60.0 * our + eps)
    cumulative_sugar = cumulative_trapezoid_rate(sugar_feed, time_h)
    feed_derivative = gradient_by_time(sugar_feed, time_h)
    do_slope = gradient_by_time(do, time_h)
    derived = np.column_stack(
        [
            substrate_mass,
            penicillin_mass,
            biomass_mass,
            rq_proxy,
            cumulative_sugar,
            feed_derivative,
            do_slope,
        ]
    ).astype(np.float32)
    return derived, derived_columns


def parse_raman(row: list[str]) -> np.ndarray:
    raw = row[RAMAN_START_INDEX : RAMAN_START_INDEX + RAMAN_COLUMN_COUNT]
    if len(raw) < RAMAN_COLUMN_COUNT:
        raw = raw + [""] * (RAMAN_COLUMN_COUNT - len(raw))
    elif len(raw) > RAMAN_COLUMN_COUNT:
        raw = raw[:RAMAN_COLUMN_COUNT]
    return np.array([parse_float(value) for value in raw], dtype=np.float32)


def robust_process_summary(process: np.ndarray, columns: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for idx, col in enumerate(columns):
        values = process[:, idx]
        finite = np.isfinite(values)
        summary[col] = {
            "non_missing": int(finite.sum()),
            "missing": int((~finite).sum()),
            "missing_ratio": float((~finite).mean()),
            "median": finite_median(values),
            "min": float(np.nanmin(values)) if finite.any() else math.nan,
            "max": float(np.nanmax(values)) if finite.any() else math.nan,
        }
    return summary


def flush_batch(
    batch: BatchBuffer,
    process_columns: list[str],
    fault_stats: dict[int, dict[str, Any]],
    output_root: Path,
    metadata_rows: list[dict[str, Any]],
    process_summaries: dict[str, Any],
) -> None:
    process = np.asarray(batch.rows, dtype=np.float32)
    if process.size == 0:
        return
    time_h = process[:, safe_column_index(process_columns, "Time (h)")].astype(np.float32)
    derived, derived_columns = make_derived(process, process_columns)
    stat = fault_stats.get(batch.batch_id, {})
    fault_label = int(stat.get("fault_label", 0))

    batch_dir = output_root / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        batch_dir / f"batch_{batch.batch_id:03d}.npz",
        process=process,
        process_columns=np.array(process_columns, dtype=object),
        derived=derived,
        derived_columns=np.array(derived_columns, dtype=object),
        time_h=time_h,
        batch_id=np.int32(batch.batch_id),
        fault_label=np.int32(fault_label),
    )

    duration_h = float(np.nanmax(time_h) - np.nanmin(time_h)) if len(time_h) else math.nan
    metadata_rows.append(
        {
            "batch_id": batch.batch_id,
            "fault_label": fault_label,
            "n_rows": int(process.shape[0]),
            "time_start_h": float(np.nanmin(time_h)) if len(time_h) else math.nan,
            "time_end_h": float(np.nanmax(time_h)) if len(time_h) else math.nan,
            "duration_h": duration_h,
            "source_row_start": batch.source_row_start,
            "source_row_end": batch.source_row_end,
            "penicillin_harvested_during_batch_kg": stat.get(
                "penicillin_harvested_during_batch_kg", math.nan
            ),
            "penicillin_harvested_end_of_batch_kg": stat.get(
                "penicillin_harvested_end_of_batch_kg", math.nan
            ),
            "penicillin_yield_total_kg": stat.get("penicillin_yield_total_kg", math.nan),
            "process_npz": str(batch_dir / f"batch_{batch.batch_id:03d}.npz"),
        }
    )
    process_summaries[f"batch_{batch.batch_id:03d}"] = robust_process_summary(
        process, process_columns
    )


def write_csv_dicts(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess IndPenSim for Phase 04")
    parser.add_argument(
        "--config",
        default="configs/data/preprocess_phase04.json",
        help="Phase 04 preprocessing config",
    )
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--raman-sample-step-hours", type=float, default=None)
    parser.add_argument(
        "--write-full-raman",
        action="store_true",
        help="Also export full Raman matrices per batch. This is large and disabled by default.",
    )
    args = parser.parse_args()

    start = time.time()
    project_root = Path.cwd()
    config = load_json(project_root / args.config)
    data_config = load_json(project_root / config["data_config"])
    variable_roles = load_json(project_root / config["variable_roles"])
    raw_csv = Path(data_config["raw_csv"])
    stats_csv = Path(data_config["statistics_csv"])
    output_root = project_root / config.get("output_root", "data/processed/phase04")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metadata").mkdir(parents=True, exist_ok=True)
    (output_root / "raman").mkdir(parents=True, exist_ok=True)
    (output_root / "stats").mkdir(parents=True, exist_ok=True)

    sample_step = (
        args.raman_sample_step_hours
        if args.raman_sample_step_hours is not None
        else float(config["raman_sampling"]["sample_step_hours"])
    )
    fault_stats = load_fault_statistics(stats_csv)
    wavelengths = infer_wavelengths()

    raman_spectra: list[np.ndarray] = []
    raman_sample_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    process_summaries: dict[str, Any] = {}
    full_raman_buffers: list[np.ndarray] = []

    with raw_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        process_columns = header[:PROCESS_COLUMN_COUNT]
        time_idx = safe_column_index(process_columns, "Time (h)")
        raman_recorded_idx = safe_column_index(process_columns, " 1-Raman spec recorded")

        batch = BatchBuffer(batch_id=1, rows=[], source_row_start=1)
        previous_time: float | None = None
        source_row = 0
        for row in reader:
            source_row += 1
            current_time = parse_float(row[time_idx] if time_idx < len(row) else None)
            if (
                previous_time is not None
                and np.isfinite(current_time)
                and np.isfinite(previous_time)
                and current_time < previous_time
            ):
                batch.source_row_end = source_row - 1
                flush_batch(
                    batch,
                    process_columns,
                    fault_stats,
                    output_root,
                    metadata_rows,
                    process_summaries,
                )
                if args.write_full_raman and full_raman_buffers:
                    raman_dir = output_root / "raman" / "full"
                    raman_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        raman_dir / f"batch_{batch.batch_id:03d}_raman.npz",
                        spectra=np.asarray(full_raman_buffers, dtype=np.float32),
                        wavelengths=wavelengths,
                    )
                if args.max_batches is not None and batch.batch_id >= args.max_batches:
                    break
                batch = BatchBuffer(
                    batch_id=batch.batch_id + 1,
                    rows=[],
                    source_row_start=source_row,
                )
                full_raman_buffers = []

            process_row = [parse_float(value) for value in row[:PROCESS_COLUMN_COUNT]]
            if len(process_row) < PROCESS_COLUMN_COUNT:
                process_row.extend([math.nan] * (PROCESS_COLUMN_COUNT - len(process_row)))
            batch.rows.append(process_row)

            raman_recorded = process_row[raman_recorded_idx]
            should_sample = (
                np.isfinite(current_time)
                and np.isfinite(raman_recorded)
                and raman_recorded >= 0.5
                and current_time - batch.last_raman_sample_time >= sample_step
            )
            if should_sample or args.write_full_raman:
                spectrum = parse_raman(row)
                if args.write_full_raman:
                    full_raman_buffers.append(spectrum)
                if should_sample and np.isfinite(spectrum).any() and float(np.nanmax(np.abs(spectrum))) > 0:
                    batch.last_raman_sample_time = current_time
                    raman_spectra.append(spectrum)
                    raman_sample_rows.append(
                        {
                            "sample_index": len(raman_spectra) - 1,
                            "batch_id": batch.batch_id,
                            "fault_label": int(fault_stats.get(batch.batch_id, {}).get("fault_label", 0)),
                            "source_row": source_row,
                            "time_h": current_time,
                            "raman_recorded_flag": raman_recorded,
                        }
                    )

            previous_time = current_time
        else:
            batch.source_row_end = source_row
            flush_batch(
                batch,
                process_columns,
                fault_stats,
                output_root,
                metadata_rows,
                process_summaries,
            )
            if args.write_full_raman and full_raman_buffers:
                raman_dir = output_root / "raman" / "full"
                raman_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    raman_dir / f"batch_{batch.batch_id:03d}_raman.npz",
                    spectra=np.asarray(full_raman_buffers, dtype=np.float32),
                    wavelengths=wavelengths,
                )

    metadata_path = output_root / "metadata" / "batch_metadata.csv"
    write_csv_dicts(metadata_path, metadata_rows)
    write_csv_dicts(output_root / "raman" / "raman_sample_rows.csv", raman_sample_rows)
    np.save(output_root / "raman" / "raman_sample_spectra.npy", np.asarray(raman_spectra, dtype=np.float32))
    np.save(output_root / "raman" / "raman_wavelengths.npy", wavelengths)

    process_summary_path = output_root / "stats" / "process_column_summaries.json"
    process_summary_path.write_text(json.dumps(process_summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    alignment_summary = {
        "raw_csv": str(raw_csv),
        "statistics_csv": str(stats_csv),
        "process_columns": process_columns,
        "process_column_count": len(process_columns),
        "raman_raw_row_start_index": RAMAN_START_INDEX,
        "raman_column_count": RAMAN_COLUMN_COUNT,
        "wavelength_start": 2400,
        "wavelength_end": 201,
        "wavelength_count": int(wavelengths.size),
        "batch_id_source": "Time (h) reset order",
        "fault_label_source": str(stats_csv),
        "raw_batch_id_used": False,
        "raw_fault_flag_used": False,
        "raman_sample_count": len(raman_sample_rows),
        "write_full_raman": bool(args.write_full_raman),
        "elapsed_seconds": round(time.time() - start, 2),
        "variable_roles": variable_roles,
    }
    (output_root / "metadata" / "alignment_summary.json").write_text(
        json.dumps(alignment_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest = {
        "phase": "04_preprocessing",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metadata_csv": str(metadata_path),
        "batch_npz_dir": str(output_root / "batches"),
        "raman_sample_rows_csv": str(output_root / "raman" / "raman_sample_rows.csv"),
        "raman_sample_spectra_npy": str(output_root / "raman" / "raman_sample_spectra.npy"),
        "raman_wavelengths_npy": str(output_root / "raman" / "raman_wavelengths.npy"),
        "process_column_summaries_json": str(process_summary_path),
        "alignment_summary_json": str(output_root / "metadata" / "alignment_summary.json"),
        "n_batches": len(metadata_rows),
        "n_raman_samples": len(raman_sample_rows),
        "full_raman_exported": bool(args.write_full_raman),
    }
    (output_root / "processed_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    main()

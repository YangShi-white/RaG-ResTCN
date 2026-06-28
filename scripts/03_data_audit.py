#!/usr/bin/env python3
"""Raw data audit for the IndPenSim dataset.

The script intentionally uses only Python standard-library modules so it can
run before the project environment is fully installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any


MISSING_VALUES = {"", "nan", "NaN", "NAN", "null", "None"}


def summarize_counter(counter: Counter[str], max_full_values: int = 20) -> dict[str, Any]:
    summary = {
        "unique_values": len(counter),
        "top_values": [
            {"value": value, "count": count}
            for value, count in counter.most_common(max_full_values)
        ],
        "full_counts_included": len(counter) <= max_full_values,
    }
    if len(counter) <= max_full_values:
        summary["full_counts"] = dict(counter)
    return summary


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def file_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
    }


def audit_statistics_csv(path: Path) -> dict[str, Any]:
    info = file_info(path)
    if not info["exists"]:
        return info

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    fault_counts = None
    fault_batches = []
    fault_field = next((col for col in header if "Fault ref" in col), None)
    if fault_field is not None:
        fault_idx = header.index(fault_field)
        batch_idx = header.index("Batch ref") if "Batch ref" in header else 0
        counter: Counter[str] = Counter()
        for row in rows:
            if fault_idx >= len(row):
                value = ""
            else:
                value = row[fault_idx].strip()
            counter[value] += 1
            if value == "1" and batch_idx < len(row):
                fault_batches.append(row[batch_idx])
        fault_counts = dict(counter)

    info.update(
        {
            "columns": len(header),
            "header": header,
            "data_rows": len(rows),
            "first_rows": rows[:5],
            "fault_field": fault_field,
            "fault_counts": fault_counts,
            "fault_batches": fault_batches,
        }
    )
    return info


def infer_sections(header: list[str]) -> dict[str, Any]:
    """Infer coarse column sections from the raw CSV header."""
    numeric_like = []
    raman_prefixed = []
    for col in header:
        stripped = col.strip()
        if stripped.startswith("Raman_"):
            raman_prefixed.append(col)
        else:
            try:
                float(stripped)
                numeric_like.append(col)
            except ValueError:
                pass

    return {
        "total_columns": len(header),
        "first_45_columns": header[:45],
        "last_20_columns": header[-20:],
        "raman_prefixed_columns": len(raman_prefixed),
        "numeric_like_columns": len(numeric_like),
        "numeric_like_first_10": numeric_like[:10],
        "numeric_like_last_10": numeric_like[-10:],
    }


def audit_raw_csv(path: Path, count_missing: bool = True) -> dict[str, Any]:
    info = file_info(path)
    if not info["exists"]:
        return info

    start_time = time.time()

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        n_cols = len(header)

        missing_counts = [0] * n_cols
        non_missing_counts = [0] * n_cols
        row_count = 0
        malformed_rows = 0
        row_length_counts: Counter[int] = Counter()
        row_length_mismatch_examples: list[dict[str, Any]] = []
        batch_lengths: list[int] = []
        current_batch_len = 0
        previous_time = None
        first_data_row = None
        last_data_row = None

        try:
            time_idx = header.index("Time (h)")
        except ValueError:
            time_idx = 0
        tracked_columns = [
            col
            for col in (
                "Batch reference(Batch_ref:Batch ref)",
                "Batch ID",
                "Fault reference(Fault_ref:Fault ref)",
                "Fault flag",
            )
            if col in header
        ]
        tracked_indices = {col: header.index(col) for col in tracked_columns}
        tracked_value_counts: dict[str, Counter[str]] = {
            col: Counter() for col in tracked_columns
        }

        for row in reader:
            row_count += 1
            original_row_len = len(row)
            row_length_counts[original_row_len] += 1

            if first_data_row is None:
                first_data_row = row[: min(20, len(row))]
            last_data_row = row[: min(20, len(row))]

            if original_row_len != n_cols:
                malformed_rows += 1
                if len(row_length_mismatch_examples) < 5:
                    row_length_mismatch_examples.append(
                        {
                            "data_row_number": row_count,
                            "expected_columns": n_cols,
                            "actual_columns": original_row_len,
                            "last_8_values": row[-8:],
                        }
                    )
                if original_row_len < n_cols:
                    row = row + [""] * (n_cols - original_row_len)
                else:
                    row = row[:n_cols]

            for col, idx in tracked_indices.items():
                value = row[idx].strip() if idx < len(row) else ""
                tracked_value_counts[col][value] += 1

            if count_missing:
                for idx, value in enumerate(row):
                    if value.strip() in MISSING_VALUES:
                        missing_counts[idx] += 1
                    else:
                        non_missing_counts[idx] += 1

            try:
                current_time = float(row[time_idx])
            except (ValueError, IndexError):
                current_time = None

            if current_time is not None:
                if previous_time is not None and current_time < previous_time:
                    batch_lengths.append(current_batch_len)
                    current_batch_len = 0
                previous_time = current_time

            current_batch_len += 1

        if current_batch_len:
            batch_lengths.append(current_batch_len)

    all_missing_columns = [
        header[idx] for idx, count in enumerate(non_missing_counts) if count == 0
    ]
    top_missing = sorted(
        (
            {
                "column": header[idx],
                "missing": missing_counts[idx],
                "missing_ratio": round(missing_counts[idx] / row_count, 6)
                if row_count
                else None,
            }
            for idx in range(n_cols)
        ),
        key=lambda item: item["missing"],
        reverse=True,
    )[:30]

    row_length_min = min(row_length_counts) if row_length_counts else None
    row_length_max = max(row_length_counts) if row_length_counts else None
    tracked_summaries = {
        col: summarize_counter(counter) for col, counter in tracked_value_counts.items()
    }
    alignment_notes: dict[str, Any] = {
        "header_columns": n_cols,
        "observed_row_length_min": row_length_min,
        "observed_row_length_max": row_length_max,
        "uniform_row_width": row_length_min == row_length_max,
    }
    if row_length_min is not None and row_length_max is not None:
        alignment_notes["header_minus_observed_row_columns"] = n_cols - row_length_min

    batch_id_summary = tracked_summaries.get("Batch ID")
    fault_flag_summary = tracked_summaries.get("Fault flag")
    if (
        batch_id_summary is not None
        and fault_flag_summary is not None
        and batch_id_summary["unique_values"] > 100
        and fault_flag_summary["unique_values"] > 100
    ):
        alignment_notes.update(
            {
                "suspected_shifted_columns": ["Batch ID", "Fault flag"],
                "do_not_use_raw_batch_id_or_fault_flag_as_labels": True,
                "recommended_batch_id_source": "infer sequential batch index from Time (h) resets",
                "recommended_fault_label_source": "100_Batches_IndPenSim_Statistics.csv",
                "recommended_raman_start_index": header.index("Batch ID")
                if "Batch ID" in header
                else None,
                "recommended_raman_column_count": 2200,
                "recommended_raman_wavelength_range": "2400 down to 201",
            }
        )

    info.update(
        {
            "columns": n_cols,
            "data_rows": row_count,
            "malformed_rows": malformed_rows,
            "row_length_min": row_length_min,
            "row_length_max": row_length_max,
            "row_length_histogram": {
                str(length): row_length_counts[length]
                for length in sorted(row_length_counts)
            },
            "row_length_mismatch_examples": row_length_mismatch_examples,
            "tracked_value_summaries": tracked_summaries,
            "alignment_notes": alignment_notes,
            "header_sections": infer_sections(header),
            "first_data_row_first_20_values": first_data_row,
            "last_data_row_first_20_values": last_data_row,
            "inferred_batches_from_time_reset": len(batch_lengths),
            "batch_lengths": batch_lengths,
            "batch_length_min": min(batch_lengths) if batch_lengths else None,
            "batch_length_max": max(batch_lengths) if batch_lengths else None,
            "batch_length_mean": round(sum(batch_lengths) / len(batch_lengths), 2)
            if batch_lengths
            else None,
            "all_missing_columns": all_missing_columns,
            "top_missing_columns": top_missing,
            "elapsed_seconds": round(time.time() - start_time, 2),
        }
    )
    return info


def write_markdown_summary(audit: dict[str, Any], output_md: Path) -> None:
    raw = audit["raw_csv"]
    stats = audit["statistics_csv"]

    lines = [
        "# Phase 03 Data Audit Auto Summary",
        "",
        "This file was generated by `scripts/03_data_audit.py`.",
        "",
        "## Raw CSV",
        "",
        f"- Path: `{raw['path']}`",
        f"- Exists: `{raw['exists']}`",
        f"- Size MB: `{raw.get('size_mb')}`",
        f"- Data rows: `{raw.get('data_rows')}`",
        f"- Columns: `{raw.get('columns')}`",
        f"- Rows with header-width mismatch: `{raw.get('malformed_rows')}`",
        f"- Data row length range: `{raw.get('row_length_min')}` to `{raw.get('row_length_max')}`",
        f"- Data row length histogram: `{raw.get('row_length_histogram')}`",
        f"- Alignment notes: `{raw.get('alignment_notes')}`",
        f"- Inferred batches: `{raw.get('inferred_batches_from_time_reset')}`",
        f"- Batch length range: `{raw.get('batch_length_min')}` to `{raw.get('batch_length_max')}`",
        f"- Batch length mean: `{raw.get('batch_length_mean')}`",
        f"- Full scan elapsed seconds: `{raw.get('elapsed_seconds')}`",
        f"- Raw tracked value summaries: `{raw.get('tracked_value_summaries')}`",
        "",
        "## Header Sections",
        "",
        "First 45 columns:",
        "",
        "```text",
        "\n".join(raw.get("header_sections", {}).get("first_45_columns", [])),
        "```",
        "",
        "Last 20 columns:",
        "",
        "```text",
        "\n".join(raw.get("header_sections", {}).get("last_20_columns", [])),
        "```",
        "",
        "## Missing Columns",
        "",
        "All-missing columns:",
        "",
        "```text",
        "\n".join(raw.get("all_missing_columns", [])) or "None",
        "```",
        "",
        "Top missing columns:",
        "",
        "| Column | Missing | Missing Ratio |",
        "|---|---:|---:|",
    ]

    for item in raw.get("top_missing_columns", []):
        lines.append(
            f"| {item['column']} | {item['missing']} | {item['missing_ratio']} |"
        )

    lines.extend(
        [
            "",
            "## Statistics CSV",
            "",
            f"- Path: `{stats['path']}`",
            f"- Exists: `{stats['exists']}`",
            f"- Size MB: `{stats.get('size_mb')}`",
            f"- Data rows: `{stats.get('data_rows')}`",
            f"- Columns: `{stats.get('columns')}`",
            f"- Fault field: `{stats.get('fault_field')}`",
            f"- Fault counts: `{stats.get('fault_counts')}`",
            f"- Fault batches: `{stats.get('fault_batches')}`",
            "",
        ]
    )

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit raw IndPenSim CSV files")
    parser.add_argument("--config", required=True, help="Path to JSON data config")
    parser.add_argument("--output-json", required=True, help="Output JSON path")
    parser.add_argument("--output-md", help="Optional output Markdown summary path")
    parser.add_argument(
        "--skip-missing-count",
        action="store_true",
        help="Skip per-column missing counts for faster header-only scan",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    raw_csv = Path(config["raw_csv"])
    statistics_csv = Path(config["statistics_csv"])

    audit = {
        "dataset_name": config.get("dataset_name", "IndPenSim"),
        "raw_csv": audit_raw_csv(raw_csv, count_missing=not args.skip_missing_count),
        "statistics_csv": audit_statistics_csv(statistics_csv),
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.output_md:
        write_markdown_summary(audit, Path(args.output_md))

    print(f"Audit JSON written to {output_json}")
    if args.output_md:
        print(f"Audit Markdown summary written to {args.output_md}")


if __name__ == "__main__":
    main()

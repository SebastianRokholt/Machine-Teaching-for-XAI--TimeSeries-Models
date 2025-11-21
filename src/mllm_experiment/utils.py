# src/mllm_experiment/utils.py
# Logging, CSV / JSONL helpers
from __future__ import annotations
import csv
from pathlib import Path
from typing import Any


def write_dicts_to_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Append a list of dictionaries to a CSV file.

    If the file does not exist, this function creates it and writes the
    header row. If it exists, it appends only the data rows.

    Args:
        rows: List of row dictionaries.
        path: Path to the CSV file.
    """
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.is_file()
    fieldnames = list(rows[0].keys())

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def log_exam_results(rows: list[dict[str, Any]], output_root: Path) -> None:
    """Log exam-phase results to exam_results.csv.

    Args:
        rows: List of exam result rows.
        output_root: Root output directory for logs.
    """
    path = output_root / "exam_results.csv"
    write_dicts_to_csv(rows, path)


def log_teaching_results(rows: list[dict[str, Any]], output_root: Path) -> None:
    """Log teaching-phase results to teaching_results.csv.

    Args:
        rows: List of teaching result rows.
        output_root: Root output directory for logs.
    """
    path = output_root / "teaching_results.csv"
    write_dicts_to_csv(rows, path)


def log_participant_summaries(rows: list[dict[str, Any]], output_root: Path) -> None:
    """Log per-participant summaries to participants.csv.

    Args:
        rows: List of participant summary rows.
        output_root: Root output directory for logs.
    """
    path = output_root / "participants.csv"
    write_dicts_to_csv(rows, path)

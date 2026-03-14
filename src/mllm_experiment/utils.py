# src/mllm_experiment/utils.py
# Logging, CSV / JSONL helpers
from __future__ import annotations
import csv
import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Mapping


_csv_locks_guard = threading.Lock()
_csv_locks: dict[Path, threading.Lock] = {}


def setup_logging(
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
    log_file: Path | None = None,
    events_log_file: Path | None = None,
    http_debug: bool = False,
) -> None:
    """Configure root logging for the experiment runner.

    This function configures stream and file handlers for readable logs
    and a JSONL handler for structured event records.

    Args:
        log_level: Logging level for experiment loggers.
        log_file: Optional path for the readable text log file.
        events_log_file: Optional path for structured JSONL event logs.
        http_debug: Flag that enables debug logs for HTTP client internals.
    """
    log_level_mapper = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    log_level_value = log_level_mapper.get(log_level.upper(), logging.INFO)

    readable_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    handlers: list[logging.Handler] = []

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(readable_formatter)
    handlers.append(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(readable_formatter)
        handlers.append(file_handler)

    if events_log_file is not None:
        events_log_file.parent.mkdir(parents=True, exist_ok=True)
        events_handler = logging.FileHandler(events_log_file, encoding="utf-8")
        events_handler.setFormatter(_EventJsonFormatter())
        events_handler.addFilter(_EventRecordFilter())
        handlers.append(events_handler)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level_value)
    for handler in handlers:
        root_logger.addHandler(handler)

    third_party_level = log_level_value if http_debug else logging.WARNING
    for logger_name in ("httpx", "httpcore", "openai", "openai._base_client"):
        logging.getLogger(logger_name).setLevel(third_party_level)


@dataclass(slots=True)
class ExperimentEventLogger:
    """Emit structured experiment events with a shared run identifier.

    Attributes:
        run_id: Stable identifier for one full experiment run.
    """

    run_id: str

    def log(
        self,
        logger: logging.Logger,
        event: str,
        level: int = logging.INFO,
        **fields: Any,
    ) -> None:
        """Write one structured event through the Python logging pipeline.

        Args:
            logger: Logger instance that emits the event.
            event: Stable event name.
            level: Logging level for the event.
            **fields: Event fields that describe context and diagnostics.
        """
        compact_fields = {k: v for k, v in fields.items() if v is not None}
        logger.log(
            level,
            _format_event_message(event=event, fields=compact_fields),
            extra={
                "event": event,
                "run_id": self.run_id,
                "event_fields": compact_fields,
            },
        )


class _EventRecordFilter(logging.Filter):
    """Keep only structured event records for the JSONL handler."""

    def filter(self, record: logging.LogRecord) -> bool:
        return bool(getattr(record, "event", None) and getattr(record, "run_id", None))


class _EventJsonFormatter(logging.Formatter):
    """Format structured event records as one-line JSON."""

    _ordered_keys = (
        "participant_id",
        "phase",
        "group",
        "exam_set_id",
        "call_type",
        "item_id",
        "mode",
        "status",
        "latency_ms",
        "prompt_tokens",
        "completion_tokens",
        "items_count",
        "parsed_answers_count",
        "expected_answers_count",
        "error_type",
        "error_message",
        "response_preview",
        "response_length",
    )

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "event": str(getattr(record, "event", "")),
            "run_id": str(getattr(record, "run_id", "")),
        }

        event_fields = getattr(record, "event_fields", {})
        if isinstance(event_fields, dict):
            for key in self._ordered_keys:
                if key in event_fields:
                    payload[key] = _serialise_json_value(event_fields[key])
            for key, value in event_fields.items():
                if key not in payload:
                    payload[key] = _serialise_json_value(value)

        return json.dumps(payload, ensure_ascii=True)


def _format_event_message(event: str, fields: Mapping[str, Any]) -> str:
    """Create a concise human-readable event message.

    Args:
        event: Stable event name.
        fields: Structured event fields.

    Returns:
        One compact message line.
    """
    if not fields:
        return event

    parts: list[str] = []
    for key, value in fields.items():
        text = _serialise_text_value(value=value)
        if len(text) > 180:
            text = f"{text[:177]}..."
        parts.append(f"{key}={text}")
    return f"{event} {' '.join(parts)}"


def _serialise_text_value(value: Any) -> str:
    """Convert an event value to a compact text representation.

    Args:
        value: Event field value.

    Returns:
        Compact string representation.
    """
    json_ready = _serialise_json_value(value)
    try:
        return json.dumps(json_ready, ensure_ascii=True, separators=(",", ":"))
    except TypeError:
        return str(json_ready)


def _serialise_json_value(value: Any) -> Any:
    """Convert an event value into JSON-serialisable data.

    Args:
        value: Event field value.

    Returns:
        JSON-serialisable value.
    """
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _serialise_json_value(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, (list, tuple, set)):
        return [_serialise_json_value(item) for item in value]
    return str(value)


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
    fieldnames = list(rows[0].keys())
    lock = _get_csv_lock(path)

    with lock:
        file_exists = path.is_file()
        if file_exists:
            existing_header = _read_csv_header(path)
            if existing_header:
                if existing_header != fieldnames:
                    msg = (
                        f"CSV header mismatch for {path}. Existing header is "
                        f"{existing_header} but new rows use {fieldnames}. "
                        "Use a fresh output directory or align the schema."
                    )
                    raise ValueError(msg)
            else:
                file_exists = False

        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)


def _get_csv_lock(path: Path) -> threading.Lock:
    """Return a shared lock for one CSV path.

    Args:
        path: Path to the CSV file.

    Returns:
        Lock instance used to serialise writes.
    """
    resolved_path = path.resolve()
    with _csv_locks_guard:
        lock = _csv_locks.get(resolved_path)
        if lock is None:
            lock = threading.Lock()
            _csv_locks[resolved_path] = lock
        return lock


def _read_csv_header(path: Path) -> list[str]:
    """Read the header row from a CSV file if available.

    Args:
        path: Path to the CSV file.

    Returns:
        Header field names. Returns an empty list for empty files.
    """
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        return next(reader, [])


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

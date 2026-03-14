# src/mllm_experiment/run_trial.py
# CLI entrypoint (python run_trial.py --participants 1 --teaching_set_dir ...)
# run_trial.py
from __future__ import annotations
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import logging
import random
import subprocess
from typing import Any
import uuid
from pathlib import Path
from tqdm.auto import tqdm

from .utils import ExperimentEventLogger, setup_logging
from .data_loading import Group
from .openai_client import OpenAIChatClient
from .trial import TrialRunner
from .config import ExperimentConfig

logger = logging.getLogger(__name__)

COMPATIBILITY_MANIFEST_FILENAME = "run_compatibility_manifest.json"
RUN_METADATA_DIRNAME = "run_metadata"
_COMPLETED_STATUS = "completed"
_FAILED_STATUS = "failed"


def _utc_timestamp_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _resolve_git_commit_hash() -> str:
    """Resolve the repository HEAD commit hash.

    Returns:
        Full commit hash string for HEAD.

    Raises:
        RuntimeError: Raised when Git metadata is unavailable.
    """
    repository_root = Path(__file__).resolve().parents[2]
    command = ["git", "rev-parse", "HEAD"]
    try:
        result = subprocess.run(
            command,
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        msg = (
            "Failed to resolve git commit hash with 'git rev-parse HEAD'. "
            "Ensure the repository has Git metadata and retry."
        )
        raise RuntimeError(msg) from exc

    commit_hash = result.stdout.strip()
    if not commit_hash:
        msg = "Git commit hash resolution returned an empty value."
        raise RuntimeError(msg)
    return commit_hash


def _materialise_effective_seed(configured_seed: int | None) -> int:
    """Materialise one seed value used for the full run.

    Args:
        configured_seed: Optional CLI seed.

    Returns:
        Integer seed used for deterministic planning and participant RNG streams.
    """
    if configured_seed is not None:
        return configured_seed
    return random.SystemRandom().randint(0, 2**31 - 1)


def _group_counts_to_serialisable(
    counts: dict[Group, int],
    enabled_groups: list[Group],
) -> dict[str, int]:
    """Convert group-count mappings to serialisable string-key dictionaries.

    Args:
        counts: Mapping from Group enum to integer counts.
        enabled_groups: Ordered list of groups enabled for the run.

    Returns:
        Dictionary keyed by group letters in enabled-group order.
    """
    return {group.value: counts[group] for group in enabled_groups}


def _read_json_dict(path: Path) -> dict[str, Any]:
    """Read a JSON file and validate that it contains an object.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON object.

    Raises:
        ValueError: Raised when parsing fails or payload is not an object.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON content in {path}: {exc}"
        raise ValueError(msg) from exc

    if not isinstance(payload, dict):
        msg = f"Expected JSON object in {path}, found {type(payload).__name__}."
        raise ValueError(msg)
    return payload


def _ensure_output_compatibility_manifest(
    output_root: Path,
    protected_fields: dict[str, str],
    manifest_created_at: str,
) -> Path:
    """Ensure one compatibility manifest exists and matches protected fields.

    Args:
        output_root: Output directory for the experiment run.
        protected_fields: Fields that must remain stable across appended runs.
        manifest_created_at: UTC timestamp used when creating a new manifest.

    Returns:
        Path to the compatibility manifest.

    Raises:
        ValueError: Raised when existing manifest fields mismatch current inputs.
    """
    manifest_path = output_root / COMPATIBILITY_MANIFEST_FILENAME
    if manifest_path.is_file():
        manifest_payload = _read_json_dict(manifest_path)
        missing_fields = [
            field_name for field_name in protected_fields if field_name not in manifest_payload
        ]
        if missing_fields:
            msg = (
                "Compatibility manifest is missing required field(s): "
                f"{', '.join(missing_fields)} in {manifest_path}. "
                "Use a new output directory for this run."
            )
            raise ValueError(msg)

        mismatches: list[str] = []
        for field_name, expected_value in protected_fields.items():
            actual_value = str(manifest_payload.get(field_name, ""))
            if actual_value != expected_value:
                mismatches.append(
                    f"{field_name}: existing={actual_value!r} current={expected_value!r}"
                )
        if mismatches:
            mismatch_text = ", ".join(mismatches)
            msg = (
                "Output directory compatibility check failed. "
                f"Manifest path: {manifest_path}. Mismatch details: {mismatch_text}. "
                "Use a new output directory for this run."
            )
            raise ValueError(msg)
        return manifest_path

    payload = {
        **protected_fields,
        "manifest_created_at": manifest_created_at,
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _load_completed_group_counts(
    output_root: Path,
    enabled_groups: list[Group],
) -> dict[Group, int]:
    """Load baseline completed-participant counts from participants.csv.

    Args:
        output_root: Output directory that may contain participants.csv.
        enabled_groups: Groups enabled for the current run.

    Returns:
        Dictionary with completed counts for enabled groups only.
    """
    counts: dict[Group, int] = {group: 0 for group in enabled_groups}
    participants_path = output_root / "participants.csv"
    if not participants_path.is_file():
        return counts

    with participants_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            logger.warning(
                "[run_trial] participants.csv is empty. Using zero baseline counts path=%s",
                participants_path,
            )
            return counts

        for row_number, row in enumerate(reader, start=2):
            raw_status = str(row.get("status", "")).strip().lower()
            if not raw_status:
                logger.warning(
                    "[run_trial] skipping row with missing status path=%s row=%d",
                    participants_path,
                    row_number,
                )
                continue
            if raw_status not in {_COMPLETED_STATUS, _FAILED_STATUS}:
                logger.warning(
                    (
                        "[run_trial] skipping row with invalid status path=%s row=%d "
                        "status=%s"
                    ),
                    participants_path,
                    row_number,
                    raw_status,
                )
                continue
            if raw_status != _COMPLETED_STATUS:
                continue

            raw_group = str(row.get("group", "")).strip().upper()
            if not raw_group:
                logger.warning(
                    "[run_trial] skipping completed row with missing group path=%s row=%d",
                    participants_path,
                    row_number,
                )
                continue
            try:
                group = Group(raw_group)
            except ValueError:
                logger.warning(
                    (
                        "[run_trial] skipping completed row with invalid group path=%s "
                        "row=%d group=%s"
                    ),
                    participants_path,
                    row_number,
                    raw_group,
                )
                continue
            if group not in counts:
                continue
            counts[group] += 1

    return counts


def _plan_balanced_assignments(
    assignment_rng: random.Random,
    enabled_groups: list[Group],
    baseline_counts: dict[Group, int],
    participants_total: int,
) -> tuple[list[Group], dict[Group, int]]:
    """Plan balanced random group assignments for the current run.

    Args:
        assignment_rng: Random stream dedicated to assignment planning.
        enabled_groups: Ordered list of enabled groups.
        baseline_counts: Completed baseline counts from prior rows.
        participants_total: Number of participants requested for this run.

    Returns:
        Tuple of:
            - ordered list of planned groups by participant index,
            - projected counts after applying this plan.
    """
    projected_counts = dict(baseline_counts)
    planned_groups: list[Group] = []

    for _ in range(participants_total):
        min_count = min(projected_counts[group] for group in enabled_groups)
        least_represented = [
            group for group in enabled_groups if projected_counts[group] == min_count
        ]
        selected_group = assignment_rng.choice(least_represented)
        planned_groups.append(selected_group)
        projected_counts[selected_group] += 1

    return planned_groups, projected_counts


def _write_run_metadata_snapshot(
    output_root: Path,
    run_id: str,
    snapshot_payload: dict[str, Any],
) -> Path:
    """Write one run metadata snapshot as JSON.

    Args:
        output_root: Output directory for the experiment.
        run_id: Stable identifier for this run.
        snapshot_payload: JSON-serialisable metadata payload.

    Returns:
        Path to the written snapshot.
    """
    metadata_root = output_root / RUN_METADATA_DIRNAME
    metadata_root.mkdir(parents=True, exist_ok=True)
    snapshot_path = metadata_root / f"{run_id}.json"
    snapshot_path.write_text(
        json.dumps(snapshot_payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return snapshot_path


def _initialise_rng_streams(effective_seed: int) -> tuple[random.Random, random.Random]:
    """Initialise deterministic RNG streams for planning and participant seeds.

    Args:
        effective_seed: Materialised run seed.

    Returns:
        Tuple with assignment RNG and participant-seed RNG.
    """
    assignment_seed = (effective_seed << 1) ^ 0x9E3779B185EBCA87
    participant_seed = (effective_seed << 1) ^ 0xC2B2AE3D27D4EB4F
    return random.Random(assignment_seed), random.Random(participant_seed)


def parse_conditions_selector(raw_selector: str) -> tuple[str, list[Group]]:
    """Parse and validate the group selection string.

    Args:
        raw_selector: Raw value passed to --conditions.

    Returns:
        Tuple of:
            - canonical requested conditions string (letters A-F),
            - enabled groups list for this run.
    """
    selector = raw_selector.strip().lower()
    if selector == "all":
        selector = "abcdef"

    if not selector:
        msg = "Group selector is empty. Use 'all' or a combination of letters a-f."
        raise ValueError(msg)

    allowed_letters = "abcdef"
    invalid_letters = sorted({char for char in selector if char not in allowed_letters})
    if invalid_letters:
        msg = (
            "Group selector contains invalid characters: "
            f"{', '.join(invalid_letters)}. Use only letters a-f or 'all'."
        )
        raise ValueError(msg)

    selected_letters = [letter for letter in allowed_letters if letter in set(selector)]
    requested_conditions = "".join(selected_letters).upper()
    if not requested_conditions:
        msg = "Group selector does not contain any valid group letters."
        raise ValueError(msg)

    enabled_groups = [Group(letter.upper()) for letter in selected_letters]
    return requested_conditions, enabled_groups


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the experiment runner.

    Returns:
        Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run MLLM teaching trials with ChatGPT 5.1.",
    )
    parser.add_argument(
        "--participants",
        type=int,
        default=1,
        help="Number of MLLM participants (trials) to run.",
    )
    parser.add_argument(
        "--teaching_set_dir",
        type=Path,
        required=True,
        help=(
            "Root directory containing teaching set images "
            "(A, B, C, D folders where E reuses D and F has no teaching images)."
        ),
    )
    parser.add_argument(
        "--exam_sets_dir",
        type=Path,
        required=True,
        help="Root directory containing exam sets (set1, set2).",
    )
    parser.add_argument(
        "--metadata_dir",
        type=Path,
        required=True,
        help="Directory containing teaching_items.csv and exam_items.csv.",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default="all",
        help=(
            "Group selection string. Use 'all' or a non-empty combination "
            "of letters a-f (e.g. abc, bcd, abcdef)."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results"),
        help="Directory where logs and raw responses are written.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-5-mini",
        help="OpenAI model name to use.",
    )
    parser.add_argument(
        "--parallel_participants",
        type=int,
        default=2,
        help="Number of participants to run in parallel.",
    )
    parser.add_argument(
        "--max_requests_per_minute",
        type=int,
        default=500,
        help="Maximum number of OpenAI API requests per minute.",
    )
    parser.add_argument(
        "--max_tokens_per_minute",
        type=int,
        default=500_000,
        help="Maximum number of OpenAI API tokens per minute.",
    )
    parser.add_argument(
        "--max_inflight_api_calls",
        type=int,
        default=None,
        help=(
            "Maximum number of concurrent OpenAI API calls. "
            "Defaults to --parallel_participants."
        ),
    )
    parser.add_argument(
        "--api_timeout_seconds",
        type=float,
        default=600.0,
        help="Timeout per OpenAI API request in seconds.",
    )
    parser.add_argument(
        "--api_retry_attempts",
        type=int,
        default=12,
        help="Number of retries after the initial failed API request.",
    )
    parser.add_argument(
        "--api_retry_base_delay_seconds",
        type=float,
        default=2.0,
        help="Base retry delay in seconds for exponential backoff.",
    )
    parser.add_argument(
        "--api_retry_max_delay_seconds",
        type=float,
        default=120.0,
        help="Maximum retry delay in seconds for exponential backoff.",
    )
    parser.add_argument(
        "--api_retry_jitter_fraction",
        type=float,
        default=0.2,
        help="Fractional jitter applied to retry delay.",
    )
    parser.add_argument(
        "--group_e_retain_retry_attempts",
        type=int,
        default=4,
        help=(
            "Number of retries for one group E teaching example when retain "
            "changes the rule-of-thumb."
        ),
    )
    parser.add_argument(
        "--post_exam_batch_size",
        type=int,
        default=5,
        help="Number of post-exam examples per model call batch.",
    )
    parser.add_argument(
        "--post_exam_missing_repair_attempts",
        type=int,
        default=2,
        help=(
            "Maximum number of follow-up repair attempts per post-exam batch "
            "when answers are missing."
        ),
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help=(
            "Base random seed for reproducibility. If None, "
            "a randomly selected seed will be applied (and printed for reproducibility)"
            "so that different runs may produce different results and "
            "results from multiple runs (with same parameters) can be appended."
        ),
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="ERROR",
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help=(
            "Run the pipeline without real OpenAI calls using a dummy client. "
            "This validates metadata loading, message building and logging."
        ),
    )
    parser.add_argument(
        "--events_log_file",
        type=Path,
        default=None,
        help=(
            "Path to the structured JSONL events log. "
            "Defaults to <output_dir>/experiment_events.jsonl."
        ),
    )
    parser.add_argument(
        "--http_debug",
        action="store_true",
        help="Enable verbose third-party HTTP and OpenAI library logging.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for running one or more trials from the command line."""
    args = parse_args()
    try:
        requested_conditions, enabled_groups = parse_conditions_selector(args.conditions)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    effective_conditions = "".join(group.value for group in enabled_groups)
    effective_seed = _materialise_effective_seed(args.random_seed)
    config = ExperimentConfig(
        log_level=args.log_level,
        teaching_root=args.teaching_set_dir,
        exam_root=args.exam_sets_dir,
        metadata_root=args.metadata_dir,
        output_root=args.output_dir,
        model_name=args.model_name,
        random_seed=effective_seed,
        parallel_participants=args.parallel_participants,
        max_requests_per_minute=args.max_requests_per_minute,
        max_tokens_per_minute=args.max_tokens_per_minute,
        max_inflight_api_calls=args.max_inflight_api_calls,
        api_timeout_seconds=args.api_timeout_seconds,
        api_retry_attempts=args.api_retry_attempts,
        api_retry_base_delay_seconds=args.api_retry_base_delay_seconds,
        api_retry_max_delay_seconds=args.api_retry_max_delay_seconds,
        api_retry_jitter_fraction=args.api_retry_jitter_fraction,
        group_e_retain_retry_attempts=args.group_e_retain_retry_attempts,
        post_exam_batch_size=args.post_exam_batch_size,
        post_exam_missing_repair_attempts=args.post_exam_missing_repair_attempts,
    )
    events_log_file = args.events_log_file or (config.output_root / "experiment_events.jsonl")
    run_id = f"run_{uuid.uuid4().hex[:12]}"

    setup_logging(
        log_level=config.log_level,
        log_file=config.logfile_path,
        events_log_file=events_log_file,
        http_debug=args.http_debug,
    )
    event_logger = ExperimentEventLogger(run_id=run_id)
    run_started_at = _utc_timestamp_iso()
    try:
        git_commit_hash = _resolve_git_commit_hash()
    except RuntimeError as exc:
        logger.error("[run_trial] %s", str(exc))
        raise SystemExit(str(exc)) from exc
    protected_manifest_fields = {
        "model_name": config.model_name,
        "git_commit_hash": git_commit_hash,
        "enabled_conditions": effective_conditions,
    }
    try:
        manifest_path = _ensure_output_compatibility_manifest(
            output_root=config.output_root,
            protected_fields=protected_manifest_fields,
            manifest_created_at=run_started_at,
        )
    except ValueError as exc:
        logger.error("[run_trial] %s", str(exc))
        raise SystemExit(str(exc)) from exc

    baseline_completed_counts = _load_completed_group_counts(
        output_root=config.output_root,
        enabled_groups=enabled_groups,
    )
    assignment_rng, participant_seed_rng = _initialise_rng_streams(effective_seed)
    planned_groups, projected_completed_counts = _plan_balanced_assignments(
        assignment_rng=assignment_rng,
        enabled_groups=enabled_groups,
        baseline_counts=baseline_completed_counts,
        participants_total=args.participants,
    )
    planned_assignment_sequence = [group.value for group in planned_groups]
    participant_seed_sequence = [
        participant_seed_rng.randint(0, 2**31 - 1)
        for _ in range(args.participants)
    ]
    participant_jobs = [
        (index, participant_seed_sequence[index], planned_groups[index])
        for index in range(args.participants)
    ]
    snapshot_payload = {
        "run_timestamp": run_started_at,
        "run_id": run_id,
        "effective_seed": effective_seed,
        "enabled_conditions": effective_conditions,
        "baseline_completed_counts": _group_counts_to_serialisable(
            counts=baseline_completed_counts,
            enabled_groups=enabled_groups,
        ),
        "planned_assignment_sequence": planned_assignment_sequence,
        "projected_completed_counts": _group_counts_to_serialisable(
            counts=projected_completed_counts,
            enabled_groups=enabled_groups,
        ),
        "participant_seed_sequence": participant_seed_sequence,
        "model_name": config.model_name,
        "git_commit_hash": git_commit_hash,
        "group_e_retain_retry_attempts": config.group_e_retain_retry_attempts,
        "post_exam_batch_size": config.post_exam_batch_size,
        "post_exam_missing_repair_attempts": config.post_exam_missing_repair_attempts,
    }
    snapshot_path = _write_run_metadata_snapshot(
        output_root=config.output_root,
        run_id=run_id,
        snapshot_payload=snapshot_payload,
    )
    logger.info(
        (
            "[run_trial] assignment plan prepared effective_seed=%d baseline_completed=%s "
            "planned_sequence=%s"
        ),
        effective_seed,
        _group_counts_to_serialisable(
            counts=baseline_completed_counts,
            enabled_groups=enabled_groups,
        ),
        planned_assignment_sequence,
    )
    if args.random_seed is None:
        logger.info(
            "[run_trial] random_seed was not provided. Generated effective_seed=%d",
            effective_seed,
        )

    if args.verbose:
        logger.debug(
            "[run_trial] config run_id=%s participants=%d dry_run=%s "
            "log_level=%s model_name=%s random_seed=%s output_root=%s "
            "logfile_path=%s events_log_file=%s "
            "conditions_requested=%s conditions_effective=%s git_commit_hash=%s "
            "manifest_path=%s snapshot_path=%s group_e_retain_retry_attempts=%d "
            "post_exam_batch_size=%d post_exam_missing_repair_attempts=%d",
            run_id,
            args.participants,
            args.dry_run,
            config.log_level,
            config.model_name,
            config.random_seed,
            config.output_root,
            config.logfile_path,
            events_log_file,
            requested_conditions,
            effective_conditions,
            git_commit_hash,
            manifest_path,
            snapshot_path,
            config.group_e_retain_retry_attempts,
            config.post_exam_batch_size,
            config.post_exam_missing_repair_attempts,
        )

    client = OpenAIChatClient(
        model=config.model_name,
        use_dummy=args.dry_run,
        event_logger=event_logger,
        max_requests_per_minute=config.max_requests_per_minute,
        max_tokens_per_minute=config.max_tokens_per_minute,
        max_inflight_api_calls=config.max_inflight_api_calls,
        timeout_seconds=config.api_timeout_seconds,
        retry_attempts=config.api_retry_attempts,
        retry_base_delay_seconds=config.api_retry_base_delay_seconds,
        retry_max_delay_seconds=config.api_retry_max_delay_seconds,
        retry_jitter_fraction=config.api_retry_jitter_fraction,
    )
    runner = TrialRunner(
        config=config,
        client=client,
        enabled_groups=enabled_groups,
        verbose=args.verbose,
        event_logger=event_logger,
        show_progress_bars=config.parallel_participants <= 1,
    )

    if args.verbose:
        runner.debug_summary()

    event_logger.log(
        logger=logger,
        event="run.start",
        level=logging.INFO,
        status="started",
        mode="dummy" if args.dry_run else "api",
        run_timestamp=run_started_at,
        participants_total=args.participants,
        model_name=config.model_name,
        effective_seed=effective_seed,
        random_seed=config.random_seed,
        git_commit_hash=git_commit_hash,
        conditions_requested=requested_conditions,
        conditions_effective=effective_conditions,
        compatibility_manifest_file=manifest_path,
        run_metadata_snapshot=snapshot_path,
        log_level=config.log_level,
        output_root=config.output_root,
        log_file=config.logfile_path,
        events_log_file=events_log_file,
        parallel_participants=config.parallel_participants,
        max_requests_per_minute=config.max_requests_per_minute,
        max_tokens_per_minute=config.max_tokens_per_minute,
        max_inflight_api_calls=config.max_inflight_api_calls,
        api_timeout_seconds=config.api_timeout_seconds,
        api_retry_attempts=config.api_retry_attempts,
        api_retry_base_delay_seconds=config.api_retry_base_delay_seconds,
        api_retry_max_delay_seconds=config.api_retry_max_delay_seconds,
        api_retry_jitter_fraction=config.api_retry_jitter_fraction,
        group_e_retain_retry_attempts=config.group_e_retain_retry_attempts,
        post_exam_batch_size=config.post_exam_batch_size,
        post_exam_missing_repair_attempts=config.post_exam_missing_repair_attempts,
    )
    event_logger.log(
        logger=logger,
        event="run.assignment_plan",
        level=logging.INFO,
        status="planned",
        effective_seed=effective_seed,
        participants_total=args.participants,
        baseline_completed_counts=_group_counts_to_serialisable(
            counts=baseline_completed_counts,
            enabled_groups=enabled_groups,
        ),
        projected_completed_counts=_group_counts_to_serialisable(
            counts=projected_completed_counts,
            enabled_groups=enabled_groups,
        ),
        planned_assignment_sequence=planned_assignment_sequence,
        participant_seed_sequence=participant_seed_sequence,
    )

    participants_completed = 0
    participants_failed = 0
    run_status = "completed"

    def run_one_participant(index: int, seed: int, assigned_group: Group) -> bool:
        rng = random.Random(seed)
        if args.verbose:
            logger.debug(
                "[run_trial] participant index=%d seed=%d assigned_group=%s",
                index,
                seed,
                assigned_group.value,
            )
        return runner.run_participant(
            rng=rng,
            index=index,
            assigned_group=assigned_group,
        )

    try:
        if config.parallel_participants <= 1:
            for idx, seed, assigned_group in tqdm(
                participant_jobs,
                total=len(participant_jobs),
                desc="Participants",
                unit="participant",
            ):
                if run_one_participant(index=idx, seed=seed, assigned_group=assigned_group):
                    participants_completed += 1
                else:
                    participants_failed += 1
        else:
            with ThreadPoolExecutor(max_workers=config.parallel_participants) as executor:
                futures = [
                    executor.submit(
                        run_one_participant,
                        index=idx,
                        seed=seed,
                        assigned_group=assigned_group,
                    )
                    for idx, seed, assigned_group in participant_jobs
                ]
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Participants",
                    unit="participant",
                ):
                    if future.result():
                        participants_completed += 1
                    else:
                        participants_failed += 1
    except Exception:
        run_status = "failed"
        raise
    finally:
        event_logger.log(
            logger=logger,
            event="run.complete",
            level=logging.INFO if run_status == "completed" else logging.ERROR,
            status=run_status,
            mode="dummy" if args.dry_run else "api",
            participants_total=args.participants,
            participants_completed=participants_completed,
            participants_failed=participants_failed,
        )


if __name__ == "__main__":
    main()

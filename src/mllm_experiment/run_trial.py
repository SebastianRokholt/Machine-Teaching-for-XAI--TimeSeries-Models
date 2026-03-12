# src/mllm_experiment/run_trial.py
# CLI entrypoint (python run_trial.py --participants 1 --teaching_set_dir ...)
# run_trial.py
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import random
import uuid
from pathlib import Path
from tqdm.auto import tqdm

from .utils import ExperimentEventLogger, setup_logging
from .data_loading import Group
from .openai_client import OpenAIChatClient
from .trial import TrialRunner
from .config import ExperimentConfig

logger = logging.getLogger(__name__)


def parse_conditions_selector(raw_selector: str) -> tuple[str, list[Group]]:
    """Parse and validate the group selection string.

    Args:
        raw_selector: Raw value passed to --conditions.

    Returns:
        Tuple of:
            - canonical requested conditions string (letters A-F),
            - enabled groups list for this stage.
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

    deferred_groups = [letter.upper() for letter in selected_letters if letter in {"e", "f"}]
    if deferred_groups:
        msg = (
            "Group(s) "
            f"{', '.join(deferred_groups)} are not implemented yet in stage 1. "
            "Use combinations of A, B, C and D."
        )
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
        help="Root directory containing teaching sets A, B, C and D.",
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
            "of letters a-f (e.g. abc, bcd, abcdef). "
            "Stage 1 supports A, B, C and D only."
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
        "--verbose",  # deprecated, use --log_level DEBUG instead
        action="store_true",
        help="Deprecated. Enables verbose logging and basic debugging output.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
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
    config = ExperimentConfig(
        log_level=args.log_level,
        teaching_root=args.teaching_set_dir,
        exam_root=args.exam_sets_dir,
        metadata_root=args.metadata_dir,
        output_root=args.output_dir,
        model_name=args.model_name,
        random_seed=args.random_seed,
        parallel_participants=args.parallel_participants,
        max_requests_per_minute=args.max_requests_per_minute,
        max_tokens_per_minute=args.max_tokens_per_minute,
        max_inflight_api_calls=args.max_inflight_api_calls,
        api_timeout_seconds=args.api_timeout_seconds,
        api_retry_attempts=args.api_retry_attempts,
        api_retry_base_delay_seconds=args.api_retry_base_delay_seconds,
        api_retry_max_delay_seconds=args.api_retry_max_delay_seconds,
        api_retry_jitter_fraction=args.api_retry_jitter_fraction,
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

    if args.verbose:
        logger.debug(
            "[run_trial] config run_id=%s participants=%d dry_run=%s "
            "log_level=%s model_name=%s random_seed=%s output_root=%s "
            "logfile_path=%s events_log_file=%s "
            "conditions_requested=%s conditions_effective=%s",
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
        participants_total=args.participants,
        model_name=config.model_name,
        random_seed=config.random_seed,
        conditions_requested=requested_conditions,
        conditions_effective=effective_conditions,
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
    )

    # This sets up the base RNG. If config.random_seed is None, the
    # sequence changes between runs, but each participant seed is logged.
    base_rng = random.Random(config.random_seed)
    participant_jobs = [
        (idx, base_rng.randint(0, 2**31 - 1))
        for idx in range(args.participants)
    ]
    participants_completed = 0
    participants_failed = 0
    run_status = "completed"

    def run_one_participant(index: int, seed: int) -> bool:
        rng = random.Random(seed)
        if args.verbose:
            logger.debug("[run_trial] participant index=%d seed=%d", index, seed)
        return runner.run_participant(rng=rng, index=index)

    try:
        if config.parallel_participants <= 1:
            for idx, seed in tqdm(
                participant_jobs,
                total=len(participant_jobs),
                desc="Participants",
                unit="participant",
            ):
                if run_one_participant(index=idx, seed=seed):
                    participants_completed += 1
                else:
                    participants_failed += 1
        else:
            with ThreadPoolExecutor(max_workers=config.parallel_participants) as executor:
                futures = [
                    executor.submit(run_one_participant, index=idx, seed=seed)
                    for idx, seed in participant_jobs
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

# src/mllm_experiment/run_trial.py
# CLI entrypoint (python run_trial.py --participants 1 --teaching_set_dir ...)
# run_trial.py
from __future__ import annotations
import argparse
import logging
import random
import uuid
from pathlib import Path
from tqdm.auto import tqdm

from .utils import ExperimentEventLogger, setup_logging
from .openai_client import OpenAIChatClient
from .trial import TrialRunner
from .config import ExperimentConfig

logger = logging.getLogger(__name__)

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
        help="Root directory containing teaching sets A, B and C.",
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
    config = ExperimentConfig(
        log_level=args.log_level,
        teaching_root=args.teaching_set_dir,
        exam_root=args.exam_sets_dir,
        metadata_root=args.metadata_dir,
        output_root=args.output_dir,
        model_name=args.model_name,
        random_seed=args.random_seed,
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
            "logfile_path=%s events_log_file=%s",
            run_id,
            args.participants,
            args.dry_run,
            config.log_level,
            config.model_name,
            config.random_seed,
            config.output_root,
            config.logfile_path,
            events_log_file,
        )

    client = OpenAIChatClient(
        model=config.model_name,
        use_dummy=args.dry_run,
        event_logger=event_logger,
    )
    runner = TrialRunner(
        config=config,
        client=client,
        verbose=args.verbose,
        event_logger=event_logger,
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
        log_level=config.log_level,
        output_root=config.output_root,
        log_file=config.logfile_path,
        events_log_file=events_log_file,
    )

    # This sets up the base RNG. If config.random_seed is None, the
    # sequence changes between runs, but each participant seed is logged.
    base_rng = random.Random(config.random_seed)
    participants_completed = 0
    run_status = "completed"

    try:
        for idx in tqdm(range(args.participants), desc="Participants", unit="participant"):
            seed = base_rng.randint(0, 2**31 - 1)
            rng = random.Random(seed)
            if args.verbose:
                logger.debug("[run_trial] participant index=%d seed=%d", idx, seed)
            runner.run_participant(rng=rng, index=idx)
            participants_completed += 1
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
        )


if __name__ == "__main__":
    main()

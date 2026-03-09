# src/mllm_experiment/run_trial.py
# CLI entrypoint (python run_trial.py --participants 1 --teaching_set_dir ...)
# run_trial.py
from __future__ import annotations
import logging
import argparse
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from typing_extensions import Literal
from tqdm.auto import tqdm

from .utils import setup_logging
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

    setup_logging(log_level=config.log_level, log_file=config.logfile_path)

    if args.verbose:
        logger.debug("[run_trial] configuration")
        logger.debug(f"  log_level:     {config.log_level}")
        logger.debug(f"  logfile_path:  {config.logfile_path}")
        logger.debug(f"  teaching_root: {config.teaching_root}")
        logger.debug(f"  exam_root:     {config.exam_root}")
        logger.debug(f"  metadata_root: {config.metadata_root}")
        logger.debug(f"  output_root:   {config.output_root}")
        logger.debug(f"  logfile_path:  {config.logfile_path}")
        logger.debug(f"  model_name:    {config.model_name}")
        logger.debug(f"  random_seed:   {config.random_seed}")
        logger.debug(f"  participants:  {args.participants}")
        logger.debug(f"  dry_run:       {args.dry_run}")

    client = OpenAIChatClient(model=config.model_name, use_dummy=args.dry_run)
    runner = TrialRunner(config=config, client=client, verbose=args.verbose)

    if args.verbose:
        runner.debug_summary()

    # sets up the base rng. If config.random_seed is None, rng will change between trials, 
    # but seed is still printed for reproducability.
    base_rng = random.Random(config.random_seed)

    for idx in tqdm(range(args.participants), desc="Participants", unit="participant"):
        # derive a per-participant seed for reproducibility
        seed = base_rng.randint(0, 2**31 - 1)
        rng = random.Random(seed)
        if args.verbose:
            logger.debug(
                f"[run_trial] running participant index={idx}, "
                f"seed={seed}",
            )
        runner.run_participant(rng=rng, index=idx)


if __name__ == "__main__":
    main()

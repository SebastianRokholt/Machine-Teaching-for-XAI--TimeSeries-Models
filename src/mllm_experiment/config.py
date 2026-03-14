# src/mllm_experiment/config.py
from dataclasses import dataclass
from pathlib import Path
from typing_extensions import Literal


GroupEPostExamContextMode = Literal["rule_only"]
GroupEPostExamTieBreaker = Literal["teaching_majority_label"]


TEACHING_METADATA_FILENAME = "teaching_items.csv"
EXAM_METADATA_FILENAME = "exam_items.csv"


@dataclass(slots=True)
class ExperimentConfig:
    """Store experiment-level configuration for the MLLM teaching study.

    This configuration keeps track of directory locations, model settings
    and basic design constants such as the number of items per phase.
    The script passes an instance of this class through the trial runner
    so that all components share the same configuration.

    Attributes:
        teaching_root: Root directory that contains the teaching set
            subdirectories A, B, C and D. Group E reuses D and
            group F has no teaching images.
        exam_root: Root directory that contains the exam sets (set1, set2).
        metadata_root: Directory that contains the CSV metadata files.
        output_root: Directory where logs and raw responses are written.
        model_name: Name of the OpenAI model to use.
        pre_exam_items: Number of pre-teaching exam items per trial.
        post_exam_items: Number of post-teaching exam items per trial.
        post_exam_batch_size: Number of post-exam items per model call batch.
        post_exam_missing_repair_attempts: Number of follow-up repair attempts
            per post-exam batch when answers are missing.
        teaching_items: Number of teaching items per trial.
        random_seed: Base random seed used for reproducibility. Default is
            None so that repeated runs can append additional participants.
            Set a fixed integer when running one large batch to keep
            the assignment reproducible.
        parallel_participants: Number of participants to run in parallel.
        max_requests_per_minute: Maximum API requests per minute.
        max_tokens_per_minute: Maximum API tokens per minute.
        max_inflight_api_calls: Maximum number of concurrent API calls.
            If None, this resolves to parallel_participants.
        api_timeout_seconds: Timeout per API request in seconds.
        api_retry_attempts: Number of retries after the first failed attempt.
        api_retry_base_delay_seconds: Base delay for exponential backoff.
        api_retry_max_delay_seconds: Maximum delay for exponential backoff.
        api_retry_jitter_fraction: Fractional jitter added to backoff.
        group_e_retain_retry_attempts: Number of retries for one group E
            teaching example when the response uses retain but changes the
            rule-of-thumb.
        group_e_teaching_context_window_examples: Number of latest
            committed teaching examples retained in group E context.
        group_e_post_exam_context_mode: Group E post-exam memory mode.
            The current implementation supports "rule_only".
        group_e_post_exam_rule_max_chars: Maximum number of characters
            kept from the final group E rule-of-thumb. This value must
            be between 1 and 1000.
        group_e_post_exam_tie_breaker: Group E tie-break strategy used
            in post-exam instructions.
    """
    teaching_root: Path
    exam_root: Path
    metadata_root: Path
    output_root: Path
    model_name: str = "gpt-5-mini"
    pre_exam_items: int = 30
    post_exam_items: int = 30
    post_exam_batch_size: int = 5
    post_exam_missing_repair_attempts: int = 2
    teaching_items: int = 60
    random_seed: int | None = None
    parallel_participants: int = 2
    max_requests_per_minute: int = 500
    max_tokens_per_minute: int = 500_000
    max_inflight_api_calls: int | None = None
    api_timeout_seconds: float = 600.0
    api_retry_attempts: int = 12
    api_retry_base_delay_seconds: float = 2.0
    api_retry_max_delay_seconds: float = 120.0
    api_retry_jitter_fraction: float = 0.2
    group_e_retain_retry_attempts: int = 4
    group_e_teaching_context_window_examples: int = 5
    group_e_post_exam_context_mode: GroupEPostExamContextMode = "rule_only"
    group_e_post_exam_rule_max_chars: int = 1000
    group_e_post_exam_tie_breaker: GroupEPostExamTieBreaker = "teaching_majority_label"
    logfile_path: Path | None = None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    def __post_init__(self) -> None:
        """Validate config values and resolve derived defaults."""
        if self.logfile_path is None:
            self.logfile_path = self.output_root / "experiment.log"

        if self.parallel_participants <= 0:
            msg = "parallel_participants must be greater than 0."
            raise ValueError(msg)
        if self.max_requests_per_minute <= 0:
            msg = "max_requests_per_minute must be greater than 0."
            raise ValueError(msg)
        if self.max_tokens_per_minute <= 0:
            msg = "max_tokens_per_minute must be greater than 0."
            raise ValueError(msg)
        if self.max_inflight_api_calls is None:
            self.max_inflight_api_calls = self.parallel_participants
        elif self.max_inflight_api_calls <= 0:
            msg = "max_inflight_api_calls must be greater than 0."
            raise ValueError(msg)
        if self.api_timeout_seconds <= 0:
            msg = "api_timeout_seconds must be greater than 0."
            raise ValueError(msg)
        if self.api_retry_attempts < 0:
            msg = "api_retry_attempts must be 0 or greater."
            raise ValueError(msg)
        if self.api_retry_base_delay_seconds <= 0:
            msg = "api_retry_base_delay_seconds must be greater than 0."
            raise ValueError(msg)
        if self.api_retry_max_delay_seconds <= 0:
            msg = "api_retry_max_delay_seconds must be greater than 0."
            raise ValueError(msg)
        if self.api_retry_max_delay_seconds < self.api_retry_base_delay_seconds:
            msg = "api_retry_max_delay_seconds must be >= api_retry_base_delay_seconds."
            raise ValueError(msg)
        if self.api_retry_jitter_fraction < 0:
            msg = "api_retry_jitter_fraction must be 0 or greater."
            raise ValueError(msg)
        if self.group_e_retain_retry_attempts < 0:
            msg = "group_e_retain_retry_attempts must be 0 or greater."
            raise ValueError(msg)
        if self.group_e_teaching_context_window_examples < 0:
            msg = "group_e_teaching_context_window_examples must be 0 or greater."
            raise ValueError(msg)
        if self.group_e_post_exam_context_mode not in ("rule_only",):
            msg = (
                "group_e_post_exam_context_mode must be one of: "
                "rule_only."
            )
            raise ValueError(msg)
        if self.group_e_post_exam_rule_max_chars <= 0:
            msg = "group_e_post_exam_rule_max_chars must be greater than 0."
            raise ValueError(msg)
        if self.group_e_post_exam_rule_max_chars > 1000:
            msg = "group_e_post_exam_rule_max_chars must be less than or equal to 1000."
            raise ValueError(msg)
        if self.group_e_post_exam_tie_breaker not in ("teaching_majority_label",):
            msg = (
                "group_e_post_exam_tie_breaker must be one of: "
                "teaching_majority_label."
            )
            raise ValueError(msg)
        if self.post_exam_batch_size <= 0:
            msg = "post_exam_batch_size must be greater than 0."
            raise ValueError(msg)
        if self.post_exam_missing_repair_attempts < 0:
            msg = "post_exam_missing_repair_attempts must be 0 or greater."
            raise ValueError(msg)

        # Ensures the output directory exists.
        self.output_root.mkdir(parents=True, exist_ok=True)

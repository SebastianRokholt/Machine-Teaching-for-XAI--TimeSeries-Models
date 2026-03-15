# src/mllm_experiment/trial.py              
# core logic for running a trial for a single participant
# trial.py
from __future__ import annotations
from collections import deque
import json
import logging
import random
import uuid
from dataclasses import dataclass
from typing import Any

from openai import APIStatusError
from tqdm.auto import tqdm

from .data_loading import (
    ExampleItem,
    Group,
    Phase,
    load_exam_metadata,
    load_teaching_metadata,
    resolve_exam_image_path,
    resolve_teaching_image_path,
)
from .utils import (
    ExperimentEventLogger,
    log_exam_results,
    log_participant_summaries,
    log_teaching_results,
)
from .openai_client import (
    ModelCallRetriesExhaustedError,
    OpenAIChatClient,
    ModelResponse,
)
from .prompts import (
    build_checkpoint_retry_correction_text,
    build_group_e_retry_correction_text,
    build_exam_user_content,
    build_exam_missing_answers_repair_content,
    build_rule_lock_retry_correction_text,
    build_rule_lock_user_content,
    build_teaching_user_content,
    exam_system_message,
)
from .config import ExperimentConfig

logger = logging.getLogger(__name__)
TEACHING_CHECKPOINT_INTERVAL = 10
STRICT_PROTOCOL_RETRY_ATTEMPTS = 2
GROUP_E_NORMAL_CUE_PLACEHOLDER = "No explicit normal cue is provided in this response."
GROUP_E_ABNORMAL_CUE_PLACEHOLDER = "No explicit abnormal cue is provided in this response."
CHECKPOINT_NORMAL_CUE_PLACEHOLDER = "No explicit normal cue is provided in this checkpoint response."
CHECKPOINT_ABNORMAL_CUE_PLACEHOLDER = "No explicit abnormal cue is provided in this checkpoint response."


@dataclass(slots=True)
class ParticipantConfig:
    """Store per-participant configuration choices.

    Attributes:
        participant_id: Unique identifier for this participant.
        group: Assigned teaching group (A, B, C, D, E or F).
        exam_set_pre: Exam set used in the pre-teaching phase.
        exam_set_post: Exam set used in the post-teaching phase.
    """

    participant_id: str
    group: Group
    exam_set_pre: str
    exam_set_post: str


class GroupEProtocolError(RuntimeError):
    """Signal that group E protocol constraints are violated."""

    def __init__(self, error_type: str, message: str) -> None:
        self.error_type = error_type
        super().__init__(message)


class TrialRunner:
    """Coordinate the full trial lifecycle for one or more participants.

    The trial runner loads metadata, applies preplanned groups, and assigns exam sets.
    It manages the three-phase protocol (pre-teaching exam, teaching phase,
    post-teaching exam) for each participant. It maintains a separate
    chat history per participant so that the MLLM can reuse context
    within the trial.

    Attributes:
        config: Experiment configuration instance.
        client: OpenAI chat client for model calls.
        verbose: Flag that controls additional debug output.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        client: OpenAIChatClient,
        enabled_groups: list[Group],
        verbose: bool,
        event_logger: ExperimentEventLogger | None = None,
        show_progress_bars: bool = True,
    ) -> None:
        self.config = config
        self.client = client
        self.enabled_groups = tuple(enabled_groups)
        self.verbose = verbose
        self.event_logger = event_logger
        self.show_progress_bars = show_progress_bars

        if not self.enabled_groups:
            msg = "At least one enabled group is required."
            logger.error(msg)
            raise ValueError(msg)

        logger.debug("[trial] loading metadata")
        self.teaching_by_group = load_teaching_metadata(config)
        self.exam_by_set = load_exam_metadata(
            config=config,
            enabled_groups=self.enabled_groups,
        )
        required_teaching_sources: list[tuple[Group, Group]] = []
        for group in self.enabled_groups:
            if group is Group.F:
                continue
            source_group = Group.D if group is Group.E else group
            required_teaching_sources.append((group, source_group))

        missing_teaching = []
        for group, source_group in required_teaching_sources:
            if self.teaching_by_group.get(source_group):
                continue
            if group is source_group:
                missing_teaching.append(group.value)
            else:
                missing_teaching.append(f"{group.value} (source {source_group.value})")
        if missing_teaching:
            msg = (
                "No teaching metadata rows found for selected teaching source(s): "
                f"{', '.join(missing_teaching)}"
            )
            logger.error(msg)
            raise ValueError(msg)

    def run_participant(
        self,
        rng: random.Random,
        index: int,
        assigned_group: Group,
    ) -> bool:
        """Run a full trial for a single participant.

        Args:
            rng: Random number generator instance.
            index: Index of the participant within this run.
            assigned_group: Preassigned teaching group for this participant.

        Returns:
            True when the participant completes all phases. False when
            retries are exhausted, API status errors occur, or protocol
            validation fails safely.
        """
        participant = self._sample_participant_config(
            rng=rng,
            index=index,
            assigned_group=assigned_group,
        )
        self._log_event(
            event="participant.start",
            level=logging.INFO,
            participant_id=participant.participant_id,
            group=participant.group.value,
            exam_set_pre=participant.exam_set_pre,
            exam_set_post=participant.exam_set_post,
            status="started",
        )

        try:
            # initialises chat history with a system message
            messages: list[dict[str, Any]] = [exam_system_message()]

            # phase 1: pre-teaching exam
            pre_rows, pre_accuracy, messages = self._run_exam_phase(
                participant=participant,
                rng=rng,
                phase=Phase.PRE,
                messages=messages,
            )
            log_exam_results(pre_rows, self.config.output_root)

            # phase 2: teaching phase
            # re-initialises chat history with only the system message
            messages = [exam_system_message()]
            teaching_rows, messages, final_rule_of_thumb = self._run_teaching_phase(
                participant=participant,
                rng=rng,
                messages=messages,
            )
            log_teaching_results(teaching_rows, self.config.output_root)

            # phase 3: post-teaching exam
            post_rows, post_accuracy, messages = self._run_exam_phase(
                participant=participant,
                rng=rng,
                phase=Phase.POST,
                messages=messages,
                fixed_rule_of_thumb=final_rule_of_thumb,
                teaching_examples_seen=len(teaching_rows),
            )
            log_exam_results(post_rows, self.config.output_root)

            summary_row = {
                "participant_id": participant.participant_id,
                "group": participant.group.value,
                "exam_set_pre": participant.exam_set_pre,
                "exam_set_post": participant.exam_set_post,
                "accuracy_pre": pre_accuracy,
                "accuracy_post": post_accuracy,
                "delta_accuracy": post_accuracy - pre_accuracy,
                "status": "completed",
                "error_type": "",
                "error_message": "",
            }
            log_participant_summaries([summary_row], self.config.output_root)

            self._log_event(
                event="participant.complete",
                level=logging.INFO,
                participant_id=participant.participant_id,
                group=participant.group.value,
                exam_set_pre=participant.exam_set_pre,
                exam_set_post=participant.exam_set_post,
                accuracy_pre=round(pre_accuracy, 6),
                accuracy_post=round(post_accuracy, 6),
                delta_accuracy=round(post_accuracy - pre_accuracy, 6),
                status="completed",
            )
            return True
        except ModelCallRetriesExhaustedError as exc:
            return self._record_participant_failure(
                participant=participant,
                error_type=exc.last_error_type,
                error_message=str(exc),
                attempts=exc.attempts,
            )
        except GroupEProtocolError as exc:
            return self._record_participant_failure(
                participant=participant,
                error_type=exc.error_type,
                error_message=str(exc),
            )
        except APIStatusError as exc:
            return self._record_participant_failure(
                participant=participant,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

    def _record_participant_failure(
        self,
        participant: ParticipantConfig,
        error_type: str,
        error_message: str,
        attempts: int | None = None,
    ) -> bool:
        """Record one participant failure summary and structured event.

        Args:
            participant: Participant configuration for the failed trial.
            error_type: Stable failure type label.
            error_message: Human-readable failure message.
            attempts: Optional number of attempts made before failing.

        Returns:
            False, which indicates participant failure to the caller.
        """
        summary_row = {
            "participant_id": participant.participant_id,
            "group": participant.group.value,
            "exam_set_pre": participant.exam_set_pre,
            "exam_set_post": participant.exam_set_post,
            "accuracy_pre": None,
            "accuracy_post": None,
            "delta_accuracy": None,
            "status": "failed",
            "error_type": error_type,
            "error_message": error_message,
        }
        log_participant_summaries([summary_row], self.config.output_root)

        event_fields: dict[str, Any] = {
            "participant_id": participant.participant_id,
            "group": participant.group.value,
            "exam_set_pre": participant.exam_set_pre,
            "exam_set_post": participant.exam_set_post,
            "status": "failed",
            "error_type": error_type,
            "error_message": error_message,
        }
        if attempts is not None:
            event_fields["attempts"] = attempts

        self._log_event(
            event="participant.failed",
            level=logging.ERROR,
            **event_fields,
        )
        logger.error(
            "[trial] participant failed participant_id=%s group=%s reason=%s",
            participant.participant_id,
            participant.group.value,
            error_message,
        )
        return False

    def _sample_participant_config(
        self,
        rng: random.Random,
        index: int,
        assigned_group: Group,
    ) -> ParticipantConfig:
        """Sample exam sets for one participant with a preassigned group.

        Args:
            rng: Random number generator instance.
            index: Participant index within the run.
            assigned_group: Preassigned teaching group for this participant.

        Returns:
            Participant configuration object.
        """
        participant_id = f"p_{index:04d}_{uuid.uuid4().hex[:8]}"

        exam_sets = list(self.exam_by_set.keys())
        if len(exam_sets) != 2:
            msg = f"Expected exactly two exam sets, found {len(exam_sets)}."
            logger.exception(msg)
            raise ValueError(msg)

        rng.shuffle(exam_sets)
        exam_set_pre, exam_set_post = exam_sets

        return ParticipantConfig(
            participant_id=participant_id,
            group=assigned_group,
            exam_set_pre=exam_set_pre,
            exam_set_post=exam_set_post,
        )

    def _run_exam_phase(
        self,
        participant: ParticipantConfig,
        rng: random.Random,
        phase: Phase,
        messages: list[dict[str, Any]],
        fixed_rule_of_thumb: str | None = None,
        teaching_examples_seen: int = 0,
    ) -> tuple[list[dict[str, Any]], float, list[dict[str, Any]]]:
        """Run either the pre- or post-teaching exam phase.

        Args:
            participant: ParticipantConfig instance.
            rng: Random number generator instance.
            phase: Phase.PRE or Phase.POST.
            messages: Current chat message history.
            fixed_rule_of_thumb: Optional fixed rule-of-thumb string
                used for groups A-E post-exam prompts.
            teaching_examples_seen: Number of teaching examples
                completed before the post exam. Used for context
                compaction diagnostics.

        Returns:
            Tuple containing:
                - list of exam result rows,
                - accuracy value in [0, 1],
                - updated message history including the model's reply.
        """
        if phase is Phase.PRE:
            exam_set_id = participant.exam_set_pre
            item_limit = self.config.pre_exam_items
        else:
            exam_set_id = participant.exam_set_post
            item_limit = self.config.post_exam_items

        items = list(self.exam_by_set[exam_set_id])

        # optionally shuffle within exam set (if desired)
        rng.shuffle(items)

        # resolve the correct image path for each example given the group
        item_paths: list[tuple[ExampleItem, Any]] = []
        for item in items[:item_limit]:
            image_path = resolve_exam_image_path(
                exam_root=self.config.exam_root,
                item=item,
                group=participant.group,
                phase=phase,
            )
            item_paths.append((item, image_path))

        modality_shown = self._resolve_exam_modality(
            group=participant.group,
            phase=phase,
        )
        effective_fixed_rule_of_thumb = fixed_rule_of_thumb
        context_policy: str | None = None
        context_window_examples: int | None = None
        context_examples_retained: int | None = None
        context_examples_dropped: int | None = None
        rule_chars: int | None = None

        if phase is Phase.POST and participant.group in (
            Group.A,
            Group.B,
            Group.C,
            Group.D,
            Group.E,
        ):
            context_mode = self.config.group_e_post_exam_context_mode
            effective_fixed_rule_of_thumb = self._compact_group_e_rule_of_thumb(
                fixed_rule_of_thumb=fixed_rule_of_thumb,
            )
            rule_chars = len(effective_fixed_rule_of_thumb)
            if context_mode == "rule_only":
                context_policy = "post_exam_rule_only"
                context_window_examples = 0
                context_examples_retained = 0
                context_examples_dropped = max(0, teaching_examples_seen)
            else:
                context_policy = f"post_exam_{context_mode}"

        self._log_event(
            event="phase.exam.start",
            level=logging.INFO,
            participant_id=participant.participant_id,
            phase=phase.value,
            group=participant.group.value,
            exam_set_id=exam_set_id,
            items_count=len(item_paths),
            modality=modality_shown,
            context_policy=context_policy,
            context_window_examples=context_window_examples,
            context_examples_retained=context_examples_retained,
            context_examples_dropped=context_examples_dropped,
            rule_chars=rule_chars,
            status="started",
        )

        allowed_item_ids = {item.item_id for item, _path in item_paths}
        answer_by_item: dict[str, str] = {}
        answer_response_by_item: dict[str, ModelResponse] = {}
        fallback_response_by_item: dict[str, ModelResponse] = {}
        repair_attempts_used_total = 0
        repaired_answers_count = 0
        batch_count = 1

        if phase is Phase.PRE:
            user_content, returned_modality = build_exam_user_content(
                group=participant.group,
                phase=phase,
                exam_items=item_paths,
                fixed_rule_of_thumb=effective_fixed_rule_of_thumb,
                group_e_post_exam_tie_breaker=self.config.group_e_post_exam_tie_breaker,
            )
            if returned_modality != modality_shown:
                logger.warning(
                    "[trial] resolved modality '%s' differs from prompt modality '%s'",
                    modality_shown,
                    returned_modality,
                )
            messages.append({"role": "user", "content": user_content})
            self._log_exam_prompt_summary(
                participant_id=participant.participant_id,
                phase=phase,
                exam_set_id=exam_set_id,
                user_content=user_content,
                item_paths=item_paths,
            )

            model_response: ModelResponse = self.client.call(
                messages,
                verbose=self.verbose,
                call_context={
                    "participant_id": participant.participant_id,
                    "phase": phase.value,
                    "group": participant.group.value,
                    "exam_set_id": exam_set_id,
                    "call_type": "exam",
                    "items_count": len(item_paths),
                    "item_ids": [item.item_id for item, _path in item_paths],
                    "modality": modality_shown,
                    "context_policy": context_policy,
                    "context_window_examples": context_window_examples,
                    "context_examples_retained": context_examples_retained,
                    "context_examples_dropped": context_examples_dropped,
                    "rule_chars": rule_chars,
                },
            )
            messages.append({"role": "assistant", "content": model_response.raw_text})

            event_context = {
                "participant_id": participant.participant_id,
                "phase": phase.value,
                "group": participant.group.value,
                "exam_set_id": exam_set_id,
                "call_type": "exam",
                "modality": modality_shown,
            }
            parsed_answers = self._parse_exam_answers(
                model_response=model_response,
                expected_count=len(item_paths),
                event_context=event_context,
                allowed_item_ids=allowed_item_ids,
            )
            for item_id, guess in parsed_answers.items():
                answer_by_item[item_id] = guess
                answer_response_by_item[item_id] = model_response
            for item, _ in item_paths:
                fallback_response_by_item[item.item_id] = model_response
        else:
            frozen_messages = self._resolve_post_exam_frozen_messages(
                participant=participant,
                prior_messages=messages,
            )
            batches = self._chunk_exam_items(
                item_paths=item_paths,
                chunk_size=self.config.post_exam_batch_size,
            )
            batch_count = len(batches)
            for batch_index, batch_item_paths in enumerate(batches, start=1):
                batch_item_ids = [item.item_id for item, _ in batch_item_paths]
                batch_user_content, returned_modality = build_exam_user_content(
                    group=participant.group,
                    phase=phase,
                    exam_items=batch_item_paths,
                    fixed_rule_of_thumb=effective_fixed_rule_of_thumb,
                    group_e_post_exam_tie_breaker=self.config.group_e_post_exam_tie_breaker,
                )
                if returned_modality != modality_shown:
                    logger.warning(
                        "[trial] resolved modality '%s' differs from prompt modality '%s'",
                        modality_shown,
                        returned_modality,
                    )
                self._log_exam_prompt_summary(
                    participant_id=participant.participant_id,
                    phase=phase,
                    exam_set_id=exam_set_id,
                    user_content=batch_user_content,
                    item_paths=batch_item_paths,
                    batch_index=batch_index,
                    batches_total=batch_count,
                )
                self._log_event(
                    event="phase.exam.batch.start",
                    level=logging.INFO,
                    participant_id=participant.participant_id,
                    phase=phase.value,
                    group=participant.group.value,
                    exam_set_id=exam_set_id,
                    modality=modality_shown,
                    batch_index=batch_index,
                    batches_total=batch_count,
                    items_count=len(batch_item_paths),
                    item_ids=batch_item_ids,
                    context_policy=context_policy,
                    context_window_examples=context_window_examples,
                    context_examples_retained=context_examples_retained,
                    context_examples_dropped=context_examples_dropped,
                    rule_chars=rule_chars,
                    status="started",
                )

                batch_messages = list(frozen_messages)
                batch_messages.append({"role": "user", "content": batch_user_content})
                batch_response = self.client.call(
                    batch_messages,
                    verbose=self.verbose,
                    call_context={
                        "participant_id": participant.participant_id,
                        "phase": phase.value,
                        "group": participant.group.value,
                        "exam_set_id": exam_set_id,
                        "call_type": "exam",
                        "modality": modality_shown,
                        "batch_index": batch_index,
                        "batches_total": batch_count,
                        "items_count": len(batch_item_paths),
                        "item_ids": batch_item_ids,
                        "context_policy": context_policy,
                        "context_window_examples": context_window_examples,
                        "context_examples_retained": context_examples_retained,
                        "context_examples_dropped": context_examples_dropped,
                        "rule_chars": rule_chars,
                    },
                )
                batch_messages.append({"role": "assistant", "content": batch_response.raw_text})
                last_batch_response = batch_response

                batch_event_context = {
                    "participant_id": participant.participant_id,
                    "phase": phase.value,
                    "group": participant.group.value,
                    "exam_set_id": exam_set_id,
                    "call_type": "exam",
                    "modality": modality_shown,
                    "batch_index": batch_index,
                    "batches_total": batch_count,
                }
                batch_answers = self._parse_exam_answers(
                    model_response=batch_response,
                    expected_count=len(batch_item_paths),
                    event_context=batch_event_context,
                    allowed_item_ids=set(batch_item_ids),
                )
                for item_id, guess in batch_answers.items():
                    answer_by_item[item_id] = guess
                    answer_response_by_item[item_id] = batch_response

                missing_item_ids = [
                    item_id for item_id in batch_item_ids if item_id not in batch_answers
                ]
                batch_repair_attempts_used = 0
                batch_recovered_answers = 0

                if missing_item_ids:
                    self._log_event(
                        event="exam.repair.scheduled",
                        level=logging.WARNING,
                        participant_id=participant.participant_id,
                        phase=phase.value,
                        group=participant.group.value,
                        exam_set_id=exam_set_id,
                        call_type="exam_repair",
                        modality=modality_shown,
                        batch_index=batch_index,
                        batches_total=batch_count,
                        missing_item_ids=list(missing_item_ids),
                        missing_count=len(missing_item_ids),
                        retries_configured=self.config.post_exam_missing_repair_attempts,
                        status="scheduled",
                    )

                for repair_attempt in range(
                    1,
                    self.config.post_exam_missing_repair_attempts + 1,
                ):
                    if not missing_item_ids:
                        break

                    batch_repair_attempts_used += 1
                    repair_attempts_used_total += 1
                    repair_user_content = build_exam_missing_answers_repair_content(
                        missing_item_ids=missing_item_ids,
                    )
                    batch_messages.append(
                        {"role": "user", "content": repair_user_content},
                    )
                    self._log_event(
                        event="exam.repair.attempt",
                        level=logging.INFO,
                        participant_id=participant.participant_id,
                        phase=phase.value,
                        group=participant.group.value,
                        exam_set_id=exam_set_id,
                        call_type="exam_repair",
                        modality=modality_shown,
                        batch_index=batch_index,
                        batches_total=batch_count,
                        repair_attempt=repair_attempt,
                        repair_attempts_total=self.config.post_exam_missing_repair_attempts,
                        missing_item_ids=list(missing_item_ids),
                        missing_count=len(missing_item_ids),
                        status="started",
                    )
                    repair_response = self.client.call(
                        batch_messages,
                        verbose=self.verbose,
                        call_context={
                            "participant_id": participant.participant_id,
                            "phase": phase.value,
                            "group": participant.group.value,
                            "exam_set_id": exam_set_id,
                            "call_type": "exam_repair",
                            "modality": modality_shown,
                            "batch_index": batch_index,
                            "batches_total": batch_count,
                            "repair_attempt": repair_attempt,
                            "repair_attempts_total": (
                                self.config.post_exam_missing_repair_attempts
                            ),
                            "items_count": len(missing_item_ids),
                            "item_ids": list(missing_item_ids),
                            "context_policy": context_policy,
                            "context_window_examples": context_window_examples,
                            "context_examples_retained": context_examples_retained,
                            "context_examples_dropped": context_examples_dropped,
                            "rule_chars": rule_chars,
                        },
                    )
                    batch_messages.append(
                        {"role": "assistant", "content": repair_response.raw_text},
                    )
                    last_batch_response = repair_response

                    repair_event_context = {
                        "participant_id": participant.participant_id,
                        "phase": phase.value,
                        "group": participant.group.value,
                        "exam_set_id": exam_set_id,
                        "call_type": "exam_repair",
                        "modality": modality_shown,
                        "batch_index": batch_index,
                        "batches_total": batch_count,
                        "repair_attempt": repair_attempt,
                        "repair_attempts_total": (
                            self.config.post_exam_missing_repair_attempts
                        ),
                    }
                    repair_answers = self._parse_exam_answers(
                        model_response=repair_response,
                        expected_count=len(missing_item_ids),
                        event_context=repair_event_context,
                        allowed_item_ids=set(missing_item_ids),
                    )
                    recovered_item_ids = [
                        item_id
                        for item_id in repair_answers
                        if item_id in missing_item_ids
                    ]
                    for item_id, guess in repair_answers.items():
                        answer_by_item[item_id] = guess
                        answer_response_by_item[item_id] = repair_response
                        batch_answers[item_id] = guess

                    if recovered_item_ids:
                        recovered_count = len(recovered_item_ids)
                        batch_recovered_answers += recovered_count
                        repaired_answers_count += recovered_count

                    missing_item_ids = [
                        item_id for item_id in batch_item_ids if item_id not in batch_answers
                    ]
                    if not missing_item_ids:
                        self._log_event(
                            event="exam.repair.complete",
                            level=logging.INFO,
                            participant_id=participant.participant_id,
                            phase=phase.value,
                            group=participant.group.value,
                            exam_set_id=exam_set_id,
                            call_type="exam_repair",
                            modality=modality_shown,
                            batch_index=batch_index,
                            batches_total=batch_count,
                            attempts_used=batch_repair_attempts_used,
                            recovered_answers_count=batch_recovered_answers,
                            status="completed",
                        )
                        break

                if missing_item_ids:
                    self._log_event(
                        event="exam.repair.exhausted",
                        level=logging.ERROR,
                        participant_id=participant.participant_id,
                        phase=phase.value,
                        group=participant.group.value,
                        exam_set_id=exam_set_id,
                        call_type="exam_repair",
                        modality=modality_shown,
                        batch_index=batch_index,
                        batches_total=batch_count,
                        attempts_used=batch_repair_attempts_used,
                        missing_item_ids=list(missing_item_ids),
                        missing_count=len(missing_item_ids),
                        recovered_answers_count=batch_recovered_answers,
                        status="failed",
                    )

                for item_id in batch_item_ids:
                    fallback_response_by_item[item_id] = last_batch_response

                parsed_answers_count = sum(
                    1 for item_id in batch_item_ids if item_id in batch_answers
                )
                self._log_event(
                    event="phase.exam.batch.complete",
                    level=logging.INFO,
                    participant_id=participant.participant_id,
                    phase=phase.value,
                    group=participant.group.value,
                    exam_set_id=exam_set_id,
                    modality=modality_shown,
                    batch_index=batch_index,
                    batches_total=batch_count,
                    items_count=len(batch_item_ids),
                    parsed_answers_count=parsed_answers_count,
                    expected_answers_count=len(batch_item_ids),
                    repair_attempts_used=batch_repair_attempts_used,
                    recovered_answers_count=batch_recovered_answers,
                    status="completed",
                )

        rows, accuracy = self._build_exam_rows(
            participant=participant,
            phase=phase,
            exam_set_id=exam_set_id,
            modality_shown=modality_shown,
            item_paths=item_paths,
            answer_by_item=answer_by_item,
            answer_response_by_item=answer_response_by_item,
            fallback_response_by_item=fallback_response_by_item,
        )

        self._log_event(
            event="phase.exam.complete",
            level=logging.INFO,
            participant_id=participant.participant_id,
            phase=phase.value,
            group=participant.group.value,
            exam_set_id=exam_set_id,
            items_count=len(item_paths),
            parsed_answers_count=len(answer_by_item),
            expected_answers_count=len(item_paths),
            modality=modality_shown,
            batch_count=batch_count,
            repair_attempts_used=repair_attempts_used_total,
            repaired_answers_count=repaired_answers_count,
            accuracy=round(accuracy, 6),
            status="completed",
        )

        return rows, accuracy, messages

    def _resolve_exam_modality(self, group: Group, phase: Phase) -> str:
        """Resolve exam modality for one group and phase pair."""
        if phase is Phase.PRE:
            return "raw_only"
        if group in (Group.A, Group.B):
            return "overlay"
        if group in (Group.D, Group.E):
            return "simplified_only"
        if group in (Group.C, Group.F):
            return "raw_only"
        msg = f"Unsupported phase/group combination: phase={phase} group={group}"
        raise ValueError(msg)

    def _chunk_exam_items(
        self,
        item_paths: list[tuple[ExampleItem, Any]],
        chunk_size: int,
    ) -> list[list[tuple[ExampleItem, Any]]]:
        """Split exam item paths into fixed-size chunks."""
        chunks: list[list[tuple[ExampleItem, Any]]] = []
        for idx in range(0, len(item_paths), chunk_size):
            chunks.append(item_paths[idx : idx + chunk_size])
        return chunks

    def _resolve_post_exam_frozen_messages(
        self,
        participant: ParticipantConfig,
        prior_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Resolve the baseline messages used for each post-exam batch.

        Args:
            participant: Participant configuration.
            prior_messages: Message history at the end of teaching.

        Returns:
            List of baseline messages reused for each post-exam batch.
        """
        if participant.group in (Group.A, Group.B, Group.C, Group.D, Group.E):
            return [exam_system_message()]

        return list(prior_messages)

    def _compact_group_e_rule_of_thumb(
        self,
        fixed_rule_of_thumb: str | None,
    ) -> str:
        """Compact one fixed rule text to configured length.

        Args:
            fixed_rule_of_thumb: Final teaching rule text.

        Returns:
            Compact rule text.
        """
        rule_text = (fixed_rule_of_thumb or "").strip()
        if not rule_text:
            return ""

        max_chars = self.config.group_e_post_exam_rule_max_chars
        if len(rule_text) <= max_chars:
            return rule_text

        compact_rule = rule_text[:max_chars].rstrip()
        logger.debug(
            "[trial] compacted post-exam rule original_chars=%d compact_chars=%d",
            len(rule_text),
            len(compact_rule),
        )
        return compact_rule

    def _build_group_e_teaching_context_messages(
        self,
        base_messages: list[dict[str, Any]],
        recent_examples: deque[tuple[list[dict[str, Any]], str]],
        current_rule_of_thumb: str | None,
        committed_examples_total: int,
    ) -> tuple[list[dict[str, Any]], int, int]:
        """Build compact group E teaching context for one example.

        Args:
            base_messages: Base participant message history.
            recent_examples: Retained teaching examples with responses.
            current_rule_of_thumb: Current group E rule-of-thumb.
            committed_examples_total: Number of examples committed so far.

        Returns:
            Tuple containing:
                - compact context messages,
                - retained example count,
                - dropped example count.
        """
        system_message = base_messages[0] if base_messages else exam_system_message()
        context_messages: list[dict[str, Any]] = [system_message]

        rule_memory_text = self._build_group_e_teaching_rule_memory_text(
            current_rule_of_thumb=current_rule_of_thumb,
        )
        if rule_memory_text:
            context_messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": rule_memory_text}],
                },
            )

        for user_content, raw_response in recent_examples:
            context_messages.append({"role": "user", "content": user_content})
            context_messages.append({"role": "assistant", "content": raw_response})

        retained_examples = len(recent_examples)
        dropped_examples = max(0, committed_examples_total - retained_examples)
        return context_messages, retained_examples, dropped_examples

    def _build_group_e_teaching_rule_memory_text(
        self,
        current_rule_of_thumb: str | None,
    ) -> str | None:
        """Build one compact rule-memory message for group E teaching.

        Args:
            current_rule_of_thumb: Current group E rule-of-thumb.

        Returns:
            Memory text when a rule exists, otherwise None.
        """
        rule_text = (current_rule_of_thumb or "").strip()
        if not rule_text:
            return None
        return (
            "Memory from earlier teaching examples. "
            f'Current rule-of-thumb: "{rule_text}". '
            "Use this memory while processing this next teaching example."
        )

    def _log_exam_prompt_summary(
        self,
        participant_id: str,
        phase: Phase,
        exam_set_id: str,
        user_content: list[dict[str, Any]],
        item_paths: list[tuple[ExampleItem, Any]],
        batch_index: int | None = None,
        batches_total: int | None = None,
    ) -> None:
        """Log one compact exam prompt summary."""
        text_part_count = sum(
            1
            for content in user_content
            if isinstance(content, dict) and content.get("type") == "text"
        )
        image_part_count = sum(
            1
            for content in user_content
            if isinstance(content, dict) and content.get("type") == "image_url"
        )
        if batch_index is None or batches_total is None:
            logger.debug(
                "[trial] exam prompt summary participant_id=%s phase=%s exam_set_id=%s "
                "items=%d text_parts=%d image_parts=%d item_ids=%s",
                participant_id,
                phase.value,
                exam_set_id,
                len(item_paths),
                text_part_count,
                image_part_count,
                [item.item_id for item, _path in item_paths[:5]],
            )
            return
        logger.debug(
            "[trial] exam prompt summary participant_id=%s phase=%s exam_set_id=%s "
            "batch=%d/%d items=%d text_parts=%d image_parts=%d item_ids=%s",
            participant_id,
            phase.value,
            exam_set_id,
            batch_index,
            batches_total,
            len(item_paths),
            text_part_count,
            image_part_count,
            [item.item_id for item, _path in item_paths[:5]],
        )

    def _build_exam_rows(
        self,
        participant: ParticipantConfig,
        phase: Phase,
        exam_set_id: str,
        modality_shown: str,
        item_paths: list[tuple[ExampleItem, Any]],
        answer_by_item: dict[str, str],
        answer_response_by_item: dict[str, ModelResponse],
        fallback_response_by_item: dict[str, ModelResponse],
    ) -> tuple[list[dict[str, Any]], float]:
        """Build per-item exam rows and compute phase accuracy."""
        rows: list[dict[str, Any]] = []
        correct = 0

        for item, _path in item_paths:
            guess = answer_by_item.get(item.item_id)
            is_correct = guess == item.ai_class
            if is_correct:
                correct += 1

            response = answer_response_by_item.get(item.item_id)
            if response is None:
                response = fallback_response_by_item.get(item.item_id)

            row = {
                "participant_id": participant.participant_id,
                "phase": phase.value,
                "group": participant.group.value,
                "exam_set_id": exam_set_id,
                "item_id": item.item_id,
                "AI_class": item.ai_class,
                "simplicity_k": item.simplicity_k,
                "margin": item.margin,
                "modality_shown": modality_shown,
                "learner_guess": guess,
                "is_correct": int(is_correct),
                "response_time_ms": response.latency_ms if response is not None else None,
                "prompt_tokens": response.prompt_tokens if response is not None else None,
                "completion_tokens": (
                    response.completion_tokens if response is not None else None
                ),
                "raw_response": response.raw_text if response is not None else "",
            }
            rows.append(row)

        accuracy = correct / len(item_paths) if item_paths else 0.0
        return rows, accuracy

    def _parse_exam_answers(
        self,
        model_response: ModelResponse,
        expected_count: int,
        event_context: dict[str, Any],
        allowed_item_ids: set[str] | None = None,
    ) -> dict[str, str]:
        """Parse the model's JSON exam answers into a dictionary.

        Args:
            model_response: ModelResponse returned by the OpenAI client.
            expected_count: Expected number of answers.
            event_context: Context fields used for structured events.
            allowed_item_ids: Optional item ID whitelist for filtering.

        Returns:
            Dictionary mapping item_id to guessed class.
        """
        answers: dict[str, str] = {}

        obj = self._extract_json_payload(model_response=model_response)
        if obj is None:
            self._log_exam_parse_failure(
                event_context=event_context,
                expected_count=expected_count,
                parsed_count=0,
                error_type="json_decode_error",
                error_message="Model response is not valid JSON.",
                raw_text=model_response.raw_text,
            )
            return answers

        if not isinstance(obj, dict):
            self._log_exam_parse_failure(
                event_context=event_context,
                expected_count=expected_count,
                parsed_count=0,
                error_type="invalid_json_shape",
                error_message="Parsed JSON response is not an object.",
                raw_text=model_response.raw_text,
            )
            return answers

        answers_list = obj.get("answers")
        if not isinstance(answers_list, list):
            self._log_exam_parse_failure(
                event_context=event_context,
                expected_count=expected_count,
                parsed_count=0,
                error_type="missing_answers_list",
                error_message="Parsed JSON response has no list in 'answers'.",
                raw_text=model_response.raw_text,
            )
            return answers

        for entry in answers_list:
            if not isinstance(entry, dict):
                continue
            raw_item_id = entry.get("item_id")
            if raw_item_id is None:
                continue
            item_id = str(raw_item_id).strip()
            if not item_id:
                continue
            if allowed_item_ids is not None and item_id not in allowed_item_ids:
                logger.warning(
                    "[trial] ignoring answer for out-of-scope item_id=%s",
                    item_id,
                )
                continue
            guess = str(entry.get("guess")).lower()
            if guess not in ("normal", "abnormal"):
                logger.warning("[trial] invalid guess label '%s' for item_id=%s", guess, item_id)
                continue
            answers[item_id] = guess

        if len(answers) != expected_count and expected_count > 0 and len(answers) == 0:
            self._log_exam_parse_failure(
                event_context=event_context,
                expected_count=expected_count,
                parsed_count=0,
                error_type="no_valid_answers",
                error_message="No valid answer entries were parsed.",
                raw_text=model_response.raw_text,
            )
        elif len(answers) != expected_count:
            logger.warning(
                "[trial] parsed %d/%d answers for participant_id=%s phase=%s",
                len(answers),
                expected_count,
                event_context.get("participant_id"),
                event_context.get("phase"),
            )

        return answers

    def _run_teaching_phase(
        self,
        participant: ParticipantConfig,
        rng: random.Random,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
        """Run the teaching phase with sequential labelled examples.

        Args:
            participant: ParticipantConfig instance.
            rng: Random number generator instance.
            messages: Current chat message history.

        Returns:
            Tuple containing:
                - list of teaching result rows,
                - updated message history including all teaching replies.
                - final locked rule-of-thumb for groups A-E, otherwise None.
        """
        if participant.group is Group.F:
            modality_shown = "none"
            self._log_event(
                event="phase.teaching.start",
                level=logging.INFO,
                participant_id=participant.participant_id,
                phase=Phase.TEACHING.value,
                group=participant.group.value,
                items_count=0,
                modality=modality_shown,
                status="started",
            )
            self._log_event(
                event="phase.teaching.complete",
                level=logging.INFO,
                participant_id=participant.participant_id,
                phase=Phase.TEACHING.value,
                group=participant.group.value,
                items_count=0,
                modality=modality_shown,
                status="completed",
            )
            return [], messages, None

        teaching_source_group = Group.D if participant.group is Group.E else participant.group
        teaching_items = list(self.teaching_by_group[teaching_source_group])
        teaching_items = teaching_items[: self.config.teaching_items]

        # group A keeps curriculum order as specified by metadata
        # groups B and C randomise the order
        if participant.group in (Group.B, Group.C):
            rng.shuffle(teaching_items)

        if participant.group in (Group.A, Group.B):
            modality_shown = "overlay"
        elif participant.group in (Group.D, Group.E):
            modality_shown = "simplified_only"
        else:
            modality_shown = "raw_only"
        self._log_event(
            event="phase.teaching.start",
            level=logging.INFO,
            participant_id=participant.participant_id,
            phase=Phase.TEACHING.value,
            group=participant.group.value,
            items_count=len(teaching_items),
            modality=modality_shown,
            status="started",
        )
        logger.debug(
            "[trial] teaching prompt summary participant_id=%s group=%s items=%d item_ids=%s",
            participant.participant_id,
            participant.group.value,
            len(teaching_items),
            [item.item_id for item in teaching_items[:5]],
        )

        rows: list[dict[str, Any]] = []
        current_rule_of_thumb: str | None = None
        latest_checkpoint_rule_of_thumb: str | None = None
        group_e_recent_examples: deque[tuple[list[dict[str, Any]], str]] | None = None
        group_e_committed_examples_total = 0
        if participant.group is Group.E:
            group_e_recent_examples = deque(
                maxlen=self.config.group_e_teaching_context_window_examples,
            )

        iterator = enumerate(teaching_items, start=1)
        progress = tqdm(
            iterator,
            desc="Teaching examples",
            unit="example",
            total=len(teaching_items),
            disable=not self.show_progress_bars,
        )
        for idx, item in progress:
            image_path = resolve_teaching_image_path(self.config.teaching_root, item)
            is_checkpoint = (
                participant.group in (Group.A, Group.B, Group.C, Group.D)
                and idx % TEACHING_CHECKPOINT_INTERVAL == 0
            )
            teaching_step_type = "checkpoint" if is_checkpoint else "example"
            prior_rule = (
                current_rule_of_thumb
                if participant.group is Group.E
                else latest_checkpoint_rule_of_thumb
            )
            base_user_content = build_teaching_user_content(
                group=participant.group,
                item=item,
                image_path=image_path,
                index=idx,
                total=len(teaching_items),
                teaching_step_type=teaching_step_type,
                current_rule_of_thumb=prior_rule,
            )

            if participant.group is Group.E:
                if group_e_recent_examples is None:
                    msg = "Group E teaching context buffer is not initialised."
                    raise RuntimeError(msg)
                (
                    context_messages,
                    context_examples_retained,
                    context_examples_dropped,
                ) = self._build_group_e_teaching_context_messages(
                    base_messages=messages,
                    recent_examples=group_e_recent_examples,
                    current_rule_of_thumb=current_rule_of_thumb,
                    committed_examples_total=group_e_committed_examples_total,
                )
                (
                    model_response,
                    current_rule_of_thumb,
                    committed_user_content,
                ) = self._run_group_e_teaching_item_with_retries(
                    participant=participant,
                    item=item,
                    index=idx,
                    total=len(teaching_items),
                    context_messages=context_messages,
                    base_user_content=base_user_content,
                    current_rule_of_thumb=current_rule_of_thumb,
                    modality_shown=modality_shown,
                    context_examples_retained=context_examples_retained,
                    context_examples_dropped=context_examples_dropped,
                )
                group_e_recent_examples.append(
                    (committed_user_content, model_response.raw_text),
                )
                group_e_committed_examples_total += 1
            else:
                if is_checkpoint:
                    model_response, checkpoint_rule_of_thumb = (
                        self._run_non_e_checkpoint_item_with_retries(
                            participant=participant,
                            item=item,
                            index=idx,
                            total=len(teaching_items),
                            messages=messages,
                            base_user_content=base_user_content,
                            modality_shown=modality_shown,
                        )
                    )
                    latest_checkpoint_rule_of_thumb = checkpoint_rule_of_thumb
                else:
                    messages.append({"role": "user", "content": base_user_content})
                    call_context = {
                        "participant_id": participant.participant_id,
                        "phase": Phase.TEACHING.value,
                        "group": participant.group.value,
                        "item_id": item.item_id,
                        "call_type": "teaching",
                        "items_count": 1,
                        "modality": modality_shown,
                        "teaching_step_type": teaching_step_type,
                    }
                    try:
                        model_response = self.client.call(
                            messages,
                            verbose=self.verbose,
                            call_context=call_context,
                        )
                    except ModelCallRetriesExhaustedError as exc:
                        if not exc.is_parse_body_bad_request:
                            raise
                        model_response = ModelResponse(
                            raw_text='{"acknowledged": true}',
                            parsed_json={"acknowledged": True},
                            latency_ms=0.0,
                            prompt_tokens=None,
                            completion_tokens=None,
                        )
                        self._log_event(
                            event="teaching.parse_body_fallback",
                            level=logging.WARNING,
                            participant_id=participant.participant_id,
                            phase=Phase.TEACHING.value,
                            group=participant.group.value,
                            item_id=item.item_id,
                            call_type="teaching",
                            modality=modality_shown,
                            teaching_step_type=teaching_step_type,
                            example_index=idx,
                            examples_total=len(teaching_items),
                            status="fallback",
                            fallback_type="synthetic_acknowledged",
                            attempts=exc.attempts,
                            error_type=exc.last_error_type,
                            error_message=str(exc.last_error),
                            error_status_code=exc.last_status_code,
                            error_is_parse_body_bad_request=exc.is_parse_body_bad_request,
                        )
                        logger.warning(
                            "[trial] using synthetic acknowledgement fallback "
                            "participant_id=%s group=%s item_id=%s attempts=%d",
                            participant.participant_id,
                            participant.group.value,
                            item.item_id,
                            exc.attempts,
                        )
                    # append assistant reply to history
                    messages.append({"role": "assistant", "content": model_response.raw_text})

            row = {
                "participant_id": participant.participant_id,
                "phase": Phase.TEACHING.value,
                "group": participant.group.value,
                "item_id": item.item_id,
                "AI_class": item.ai_class,
                "simplicity_k": item.simplicity_k,
                "margin": item.margin,
                "modality_shown": modality_shown,
                "time_spent_ms": model_response.latency_ms,
                "prompt_tokens": model_response.prompt_tokens,
                "completion_tokens": model_response.completion_tokens,
                "raw_response": model_response.raw_text,
                "teaching_step_type": teaching_step_type,
            }
            rows.append(row)

        if participant.group is Group.E and not current_rule_of_thumb:
            msg = "Group E teaching session completes without a valid final rule-of-thumb."
            raise GroupEProtocolError(
                error_type="missing_final_rule",
                message=msg,
            )

        locked_rule_for_post = self._run_final_rule_lock_with_retries(
            participant=participant,
            messages=messages,
            modality_shown=modality_shown,
            teaching_examples_seen=len(teaching_items),
            current_rule_of_thumb=(
                current_rule_of_thumb
                if participant.group is Group.E
                else latest_checkpoint_rule_of_thumb
            ),
            group_e_recent_examples=group_e_recent_examples,
            group_e_committed_examples_total=group_e_committed_examples_total,
        )

        self._log_event(
            event="phase.teaching.complete",
            level=logging.INFO,
            participant_id=participant.participant_id,
            phase=Phase.TEACHING.value,
            group=participant.group.value,
            items_count=len(rows),
            modality=modality_shown,
            rule_chars=len(locked_rule_for_post or ""),
            status="completed",
        )
        return rows, messages, locked_rule_for_post

    def _run_non_e_checkpoint_item_with_retries(
        self,
        participant: ParticipantConfig,
        item: ExampleItem,
        index: int,
        total: int,
        messages: list[dict[str, Any]],
        base_user_content: list[dict[str, Any]],
        modality_shown: str,
    ) -> tuple[ModelResponse, str]:
        """Run one non-E checkpoint item with strict retries."""
        max_retries = STRICT_PROTOCOL_RETRY_ATTEMPTS
        total_attempts = max_retries + 1
        retry_error_message: str | None = None

        for attempt in range(1, total_attempts + 1):
            if attempt == 1:
                user_content = base_user_content
            else:
                retry_instruction = build_checkpoint_retry_correction_text(
                    error_message=retry_error_message
                    or "Checkpoint response violates the required JSON schema.",
                )
                user_content = list(base_user_content)
                user_content.append({"type": "text", "text": retry_instruction})

            messages.append({"role": "user", "content": user_content})
            model_response = self.client.call(
                messages,
                verbose=self.verbose,
                call_context={
                    "participant_id": participant.participant_id,
                    "phase": Phase.TEACHING.value,
                    "group": participant.group.value,
                    "item_id": item.item_id,
                    "call_type": "teaching",
                    "items_count": 1,
                    "modality": modality_shown,
                    "teaching_step_type": "checkpoint",
                    "example_index": index,
                    "examples_total": total,
                    "example_retry_attempt": attempt,
                    "example_retry_attempts_total": total_attempts,
                },
            )
            messages.append({"role": "assistant", "content": model_response.raw_text})
            try:
                checkpoint_rule_of_thumb = self._parse_non_e_checkpoint_response(
                    participant=participant,
                    item=item,
                    index=index,
                    total=total,
                    model_response=model_response,
                    modality_shown=modality_shown,
                    example_attempt=attempt,
                    example_attempts_total=total_attempts,
                )
            except GroupEProtocolError as exc:
                retries_remaining = max_retries - attempt + 1
                if retries_remaining > 0:
                    retry_error_message = str(exc)
                    self._log_event(
                        event="teaching.checkpoint.retry_scheduled",
                        level=logging.WARNING,
                        participant_id=participant.participant_id,
                        phase=Phase.TEACHING.value,
                        group=participant.group.value,
                        item_id=item.item_id,
                        call_type="teaching",
                        modality=modality_shown,
                        teaching_step_type="checkpoint",
                        example_index=index,
                        examples_total=total,
                        example_retry_attempt=attempt,
                        retries_remaining=retries_remaining,
                        status="retrying",
                        error_type=exc.error_type,
                        error_message=str(exc),
                    )
                    continue
                self._log_event(
                    event="teaching.checkpoint.retry_exhausted",
                    level=logging.ERROR,
                    participant_id=participant.participant_id,
                    phase=Phase.TEACHING.value,
                    group=participant.group.value,
                    item_id=item.item_id,
                    call_type="teaching",
                    modality=modality_shown,
                    teaching_step_type="checkpoint",
                    example_index=index,
                    examples_total=total,
                    example_retry_attempt=attempt,
                    retries_configured=max_retries,
                    status="failed",
                    error_type=exc.error_type,
                    error_message=str(exc),
                )
                raise

            return model_response, checkpoint_rule_of_thumb

        msg = "Checkpoint retry loop reaches an invalid terminal state."
        raise RuntimeError(msg)

    def _run_final_rule_lock_with_retries(
        self,
        participant: ParticipantConfig,
        messages: list[dict[str, Any]],
        modality_shown: str,
        teaching_examples_seen: int,
        current_rule_of_thumb: str | None,
        group_e_recent_examples: deque[tuple[list[dict[str, Any]], str]] | None,
        group_e_committed_examples_total: int,
    ) -> str:
        """Run the final locked-rule snapshot with strict retries."""
        max_retries = STRICT_PROTOCOL_RETRY_ATTEMPTS
        total_attempts = max_retries + 1
        retry_error_message: str | None = None
        context_messages: list[dict[str, Any]]
        context_examples_retained: int | None = None
        context_examples_dropped: int | None = None

        if participant.group is Group.E:
            if group_e_recent_examples is None:
                msg = "Group E final rule lock requires a context buffer."
                raise RuntimeError(msg)
            (
                context_messages,
                context_examples_retained,
                context_examples_dropped,
            ) = self._build_group_e_teaching_context_messages(
                base_messages=messages,
                recent_examples=group_e_recent_examples,
                current_rule_of_thumb=current_rule_of_thumb,
                committed_examples_total=group_e_committed_examples_total,
            )
        else:
            context_messages = list(messages)

        self._log_event(
            event="phase.teaching.rule_lock.start",
            level=logging.INFO,
            participant_id=participant.participant_id,
            phase=Phase.TEACHING.value,
            group=participant.group.value,
            call_type="rule_lock",
            modality=modality_shown,
            items_count=0,
            context_examples_retained=context_examples_retained,
            context_examples_dropped=context_examples_dropped,
            status="started",
        )

        for attempt in range(1, total_attempts + 1):
            user_content = build_rule_lock_user_content(
                group=participant.group,
                teaching_examples_seen=teaching_examples_seen,
                current_rule_of_thumb=current_rule_of_thumb,
            )
            if attempt > 1:
                retry_instruction = build_rule_lock_retry_correction_text(
                    error_message=retry_error_message
                    or "Rule-lock response violates the required JSON schema.",
                )
                user_content.append({"type": "text", "text": retry_instruction})

            call_messages = list(context_messages)
            call_messages.append({"role": "user", "content": user_content})
            model_response = self.client.call(
                call_messages,
                verbose=self.verbose,
                call_context={
                    "participant_id": participant.participant_id,
                    "phase": Phase.TEACHING.value,
                    "group": participant.group.value,
                    "call_type": "rule_lock",
                    "items_count": 0,
                    "modality": modality_shown,
                    "example_retry_attempt": attempt,
                    "example_retry_attempts_total": total_attempts,
                    "context_examples_retained": context_examples_retained,
                    "context_examples_dropped": context_examples_dropped,
                },
            )
            try:
                locked_rule_payload = self._parse_locked_rule_snapshot(
                    participant=participant,
                    model_response=model_response,
                    modality_shown=modality_shown,
                    example_attempt=attempt,
                    example_attempts_total=total_attempts,
                )
            except GroupEProtocolError as exc:
                retries_remaining = max_retries - attempt + 1
                if retries_remaining > 0:
                    retry_error_message = str(exc)
                    self._log_event(
                        event="teaching.rule_lock.retry_scheduled",
                        level=logging.WARNING,
                        participant_id=participant.participant_id,
                        phase=Phase.TEACHING.value,
                        group=participant.group.value,
                        call_type="rule_lock",
                        modality=modality_shown,
                        example_retry_attempt=attempt,
                        retries_remaining=retries_remaining,
                        status="retrying",
                        error_type=exc.error_type,
                        error_message=str(exc),
                    )
                    continue
                self._log_event(
                    event="teaching.rule_lock.retry_exhausted",
                    level=logging.ERROR,
                    participant_id=participant.participant_id,
                    phase=Phase.TEACHING.value,
                    group=participant.group.value,
                    call_type="rule_lock",
                    modality=modality_shown,
                    example_retry_attempt=attempt,
                    retries_configured=max_retries,
                    status="failed",
                    error_type=exc.error_type,
                    error_message=str(exc),
                )
                raise

            locked_rule_text = self._format_locked_rule_for_exam(locked_rule_payload)
            compact_locked_rule_text = self._compact_group_e_rule_of_thumb(locked_rule_text)
            self._log_event(
                event="phase.teaching.rule_lock.complete",
                level=logging.INFO,
                participant_id=participant.participant_id,
                phase=Phase.TEACHING.value,
                group=participant.group.value,
                call_type="rule_lock",
                modality=modality_shown,
                normal_cues_count=len(locked_rule_payload["normal_cues"]),
                abnormal_cues_count=len(locked_rule_payload["abnormal_cues"]),
                exceptions_count=len(locked_rule_payload["exceptions"]),
                confidence=locked_rule_payload["confidence"],
                rule_chars=len(compact_locked_rule_text),
                status="completed",
            )
            return compact_locked_rule_text

        msg = "Final rule-lock retry loop reaches an invalid terminal state."
        raise RuntimeError(msg)

    def _run_group_e_teaching_item_with_retries(
        self,
        participant: ParticipantConfig,
        item: ExampleItem,
        index: int,
        total: int,
        context_messages: list[dict[str, Any]],
        base_user_content: list[dict[str, Any]],
        current_rule_of_thumb: str | None,
        modality_shown: str,
        context_examples_retained: int,
        context_examples_dropped: int,
    ) -> tuple[ModelResponse, str, list[dict[str, Any]]]:
        """Run one group E teaching item with retain-action retries.

        Args:
            participant: Participant configuration.
            item: Current teaching item.
            index: Example index in the teaching sequence.
            total: Number of teaching examples.
            context_messages: Compact context history for this example.
            base_user_content: Standard user content for this example.
            current_rule_of_thumb: Current group E rule-of-thumb before this item.
            modality_shown: Modality label used for logging.
            context_examples_retained: Number of retained teaching examples.
            context_examples_dropped: Number of dropped teaching examples.

        Returns:
            Tuple with model response, updated rule-of-thumb and committed user content.

        Raises:
            GroupEProtocolError: If a non-retryable protocol error occurs or retries
                for retain-rule mismatch are exhausted.
        """
        max_retries = STRICT_PROTOCOL_RETRY_ATTEMPTS
        total_attempts = max_retries + 1
        retry_error_message: str | None = None

        for attempt in range(1, total_attempts + 1):
            if attempt == 1:
                user_content = base_user_content
            else:
                retry_instruction = build_group_e_retry_correction_text(
                    current_rule_of_thumb=(current_rule_of_thumb or "").strip(),
                    error_message=retry_error_message
                    or "Group E 'retain' action must keep rule_of_thumb unchanged.",
                )
                user_content = list(base_user_content)
                user_content.append({"type": "text", "text": retry_instruction})

            rule_chars = len((current_rule_of_thumb or "").strip())
            call_messages = list(context_messages)
            call_messages.append({"role": "user", "content": user_content})
            model_response = self.client.call(
                call_messages,
                verbose=self.verbose,
                call_context={
                    "participant_id": participant.participant_id,
                    "phase": Phase.TEACHING.value,
                    "group": participant.group.value,
                    "item_id": item.item_id,
                    "call_type": "teaching",
                    "items_count": 1,
                    "modality": modality_shown,
                    "teaching_step_type": "example",
                    "example_index": index,
                    "examples_total": total,
                    "example_retry_attempt": attempt,
                    "example_retry_attempts_total": total_attempts,
                    "context_policy": "group_e_teaching_window",
                    "context_window_examples": (
                        self.config.group_e_teaching_context_window_examples
                    ),
                    "context_examples_retained": context_examples_retained,
                    "context_examples_dropped": context_examples_dropped,
                    "rule_chars": rule_chars,
                },
            )
            try:
                updated_rule_of_thumb = self._parse_group_e_rule_update(
                    participant=participant,
                    item=item,
                    index=index,
                    total=total,
                    current_rule_of_thumb=current_rule_of_thumb,
                    model_response=model_response,
                    modality_shown=modality_shown,
                    example_attempt=attempt,
                    example_attempts_total=total_attempts,
                )
            except GroupEProtocolError as exc:
                retries_remaining = max_retries - attempt + 1
                if retries_remaining > 0:
                    retry_error_message = str(exc)
                    self._log_event(
                        event="teaching.retry_scheduled",
                        level=logging.WARNING,
                        participant_id=participant.participant_id,
                        phase=Phase.TEACHING.value,
                        group=participant.group.value,
                        item_id=item.item_id,
                        call_type="teaching",
                        modality=modality_shown,
                        example_index=index,
                        examples_total=total,
                        example_retry_attempt=attempt,
                        retries_remaining=retries_remaining,
                        status="retrying",
                        error_type=exc.error_type,
                        error_message=str(exc),
                    )
                    continue
                self._log_event(
                    event="teaching.retry_exhausted",
                    level=logging.ERROR,
                    participant_id=participant.participant_id,
                    phase=Phase.TEACHING.value,
                    group=participant.group.value,
                    item_id=item.item_id,
                    call_type="teaching",
                    modality=modality_shown,
                    example_index=index,
                    examples_total=total,
                    example_retry_attempt=attempt,
                    retries_configured=max_retries,
                    status="failed",
                    error_type=exc.error_type,
                    error_message=str(exc),
                )
                raise

            return model_response, updated_rule_of_thumb, user_content

        msg = "Group E teaching retry loop reaches an invalid terminal state."
        raise RuntimeError(msg)

    def _parse_group_e_rule_update(
        self,
        participant: ParticipantConfig,
        item: ExampleItem,
        index: int,
        total: int,
        current_rule_of_thumb: str | None,
        model_response: ModelResponse,
        modality_shown: str,
        example_attempt: int = 1,
        example_attempts_total: int = 1,
    ) -> str:
        """Parse and validate one group E rule-update response."""
        event_context = {
            "participant_id": participant.participant_id,
            "phase": Phase.TEACHING.value,
            "group": participant.group.value,
            "item_id": item.item_id,
            "call_type": "teaching",
            "modality": modality_shown,
            "example_index": index,
            "examples_total": total,
            "example_retry_attempt": example_attempt,
            "example_retry_attempts_total": example_attempts_total,
        }

        obj = self._extract_json_payload(model_response=model_response)
        if not isinstance(obj, dict):
            msg = (
                "Group E teaching response is not a valid JSON object."
            )
            self._log_teaching_parse_failure(
                event_context=event_context,
                error_type="invalid_json",
                error_message=msg,
                raw_text=model_response.raw_text,
            )
            raise GroupEProtocolError("invalid_json", msg)

        description_sentence = str(obj.get("description_sentence", "")).strip()
        rule_action = str(obj.get("rule_action", "")).strip().lower()
        rule_of_thumb = str(obj.get("rule_of_thumb", "")).strip()
        normal_cues, normal_cues_autofill = self._parse_text_list_with_autofill(
            obj=obj,
            key="normal_cues",
            fallback_value=GROUP_E_NORMAL_CUE_PLACEHOLDER,
        )
        abnormal_cues, abnormal_cues_autofill = self._parse_text_list_with_autofill(
            obj=obj,
            key="abnormal_cues",
            fallback_value=GROUP_E_ABNORMAL_CUE_PLACEHOLDER,
        )
        cue_autofill_reasons: dict[str, str] = {}
        if normal_cues_autofill is not None:
            cue_autofill_reasons["normal_cues"] = normal_cues_autofill
        if abnormal_cues_autofill is not None:
            cue_autofill_reasons["abnormal_cues"] = abnormal_cues_autofill
        self._log_protocol_autofill(
            event_context=event_context,
            autofill_reasons_by_key=cue_autofill_reasons,
            raw_text=model_response.raw_text,
        )

        if not description_sentence:
            msg = "Group E response is missing a non-empty description_sentence."
            self._log_teaching_parse_failure(
                event_context=event_context,
                error_type="missing_description_sentence",
                error_message=msg,
                raw_text=model_response.raw_text,
            )
            raise GroupEProtocolError("missing_description_sentence", msg)
        if not self._is_single_sentence(text=description_sentence):
            msg = "Group E description_sentence must contain exactly one sentence."
            self._log_teaching_parse_failure(
                event_context=event_context,
                error_type="invalid_description_sentence",
                error_message=msg,
                raw_text=model_response.raw_text,
            )
            raise GroupEProtocolError("invalid_description_sentence", msg)

        allowed_actions = {"write", "retain", "rephrase"}
        if rule_action not in allowed_actions:
            msg = (
                "Group E rule_action must be one of write, retain or rephrase."
            )
            self._log_teaching_parse_failure(
                event_context=event_context,
                error_type="invalid_rule_action",
                error_message=msg,
                raw_text=model_response.raw_text,
            )
            raise GroupEProtocolError("invalid_rule_action", msg)

        if not rule_of_thumb:
            prior_rule = (current_rule_of_thumb or "").strip()
            if rule_action == "retain" and prior_rule:
                rule_of_thumb = prior_rule
                rule_autofill_source = "retain_prior_rule"
            else:
                rule_of_thumb = self._synthesise_rule_of_thumb_from_cues(
                    normal_cues=normal_cues,
                    abnormal_cues=abnormal_cues,
                )
                rule_autofill_source = "synthesised_from_cues"
            self._log_event(
                event="teaching.rule_autofill",
                level=logging.WARNING,
                **event_context,
                status="autofilled",
                autofill_source=rule_autofill_source,
                rule_action=rule_action,
                rule_of_thumb=rule_of_thumb,
            )
        if self._contains_disallowed_default_fallback(rule_of_thumb):
            msg = (
                "Group E rule_of_thumb must not use default fallback wording such as "
                "'abnormal otherwise' or 'otherwise abnormal'."
            )
            self._log_teaching_parse_failure(
                event_context=event_context,
                error_type="disallowed_default_fallback",
                error_message=msg,
                raw_text=model_response.raw_text,
            )
            raise GroupEProtocolError("disallowed_default_fallback", msg)

        if index == 1 and rule_action != "write":
            msg = (
                "Group E first teaching example must use rule_action 'write'."
            )
            self._log_teaching_parse_failure(
                event_context=event_context,
                error_type="invalid_first_rule_action",
                error_message=msg,
                raw_text=model_response.raw_text,
            )
            raise GroupEProtocolError("invalid_first_rule_action", msg)

        if rule_action == "retain":
            prior_rule = (current_rule_of_thumb or "").strip()
            if not prior_rule:
                msg = (
                    "Group E uses 'retain' before a prior rule-of-thumb exists."
                )
                self._log_teaching_parse_failure(
                    event_context=event_context,
                    error_type="retain_without_prior_rule",
                    error_message=msg,
                    raw_text=model_response.raw_text,
                )
                raise GroupEProtocolError("retain_without_prior_rule", msg)
            if rule_of_thumb != prior_rule:
                msg = (
                    "Group E 'retain' action must keep rule_of_thumb unchanged."
                )
                self._log_teaching_parse_failure(
                    event_context=event_context,
                    error_type="retain_rule_changed",
                    error_message=msg,
                    raw_text=model_response.raw_text,
                )
                raise GroupEProtocolError("retain_rule_changed", msg)

        self._log_event(
            event="phase.teaching.rule_update",
            level=logging.INFO,
            **event_context,
            status="completed",
            rule_action=rule_action,
            rule_of_thumb=rule_of_thumb,
            description_sentence=description_sentence,
            normal_cues_count=len(normal_cues),
            abnormal_cues_count=len(abnormal_cues),
        )
        return rule_of_thumb

    def _parse_non_e_checkpoint_response(
        self,
        participant: ParticipantConfig,
        item: ExampleItem,
        index: int,
        total: int,
        model_response: ModelResponse,
        modality_shown: str,
        example_attempt: int = 1,
        example_attempts_total: int = 1,
    ) -> str:
        """Parse and validate one non-E checkpoint response."""
        event_context = {
            "participant_id": participant.participant_id,
            "phase": Phase.TEACHING.value,
            "group": participant.group.value,
            "item_id": item.item_id,
            "call_type": "teaching",
            "modality": modality_shown,
            "teaching_step_type": "checkpoint",
            "example_index": index,
            "examples_total": total,
            "example_retry_attempt": example_attempt,
            "example_retry_attempts_total": example_attempts_total,
        }

        obj = self._extract_json_payload(model_response=model_response)
        if not isinstance(obj, dict):
            msg = "Checkpoint response is not a valid JSON object."
            self._log_teaching_parse_failure(
                event_context=event_context,
                error_type="invalid_checkpoint_json",
                error_message=msg,
                raw_text=model_response.raw_text,
            )
            raise GroupEProtocolError("invalid_checkpoint_json", msg)

        acknowledged = obj.get("acknowledged")
        if acknowledged is not True:
            msg = "Checkpoint response requires acknowledged to be true."
            self._log_teaching_parse_failure(
                event_context=event_context,
                error_type="invalid_checkpoint_acknowledged",
                error_message=msg,
                raw_text=model_response.raw_text,
            )
            raise GroupEProtocolError("invalid_checkpoint_acknowledged", msg)

        normal_cues, normal_cues_autofill = self._parse_text_list_with_autofill(
            obj=obj,
            key="normal_cues",
            fallback_value=CHECKPOINT_NORMAL_CUE_PLACEHOLDER,
        )
        abnormal_cues, abnormal_cues_autofill = self._parse_text_list_with_autofill(
            obj=obj,
            key="abnormal_cues",
            fallback_value=CHECKPOINT_ABNORMAL_CUE_PLACEHOLDER,
        )
        cue_autofill_reasons: dict[str, str] = {}
        if normal_cues_autofill is not None:
            cue_autofill_reasons["normal_cues"] = normal_cues_autofill
        if abnormal_cues_autofill is not None:
            cue_autofill_reasons["abnormal_cues"] = abnormal_cues_autofill
        self._log_protocol_autofill(
            event_context=event_context,
            autofill_reasons_by_key=cue_autofill_reasons,
            raw_text=model_response.raw_text,
        )
        exceptions = self._parse_text_list(
            obj=obj,
            key="exceptions",
            event_context=event_context,
            raw_text=model_response.raw_text,
            error_type="invalid_checkpoint_exceptions",
            error_message="Checkpoint response requires exceptions as a text list.",
        )
        confidence = self._parse_confidence_value(
            obj=obj,
            key="confidence",
            event_context=event_context,
            raw_text=model_response.raw_text,
            error_type="invalid_checkpoint_confidence",
            error_message="Checkpoint response requires confidence between 0 and 1.",
        )
        rule_of_thumb = str(obj.get("rule_of_thumb", "")).strip()
        if not rule_of_thumb:
            msg = "Checkpoint response is missing a non-empty rule_of_thumb."
            self._log_teaching_parse_failure(
                event_context=event_context,
                error_type="missing_checkpoint_rule_of_thumb",
                error_message=msg,
                raw_text=model_response.raw_text,
            )
            raise GroupEProtocolError("missing_checkpoint_rule_of_thumb", msg)

        self._log_event(
            event="phase.teaching.checkpoint",
            level=logging.INFO,
            **event_context,
            status="completed",
            normal_cues_count=len(normal_cues),
            abnormal_cues_count=len(abnormal_cues),
            exceptions_count=len(exceptions),
            confidence=confidence,
            rule_of_thumb=rule_of_thumb,
        )
        return rule_of_thumb

    def _parse_locked_rule_snapshot(
        self,
        participant: ParticipantConfig,
        model_response: ModelResponse,
        modality_shown: str,
        example_attempt: int = 1,
        example_attempts_total: int = 1,
    ) -> dict[str, Any]:
        """Parse and validate one final locked-rule snapshot response."""
        event_context = {
            "participant_id": participant.participant_id,
            "phase": Phase.TEACHING.value,
            "group": participant.group.value,
            "call_type": "rule_lock",
            "modality": modality_shown,
            "example_retry_attempt": example_attempt,
            "example_retry_attempts_total": example_attempts_total,
        }
        obj = self._extract_json_payload(model_response=model_response)
        if not isinstance(obj, dict):
            msg = "Final rule-lock response is not a valid JSON object."
            self._log_teaching_parse_failure(
                event_context=event_context,
                error_type="invalid_rule_lock_json",
                error_message=msg,
                raw_text=model_response.raw_text,
            )
            raise GroupEProtocolError("invalid_rule_lock_json", msg)

        locked_rule_of_thumb = str(obj.get("locked_rule_of_thumb", "")).strip()
        if not locked_rule_of_thumb:
            msg = "Final rule-lock response is missing locked_rule_of_thumb."
            self._log_teaching_parse_failure(
                event_context=event_context,
                error_type="missing_locked_rule_of_thumb",
                error_message=msg,
                raw_text=model_response.raw_text,
            )
            raise GroupEProtocolError("missing_locked_rule_of_thumb", msg)
        if participant.group is Group.E and self._contains_disallowed_default_fallback(
            locked_rule_of_thumb,
        ):
            msg = (
                "Final group E locked rule must not use default fallback wording such as "
                "'abnormal otherwise' or 'otherwise abnormal'."
            )
            self._log_teaching_parse_failure(
                event_context=event_context,
                error_type="disallowed_locked_rule_default_fallback",
                error_message=msg,
                raw_text=model_response.raw_text,
            )
            raise GroupEProtocolError("disallowed_locked_rule_default_fallback", msg)

        normal_cues = self._parse_non_empty_text_list(
            obj=obj,
            key="normal_cues",
            event_context=event_context,
            raw_text=model_response.raw_text,
            error_type="missing_locked_normal_cues",
            error_message="Final rule-lock response is missing non-empty normal_cues.",
        )
        abnormal_cues = self._parse_non_empty_text_list(
            obj=obj,
            key="abnormal_cues",
            event_context=event_context,
            raw_text=model_response.raw_text,
            error_type="missing_locked_abnormal_cues",
            error_message="Final rule-lock response is missing non-empty abnormal_cues.",
        )
        exceptions = self._parse_text_list(
            obj=obj,
            key="exceptions",
            event_context=event_context,
            raw_text=model_response.raw_text,
            error_type="invalid_locked_exceptions",
            error_message="Final rule-lock response requires exceptions as a text list.",
        )
        confidence = self._parse_confidence_value(
            obj=obj,
            key="confidence",
            event_context=event_context,
            raw_text=model_response.raw_text,
            error_type="invalid_locked_confidence",
            error_message="Final rule-lock response requires confidence between 0 and 1.",
        )
        return {
            "locked_rule_of_thumb": locked_rule_of_thumb,
            "normal_cues": normal_cues,
            "abnormal_cues": abnormal_cues,
            "exceptions": exceptions,
            "confidence": confidence,
        }

    def _format_locked_rule_for_exam(self, payload: dict[str, Any]) -> str:
        """Format one locked-rule payload into compact exam carryover text."""
        locked_rule = str(payload.get("locked_rule_of_thumb", "")).strip()
        normal_cues = payload.get("normal_cues", [])
        abnormal_cues = payload.get("abnormal_cues", [])
        exceptions = payload.get("exceptions", [])
        confidence = payload.get("confidence")
        return (
            f"locked_rule_of_thumb={locked_rule}; "
            f"normal_cues={normal_cues}; "
            f"abnormal_cues={abnormal_cues}; "
            f"exceptions={exceptions}; "
            f"confidence={confidence}"
        )

    def _synthesise_rule_of_thumb_from_cues(
        self,
        normal_cues: list[str],
        abnormal_cues: list[str],
    ) -> str:
        """Build one fallback rule-of-thumb from parsed cue lists."""
        normal_text = ", ".join(normal_cues)
        abnormal_text = ", ".join(abnormal_cues)
        return f"Normal cues: {normal_text}. Abnormal cues: {abnormal_text}."

    def _log_protocol_autofill(
        self,
        event_context: dict[str, Any],
        autofill_reasons_by_key: dict[str, str],
        raw_text: str,
    ) -> None:
        """Record one structured autofill event for recoverable protocol issues."""
        if not autofill_reasons_by_key:
            return
        self._log_event(
            event="teaching.protocol_autofill",
            level=logging.WARNING,
            **event_context,
            status="autofilled",
            autofilled_keys=sorted(autofill_reasons_by_key.keys()),
            autofill_reasons=autofill_reasons_by_key,
            response_preview=self._response_preview(raw_text=raw_text),
            response_length=len(raw_text),
        )

    def _parse_text_list_with_autofill(
        self,
        obj: dict[str, Any],
        key: str,
        fallback_value: str,
    ) -> tuple[list[str], str | None]:
        """Parse one text list and autofill when the list is missing or empty."""
        raw_values = obj.get(key)
        if not isinstance(raw_values, list):
            return [fallback_value], "missing_or_invalid_list"
        values: list[str] = []
        for raw_value in raw_values:
            text = str(raw_value).strip()
            if text:
                values.append(text)
        if values:
            return values, None
        return [fallback_value], "empty_list"

    def _parse_non_empty_text_list(
        self,
        obj: dict[str, Any],
        key: str,
        event_context: dict[str, Any],
        raw_text: str,
        error_type: str,
        error_message: str,
    ) -> list[str]:
        """Parse one required non-empty list of non-empty strings."""
        values = self._parse_text_list(
            obj=obj,
            key=key,
            event_context=event_context,
            raw_text=raw_text,
            error_type=error_type,
            error_message=error_message,
        )
        if not values:
            self._log_teaching_parse_failure(
                event_context=event_context,
                error_type=error_type,
                error_message=error_message,
                raw_text=raw_text,
            )
            raise GroupEProtocolError(error_type, error_message)
        return values

    def _parse_text_list(
        self,
        obj: dict[str, Any],
        key: str,
        event_context: dict[str, Any],
        raw_text: str,
        error_type: str,
        error_message: str,
    ) -> list[str]:
        """Parse one list of non-empty strings."""
        raw_values = obj.get(key)
        if not isinstance(raw_values, list):
            self._log_teaching_parse_failure(
                event_context=event_context,
                error_type=error_type,
                error_message=error_message,
                raw_text=raw_text,
            )
            raise GroupEProtocolError(error_type, error_message)
        values: list[str] = []
        for raw_value in raw_values:
            text = str(raw_value).strip()
            if text:
                values.append(text)
        return values

    def _parse_confidence_value(
        self,
        obj: dict[str, Any],
        key: str,
        event_context: dict[str, Any],
        raw_text: str,
        error_type: str,
        error_message: str,
    ) -> float:
        """Parse one confidence value in the closed interval [0, 1]."""
        raw_value = obj.get(key)
        try:
            confidence = float(raw_value)
        except (TypeError, ValueError):
            self._log_teaching_parse_failure(
                event_context=event_context,
                error_type=error_type,
                error_message=error_message,
                raw_text=raw_text,
            )
            raise GroupEProtocolError(error_type, error_message) from None
        if confidence < 0.0 or confidence > 1.0:
            self._log_teaching_parse_failure(
                event_context=event_context,
                error_type=error_type,
                error_message=error_message,
                raw_text=raw_text,
            )
            raise GroupEProtocolError(error_type, error_message)
        return confidence

    def _contains_disallowed_default_fallback(self, rule_of_thumb: str) -> bool:
        """Check whether one rule uses disallowed default fallback wording."""
        text = rule_of_thumb.lower()
        return (
            "abnormal otherwise" in text
            or "normal otherwise" in text
            or "otherwise abnormal" in text
            or "otherwise normal" in text
        )

    def _extract_json_payload(self, model_response: ModelResponse) -> Any | None:
        """Extract JSON payload from a model response."""
        if model_response.parsed_json is not None:
            return model_response.parsed_json

        text = model_response.raw_text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        try:
            return json.loads(text)
        except Exception:
            return None

    def _is_single_sentence(self, text: str) -> bool:
        """Check whether text appears to contain a single sentence."""
        stripped = text.strip()
        if not stripped:
            return False
        if "\n" in stripped or "\r" in stripped:
            return False
        sentence_endings = sum(stripped.count(ch) for ch in (".", "!", "?"))
        return sentence_endings <= 1

    def _log_event(self, event: str, level: int = logging.INFO, **fields: Any) -> None:
        """Write one structured event if the event logger is available.

        Args:
            event: Stable event name.
            level: Logging level for the event.
            **fields: Event context and diagnostic fields.
        """
        if self.event_logger is None:
            return
        self.event_logger.log(
            logger=logger,
            event=event,
            level=level,
            **fields,
        )

    def _log_exam_parse_failure(
        self,
        event_context: dict[str, Any],
        expected_count: int,
        parsed_count: int,
        error_type: str,
        error_message: str,
        raw_text: str,
    ) -> None:
        """Record one structured parse failure event.

        Args:
            event_context: Context fields for the event.
            expected_count: Expected number of answers.
            parsed_count: Number of parsed valid answers.
            error_type: Stable parse error type.
            error_message: Human-readable parse failure message.
            raw_text: Raw model response text.
        """
        preview = self._response_preview(raw_text=raw_text)
        self._log_event(
            event="exam.parse_failed",
            level=logging.ERROR,
            **event_context,
            status="failed",
            expected_answers_count=expected_count,
            parsed_answers_count=parsed_count,
            error_type=error_type,
            error_message=error_message,
            response_preview=preview,
            response_length=len(raw_text),
        )
        logger.error(
            "[trial] exam parse failure participant_id=%s phase=%s reason=%s",
            event_context.get("participant_id"),
            event_context.get("phase"),
            error_message,
        )

    def _log_teaching_parse_failure(
        self,
        event_context: dict[str, Any],
        error_type: str,
        error_message: str,
        raw_text: str,
    ) -> None:
        """Record one structured teaching parse failure event."""
        preview = self._response_preview(raw_text=raw_text)
        self._log_event(
            event="teaching.parse_failed",
            level=logging.ERROR,
            **event_context,
            status="failed",
            error_type=error_type,
            error_message=error_message,
            response_preview=preview,
            response_length=len(raw_text),
        )
        logger.error(
            "[trial] teaching parse failure participant_id=%s item_id=%s reason=%s",
            event_context.get("participant_id"),
            event_context.get("item_id"),
            error_message,
        )

    def _response_preview(self, raw_text: str, max_chars: int = 220) -> str:
        """Create a compact one-line response preview.

        Args:
            raw_text: Raw text to preview.
            max_chars: Maximum preview size.

        Returns:
            Sanitised single-line preview.
        """
        clean = raw_text.replace("\n", " ").replace("\r", " ").strip()
        if len(clean) <= max_chars:
            return clean
        return f"{clean[: max_chars - 3]}..."

    def debug_summary(self) -> None:
        """Print a small summary of loaded metadata for debugging."""
        logger.debug("[trial] metadata summary")
        logger.debug(
            "  enabled groups: %s",
            [group.value for group in self.enabled_groups],
        )
        for group, items in self.teaching_by_group.items():
            logger.debug(f"  group {group.value}: {len(items)} teaching items")
        for exam_set_id, items in self.exam_by_set.items():
            logger.debug(f"  exam set {exam_set_id}: {len(items)} exam items")

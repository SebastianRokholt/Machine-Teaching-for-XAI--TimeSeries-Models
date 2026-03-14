# src/mllm_experiment/trial.py              
# core logic for running a trial for a single participant
# trial.py
from __future__ import annotations
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
    build_group_e_retry_correction_text,
    build_exam_user_content,
    build_teaching_user_content,
    exam_system_message,
)
from .config import ExperimentConfig

logger = logging.getLogger(__name__)


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
    ) -> tuple[list[dict[str, Any]], float, list[dict[str, Any]]]:
        """Run either the pre- or post-teaching exam phase.

        Args:
            participant: ParticipantConfig instance.
            rng: Random number generator instance.
            phase: Phase.PRE or Phase.POST.
            messages: Current chat message history.
            fixed_rule_of_thumb: Optional fixed rule-of-thumb string
                used for group E post-exam prompts.

        Returns:
            Tuple containing:
                - list of exam result rows,
                - accuracy value in [0, 1],
                - updated message history including the model's reply.
        """
        if phase is Phase.PRE:
            exam_set_id = participant.exam_set_pre
        else:
            exam_set_id = participant.exam_set_post

        items = list(self.exam_by_set[exam_set_id])

        # optionally shuffle within exam set (if desired)
        rng.shuffle(items)

        # resolve the correct image path for each example given the group
        item_paths: list[tuple[ExampleItem, Any]] = []
        for item in items[: self.config.pre_exam_items]:
            image_path = resolve_exam_image_path(
                exam_root=self.config.exam_root, 
                item=item, 
                group=participant.group,
                phase=phase)
            item_paths.append((item, image_path))

        user_content, modality_shown = build_exam_user_content(
            group=participant.group,
            phase=phase,
            exam_items=item_paths,
            fixed_rule_of_thumb=fixed_rule_of_thumb,
        )
        messages.append({"role": "user", "content": user_content})
        self._log_event(
            event="phase.exam.start",
            level=logging.INFO,
            participant_id=participant.participant_id,
            phase=phase.value,
            group=participant.group.value,
            exam_set_id=exam_set_id,
            items_count=len(item_paths),
            modality=modality_shown,
            status="started",
        )

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
        logger.debug(
            "[trial] exam prompt summary participant_id=%s phase=%s exam_set_id=%s "
            "items=%d text_parts=%d image_parts=%d item_ids=%s",
            participant.participant_id,
            phase.value,
            exam_set_id,
            len(item_paths),
            text_part_count,
            image_part_count,
            [item.item_id for item, _path in item_paths[:5]],
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
            },
        )

        # append assistant reply to history
        messages.append({"role": "assistant", "content": model_response.raw_text})

        event_context = {
            "participant_id": participant.participant_id,
            "phase": phase.value,
            "group": participant.group.value,
            "exam_set_id": exam_set_id,
            "call_type": "exam",
            "modality": modality_shown,
        }
        answers = self._parse_exam_answers(
            model_response=model_response,
            expected_count=len(item_paths),
            event_context=event_context,
        )
        rows: list[dict[str, Any]] = []
        correct = 0

        for item, _path in item_paths:
            guess = answers.get(item.item_id)
            is_correct = guess == item.ai_class
            if is_correct:
                correct += 1

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
                "response_time_ms": model_response.latency_ms,
                "prompt_tokens": model_response.prompt_tokens,
                "completion_tokens": model_response.completion_tokens,
                "raw_response": model_response.raw_text,
            }
            rows.append(row)

        accuracy = correct / len(item_paths) if item_paths else 0.0
        self._log_event(
            event="phase.exam.complete",
            level=logging.INFO,
            participant_id=participant.participant_id,
            phase=phase.value,
            group=participant.group.value,
            exam_set_id=exam_set_id,
            items_count=len(item_paths),
            parsed_answers_count=len(answers),
            expected_answers_count=len(item_paths),
            modality=modality_shown,
            accuracy=round(accuracy, 6),
            status="completed",
        )

        return rows, accuracy, messages

    def _parse_exam_answers(
        self,
        model_response: ModelResponse,
        expected_count: int,
        event_context: dict[str, Any],
    ) -> dict[str, str]:
        """Parse the model's JSON exam answers into a dictionary.

        Args:
            model_response: ModelResponse returned by the OpenAI client.
            expected_count: Expected number of answers.
            event_context: Context fields used for structured events.

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
            item_id = str(entry.get("item_id"))
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
                - final rule-of-thumb for group E, otherwise None.
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
            base_user_content = build_teaching_user_content(
                group=participant.group,
                item=item,
                image_path=image_path,
                index=idx,
                total=len(teaching_items),
                current_rule_of_thumb=current_rule_of_thumb,
            )

            if participant.group is Group.E:
                (
                    model_response,
                    current_rule_of_thumb,
                    committed_user_content,
                ) = self._run_group_e_teaching_item_with_retries(
                    participant=participant,
                    item=item,
                    index=idx,
                    total=len(teaching_items),
                    messages=messages,
                    base_user_content=base_user_content,
                    current_rule_of_thumb=current_rule_of_thumb,
                    modality_shown=modality_shown,
                )
                messages.append({"role": "user", "content": committed_user_content})
                messages.append({"role": "assistant", "content": model_response.raw_text})
            else:
                messages.append({"role": "user", "content": base_user_content})
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
                    },
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
            }
            rows.append(row)

        self._log_event(
            event="phase.teaching.complete",
            level=logging.INFO,
            participant_id=participant.participant_id,
            phase=Phase.TEACHING.value,
            group=participant.group.value,
            items_count=len(rows),
            modality=modality_shown,
            status="completed",
        )

        if participant.group is Group.E and not current_rule_of_thumb:
            msg = "Group E teaching session completes without a valid final rule-of-thumb."
            raise GroupEProtocolError(
                error_type="missing_final_rule",
                message=msg,
            )

        return rows, messages, current_rule_of_thumb

    def _run_group_e_teaching_item_with_retries(
        self,
        participant: ParticipantConfig,
        item: ExampleItem,
        index: int,
        total: int,
        messages: list[dict[str, Any]],
        base_user_content: list[dict[str, Any]],
        current_rule_of_thumb: str | None,
        modality_shown: str,
    ) -> tuple[ModelResponse, str, list[dict[str, Any]]]:
        """Run one group E teaching item with retain-action retries.

        Args:
            participant: Participant configuration.
            item: Current teaching item.
            index: Example index in the teaching sequence.
            total: Number of teaching examples.
            messages: Persistent participant chat history.
            base_user_content: Standard user content for this example.
            current_rule_of_thumb: Current group E rule-of-thumb before this item.
            modality_shown: Modality label used for logging.

        Returns:
            Tuple with model response, updated rule-of-thumb and committed user content.

        Raises:
            GroupEProtocolError: If a non-retryable protocol error occurs or retries
                for retain-rule mismatch are exhausted.
        """
        max_retries = self.config.group_e_retain_retry_attempts
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

            call_messages = list(messages)
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
                    "example_index": index,
                    "examples_total": total,
                    "example_retry_attempt": attempt,
                    "example_retry_attempts_total": total_attempts,
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
                retryable_error = exc.error_type == "retain_rule_changed"
                retries_remaining = max_retries - attempt + 1
                if retryable_error and retries_remaining > 0:
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
                if retryable_error:
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
            msg = "Group E response is missing a non-empty rule_of_thumb."
            self._log_teaching_parse_failure(
                event_context=event_context,
                error_type="missing_rule_of_thumb",
                error_message=msg,
                raw_text=model_response.raw_text,
            )
            raise GroupEProtocolError("missing_rule_of_thumb", msg)

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
        )
        return rule_of_thumb

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

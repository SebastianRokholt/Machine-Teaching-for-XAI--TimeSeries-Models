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
        group: Assigned teaching group (A, B, C or D).
        exam_set_pre: Exam set used in the pre-teaching phase.
        exam_set_post: Exam set used in the post-teaching phase.
    """

    participant_id: str
    group: Group
    exam_set_pre: str
    exam_set_post: str


class TrialRunner:
    """Coordinate the full trial lifecycle for one or more participants.

    The trial runner loads metadata, assigns groups and exam sets, and
    manages the three-phase protocol (pre-teaching exam, teaching phase,
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
        missing_teaching = [
            group.value
            for group in self.enabled_groups
            if not self.teaching_by_group.get(group)
        ]
        if missing_teaching:
            msg = (
                "No teaching metadata rows found for selected group(s): "
                f"{', '.join(missing_teaching)}"
            )
            logger.error(msg)
            raise ValueError(msg)

    def run_participant(self, rng: random.Random, index: int) -> bool:
        """Run a full trial for a single participant.

        Args:
            rng: Random number generator instance.
            index: Index of the participant within this run.

        Returns:
            True when the participant completes all phases. False when
            API retries are exhausted and the participant fails safely.
        """
        participant = self._sample_participant_config(rng, index)
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
            teaching_rows, messages = self._run_teaching_phase(
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
            summary_row = {
                "participant_id": participant.participant_id,
                "group": participant.group.value,
                "exam_set_pre": participant.exam_set_pre,
                "exam_set_post": participant.exam_set_post,
                "accuracy_pre": None,
                "accuracy_post": None,
                "delta_accuracy": None,
                "status": "failed",
                "error_type": exc.last_error_type,
                "error_message": str(exc),
            }
            log_participant_summaries([summary_row], self.config.output_root)
            self._log_event(
                event="participant.failed",
                level=logging.ERROR,
                participant_id=participant.participant_id,
                group=participant.group.value,
                exam_set_pre=participant.exam_set_pre,
                exam_set_post=participant.exam_set_post,
                status="failed",
                error_type=exc.last_error_type,
                error_message=str(exc),
                attempts=exc.attempts,
            )
            logger.error(
                "[trial] participant failed participant_id=%s group=%s reason=%s",
                participant.participant_id,
                participant.group.value,
                str(exc),
            )
            return False

    def _sample_participant_config(self, rng: random.Random, index: int) -> ParticipantConfig:
        """Sample group and exam sets for one participant."""
        participant_id = f"p_{index:04d}_{uuid.uuid4().hex[:8]}"

        group = rng.choice(self.enabled_groups)

        exam_sets = list(self.exam_by_set.keys())
        if len(exam_sets) != 2:
            msg = f"Expected exactly two exam sets, found {len(exam_sets)}."
            logger.exception(msg)
            raise ValueError(msg)

        rng.shuffle(exam_sets)
        exam_set_pre, exam_set_post = exam_sets

        return ParticipantConfig(
            participant_id=participant_id,
            group=group,
            exam_set_pre=exam_set_pre,
            exam_set_post=exam_set_post,
        )

    def _run_exam_phase(
        self,
        participant: ParticipantConfig,
        rng: random.Random,
        phase: Phase,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], float, list[dict[str, Any]]]:
        """Run either the pre- or post-teaching exam phase.

        Args:
            participant: ParticipantConfig instance.
            rng: Random number generator instance.
            phase: Phase.PRE or Phase.POST.
            messages: Current chat message history.

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

        obj: Any | None = model_response.parsed_json
        if obj is None:
            # try to be robust to small wrappers like ```json ...```
            text = model_response.raw_text.strip()
            if text.startswith("```"):
                text = text.strip("`")
                if text.lower().startswith("json"):
                    text = text[4:].strip()
            try:
                obj = json.loads(text)
            except Exception:
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
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Run the teaching phase with sequential labelled examples.

        Args:
            participant: ParticipantConfig instance.
            rng: Random number generator instance.
            messages: Current chat message history.

        Returns:
            Tuple containing:
                - list of teaching result rows,
                - updated message history including all teaching replies.
        """
        teaching_items = list(self.teaching_by_group[participant.group])
        # limit to configured number of teaching items
        teaching_items = teaching_items[: self.config.teaching_items]

        # group A keeps curriculum order as specified by metadata
        # groups B and C randomise the order
        if participant.group in (Group.B, Group.C):
            rng.shuffle(teaching_items)

        if participant.group in (Group.A, Group.B):
            modality_shown = "overlay"
        elif participant.group is Group.D:
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
            user_content = build_teaching_user_content(
                group=participant.group,
                item=item,
                image_path=image_path,
                index=idx,
                total=len(teaching_items),
            )
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

        return rows, messages

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

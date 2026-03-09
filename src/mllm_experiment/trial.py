# src/mllm_experiment/trial.py              
# core logic for running a trial for a single participant
# trial.py
from __future__ import annotations
import json
import logging
from pathlib import Path
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
    log_exam_results,
    log_participant_summaries,
    log_teaching_results,
)
from .openai_client import OpenAIChatClient, ModelResponse
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
        group: Assigned teaching group (A, B or C).
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

    def __init__(self, config: ExperimentConfig, client: OpenAIChatClient, verbose: bool) -> None:
        self.config = config
        self.client = client
        self.verbose = verbose

        logger.debug("[trial] loading metadata")
        self.teaching_by_group = load_teaching_metadata(config)
        self.exam_by_set = load_exam_metadata(config)

    def run_participant(self, rng: random.Random, index: int) -> None:
        """Run a full trial for a single participant.

        Args:
            rng: Random number generator instance.
            index: Index of the participant within this run.
        """
        participant = self._sample_participant_config(rng, index)
        logger.info(
            "[trial] starting participant %s in group %s, pre=%s, post=%s",
            participant.participant_id,
            participant.group.value,
            participant.exam_set_pre,
            participant.exam_set_post,
        )

        # initialise chat history with a system message
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
        # re-initialises the chat history with only the system message
        messages: list[dict[str, Any]] = [exam_system_message()]
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

        # participant summary
        summary_row = {
            "participant_id": participant.participant_id,
            "group": participant.group.value,
            "exam_set_pre": participant.exam_set_pre,
            "exam_set_post": participant.exam_set_post,
            "accuracy_pre": pre_accuracy,
            "accuracy_post": post_accuracy,
            "delta_accuracy": post_accuracy - pre_accuracy,
        }
        log_participant_summaries([summary_row], self.config.output_root)

        logger.info(
            "[trial] completed participant %s (accuracy_pre=%.3f, accuracy_post=%.3f, delta=%.3f)",
            participant.participant_id,
            pre_accuracy,
            post_accuracy,
            post_accuracy - pre_accuracy,
        )

    def _sample_participant_config(self, rng: random.Random, index: int) -> ParticipantConfig:
        """Sample group and exam sets for one participant."""
        participant_id = f"p_{index:04d}_{uuid.uuid4().hex[:8]}"

        group = rng.choice(list(Group))

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
        logger.info("[trial] sending exam prompt with %d items to model.", len(item_paths))
        logger.debug("[trial] exam prompt content preview: %s", [c for c in user_content if c.get("type") == "text"])

        model_response: ModelResponse = self.client.call(messages, verbose=self.verbose)

        # append assistant reply to history
        messages.append({"role": "assistant", "content": model_response.raw_text})

        answers = self._parse_exam_answers(model_response, len(item_paths))
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
        logger.info(
            "[trial] %s phase=%s accuracy=%.3f",
            participant.participant_id,
            phase.value,
            accuracy,
        )

        return rows, accuracy, messages

    def _parse_exam_answers(
        self,
        model_response: ModelResponse,
        expected_count: int,
    ) -> dict[str, str]:
        """Parse the model's JSON exam answers into a dictionary.

        Args:
            model_response: ModelResponse returned by the OpenAI client.
            expected_count: Expected number of answers.

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
                logger.error("[trial] failed to parse model response as JSON. Response: %s", model_response.raw_text)
                return answers

        if not isinstance(obj, dict):
            logger.error("[trial] parsed JSON exam answers are not a dict")
            return answers

        answers_list = obj.get("answers")
        if not isinstance(answers_list, list):
            logger.error("[trial] parsed JSON exam answers 'answers' field is not a list")
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
            logger.error(
                "[trial] warning: could not parse any valid answers "
                f"(expected {expected_count})"
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

        rows: list[dict[str, Any]] = []

        for idx, item in tqdm(enumerate(teaching_items, start=1), desc="Teaching examples", unit="example"):
            image_path = resolve_teaching_image_path(self.config.teaching_root, item)
            user_content = build_teaching_user_content(
                group=participant.group,
                item=item,
                image_path=image_path,
                index=idx,
                total=len(teaching_items),
            )
            messages.append({"role": "user", "content": user_content})

            model_response = self.client.call(messages, verbose=self.verbose)

            # append assistant reply to history
            messages.append({"role": "assistant", "content": model_response.raw_text})

            modality_shown = "overlay" if participant.group in (Group.A, Group.B) else "raw_only"

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

        logger.info(f"[trial] teaching phase completed for {participant.participant_id}")

        return rows, messages

    def debug_summary(self) -> None:
        """Print a small summary of loaded metadata for debugging."""
        logger.debug("[trial] metadata summary")
        for group, items in self.teaching_by_group.items():
            logger.debug(f"  group {group.value}: {len(items)} teaching items")
        for exam_set_id, items in self.exam_by_set.items():
            logger.debug(f"  exam set {exam_set_id}: {len(items)} exam items")

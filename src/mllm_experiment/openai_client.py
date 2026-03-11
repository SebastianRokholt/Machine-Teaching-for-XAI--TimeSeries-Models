# src/mllm_experiment/openai_client.py
# wrapper around OpenAI Responses API
from __future__ import annotations
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any
from openai import OpenAI

from .utils import ExperimentEventLogger

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ModelResponse:
    """Store the result of a single model call.

    Attributes:
        raw_text: Raw text returned by the model.
        parsed_json: Parsed JSON object if the response is valid JSON,
            otherwise None.
        latency_ms: Time between sending the request and receiving the
            response, in milliseconds.
        prompt_tokens: Number of prompt tokens reported by the API, or
            None if unavailable.
        completion_tokens: Number of completion tokens reported by the
            API, or None if unavailable.
    """

    raw_text: str
    parsed_json: Any | None
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None


@dataclass(slots=True)
class OpenAIChatClient:
    """Lightweight wrapper around the OpenAI chat completions API.

    This client is initialised with a model name and basic decoding
    parameters. It expects the OPENAI_API_KEY environment variable to
    be set before use, unless the dummy mode is enabled.

    Attributes:
        model: Name of the OpenAI model to use.
        temperature: Sampling temperature for the model.
        max_completion_tokens: Maximum number of tokens in the completion.
        use_dummy: If true, skip real API calls and return synthetic
            responses for dry-run debugging.
    """

    model: str
    temperature: float | None = None
    max_completion_tokens: int | None = None
    use_dummy: bool = False
    event_logger: ExperimentEventLogger | None = None
    client: Any | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        """Initialise the underlying OpenAI client or dummy mode.

        When use_dummy is true, this method skips environment checks
        and avoids initialising the real OpenAI client. All calls will
        then be handled by the dummy call path.
        """
        if self.use_dummy:
            logger.info("OpenAIChatClient initialised in dummy mode")
            self.client = None
            return
        if not os.getenv("OPENAI_API_KEY"):
            from dotenv import load_dotenv
            load_dotenv()
            if not os.getenv("OPENAI_API_KEY"):
                msg = (
                    "OPENAI_API_KEY is not set. Please export your API key before "
                    "running the experiment or use --dry-run."
                )
                logger.critical(msg)
                raise RuntimeError(msg)
        self.client = OpenAI()
        logger.debug("OpenAIChatClient initialised with model %s", self.model)

    def call(
        self,
        messages: list[dict[str, Any]],
        verbose: bool = False,
        call_context: dict[str, Any] | None = None,
    ) -> ModelResponse:
        """Call the OpenAI chat API or generate a dummy response.

        Args:
            messages: Full chat history including system, user and
                assistant messages.
            verbose: Flag that controls extra debug printing.
            call_context: Optional context fields for structured logging.

        Returns:
            ModelResponse with raw text, parsed JSON and usage metadata.
        """
        if self.client is None and not self.use_dummy:
            logger.critical("OpenAI client is not initialised.")
            raise RuntimeError("OpenAI client is not initialised.")

        if self.use_dummy:
            return self._dummy_call(
                messages=messages,
                verbose=verbose,
                call_context=call_context,
            )

        logger.debug(
            "[openai_client] sending model request model=%s messages=%d",
            self.model,
            len(messages),
        )

        start = time.perf_counter()

        is_mini_model = "mini" in self.model

        # uses kwargs so we can easily omit unsupported parameters.
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            # Force strict JSON output for all phases (exam + teaching).
            "response_format": {"type": "json_object"},
        }
        if self.temperature is not None and not is_mini_model:
            # For models like gpt-5-mini that only support the default
            # temperature, leave temperature=None so it is omitted.
            kwargs["temperature"] = self.temperature
        if self.max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = self.max_completion_tokens

        response = self.client.chat.completions.create(**kwargs)
        end = time.perf_counter()
        latency_ms = (end - start) * 1000.0
        message = response.choices[0].message

        # message.content is usually a string, but be defensive in case
        # the API ever returns a list of content parts.
        raw_content = message.content
        if isinstance(raw_content, str):
            raw_text = raw_content
        elif isinstance(raw_content, list):
            pieces: list[str] = []
            for part in raw_content:
                if isinstance(part, dict) and part.get("type") == "text":
                    pieces.append(str(part.get("text", "")))
            raw_text = "".join(pieces)
        else:
            raw_text = "" if raw_content is None else str(raw_content)

        parsed: Any | None
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            parsed = None
            preview_source = raw_text if isinstance(raw_text, str) else str(raw_text)
            preview = preview_source[:300].replace("\n", " ")
            logger.warning(
                "[openai_client] response is not valid JSON; preview: %s ...",
                preview,
            )

        prompt_tokens = getattr(response.usage, "prompt_tokens", None)
        completion_tokens = getattr(response.usage, "completion_tokens", None)

        logger.debug("[openai_client] latency_ms=%.1f", latency_ms)

        self._log_model_call_complete(
            call_context=call_context,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            response_length=len(raw_text),
            message_count=len(messages),
        )

        return ModelResponse(
            raw_text=raw_text,
            parsed_json=parsed,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def _dummy_call(
        self,
        messages: list[dict[str, Any]],
        verbose: bool = False,
        call_context: dict[str, Any] | None = None,
    ) -> ModelResponse:
        """Generate a synthetic response for dry-run debugging.

        This method first checks the structured call context to detect
        whether the query corresponds to an exam phase or a teaching
        phase. It falls back to prompt-text heuristics only when the
        context does not include a call type.

        For exam phases, it returns a JSON object of the form
        {"answers": [{"item_id": "...", "guess": "normal"}, ...]} where
        the item identifiers are extracted from the prompt.

        For teaching phases, it returns {"acknowledged": true}.

        Args:
            messages: Full chat history including system, user and
                assistant messages.
            verbose: Flag that controls extra debug printing.
            call_context: Optional context fields for structured logging.

        Returns:
            ModelResponse with synthetic JSON and zeroed usage metrics.
        """
        logger.debug("[openai_client] dummy mode: generating synthetic response")

        if not messages:
            raw_text = '{"acknowledged": true}'
            parsed = json.loads(raw_text)
            self._log_model_call_complete(
                call_context=call_context,
                latency_ms=0.0,
                prompt_tokens=None,
                completion_tokens=None,
                response_length=len(raw_text),
                message_count=0,
            )
            return ModelResponse(
                raw_text=raw_text,
                parsed_json=parsed,
                latency_ms=0.0,
                prompt_tokens=None,
                completion_tokens=None,
            )

        last = messages[-1]
        content = last.get("content", [])
        text_parts: list[str] = []
        prompt_item_ids: list[str] = []

        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                text = str(part.get("text", ""))
                text_parts.append(text)
                if "item_id=" in text:
                    start = text.find("item_id=") + len("item_id=")
                    end = text.find(")", start)
                    if end == -1:
                        end = len(text)
                    item_id = text[start:end].strip()
                    if item_id:
                        prompt_item_ids.append(item_id)

        context_item_ids = self._item_ids_from_call_context(call_context=call_context)
        call_type = str((call_context or {}).get("call_type", "")).strip().lower()
        if call_type == "exam":
            exam_prompt = True
            detection_source = "call_context"
        elif call_type == "teaching":
            exam_prompt = False
            detection_source = "call_context"
        else:
            exam_prompt = self._is_exam_prompt(text_parts=text_parts)
            detection_source = "prompt_heuristic"

        if exam_prompt:
            item_ids = context_item_ids or prompt_item_ids
        else:
            item_ids = prompt_item_ids

        logger.debug(
            "[openai_client] dummy prompt routing source=%s call_type=%s exam_prompt=%s "
            "context_item_ids=%d prompt_item_ids=%d",
            detection_source,
            call_type or "n/a",
            exam_prompt,
            len(context_item_ids),
            len(prompt_item_ids),
        )

        if exam_prompt:
            answers = [{"item_id": iid, "guess": "normal"} for iid in item_ids]
            if not answers:
                logger.warning(
                    "[openai_client] dummy exam response contains no item_ids "
                    "(call_type=%s detection_source=%s)",
                    call_type or "n/a",
                    detection_source,
                )
            raw_text = json.dumps({"answers": answers})
            parsed = {"answers": answers}
        else:
            raw_text = '{"acknowledged": true}'
            parsed = {"acknowledged": True}

        if exam_prompt:
            logger.debug(f"[openai_client] dummy exam response with {len(item_ids)} answers")
        else:
            logger.debug("[openai_client] dummy teaching acknowledgement")

        self._log_model_call_complete(
            call_context=call_context,
            latency_ms=0.0,
            prompt_tokens=None,
            completion_tokens=None,
            response_length=len(raw_text),
            message_count=len(messages),
        )

        return ModelResponse(
            raw_text=raw_text,
            parsed_json=parsed,
            latency_ms=0.0,
            prompt_tokens=None,
            completion_tokens=None,
        )

    def _item_ids_from_call_context(
        self,
        call_context: dict[str, Any] | None,
    ) -> list[str]:
        """Extract exam item identifiers from structured call context.

        Args:
            call_context: Optional context dictionary passed by the caller.

        Returns:
            List of string item identifiers in context order.
        """
        if call_context is None:
            return []
        raw_item_ids = call_context.get("item_ids")
        if not isinstance(raw_item_ids, list):
            return []

        item_ids: list[str] = []
        for item_id in raw_item_ids:
            text = str(item_id).strip()
            if text:
                item_ids.append(text)
        return item_ids

    def _is_exam_prompt(self, text_parts: list[str]) -> bool:
        """Detect whether the current prompt is an exam prompt.

        Args:
            text_parts: Text segments from the last user message.

        Returns:
            True when the message asks for batch exam answers.
        """
        if not text_parts:
            return False

        text = " ".join(text_parts).lower()
        has_answers = "'answers'" in text or '"answers"' in text
        has_guess = "'guess'" in text or '"guess"' in text or "guess" in text
        has_batch_exam = "batch" in text and "exam" in text
        has_item_id = "item_id" in text
        return (has_batch_exam and has_guess) or (has_answers and has_guess and has_item_id)

    def _log_model_call_complete(
        self,
        call_context: dict[str, Any] | None,
        latency_ms: float,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        response_length: int,
        message_count: int,
    ) -> None:
        """Emit one structured model call summary event.

        Args:
            call_context: Optional context fields from the caller.
            latency_ms: Call latency in milliseconds.
            prompt_tokens: Prompt token count when available.
            completion_tokens: Completion token count when available.
            response_length: Length of raw response text.
            message_count: Number of chat messages sent to the model.
        """
        if self.event_logger is None:
            return

        fields = dict(call_context or {})
        fields.update(
            {
                "status": "completed",
                "mode": "dummy" if self.use_dummy else "api",
                "latency_ms": round(latency_ms, 3),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "response_length": response_length,
                "message_count": message_count,
            }
        )
        self.event_logger.log(
            logger=logger,
            event="model.call.complete",
            level=logging.INFO,
            **fields,
        )

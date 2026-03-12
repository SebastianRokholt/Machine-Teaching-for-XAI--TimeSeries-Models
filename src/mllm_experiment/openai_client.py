# src/mllm_experiment/openai_client.py
# wrapper around OpenAI Responses API
from __future__ import annotations
import json
import logging
import os
import random
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    ConflictError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

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


class ModelCallRetriesExhaustedError(RuntimeError):
    """Signal that all API retry attempts are exhausted."""

    def __init__(self, attempts: int, last_error: Exception) -> None:
        self.attempts = attempts
        self.last_error = last_error
        self.last_error_type = type(last_error).__name__
        message = (
            "OpenAI API retries are exhausted "
            f"after {attempts} attempts. "
            f"Last error type is {self.last_error_type}."
        )
        super().__init__(message)


@dataclass(slots=True)
class _RateLimitReservation:
    """Track one provisional token reservation for an in-flight call."""

    reserved_tokens: int


class _SharedRateLimiter:
    """Coordinate request and token budgets across worker threads.

    This limiter enforces per-minute request and token constraints and
    caps in-flight requests with a semaphore. It blocks callers until
    both budgets have capacity.
    """

    def __init__(
        self,
        requests_per_minute: int,
        tokens_per_minute: int,
        max_inflight_calls: int,
    ) -> None:
        self.requests_per_minute = max(1, requests_per_minute)
        self.tokens_per_minute = max(1, tokens_per_minute)
        self.window_seconds = 60.0

        self._semaphore = threading.Semaphore(max(1, max_inflight_calls))
        self._condition = threading.Condition(threading.Lock())
        self._request_timestamps: deque[float] = deque()
        self._token_events: deque[tuple[float, int]] = deque()
        self._provisional_tokens = 0

    def acquire(self, reserved_tokens: int) -> tuple[_RateLimitReservation, float]:
        """Acquire budget capacity for one API call.

        Args:
            reserved_tokens: Estimated tokens reserved for this call.

        Returns:
            Tuple of reservation object and total wait seconds.
        """
        reserved_tokens = max(1, int(reserved_tokens))
        self._semaphore.acquire()

        waited_seconds = 0.0
        with self._condition:
            while True:
                now = time.monotonic()
                self._prune(now=now)

                request_wait = self._request_wait_seconds(now=now)
                token_wait = self._token_wait_seconds(
                    now=now,
                    reserved_tokens=reserved_tokens,
                )
                wait_seconds = max(request_wait, token_wait)

                if wait_seconds <= 0.0:
                    self._request_timestamps.append(now)
                    self._provisional_tokens += reserved_tokens
                    self._condition.notify_all()
                    return _RateLimitReservation(reserved_tokens=reserved_tokens), waited_seconds

                wait_start = time.monotonic()
                self._condition.wait(timeout=wait_seconds)
                waited_seconds += max(0.0, time.monotonic() - wait_start)

    def finalise(
        self,
        reservation: _RateLimitReservation,
        actual_tokens: int | None,
    ) -> None:
        """Release one reservation and record actual token usage.

        Args:
            reservation: Reservation object returned by acquire.
            actual_tokens: Actual tokens used for the completed request.
                When None, the reserved estimate is used conservatively.
        """
        completed_tokens = reservation.reserved_tokens
        if actual_tokens is not None:
            completed_tokens = max(0, int(actual_tokens))

        with self._condition:
            self._provisional_tokens = max(
                0,
                self._provisional_tokens - reservation.reserved_tokens,
            )
            self._token_events.append((time.monotonic(), completed_tokens))
            self._prune(now=time.monotonic())
            self._condition.notify_all()

        self._semaphore.release()

    def _prune(self, now: float) -> None:
        """Drop usage records older than the active window."""
        cutoff = now - self.window_seconds
        while self._request_timestamps and self._request_timestamps[0] <= cutoff:
            self._request_timestamps.popleft()
        while self._token_events and self._token_events[0][0] <= cutoff:
            self._token_events.popleft()

    def _request_wait_seconds(self, now: float) -> float:
        """Compute wait seconds for request-per-minute capacity."""
        if len(self._request_timestamps) < self.requests_per_minute:
            return 0.0
        oldest = self._request_timestamps[0]
        return max(0.0, (oldest + self.window_seconds) - now)

    def _token_wait_seconds(self, now: float, reserved_tokens: int) -> float:
        """Compute wait seconds for token-per-minute capacity."""
        used_tokens = self._token_usage()
        if used_tokens + reserved_tokens <= self.tokens_per_minute:
            return 0.0

        projected_tokens = used_tokens
        for timestamp, tokens in self._token_events:
            projected_tokens -= tokens
            if projected_tokens + reserved_tokens <= self.tokens_per_minute:
                return max(0.0, (timestamp + self.window_seconds) - now)

        return self.window_seconds

    def _token_usage(self) -> int:
        """Return used plus provisional tokens in the current window."""
        completed_tokens = sum(tokens for _ts, tokens in self._token_events)
        return completed_tokens + self._provisional_tokens


@dataclass(slots=True)
class OpenAIChatClient:
    """Lightweight wrapper around the OpenAI chat completions API.

    This client is initialised with model and retry settings. It expects
    OPENAI_API_KEY to be set before use, unless dummy mode is enabled.

    Attributes:
        model: Name of the OpenAI model to use.
        temperature: Sampling temperature for the model.
        max_completion_tokens: Maximum number of completion tokens.
        use_dummy: If true, skip real API calls and return synthetic
            responses for dry-run debugging.
        max_requests_per_minute: Request-per-minute budget.
        max_tokens_per_minute: Token-per-minute budget.
        max_inflight_api_calls: Maximum in-flight API calls.
        timeout_seconds: Timeout per API request in seconds.
        retry_attempts: Number of retries after the initial attempt.
        retry_base_delay_seconds: Base delay for retry backoff.
        retry_max_delay_seconds: Maximum delay for retry backoff.
        retry_jitter_fraction: Fractional jitter for retry backoff.
    """

    model: str
    temperature: float | None = None
    max_completion_tokens: int | None = None
    use_dummy: bool = False
    max_requests_per_minute: int = 500
    max_tokens_per_minute: int = 500_000
    max_inflight_api_calls: int = 2
    timeout_seconds: float = 600.0
    retry_attempts: int = 12
    retry_base_delay_seconds: float = 2.0
    retry_max_delay_seconds: float = 120.0
    retry_jitter_fraction: float = 0.2
    event_logger: ExperimentEventLogger | None = None
    client: Any | None = field(init=False, default=None)
    rate_limiter: _SharedRateLimiter | None = field(init=False, default=None)

    # Conservative per-image estimate for vision requests.
    image_token_estimate: int = 1000
    default_completion_token_estimate: int = 2048

    def __post_init__(self) -> None:
        """Initialise the OpenAI client and the shared limiter."""
        if self.max_inflight_api_calls <= 0:
            msg = "max_inflight_api_calls must be greater than 0."
            raise ValueError(msg)

        self.rate_limiter = _SharedRateLimiter(
            requests_per_minute=self.max_requests_per_minute,
            tokens_per_minute=self.max_tokens_per_minute,
            max_inflight_calls=self.max_inflight_api_calls,
        )

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

        self.client = OpenAI(
            timeout=self.timeout_seconds,
            max_retries=0,
        )
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

        Raises:
            ModelCallRetriesExhaustedError: If all retries are exhausted.
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

        is_mini_model = "mini" in self.model

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        if self.temperature is not None and not is_mini_model:
            kwargs["temperature"] = self.temperature
        if self.max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = self.max_completion_tokens

        estimated_tokens = self._estimate_call_tokens(messages=messages)

        for attempt_idx in range(self.retry_attempts + 1):
            attempt = attempt_idx + 1
            if self.rate_limiter is None:
                msg = "Rate limiter is not initialised."
                raise RuntimeError(msg)

            reservation, wait_seconds = self.rate_limiter.acquire(
                reserved_tokens=estimated_tokens,
            )
            if wait_seconds > 0.0:
                self._log_model_call_throttled(
                    call_context=call_context,
                    wait_seconds=wait_seconds,
                    reserved_tokens=estimated_tokens,
                    attempt=attempt,
                )

            start = time.perf_counter()
            try:
                response = self.client.chat.completions.create(**kwargs)
            except APIStatusError as exc:
                self.rate_limiter.finalise(
                    reservation=reservation,
                    actual_tokens=None,
                )
                if self._is_retryable_status_error(exc):
                    if attempt > self.retry_attempts:
                        self._log_model_call_failed(
                            call_context=call_context,
                            error=exc,
                            attempts=attempt,
                        )
                        raise ModelCallRetriesExhaustedError(
                            attempts=attempt,
                            last_error=exc,
                        ) from exc
                    delay_seconds = self._retry_delay_seconds(
                        attempt_idx=attempt_idx,
                        error=exc,
                    )
                    self._log_model_call_retry(
                        call_context=call_context,
                        error=exc,
                        attempt=attempt,
                        retry_delay_seconds=delay_seconds,
                    )
                    time.sleep(delay_seconds)
                    continue

                self._log_model_call_failed(
                    call_context=call_context,
                    error=exc,
                    attempts=attempt,
                )
                raise
            except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError, ConflictError) as exc:
                self.rate_limiter.finalise(
                    reservation=reservation,
                    actual_tokens=None,
                )
                if attempt > self.retry_attempts:
                    self._log_model_call_failed(
                        call_context=call_context,
                        error=exc,
                        attempts=attempt,
                    )
                    raise ModelCallRetriesExhaustedError(
                        attempts=attempt,
                        last_error=exc,
                    ) from exc

                delay_seconds = self._retry_delay_seconds(
                    attempt_idx=attempt_idx,
                    error=exc,
                )
                self._log_model_call_retry(
                    call_context=call_context,
                    error=exc,
                    attempt=attempt,
                    retry_delay_seconds=delay_seconds,
                )
                time.sleep(delay_seconds)
                continue
            except Exception as exc:
                self.rate_limiter.finalise(
                    reservation=reservation,
                    actual_tokens=None,
                )
                self._log_model_call_failed(
                    call_context=call_context,
                    error=exc,
                    attempts=attempt,
                )
                raise

            end = time.perf_counter()
            latency_ms = (end - start) * 1000.0
            message = response.choices[0].message

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

            self.rate_limiter.finalise(
                reservation=reservation,
                actual_tokens=self._actual_tokens_used(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                ),
            )

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

        msg = "Unreachable call path in OpenAIChatClient.call"
        raise RuntimeError(msg)

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
        item identifiers are extracted from the prompt.

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
            group_value = str((call_context or {}).get("group", "")).strip().upper()
            if call_type == "teaching" and group_value == "E":
                example_index = self._teaching_example_index(text_parts=text_parts)
                if example_index is None:
                    example_index = 1

                if example_index == 1:
                    rule_action = "write"
                    rule_of_thumb = (
                        "The AI labels a charging session as abnormal when simplified "
                        "power or SOC shows clear abrupt deviations from smooth behaviour."
                    )
                else:
                    rule_action = "rephrase"
                    rule_of_thumb = (
                        "The AI tends to classify as abnormal when simplified power or "
                        "SOC departs sharply from a smooth charging profile."
                    )
                description_sentence = (
                    "The simplified power and SOC curves show one clear charging behaviour pattern."
                )
                parsed = {
                    "description_sentence": description_sentence,
                    "rule_action": rule_action,
                    "rule_of_thumb": rule_of_thumb,
                }
                raw_text = json.dumps(parsed)
            else:
                raw_text = '{"acknowledged": true}'
                parsed = {"acknowledged": True}

        if exam_prompt:
            logger.debug("[openai_client] dummy exam response with %d answers", len(item_ids))
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

    def _teaching_example_index(self, text_parts: list[str]) -> int | None:
        """Extract teaching example index from prompt text when available."""
        text = " ".join(text_parts)
        match = re.search(
            r"teaching example\s+(\d+)\s+of\s+\d+",
            text,
            flags=re.IGNORECASE,
        )
        if match is None:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

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

    def _estimate_call_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Estimate total tokens for one call before dispatch.

        Args:
            messages: Message history sent to the model.

        Returns:
            Conservative token estimate for prompt plus completion.
        """
        prompt_estimate = 0
        for message in messages:
            prompt_estimate += 16
            prompt_estimate += self._estimate_content_tokens(message.get("content"))

        completion_estimate = (
            self.max_completion_tokens
            if self.max_completion_tokens is not None
            else self.default_completion_token_estimate
        )
        total_estimate = prompt_estimate + max(1, int(completion_estimate))
        return max(1, total_estimate)

    def _estimate_content_tokens(self, content: Any) -> int:
        """Estimate tokens for one chat content payload."""
        if isinstance(content, str):
            return max(1, len(content) // 4)

        if isinstance(content, list):
            estimate = 0
            for part in content:
                if not isinstance(part, dict):
                    estimate += max(1, len(str(part)) // 4)
                    continue

                part_type = part.get("type")
                if part_type == "text":
                    estimate += max(1, len(str(part.get("text", ""))) // 4)
                elif part_type == "image_url":
                    estimate += self.image_token_estimate
                else:
                    estimate += max(1, len(str(part)) // 4)
            return max(1, estimate)

        if content is None:
            return 1
        return max(1, len(str(content)) // 4)

    def _actual_tokens_used(
        self,
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> int | None:
        """Convert usage fields to total used tokens when available."""
        total = 0
        has_usage = False
        if isinstance(prompt_tokens, int):
            total += max(0, prompt_tokens)
            has_usage = True
        if isinstance(completion_tokens, int):
            total += max(0, completion_tokens)
            has_usage = True
        if has_usage:
            return total
        return None

    def _is_retryable_status_error(self, error: APIStatusError) -> bool:
        """Return True for retryable status-code errors."""
        status_code = getattr(error, "status_code", None)
        if isinstance(status_code, int) and status_code >= 500:
            return True
        return False

    def _retry_delay_seconds(self, attempt_idx: int, error: Exception) -> float:
        """Calculate retry backoff with jitter and Retry-After support."""
        backoff = self.retry_base_delay_seconds * (2 ** attempt_idx)
        capped = min(backoff, self.retry_max_delay_seconds)
        jitter = capped * self.retry_jitter_fraction * random.random()
        delay_seconds = capped + jitter

        retry_after = self._retry_after_seconds(error=error)
        if retry_after is not None:
            delay_seconds = max(delay_seconds, retry_after)

        return max(0.0, delay_seconds)

    def _retry_after_seconds(self, error: Exception) -> float | None:
        """Extract Retry-After from API error headers when available."""
        response = getattr(error, "response", None)
        if response is None:
            return None

        headers = getattr(response, "headers", None)
        if headers is None:
            return None

        value = headers.get("retry-after")
        if value is None:
            return None

        try:
            retry_after = float(str(value).strip())
        except ValueError:
            return None

        if retry_after < 0:
            return None
        return retry_after

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

    def _log_model_call_throttled(
        self,
        call_context: dict[str, Any] | None,
        wait_seconds: float,
        reserved_tokens: int,
        attempt: int,
    ) -> None:
        """Emit one structured throttling event."""
        if self.event_logger is None:
            return

        fields = dict(call_context or {})
        fields.update(
            {
                "status": "throttled",
                "mode": "dummy" if self.use_dummy else "api",
                "wait_seconds": round(wait_seconds, 3),
                "reserved_tokens": reserved_tokens,
                "attempt": attempt,
            }
        )
        self.event_logger.log(
            logger=logger,
            event="model.call.throttled",
            level=logging.INFO,
            **fields,
        )

    def _log_model_call_retry(
        self,
        call_context: dict[str, Any] | None,
        error: Exception,
        attempt: int,
        retry_delay_seconds: float,
    ) -> None:
        """Emit one structured retry event."""
        if self.event_logger is None:
            return

        fields = dict(call_context or {})
        fields.update(
            {
                "status": "retrying",
                "mode": "dummy" if self.use_dummy else "api",
                "attempt": attempt,
                "retry_delay_seconds": round(retry_delay_seconds, 3),
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
        self.event_logger.log(
            logger=logger,
            event="model.call.retry",
            level=logging.WARNING,
            **fields,
        )

    def _log_model_call_failed(
        self,
        call_context: dict[str, Any] | None,
        error: Exception,
        attempts: int,
    ) -> None:
        """Emit one structured call failure event."""
        if self.event_logger is None:
            return

        fields = dict(call_context or {})
        fields.update(
            {
                "status": "failed",
                "mode": "dummy" if self.use_dummy else "api",
                "attempts": attempts,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
        self.event_logger.log(
            logger=logger,
            event="model.call.failed",
            level=logging.ERROR,
            **fields,
        )

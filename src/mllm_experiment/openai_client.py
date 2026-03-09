# src/mllm_experiment/openai_client.py
# wrapper around OpenAI Responses API
from __future__ import annotations
import json
import os
import logging
import time
from dataclasses import dataclass, field
from typing import Any
from openai import OpenAI
from pyparsing import Optional

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
        max_output_tokens: Maximum number of tokens in the completion.
        use_dummy: If true, skip real API calls and return synthetic
            responses for dry-run debugging.
    """

    model: str
    temperature: Optional[float] = None
    max_completion_tokens: int = None
    use_dummy: bool = False
    client: Optional[Any] = field(init=False, default=None)
    
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
    ) -> ModelResponse:
        """Call the OpenAI chat API or generate a dummy response.

        Args:
            messages: Full chat history including system, user and
                assistant messages.
            verbose: Flag that controls extra debug printing.

        Returns:
            ModelResponse with raw text, parsed JSON and usage metadata.
        """
        if self.client is None and not self.use_dummy:
            logger.critical("OpenAI client is not initialised.")
            raise RuntimeError("OpenAI client is not initialised.")
        
        if self.use_dummy:
            return self._dummy_call(messages, verbose=verbose)

        logging.debug(f"[openai_client] sending {len(messages)} messages to model {self.model}")

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
        latency_s = (end - start) * 1000000.0
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

        logger.debug(f"[openai_client] latency: {latency_s:.1f}s")

        return ModelResponse(
            raw_text=raw_text,
            parsed_json=parsed,
            latency_ms=latency_s,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def _dummy_call(
        self,
        messages: list[dict[str, Any]],
        verbose: bool = False,
    ) -> ModelResponse:
        """Generate a synthetic response for dry-run debugging.

        This method inspects the last user message to detect whether the
        query corresponds to an exam phase or a teaching phase.

        For exam phases, it returns a JSON object of the form
        {"answers": [{"item_id": "...", "guess": "normal"}, ...]} where
        the item identifiers are extracted from the prompt.

        For teaching phases, it returns {"acknowledged": true}.

        Args:
            messages: Full chat history including system, user and
                assistant messages.
            verbose: Flag that controls extra debug printing.

        Returns:
            ModelResponse with synthetic JSON and zeroed usage metrics.
        """
        logger.debug("[openai_client] dummy mode: generating synthetic response")

        if not messages:
            raw_text = '{"acknowledged": true}'
            parsed = json.loads(raw_text)
            return ModelResponse(
                raw_text=raw_text,
                parsed_json=parsed,
                latency_ms=0.0,
                prompt_tokens=None,
                completion_tokens=None,
            )

        last = messages[-1]
        content = last.get("content", [])
        exam_prompt = False
        item_ids: list[str] = []

        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                text = str(part.get("text", ""))
                if "answer with a JSON object of the form" in text:
                    exam_prompt = True
                if "item_id=" in text:
                    start = text.find("item_id=") + len("item_id=")
                    end = text.find(")", start)
                    if end == -1:
                        end = len(text)
                    item_id = text[start:end].strip()
                    if item_id:
                        item_ids.append(item_id)

        if exam_prompt and item_ids:
            answers = [{"item_id": iid, "guess": "normal"} for iid in item_ids]
            raw_text = json.dumps({"answers": answers})
            parsed = {"answers": answers}
        else:
            raw_text = '{"acknowledged": true}'
            parsed = {"acknowledged": True}

        if exam_prompt:
            logger.debug(f"[openai_client] dummy exam response with {len(item_ids)} answers")
        else:
            logger.debug("[openai_client] dummy teaching acknowledgement")

        return ModelResponse(
            raw_text=raw_text,
            parsed_json=parsed,
            latency_ms=0.0,
            prompt_tokens=None,
            completion_tokens=None,
        )

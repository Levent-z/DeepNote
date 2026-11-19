"""OpenAI-compatible generation client for separately deployed models."""

import logging
import os
import time
import uuid
from urllib.parse import urlsplit

from prediction_report import question_id_context
from pathlib import Path

logger = logging.getLogger(__name__)


class GenerationAPI:
    def __init__(self, settings, model=None):
        import openai
        OpenAI = openai.OpenAI
        self.connection_error = getattr(openai, "APIConnectionError", ConnectionError)
        if not settings.get("base_url"):
            raise ValueError("Set generation.base_url to the Jiugе service URL ending in /v1")
        self.model = model or settings.get("model")
        if not self.model:
            raise ValueError("Set generation.model or pass --model with the served model name")
        self.temperature = float(settings.get("temperature", 0.6))
        self.max_attempts = int(settings.get("max_attempts", 3))
        self.retry_delay = float(settings.get("retry_delay", 2))
        if self.max_attempts < 1 or self.retry_delay < 0:
            raise ValueError("max_attempts must be positive and retry_delay nonnegative")
        self.enable_thinking = settings.get("enable_thinking")
        if self.enable_thinking is not None and not isinstance(self.enable_thinking, bool):
            raise ValueError("generation.enable_thinking must be true, false, or null")
        self.strip_thinking = settings.get("strip_thinking", True)
        self.timeout = float(settings.get("timeout", 120))
        endpoint = urlsplit(settings["base_url"])
        self.endpoint = f"{endpoint.scheme}://{endpoint.hostname}:{endpoint.port}" if endpoint.port else f"{endpoint.scheme}://{endpoint.hostname}"
        self.client = OpenAI(
            base_url=settings["base_url"],
            api_key=os.environ.get("DEEPNOTE_GENERATION_API_KEY") or settings.get("api_key") or "EMPTY",
            timeout=self.timeout,
            max_retries=0,  # Explicit loop below caps TOTAL attempts, not retries.
        )

    def __call__(self, prompt_file, variables):
        prompt = Path(prompt_file).read_text(encoding="utf-8").format(**variables)
        request = {
            "model": self.model, "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        if self.enable_thinking is not None:
            request["extra_body"] = {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
        request_id = uuid.uuid4().hex[:8]
        question_id = question_id_context.get()
        for attempt in range(self.max_attempts):
            started = time.perf_counter()
            logger.info("Generation request: question_id=%s request_id=%s stage=%s endpoint=%s model=%s prompt_chars=%d attempt=%d/%d timeout=%.1fs enable_thinking=%s",
                        question_id, request_id, Path(prompt_file).name, self.endpoint, self.model, len(prompt),
                        attempt + 1, self.max_attempts, self.timeout, self.enable_thinking)
            try:
                response = self.client.chat.completions.create(**request)
                logger.info("Generation response: question_id=%s request_id=%s stage=%s elapsed=%.1fs finish_reason=%s usage=%s",
                            question_id, request_id, Path(prompt_file).name, time.perf_counter() - started,
                            getattr(response.choices[0], "finish_reason", None), getattr(response, "usage", None))
                break
            except Exception as exc:
                logger.warning("Generation request failed: question_id=%s request_id=%s stage=%s attempt=%d/%d elapsed=%.1fs error=%s: %s cause=%s",
                               question_id, request_id, Path(prompt_file).name, attempt + 1, self.max_attempts,
                               time.perf_counter() - started, type(exc).__name__, exc,
                               type(exc.__cause__).__name__ if exc.__cause__ else "none")
                status = getattr(exc, "status_code", None)
                retryable = isinstance(exc, (self.connection_error, ConnectionError, TimeoutError)) or (
                    isinstance(status, int) and (status in (408, 409, 429) or status >= 500)
                )
                if not retryable or attempt + 1 >= self.max_attempts:
                    raise
                logger.warning("Generation retry scheduled: question_id=%s request_id=%s delay=%.1fs", question_id, request_id, self.retry_delay)
                time.sleep(self.retry_delay)
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Generation service returned an empty answer")
        if self.strip_thinking and "</think>" in content:
            content = content.rsplit("</think>", 1)[1].strip()
        if not content.strip():
            raise ValueError("Generation service returned thinking without an answer")
        return content

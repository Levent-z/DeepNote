"""OpenAI-compatible generation client for separately deployed models."""

import os
import time
from pathlib import Path


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
        self.client = OpenAI(
            base_url=settings["base_url"],
            api_key=os.environ.get("DEEPNOTE_GENERATION_API_KEY") or settings.get("api_key") or "EMPTY",
            timeout=float(settings.get("timeout", 120)),
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
        for attempt in range(self.max_attempts):
            try:
                response = self.client.chat.completions.create(**request)
                break
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                retryable = isinstance(exc, (self.connection_error, ConnectionError, TimeoutError)) or (
                    isinstance(status, int) and (status in (408, 409, 429) or status >= 500)
                )
                if not retryable or attempt + 1 >= self.max_attempts:
                    raise
                time.sleep(self.retry_delay)
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Generation service returned an empty answer")
        if self.strip_thinking and "</think>" in content:
            content = content.rsplit("</think>", 1)[1].strip()
        if not content.strip():
            raise ValueError("Generation service returned thinking without an answer")
        return content

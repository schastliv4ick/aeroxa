"""OpenAI-compatible chat client.

Deliberately protocol-level rather than vendor-level: the same code talks to a
self-hosted vLLM serving Qwen3, to Ollama, or to a hosted gateway. Production must point
at a self-hosted endpoint inside the perimeter (ТЗ slide 10); any external endpoint is a
development convenience only and must never see client data.
"""

from __future__ import annotations

import logging

from openai import AsyncOpenAI, APIError

from app.config import get_settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings
        self._client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout_s,
        )

    @property
    def model(self) -> str:
        return self._settings.llm_model

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self._settings.llm_model,
                messages=messages,  # type: ignore[arg-type]
                temperature=(
                    self._settings.llm_temperature if temperature is None else temperature
                ),
                max_tokens=max_tokens or self._settings.llm_max_tokens,
            )
        except APIError as error:
            raise LLMError(f"LLM request failed: {error}") from error

        content = response.choices[0].message.content
        if not content:
            raise LLMError("LLM returned an empty response")
        return content.strip()


_client: LLMClient | None = None


def get_llm() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client

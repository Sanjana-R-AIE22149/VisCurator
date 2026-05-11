"""
VisCurator / CVAgent — NVIDIA NIM Client
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Thin async wrapper around the OpenAI Python SDK configured for
NVIDIA's NIM inference endpoint.
"""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncIterator

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk, ChatCompletion

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "meta/llama-3.1-70b-instruct"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 4096


class NIMClient:
    """Async client for NVIDIA NIM's OpenAI-compatible chat API.

    Usage::

        client = NIMClient()
        # Non-streaming
        reply = await client.chat(messages=[...])

        # Streaming
        async for chunk in client.chat_stream(messages=[...]):
            print(chunk)
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = NIM_BASE_URL,
        model: str = DEFAULT_MODEL,
    ) -> None:
        resolved_key = api_key or os.getenv("NVIDIA_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "NVIDIA_API_KEY is not set. "
                "Pass it explicitly or set the environment variable."
            )

        self._model = model
        self._client = AsyncOpenAI(
            api_key=resolved_key,
            base_url=base_url,
        )
        logger.info(
            "NIMClient initialised — model=%s  base_url=%s",
            self._model,
            base_url,
        )

    # ── Public API ───────────────────────────────────────────

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Send a chat completion request.

        Parameters
        ----------
        messages:
            OpenAI-style message list (role + content dicts).
        temperature:
            Sampling temperature (lower → more deterministic).
        max_tokens:
            Maximum tokens in the response.
        stream:
            If ``True``, returns an async iterator of ``ChatCompletionChunk``.
        **kwargs:
            Forwarded to the underlying ``client.chat.completions.create`` call.

        Returns
        -------
        ``ChatCompletion`` when ``stream=False``, or an async iterator of
        ``ChatCompletionChunk`` when ``stream=True``.
        """
        logger.debug(
            "NIM chat request — messages=%d  stream=%s  model=%s",
            len(messages),
            stream,
            self._model,
        )

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            **kwargs,
        )

        return response

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Convenience wrapper that yields decoded content strings.

        Internally calls :meth:`chat` with ``stream=True`` and extracts
        the ``delta.content`` from each chunk.
        """
        stream = await self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )

        async for chunk in stream:  # type: ignore[union-attr]
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    # ── Lifecycle ────────────────────────────────────────────

    async def close(self) -> None:
        """Gracefully close the underlying HTTP transport."""
        await self._client.close()
        logger.info("NIMClient connection closed.")

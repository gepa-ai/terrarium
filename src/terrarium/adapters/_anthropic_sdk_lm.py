"""Direct Anthropic SDK reflection LM — bypasses litellm.

Use when the litellm + httpx path wedges on long extended-thinking responses.
Implements GEPA's ``LanguageModel`` protocol and exposes ``total_cost`` for
:class:`gepa.utils.stop_condition.MaxReflectionCostStopper`.

Install: ``pip install terrarium[anthropic_sdk]``. The import is deferred to
``AnthropicSdkLM.__init__`` so ``terrarium`` stays importable without it.
"""

from __future__ import annotations

import threading
from typing import Any

# (input_per_mtok, output_per_mtok) USD; overrides accepted as ctor kwargs.
_PRICE_TABLE: dict[str, tuple[float, float]] = {
    "claude-opus-4-7":              (15.0, 75.0),
    "claude-opus-4-5":              (15.0, 75.0),
    "claude-opus-4-5-20251101":     (15.0, 75.0),
    "claude-sonnet-4-6":            ( 3.0, 15.0),
    "claude-sonnet-4-5":            ( 3.0, 15.0),
    "claude-haiku-4-5":             ( 1.0,  5.0),
    "claude-haiku-4-5-20251001":    ( 1.0,  5.0),
}


def _lookup_price(model: str) -> tuple[float, float]:
    if model in _PRICE_TABLE:
        return _PRICE_TABLE[model]
    for known, price in _PRICE_TABLE.items():
        if known in model:
            return price
    return (3.0, 15.0)  # default to Sonnet pricing for unknown models


class AnthropicSdkLM:
    """Reflection LM backed by the official Anthropic SDK.

    Args:
        model: Anthropic model id (e.g. ``claude-sonnet-4-6``). The
            ``anthropic/`` litellm prefix is stripped if present.
        max_tokens: Cap on response tokens. Must exceed
            ``max_thinking_tokens`` when thinking is enabled.
        max_thinking_tokens: When set, enables extended thinking with this
            budget. ``None`` disables thinking.
        timeout: Per-request timeout (seconds). Anthropic SDK uses httpx
            internally; this is the hard cap on a single ``messages.create``.
        max_retries: SDK-level retries on transient 5xx / network errors.
        input_price_per_mtok / output_price_per_mtok: USD per 1M tokens.
            Override the price table; useful for new model ids.
    """

    def __init__(
        self,
        model: str,
        *,
        max_tokens: int = 8000,
        max_thinking_tokens: int | None = None,
        timeout: float = 600.0,
        max_retries: int = 2,
        input_price_per_mtok: float | None = None,
        output_price_per_mtok: float | None = None,
    ) -> None:
        from anthropic import Anthropic

        self.model = model[len("anthropic/"):] if model.startswith("anthropic/") else model
        self.max_tokens = max_tokens
        self.max_thinking_tokens = max_thinking_tokens
        if max_thinking_tokens is not None and max_tokens <= max_thinking_tokens:
            self.max_tokens = max_thinking_tokens + 1024

        in_price, out_price = _lookup_price(self.model)
        self._in_price = input_price_per_mtok if input_price_per_mtok is not None else in_price
        self._out_price = output_price_per_mtok if output_price_per_mtok is not None else out_price

        self._client = Anthropic(timeout=timeout, max_retries=max_retries)
        self._lock = threading.Lock()
        self.total_cost: float = 0.0
        self.total_tokens_in: int = 0
        self.total_tokens_out: int = 0
        self.call_count: int = 0

    def _build_messages(self, prompt: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        # role=system is stripped here and re-attached via the ``system`` kwarg
        # in ``__call__`` — the SDK rejects it inside ``messages``.
        msgs: list[dict[str, Any]] = []
        for m in prompt:
            role = m.get("role", "user")
            if role == "system":
                continue
            msgs.append({"role": role, "content": m.get("content", "")})
        return msgs

    def _extract_system(self, prompt: str | list[dict[str, Any]]) -> str | None:
        if isinstance(prompt, str):
            return None
        sys_parts = [m.get("content", "") for m in prompt if m.get("role") == "system"]
        return "\n\n".join(p for p in sys_parts if p) or None

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self._build_messages(prompt),
        }
        sys_prompt = self._extract_system(prompt)
        if sys_prompt:
            kwargs["system"] = sys_prompt
        if self.max_thinking_tokens is not None:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": self.max_thinking_tokens}
            kwargs["temperature"] = 1.0  # extended thinking requires temperature=1

        response = self._client.messages.create(**kwargs)

        text_parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
        text = "\n".join(text_parts)

        usage = response.usage
        in_toks = getattr(usage, "input_tokens", 0) or 0
        out_toks = getattr(usage, "output_tokens", 0) or 0
        cost = (in_toks * self._in_price + out_toks * self._out_price) / 1_000_000

        with self._lock:
            self.total_tokens_in += in_toks
            self.total_tokens_out += out_toks
            self.total_cost += cost
            self.call_count += 1

        return text

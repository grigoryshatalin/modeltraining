"""OpenAI decision-model adapter — a working second provider.

Demonstrates that the `TradingModel` interface is genuinely provider-agnostic.
The `openai` package is an optional dependency: `pip install '.[openai]'`.
"""

from __future__ import annotations

from ..context import MarketContext
from .base import ModelDecisionError, TradingModel
from .prompt import SYSTEM_PROMPT, render_context
from .schema import TradeDecision


class OpenAIModel(TradingModel):
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        client: object | None = None,
    ) -> None:
        if not model:
            raise ValueError("OPENAI_MODEL must be set to a structured-output-capable model.")
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - optional dep
                raise ImportError(
                    "The 'openai' package is not installed. Install it with: "
                    "pip install '.[openai]'"
                ) from exc
            client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self._client = client
        self._model = model

    def decide(self, context: MarketContext) -> TradeDecision:
        try:
            completion = self._client.beta.chat.completions.parse(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": render_context(context)},
                ],
                response_format=TradeDecision,
            )
        except Exception as exc:  # pragma: no cover - network path
            raise ModelDecisionError(
                f"OpenAI request failed for {context.symbol}: {exc}"
            ) from exc

        decision = completion.choices[0].message.parsed
        if decision is None:
            raise ModelDecisionError(
                f"OpenAI returned no parseable decision for {context.symbol}."
            )
        return decision

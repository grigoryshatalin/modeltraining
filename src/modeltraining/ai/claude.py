"""Claude (Anthropic) decision-model adapter — the reference implementation.

Uses the Messages API structured-output helper (`messages.parse`) so the model
returns a validated `TradeDecision` directly, with adaptive thinking enabled.
"""

from __future__ import annotations

import anthropic

from ..context import MarketContext
from .base import ModelDecisionError, TradingModel
from .prompt import SYSTEM_PROMPT, render_context
from .schema import TradeDecision


class ClaudeModel(TradingModel):
    def __init__(
        self,
        model: str = "claude-opus-5",
        *,
        api_key: str | None = None,
        client: anthropic.Anthropic | None = None,
        max_tokens: int = 8000,
    ) -> None:
        # A bare Anthropic() reads ANTHROPIC_API_KEY (or an `ant auth login`
        # profile) from the environment; pass api_key only to override.
        self._client = client or (
            anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        )
        self._model = model
        self._max_tokens = max_tokens

    def decide(self, context: MarketContext) -> TradeDecision:
        try:
            response = self._client.messages.parse(
                model=self._model,
                max_tokens=self._max_tokens,
                thinking={"type": "adaptive"},
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": render_context(context)}],
                output_format=TradeDecision,
            )
        except anthropic.APIError as exc:  # pragma: no cover - network path
            raise ModelDecisionError(f"Claude request failed for {context.symbol}: {exc}") from exc

        decision = response.parsed_output
        if decision is None:
            raise ModelDecisionError(
                f"Claude returned no parseable decision for {context.symbol} "
                f"(stop_reason={response.stop_reason})."
            )
        return decision

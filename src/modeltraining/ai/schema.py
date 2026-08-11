"""Structured output schema the AI model must return for each symbol."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class TradeAction(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class TradeDecision(BaseModel):
    """A single trading decision for one symbol.

    The symbol itself is supplied by the engine (not echoed by the model) to
    avoid any chance of a mismatch. Numeric fields are normalized on the way in
    so a slightly out-of-range model response never crashes a run.
    """

    action: TradeAction = Field(description="buy, sell, or hold")
    quantity: int = Field(
        description="Whole shares to trade. Use 0 when action is hold.",
    )
    confidence: float = Field(
        description="Confidence in this decision, from 0.0 (none) to 1.0 (certain).",
    )
    rationale: str = Field(
        description="One or two sentences justifying the decision from the data.",
    )

    @field_validator("quantity")
    @classmethod
    def _non_negative_qty(cls, v: int) -> int:
        return abs(int(v))

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

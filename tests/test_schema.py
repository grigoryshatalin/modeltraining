from modeltraining.ai.schema import TradeAction, TradeDecision


def test_action_parses_from_string():
    d = TradeDecision(action="buy", quantity=3, confidence=0.7, rationale="x")
    assert d.action is TradeAction.BUY


def test_quantity_is_non_negative():
    d = TradeDecision(action="sell", quantity=-5, confidence=0.5, rationale="x")
    assert d.quantity == 5


def test_confidence_is_clamped():
    high = TradeDecision(action="hold", quantity=0, confidence=1.5, rationale="x")
    low = TradeDecision(action="hold", quantity=0, confidence=-0.4, rationale="x")
    assert high.confidence == 1.0
    assert low.confidence == 0.0

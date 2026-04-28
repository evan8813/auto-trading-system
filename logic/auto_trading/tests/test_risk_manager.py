"""
test_risk_manager.py
────────────────────
步驟 5：驗證風控計算邏輯。

測試項目：
  1. risk_amount()          ── equity × risk_pct
  2. position_size_lots()   ── 根據 ATR 計算張數，ATR=0 回傳 0
  3. transaction_cost()     ── buy / sell 成本計算（賣出含證交稅）
"""

import pytest
from config import TradingConfig
from risk_manager import RiskManager


def make_rm(**kwargs) -> RiskManager:
    """建立測試用 RiskManager"""
    cfg = TradingConfig(**kwargs)
    return RiskManager(cfg)


class TestRiskAmount:
    """risk_amount(equity)"""

    def test_basic_calculation(self):
        rm = make_rm(risk_pct=0.002)
        assert rm.risk_amount(1_000_000) == pytest.approx(2000.0)

    def test_higher_equity_higher_risk(self):
        rm = make_rm(risk_pct=0.002)
        assert rm.risk_amount(2_000_000) == pytest.approx(4000.0)

    def test_zero_equity(self):
        rm = make_rm(risk_pct=0.002)
        assert rm.risk_amount(0) == pytest.approx(0.0)

    def test_risk_pct_1_percent(self):
        rm = make_rm(risk_pct=0.01)
        assert rm.risk_amount(500_000) == pytest.approx(5000.0)


class TestPositionSizeLots:
    """position_size_lots(equity, atr)"""

    def test_basic_lot_calculation(self):
        """
        equity=1,000,000, risk_pct=0.002, ATR=2.0, point_value=1
        risk_amount = 2000
        raw_shares  = 2000 / (2.0 × 1.0) = 1000
        lots        = int(1000 / 1000) = 1
        """
        rm = make_rm(risk_pct=0.002, point_value=1.0)
        assert rm.position_size_lots(equity=1_000_000, atr=2.0) == 1

    def test_larger_atr_fewer_lots(self):
        """ATR 越大，張數越少"""
        rm = make_rm(risk_pct=0.002, point_value=1.0)
        lots_low_atr  = rm.position_size_lots(1_000_000, atr=1.0)
        lots_high_atr = rm.position_size_lots(1_000_000, atr=10.0)
        assert lots_low_atr > lots_high_atr

    def test_zero_atr_returns_zero(self):
        rm = make_rm()
        assert rm.position_size_lots(1_000_000, atr=0.0) == 0

    def test_negative_atr_returns_zero(self):
        rm = make_rm()
        assert rm.position_size_lots(1_000_000, atr=-5.0) == 0

    def test_very_large_atr_returns_zero(self):
        """ATR 極大時，lots 計算結果 < 1 → 回傳 0"""
        rm = make_rm(risk_pct=0.002, initial_equity=1_000_000)
        # risk_amount = 2000; raw_shares = 2000/10000 = 0.2; lots=0
        assert rm.position_size_lots(1_000_000, atr=10_000.0) == 0

    def test_result_is_integer(self):
        rm = make_rm()
        result = rm.position_size_lots(1_000_000, atr=2.0)
        assert isinstance(result, int)


class TestTransactionCost:
    """transaction_cost(price, shares, side)"""

    def test_buy_side_no_tax(self):
        """買方不含證交稅"""
        rm = make_rm(commission_rate=0.001425, slippage=0.001, transaction_tax=0.003)
        # notional = 100 × 1000 = 100,000
        # commission = 100,000 × 0.001425 = 142.5
        # slippage   = 100,000 × 0.001    = 100.0
        # tax        = 0（買方）
        # total      = 242.5
        cost = rm.transaction_cost(price=100.0, shares=1000, side="buy")
        assert cost == pytest.approx(242.5)

    def test_sell_side_includes_tax(self):
        """賣方含證交稅"""
        rm = make_rm(commission_rate=0.001425, slippage=0.001, transaction_tax=0.003)
        # notional = 100 × 1000 = 100,000
        # commission = 142.5
        # slippage   = 100.0
        # tax        = 100,000 × 0.003 = 300.0
        # total      = 542.5
        cost = rm.transaction_cost(price=100.0, shares=1000, side="sell")
        assert cost == pytest.approx(542.5)

    def test_sell_cost_greater_than_buy(self):
        rm = make_rm()
        cost_buy  = rm.transaction_cost(100.0, 1000, "buy")
        cost_sell = rm.transaction_cost(100.0, 1000, "sell")
        assert cost_sell > cost_buy

    def test_zero_shares(self):
        rm = make_rm()
        assert rm.transaction_cost(100.0, 0, "buy") == pytest.approx(0.0)
        assert rm.transaction_cost(100.0, 0, "sell") == pytest.approx(0.0)

    def test_cost_proportional_to_price(self):
        """成本應與成交價格成正比"""
        rm = make_rm()
        cost_100 = rm.transaction_cost(100.0, 1000, "buy")
        cost_200 = rm.transaction_cost(200.0, 1000, "buy")
        assert cost_200 == pytest.approx(cost_100 * 2, rel=1e-6)

    def test_cost_proportional_to_shares(self):
        """成本應與股數成正比"""
        rm = make_rm()
        cost_1000 = rm.transaction_cost(100.0, 1000, "sell")
        cost_2000 = rm.transaction_cost(100.0, 2000, "sell")
        assert cost_2000 == pytest.approx(cost_1000 * 2, rel=1e-6)

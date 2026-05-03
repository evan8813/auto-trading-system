"""
test_risk_manager.py
────────────────────
步驟 5：驗證風控計算邏輯。

測試項目：
  1. risk_amount()          ── min(equity × risk_pct, max_risk_amount)
  2. position_size_lots()   ── 主公式（完整停損距離）+ 容忍區（最低 1 張）
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
    """risk_amount(equity) = min(equity × risk_pct, max_risk_amount)"""

    def test_basic_calculation(self):
        """equity 未達上限時：risk = equity × risk_pct"""
        rm = make_rm(risk_pct=0.002, max_risk_amount=2_500.0)
        # 1,000,000 × 0.002 = 2,000 < 2,500 → 不觸發上限
        assert rm.risk_amount(1_000_000) == pytest.approx(2_000.0)

    def test_higher_equity_higher_risk(self):
        """equity 都在上限內時，equity 越大 risk 越大"""
        rm = make_rm(risk_pct=0.002, max_risk_amount=2_500.0)
        # 100,000 × 0.002 = 200；200,000 × 0.002 = 400，兩者都 < 2,500
        assert rm.risk_amount(200_000) > rm.risk_amount(100_000)

    def test_zero_equity(self):
        rm = make_rm(risk_pct=0.002)
        assert rm.risk_amount(0) == pytest.approx(0.0)

    def test_risk_pct_1_percent(self):
        """risk_pct=1%，equity 未達上限"""
        rm = make_rm(risk_pct=0.01, max_risk_amount=2_500.0)
        # 200,000 × 0.01 = 2,000 < 2,500 → 不觸發上限
        assert rm.risk_amount(200_000) == pytest.approx(2_000.0)

    def test_risk_amount_capped_at_max(self):
        """equity × risk_pct 超過 max_risk_amount 時，以上限為準"""
        rm = make_rm(risk_pct=0.01, max_risk_amount=2_500.0)
        # 1,000,000 × 0.01 = 10,000 > 2,500 → 上限 2,500
        assert rm.risk_amount(1_000_000) == pytest.approx(2_500.0)

    def test_cap_kicks_in_at_250k(self):
        """risk_pct=1%, max_risk_amount=2,500 → equity=250,000 剛好等於上限"""
        rm = make_rm(risk_pct=0.01, max_risk_amount=2_500.0)
        assert rm.risk_amount(250_000) == pytest.approx(2_500.0)


class TestPositionSizeLots:
    """position_size_lots(equity, atr)"""

    def test_main_formula_returns_correct_lots(self):
        """
        主公式正常運作：risk / (atr_multiplier × ATR × 1000) ≥ 1
        equity=500,000, risk_pct=0.002, max_risk=2500, ATR=0.3, atr_mult=3
        risk = min(1000, 2500) = 1000
        lots = int(1000 / (3 × 0.3 × 1000)) = int(1000/900) = 1
        """
        rm = make_rm(risk_pct=0.002, max_risk_amount=2_500.0,
                     atr_multiplier=3.0, point_value=1.0)
        assert rm.position_size_lots(equity=500_000, atr=0.3) == 1

    def test_main_formula_multiple_lots(self):
        """主公式給出多張：risk 足夠大，ATR 夠小"""
        rm = make_rm(risk_pct=0.002, max_risk_amount=2_500.0,
                     atr_multiplier=3.0, point_value=1.0)
        # risk=2000, lots=int(2000/(3×0.2×1000))=int(2000/600)=3
        assert rm.position_size_lots(equity=1_000_000, atr=0.2) == 3

    def test_larger_atr_fewer_lots(self):
        """ATR 越大，張數越少"""
        rm = make_rm(risk_pct=0.01, point_value=1.0)
        lots_low  = rm.position_size_lots(1_000_000, atr=0.1)
        lots_high = rm.position_size_lots(1_000_000, atr=1.0)
        assert lots_low > lots_high

    def test_zero_atr_returns_zero(self):
        rm = make_rm()
        assert rm.position_size_lots(1_000_000, atr=0.0) == 0

    def test_negative_atr_returns_zero(self):
        rm = make_rm()
        assert rm.position_size_lots(1_000_000, atr=-5.0) == 0

    def test_very_large_atr_exceeds_tolerance_returns_zero(self):
        """ATR 極大時，主公式給 0，1 張風險也超過容忍區 → 回傳 0"""
        rm = make_rm(risk_pct=0.01, max_risk_amount=2_500.0,
                     atr_multiplier=3.0, point_value=1.0)
        # 1-lot risk = 3 × 10,000 × 1,000 = 30,000,000 >> equity × 2%
        assert rm.position_size_lots(180_000, atr=10_000.0) == 0

    def test_result_is_integer(self):
        rm = make_rm()
        result = rm.position_size_lots(1_000_000, atr=0.3)
        assert isinstance(result, int)

    def test_tolerance_zone_returns_one_lot(self):
        """
        主公式給 0，但 1 張風險 ≤ equity × 2% → 進 1 張
        equity=180,000, risk_pct=0.01, ATR=0.8, atr_mult=3
        risk=1800, lots=int(1800/2400)=0
        1-lot risk=2400, equity×2%=3600 → 2400 ≤ 3600 → 1
        """
        rm = make_rm(risk_pct=0.01, max_risk_amount=2_500.0,
                     atr_multiplier=3.0, point_value=1.0)
        assert rm.position_size_lots(equity=180_000, atr=0.8) == 1

    def test_exceeds_tolerance_returns_zero(self):
        """
        主公式給 0，且 1 張風險 > equity × 2% → 跳過
        equity=180,000, risk_pct=0.01, ATR=2.0, atr_mult=3
        risk=1800, lots=int(1800/6000)=0
        1-lot risk=6000, equity×2%=3600 → 6000 > 3600 → 0
        """
        rm = make_rm(risk_pct=0.01, max_risk_amount=2_500.0,
                     atr_multiplier=3.0, point_value=1.0)
        assert rm.position_size_lots(equity=180_000, atr=2.0) == 0

    def test_tolerance_fades_as_equity_grows(self):
        """
        equity 成長後，主公式自然給出 ≥ 1 張，不需容忍區
        equity=360,000（原本 180K 的兩倍），同樣 ATR=0.8
        risk=min(3600,2500)=2500, lots=int(2500/2400)=1 → 主公式直接給 1
        """
        rm = make_rm(risk_pct=0.01, max_risk_amount=2_500.0,
                     atr_multiplier=3.0, point_value=1.0)
        assert rm.position_size_lots(equity=360_000, atr=0.8) == 1


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
        rm = make_rm()
        cost_100 = rm.transaction_cost(100.0, 1000, "buy")
        cost_200 = rm.transaction_cost(200.0, 1000, "buy")
        assert cost_200 == pytest.approx(cost_100 * 2, rel=1e-6)

    def test_cost_proportional_to_shares(self):
        rm = make_rm()
        cost_1000 = rm.transaction_cost(100.0, 1000, "sell")
        cost_2000 = rm.transaction_cost(100.0, 2000, "sell")
        assert cost_2000 == pytest.approx(cost_1000 * 2, rel=1e-6)

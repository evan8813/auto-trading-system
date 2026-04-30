"""
test_config.py
──────────────
步驟 1：驗證 TradingConfig 的預設值與自訂值是否正確。

測試項目：
  1. 所有預設值是否符合規格
  2. 自訂參數是否能正確覆蓋
  3. 參數型別是否正確（int / float / str）
  4. 組合驗算：risk_pct × initial_equity 邏輯
"""

import pytest
from config import TradingConfig


class TestTradingConfigDefaults:
    """驗證預設值"""

    def setup_method(self):
        self.cfg = TradingConfig()

    # ── 資金設定 ──
    def test_initial_equity_default(self):
        assert self.cfg.initial_equity == 180_000.0

    def test_risk_pct_default(self):
        assert self.cfg.risk_pct == 0.01

    # ── 交易成本 ──
    def test_commission_rate_default(self):
        assert self.cfg.commission_rate == 0.001425

    def test_transaction_tax_default(self):
        assert self.cfg.transaction_tax == 0.003

    def test_slippage_default(self):
        assert self.cfg.slippage == 0.001

    # ── 指標參數 ──
    def test_breakout_window_default(self):
        assert self.cfg.breakout_window == 50

    def test_ma_fast_default(self):
        assert self.cfg.ma_fast == 50

    def test_ma_slow_default(self):
        assert self.cfg.ma_slow == 100

    def test_atr_period_default(self):
        assert self.cfg.atr_period == 100

    def test_atr_multiplier_default(self):
        assert self.cfg.atr_multiplier == 3.0

    def test_week52_default(self):
        assert self.cfg.week52 == 252

    # ── 篩選條件 ──
    def test_min_avg_amount_default(self):
        assert self.cfg.min_avg_amount == 5_000_000

    # ── 部位限制 ──
    def test_max_positions_default(self):
        assert self.cfg.max_positions == 5

    def test_max_trade_cost_default(self):
        assert self.cfg.max_trade_cost == 36_000.0

    def test_point_value_default(self):
        assert self.cfg.point_value == 1.0

    # ── 回測區間 ──
    def test_backtest_start_default(self):
        assert self.cfg.backtest_start == "2010-01-01"

    def test_backtest_end_default(self):
        assert self.cfg.backtest_end == "2023-12-29"


class TestTradingConfigCustom:
    """驗證自訂參數覆蓋"""

    def test_custom_initial_equity(self):
        cfg = TradingConfig(initial_equity=2_000_000)
        assert cfg.initial_equity == 2_000_000.0

    def test_custom_risk_pct(self):
        cfg = TradingConfig(risk_pct=0.005)
        assert cfg.risk_pct == 0.005

    def test_custom_max_positions(self):
        cfg = TradingConfig(max_positions=20)
        assert cfg.max_positions == 20

    def test_custom_backtest_start(self):
        cfg = TradingConfig(backtest_start="2015-01-01")
        assert cfg.backtest_start == "2015-01-01"

    def test_custom_backtest_end(self):
        cfg = TradingConfig(backtest_end="2020-12-31")
        assert cfg.backtest_end == "2020-12-31"

    def test_multiple_custom_params(self):
        cfg = TradingConfig(
            initial_equity=500_000,
            risk_pct=0.001,
            max_positions=5,
            atr_multiplier=2.5,
        )
        assert cfg.initial_equity == 500_000.0
        assert cfg.risk_pct == 0.001
        assert cfg.max_positions == 5
        assert cfg.atr_multiplier == 2.5


class TestTradingConfigTypes:
    """驗證欄位型別"""

    def setup_method(self):
        self.cfg = TradingConfig()

    # ── float 欄位 ──
    def test_initial_equity_is_float(self):
        assert isinstance(self.cfg.initial_equity, float)

    def test_risk_pct_is_float(self):
        assert isinstance(self.cfg.risk_pct, float)

    def test_commission_rate_is_float(self):
        assert isinstance(self.cfg.commission_rate, float)

    def test_transaction_tax_is_float(self):
        assert isinstance(self.cfg.transaction_tax, float)

    def test_slippage_is_float(self):
        assert isinstance(self.cfg.slippage, float)

    def test_atr_multiplier_is_float(self):
        assert isinstance(self.cfg.atr_multiplier, float)

    def test_min_avg_amount_is_float(self):
        assert isinstance(self.cfg.min_avg_amount, float)

    def test_max_trade_cost_is_float(self):
        assert isinstance(self.cfg.max_trade_cost, float)

    def test_point_value_is_float(self):
        assert isinstance(self.cfg.point_value, float)

    # ── int 欄位 ──
    def test_breakout_window_is_int(self):
        assert isinstance(self.cfg.breakout_window, int)

    def test_ma_fast_is_int(self):
        assert isinstance(self.cfg.ma_fast, int)

    def test_ma_slow_is_int(self):
        assert isinstance(self.cfg.ma_slow, int)

    def test_atr_period_is_int(self):
        assert isinstance(self.cfg.atr_period, int)

    def test_week52_is_int(self):
        assert isinstance(self.cfg.week52, int)

    def test_max_positions_is_int(self):
        assert isinstance(self.cfg.max_positions, int)

    # ── str 欄位 ──
    def test_backtest_start_is_str(self):
        assert isinstance(self.cfg.backtest_start, str)

    def test_backtest_end_is_str(self):
        assert isinstance(self.cfg.backtest_end, str)


class TestTradingConfigLogic:
    """驗證參數之間的邏輯關係"""

    def test_ma_fast_less_than_ma_slow(self):
        """快線週期應小於慢線週期"""
        cfg = TradingConfig()
        assert cfg.ma_fast < cfg.ma_slow

    def test_risk_per_trade_calculation(self):
        """每筆風險金額 = cfg.initial_equity × cfg.risk_pct"""
        cfg = TradingConfig(initial_equity=1_000_000, risk_pct=0.002)
        assert cfg.initial_equity * cfg.risk_pct == pytest.approx(2000.0)

    def test_week52_greater_than_ma_slow(self):
        """52週均線週期應大於慢線均線週期"""
        cfg = TradingConfig()
        assert cfg.week52 > cfg.ma_slow

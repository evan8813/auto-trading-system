"""
test_backtester.py
──────────────────
步驟 9：驗證回測引擎的整合流程與績效計算。

測試項目：
  1. run() 使用合成資料 → 回傳 equity_curve / trades / metrics
  2. equity_curve 長度符合預期，equity 為正數
  3. metrics 包含所有必要的 key
  4. _compute_metrics() 手動驗算：total_return / max_drawdown / win_rate
  5. _close_position() 做多損益計算正確
  6. _close_position() 做空損益計算正確
  7. 無交易時 metrics 仍完整（total_trades=0）
"""

import pytest
import numpy as np
import pandas as pd

from config import TradingConfig
from backtester import Backtester
from data_loader import DataLoader
from models import Position


# ── 共用輔助 ──────────────────────────────────

def make_cfg(**kwargs) -> TradingConfig:
    defaults = dict(
        initial_equity=1_000_000.0,
        backtest_start="2020-01-01",
        backtest_end="2021-12-31",
        max_positions=3,
        risk_pct=0.002,
    )
    defaults.update(kwargs)
    return TradingConfig(**defaults)


def make_backtester(**kwargs) -> Backtester:
    return Backtester(make_cfg(**kwargs))


def make_long_position(
    ticker="2330",
    adj_entry=100.0,
    raw_entry=100.0,
    shares=1000,
    trail_high=110.0,
    entry_date="2023-01-10",
    atr=5.0,
) -> Position:
    return Position(
        ticker=ticker,
        direction="long",
        entry_date=pd.Timestamp(entry_date),
        lots=1,
        shares=shares,
        adj_entry_price=adj_entry,
        raw_entry_price=raw_entry,
        trail_high=trail_high,
        trail_low=90.0,
        atr_at_entry=atr,
    )


def make_short_position(
    ticker="2317",
    adj_entry=100.0,
    raw_entry=100.0,
    shares=1000,
    trail_low=90.0,
    entry_date="2023-01-10",
    atr=5.0,
) -> Position:
    return Position(
        ticker=ticker,
        direction="short",
        entry_date=pd.Timestamp(entry_date),
        lots=1,
        shares=shares,
        adj_entry_price=adj_entry,
        raw_entry_price=raw_entry,
        trail_high=110.0,
        trail_low=trail_low,
        atr_at_entry=atr,
    )


class TestBacktesterRun:
    """run() 整合測試（使用合成資料）"""

    def setup_method(self):
        self.cfg  = make_cfg()
        self.bt   = Backtester(self.cfg)
        self.data = DataLoader.generate_synthetic(
            ["2330", "2317", "0050"],
            start=self.cfg.backtest_start,
            end=self.cfg.backtest_end,
            seed=42,
        )

    def test_returns_dict_with_required_keys(self):
        result = self.bt.run(self.data)
        assert "equity_curve" in result
        assert "trades"       in result
        assert "metrics"      in result

    def test_equity_curve_is_dataframe(self):
        result = self.bt.run(self.data)
        assert isinstance(result["equity_curve"], pd.DataFrame)

    def test_equity_curve_has_equity_column(self):
        result = self.bt.run(self.data)
        assert "equity" in result["equity_curve"].columns

    def test_equity_curve_non_negative(self):
        result = self.bt.run(self.data)
        assert (result["equity_curve"]["equity"] > 0).all()

    def test_metrics_contains_all_keys(self):
        result  = self.bt.run(self.data)
        metrics = result["metrics"]
        for key in ["total_return_pct", "cagr_pct", "max_drawdown_pct",
                    "win_rate_pct", "total_trades", "avg_win", "avg_loss",
                    "profit_factor"]:
            assert key in metrics, f"缺少 metrics key：{key}"

    def test_max_drawdown_is_negative_or_zero(self):
        result = self.bt.run(self.data)
        assert result["metrics"]["max_drawdown_pct"] <= 0

    def test_win_rate_between_0_and_100(self):
        result = self.bt.run(self.data)
        wr = result["metrics"]["win_rate_pct"]
        assert 0.0 <= wr <= 100.0

    def test_initial_equity_as_first_equity_value(self):
        result = self.bt.run(self.data)
        first_equity = result["equity_curve"]["equity"].iloc[0]
        assert first_equity == pytest.approx(self.cfg.initial_equity)


class TestComputeMetrics:
    """_compute_metrics() 手動驗算"""

    def setup_method(self):
        self.bt = make_backtester()

    def _run_metrics(self, equity_values, trades):
        dates = pd.bdate_range("2020-01-01", periods=len(equity_values))
        ec    = [{"date": d, "equity": e} for d, e in zip(dates, equity_values)]
        return self.bt._compute_metrics(ec, trades)

    def test_total_return_positive(self):
        m = self._run_metrics([1_000_000, 1_100_000, 1_200_000], [])
        assert m["total_return_pct"] == pytest.approx(20.0, rel=1e-3)

    def test_total_return_negative(self):
        m = self._run_metrics([1_000_000, 950_000, 900_000], [])
        assert m["total_return_pct"] == pytest.approx(-10.0, rel=1e-3)

    def test_max_drawdown_zero_for_monotone_increase(self):
        m = self._run_metrics([1_000_000, 1_100_000, 1_200_000], [])
        assert m["max_drawdown_pct"] == pytest.approx(0.0, abs=1e-6)

    def test_max_drawdown_correct_calculation(self):
        # 100 → 120 → 90 → 105；peak=120 最大回撤 = (90-120)/120 = -25%
        m = self._run_metrics([100_000, 120_000, 90_000, 105_000], [])
        assert m["max_drawdown_pct"] == pytest.approx(-25.0, rel=1e-3)

    def test_zero_trades(self):
        m = self._run_metrics([1_000_000, 1_050_000], [])
        assert m["total_trades"]  == 0
        assert m["win_rate_pct"]  == 0.0
        assert m["profit_factor"] == 0.0

    def test_win_rate_all_wins(self):
        trades = [{"pnl_net": 1000}, {"pnl_net": 2000}, {"pnl_net": 500}]
        m = self._run_metrics([1_000_000, 1_100_000], trades)
        assert m["win_rate_pct"] == pytest.approx(100.0)

    def test_win_rate_half_wins(self):
        trades = [{"pnl_net": 1000}, {"pnl_net": -1000}]
        m = self._run_metrics([1_000_000, 1_100_000], trades)
        assert m["win_rate_pct"] == pytest.approx(50.0)

    def test_profit_factor_calculation(self):
        """profit_factor = sum(wins) / abs(sum(losses))"""
        trades = [
            {"pnl_net": 3000},
            {"pnl_net": -1000},
        ]
        m = self._run_metrics([1_000_000, 1_100_000], trades)
        assert m["profit_factor"] == pytest.approx(3.0)

    def test_total_trades_count(self):
        trades = [{"pnl_net": v} for v in [100, -50, 200, -30, 10]]
        m = self._run_metrics([1_000_000, 1_050_000], trades)
        assert m["total_trades"] == 5


class TestClosePositionLong:
    """_close_position() 做多損益計算"""

    def setup_method(self):
        self.bt = make_backtester()

    def test_long_profit_calculation(self):
        """
        adj_entry=100, exit=110, shares=1000, slippage=0.001
        actual_exit = 110 × (1 - 0.001) = 109.89
        gross_pnl   = (109.89 - 100) × 1000 = 9890
        cost_entry  = 100 × 1000 × (0.001425 + 0.001) = 242.5
        cost_exit   = 109.89 × 1000 × (0.001425 + 0.001 + 0.003) ≈ 710.6
        pnl_net     ≈ 9890 - 242.5 - 710.6 ≈ 8936.9
        """
        pos = make_long_position(adj_entry=100.0, shares=1000)
        trade = self.bt._close_position(
            pos=pos,
            exit_date=pd.Timestamp("2023-02-01"),
            exit_price=110.0,
            reason="signal",
        )
        assert trade["gross_pnl"] > 0
        assert trade["pnl_net"]   < trade["gross_pnl"]  # 扣成本後更少
        assert trade["direction"] == "long"
        assert trade["ticker"]    == "2330"

    def test_long_loss_calculation(self):
        """出場價低於進場價 → pnl_net < 0"""
        pos = make_long_position(adj_entry=100.0, shares=1000)
        trade = self.bt._close_position(
            pos=pos,
            exit_date=pd.Timestamp("2023-02-01"),
            exit_price=90.0,
            reason="signal",
        )
        assert trade["pnl_net"] < 0

    def test_long_trade_record_keys(self):
        pos = make_long_position()
        trade = self.bt._close_position(pos, pd.Timestamp("2023-02-01"), 110.0, "signal")
        for key in ["ticker", "direction", "lots", "shares",
                    "entry_date", "exit_date", "hold_days",
                    "adj_entry_price", "adj_exit_price",
                    "raw_entry_price", "raw_exit_price",
                    "gross_pnl", "total_cost", "pnl_net",
                    "atr_at_entry", "exit_reason"]:
            assert key in trade, f"缺少 trade key：{key}"

    def test_hold_days_correct(self):
        entry = pd.Timestamp("2023-01-10")
        exit_ = pd.Timestamp("2023-01-20")
        pos   = make_long_position(entry_date=entry.strftime("%Y-%m-%d"))
        trade = self.bt._close_position(pos, exit_, 105.0, "signal")
        assert trade["hold_days"] == 10


class TestClosePositionShort:
    """_close_position() 做空損益計算"""

    def setup_method(self):
        self.bt = make_backtester()

    def test_short_profit_when_price_falls(self):
        """做空進場100，出場90 → 獲利"""
        pos = make_short_position(adj_entry=100.0, shares=1000)
        trade = self.bt._close_position(
            pos=pos,
            exit_date=pd.Timestamp("2023-02-01"),
            exit_price=90.0,
            reason="signal",
        )
        assert trade["gross_pnl"] > 0

    def test_short_loss_when_price_rises(self):
        """做空進場100，出場110 → 虧損"""
        pos = make_short_position(adj_entry=100.0, shares=1000)
        trade = self.bt._close_position(
            pos=pos,
            exit_date=pd.Timestamp("2023-02-01"),
            exit_price=110.0,
            reason="signal",
        )
        assert trade["pnl_net"] < 0

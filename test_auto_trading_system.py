"""
單元測試：auto_trading_system.py

測試策略：
  - 每個測試都使用「自己造的假資料」，讓預期答案完全可控。
  - 測試分四層：
      1. Indicators   — 技術指標計算是否正確
      2. UniverseFilter — 選股條件是否按規格篩選
      3. SignalGenerator — 進出場訊號邏輯是否正確
      4. RiskManager  — 部位大小與交易成本計算是否正確

執行方式：
  pip install pytest
  pytest test_auto_trading_system.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "logic"))

import numpy as np
import pandas as pd
import pytest

from auto_trading_system import (
    TradingConfig,
    Indicators,
    UniverseFilter,
    SignalGenerator,
    RiskManager,
    Position,
)

# ══════════════════════════════════════════════
# 輔助函式
# ══════════════════════════════════════════════

def make_df(n: int, close: float = 100.0, high_offset: float = 2.0,
            low_offset: float = 2.0, amount: float = 10_000_000.0) -> pd.DataFrame:
    """
    建立固定價格的假 OHLCV DataFrame（n 個交易日）。
    High = close + high_offset
    Low  = close - low_offset
    用於需要「穩定數值」的測試。
    """
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({
        "Open":   [close] * n,
        "High":   [close + high_offset] * n,
        "Low":    [close - low_offset] * n,
        "Close":  [close] * n,
        "Volume": [1_000_000] * n,
        "Amount": [amount] * n,
    }, index=dates)


def make_df_with_close(closes: list, amount: float = 10_000_000.0) -> pd.DataFrame:
    """
    用指定的收盤價序列建立 DataFrame。
    High = close * 1.02, Low = close * 0.98
    用於需要「特定趨勢」的測試。
    """
    n = len(closes)
    dates = pd.bdate_range("2020-01-01", periods=n)
    closes_arr = np.array(closes, dtype=float)
    return pd.DataFrame({
        "Open":   closes_arr,
        "High":   closes_arr * 1.02,
        "Low":    closes_arr * 0.98,
        "Close":  closes_arr,
        "Volume": [1_000_000] * n,
        "Amount": [amount] * n,
    }, index=dates)


# ══════════════════════════════════════════════
# 1. Indicators 測試
# ══════════════════════════════════════════════

class TestIndicators:
    """驗證技術指標計算是否正確"""

    def test_atr_constant_range(self):
        """
        當 High-Low 固定（無缺口），ATR 應等於 High-Low 差距。

        設計：High=102, Low=98 → TrueRange 每天都是 4
        預期：ATR(14) 穩定後 = 4.0
        """
        df = make_df(n=50, close=100.0, high_offset=2.0, low_offset=2.0)
        atr = Indicators.atr(df, period=14)

        # 前 13 筆因 rolling window 不足，應為 NaN
        assert atr.iloc[:13].isna().all(), "前 13 筆應為 NaN（rolling 不足）"

        # 第 14 筆之後，ATR 應等於 4.0（High-Low=4，無跳空）
        assert abs(atr.iloc[14] - 4.0) < 1e-6, f"ATR 應為 4.0，實際得到 {atr.iloc[14]}"
        assert abs(atr.iloc[-1] - 4.0) < 1e-6, f"最後一筆 ATR 應為 4.0，實際得到 {atr.iloc[-1]}"

    def test_ma_fast_calculation(self):
        """
        MA(5) 的值應等於過去 5 天收盤的平均。

        設計：收盤價 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        預期：MA(5) 的第 5 筆（index=4）= (1+2+3+4+5)/5 = 3.0
              MA(5) 的最後一筆（index=9）= (6+7+8+9+10)/5 = 8.0
        """
        closes = list(range(1, 11))  # [1, 2, ..., 10]
        df = make_df_with_close(closes)

        cfg = TradingConfig(ma_fast=5, ma_slow=10, breakout_window=5,
                            atr_period=3, week52=5)
        df_ind = Indicators.add_all(df, cfg)

        assert abs(df_ind["MA_Fast"].iloc[4] - 3.0) < 1e-6, \
            f"MA(5) 第 5 筆應為 3.0，實際 {df_ind['MA_Fast'].iloc[4]}"
        assert abs(df_ind["MA_Fast"].iloc[-1] - 8.0) < 1e-6, \
            f"MA(5) 最後筆應為 8.0，實際 {df_ind['MA_Fast'].iloc[-1]}"

    def test_high_n_is_rolling_max_of_close(self):
        """
        High_N 是收盤價的 rolling max，應等於過去 N 天的最高收盤。

        設計：收盤 [10, 20, 15, 18, 12]，window=3
        第 3 筆（index=2）: max(10,20,15) = 20
        第 5 筆（index=4）: max(15,18,12) = 18
        """
        closes = [10, 20, 15, 18, 12]
        df = make_df_with_close(closes)
        cfg = TradingConfig(ma_fast=2, ma_slow=3, breakout_window=3,
                            atr_period=2, week52=3)
        df_ind = Indicators.add_all(df, cfg)

        assert abs(df_ind["High_N"].iloc[2] - 20.0) < 1e-6, \
            f"High_N 第 3 筆應為 20，實際 {df_ind['High_N'].iloc[2]}"
        assert abs(df_ind["High_N"].iloc[4] - 18.0) < 1e-6, \
            f"High_N 最後筆應為 18，實際 {df_ind['High_N'].iloc[4]}"

    def test_high_52w_uses_high_column(self):
        """
        High_52W 應取 High 欄位（最高價）的 52 週最大值，不是收盤。

        設計：收盤固定 100，但 High = close * 1.02 = 102
              week52=10，最後一筆 High_52W 應等於 102（High 欄）
        """
        closes = [100.0] * 30
        df = make_df_with_close(closes)
        cfg = TradingConfig(ma_fast=3, ma_slow=5, breakout_window=3,
                            atr_period=3, week52=10)
        df_ind = Indicators.add_all(df, cfg)

        expected = 100.0 * 1.02  # High = close * 1.02
        assert abs(df_ind["High_52W"].iloc[-1] - expected) < 1e-6, \
            f"High_52W 應為 {expected}（High 欄），實際 {df_ind['High_52W'].iloc[-1]}"

    def test_low_52w_uses_low_column(self):
        """
        Low_52W 應取 Low 欄位（最低價）的 52 週最小值，不是收盤。

        設計：收盤固定 100，但 Low = close * 0.98 = 98
              week52=10，最後一筆 Low_52W 應等於 98（Low 欄）
        """
        closes = [100.0] * 30
        df = make_df_with_close(closes)
        cfg = TradingConfig(ma_fast=3, ma_slow=5, breakout_window=3,
                            atr_period=3, week52=10)
        df_ind = Indicators.add_all(df, cfg)

        expected = 100.0 * 0.98  # Low = close * 0.98
        assert abs(df_ind["Low_52W"].iloc[-1] - expected) < 1e-6, \
            f"Low_52W 應為 {expected}（Low 欄），實際 {df_ind['Low_52W'].iloc[-1]}"


# ══════════════════════════════════════════════
# 2. UniverseFilter 測試
# ══════════════════════════════════════════════

class TestUniverseFilter:
    """驗證選股條件篩選是否符合規格"""

    CFG = TradingConfig(
        min_avg_amount=5_000_000,
        week52=252,
        ma_fast=50, ma_slow=100,
        breakout_window=50, atr_period=14,
    )

    def _make_ind(self, closes: list, amount: float) -> pd.DataFrame:
        df = make_df_with_close(closes, amount=amount)
        return Indicators.add_all(df, self.CFG)

    # ── 入選條件 ──────────────────────────────

    def test_select_when_high_breaks_52w_high(self):
        """
        金額充足 + 今日最高價 > 52 週最高價 → 應入選（突破新高）。

        設計：前 252 天 High 固定 102，最後一天 High = 103（突破）
        """
        # 前 252 天 close=100（High=102），最後一天 close=101（High=102.02）
        # 為了讓最後一天 High 確實 > High_52W，最後一天給更高的 close
        closes = [100.0] * 252 + [105.0]
        df_ind = self._make_ind(closes, amount=10_000_000)
        date = df_ind.index[-1]

        result = UniverseFilter(self.CFG).filter({"AAAA": df_ind}, date)

        assert "AAAA" in result, "最高價突破 52 週高，應入選"

    def test_select_when_low_breaks_52w_low(self):
        """
        金額充足 + 今日最低價 < 52 週最低價 → 應入選（突破新低）。

        設計：前 252 天 Low 固定 98，最後一天 Low 更低
        """
        closes = [100.0] * 252 + [95.0]
        df_ind = self._make_ind(closes, amount=10_000_000)
        date = df_ind.index[-1]

        result = UniverseFilter(self.CFG).filter({"BBBB": df_ind}, date)

        assert "BBBB" in result, "最低價突破 52 週低，應入選"

    # ── 排除條件 ──────────────────────────────

    def test_reject_when_amount_too_low(self):
        """
        成交金額不足門檻（1M < 5M）→ 不應入選。
        """
        closes = [100.0] * 252 + [105.0]
        df_ind = self._make_ind(closes, amount=1_000_000)
        date = df_ind.index[-1]

        result = UniverseFilter(self.CFG).filter({"CCCC": df_ind}, date)

        assert "CCCC" not in result, "成交金額不足，不應入選"

    def test_reject_when_no_52w_breakout(self):
        """
        金額充足但最高價 / 最低價都沒有突破 52 週 → 不應入選。

        設計：全部 253 天收盤固定 100，最後一天沒有突破
        """
        closes = [100.0] * 253
        df_ind = self._make_ind(closes, amount=10_000_000)
        date = df_ind.index[-1]

        result = UniverseFilter(self.CFG).filter({"DDDD": df_ind}, date)

        assert "DDDD" not in result, "沒有突破 52 週高低，不應入選"

    def test_reject_when_nan_indicator(self):
        """
        資料不足導致 High_52W / Low_52W 為 NaN → 不應入選。

        設計：只給 50 筆，week52=252 → 指標全為 NaN
        """
        df = make_df(n=50, close=100.0, amount=10_000_000)
        df_ind = Indicators.add_all(df, self.CFG)
        date = df_ind.index[-1]

        result = UniverseFilter(self.CFG).filter({"EEEE": df_ind}, date)

        assert "EEEE" not in result, "指標 NaN 時不應入選"


# ══════════════════════════════════════════════
# 3. SignalGenerator 測試
# ══════════════════════════════════════════════

class TestSignalGenerator:
    """驗證進出場訊號邏輯"""

    def _make_row(self, close, high_n, ma_fast, ma_slow, atr,
                  high=None, low=None) -> pd.Series:
        """建立一筆指標列"""
        return pd.Series({
            "Close":   close,
            "High":    high if high is not None else close + 2,
            "Low":     low  if low  is not None else close - 2,
            "High_N":  high_n,
            "Low_N":   low,
            "MA_Fast": ma_fast,
            "MA_Slow": ma_slow,
            "ATR":     atr,
        })

    # ── 做多進場 ──────────────────────────────

    def test_long_entry_true_when_breakout_and_ma_aligned(self):
        """
        今日收盤 > 昨日 N 日高 且 MA_Fast > MA_Slow → 應觸發做多。
        """
        row      = self._make_row(close=105, high_n=110, ma_fast=102, ma_slow=95, atr=3)
        prev_row = self._make_row(close=100, high_n=104, ma_fast=101, ma_slow=95, atr=3)

        assert SignalGenerator.long_entry(row, prev_row), \
            "收盤 105 > 前日 High_N 104，MA_Fast > MA_Slow，應進場做多"

    def test_long_entry_false_when_no_breakout(self):
        """
        今日收盤 <= 昨日 N 日高 → 不觸發做多。
        """
        row      = self._make_row(close=100, high_n=110, ma_fast=102, ma_slow=95, atr=3)
        prev_row = self._make_row(close=99,  high_n=101, ma_fast=101, ma_slow=95, atr=3)

        assert not SignalGenerator.long_entry(row, prev_row), \
            "收盤 100 <= 前日 High_N 101，不應進場"

    def test_long_entry_false_when_ma_not_aligned(self):
        """
        有突破但 MA_Fast < MA_Slow → 趨勢不對，不觸發做多。
        """
        row      = self._make_row(close=105, high_n=110, ma_fast=90, ma_slow=100, atr=3)
        prev_row = self._make_row(close=100, high_n=104, ma_fast=89, ma_slow=100, atr=3)

        assert not SignalGenerator.long_entry(row, prev_row), \
            "MA_Fast(90) < MA_Slow(100)，趨勢不對，不應進場"

    def test_long_entry_false_when_nan(self):
        """
        任一指標為 NaN → 保護性回傳 False，不觸發訊號。
        """
        row      = self._make_row(close=float("nan"), high_n=104, ma_fast=102, ma_slow=95, atr=3)
        prev_row = self._make_row(close=100, high_n=104, ma_fast=101, ma_slow=95, atr=3)

        assert SignalGenerator.long_entry(row, prev_row) is False, \
            "Close 為 NaN 時，應回傳 False，不觸發訊號"

    # ── 做多出場 ──────────────────────────────

    def test_long_exit_true_when_close_below_trail_stop(self):
        """
        收盤 < trail_high - atr_mult * ATR → 應觸發停損出場。

        設計：trail_high=120, ATR=5, atr_mult=3
              停損線 = 120 - 3×5 = 105
              收盤 = 100 < 105 → 應出場
        """
        row = pd.Series({"Close": 100.0, "ATR": 5.0})
        assert SignalGenerator.long_exit(row, trail_high=120.0, atr_mult=3.0), \
            "收盤 100 < 停損線 105，應出場"

    def test_long_exit_false_when_close_above_trail_stop(self):
        """
        收盤 > 停損線 → 不出場，繼續持有。

        設計：trail_high=120, ATR=5, atr_mult=3
              停損線 = 105，收盤 = 110 > 105 → 不出場
        """
        row = pd.Series({"Close": 110.0, "ATR": 5.0})
        assert not SignalGenerator.long_exit(row, trail_high=120.0, atr_mult=3.0), \
            "收盤 110 > 停損線 105，不應出場"

    def test_long_exit_false_when_atr_nan(self):
        """ATR 為 NaN → 保護性回傳 False"""
        row = pd.Series({"Close": 100.0, "ATR": float("nan")})
        assert SignalGenerator.long_exit(row, trail_high=120.0, atr_mult=3.0) is False, \
            "ATR 為 NaN 時，不應觸發出場"

    # ── 做空對稱測試 ──────────────────────────

    def test_short_entry_true_when_breakdown_and_ma_inverted(self):
        """
        今日收盤 < 昨日 N 日低 且 MA_Fast < MA_Slow → 應觸發做空。
        """
        row = pd.Series({
            "Close": 95.0, "Low_N": 97.0,
            "MA_Fast": 98.0, "MA_Slow": 105.0, "ATR": 3.0,
        })
        prev_row = pd.Series({
            "Close": 100.0, "Low_N": 96.0,
            "MA_Fast": 99.0, "MA_Slow": 105.0, "ATR": 3.0,
        })
        assert SignalGenerator.short_entry(row, prev_row), \
            "收盤 95 < 前日 Low_N 96，MA_Fast < MA_Slow，應進場做空"

    def test_short_exit_true_when_close_above_trail_stop(self):
        """
        做空停損：收盤 > trail_low + atr_mult * ATR → 應出場。

        設計：trail_low=80, ATR=5, atr_mult=3
              停損線 = 80 + 15 = 95，收盤 = 100 > 95 → 應出場
        """
        row = pd.Series({"Close": 100.0, "ATR": 5.0})
        assert SignalGenerator.short_exit(row, trail_low=80.0, atr_mult=3.0), \
            "收盤 100 > 做空停損線 95，應出場"


# ══════════════════════════════════════════════
# 4. RiskManager 測試
# ══════════════════════════════════════════════

class TestRiskManager:
    """驗證部位大小與交易成本計算"""

    def setup_method(self):
        self.cfg = TradingConfig(
            initial_equity  = 1_000_000,
            risk_pct        = 0.002,       # 0.2%
            commission_rate = 0.001425,
            transaction_tax = 0.003,
            slippage        = 0.001,
            point_value     = 1.0,
        )
        self.rm = RiskManager(self.cfg)

    def test_risk_amount(self):
        """
        每筆風險金額 = 資金 × risk_pct。
        設計：1,000,000 × 0.002 = 2,000
        """
        assert abs(self.rm.risk_amount(1_000_000) - 2_000) < 1e-6, \
            f"風險金額應為 2000，實際 {self.rm.risk_amount(1_000_000)}"

    def test_position_size_lots_basic(self):
        """
        張數 = floor(risk_amount / (ATR × point_value) / 1000)。
        設計：equity=1,000,000, risk_pct=0.002, ATR=2.0
              risk_amount = 2000
              raw_shares  = 2000 / (2.0 × 1.0) = 1000 股
              lots        = floor(1000 / 1000) = 1 張
        """
        lots = self.rm.position_size_lots(equity=1_000_000, atr=2.0)
        assert lots == 1, f"張數應為 1，實際 {lots}"

    def test_position_size_lots_zero_when_atr_zero(self):
        """ATR = 0 時，不應開倉，應回傳 0"""
        lots = self.rm.position_size_lots(equity=1_000_000, atr=0.0)
        assert lots == 0, "ATR=0 時張數應為 0"

    def test_position_size_lots_zero_when_atr_negative(self):
        """ATR < 0（異常資料）時，應回傳 0"""
        lots = self.rm.position_size_lots(equity=1_000_000, atr=-1.0)
        assert lots == 0, "ATR<0 時張數應為 0"

    def test_transaction_cost_buy_no_tax(self):
        """
        買進成本 = 手續費 + 滑價（無交易稅）。
        設計：price=100, shares=1000
              notional   = 100,000
              commission = 100,000 × 0.001425 = 142.5
              slippage   = 100,000 × 0.001    = 100.0
              total      = 242.5（無稅）
        """
        cost = self.rm.transaction_cost(price=100.0, shares=1000, side="buy")
        expected = 100_000 * (0.001425 + 0.001)
        assert abs(cost - expected) < 1e-6, \
            f"買進成本應為 {expected:.2f}，實際 {cost:.2f}"

    def test_transaction_cost_sell_has_tax(self):
        """
        賣出成本 = 手續費 + 滑價 + 證交稅。
        設計：price=100, shares=1000
              notional   = 100,000
              commission = 142.5
              slippage   = 100.0
              tax        = 100,000 × 0.003 = 300.0
              total      = 542.5
        """
        cost = self.rm.transaction_cost(price=100.0, shares=1000, side="sell")
        expected = 100_000 * (0.001425 + 0.001 + 0.003)
        assert abs(cost - expected) < 1e-6, \
            f"賣出成本應為 {expected:.2f}，實際 {cost:.2f}"

    def test_sell_cost_greater_than_buy_cost(self):
        """賣出成本一定大於買進成本（因為多了交易稅）"""
        buy_cost  = self.rm.transaction_cost(100.0, 1000, "buy")
        sell_cost = self.rm.transaction_cost(100.0, 1000, "sell")
        assert sell_cost > buy_cost, "賣出成本應大於買進成本（含交易稅）"


# ══════════════════════════════════════════════
# 5. Position.update_trail 測試
# ══════════════════════════════════════════════

class TestPosition:
    """驗證追蹤停損更新邏輯"""

    def _make_long_pos(self) -> Position:
        return Position(
            ticker="TEST", direction="long",
            entry_date=pd.Timestamp("2024-01-01"),
            lots=1, shares=1000,
            adj_entry_price=100.0, raw_entry_price=100.0,
            trail_high=100.0, trail_low=100.0,
            atr_at_entry=2.0,
        )

    def test_trail_high_updates_on_new_high(self):
        """做多持倉：新高出現時 trail_high 應更新"""
        pos = self._make_long_pos()
        pos.update_trail(high=110.0, low=95.0)
        assert pos.trail_high == 110.0, "新高 110 應更新 trail_high"

    def test_trail_high_does_not_decrease(self):
        """做多持倉：trail_high 只升不降"""
        pos = self._make_long_pos()
        pos.update_trail(high=110.0, low=95.0)
        pos.update_trail(high=105.0, low=95.0)  # 低於之前的 110
        assert pos.trail_high == 110.0, "trail_high 不應因新的較低高點而下降"

    def test_trail_low_updates_for_short(self):
        """做空持倉：新低出現時 trail_low 應更新"""
        pos = self._make_long_pos()
        pos.direction = "short"
        pos.trail_low = 100.0
        pos.update_trail(high=102.0, low=90.0)
        assert pos.trail_low == 90.0, "做空時新低 90 應更新 trail_low"

    def test_trail_low_does_not_increase_for_short(self):
        """做空持倉：trail_low 只降不升"""
        pos = self._make_long_pos()
        pos.direction = "short"
        pos.trail_low = 100.0
        pos.update_trail(high=102.0, low=90.0)
        pos.update_trail(high=105.0, low=95.0)  # 高於之前的 90
        assert pos.trail_low == 90.0, "trail_low 不應因新的較高低點而上升"


# ══════════════════════════════════════════════
# 執行
# ══════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

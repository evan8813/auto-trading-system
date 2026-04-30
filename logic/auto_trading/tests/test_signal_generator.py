"""
test_signal_generator.py
────────────────────────
步驟 4：驗證進出場訊號邏輯。

測試項目：
  1. long_entry()  ── 突破 + 趨勢向上 → True
  2. long_entry()  ── 任一條件不符 → False
  3. long_exit()   ── 跌破追蹤停損 → True
  4. long_exit()   ── 未跌破 → False
  5. short_entry() ── 跌破 + 趨勢向下 → True
  6. short_exit()  ── 突破追蹤停損 → True
  7. NaN 輸入保護 → 一律回傳 False
"""

import pytest
import numpy as np
import pandas as pd

from signal_generator import SignalGenerator


def make_row(**kwargs) -> pd.Series:
    """建立測試用 row（pd.Series），未指定的欄位填 0"""
    defaults = {
        "Open": 100.0, "High": 105.0, "Low": 95.0, "Close": 102.0,
        "ATR": 5.0, "MA_Fast": 110.0, "MA_Slow": 100.0,
        "High_N": 108.0, "Low_N": 90.0,
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


class TestLongEntry:
    """long_entry(row, prev_row)"""

    def test_triggers_when_breakout_and_trend_up(self):
        """收盤突破前一日 High_N 且 MA_Fast > MA_Slow"""
        row      = make_row(Close=110.0, MA_Fast=115.0, MA_Slow=100.0)
        prev_row = make_row(High_N=108.0)
        assert SignalGenerator.long_entry(row, prev_row)

    def test_no_trigger_without_breakout(self):
        """收盤未突破 High_N"""
        row      = make_row(Close=100.0, MA_Fast=115.0, MA_Slow=100.0)
        prev_row = make_row(High_N=108.0)
        assert not SignalGenerator.long_entry(row, prev_row)

    def test_no_trigger_without_uptrend(self):
        """MA_Fast <= MA_Slow（趨勢不向上）"""
        row      = make_row(Close=110.0, MA_Fast=95.0, MA_Slow=100.0)
        prev_row = make_row(High_N=108.0)
        assert not SignalGenerator.long_entry(row, prev_row)

    def test_no_trigger_when_close_equals_high_n(self):
        """收盤等於 High_N（非突破，需嚴格大於）"""
        row      = make_row(Close=108.0, MA_Fast=115.0, MA_Slow=100.0)
        prev_row = make_row(High_N=108.0)
        assert not SignalGenerator.long_entry(row, prev_row)

    def test_nan_close_returns_false(self):
        row      = make_row(Close=np.nan)
        prev_row = make_row(High_N=108.0)
        assert not SignalGenerator.long_entry(row, prev_row)

    def test_nan_high_n_returns_false(self):
        row      = make_row(Close=110.0)
        prev_row = make_row(High_N=np.nan)
        assert not SignalGenerator.long_entry(row, prev_row)

    def test_nan_ma_fast_returns_false(self):
        row      = make_row(Close=110.0, MA_Fast=np.nan)
        prev_row = make_row(High_N=108.0)
        assert not SignalGenerator.long_entry(row, prev_row)

    def test_nan_ma_slow_returns_false(self):
        row      = make_row(Close=110.0, MA_Slow=np.nan)
        prev_row = make_row(High_N=108.0)
        assert not SignalGenerator.long_entry(row, prev_row)


class TestLongExit:
    """long_exit(row, trail_high, atr_mult)"""

    def test_triggers_when_close_below_stop(self):
        """Close < trail_high - atr_mult × ATR → 出場"""
        # stop = 120 - 3 × 5 = 105；Close = 103 < 105 → 出場
        row = make_row(Close=103.0, ATR=5.0)
        assert SignalGenerator.long_exit(row, trail_high=120.0, atr_mult=3.0)

    def test_no_trigger_when_close_above_stop(self):
        """Close >= stop → 不出場"""
        # stop = 120 - 3 × 5 = 105；Close = 106 > 105 → 持倉
        row = make_row(Close=106.0, ATR=5.0)
        assert not SignalGenerator.long_exit(row, trail_high=120.0, atr_mult=3.0)

    def test_no_trigger_when_close_equals_stop(self):
        """Close 等於 stop（非嚴格小於）→ 不出場"""
        row = make_row(Close=105.0, ATR=5.0)
        assert not SignalGenerator.long_exit(row, trail_high=120.0, atr_mult=3.0)

    def test_nan_atr_returns_false(self):
        row = make_row(Close=100.0, ATR=np.nan)
        assert not SignalGenerator.long_exit(row, trail_high=120.0, atr_mult=3.0)

    def test_nan_close_returns_false(self):
        row = make_row(Close=np.nan, ATR=5.0)
        assert not SignalGenerator.long_exit(row, trail_high=120.0, atr_mult=3.0)


class TestShortEntry:
    """short_entry(row, prev_row)"""

    def test_triggers_when_breakdown_and_trend_down(self):
        """收盤跌破前一日 Low_N 且 MA_Fast < MA_Slow"""
        row      = make_row(Close=88.0, MA_Fast=90.0, MA_Slow=100.0)
        prev_row = make_row(Low_N=92.0)
        assert SignalGenerator.short_entry(row, prev_row)

    def test_no_trigger_without_breakdown(self):
        """收盤未跌破 Low_N"""
        row      = make_row(Close=95.0, MA_Fast=90.0, MA_Slow=100.0)
        prev_row = make_row(Low_N=92.0)
        assert not SignalGenerator.short_entry(row, prev_row)

    def test_no_trigger_without_downtrend(self):
        """MA_Fast >= MA_Slow（趨勢不向下）"""
        row      = make_row(Close=88.0, MA_Fast=110.0, MA_Slow=100.0)
        prev_row = make_row(Low_N=92.0)
        assert not SignalGenerator.short_entry(row, prev_row)

    def test_nan_close_returns_false(self):
        row      = make_row(Close=np.nan)
        prev_row = make_row(Low_N=92.0)
        assert not SignalGenerator.short_entry(row, prev_row)

    def test_nan_low_n_returns_false(self):
        row      = make_row(Close=88.0, MA_Fast=90.0, MA_Slow=100.0)
        prev_row = make_row(Low_N=np.nan)
        assert not SignalGenerator.short_entry(row, prev_row)

    def test_no_trigger_when_close_equals_low_n(self):
        """收盤等於 Low_N（非跌破，需嚴格小於）"""
        row      = make_row(Close=92.0, MA_Fast=90.0, MA_Slow=100.0)
        prev_row = make_row(Low_N=92.0)
        assert not SignalGenerator.short_entry(row, prev_row)

    def test_nan_ma_fast_returns_false(self):
        row      = make_row(Close=88.0, MA_Fast=np.nan, MA_Slow=100.0)
        prev_row = make_row(Low_N=92.0)
        assert not SignalGenerator.short_entry(row, prev_row)

    def test_nan_ma_slow_returns_false(self):
        row      = make_row(Close=88.0, MA_Fast=90.0, MA_Slow=np.nan)
        prev_row = make_row(Low_N=92.0)
        assert not SignalGenerator.short_entry(row, prev_row)


class TestShortExit:
    """short_exit(row, trail_low, atr_mult)"""

    def test_triggers_when_close_above_stop(self):
        """Close > trail_low + atr_mult × ATR → 出場"""
        # stop = 80 + 3 × 5 = 95；Close = 97 > 95 → 出場
        row = make_row(Close=97.0, ATR=5.0)
        assert SignalGenerator.short_exit(row, trail_low=80.0, atr_mult=3.0)

    def test_no_trigger_when_close_below_stop(self):
        """Close <= stop → 持倉"""
        # stop = 80 + 3 × 5 = 95；Close = 94 < 95 → 持倉
        row = make_row(Close=94.0, ATR=5.0)
        assert not SignalGenerator.short_exit(row, trail_low=80.0, atr_mult=3.0)

    def test_nan_atr_returns_false(self):
        row = make_row(Close=97.0, ATR=np.nan)
        assert not SignalGenerator.short_exit(row, trail_low=80.0, atr_mult=3.0)

    def test_nan_close_returns_false(self):
        row = make_row(Close=np.nan, ATR=5.0)
        assert not SignalGenerator.short_exit(row, trail_low=80.0, atr_mult=3.0)

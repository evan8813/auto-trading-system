"""
test_signal_generator.py
────────────────────────
步驟 4：驗證進出場訊號邏輯。

測試項目：
  1. long_entry()  ── 突破 + MACD > 0 + MACD 上升 → True
  2. long_entry()  ── 任一條件不符 → False
  3. long_exit()   ── Phase 1（未獲利）：跌破 Low_Stop → True
  4. long_exit()   ── Phase 2（已獲利）：跌破 ATR 追蹤停損 → True
  5. short_entry() ── 跌破 + MACD < 0 + MACD 下降 → True
  6. short_exit()  ── Phase 1（未獲利）：突破 High_Stop → True
  7. short_exit()  ── Phase 2（已獲利）：突破 ATR 追蹤停損 → True
  8. NaN 輸入保護 → 一律回傳 False
"""

import pytest
import numpy as np
import pandas as pd

from signal_generator import SignalGenerator


def make_row(**kwargs) -> pd.Series:
    """建立測試用 row（pd.Series），未指定的欄位填預設值"""
    defaults = {
        "Open": 100.0, "High": 105.0, "Low": 95.0, "Close": 102.0,
        "ATR": 5.0,
        "High_N": 108.0, "Low_N": 90.0,
        "High_Stop": 106.0, "Low_Stop": 94.0,
        "MACD": 1.0,
        "Volume": 200_000.0,    # 預設量：2 倍均量（符合 1.5x 門檻）
        "Vol_MA20": 100_000.0,  # 預設 20 日均量
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


# ════════════════════════════════════════════════
# 做多進場
# ════════════════════════════════════════════════

class TestLongEntry:
    """long_entry(row, prev_row)"""

    def test_triggers_when_all_conditions_met(self):
        """收盤突破前日 High_N，MACD > 0 且上升"""
        row      = make_row(Close=110.0, MACD=2.0)   # MACD 今=2.0 > 昨=1.0，且 > 0
        prev_row = make_row(High_N=108.0, MACD=1.0)
        assert SignalGenerator.long_entry(row, prev_row)

    def test_no_trigger_without_breakout(self):
        """收盤未突破 High_N"""
        row      = make_row(Close=100.0, MACD=2.0)
        prev_row = make_row(High_N=108.0, MACD=1.0)
        assert not SignalGenerator.long_entry(row, prev_row)

    def test_no_trigger_when_close_equals_high_n(self):
        """收盤等於 High_N（需嚴格大於）"""
        row      = make_row(Close=108.0, MACD=2.0)
        prev_row = make_row(High_N=108.0, MACD=1.0)
        assert not SignalGenerator.long_entry(row, prev_row)

    def test_no_trigger_when_macd_negative(self):
        """MACD ≤ 0 → 不在多頭區間"""
        row      = make_row(Close=110.0, MACD=-0.5)
        prev_row = make_row(High_N=108.0, MACD=-1.0)
        assert not SignalGenerator.long_entry(row, prev_row)

    def test_no_trigger_when_macd_not_rising(self):
        """MACD 今 ≤ MACD 昨（未上升）"""
        row      = make_row(Close=110.0, MACD=1.0)   # 今=1.0 = 昨=1.0，未上升
        prev_row = make_row(High_N=108.0, MACD=1.0)
        assert not SignalGenerator.long_entry(row, prev_row)

    def test_no_trigger_when_macd_falling(self):
        """MACD 今 < MACD 昨（下降中）"""
        row      = make_row(Close=110.0, MACD=0.5)
        prev_row = make_row(High_N=108.0, MACD=2.0)
        assert not SignalGenerator.long_entry(row, prev_row)

    def test_nan_close_returns_false(self):
        row      = make_row(Close=np.nan)
        prev_row = make_row(High_N=108.0, MACD=1.0)
        assert not SignalGenerator.long_entry(row, prev_row)

    def test_nan_high_n_returns_false(self):
        row      = make_row(Close=110.0, MACD=2.0)
        prev_row = make_row(High_N=np.nan, MACD=1.0)
        assert not SignalGenerator.long_entry(row, prev_row)

    def test_nan_macd_today_returns_false(self):
        row      = make_row(Close=110.0, MACD=np.nan)
        prev_row = make_row(High_N=108.0, MACD=1.0)
        assert not SignalGenerator.long_entry(row, prev_row)

    def test_nan_macd_prev_returns_false(self):
        row      = make_row(Close=110.0, MACD=2.0)
        prev_row = make_row(High_N=108.0, MACD=np.nan)
        assert not SignalGenerator.long_entry(row, prev_row)

    def test_no_trigger_when_volume_below_threshold(self):
        """量不足（Volume < Vol_MA20 × 1.5）→ 不進場"""
        row      = make_row(Close=110.0, MACD=2.0, Volume=100_000.0, Vol_MA20=100_000.0)
        prev_row = make_row(High_N=108.0, MACD=1.0)
        assert not SignalGenerator.long_entry(row, prev_row)

    def test_no_trigger_when_volume_equals_threshold(self):
        """量剛好等於 1.5 倍均量（需嚴格大於）→ 不進場"""
        row      = make_row(Close=110.0, MACD=2.0, Volume=150_000.0, Vol_MA20=100_000.0)
        prev_row = make_row(High_N=108.0, MACD=1.0)
        assert not SignalGenerator.long_entry(row, prev_row)

    def test_triggers_with_volume_surge(self):
        """量超過 1.5 倍均量 → 進場"""
        row      = make_row(Close=110.0, MACD=2.0, Volume=160_000.0, Vol_MA20=100_000.0)
        prev_row = make_row(High_N=108.0, MACD=1.0)
        assert SignalGenerator.long_entry(row, prev_row)

    def test_nan_volume_returns_false(self):
        row      = make_row(Close=110.0, MACD=2.0, Volume=np.nan)
        prev_row = make_row(High_N=108.0, MACD=1.0)
        assert not SignalGenerator.long_entry(row, prev_row)

    def test_nan_vol_ma20_returns_false(self):
        row      = make_row(Close=110.0, MACD=2.0, Vol_MA20=np.nan)
        prev_row = make_row(High_N=108.0, MACD=1.0)
        assert not SignalGenerator.long_entry(row, prev_row)


# ════════════════════════════════════════════════
# 做多出場（ATR 追蹤停損）
# ════════════════════════════════════════════════

class TestLongExit:
    """long_exit(row, trail_high, atr_mult)
    stop = trail_high - atr_mult × ATR，從第一天開始，無 Phase 區分。
    """

    def test_triggers_when_close_below_atr_stop(self):
        """stop = 100 - 3×5 = 85，Close=84 < 85 → 出場"""
        row = make_row(Close=84.0, ATR=5.0)
        assert SignalGenerator.long_exit(row, trail_high=100.0, atr_mult=3.0)

    def test_no_trigger_when_close_above_atr_stop(self):
        """stop = 100 - 3×5 = 85，Close=86 > 85 → 不出場"""
        row = make_row(Close=86.0, ATR=5.0)
        assert not SignalGenerator.long_exit(row, trail_high=100.0, atr_mult=3.0)

    def test_no_trigger_when_close_equals_atr_stop(self):
        """stop = 100 - 3×5 = 85，Close=85（需嚴格小於）→ 不出場"""
        row = make_row(Close=85.0, ATR=5.0)
        assert not SignalGenerator.long_exit(row, trail_high=100.0, atr_mult=3.0)

    def test_higher_trail_high_raises_stop(self):
        """trail_high 越高，stop 越高：trail=120，stop=120-15=105，Close=104 → 出場"""
        row = make_row(Close=104.0, ATR=5.0)
        assert SignalGenerator.long_exit(row, trail_high=120.0, atr_mult=3.0)

    def test_higher_trail_high_no_trigger_above_stop(self):
        """trail=120，stop=105，Close=106 → 不出場"""
        row = make_row(Close=106.0, ATR=5.0)
        assert not SignalGenerator.long_exit(row, trail_high=120.0, atr_mult=3.0)

    def test_day1_stop_at_entry_price(self):
        """進場第一天：trail_high = entry_price = 100，stop=85，Close=84 → 出場"""
        row = make_row(Close=84.0, ATR=5.0)
        assert SignalGenerator.long_exit(row, trail_high=100.0, atr_mult=3.0)

    def test_nan_atr_returns_false(self):
        row = make_row(Close=84.0, ATR=np.nan)
        assert not SignalGenerator.long_exit(row, trail_high=100.0, atr_mult=3.0)

    def test_nan_close_returns_false(self):
        row = make_row(Close=np.nan, ATR=5.0)
        assert not SignalGenerator.long_exit(row, trail_high=100.0, atr_mult=3.0)


# ════════════════════════════════════════════════
# 做空進場
# ════════════════════════════════════════════════

class TestShortEntry:
    """short_entry(row, prev_row)"""

    def test_triggers_when_all_conditions_met(self):
        """收盤跌破前日 Low_N，MACD < 0 且下降"""
        row      = make_row(Close=88.0, MACD=-2.0)   # 今=-2.0 < 昨=-1.0，且 < 0
        prev_row = make_row(Low_N=92.0, MACD=-1.0)
        assert SignalGenerator.short_entry(row, prev_row)

    def test_no_trigger_without_breakdown(self):
        """收盤未跌破 Low_N"""
        row      = make_row(Close=95.0, MACD=-2.0)
        prev_row = make_row(Low_N=92.0, MACD=-1.0)
        assert not SignalGenerator.short_entry(row, prev_row)

    def test_no_trigger_when_close_equals_low_n(self):
        """收盤等於 Low_N（需嚴格小於）"""
        row      = make_row(Close=92.0, MACD=-2.0)
        prev_row = make_row(Low_N=92.0, MACD=-1.0)
        assert not SignalGenerator.short_entry(row, prev_row)

    def test_no_trigger_when_macd_positive(self):
        """MACD >= 0 → 不在空頭區間"""
        row      = make_row(Close=88.0, MACD=0.5)
        prev_row = make_row(Low_N=92.0, MACD=-1.0)
        assert not SignalGenerator.short_entry(row, prev_row)

    def test_no_trigger_when_macd_not_falling(self):
        """MACD 今 >= MACD 昨（未下降）"""
        row      = make_row(Close=88.0, MACD=-1.0)   # 今=-1.0 = 昨=-1.0
        prev_row = make_row(Low_N=92.0, MACD=-1.0)
        assert not SignalGenerator.short_entry(row, prev_row)

    def test_nan_close_returns_false(self):
        row      = make_row(Close=np.nan)
        prev_row = make_row(Low_N=92.0, MACD=-1.0)
        assert not SignalGenerator.short_entry(row, prev_row)

    def test_nan_low_n_returns_false(self):
        row      = make_row(Close=88.0, MACD=-2.0)
        prev_row = make_row(Low_N=np.nan, MACD=-1.0)
        assert not SignalGenerator.short_entry(row, prev_row)

    def test_nan_macd_returns_false(self):
        row      = make_row(Close=88.0, MACD=np.nan)
        prev_row = make_row(Low_N=92.0, MACD=-1.0)
        assert not SignalGenerator.short_entry(row, prev_row)

    def test_no_trigger_when_volume_below_threshold(self):
        """量不足 → 不進場"""
        row      = make_row(Close=88.0, MACD=-2.0, Volume=100_000.0, Vol_MA20=100_000.0)
        prev_row = make_row(Low_N=92.0, MACD=-1.0)
        assert not SignalGenerator.short_entry(row, prev_row)

    def test_triggers_with_volume_surge(self):
        """量超過 1.5 倍均量 → 進場"""
        row      = make_row(Close=88.0, MACD=-2.0, Volume=160_000.0, Vol_MA20=100_000.0)
        prev_row = make_row(Low_N=92.0, MACD=-1.0)
        assert SignalGenerator.short_entry(row, prev_row)

    def test_nan_volume_returns_false(self):
        row      = make_row(Close=88.0, MACD=-2.0, Volume=np.nan)
        prev_row = make_row(Low_N=92.0, MACD=-1.0)
        assert not SignalGenerator.short_entry(row, prev_row)


# ════════════════════════════════════════════════
# 做空出場（ATR 追蹤停損）
# ════════════════════════════════════════════════

class TestShortExit:
    """short_exit(row, trail_low, atr_mult)
    stop = trail_low + atr_mult × ATR，從第一天開始，無 Phase 區分。
    """

    def test_triggers_when_close_above_atr_stop(self):
        """stop = 100 + 3×5 = 115，Close=116 > 115 → 出場"""
        row = make_row(Close=116.0, ATR=5.0)
        assert SignalGenerator.short_exit(row, trail_low=100.0, atr_mult=3.0)

    def test_no_trigger_when_close_below_atr_stop(self):
        """stop = 100 + 3×5 = 115，Close=114 < 115 → 不出場"""
        row = make_row(Close=114.0, ATR=5.0)
        assert not SignalGenerator.short_exit(row, trail_low=100.0, atr_mult=3.0)

    def test_no_trigger_when_close_equals_atr_stop(self):
        """stop = 115，Close=115（需嚴格大於）→ 不出場"""
        row = make_row(Close=115.0, ATR=5.0)
        assert not SignalGenerator.short_exit(row, trail_low=100.0, atr_mult=3.0)

    def test_lower_trail_low_lowers_stop(self):
        """trail_low 越低，stop 越低：trail=80，stop=80+15=95，Close=94 → 不出場"""
        row = make_row(Close=94.0, ATR=5.0)
        assert not SignalGenerator.short_exit(row, trail_low=80.0, atr_mult=3.0)

    def test_lower_trail_low_triggers_above_stop(self):
        """trail=80，stop=95，Close=96 → 出場"""
        row = make_row(Close=96.0, ATR=5.0)
        assert SignalGenerator.short_exit(row, trail_low=80.0, atr_mult=3.0)

    def test_day1_stop_at_entry_price(self):
        """進場第一天：trail_low = entry_price = 100，stop=115，Close=116 → 出場"""
        row = make_row(Close=116.0, ATR=5.0)
        assert SignalGenerator.short_exit(row, trail_low=100.0, atr_mult=3.0)

    def test_nan_atr_returns_false(self):
        row = make_row(Close=116.0, ATR=np.nan)
        assert not SignalGenerator.short_exit(row, trail_low=100.0, atr_mult=3.0)

    def test_nan_close_returns_false(self):
        row = make_row(Close=np.nan, ATR=5.0)
        assert not SignalGenerator.short_exit(row, trail_low=100.0, atr_mult=3.0)

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


# ════════════════════════════════════════════════
# 做多出場（兩段式）
# ════════════════════════════════════════════════

class TestLongExit:
    """long_exit(row, trail_high, atr_mult, entry_price)"""

    # ── Phase 1：trail_high <= entry_price，用 Low_Stop ──

    def test_phase1_triggers_when_close_below_low_stop(self):
        """trail_high = entry_price（未獲利），收盤跌破 Low_Stop"""
        row = make_row(Close=93.0, Low_Stop=94.0, ATR=5.0)
        assert SignalGenerator.long_exit(row, trail_high=100.0, atr_mult=3.0, entry_price=100.0)

    def test_phase1_no_trigger_when_close_above_low_stop(self):
        """trail_high = entry_price，收盤仍在 Low_Stop 之上"""
        row = make_row(Close=95.0, Low_Stop=94.0, ATR=5.0)
        assert not SignalGenerator.long_exit(row, trail_high=100.0, atr_mult=3.0, entry_price=100.0)

    def test_phase1_no_trigger_when_close_equals_low_stop(self):
        """收盤等於 Low_Stop（需嚴格小於）"""
        row = make_row(Close=94.0, Low_Stop=94.0, ATR=5.0)
        assert not SignalGenerator.long_exit(row, trail_high=100.0, atr_mult=3.0, entry_price=100.0)

    def test_phase1_nan_low_stop_returns_false(self):
        row = make_row(Close=93.0, Low_Stop=np.nan, ATR=5.0)
        assert not SignalGenerator.long_exit(row, trail_high=100.0, atr_mult=3.0, entry_price=100.0)

    # ── Phase 2：trail_high > entry_price，用 ATR 追蹤停損 ──

    def test_phase2_triggers_when_close_below_atr_stop(self):
        """trail_high > entry_price，stop = 120 - 3×5 = 105，Close=103 < 105"""
        row = make_row(Close=103.0, ATR=5.0, Low_Stop=94.0)
        assert SignalGenerator.long_exit(row, trail_high=120.0, atr_mult=3.0, entry_price=100.0)

    def test_phase2_no_trigger_when_close_above_atr_stop(self):
        """trail_high > entry_price，stop = 120 - 3×5 = 105，Close=106 > 105"""
        row = make_row(Close=106.0, ATR=5.0, Low_Stop=94.0)
        assert not SignalGenerator.long_exit(row, trail_high=120.0, atr_mult=3.0, entry_price=100.0)

    def test_phase2_ignores_low_stop(self):
        """Phase 2 時，即使 Close < Low_Stop，只要未跌破 ATR 停損就不出場"""
        # stop_atr = 120 - 3×5 = 105；Close=93 < Low_Stop=94，但 93 < 105 → 出場
        # 確認：出場是因為 ATR 停損，不是因為 Low_Stop
        row = make_row(Close=93.0, ATR=5.0, Low_Stop=94.0)
        # ATR 停損觸發
        assert SignalGenerator.long_exit(row, trail_high=120.0, atr_mult=3.0, entry_price=100.0)
        # 若 ATR 停損沒觸發（Close 很高），即使 Close < Low_Stop 也不出場
        row_high = make_row(Close=106.0, ATR=5.0, Low_Stop=94.0)
        assert not SignalGenerator.long_exit(row_high, trail_high=120.0, atr_mult=3.0, entry_price=100.0)

    def test_phase2_nan_atr_returns_false(self):
        row = make_row(Close=103.0, ATR=np.nan)
        assert not SignalGenerator.long_exit(row, trail_high=120.0, atr_mult=3.0, entry_price=100.0)

    def test_nan_close_returns_false(self):
        row = make_row(Close=np.nan, ATR=5.0)
        assert not SignalGenerator.long_exit(row, trail_high=120.0, atr_mult=3.0, entry_price=100.0)


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


# ════════════════════════════════════════════════
# 做空出場（兩段式）
# ════════════════════════════════════════════════

class TestShortExit:
    """short_exit(row, trail_low, atr_mult, entry_price)"""

    # ── Phase 1：trail_low >= entry_price，用 High_Stop ──

    def test_phase1_triggers_when_close_above_high_stop(self):
        """trail_low = entry_price（未獲利），收盤突破 High_Stop"""
        row = make_row(Close=107.0, High_Stop=106.0, ATR=5.0)
        assert SignalGenerator.short_exit(row, trail_low=100.0, atr_mult=3.0, entry_price=100.0)

    def test_phase1_no_trigger_when_close_below_high_stop(self):
        """trail_low = entry_price，收盤仍在 High_Stop 之下"""
        row = make_row(Close=104.0, High_Stop=106.0, ATR=5.0)
        assert not SignalGenerator.short_exit(row, trail_low=100.0, atr_mult=3.0, entry_price=100.0)

    def test_phase1_no_trigger_when_close_equals_high_stop(self):
        """收盤等於 High_Stop（需嚴格大於）"""
        row = make_row(Close=106.0, High_Stop=106.0, ATR=5.0)
        assert not SignalGenerator.short_exit(row, trail_low=100.0, atr_mult=3.0, entry_price=100.0)

    def test_phase1_nan_high_stop_returns_false(self):
        row = make_row(Close=107.0, High_Stop=np.nan, ATR=5.0)
        assert not SignalGenerator.short_exit(row, trail_low=100.0, atr_mult=3.0, entry_price=100.0)

    # ── Phase 2：trail_low < entry_price，用 ATR 追蹤停損 ──

    def test_phase2_triggers_when_close_above_atr_stop(self):
        """trail_low < entry_price，stop = 80 + 3×5 = 95，Close=97 > 95"""
        row = make_row(Close=97.0, ATR=5.0, High_Stop=106.0)
        assert SignalGenerator.short_exit(row, trail_low=80.0, atr_mult=3.0, entry_price=100.0)

    def test_phase2_no_trigger_when_close_below_atr_stop(self):
        """trail_low < entry_price，stop = 80 + 3×5 = 95，Close=94 < 95"""
        row = make_row(Close=94.0, ATR=5.0, High_Stop=106.0)
        assert not SignalGenerator.short_exit(row, trail_low=80.0, atr_mult=3.0, entry_price=100.0)

    def test_phase2_nan_atr_returns_false(self):
        row = make_row(Close=97.0, ATR=np.nan)
        assert not SignalGenerator.short_exit(row, trail_low=80.0, atr_mult=3.0, entry_price=100.0)

    def test_nan_close_returns_false(self):
        row = make_row(Close=np.nan, ATR=5.0)
        assert not SignalGenerator.short_exit(row, trail_low=80.0, atr_mult=3.0, entry_price=100.0)

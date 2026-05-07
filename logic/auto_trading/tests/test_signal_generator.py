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
    """long_exit(row, trail_high, atr_mult, entry_price, atr_at_entry, phase1_atr_mult)
    Phase 1 固定停損 = entry_price - phase1_atr_mult × atr_at_entry
    """

    # ── Phase 1：trail_high <= entry_price，固定停損（進場當下鎖死）──
    # entry=100, atr_at_entry=5, phase1_atr_mult=1.5 → fixed_stop=92.5

    def test_phase1_triggers_when_close_below_fixed_stop(self):
        """收盤低於固定停損 92.5 → 出場"""
        row = make_row(Close=91.0, ATR=5.0)
        assert SignalGenerator.long_exit(row, trail_high=100.0, atr_mult=3.0,
                                         entry_price=100.0, atr_at_entry=5.0, phase1_atr_mult=1.5)

    def test_phase1_no_trigger_when_close_above_fixed_stop(self):
        """收盤高於固定停損 92.5 → 不出場"""
        row = make_row(Close=93.0, ATR=5.0)
        assert not SignalGenerator.long_exit(row, trail_high=100.0, atr_mult=3.0,
                                              entry_price=100.0, atr_at_entry=5.0, phase1_atr_mult=1.5)

    def test_phase1_no_trigger_when_close_equals_fixed_stop(self):
        """收盤等於固定停損（需嚴格小於）→ 不出場"""
        row = make_row(Close=92.5, ATR=5.0)
        assert not SignalGenerator.long_exit(row, trail_high=100.0, atr_mult=3.0,
                                              entry_price=100.0, atr_at_entry=5.0, phase1_atr_mult=1.5)

    def test_phase1_stop_does_not_move_with_price(self):
        """固定停損不隨股價更新：Close=91 仍觸發，不管目前 ATR 多少"""
        row = make_row(Close=91.0, ATR=2.0)   # ATR 縮小，固定停損不變
        assert SignalGenerator.long_exit(row, trail_high=100.0, atr_mult=3.0,
                                         entry_price=100.0, atr_at_entry=5.0, phase1_atr_mult=1.5)

    # ── Phase 2：trail_high > entry_price，用 ATR 追蹤停損 ──

    def test_phase2_triggers_when_close_below_atr_stop(self):
        """trail_high > entry_price，stop = 120 - 3×5 = 105，Close=103 < 105"""
        row = make_row(Close=103.0, ATR=5.0)
        assert SignalGenerator.long_exit(row, trail_high=120.0, atr_mult=3.0,
                                         entry_price=100.0, atr_at_entry=5.0, phase1_atr_mult=1.5)

    def test_phase2_no_trigger_when_close_above_atr_stop(self):
        """trail_high > entry_price，stop = 120 - 3×5 = 105，Close=106 > 105"""
        row = make_row(Close=106.0, ATR=5.0)
        assert not SignalGenerator.long_exit(row, trail_high=120.0, atr_mult=3.0,
                                              entry_price=100.0, atr_at_entry=5.0, phase1_atr_mult=1.5)

    def test_phase2_uses_atr_not_fixed_stop(self):
        """Phase 2 只看 ATR 追蹤停損，不看固定停損"""
        # Close=91 < fixed_stop=92.5，但 Phase 2 用 ATR stop=120-3×5=105
        # Close=91 < 105 → 出場（原因是 ATR 停損，不是固定停損）
        row = make_row(Close=91.0, ATR=5.0)
        assert SignalGenerator.long_exit(row, trail_high=120.0, atr_mult=3.0,
                                         entry_price=100.0, atr_at_entry=5.0, phase1_atr_mult=1.5)
        # Close=106 > ATR stop=105 → 不出場
        row2 = make_row(Close=106.0, ATR=5.0)
        assert not SignalGenerator.long_exit(row2, trail_high=120.0, atr_mult=3.0,
                                              entry_price=100.0, atr_at_entry=5.0, phase1_atr_mult=1.5)

    def test_phase2_nan_atr_returns_false(self):
        row = make_row(Close=103.0, ATR=np.nan)
        assert not SignalGenerator.long_exit(row, trail_high=120.0, atr_mult=3.0,
                                              entry_price=100.0, atr_at_entry=5.0, phase1_atr_mult=1.5)

    def test_nan_close_returns_false(self):
        row = make_row(Close=np.nan, ATR=5.0)
        assert not SignalGenerator.long_exit(row, trail_high=120.0, atr_mult=3.0,
                                              entry_price=100.0, atr_at_entry=5.0, phase1_atr_mult=1.5)


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
    """short_exit(row, trail_low, atr_mult, entry_price, atr_at_entry, phase1_atr_mult)
    Phase 1 固定停損 = entry_price + phase1_atr_mult × atr_at_entry
    """

    # ── Phase 1：trail_low >= entry_price，固定停損（進場當下鎖死）──
    # entry=100, atr_at_entry=5, phase1_atr_mult=1.5 → fixed_stop=107.5

    def test_phase1_triggers_when_close_above_fixed_stop(self):
        """收盤高於固定停損 107.5 → 出場"""
        row = make_row(Close=109.0, ATR=5.0)
        assert SignalGenerator.short_exit(row, trail_low=100.0, atr_mult=3.0,
                                          entry_price=100.0, atr_at_entry=5.0, phase1_atr_mult=1.5)

    def test_phase1_no_trigger_when_close_below_fixed_stop(self):
        """收盤低於固定停損 107.5 → 不出場"""
        row = make_row(Close=106.0, ATR=5.0)
        assert not SignalGenerator.short_exit(row, trail_low=100.0, atr_mult=3.0,
                                               entry_price=100.0, atr_at_entry=5.0, phase1_atr_mult=1.5)

    def test_phase1_no_trigger_when_close_equals_fixed_stop(self):
        """收盤等於固定停損（需嚴格大於）→ 不出場"""
        row = make_row(Close=107.5, ATR=5.0)
        assert not SignalGenerator.short_exit(row, trail_low=100.0, atr_mult=3.0,
                                               entry_price=100.0, atr_at_entry=5.0, phase1_atr_mult=1.5)

    def test_phase1_stop_does_not_move_with_price(self):
        """固定停損不隨股價更新：Close=109 仍觸發，不管目前 ATR 多少"""
        row = make_row(Close=109.0, ATR=2.0)
        assert SignalGenerator.short_exit(row, trail_low=100.0, atr_mult=3.0,
                                          entry_price=100.0, atr_at_entry=5.0, phase1_atr_mult=1.5)

    # ── Phase 2：trail_low < entry_price，用 ATR 追蹤停損 ──

    def test_phase2_triggers_when_close_above_atr_stop(self):
        """trail_low < entry_price，stop = 80 + 3×5 = 95，Close=97 > 95"""
        row = make_row(Close=97.0, ATR=5.0)
        assert SignalGenerator.short_exit(row, trail_low=80.0, atr_mult=3.0,
                                          entry_price=100.0, atr_at_entry=5.0, phase1_atr_mult=1.5)

    def test_phase2_no_trigger_when_close_below_atr_stop(self):
        """trail_low < entry_price，stop = 80 + 3×5 = 95，Close=94 < 95"""
        row = make_row(Close=94.0, ATR=5.0)
        assert not SignalGenerator.short_exit(row, trail_low=80.0, atr_mult=3.0,
                                               entry_price=100.0, atr_at_entry=5.0, phase1_atr_mult=1.5)

    def test_phase2_nan_atr_returns_false(self):
        row = make_row(Close=97.0, ATR=np.nan)
        assert not SignalGenerator.short_exit(row, trail_low=80.0, atr_mult=3.0,
                                               entry_price=100.0, atr_at_entry=5.0, phase1_atr_mult=1.5)

    def test_nan_close_returns_false(self):
        row = make_row(Close=np.nan, ATR=5.0)
        assert not SignalGenerator.short_exit(row, trail_low=80.0, atr_mult=3.0,
                                               entry_price=100.0, atr_at_entry=5.0, phase1_atr_mult=1.5)

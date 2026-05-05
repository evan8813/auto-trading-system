"""
test_two_phase_exit.py
──────────────────────
驗證兩段式停損（Two-Phase Exit）機制的端對端正確性。

切換條件：
  trail_high > entry_price（多方）→ Phase 2
  trail_low  < entry_price（空方）→ Phase 2
  否則 → Phase 1

Phase 1（未獲利）：
  多方出場：Close < Low_Stop（前日 stop_window 日最低）
  空方出場：Close > High_Stop（前日 stop_window 日最高）

Phase 2（已獲利）：
  多方出場：Close < trail_high - atr_mult × ATR
  空方出場：Close > trail_low  + atr_mult × ATR

測試結構：
  1. TestLowHighStopBreachable    ── shift(1) 後停損值確實可被穿越
  2. TestPhaseExitWithRealIndicators ── 用 add_all() 的真實指標驗證兩段出場
  3. TestPhaseBoundaryCondition   ── trail_high 邊界值切換 Phase 行為
"""

import numpy as np
import pandas as pd
import pytest

from config import TradingConfig
from indicators import Indicators
from signal_generator import SignalGenerator


# ══════════════════════════════════════════════
# 共用資料產生器
# ══════════════════════════════════════════════

def _declining_df(n: int = 40, start: float = 50.0) -> pd.DataFrame:
    """每日收盤價穩定下跌的 OHLCV DataFrame。

    蠟燭高低差設為 0.3（小於每日跌幅 ≈ 1.15），
    確保 Close[t] < Low[t-1]（即 Close < Low_Stop 可發生）。
    """
    dates  = pd.bdate_range("2023-01-01", periods=n)
    closes = np.linspace(start, start * 0.1, n)   # 90% 跌幅，日跌幅 ≈ 1.15
    lows   = closes - 0.3
    highs  = closes + 0.3
    return pd.DataFrame(
        {"Open": closes, "High": highs, "Low": lows, "Close": closes,
         "Volume": 10_000.0, "Amount": closes * 10_000.0},
        index=dates,
    )


def _rising_df(n: int = 40, start: float = 20.0) -> pd.DataFrame:
    """每日收盤價穩定上漲的 OHLCV DataFrame。

    蠟燭高低差設為 0.3（小於每日漲幅 ≈ 1.54），
    確保 Close[t] > High[t-1]（即 Close > High_Stop 可發生）。
    """
    dates  = pd.bdate_range("2023-01-01", periods=n)
    closes = np.linspace(start, start * 4.0, n)   # 300% 漲幅，日漲幅 ≈ 1.54
    lows   = closes - 0.3
    highs  = closes + 0.3
    return pd.DataFrame(
        {"Open": closes, "High": highs, "Low": lows, "Close": closes,
         "Volume": 10_000.0, "Amount": closes * 10_000.0},
        index=dates,
    )


def _rally_then_drop_df(n_rise: int = 25, n_drop: int = 15,
                        start: float = 50.0) -> pd.DataFrame:
    """先漲後跌的 OHLCV DataFrame"""
    dates  = pd.bdate_range("2023-01-01", periods=n_rise + n_drop)
    peak   = start * 1.5
    closes = np.concatenate([
        np.linspace(start, peak, n_rise),
        np.linspace(peak, start * 0.7, n_drop),
    ])
    lows   = closes - 0.3
    highs  = closes + 0.3
    return pd.DataFrame(
        {"Open": closes, "High": highs, "Low": lows, "Close": closes,
         "Volume": 10_000.0, "Amount": closes * 10_000.0},
        index=dates,
    )


def _drop_then_rise_df(n_drop: int = 25, n_rise: int = 15,
                       start: float = 50.0) -> pd.DataFrame:
    """先跌後漲的 OHLCV DataFrame"""
    dates  = pd.bdate_range("2023-01-01", periods=n_drop + n_rise)
    trough = start * 0.65
    closes = np.concatenate([
        np.linspace(start, trough, n_drop),
        np.linspace(trough, start * 1.1, n_rise),
    ])
    lows   = closes - 0.3
    highs  = closes + 0.3
    return pd.DataFrame(
        {"Open": closes, "High": highs, "Low": lows, "Close": closes,
         "Volume": 10_000.0, "Amount": closes * 10_000.0},
        index=dates,
    )


# ══════════════════════════════════════════════
# 1. 停損值可被穿越（驗證 shift(1) 修正）
# ══════════════════════════════════════════════

class TestLowHighStopBreachable:
    """
    Low_Stop / High_Stop 使用 .shift(1)，確保今日 Close 確實可突破停損值。
    若無 shift，Close[t] ≥ Low[t] ≥ Low_Stop[t]，Phase 1 永遠無法觸發。
    """

    def test_declining_close_can_breach_low_stop(self):
        """下跌序列中，至少存在一天 Close < Low_Stop（Phase 1 多方出場可被觸發）"""
        cfg    = TradingConfig(stop_window=5)
        result = Indicators.add_all(_declining_df(n=40), cfg)
        valid  = result.dropna(subset=["Low_Stop"])
        assert (valid["Close"] < valid["Low_Stop"]).any(), (
            "下跌序列中 Close 從未低於 Low_Stop——"
            "請確認 indicators.py 的 Low_Stop 包含 .shift(1)"
        )

    def test_rising_close_can_breach_high_stop(self):
        """上漲序列中，至少存在一天 Close > High_Stop（Phase 1 空方出場可被觸發）"""
        cfg    = TradingConfig(stop_window=5)
        result = Indicators.add_all(_rising_df(n=40), cfg)
        valid  = result.dropna(subset=["High_Stop"])
        assert (valid["Close"] > valid["High_Stop"]).any(), (
            "上漲序列中 Close 從未高於 High_Stop——"
            "請確認 indicators.py 的 High_Stop 包含 .shift(1)"
        )

    def test_low_stop_excludes_todays_low(self):
        """Low_Stop[t] 不包含今日的 Low（若含今日，Close < Low_Stop 在邏輯上不可能）"""
        cfg    = TradingConfig(stop_window=3)
        df     = _declining_df(n=20)
        result = Indicators.add_all(df, cfg)

        # 對每一列：Low_Stop[t] 應 <= Low[t-1]（前一日最低）
        # 若 Low_Stop 含今日，Low_Stop[t] <= Low[t] <= Close[t]，永遠不會觸發
        for i in range(4, len(df)):
            idx      = df.index[i]
            prev_idx = df.index[i - 1]
            low_stop = result.loc[idx, "Low_Stop"]
            if pd.isna(low_stop):
                continue
            prev_low = df.loc[prev_idx, "Low"]
            # 確認 Low_Stop 的值 <= 前一日的 Low（不含今日高點）
            assert low_stop <= prev_low + 1e-9, (
                f"Low_Stop[{idx.date()}]={low_stop:.4f} 超過前日 Low={prev_low:.4f}，"
                "可能誤含今日資料"
            )


# ══════════════════════════════════════════════
# 2. 用真實指標值驗證兩段出場
# ══════════════════════════════════════════════

class TestPhaseExitWithRealIndicators:
    """
    用 Indicators.add_all() 計算出的真實指標（含 shift(1) 的 Low_Stop / High_Stop），
    餵進 SignalGenerator，驗證兩段出場都能正確觸發。
    """

    def test_phase1_long_exit_triggers_via_low_stop(self):
        """
        下跌序列 + trail_high == entry_price（未獲利）→ Phase 1：
        找到 Close < Low_Stop 的列，確認 long_exit() 回傳 True。
        """
        cfg    = TradingConfig(stop_window=5, atr_period=5)
        result = Indicators.add_all(_declining_df(n=40), cfg)

        valid        = result.dropna(subset=["Low_Stop", "ATR"])
        breach_rows  = valid[valid["Close"] < valid["Low_Stop"]]
        assert not breach_rows.empty, "測試前提：下跌序列需有 Close < Low_Stop 的列"

        row         = breach_rows.iloc[0]
        entry_price = row["Close"] + 5.0   # trail_high == entry_price（未曾獲利）

        assert SignalGenerator.long_exit(
            row, trail_high=entry_price, atr_mult=3.0, entry_price=entry_price
        ), "Phase 1 多方：Close < Low_Stop 應觸發出場"

    def test_phase2_long_exit_triggers_via_atr(self):
        """
        先漲後跌序列 + trail_high > entry_price（已獲利）→ Phase 2：
        跌幅超過 atr_mult × ATR 後，long_exit() 應回傳 True。
        """
        cfg    = TradingConfig(stop_window=5, atr_period=5, atr_multiplier=2.0)
        df     = _rally_then_drop_df(n_rise=25, n_drop=15)
        result = Indicators.add_all(df, cfg)

        entry_price = df["Close"].iloc[0]
        trail_high  = df["High"].max()  # 追蹤高點 = 序列最高點

        valid = result.dropna(subset=["ATR"])
        drop_section = valid.iloc[-10:]   # 跌段末尾

        triggered = any(
            SignalGenerator.long_exit(
                row, trail_high=trail_high,
                atr_mult=cfg.atr_multiplier,
                entry_price=entry_price,
            )
            for _, row in drop_section.iterrows()
        )
        assert triggered, (
            "Phase 2 多方：跌段末尾應有 Close < trail_high - 2×ATR，但未觸發出場"
        )

    def test_phase1_short_exit_triggers_via_high_stop(self):
        """
        上漲序列 + trail_low == entry_price（未獲利）→ Phase 1 空方：
        Close > High_Stop 應觸發出場。
        """
        cfg    = TradingConfig(stop_window=5, atr_period=5)
        result = Indicators.add_all(_rising_df(n=40), cfg)

        valid       = result.dropna(subset=["High_Stop", "ATR"])
        breach_rows = valid[valid["Close"] > valid["High_Stop"]]
        assert not breach_rows.empty, "測試前提：上漲序列需有 Close > High_Stop 的列"

        row         = breach_rows.iloc[0]
        entry_price = row["Close"] - 5.0   # trail_low == entry_price（未曾獲利）

        assert SignalGenerator.short_exit(
            row, trail_low=entry_price, atr_mult=3.0, entry_price=entry_price
        ), "Phase 1 空方：Close > High_Stop 應觸發出場"

    def test_phase2_short_exit_triggers_via_atr(self):
        """
        先跌後漲序列 + trail_low < entry_price（已獲利）→ Phase 2 空方：
        漲幅超過 atr_mult × ATR 後，short_exit() 應回傳 True。
        """
        cfg    = TradingConfig(stop_window=5, atr_period=5, atr_multiplier=2.0)
        df     = _drop_then_rise_df(n_drop=25, n_rise=15)
        result = Indicators.add_all(df, cfg)

        entry_price = df["Close"].iloc[0]
        trail_low   = df["Low"].min()   # 追蹤低點 = 序列最低點

        valid        = result.dropna(subset=["ATR"])
        rise_section = valid.iloc[-10:]

        triggered = any(
            SignalGenerator.short_exit(
                row, trail_low=trail_low,
                atr_mult=cfg.atr_multiplier,
                entry_price=entry_price,
            )
            for _, row in rise_section.iterrows()
        )
        assert triggered, (
            "Phase 2 空方：漲段末尾應有 Close > trail_low + 2×ATR，但未觸發出場"
        )


# ══════════════════════════════════════════════
# 3. Phase 邊界條件
# ══════════════════════════════════════════════

class TestPhaseBoundaryCondition:
    """
    trail_high == entry_price 是 Phase 1 / Phase 2 的精確切換點。
    同一根 K 棒，因 trail_high 微小差異，出場行為截然不同。
    """

    @staticmethod
    def _row(close: float, low_stop: float, atr: float) -> pd.Series:
        return pd.Series({
            "Close": close, "Low_Stop": low_stop, "ATR": atr,
            "High_Stop": close + 20.0,
            "Open": close, "High": close + 0.5, "Low": close - 0.5,
            "High_N": close + 10.0, "Low_N": close - 10.0, "MACD": 1.0,
        })

    def test_trail_equal_entry_activates_phase1(self):
        """trail_high == entry_price → Phase 1：Close < Low_Stop 觸發出場"""
        row = self._row(close=93.0, low_stop=94.0, atr=5.0)
        # Phase 1: 93 < 94 → 出場
        assert SignalGenerator.long_exit(
            row, trail_high=100.0, atr_mult=3.0, entry_price=100.0
        )

    def test_trail_above_entry_activates_phase2(self):
        """trail_high > entry_price → Phase 2：改看 ATR 追蹤停損"""
        row = self._row(close=93.0, low_stop=94.0, atr=5.0)
        # Phase 2: stop = 100.01 - 3×5 = 85.01；Close=93 > 85.01 → 不出場
        # （即使 Close=93 < Low_Stop=94，Phase 2 不參考 Low_Stop）
        assert not SignalGenerator.long_exit(
            row, trail_high=100.01, atr_mult=3.0, entry_price=100.0
        )

    def test_phase1_ignores_atr_stop_when_low_stop_not_breached(self):
        """
        Phase 1 時，若 Close > Low_Stop，就算 ATR 追蹤停損被突破也不出場。
        entry=100, trail_high=100, ATR=20 → atr_stop=40；
        Low_Stop=80, Close=84：ATR stop 已被突破，但 Low_Stop 未被突破 → 不出場。
        """
        row = self._row(close=84.0, low_stop=80.0, atr=20.0)
        # Phase 1: 84 > 80 → 不出場（即使 atr_stop=40 遠低於 84）
        assert not SignalGenerator.long_exit(
            row, trail_high=100.0, atr_mult=3.0, entry_price=100.0
        )

    def test_phase_switch_changes_exit_behavior_on_same_candle(self):
        """
        同一根 K 棒（Close=84, Low_Stop=80, ATR=5）：
          Phase 1（trail_high=100） → 不出場（84 > Low_Stop=80）
          Phase 2（trail_high=100.01）→ 出場（84 < atr_stop=100.01-15=85.01）
        """
        row = self._row(close=84.0, low_stop=80.0, atr=5.0)

        # Phase 1：Low_Stop=80 未被穿越 → 不出場
        assert not SignalGenerator.long_exit(
            row, trail_high=100.0, atr_mult=3.0, entry_price=100.0
        )
        # Phase 2：atr_stop ≈ 85 被穿越 → 出場
        assert SignalGenerator.long_exit(
            row, trail_high=100.01, atr_mult=3.0, entry_price=100.0
        )

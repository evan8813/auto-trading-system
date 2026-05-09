"""
test_two_phase_exit.py
──────────────────────
驗證 ATR 追蹤停損（從進場第一天開始，無 Phase 區分）。

出場條件：
  做多：Close < trail_high - atr_mult × ATR
  做空：Close > trail_low  + atr_mult × ATR

trail_high 初始 = 進場價，每日只往上更新（停損線只升不降）。
trail_low  初始 = 進場價，每日只往下更新（停損線只降不升）。

測試結構：
  1. TestLowHighStopBreachable    ── 確認指標計算正確（ATR 不為 0）
  2. TestAtrTrailWithRealIndicators ── 用 add_all() 真實指標驗證出場
  3. TestTrailHighLowBoundary     ── trail 邊界值行為
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
    dates  = pd.bdate_range("2023-01-01", periods=n)
    closes = np.linspace(start, start * 0.1, n)
    lows   = closes - 0.3
    highs  = closes + 0.3
    return pd.DataFrame(
        {"Open": closes, "High": highs, "Low": lows, "Close": closes,
         "Volume": 10_000.0, "Amount": closes * 10_000.0},
        index=dates,
    )


def _rising_df(n: int = 40, start: float = 20.0) -> pd.DataFrame:
    dates  = pd.bdate_range("2023-01-01", periods=n)
    closes = np.linspace(start, start * 4.0, n)
    lows   = closes - 0.3
    highs  = closes + 0.3
    return pd.DataFrame(
        {"Open": closes, "High": highs, "Low": lows, "Close": closes,
         "Volume": 10_000.0, "Amount": closes * 10_000.0},
        index=dates,
    )


def _rally_then_drop_df(n_rise: int = 25, n_drop: int = 15,
                        start: float = 50.0) -> pd.DataFrame:
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
# 1. 指標基本正確性
# ══════════════════════════════════════════════

class TestLowHighStopBreachable:
    """確認 ATR 指標正常計算，不為 0 或全 NaN。"""

    def test_atr_non_zero_after_warmup(self):
        """ATR 暖機後應 > 0"""
        cfg    = TradingConfig(atr_period=5)
        result = Indicators.add_all(_declining_df(n=40), cfg)
        valid  = result.dropna(subset=["ATR"])
        assert (valid["ATR"] > 0).all()

    def test_atr_available_for_rising(self):
        cfg    = TradingConfig(atr_period=5)
        result = Indicators.add_all(_rising_df(n=40), cfg)
        valid  = result.dropna(subset=["ATR"])
        assert len(valid) > 0

    def test_low_stop_shift_excludes_today(self):
        """Low_Stop[t] 應 <= Low[t-1]（shift(1) 正確）"""
        cfg    = TradingConfig(stop_window=3)
        df     = _declining_df(n=20)
        result = Indicators.add_all(df, cfg)
        for i in range(4, len(df)):
            idx      = df.index[i]
            prev_idx = df.index[i - 1]
            low_stop = result.loc[idx, "Low_Stop"]
            if pd.isna(low_stop):
                continue
            assert low_stop <= df.loc[prev_idx, "Low"] + 1e-9


# ══════════════════════════════════════════════
# 2. 用真實指標驗證出場
# ══════════════════════════════════════════════

class TestAtrTrailWithRealIndicators:
    """用 Indicators.add_all() 計算真實 ATR，驗證 ATR 追蹤停損觸發。"""

    def test_long_exit_triggers_on_declining_stock(self):
        """下跌序列：trail_high = entry，stop = entry - 3×ATR，Close 最終跌穿"""
        cfg    = TradingConfig(atr_period=5, atr_multiplier=2.0)
        result = Indicators.add_all(_declining_df(n=40), cfg)
        valid  = result.dropna(subset=["ATR"])

        entry_price = valid["Close"].iloc[0]
        trail_high  = entry_price   # 一路下跌，trail_high 從未更新

        triggered = any(
            SignalGenerator.long_exit(row, trail_high=trail_high, atr_mult=cfg.atr_multiplier)
            for _, row in valid.iterrows()
        )
        assert triggered, "下跌序列中 ATR 追蹤停損應觸發"

    def test_long_exit_triggers_after_rally_drop(self):
        """先漲後跌序列：trail_high 更新至高點，跌段觸發 ATR 停損"""
        cfg    = TradingConfig(atr_period=5, atr_multiplier=2.0)
        df     = _rally_then_drop_df(n_rise=25, n_drop=15)
        result = Indicators.add_all(df, cfg)
        valid  = result.dropna(subset=["ATR"])

        trail_high   = df["High"].max()
        entry_price  = df["Close"].iloc[0]
        drop_section = valid.iloc[-10:]

        triggered = any(
            SignalGenerator.long_exit(row, trail_high=trail_high, atr_mult=cfg.atr_multiplier)
            for _, row in drop_section.iterrows()
        )
        assert triggered, "跌段末尾應觸發 ATR 追蹤停損"

    def test_short_exit_triggers_on_rising_stock(self):
        """上漲序列：trail_low = entry，stop = entry + 3×ATR，Close 最終突破"""
        cfg    = TradingConfig(atr_period=5, atr_multiplier=2.0)
        result = Indicators.add_all(_rising_df(n=40), cfg)
        valid  = result.dropna(subset=["ATR"])

        entry_price = valid["Close"].iloc[0]
        trail_low   = entry_price

        triggered = any(
            SignalGenerator.short_exit(row, trail_low=trail_low, atr_mult=cfg.atr_multiplier)
            for _, row in valid.iterrows()
        )
        assert triggered, "上漲序列中 ATR 追蹤停損應觸發"

    def test_short_exit_triggers_after_drop_rise(self):
        """先跌後漲序列：trail_low 更新至低點，漲段觸發 ATR 停損"""
        cfg    = TradingConfig(atr_period=5, atr_multiplier=2.0)
        df     = _drop_then_rise_df(n_drop=25, n_rise=15)
        result = Indicators.add_all(df, cfg)
        valid  = result.dropna(subset=["ATR"])

        trail_low    = df["Low"].min()
        rise_section = valid.iloc[-10:]

        triggered = any(
            SignalGenerator.short_exit(row, trail_low=trail_low, atr_mult=cfg.atr_multiplier)
            for _, row in rise_section.iterrows()
        )
        assert triggered, "漲段末尾應觸發 ATR 追蹤停損"


# ══════════════════════════════════════════════
# 3. trail_high / trail_low 邊界條件
# ══════════════════════════════════════════════

class TestTrailHighLowBoundary:
    """trail_high / trail_low 的邊界行為。"""

    @staticmethod
    def _row(close: float, atr: float) -> pd.Series:
        return pd.Series({
            "Close": close, "ATR": atr,
            "Open": close, "High": close + 0.5, "Low": close - 0.5,
            "High_N": close + 10.0, "Low_N": close - 10.0, "MACD": 1.0,
        })

    def test_long_exit_at_entry_trail_tight(self):
        """trail_high = entry = 100，stop = 100 - 3×5 = 85，Close=84 → 出場"""
        row = self._row(close=84.0, atr=5.0)
        assert SignalGenerator.long_exit(row, trail_high=100.0, atr_mult=3.0)

    def test_long_exit_at_entry_trail_just_safe(self):
        """trail_high = entry = 100，stop=85，Close=86 → 不出場"""
        row = self._row(close=86.0, atr=5.0)
        assert not SignalGenerator.long_exit(row, trail_high=100.0, atr_mult=3.0)

    def test_long_exit_higher_trail_raises_stop(self):
        """trail_high 提升後，同一 Close 可能觸發：trail=110，stop=95，Close=94 → 出場"""
        row = self._row(close=94.0, atr=5.0)
        assert SignalGenerator.long_exit(row, trail_high=110.0, atr_mult=3.0)

    def test_short_exit_at_entry_trail_tight(self):
        """trail_low = entry = 100，stop = 100 + 3×5 = 115，Close=116 → 出場"""
        row = self._row(close=116.0, atr=5.0)
        assert SignalGenerator.short_exit(row, trail_low=100.0, atr_mult=3.0)

    def test_short_exit_at_entry_trail_just_safe(self):
        """trail_low = entry = 100，stop=115，Close=114 → 不出場"""
        row = self._row(close=114.0, atr=5.0)
        assert not SignalGenerator.short_exit(row, trail_low=100.0, atr_mult=3.0)

    def test_short_exit_lower_trail_lowers_stop(self):
        """trail_low 下降後，同一 Close 可能不觸發：trail=90，stop=105，Close=104 → 不出場"""
        row = self._row(close=104.0, atr=5.0)
        assert not SignalGenerator.short_exit(row, trail_low=90.0, atr_mult=3.0)

"""
signal_generator.py
───────────────────
職責：根據指標數值判斷「進場 / 出場訊號」，回傳 True / False。
      只做邏輯判斷，不做下單、部位計算或狀態修改。

擴充指引：
  - 新增策略 → 新增 @staticmethod，在 Backtester._check_entries() 裡呼叫即可。
  - 保持純函式風格：輸入 pd.Series，輸出 bool，無副作用。

進出場規格：
  做多進場：收盤 > 前日 High_N(20)  且  MACD > 0  且  MACD 今 > MACD 昨
  做多出場（兩段式）：
    Phase 1  trail_high <= entry_price  →  收盤 < Low_Stop（10 日低）
    Phase 2  trail_high >  entry_price  →  收盤 < trail_high - atr_mult × ATR
  做空進場：收盤 < 前日 Low_N(20)   且  MACD < 0  且  MACD 今 < MACD 昨
  做空出場（兩段式）：
    Phase 1  trail_low >= entry_price  →  收盤 > High_Stop（10 日高）
    Phase 2  trail_low <  entry_price  →  收盤 > trail_low + atr_mult × ATR
"""

from __future__ import annotations

import pandas as pd


class SignalGenerator:
    """
    提供四個靜態方法：
      long_entry  / long_exit
      short_entry / short_exit
    """

    @staticmethod
    def long_entry(row: pd.Series, prev_row: pd.Series) -> bool:
        """
        做多進場條件：
          1. 當日收盤突破「前一日」的 N 日高點
          2. MACD > 0（多頭區間）
          3. MACD 今 > MACD 昨（MACD 上升中）
        """
        if any(pd.isna([row["Close"], prev_row["High_N"],
                        row["MACD"], prev_row["MACD"]])):
            return False
        breakout     = row["Close"]  > prev_row["High_N"]
        macd_pos     = row["MACD"]   > 0
        macd_rising  = row["MACD"]   > prev_row["MACD"]
        return breakout and macd_pos and macd_rising

    @staticmethod
    def long_exit(
        row:         pd.Series,
        trail_high:  float,
        atr_mult:    float,
        entry_price: float,
    ) -> bool:
        """
        做多出場條件（兩段式追蹤停損）：
          Phase 1（尚未獲利，trail_high <= entry_price）：
            收盤 < Low_Stop（10 日低點）
          Phase 2（已有獲利，trail_high > entry_price）：
            收盤 < trail_high - atr_mult × ATR
        """
        if pd.isna(row["Close"]):
            return False
        if trail_high > entry_price:
            if pd.isna(row["ATR"]):
                return False
            return row["Close"] < trail_high - atr_mult * row["ATR"]
        else:
            if pd.isna(row["Low_Stop"]):
                return False
            return row["Close"] < row["Low_Stop"]

    @staticmethod
    def short_entry(row: pd.Series, prev_row: pd.Series) -> bool:
        """
        做空進場條件：
          1. 當日收盤跌破「前一日」的 N 日低點
          2. MACD < 0（空頭區間）
          3. MACD 今 < MACD 昨（MACD 下降中）
        """
        if any(pd.isna([row["Close"], prev_row["Low_N"],
                        row["MACD"], prev_row["MACD"]])):
            return False
        breakdown     = row["Close"]  < prev_row["Low_N"]
        macd_neg      = row["MACD"]   < 0
        macd_falling  = row["MACD"]   < prev_row["MACD"]
        return breakdown and macd_neg and macd_falling

    @staticmethod
    def short_exit(
        row:         pd.Series,
        trail_low:   float,
        atr_mult:    float,
        entry_price: float,
    ) -> bool:
        """
        做空出場條件（兩段式追蹤停損）：
          Phase 1（尚未獲利，trail_low >= entry_price）：
            收盤 > High_Stop（10 日高點）
          Phase 2（已有獲利，trail_low < entry_price）：
            收盤 > trail_low + atr_mult × ATR
        """
        if pd.isna(row["Close"]):
            return False
        if trail_low < entry_price:
            if pd.isna(row["ATR"]):
                return False
            return row["Close"] > trail_low + atr_mult * row["ATR"]
        else:
            if pd.isna(row["High_Stop"]):
                return False
            return row["Close"] > row["High_Stop"]

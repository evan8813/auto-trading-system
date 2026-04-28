"""
signal_generator.py
───────────────────
職責：根據指標數值判斷「進場 / 出場訊號」，回傳 True / False。
      只做邏輯判斷，不做下單、部位計算或狀態修改。

擴充指引：
  - 新增策略（例如 RSI 超賣進場）→ 新增 @staticmethod，
    在 Backtester._check_entries() 裡呼叫即可。
  - 保持純函式風格：輸入 pd.Series，輸出 bool，無副作用。
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
          2. MA_Fast > MA_Slow（趨勢向上確認）
        """
        if any(pd.isna([row["Close"], prev_row["High_N"],
                        row["MA_Fast"], row["MA_Slow"]])):
            return False
        breakout = row["Close"] > prev_row["High_N"]
        trend    = row["MA_Fast"] > row["MA_Slow"]
        return breakout and trend

    @staticmethod
    def long_exit(row: pd.Series, trail_high: float, atr_mult: float) -> bool:
        """
        做多出場條件（追蹤停損）：
          收盤 < 持倉期間最高價 − atr_mult × ATR
        """
        if pd.isna(row["ATR"]) or pd.isna(row["Close"]):
            return False
        stop = trail_high - atr_mult * row["ATR"]
        return row["Close"] < stop

    @staticmethod
    def short_entry(row: pd.Series, prev_row: pd.Series) -> bool:
        """
        做空進場條件：
          1. 當日收盤跌破「前一日」的 N 日低點
          2. MA_Fast < MA_Slow（趨勢向下確認）
        """
        if any(pd.isna([row["Close"], prev_row["Low_N"],
                        row["MA_Fast"], row["MA_Slow"]])):
            return False
        breakdown = row["Close"] < prev_row["Low_N"]
        trend     = row["MA_Fast"] < row["MA_Slow"]
        return breakdown and trend

    @staticmethod
    def short_exit(row: pd.Series, trail_low: float, atr_mult: float) -> bool:
        """
        做空出場條件（追蹤停損）：
          收盤 > 持倉期間最低價 + atr_mult × ATR
        """
        if pd.isna(row["ATR"]) or pd.isna(row["Close"]):
            return False
        stop = trail_low + atr_mult * row["ATR"]
        return row["Close"] > stop

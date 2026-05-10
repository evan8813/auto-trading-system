"""
signal_generator.py
───────────────────
職責：根據指標數值判斷「進場 / 出場訊號」，回傳 True / False。
      只做邏輯判斷，不做下單、部位計算或狀態修改。

擴充指引：
  - 新增策略 → 新增 @staticmethod，在 Backtester._check_entries() 裡呼叫即可。
  - 保持純函式風格：輸入 pd.Series，輸出 bool，無副作用。

進出場規格：
  做多進場：收盤 > 前日 High_N(20)  且  MACD > 0  且  MACD 今 > MACD 昨  且  量 > Vol_MA20 × vol_mult
  做多出場：收盤 < trail_high - atr_mult × ATR（ATR 追蹤停損，從第一天開始）
  做空進場：收盤 < 前日 Low_N(20)   且  MACD < 0  且  MACD 今 < MACD 昨  且  量 > Vol_MA20 × vol_mult
  做空出場：收盤 > trail_low + atr_mult × ATR（ATR 追蹤停損，從第一天開始）
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
    def long_entry(row: pd.Series, prev_row: pd.Series, vol_mult: float = 1.5) -> bool:
        """
        做多進場條件：
          1. 當日收盤突破「前一日」的 N 日高點
          2. MACD > 0（多頭區間）
          3. MACD 今 > MACD 昨（MACD 上升中）
          4. 當日成交量 > 20 日均量 × vol_mult（突破量能確認）
        """
        if any(pd.isna([row["Close"], prev_row["High_N"],
                        row["MACD"], prev_row["MACD"],
                        row["Volume"], row["Vol_MA20"]])):
            return False
        breakout     = row["Close"]  > prev_row["High_N"]
        macd_pos     = row["MACD"]   > 0
        macd_rising  = row["MACD"]   > prev_row["MACD"]
        vol_surge    = row["Volume"] > row["Vol_MA20"] * vol_mult
        return breakout and macd_pos and macd_rising and vol_surge

    @staticmethod
    def long_exit(
        row:        pd.Series,
        trail_high: float,
        atr_mult:   float,
    ) -> bool:
        """
        做多出場條件（ATR 追蹤停損，從第一天開始）：
          收盤 < trail_high - atr_mult × ATR
        trail_high 初始值 = 進場價，每日只往上更新，確保停損線只升不降。
        """
        if pd.isna(row["Close"]) or pd.isna(row["ATR"]):
            return False
        return row["Close"] < trail_high - atr_mult * row["ATR"]

    @staticmethod
    def short_entry(row: pd.Series, prev_row: pd.Series, vol_mult: float = 1.5) -> bool:
        """
        做空進場條件：
          1. 當日收盤跌破「前一日」的 N 日低點
          2. MACD < 0（空頭區間）
          3. MACD 今 < MACD 昨（MACD 下降中）
          4. 當日成交量 > 20 日均量 × vol_mult（跌破量能確認）
        """
        if any(pd.isna([row["Close"], prev_row["Low_N"],
                        row["MACD"], prev_row["MACD"],
                        row["Volume"], row["Vol_MA20"]])):
            return False
        breakdown     = row["Close"]  < prev_row["Low_N"]
        macd_neg      = row["MACD"]   < 0
        macd_falling  = row["MACD"]   < prev_row["MACD"]
        vol_surge     = row["Volume"] > row["Vol_MA20"] * vol_mult
        return breakdown and macd_neg and macd_falling and vol_surge

    @staticmethod
    def short_exit(
        row:       pd.Series,
        trail_low: float,
        atr_mult:  float,
    ) -> bool:
        """
        做空出場條件（ATR 追蹤停損，從第一天開始）：
          收盤 > trail_low + atr_mult × ATR
        trail_low 初始值 = 進場價，每日只往下更新，確保停損線只降不升。
        """
        if pd.isna(row["Close"]) or pd.isna(row["ATR"]):
            return False
        return row["Close"] > trail_low + atr_mult * row["ATR"]

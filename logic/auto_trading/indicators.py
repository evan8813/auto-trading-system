"""
indicators.py
─────────────
職責：在 OHLCV DataFrame 上計算並附加技術指標欄位。
      只做「計算」，不做篩選或訊號判斷。

擴充指引：
  - 新增指標（例如 RSI、MACD）→ 在 Indicators 加 @staticmethod，
    再於 add_all() 呼叫即可。
  - 所有指標函式輸入 pd.DataFrame / pd.Series，輸出 pd.Series，
    保持無副作用的純函式風格。
"""

from __future__ import annotations

import pandas as pd

from config import TradingConfig


class Indicators:
    """在 DataFrame 上計算並附加所有技術指標"""

    # ── 單一指標（純函式）────────────────────

    @staticmethod
    def atr(df: pd.DataFrame, period: int) -> pd.Series:
        """Average True Range"""
        high, low, close = df["High"], df["Low"], df["Close"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def sma(series: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(window).mean()

    @staticmethod
    def rolling_max(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window).max()

    @staticmethod
    def rolling_min(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window).min()

    # ── 批次附加（供 Backtester 呼叫）────────

    @staticmethod
    def add_all(df: pd.DataFrame, cfg: TradingConfig) -> pd.DataFrame:
        """
        複製 DataFrame 並附加所有策略所需指標欄位。
        原始 df 不被修改。
        """
        df = df.copy()
        c  = df["Close"]

        df["ATR"]      = Indicators.atr(df, cfg.atr_period)
        df["MA_Fast"]  = Indicators.sma(c, cfg.ma_fast)
        df["MA_Slow"]  = Indicators.sma(c, cfg.ma_slow)
        df["High_N"]   = Indicators.rolling_max(c, cfg.breakout_window)
        df["Low_N"]    = Indicators.rolling_min(c, cfg.breakout_window)
        df["High_52W"] = Indicators.rolling_max(c, cfg.week52)
        df["Low_52W"]  = Indicators.rolling_min(c, cfg.week52)

        # 成交金額（若有 Amount 欄使用之，否則以 Volume × Close 估算）
        if "Amount" in df.columns:
            df["Avg_Amount_20"] = df["Amount"].rolling(20).mean()
        else:
            df["Avg_Amount_20"] = (df["Volume"] * df["Close"]).rolling(20).mean()

        return df

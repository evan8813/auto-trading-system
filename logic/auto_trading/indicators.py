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
    def macd(series: pd.Series, fast: int, slow: int, signal: int
             ) -> tuple[pd.Series, pd.Series]:
        """MACD 線與 Signal 線（EMA-based）"""
        ema_fast   = series.ewm(span=fast,   adjust=False).mean()
        ema_slow   = series.ewm(span=slow,   adjust=False).mean()
        macd_line  = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line

    @staticmethod
    def add_all(df: pd.DataFrame, cfg: TradingConfig) -> pd.DataFrame:
        """
        複製 DataFrame 並附加所有策略所需指標欄位。
        原始 df 不被修改。
        """
        df = df.copy()
        c  = df["Close"]

        df["ATR"]       = Indicators.atr(df, cfg.atr_period)
        df["High_N"]    = Indicators.rolling_max(c, cfg.breakout_window)
        df["Low_N"]     = Indicators.rolling_min(c, cfg.breakout_window)
        # .shift(1): stop price references yesterday's rolling window, so today's
        # candle can actually breach it (Close < Low_Stop is otherwise impossible
        # because Low_Stop[t] <= Low[t] <= Close[t] without the shift).
        df["High_Stop"] = Indicators.rolling_max(df["High"], cfg.stop_window).shift(1)
        df["Low_Stop"]  = Indicators.rolling_min(df["Low"],  cfg.stop_window).shift(1)
        df["High_52W"]  = Indicators.rolling_max(df["High"], cfg.week52)
        df["Low_52W"]   = Indicators.rolling_min(df["Low"],  cfg.week52)

        macd_line, signal_line = Indicators.macd(
            c, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
        df["MACD"]        = macd_line
        df["MACD_signal"] = signal_line

        # 成交金額（若有 Amount 欄使用之，否則以 Volume × Close 估算）
        if "Amount" in df.columns:
            df["Avg_Amount_20"] = df["Amount"].rolling(20).mean()
        else:
            df["Avg_Amount_20"] = (df["Volume"] * df["Close"]).rolling(20).mean()

        # ROC（2週/5週/7週 平均動能）
        df["ROC_10"] = c.pct_change(periods=10) * 100   # 2 週
        df["ROC_25"] = c.pct_change(periods=25) * 100   # 5 週
        df["ROC_35"] = c.pct_change(periods=35) * 100   # 7 週
        df["ROC_avg"] = (df["ROC_10"] + df["ROC_25"] + df["ROC_35"]) / 3

        return df

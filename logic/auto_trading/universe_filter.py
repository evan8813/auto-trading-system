"""
universe_filter.py
──────────────────
職責：每個交易日篩選出「符合條件的候選股票清單」。
      只負責篩選，不負責訊號判斷或部位計算。

擴充指引：
  - 新增篩選條件（例如排除特定產業、市值下限）→ 在 filter() 裡加條件
  - 若想支援多套篩選規則，可讓 UniverseFilter 接受條件清單（策略模式）
"""

from __future__ import annotations

import pandas as pd

from config import TradingConfig


class UniverseFilter:
    """
    每日執行股票篩選，回傳當日符合條件的股票代號清單。

    篩選條件：
      1. 20 日平均成交金額 >= min_avg_amount（流動性門檻）
      2. 今日 High > 昨日 High_52W 或 今日 Low < 昨日 Low_52W（52 週突破）
      3. 收盤價 × 1000 <= max_trade_cost（單倉買得起至少 1 張）
    """

    def __init__(self, cfg: TradingConfig) -> None:
        self.cfg = cfg
        self._max_price: float = cfg.max_trade_cost / 1000

    def filter(
        self,
        data_dict: dict[str, pd.DataFrame],
        date:      pd.Timestamp,
    ) -> list[str]:
        """
        回傳在指定日期符合所有篩選條件的股票代號。

        Parameters
        ----------
        data_dict : 已附加指標的 OHLCV dict（key = ticker）
        date      : 當日日期

        Returns
        -------
        list[str]  符合條件的股票代號清單
        """
        candidates: list[str] = []

        for ticker, df in data_dict.items():
            if date not in df.index:
                continue

            idx = df.index.get_loc(date)
            if idx == 0:
                continue  # 無前一日，無法判斷 52 週突破

            row      = df.iloc[idx]
            prev_row = df.iloc[idx - 1]

            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]

            if not self._liquidity_ok(row):
                continue
            if not self._52w_breakout(row, prev_row):
                continue
            if not self._affordable(row):
                continue

            candidates.append(ticker)

        return candidates

    # ── 私有篩選條件（每個條件獨立一個方法，方便單獨測試）──

    def _liquidity_ok(self, row: pd.Series) -> bool:
        """20 日平均成交金額是否達門檻"""
        val = row.get("Avg_Amount_20")
        if pd.isna(val):
            return False
        return float(val) >= self.cfg.min_avg_amount

    def _52w_breakout(self, row: pd.Series, prev_row: pd.Series) -> bool:
        """今日 High > 昨日 High_52W 或 今日 Low < 昨日 Low_52W"""
        high_52w = prev_row.get("High_52W")
        low_52w  = prev_row.get("Low_52W")

        breakout_high = (not pd.isna(high_52w)) and float(row["High"]) > float(high_52w)
        breakout_low  = (not pd.isna(low_52w))  and float(row["Low"])  < float(low_52w)

        return breakout_high or breakout_low

    def _affordable(self, row: pd.Series) -> bool:
        """收盤 × 1000 <= max_trade_cost（單倉最高成本）"""
        close = row.get("Close")
        if pd.isna(close):
            return False
        return float(close) <= self._max_price

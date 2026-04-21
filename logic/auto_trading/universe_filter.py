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
      2. 收盤價在 52 週最高價的 90% 以上（近強勢新高）
    """

    def __init__(self, cfg: TradingConfig) -> None:
        self.cfg = cfg

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

            row = df.loc[date]

            if not self._liquidity_ok(row):
                continue
            if not self._near_52w_high(row):
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

    def _near_52w_high(self, row: pd.Series) -> bool:
        """收盤是否在 52 週高點的 90% 以上"""
        high_52w = row.get("High_52W")
        if pd.isna(high_52w) or float(high_52w) <= 0:
            return False
        return float(row["Close"]) >= float(high_52w) * 0.90

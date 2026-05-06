"""
universe_filter.py
──────────────────
職責：每個交易日篩選出「符合條件的候選股票清單」。
      只負責篩選，不負責訊號判斷或部位計算。

篩選條件（依 CLAUDE.md 規格）：
  1. 20 日平均成交金額 >= min_avg_amount（流動性門檻）
  2. 52 週突破：今日 High > 昨日 High_52W（多方）或 Low < 昨日 Low_52W（空方）
  3. 收盤 × 1000 <= equity / max_positions（單倉動態上限，買得起至少 1 張）
  4. 股價下限：做多時收盤 >= min_long_price；做空時收盤 >= min_short_price
  5. 大盤環境：加權指數 > 200 日 EMA → 只做多；< 200 日 EMA → 只做空

擴充指引：
  - 新增篩選條件（例如排除特定產業、市值下限）→ 在 filter() 裡加條件並新增私有方法
  - 若想支援多套篩選規則，可讓 UniverseFilter 接受條件清單（策略模式）
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from config import TradingConfig


class UniverseFilter:
    """
    每日執行股票篩選，回傳當日符合條件的股票代號清單。

    篩選條件：
      1. 20 日平均成交金額 >= min_avg_amount（流動性門檻）
      2. 52 週突破（多方高點突破 / 空方低點跌破）
      3. 收盤 × 1000 <= equity / max_positions（單倉動態上限）
      4. 股價下限：做多 >= min_long_price；做空 >= min_short_price
      5. 大盤 200 日 EMA 多空環境過濾
    """

    def __init__(self, cfg: TradingConfig) -> None:
        self.cfg = cfg
        self._taiex: Optional[pd.DataFrame] = self._load_taiex()

    def _load_taiex(self) -> Optional[pd.DataFrame]:
        """載入加權指數 CSV，計算 200 日 EMA"""
        if not self.cfg.taiex_csv_path:
            return None
        path = Path(self.cfg.taiex_csv_path)
        if not path.exists():
            return None
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        df["EMA_200"] = df["Close"].ewm(span=self.cfg.taiex_ema_period, adjust=False).mean()
        return df

    def filter(
        self,
        data_dict: dict[str, pd.DataFrame],
        date:      pd.Timestamp,
        equity:    float,
    ) -> list[str]:
        """
        回傳在指定日期符合所有篩選條件的股票代號。

        Parameters
        ----------
        data_dict : 已附加指標的 OHLCV dict（key = ticker）
        date      : 當日日期
        equity    : 當日總淨值（用於計算動態單倉上限）

        Returns
        -------
        list[str]  符合條件的股票代號清單
        """
        max_price = equity / (self.cfg.max_positions * 1000)
        regime    = self._market_regime(date)
        candidates: list[str] = []

        for ticker, df in data_dict.items():
            if date not in df.index:
                continue

            loc = df.index.get_loc(date)
            # get_loc 在重複索引時回傳 slice 或 ndarray，統一轉成整數
            if isinstance(loc, slice):
                idx = loc.stop - 1
            elif hasattr(loc, '__len__'):
                import numpy as np
                idx = int(np.where(loc)[0][-1])
            else:
                idx = int(loc)

            if idx == 0:
                continue  # 無前一日，無法判斷 52 週突破

            row      = df.iloc[idx]
            prev_row = df.iloc[idx - 1]

            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]

            if not self._liquidity_ok(row):
                continue

            direction = self._breakout_direction(row, prev_row)
            if direction is None:
                continue

            allowed = self._intersect_direction(direction, regime)
            if allowed is None:
                continue

            if not self._price_floor_ok(row, allowed):
                continue

            if not self._affordable(row, max_price):
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

    def _breakout_direction(self, row: pd.Series, prev_row: pd.Series) -> Optional[str]:
        """
        判斷 52 週突破方向。
        回傳 'long'、'short'、'both' 或 None（無突破）。
        """
        high_52w = prev_row.get("High_52W")
        low_52w  = prev_row.get("Low_52W")

        breakout_high = (not pd.isna(high_52w)) and float(row["High"]) > float(high_52w)
        breakout_low  = (not pd.isna(low_52w))  and float(row["Low"])  < float(low_52w)

        if breakout_high and breakout_low:
            return "both"
        if breakout_high:
            return "long"
        if breakout_low:
            return "short"
        return None

    def _market_regime(self, date: pd.Timestamp) -> str:
        """
        根據加權指數與 200 日 EMA 判斷大盤環境。
        Close > EMA → 'long'；Close < EMA → 'short'；無資料 → 'both'。
        """
        if self._taiex is None:
            return "both"
        past = self._taiex[self._taiex.index <= date]
        if past.empty:
            return "both"
        latest  = past.iloc[-1]
        t_close = latest.get("Close")
        t_ema   = latest.get("EMA_200")
        if pd.isna(t_close) or pd.isna(t_ema):
            return "both"
        return "long" if float(t_close) > float(t_ema) else "short"

    @staticmethod
    def _intersect_direction(breakout: str, regime: str) -> Optional[str]:
        """
        取突破方向與大盤環境的交集。
        無交集 → None（不入選）。
        """
        if regime == "both":
            return breakout
        if breakout == "both":
            return regime
        if breakout == regime:
            return regime
        return None

    def _price_floor_ok(self, row: pd.Series, direction: str) -> bool:
        """
        股價下限：做多 >= min_long_price；做空 >= min_short_price。
        direction='both' 時，符合任一方向即通過。
        """
        close = row.get("Close")
        if pd.isna(close):
            return False
        close = float(close)
        if direction in ("long", "both") and close >= self.cfg.min_long_price:
            return True
        if direction in ("short", "both") and close >= self.cfg.min_short_price:
            return True
        return False

    def _affordable(self, row: pd.Series, max_price: float) -> bool:
        """收盤 × 1000 <= equity / max_positions（單倉動態上限）"""
        close = row.get("Close")
        if pd.isna(close):
            return False
        return float(close) <= max_price

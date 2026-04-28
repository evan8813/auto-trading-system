"""
corporate_action.py
───────────────────
職責：管理除權息事件的「載入」與「套用到持倉」。
      與 Position 互動，但不依賴 Backtester 或 LiveTrader。

擴充指引：
  - 新增事件來源（例如從 API 自動抓取）→ 新增 load_from_api() 方法
  - 新增事件類型（例如減資）→ 在 apply_to_position() 加分支
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import pandas as pd

from models import CorporateEvent, Position

logger = logging.getLogger(__name__)


class CorporateActionLog:
    """
    管理所有除權息事件，在實盤持倉更新時自動套用。

    CSV 格式（corporate_actions.csv）：
      ticker, event_date, event_type, cash_dividend, stock_ratio, split_ratio, note

    典型使用流程（實盤）：
      ca_log = CorporateActionLog()
      ca_log.load_csv("corporate_actions.csv")

      # 每日盤前：
      for pos in positions:
          ca_log.apply_to_position(pos, today)
    """

    def __init__(self) -> None:
        self._events: list[CorporateEvent] = []

    # ── 載入 ──────────────────────────────────

    def add(self, event: CorporateEvent) -> None:
        """手動新增一筆事件（測試 / 程式化輸入用）"""
        self._events.append(event)

    def load_csv(self, path: str) -> None:
        """從 CSV 批次載入除權息事件"""
        count_before = len(self._events)
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._events.append(CorporateEvent(
                    ticker        = row["ticker"].strip(),
                    event_date    = pd.Timestamp(row["event_date"].strip()),
                    event_type    = row.get("event_type", "dividend").strip(),
                    cash_dividend = float(row.get("cash_dividend") or 0),
                    stock_ratio   = float(row.get("stock_ratio")   or 0),
                    split_ratio   = float(row.get("split_ratio")   or 1),
                    note          = row.get("note", "").strip(),
                ))
        loaded = len(self._events) - count_before
        logger.info(f"載入 {loaded} 筆除權息事件（{path}）。")

    # ── 查詢 ──────────────────────────────────

    def get_events(
        self,
        ticker:    str,
        from_date: pd.Timestamp,
        to_date:   pd.Timestamp,
    ) -> list[CorporateEvent]:
        """取得指定股票在日期區間內的所有事件"""
        return [
            e for e in self._events
            if e.ticker == ticker and from_date <= e.event_date <= to_date
        ]

    # ── 套用到持倉 ────────────────────────────

    def apply_to_position(
        self,
        pos:   Position,
        as_of: pd.Timestamp,
    ) -> None:
        """
        將 as_of 當日的除權息事件套用到持倉。
          - 現金股息 → 累計到 pos.dividend_received（元 / 股）
          - 股票分割 → 更新 pos.split_ratio 與 pos.shares
          - 股票股利 → 累加持股數
        """
        for e in self.get_events(pos.ticker, as_of, as_of):
            self._apply_event(pos, e, as_of)

    # ── 私有 ──────────────────────────────────

    @staticmethod
    def _apply_event(
        pos:   Position,
        e:     CorporateEvent,
        as_of: pd.Timestamp,
    ) -> None:
        if e.cash_dividend > 0:
            pos.dividend_received += e.cash_dividend
            logger.info(
                f"[除息] {pos.ticker} {as_of.date()}"
                f"  現金股息 +{e.cash_dividend:.4f} 元/股"
            )
        if e.split_ratio != 1.0:
            pos.split_ratio *= e.split_ratio
            pos.shares       = int(pos.shares * e.split_ratio)
            logger.info(
                f"[分割] {pos.ticker} {as_of.date()}"
                f"  分割比 {e.split_ratio}  持股 → {pos.shares} 股"
            )
        if e.stock_ratio > 0:
            extra      = int(pos.shares * e.stock_ratio)
            pos.shares += extra
            logger.info(
                f"[股利] {pos.ticker} {as_of.date()}"
                f"  股票股利 +{extra} 股  持股 → {pos.shares} 股"
            )

"""
models.py
─────────
純資料結構（dataclass）。
不含任何業務邏輯，只負責「存放狀態」。

擴充指引：
  - 新增持倉欄位（例如 stop_loss_price）→ 直接在 Position 加 field
  - 新增除權息事件類型 → 在 CorporateEvent.event_type 加說明
"""

from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class Position:
    """
    一筆持倉，同時記錄調整後價格（回測用）與原始市場價格（實盤對帳用）。
    """
    ticker:          str
    direction:       str              # "long" | "short"
    entry_date:      pd.Timestamp
    lots:            int              # 張數
    shares:          int              # 股數 = lots × 1000

    # ── 回測使用（還原股價）──
    adj_entry_price: float            # 調整後進場價（回測損益計算基準）

    # ── 實盤使用（原始市場價格）──
    raw_entry_price: float            # 原始進場價（Shioaji 成交均價）

    # ── 追蹤停損 ──
    trail_high:      float            # 持倉中累計最高價（做多停損基準）
    trail_low:       float            # 持倉中累計最低價（做空停損基準）
    atr_at_entry:    float

    # ── 除權息事件累計（實盤用）──
    dividend_received: float = 0.0    # 持倉期間累計現金股息（元 / 股）
    split_ratio:       float = 1.0    # 累計股票分割比（> 1 = 股本膨脹）

    def update_trail(self, high: float, low: float) -> None:
        """每日收盤後更新追蹤停損基準價"""
        if self.direction == "long":
            self.trail_high = max(self.trail_high, high)
        else:
            self.trail_low  = min(self.trail_low, low)


@dataclass
class CorporateEvent:
    """
    記錄一筆除權息事件。

    cash_dividend : 現金股息（元 / 股），除息日後入帳。
    stock_ratio   : 股票股利比（例如 0.1 = 每股配 0.1 股）。
    split_ratio   : 股票分割比（例如 2.0 = 1 股變 2 股）。

    真實損益計算：
      total_pnl = 價差損益 + cash_dividend × shares_held + 股利市值
    """
    ticker:        str
    event_date:    pd.Timestamp
    event_type:    str               # "dividend" | "split" | "rights"
    cash_dividend: float = 0.0      # 元 / 股
    stock_ratio:   float = 0.0      # 股 / 股
    split_ratio:   float = 1.0      # 分割比
    note:          str  = ""

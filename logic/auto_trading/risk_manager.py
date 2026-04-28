"""
risk_manager.py
───────────────
職責：計算「每筆風險金額」、「部位大小（張數）」、「交易成本」。
      只負責數字計算，不持有任何狀態，全部為純函式。

擴充指引：
  - 換用其他部位計算公式（例如 Kelly Criterion）→ 新增方法，
    在 Backtester._check_entries() 指定使用哪個方法即可。
  - 新增費用項目（例如借券費）→ 在 transaction_cost() 新增欄位。
"""

from __future__ import annotations

from config import TradingConfig


class RiskManager:
    """
    負責所有風險與成本計算。

    核心公式：
      Risk Amount  = current_equity × risk_pct
      Position Lots = Risk Amount ÷ (ATR × point_value) ÷ 1000
    """

    def __init__(self, cfg: TradingConfig) -> None:
        self.cfg = cfg

    def risk_amount(self, equity: float) -> float:
        """每筆交易可承擔的最大損失金額（元）"""
        return equity * self.cfg.risk_pct

    def position_size_lots(self, equity: float, atr: float) -> int:
        """
        計算部位大小（張數，1 張 = 1000 股）。
        ATR 越大 → 張數越少（波動大時自動縮小部位）。
        回傳 0 表示此筆交易不符合下單條件。
        """
        if not (atr > 0):
            return 0
        raw_shares = self.risk_amount(equity) / (atr * self.cfg.point_value)
        return max(int(raw_shares / 1000), 0)

    def transaction_cost(self, price: float, shares: int, side: str) -> float:
        """
        計算單邊交易成本（手續費 + 證交稅 + 滑價）。

        Parameters
        ----------
        price  : 成交價格
        shares : 股數
        side   : "buy" 或 "sell"（賣出時額外收證交稅）
        """
        notional      = price * shares
        commission    = notional * self.cfg.commission_rate
        slippage_cost = notional * self.cfg.slippage
        tax           = notional * self.cfg.transaction_tax if side == "sell" else 0.0
        return commission + slippage_cost + tax

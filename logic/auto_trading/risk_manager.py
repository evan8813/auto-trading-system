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
      risk_amount  = min(equity × risk_pct, max_risk_amount)
      lots（主公式）= risk_amount ÷ (atr_multiplier × ATR × point_value × 1000)
      lots（容忍區）= 主公式給 0 但 1 張風險 ≤ equity × 2% 時，進 1 張
    """

    def __init__(self, cfg: TradingConfig) -> None:
        self.cfg = cfg

    def risk_amount(self, equity: float) -> float:
        """
        每筆交易可承擔的最大損失金額（元）。
        上限 max_risk_amount：防止資產大幅成長後單筆風險失控。
        """
        return min(equity * self.cfg.risk_pct, self.cfg.max_risk_amount)

    def position_size_lots(self, equity: float, atr: float) -> int:
        """
        計算部位大小（張數，1 張 = 1000 股）。

        主公式：使用完整停損距離（atr_multiplier × ATR），
                確保打到停損時實際虧損 = risk_amount（= equity × risk_pct）。

        容忍區：主公式給 0 張，但 1 張風險 ≤ equity × 2% 時，
                仍進 1 張（資本不足時的最低單位保護）。
                隨 equity 成長，1% 公式自然給出 ≥ 1 張，容忍區逐漸不觸發。

        回傳 0：波動太大或 ATR <= 0，跳過此筆。
        """
        if not (atr > 0):
            return 0

        stop_distance = self.cfg.atr_multiplier * atr * self.cfg.point_value
        risk          = self.risk_amount(equity)

        # 主公式
        lots = int(risk / (stop_distance * 1000))
        if lots >= 1:
            return lots

        # 容忍區：1 張實際風險 ≤ 2% of equity → 進 1 張
        one_lot_risk = stop_distance * 1000
        if one_lot_risk <= equity * self.cfg.risk_pct * 2:
            return 1

        return 0

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

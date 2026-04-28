"""
live_trader.py
──────────────
職責：串接永豐金 Shioaji API，負責實盤下單、停損監控、除權息套用、
      以及建立含股息的完整實盤交易紀錄。
      只負責「與交易所溝通」及「實盤損益記錄」。

除權息處理說明：
  台股除息日股價向下跳空，若以原始股價計算停損，
  除息當天極易誤觸停損。本模組的解法：
    1. 每日盤前呼叫 apply_corporate_actions()，
       更新 pos.dividend_received 與 pos.shares。
    2. 停損判斷使用還原後股價（latest_data 應傳調整後資料），
       避免除息日誤觸停損。
    3. build_live_trade_record() 以「原始成交價 + 累計股息」
       計算實際損益，作為報稅與對帳的依據。

安裝依賴：pip install shioaji
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from config import TradingConfig
from corporate_action import CorporateActionLog
from models import Position
from risk_manager import RiskManager
from signal_generator import SignalGenerator

logger = logging.getLogger(__name__)

try:
    import shioaji as sj
    _SHIOAJI_AVAILABLE = True
except ImportError:
    _SHIOAJI_AVAILABLE = False


class LiveTrader:
    """
    實盤交易介面，搭配永豐金 Shioaji API。

    Parameters
    ----------
    cfg        : TradingConfig
    api_key    : Shioaji API 金鑰
    secret_key : Shioaji 密鑰
    ca_path    : 憑證路徑（空字串 = 不啟用 CA）
    ca_passwd  : 憑證密碼
    sim        : True = 模擬模式（不會真實下單）
    ca_log     : CorporateActionLog 實例；None = 空的（不處理除權息）
    """

    def __init__(
        self,
        cfg:        TradingConfig,
        api_key:    str,
        secret_key: str,
        ca_path:    str = "",
        ca_passwd:  str = "",
        sim:        bool = True,
        ca_log:     Optional[CorporateActionLog] = None,
    ) -> None:
        if not _SHIOAJI_AVAILABLE:
            raise ImportError("請先安裝 shioaji：pip install shioaji")

        self.cfg      = cfg
        self.risk_mgr = RiskManager(cfg)
        self.ca_log   = ca_log or CorporateActionLog()

        self.api  = sj.Shioaji(simulation=sim)
        accounts  = self.api.login(api_key=api_key, secret_key=secret_key)

        if ca_path:
            self.api.activate_ca(
                ca_path   = ca_path,
                ca_passwd = ca_passwd,
                person_id = accounts[0].person_id,
            )

        logger.info(f"Shioaji 登入成功（simulation={sim}）")

    # ── 下單 ──────────────────────────────────

    def place_order(
        self,
        ticker:    str,
        direction: str,    # "long"（買進）| "short"（賣出）
        lots:      int,
        price:     float,
    ) -> object:
        """送出市價 ROD 委託，回傳 Shioaji Trade 物件"""
        action   = sj.constant.Action.Buy if direction == "long" else sj.constant.Action.Sell
        contract = self.api.Contracts.Stocks[ticker]
        order    = self.api.Order(
            price      = price,
            quantity   = lots,
            action     = action,
            price_type = sj.constant.StockPriceType.MKT,
            order_type = sj.constant.OrderType.ROD,
        )
        trade = self.api.place_order(contract, order)
        logger.info(
            f"委託送出：{ticker} {direction.upper()} {lots}張"
            f" @ {price:.2f}  (sim={self.api.simulation})"
        )
        return trade

    # ── 除權息套用 ─────────────────────────────

    def apply_corporate_actions(
        self,
        positions: list[Position],
        as_of:     pd.Timestamp,
    ) -> None:
        """
        每日開盤前呼叫。
        副作用：更新 pos.dividend_received / pos.shares / pos.split_ratio。
        """
        for pos in positions:
            self.ca_log.apply_to_position(pos, as_of)

    # ── 停損監控 ───────────────────────────────

    def monitor_and_exit(
        self,
        positions:   list[Position],
        latest_data: dict[str, pd.DataFrame],
        today:       pd.Timestamp,
    ) -> list[Position]:
        """
        掃描所有持倉，觸發停損時自動送出出場委託。
        回傳剩餘持倉（已出場者移除）。

        注意：實際成交價由 Shioaji callback 取得，
              請在 on_order_cb 中呼叫 build_live_trade_record() 補上 raw_exit_price。
        """
        sig_gen   = SignalGenerator()
        remaining: list[Position] = []

        for pos in positions:
            df = latest_data.get(pos.ticker)
            if df is None or today not in df.index:
                remaining.append(pos)
                continue

            row = df.loc[today]
            pos.update_trail(row["High"], row["Low"])

            if pos.direction == "long":
                should_exit = sig_gen.long_exit(
                    row, pos.trail_high, self.cfg.atr_multiplier)
            else:
                should_exit = sig_gen.short_exit(
                    row, pos.trail_low, self.cfg.atr_multiplier)

            if should_exit:
                logger.warning(
                    f"⚠️  停損觸發：{pos.ticker} {pos.direction}"
                    f"  trail_high={pos.trail_high:.2f}"
                    f"  trail_low={pos.trail_low:.2f}"
                )
                exit_dir = "short" if pos.direction == "long" else "long"
                self.place_order(pos.ticker, exit_dir, pos.lots, row["Close"])
            else:
                remaining.append(pos)

        return remaining

    # ── 實盤交易紀錄 ───────────────────────────

    def build_live_trade_record(
        self,
        pos:            Position,
        exit_date:      pd.Timestamp,
        raw_exit_price: float,
        exit_reason:    str = "signal",
    ) -> dict:
        """
        建立完整的實盤交易紀錄（使用原始市場成交價）。

        raw_exit_price 應填入 Shioaji 成交回報的實際均價。

        損益計算：
          raw_pnl       = 價差損益（原始成交價差 × 股數）
          dividend_income = 持倉期間現金股息（元/股 × 股數）
          pnl_net       = raw_pnl + dividend_income − 手續費 − 稅 − 滑價
        """
        if pos.direction == "long":
            raw_pnl = (raw_exit_price - pos.raw_entry_price) * pos.shares
        else:
            raw_pnl = (pos.raw_entry_price - raw_exit_price) * pos.shares

        cost = (
            self.risk_mgr.transaction_cost(pos.raw_entry_price, pos.shares, "buy")
            + self.risk_mgr.transaction_cost(raw_exit_price, pos.shares, "sell")
        )

        dividend_income = pos.dividend_received * pos.shares
        pnl_net         = raw_pnl - cost + dividend_income

        return {
            "ticker":                      pos.ticker,
            "direction":                   pos.direction,
            "lots":                        pos.lots,
            "shares":                      pos.shares,
            "entry_date":                  pos.entry_date,
            "exit_date":                   exit_date,
            "hold_days":                   (exit_date - pos.entry_date).days,
            "raw_entry_price":             round(pos.raw_entry_price, 4),
            "raw_exit_price":              round(raw_exit_price, 4),
            "adj_entry_price":             round(pos.adj_entry_price, 4),
            "dividend_received_per_share": round(pos.dividend_received, 4),
            "dividend_income":             round(dividend_income, 2),
            "split_ratio":                 round(pos.split_ratio, 6),
            "raw_pnl":                     round(raw_pnl, 2),
            "total_cost":                  round(cost, 2),
            "pnl_net":                     round(pnl_net, 2),
            "atr_at_entry":                round(pos.atr_at_entry, 4),
            "exit_reason":                 exit_reason,
        }

    def logout(self) -> None:
        self.api.logout()
        logger.info("Shioaji 已登出。")

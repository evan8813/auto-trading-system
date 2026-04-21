"""
backtester.py
─────────────
職責：逐日模擬交易，輸出 equity_curve / trades / metrics。
      只負責「回測執行流程」的協調，
      各子任務（指標、篩選、訊號、風控）委派給各自模組。

時序規則：
  T   日收盤 → 判斷訊號
  T+1 日開盤 → 執行進出場（使用 T+1 開盤價）

擴充指引：
  - 換用不同策略 → 替換 SignalGenerator / UniverseFilter 的實作
  - 新增止盈條件 → 在 _check_exits() 加判斷分支
  - 支援多策略並行 → 讓 Backtester 接受策略清單
"""

from __future__ import annotations

import logging

import pandas as pd

from config import TradingConfig
from indicators import Indicators
from models import Position
from risk_manager import RiskManager
from signal_generator import SignalGenerator
from universe_filter import UniverseFilter

logger = logging.getLogger(__name__)


class Backtester:
    """
    逐日回測引擎（使用調整後股價）。

    run() 回傳：
      equity_curve : pd.DataFrame（index = Date, col = equity）
      trades       : pd.DataFrame（每筆平倉紀錄）
      metrics      : dict（績效指標）
    """

    def __init__(self, cfg: TradingConfig) -> None:
        self.cfg      = cfg
        self.risk_mgr = RiskManager(cfg)
        self.sig_gen  = SignalGenerator()
        self.uni_flt  = UniverseFilter(cfg)

    # ══ 主入口 ════════════════════════════════

    def run(self, raw_data: dict[str, pd.DataFrame]) -> dict:
        """
        Parameters
        ----------
        raw_data : key = 股票代號, value = OHLCV DataFrame（已清洗，尚未加指標）

        Returns
        -------
        dict with keys: equity_curve, trades, metrics
        """
        # 附加技術指標（不修改原始資料）
        data = {t: Indicators.add_all(df, self.cfg) for t, df in raw_data.items()}

        # 建立回測日期序列
        start     = pd.Timestamp(self.cfg.backtest_start)
        end       = pd.Timestamp(self.cfg.backtest_end)
        all_dates = sorted({
            d for df in data.values()
            for d in df.index
            if start <= d <= end
        })

        equity         = self.cfg.initial_equity
        positions:     list[Position] = []
        closed_trades: list[dict]    = []
        equity_curve:  list[dict]    = []

        for i, date in enumerate(all_dates):
            if i == 0:
                equity_curve.append({"date": date, "equity": equity})
                continue

            prev_date = all_dates[i - 1]

            # ── (A) 出場：T日訊號觸發 → T日開盤出場 ──
            positions, exited = self._check_exits(positions, data, date)
            for trade in exited:
                equity += trade["pnl_net"]
                closed_trades.append(trade)

            # ── (B) 進場：T-1日訊號觸發 → T日開盤進場 ──
            if len(positions) < self.cfg.max_positions:
                candidates   = self.uni_flt.filter(data, date)
                held_tickers = {p.ticker for p in positions}
                new_pos      = self._check_entries(
                    candidates, data, date, prev_date, equity, held_tickers
                )
                positions.extend(new_pos)

            equity_curve.append({"date": date, "equity": equity})

        metrics   = self._compute_metrics(equity_curve, closed_trades)
        trades_df = pd.DataFrame(closed_trades) if closed_trades else pd.DataFrame()

        return {
            "equity_curve": pd.DataFrame(equity_curve).set_index("date"),
            "trades":       trades_df,
            "metrics":      metrics,
        }

    # ══ 出場邏輯 ══════════════════════════════

    def _check_exits(
        self,
        positions: list[Position],
        data:      dict[str, pd.DataFrame],
        date:      pd.Timestamp,
    ) -> tuple[list[Position], list[dict]]:
        remaining: list[Position] = []
        exited:    list[dict]    = []

        for pos in positions:
            df = data.get(pos.ticker)
            if df is None or date not in df.index:
                remaining.append(pos)
                continue

            row = df.loc[date]
            pos.update_trail(row["High"], row["Low"])

            if pos.direction == "long":
                should_exit = self.sig_gen.long_exit(
                    row, pos.trail_high, self.cfg.atr_multiplier)
            else:
                should_exit = self.sig_gen.short_exit(
                    row, pos.trail_low, self.cfg.atr_multiplier)

            if should_exit:
                exit_px = row["Open"] if not pd.isna(row["Open"]) else row["Close"]
                trade   = self._close_position(pos, date, exit_px, "signal")
                exited.append(trade)
                logger.debug(
                    f"EXIT  {pos.ticker} {pos.direction} @ {exit_px:.2f}"
                    f"  PnL={trade['pnl_net']:+.0f}"
                )
            else:
                remaining.append(pos)

        return remaining, exited

    def _close_position(
        self,
        pos:        Position,
        exit_date:  pd.Timestamp,
        exit_price: float,
        reason:     str,
    ) -> dict:
        """計算平倉損益，回傳 trade dict（回測版，使用調整後價格）"""
        cfg = self.cfg

        # 滑價調整（做多賣出 → 價格偏低；做空回補 → 價格偏高）
        actual_exit = (
            exit_price * (1 - cfg.slippage) if pos.direction == "long"
            else exit_price * (1 + cfg.slippage)
        )

        # 毛利
        if pos.direction == "long":
            gross_pnl = (actual_exit - pos.adj_entry_price) * pos.shares
        else:
            gross_pnl = (pos.adj_entry_price - actual_exit) * pos.shares

        # 進出場總成本
        cost_entry = self.risk_mgr.transaction_cost(pos.adj_entry_price, pos.shares, "buy")
        cost_exit  = self.risk_mgr.transaction_cost(actual_exit, pos.shares, "sell")
        total_cost = cost_entry + cost_exit
        pnl_net    = gross_pnl - total_cost

        return {
            "ticker":          pos.ticker,
            "direction":       pos.direction,
            "lots":            pos.lots,
            "shares":          pos.shares,
            "entry_date":      pos.entry_date,
            "exit_date":       exit_date,
            "hold_days":       (exit_date - pos.entry_date).days,
            "adj_entry_price": round(pos.adj_entry_price, 4),
            "adj_exit_price":  round(actual_exit, 4),
            "raw_entry_price": round(pos.raw_entry_price, 4),
            "raw_exit_price":  round(exit_price, 4),
            "gross_pnl":       round(gross_pnl, 2),
            "total_cost":      round(total_cost, 2),
            "pnl_net":         round(pnl_net, 2),
            "atr_at_entry":    round(pos.atr_at_entry, 4),
            "exit_reason":     reason,
        }

    # ══ 進場邏輯 ══════════════════════════════

    def _check_entries(
        self,
        candidates:   list[str],
        data:         dict[str, pd.DataFrame],
        date:         pd.Timestamp,
        prev_date:    pd.Timestamp,
        equity:       float,
        held_tickers: set[str],
    ) -> list[Position]:
        new_positions: list[Position] = []

        for ticker in candidates:
            if ticker in held_tickers:
                continue
            if len(held_tickers) + len(new_positions) >= self.cfg.max_positions:
                break

            df = data.get(ticker)
            if df is None or date not in df.index or prev_date not in df.index:
                continue

            row      = df.loc[date]
            prev_row = df.loc[prev_date]
            atr      = row["ATR"]

            if pd.isna(atr) or atr <= 0:
                continue

            direction = self._resolve_direction(row, prev_row)
            if direction is None:
                continue

            lots = self.risk_mgr.position_size_lots(equity, atr)
            if lots == 0:
                continue

            shares      = lots * 1000
            entry_price = row["Open"]
            if pd.isna(entry_price):
                continue

            adj_entry = (
                entry_price * (1 + self.cfg.slippage) if direction == "long"
                else entry_price * (1 - self.cfg.slippage)
            )

            pos = Position(
                ticker          = ticker,
                direction       = direction,
                entry_date      = date,
                lots            = lots,
                shares          = shares,
                adj_entry_price = adj_entry,
                raw_entry_price = entry_price,
                trail_high      = row["High"],
                trail_low       = row["Low"],
                atr_at_entry    = atr,
            )
            new_positions.append(pos)
            held_tickers.add(ticker)
            logger.debug(
                f"ENTRY {ticker} {direction.upper()} @ {adj_entry:.2f}"
                f"  lots={lots}  ATR={atr:.2f}"
            )

        return new_positions

    def _resolve_direction(
        self,
        row:      pd.Series,
        prev_row: pd.Series,
    ) -> str | None:
        """判斷進場方向，優先做多，其次做空，都不符合回傳 None"""
        if self.sig_gen.long_entry(row, prev_row):
            return "long"
        if self.sig_gen.short_entry(row, prev_row):
            return "short"
        return None

    # ══ 績效計算 ══════════════════════════════

    def _compute_metrics(
        self,
        equity_curve: list[dict],
        trades:       list[dict],
    ) -> dict:
        ec = pd.DataFrame(equity_curve).set_index("date")["equity"]

        total_return = (ec.iloc[-1] / ec.iloc[0] - 1) * 100

        roll_max = ec.cummax()
        drawdown = (ec - roll_max) / roll_max
        mdd      = drawdown.min() * 100

        n_years = (ec.index[-1] - ec.index[0]).days / 365.25
        cagr    = (
            ((ec.iloc[-1] / ec.iloc[0]) ** (1 / n_years) - 1) * 100
            if n_years > 0 else 0.0
        )

        win_rate = avg_win = avg_loss = profit_factor = 0.0
        if trades:
            df_t          = pd.DataFrame(trades)
            wins          = df_t.loc[df_t["pnl_net"] > 0,  "pnl_net"]
            losses        = df_t.loc[df_t["pnl_net"] <= 0, "pnl_net"]
            win_rate      = len(wins) / len(df_t) * 100
            avg_win       = wins.mean()   if len(wins)   else 0.0
            avg_loss      = losses.mean() if len(losses) else 0.0
            profit_factor = (
                wins.sum() / abs(losses.sum())
                if losses.sum() != 0 else float("inf")
            )

        return {
            "total_return_pct": round(total_return, 2),
            "cagr_pct":         round(cagr, 2),
            "max_drawdown_pct": round(mdd, 2),
            "win_rate_pct":     round(win_rate, 2),
            "total_trades":     len(trades),
            "avg_win":          round(avg_win, 0),
            "avg_loss":         round(avg_loss, 0),
            "profit_factor":    round(profit_factor, 3),
        }

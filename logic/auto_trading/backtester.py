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

        # ── 動態偵測指標欄位：排除原始 OHLCV 欄，其餘都視為指標欄 ──
        _base_cols = {"Open", "High", "Low", "Close",
                      "Volume", "Amount", "Change", "Transactions"}
        indicator_cols: list[str] = []
        for df in data.values():
            indicator_cols = [c for c in df.columns if c not in _base_cols]
            break   # 所有股票的指標欄位相同，取第一支即可

        # ── 自動跳過暖機期：找到第一個「所有指標欄皆有值」的交易日 ──
        # 條件：至少一支股票在該日的所有指標欄均非 NaN
        first_valid: pd.Timestamp | None = None
        for d in all_dates:
            for df in data.values():
                if d not in df.index:
                    continue
                row = df.loc[d]
                if all(not pd.isna(row[c]) for c in indicator_cols if c in row.index):
                    first_valid = d
                    break
            if first_valid is not None:
                break

        if first_valid is not None and first_valid > all_dates[0]:
            skipped = sum(1 for d in all_dates if d < first_valid)
            logger.info(
                f"自動跳過暖機期 {skipped} 個交易日"
                f"（等待指標欄：{indicator_cols}），"
                f"實際回測起始日：{first_valid.date()}"
            )
            all_dates = [d for d in all_dates if d >= first_valid]

        equity         = self.cfg.initial_equity
        positions:     list[Position] = []
        closed_trades: list[dict]    = []
        equity_curve:  list[dict]    = []

        # T 日收盤產生的待執行訊號，T+1 日開盤才執行
        pending_exit_tickers: set[str]                     = set()
        pending_entries:      list[tuple[str, str, float]] = []  # (ticker, direction, atr)

        for i, date in enumerate(all_dates):

            # ── (A) 執行前一日收盤訊號（T+1 開盤成交）──
            if pending_exit_tickers or pending_entries:
                positions, exited = self._execute_exits(
                    positions, data, date, pending_exit_tickers)
                for trade in exited:
                    equity += trade["pnl_net"]
                    closed_trades.append(trade)

                held = {p.ticker for p in positions}
                if len(positions) < self.cfg.max_positions:
                    deployed  = sum(p.adj_entry_price * p.shares for p in positions)
                    available = max(equity - deployed, 0.0)
                    new_pos   = self._execute_entries(
                        pending_entries, data, date, held, available)
                    positions.extend(new_pos)

            pending_exit_tickers = set()
            pending_entries      = []

            # ── (B) 更新 trail（用今日 High/Low），再判斷今日收盤出場訊號 ──
            for pos in positions:
                df = data.get(pos.ticker)
                if df is None or date not in df.index:
                    continue
                row = df.loc[date]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]
                pos.update_trail(row["High"], row["Low"])

            # ── (C) 今日收盤：判斷出場訊號（T+1 執行）──
            for pos in positions:
                df = data.get(pos.ticker)
                if df is None or date not in df.index:
                    continue
                row = df.loc[date]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]
                if pos.direction == "long":
                    if self.sig_gen.long_exit(row, pos.trail_high, self.cfg.atr_multiplier):
                        pending_exit_tickers.add(pos.ticker)
                else:
                    if self.sig_gen.short_exit(row, pos.trail_low, self.cfg.atr_multiplier):
                        pending_exit_tickers.add(pos.ticker)

            # ── (D) 今日收盤：篩選 + 判斷進場訊號（T+1 執行）──
            prev_date    = all_dates[i - 1] if i > 0 else date
            candidates   = self.uni_flt.filter(data, date)
            held_tickers = {p.ticker for p in positions}
            effective_held = held_tickers - pending_exit_tickers
            available_slots = self.cfg.max_positions - len(effective_held)

            # 按 ROC_avg 由高到低排序，確保動能最強的股票優先進場
            def get_roc(ticker):
                df_ = data.get(ticker)
                if df_ is None or date not in df_.index:
                    return float("-inf")
                r = df_.loc[date]
                if isinstance(r, pd.DataFrame): r = r.iloc[-1]
                return float(r.get("ROC_avg", float("-inf")))

            candidates = sorted(candidates, key=get_roc, reverse=True)

            for ticker in candidates:
                if available_slots <= len(pending_entries):
                    break
                if ticker in effective_held or ticker in pending_exit_tickers or ticker in {e[0] for e in pending_entries}:
                    continue
                df = data.get(ticker)
                if df is None or date not in df.index or prev_date not in df.index:
                    continue
                row      = df.loc[date]
                prev_row = df.loc[prev_date]
                if isinstance(row,      pd.DataFrame): row      = row.iloc[-1]
                if isinstance(prev_row, pd.DataFrame): prev_row = prev_row.iloc[-1]
                atr = row["ATR"]
                if pd.isna(atr) or atr <= 0:
                    continue
                direction = self._resolve_direction(row, prev_row)
                if direction:
                    pending_entries.append((ticker, direction, atr))

            equity_curve.append({"date": date, "equity": equity})

        metrics   = self._compute_metrics(equity_curve, closed_trades)
        trades_df = pd.DataFrame(closed_trades) if closed_trades else pd.DataFrame()

        return {
            "equity_curve": pd.DataFrame(equity_curve).set_index("date"),
            "trades":       trades_df,
            "metrics":      metrics,
        }

    # ══ 執行出場（T+1 開盤）══════════════════════

    def _execute_exits(
        self,
        positions:    list[Position],
        data:         dict[str, pd.DataFrame],
        exec_date:    pd.Timestamp,
        exit_tickers: set[str],
    ) -> tuple[list[Position], list[dict]]:
        remaining: list[Position] = []
        exited:    list[dict]    = []

        for pos in positions:
            if pos.ticker not in exit_tickers:
                remaining.append(pos)
                continue
            df = data.get(pos.ticker)
            if df is None or exec_date not in df.index:
                remaining.append(pos)   # 停牌→留倉等下一天
                continue
            row = df.loc[exec_date]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            exit_px = row["Open"] if not pd.isna(row["Open"]) else row["Close"]
            trade   = self._close_position(pos, exec_date, exit_px, "signal")
            exited.append(trade)
            logger.debug(
                f"EXIT  {pos.ticker} {pos.direction} @ {exit_px:.2f}"
                f"  PnL={trade['pnl_net']:+.0f}"
            )

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
            "atr_at_entry":     round(pos.atr_at_entry, 4),
            "equity_at_entry":  round(pos.equity_at_entry, 2),
            "exit_reason":      reason,
        }

    # ══ 執行進場（T+1 開盤）══════════════════════

    def _execute_entries(
        self,
        pending_entries: list[tuple[str, str, float]],
        data:            dict[str, pd.DataFrame],
        exec_date:       pd.Timestamp,
        held_tickers:    set[str],
        equity:          float,
    ) -> list[Position]:
        new_positions: list[Position] = []
        initial_held_count = len(held_tickers)

        for ticker, direction, atr_at_signal in pending_entries:
            if ticker in held_tickers:
                continue
            if initial_held_count + len(new_positions) >= self.cfg.max_positions:
                break

            df = data.get(ticker)
            if df is None or exec_date not in df.index:
                continue

            row = df.loc[exec_date]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            entry_price = row["Open"]
            if pd.isna(entry_price):
                continue

            lots = self.risk_mgr.position_size_lots(equity, atr_at_signal)
            if lots == 0:
                continue

            shares    = lots * 1000
            adj_entry = (
                entry_price * (1 + self.cfg.slippage) if direction == "long"
                else entry_price * (1 - self.cfg.slippage)
            )

            # ── 單筆資金上限 ──
            entry_cost = adj_entry * shares
            if entry_cost > self.cfg.max_trade_cost:
                lots = max(int(self.cfg.max_trade_cost / (adj_entry * 1000)), 0)
                if lots == 0:
                    continue
                shares     = lots * 1000
                entry_cost = adj_entry * shares

            # ── 可用現金檢查 ──
            if entry_cost > equity:
                lots = max(int(equity / (adj_entry * 1000)), 0)
                if lots == 0:
                    continue
                shares = lots * 1000

            pos = Position(
                ticker           = ticker,
                direction        = direction,
                entry_date       = exec_date,
                lots             = lots,
                shares           = shares,
                adj_entry_price  = adj_entry,
                raw_entry_price  = entry_price,
                trail_high       = entry_price,
                trail_low        = entry_price,
                atr_at_entry     = atr_at_signal,
                equity_at_entry  = equity,       # 進場時可用資金
            )
            new_positions.append(pos)
            held_tickers.add(ticker)
            logger.debug(
                f"ENTRY {ticker} {direction.upper()} @ {adj_entry:.2f}"
                f"  lots={lots}  ATR={atr_at_signal:.2f}"
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

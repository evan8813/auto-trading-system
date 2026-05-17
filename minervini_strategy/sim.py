"""
sim.py
本地回測引擎，替代 finlab.backtest.sim()。
支援每日掃描模式：有訊號就進場，停損或訊號消失就出場。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── 報告物件 ──────────────────────────────────────────────────────────────────

class Report:
    def __init__(self, equity: pd.Series, trades: pd.DataFrame):
        self.equity = equity
        self.trades = trades

    def get_stats(self) -> dict:
        ec = self.equity.dropna()
        if len(ec) < 2:
            return {"cagr": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "total_return": 0.0}

        years     = (ec.index[-1] - ec.index[0]).days / 365.25
        total_ret = ec.iloc[-1] / ec.iloc[0] - 1
        cagr      = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0.0

        roll_max = ec.cummax()
        max_dd   = (ec / roll_max - 1).min()

        monthly = ec.resample("ME").last().pct_change().dropna()
        sharpe  = (monthly.mean() / monthly.std() * np.sqrt(12)
                   if monthly.std() > 0 else 0.0)

        win_trades  = self.trades[self.trades["pnl"] > 0]
        win_rate    = len(win_trades) / len(self.trades) if len(self.trades) > 0 else 0.0
        avg_win     = win_trades["pnl"].mean() if len(win_trades) > 0 else 0.0
        loss_trades = self.trades[self.trades["pnl"] <= 0]
        avg_loss    = loss_trades["pnl"].mean() if len(loss_trades) > 0 else 0.0

        return {
            "total_return":  total_ret,
            "cagr":          cagr,
            "max_drawdown":  max_dd,
            "sharpe":        sharpe,
            "n_trades":      len(self.trades),
            "win_rate":      win_rate,
            "avg_win":       avg_win,
            "avg_loss":      avg_loss,
        }

    def print_stats(self):
        s = self.get_stats()
        print("=" * 42)
        print(f"  總報酬：    {s['total_return']:>8.2%}")
        print(f"  年化報酬：  {s['cagr']:>8.2%}")
        print(f"  最大回撤：  {s['max_drawdown']:>8.2%}")
        print(f"  月化夏普：  {s['sharpe']:>8.2f}")
        print(f"  交易筆數：  {s['n_trades']:>8d}")
        print(f"  勝率：      {s['win_rate']:>8.2%}")
        print(f"  平均獲利：  {s['avg_win']:>8.0f} 元")
        print(f"  平均虧損：  {s['avg_loss']:>8.0f} 元")
        print("=" * 42)

    def plot(self, title: str = "Equity Curve"):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(13, 5))
        self.equity.plot(ax=ax, color="steelblue", linewidth=1.2)
        ax.axhline(self.equity.iloc[0], color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel("Portfolio Value (NTD)")
        plt.tight_layout()
        plt.show()


# ── 回測引擎 ──────────────────────────────────────────────────────────────────

def sim(
    position:       pd.DataFrame,
    close:          pd.DataFrame,
    stop_loss:      float | None = 0.08,
    take_profit:    float | None = None,
    max_positions:  int          = 5,
    fee_ratio:      float        = 1.425 / 1000 / 3,
    tax_ratio:      float        = 3 / 1000,
    initial_equity: float        = 1_000_000,
) -> Report:
    """
    每日掃描回測引擎。

    Parameters
    ----------
    position       : bool DataFrame（True = 持有訊號），index=日期, columns=股票代號
    close          : 收盤價 DataFrame，同 index/columns
    stop_loss      : 停損比例，e.g. 0.08 = 8%
    take_profit    : 停利比例，e.g. 0.25 = 25%
    max_positions  : 最大同時持倉數
    fee_ratio      : 手續費（買賣各收）
    tax_ratio      : 交易稅（賣時收）
    initial_equity : 起始資金（元）

    邏輯
    ----
    進場：position 由 False → True，且持倉數 < max_positions
    出場：position 變 False  OR  停損 / 停利觸發
    部位大小：進場時 portfolio_value / max_positions（等權重）
    """
    # 對齊 index，只取 close 有資料的日期
    pos   = position.reindex(close.index, method="ffill").fillna(False)
    dates = close.index

    equity   = initial_equity
    holdings: dict[str, dict] = {}  # ticker -> {shares, entry_price}
    trade_log: list[dict]     = []
    equity_curve: dict        = {}

    prev_pos = pd.Series(False, index=pos.columns)

    for dt in dates:
        curr_pos = pos.loc[dt] if dt in pos.index else pd.Series(False, index=pos.columns)

        # ── 1. 停損 / 停利 ────────────────────────────────
        exits_sl = []
        for tk, h in holdings.items():
            if tk not in close.columns:
                exits_sl.append((tk, "delisted"))
                continue
            px  = close.at[dt, tk]
            if np.isnan(px):
                continue
            ret = px / h["entry_price"] - 1
            if stop_loss   is not None and ret <= -stop_loss:
                exits_sl.append((tk, "stop_loss"))
            elif take_profit is not None and ret >= take_profit:
                exits_sl.append((tk, "take_profit"))

        for tk, reason in exits_sl:
            h  = holdings.pop(tk)
            px = close.at[dt, tk] if tk in close.columns and not np.isnan(close.at[dt, tk]) else h["entry_price"]
            proceeds = h["shares"] * px * (1 - fee_ratio - tax_ratio)
            equity  += proceeds
            trade_log.append({
                "ticker":       tk,
                "entry_date":   h["entry_date"],
                "exit_date":    dt,
                "entry_price":  h["entry_price"],
                "exit_price":   px,
                "shares":       h["shares"],
                "pnl":          proceeds - h["cost"],
                "exit_reason":  reason,
            })

        # ── 2. 訊號消失 → 出場 ───────────────────────────
        exits_sig = [tk for tk in list(holdings) if not curr_pos.get(tk, False)]
        for tk in exits_sig:
            if tk in holdings:
                h  = holdings.pop(tk)
                px = close.at[dt, tk] if tk in close.columns and not np.isnan(close.at[dt, tk]) else h["entry_price"]
                proceeds = h["shares"] * px * (1 - fee_ratio - tax_ratio)
                equity  += proceeds
                trade_log.append({
                    "ticker":       tk,
                    "entry_date":   h["entry_date"],
                    "exit_date":    dt,
                    "entry_price":  h["entry_price"],
                    "exit_price":   px,
                    "shares":       h["shares"],
                    "pnl":          proceeds - h["cost"],
                    "exit_reason":  "signal_off",
                })

        # ── 3. 新訊號 → 進場 ─────────────────────────────
        new_signals = [
            tk for tk in curr_pos[curr_pos].index
            if tk not in holdings
            and not prev_pos.get(tk, False)   # 今天才剛突破（False → True）
            and tk in close.columns
        ]

        slots = max_positions - len(holdings)
        if slots > 0 and new_signals:
            # 計算當前 portfolio value 做等權配置
            port_val = equity
            for tk, h in holdings.items():
                if tk in close.columns:
                    px = close.at[dt, tk]
                    port_val += h["shares"] * (px if not np.isnan(px) else h["entry_price"])

            per_pos = port_val / max_positions

            for tk in new_signals[:slots]:
                px = close.at[dt, tk]
                if np.isnan(px) or px <= 0 or per_pos > equity:
                    continue
                shares = per_pos * (1 - fee_ratio) / px
                equity -= per_pos
                holdings[tk] = {
                    "shares":      shares,
                    "entry_price": px,
                    "entry_date":  dt,
                    "cost":        per_pos,
                }

        # ── 4. Mark-to-market ────────────────────────────
        port_val = equity
        for tk, h in holdings.items():
            if tk in close.columns:
                px = close.at[dt, tk]
                port_val += h["shares"] * (px if not np.isnan(px) else h["entry_price"])
        equity_curve[dt] = port_val

        prev_pos = curr_pos

    trades_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame(
        columns=["ticker", "entry_date", "exit_date", "entry_price",
                 "exit_price", "shares", "pnl", "exit_reason"]
    )
    return Report(pd.Series(equity_curve), trades_df)

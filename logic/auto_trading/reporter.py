"""
reporter.py
───────────
職責：將回測結果輸出為可讀格式（終端機印表、CSV、PNG 圖表）。
      只負責「呈現」，不做任何計算或策略邏輯。

擴充指引：
  - 新增輸出格式（例如 HTML 報表、Excel）→ 新增靜態方法
  - 調整圖表樣式 → 修改 plot_equity_curve()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # 無 GUI 環境也能存圖
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


class Reporter:
    """提供三種輸出方式：終端機報告、CSV 交易紀錄、Equity Curve 圖"""

    # ── 終端機報告 ────────────────────────────

    @staticmethod
    def print_metrics(metrics: dict) -> None:
        """在終端機印出績效摘要"""
        print("\n" + "=" * 48)
        print("  回測績效報告")
        print("=" * 48)
        print(f"  總報酬率        : {metrics['total_return_pct']:>9.2f} %")
        print(f"  年化報酬 (CAGR) : {metrics['cagr_pct']:>9.2f} %")
        print(f"  最大回撤 (MDD)  : {metrics['max_drawdown_pct']:>9.2f} %")
        print(f"  勝率            : {metrics['win_rate_pct']:>9.2f} %")
        print(f"  總交易次數      : {metrics['total_trades']:>9d}")
        print(f"  平均獲利 / 筆   : {metrics['avg_win']:>9.0f} 元")
        print(f"  平均虧損 / 筆   : {metrics['avg_loss']:>9.0f} 元")
        print(f"  獲利因子        : {metrics['profit_factor']:>9.3f}")
        print("=" * 48 + "\n")

    # ── CSV 交易紀錄 ──────────────────────────

    @staticmethod
    def save_trade_log(
        trades: pd.DataFrame,
        path:   str = "trade_log.csv",
    ) -> None:
        """
        輸出每筆交易紀錄 CSV。

        欄位說明：
          adj_entry/exit_price  → 調整後股價（回測損益依此計算）
          raw_entry/exit_price  → 原始市場成交價（實盤對帳 / 報稅用）
          pnl_net               → 扣除手續費、稅、滑價後的淨損益
          dividend_income       → 實盤現金股息總收入（元）[僅實盤有值]
        """
        if trades.empty:
            logger.warning("無交易紀錄可輸出。")
            return
        trades.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  交易紀錄已儲存：{path}  （共 {len(trades)} 筆）")

    # ── Equity Curve 圖 ───────────────────────

    @staticmethod
    def plot_equity_curve(
        equity_curve: pd.DataFrame,
        save_path:    Optional[str] = "equity_curve.png",
    ) -> None:
        """
        輸出雙圖表 PNG：上圖為淨值曲線，下圖為回撤百分比。

        Parameters
        ----------
        equity_curve : DataFrame，index = Date，欄位包含 "equity"
        save_path    : 輸出路徑；None = 直接顯示（需有 GUI）
        """
        fig, axes = plt.subplots(
            2, 1, figsize=(14, 8), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        fig.suptitle("自動化交易系統 — Equity Curve", fontsize=14, fontweight="bold")

        ec       = equity_curve["equity"]
        roll_max = ec.cummax()
        drawdown = (ec - roll_max) / roll_max * 100

        # 上圖：淨值曲線
        axes[0].plot(ec.index, ec.values, color="#1565C0", linewidth=1.5, label="Equity")
        axes[0].fill_between(ec.index, ec.values, alpha=0.08, color="#1565C0")
        axes[0].set_ylabel("資產淨值（元）")
        axes[0].yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x / 1_000_000:.1f}M"))
        axes[0].legend(loc="upper left")
        axes[0].grid(alpha=0.3)

        # 下圖：回撤
        axes[1].fill_between(
            ec.index, drawdown.values, color="#C62828", alpha=0.6, label="Drawdown")
        axes[1].set_ylabel("回撤 (%)")
        axes[1].set_xlabel("日期")
        axes[1].legend(loc="lower left")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Equity Curve 已儲存：{save_path}")
        else:
            plt.show()
        plt.close()

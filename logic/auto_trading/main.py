"""
main.py
───────
職責：程式的唯一入口點。
      組裝各模組、執行回測、輸出結果。
      不含任何業務邏輯，只負責「協調流程」。

用法：
  python main.py                                   # 合成資料測試
  python main.py /path/to/csv_folder               # 載入所有股票
  python main.py /path/to/csv_folder 2330 2317     # 指定股票
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from backtester import Backtester
from checkpoint import run_checkpoint
from config import TradingConfig
from data_loader import DataLoader
from reporter import Reporter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_backtest(
    data_folder: Optional[str] = None,
    tickers:     Optional[list[str]] = None,
    cfg:         Optional[TradingConfig] = None,
    output_dir:  str = "output",
) -> dict:
    """
    執行回測的主入口。

    Parameters
    ----------
    data_folder : 存放 TWSE CSV 的資料夾路徑；None = 使用合成資料。
    tickers     : 指定股票代號清單；None = 自動掃描資料夾。
    cfg         : TradingConfig；None = 使用預設值。
    output_dir  : 輸出 equity_curve.png 與 trade_log.csv 的目錄。

    Returns
    -------
    dict  包含 equity_curve, trades, metrics
    """
    cfg = cfg or TradingConfig()

    # ── 1. 載入資料 ──
    if data_folder is not None:
        logger.info(f"從 CSV 資料夾載入：{data_folder}")
        data = DataLoader.load_folder(data_folder, tickers=tickers, adjusted=True)
    else:
        logger.info("未指定資料夾，使用合成資料（測試模式）。")
        synthetic_tickers = tickers or [f"{2000 + i}" for i in range(50)]
        data = DataLoader.generate_synthetic(
            synthetic_tickers, cfg.backtest_start, cfg.backtest_end
        )

    if not data:
        logger.error("無可用資料，請確認資料夾路徑與 CSV 格式。")
        return {}

    # ── 2. 執行回測 ──
    engine = Backtester(cfg)
    logger.info(
        f"開始回測（{cfg.backtest_start} ~ {cfg.backtest_end}），"
        f"共 {len(data)} 支股票…"
    )
    results = engine.run(data)

    # ── 3. 輸出報告 ──
    Reporter.print_metrics(results["metrics"])

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    Reporter.plot_equity_curve(
        results["equity_curve"],
        save_path=str(out / "equity_curve.png"),
    )
    Reporter.save_trade_log(
        results["trades"],
        path=str(out / "trade_log.csv"),
    )

    # 終端機預覽最後 10 筆交易
    if not results["trades"].empty:
        display_cols = [
            "ticker", "direction", "lots",
            "entry_date", "exit_date", "hold_days",
            "adj_entry_price", "adj_exit_price",
            "raw_entry_price", "raw_exit_price",
            "pnl_net", "exit_reason",
        ]
        cols = [c for c in display_cols if c in results["trades"].columns]
        print("\n  最近 10 筆交易：")
        print(results["trades"][cols].tail(10).to_string(index=False))

    # ── 4. Checkpoint 驗證 ──
    run_checkpoint(results, data, cfg)

    return results


if __name__ == "__main__":
    folder  = sys.argv[1] if len(sys.argv) > 1 else None
    symbols = sys.argv[2:] if len(sys.argv) > 2 else None

    cfg = TradingConfig(
        initial_equity   = 180_000,       # 初始資金 18 萬（可負擔最多 1 張 180 元以下股票）
        risk_pct         = 0.002,         # 每筆風險 0.2%
        commission_rate  = 0.001425,
        transaction_tax  = 0.003,
        slippage         = 0.001,
        max_positions    = 10,
        max_trade_cost   = 5_000,         # 單筆買入上限 5,000 元
        min_avg_amount   = 5_000_000,
        min_long_price   = 10.0,          # 做多股價下限
        min_short_price  = 20.0,          # 做空股價下限
        taiex_csv_path   = str(Path(__file__).parent.parent.parent / "taiex.csv"),
        backtest_start   = "2010-01-01",
        backtest_end     = "2023-12-29",
    )

    run_backtest(
        data_folder = folder,
        tickers     = symbols,
        cfg         = cfg,
        output_dir  = "output",
    )

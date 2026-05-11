"""
experiment.py
─────────────
測試用執行檔。想改變數值測試時，在這裡修改再執行，不影響 main.py。

執行方式：
  python experiment.py                            # 合成資料
  python experiment.py /path/to/stocks_full       # 完整資料
  python experiment.py /path/to/stocks_full 2330  # 指定股票

結果輸出到 output/experiment/，不會覆蓋 main.py 的 output/。
"""

import sys
from config import TradingConfig
from main import run_backtest

# ── 在這裡修改你想測試的參數 ─────────────────────────────────────────
# 沒有寫的參數會自動沿用 config.py 的預設值，不需要全部列出
cfg = TradingConfig(
    # initial_equity = 180_000,    # 預設值，需要改時取消 # 即可
    # max_positions  = 5,
    # max_trade_cost = 36_000,
    # risk_pct       = 0.01,
    # atr_multiplier = 5.0,
    # vol_mult       = 1.5,
    max_positions_per_sector = 2,  # 同產業最多 2 席（0 = 不限制）
)
# ────────────────────────────────────────────────────────────────────

folder  = sys.argv[1] if len(sys.argv) > 1 else None
symbols = sys.argv[2:] if len(sys.argv) > 2 else None

run_backtest(
    data_folder = folder,
    tickers     = symbols,
    cfg         = cfg,
    output_dir  = "output/experiment",
)

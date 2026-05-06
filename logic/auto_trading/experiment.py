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
    initial_equity = 1_000_000,    # 測試 100 萬資金
    max_positions  = 10,           # 最大持倉數
    max_trade_cost = 100_000,      # 單筆上限 = 100萬 ÷ 10
    # risk_pct     = 0.01,         # 其他參數保持預設，需要時取消 # 即可
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

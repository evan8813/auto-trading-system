"""
experiment.py
─────────────
測試用執行檔。想改變數值測試時，在這裡修改再執行，不影響 main.py。

執行方式：
  python experiment.py                            # 合成資料
  python experiment.py /path/to/stocks_full       # 完整資料
  python experiment.py /path/to/stocks_full 2330  # 指定股票

結果輸出到 output/experiment/，不會覆蓋 main.py 的 output/。

════════════════════════════════════════════════════════
  歷次實驗紀錄（基準 main.py：+69.6%、MDD -29.7%、Calmar 0.14）
════════════════════════════════════════════════════════

  【實驗 A】產業集中度上限 max_positions_per_sector=2
    結論：+57.4%、MDD -33.3%、Calmar 0.107 → 更差
          產業集中不是問題，限制反而砍掉好機會

  【實驗 B】資金規模 1M（max_risk_amount 未調整）
    結論：+20.7%、MDD -11.3%、Calmar 0.130
          max_risk_amount=2,500 被卡住，position size 沒等比放大，結果失真

  【實驗 C】資金規模 1M + max_risk_amount=10,000
    結論：績效不佳（未記錄詳細數字）
          資金規模不是主因

  【待測】ATR 倍數調整（5x → 3.5x）
  【待測】時間段分析（先做 Section 15 圖表再決定參數方向）
════════════════════════════════════════════════════════
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

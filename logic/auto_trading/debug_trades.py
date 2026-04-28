"""
debug_trades.py
───────────────
診斷回測中的實際交易細節：
 - 顯示前 20 筆進場：張數、進場價、買入金額
 - 確認 max_trade_cost 是否確實生效
 - 印出 config 中的關鍵參數
"""

import sys
import logging
import pandas as pd

sys.path.insert(0, '.')

from config import TradingConfig
from data_loader import DataLoader
from backtester import Backtester
from indicators import Indicators

logging.basicConfig(level=logging.WARNING)   # 關掉 INFO 雜訊

# ── 使用與 main.py 相同的設定 ──
cfg = TradingConfig(
    initial_equity  = 1_000_000,    # main.py 裡的原始設定
    risk_pct        = 0.002,
    commission_rate = 0.001425,
    transaction_tax = 0.003,
    slippage        = 0.001,
    max_positions   = 10,
    min_avg_amount  = 5_000_000,
    backtest_start  = "2010-01-01",
    backtest_end    = "2023-12-29",
)

print("=== CONFIG 參數 ===")
print(f"  initial_equity  : {cfg.initial_equity:,.0f}")
print(f"  max_trade_cost  : {cfg.max_trade_cost:,.0f}")
print(f"  risk_pct        : {cfg.risk_pct}")
print(f"  max_positions   : {cfg.max_positions}")

folder = sys.argv[1] if len(sys.argv) > 1 else None
if folder is None:
    print("\n用法: python debug_trades.py <資料夾路徑>")
    sys.exit(1)

print(f"\n載入資料: {folder}")
data = DataLoader.load_folder(folder, adjusted=True)
print(f"載入 {len(data)} 支股票")

# ── 為了快速診斷，只跑前 2 年 ──
cfg2 = TradingConfig(
    initial_equity  = 1_000_000,
    risk_pct        = 0.002,
    commission_rate = 0.001425,
    transaction_tax = 0.003,
    slippage        = 0.001,
    max_positions   = 10,
    min_avg_amount  = 5_000_000,
    backtest_start  = "2010-01-01",
    backtest_end    = "2012-12-31",      # 只跑 2 年，快速看第一批交易
    max_trade_cost  = 5_000.0,
)

engine = Backtester(cfg2)
data_ind = {t: Indicators.add_all(df, cfg2) for t, df in data.items()}

print("\n開始回測（2010~2012）...")
results = engine.run(data)

trades = results["trades"]
metrics = results["metrics"]

print("\n=== 績效指標（2010~2012）===")
print(f"  總報酬率   : {metrics['total_return_pct']:,.2f} %")
print(f"  總交易次數 : {metrics['total_trades']}")
print(f"  勝率       : {metrics['win_rate_pct']:.2f} %")

if not trades.empty:
    print(f"\n=== 前 20 筆交易 ===")
    cols = ["ticker", "direction", "lots", "shares",
            "entry_date", "adj_entry_price", "adj_exit_price",
            "pnl_net", "exit_reason"]
    cols = [c for c in cols if c in trades.columns]
    df_show = trades[cols].head(20).copy()
    df_show["entry_cost"] = df_show["adj_entry_price"] * df_show["shares"]
    print(df_show.to_string(index=False))

    print(f"\n=== 買入金額統計 ===")
    trades["entry_cost"] = trades["adj_entry_price"] * trades["shares"]
    print(f"  最大買入金額 : {trades['entry_cost'].max():,.0f}")
    print(f"  最小買入金額 : {trades['entry_cost'].min():,.0f}")
    print(f"  平均買入金額 : {trades['entry_cost'].mean():,.0f}")
    print(f"  超過 5000 元筆數 : {(trades['entry_cost'] > 5000).sum()}")

    print(f"\n=== 股價分布 ===")
    print(f"  最高進場價 : {trades['adj_entry_price'].max():,.2f}")
    print(f"  最低進場價 : {trades['adj_entry_price'].min():,.4f}")
    print(f"  最大張數   : {trades['lots'].max()}")
    print(f"  最小張數   : {trades['lots'].min()}")
    print(f"  最大 pnl_net : {trades['pnl_net'].max():,.0f}")
    print(f"  最小 pnl_net : {trades['pnl_net'].min():,.0f}")
else:
    print("\n無任何交易發生（可能全被篩選掉了）")

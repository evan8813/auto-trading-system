"""
verify.py
─────────
驗證腳本。修改下方各區塊的設定後執行即可。

執行方式：
  cd C:/Users/USER/Desktop/jupyter/TW-stock/logic/auto_trading
  python verify.py

注意：SAMPLE_DATES 填的是「信號日」= 進場日的前一個交易日
  例如 trade log 顯示 entry_date = 2013-01-15
  -> 這裡填 2013-01-14
"""

import sys
sys.path.insert(0, ".")

import pandas as pd

from config import TradingConfig
from data_loader import DataLoader
from backtester import Backtester
from checkpoint import run_checkpoint, sample_period

# ══════════════════════════════════════════════
# 設定區（每次使用只改這裡）
# ══════════════════════════════════════════════

# 資料路徑
DATA_FOLDER = r"C:\Users\USER\Desktop\jupyter\TW-stock\stock_full"

# 要抽查的信號日（可加任意多個）
SAMPLE_DATES = [
    "2013-01-14",   # entry_date 2013-01-15 的前一天
    "2013-02-21",   # entry_date 2013-02-22 的前一天
    "2013-03-05",   # entry_date 2013-03-06 的前一天
]

# ══════════════════════════════════════════════
# 載入資料 + 跑回測（固定，不需修改）
# ══════════════════════════════════════════════

cfg     = TradingConfig()
data    = DataLoader.load_folder(DATA_FOLDER, adjusted=True)
results = Backtester(cfg).run(data)

# ══════════════════════════════════════════════
# 績效摘要
# ══════════════════════════════════════════════
m = results["metrics"]
print("\n" + "=" * 60)
print("  績效摘要")
print("=" * 60)
print(f"  總報酬率       : {m['total_return_pct']:+.2f}%")
print(f"  年化報酬 (CAGR): {m['cagr_pct']:+.2f}%")
print(f"  最大回撤       : {m['max_drawdown_pct']:.2f}%")
print(f"  勝率           : {m['win_rate_pct']:.1f}%")
print(f"  總交易筆數     : {m['total_trades']}")
print(f"  平均獲利       : {m['avg_win']:,.0f} 元")
print(f"  平均虧損       : {m['avg_loss']:,.0f} 元")
print(f"  獲利因子       : {m['profit_factor']:.3f}")
print("-" * 60)

# ── 資金使用率分析 ──
trades_df_m = results["trades"]
if not trades_df_m.empty:
    per_trade     = trades_df_m["lots"] * 1000 * trades_df_m["raw_entry_price"]
    avg_deployed  = per_trade.mean()
    max_deployed  = per_trade.max()
    avg_pnl_pct   = (trades_df_m["pnl_net"] / per_trade * 100).mean()
    print(f"  每筆平均投入   : {avg_deployed:,.0f} 元（{avg_deployed/cfg.initial_equity*100:.1f}% 資金）")
    print(f"  單筆最大投入   : {max_deployed:,.0f} 元（{max_deployed/cfg.initial_equity*100:.1f}% 資金）")
    print(f"  每筆平均報酬率 : {avg_pnl_pct:+.2f}%（以投入金額計）")
print("=" * 60)

# ══════════════════════════════════════════════
# 逐年報酬 + 最大回撤
# ══════════════════════════════════════════════
ec = results["equity_curve"]["equity"]

print("\n" + "=" * 52)
print("  逐年績效")
print(f"  {'年份':<6} {'年初資金':>10} {'年末資金':>10} {'年報酬':>8} {'年內最大回撤':>12}")
print("-" * 52)

for year, group in ec.groupby(ec.index.year):
    start_eq  = group.iloc[0]
    end_eq    = group.iloc[-1]
    ret_pct   = (end_eq / start_eq - 1) * 100
    roll_max  = group.cummax()
    mdd_pct   = ((group - roll_max) / roll_max).min() * 100
    print(f"  {year:<6} {start_eq:>10,.0f} {end_eq:>10,.0f} {ret_pct:>+7.2f}% {mdd_pct:>11.2f}%")

print("=" * 52)

# ══════════════════════════════════════════════
# 最大虧損前 10 筆
# ══════════════════════════════════════════════
trades_df = results["trades"]
if not trades_df.empty:
    worst = (
        trades_df.nsmallest(10, "pnl_net")[
            ["ticker", "direction", "entry_date", "exit_date",
             "hold_days", "lots", "raw_entry_price", "raw_exit_price",
             "pnl_net", "atr_at_entry"]
        ].copy()
    )
    worst["entry_date"] = pd.to_datetime(worst["entry_date"]).dt.strftime("%Y-%m-%d")
    worst["exit_date"]  = pd.to_datetime(worst["exit_date"]).dt.strftime("%Y-%m-%d")
    print("\n" + "=" * 80)
    print("  最大虧損前 10 筆")
    print("=" * 80)
    print(worst.to_string(index=False))
    print("=" * 80)

# ══════════════════════════════════════════════
# 逐筆交易明細（預設關閉，改成 True 即可顯示）
# ══════════════════════════════════════════════
SHOW_TRADE_DETAIL = False

trades_df = results["trades"]

if SHOW_TRADE_DETAIL:
    print("\n" + "=" * 90)
    print("  逐筆交易明細")
    print("=" * 90)

    if trades_df.empty:
        print("  無交易紀錄。")
    else:
        show_cols = [
            "ticker", "direction", "lots",
            "entry_date", "raw_entry_price",
            "exit_date",  "raw_exit_price",
            "hold_days",
            "gross_pnl", "total_cost", "pnl_net",
            "atr_at_entry", "equity_at_entry",
            "exit_reason",
        ]
        show_cols = [c for c in show_cols if c in trades_df.columns]

        df_show = trades_df[show_cols].copy()
        df_show["entry_date"] = pd.to_datetime(df_show["entry_date"]).dt.strftime("%Y-%m-%d")
        df_show["exit_date"]  = pd.to_datetime(df_show["exit_date"]).dt.strftime("%Y-%m-%d")

        for col in ["raw_entry_price", "raw_exit_price", "atr_at_entry"]:
            if col in df_show.columns:
                df_show[col] = df_show[col].map("{:.2f}".format)
        for col in ["gross_pnl", "total_cost", "pnl_net", "equity_at_entry"]:
            if col in df_show.columns:
                df_show[col] = df_show[col].map("{:,.0f}".format)

        print(df_show.to_string(index=False))

        print("-" * 90)
        total     = len(trades_df)
        wins      = (trades_df["pnl_net"] > 0).sum()
        losses    = total - wins
        total_pnl = trades_df["pnl_net"].sum()
        avg_hold  = trades_df["hold_days"].mean()
        print(f"  總筆數: {total}  獲利: {wins}  虧損: {losses}  "
              f"勝率: {wins/total*100:.1f}%  總損益: {total_pnl:,.0f}  "
              f"平均持倉天數: {avg_hold:.1f}")
    print("=" * 90)

# ══════════════════════════════════════════════
# 驗證 1：全段回測（所有時間點，只看對不對）
# ══════════════════════════════════════════════
#
# 跑全部交易日，驗證每一筆交易：
#   - 進場股票確實在選股池內
#   - 進場訊號確實觸發
# 只印 PASS / FAIL 統計，不印細節
#
from checkpoint import check_trades, check_execution
from indicators import Indicators

print("\n" + "=" * 60)
print("  全段驗證（所有交易）")
print("=" * 60)
cp = check_trades(results["trades"], data, cfg)
if cp.empty:
    print("  無交易紀錄。")
else:
    total  = len(cp)
    passed = cp["ALL_PASS"].sum()
    failed = total - passed
    print(f"  共 {total} 筆  PASS {passed}  FAIL {failed}")
    print(f"  A 在選股池內          : {cp['A_in_pool'].sum()}/{total}")
    print(f"  B 訊號確實觸發        : {cp['B_signal_ok'].sum()}/{total}")
    print(f"  C 更高ROC股票無訊號   : {cp['C_higher_no_signal'].sum()}/{total}")
    if failed > 0:
        print("\n  FAIL 明細：")
        print(cp[~cp["ALL_PASS"]].to_string(index=False))
print("=" * 60)

# ══════════════════════════════════════════════
# 驗證 2：進出場執行細節（價格 / 部位大小 / 停損）
# ══════════════════════════════════════════════
#
# 逐筆確認：
#   D 進場價格  → raw_entry_price == 進場日 T+1 開盤價
#   E 部位大小  → lots 符合 equity × risk_pct ÷ ATR ÷ 1000（含 max_trade_cost 限制）
#   F 出場價格  → raw_exit_price == 出場日 T+1 開盤價
#   G 停損觸發  → 出場訊號日 T 滿足追蹤停損條件
#                 做多：Close < trail_high − atr_mult × ATR
#                 做空：Close > trail_low  + atr_mult × ATR
#
print("\n" + "=" * 60)
print("  執行細節驗證（價格 / 部位大小 / 停損）")
print("=" * 60)
cp3 = check_execution(results["trades"], data, cfg)
if cp3.empty:
    print("  無交易紀錄。")
else:
    total3  = len(cp3)
    passed3 = cp3["ALL_PASS"].sum()
    failed3 = total3 - passed3
    print(f"  共 {total3} 筆  PASS {passed3}  FAIL {failed3}")
    print(f"  D 進場條件正確  : {cp3['D_entry_cond'].sum()}/{total3}")
    print(f"  E 進場價格正確  : {cp3['E_entry_price'].sum()}/{total3}")
    print(f"  F 部位大小正確  : {cp3['F_lots'].sum()}/{total3}")
    print(f"  G 出場價格正確  : {cp3['G_exit_price'].sum()}/{total3}")
    print(f"  H 停損確實觸發  : {cp3['H_stop_trigger'].sum()}/{total3}")
    if failed3 > 0:
        print("\n  FAIL 明細：")
        for _, row in cp3[~cp3["ALL_PASS"]].iterrows():
            print(f"  {row['ticker']} {row['direction']} {row['entry_date']} → {row['exit_date']}")
            if not row["D_entry_cond"]:   print(f"    D FAIL: {row['D_note']}")
            if not row["E_entry_price"]:  print(f"    E FAIL: {row['E_note']}")
            if not row["F_lots"]:         print(f"    F FAIL: {row['F_note']}")
            if not row["G_exit_price"]:   print(f"    G FAIL: {row['G_note']}")
            if not row["H_stop_trigger"]: print(f"    H FAIL: {row['H_note']}")
    else:
        print("  全部 PASS")
print("=" * 60)

# ══════════════════════════════════════════════
# 驗證 2：抽樣特定時間點
# ══════════════════════════════════════════════
#
# 對每個日期輸出三段：
#
# ① 選股池（依 ROC 排名）
#    欄位說明：
#      rank          → ROC 排名（1 = 動能最強）
#      close         → 當日收盤價
#      close/52w_pct → 收盤 / 52週最高價（越接近100%越強）
#      avg_amount_20 → 20日平均成交金額（百萬元）
#      roc_avg       → (ROC_10 + ROC_25 + ROC_35) / 3，三週期平均動能
#      signal        → LONG / SHORT / none
#                      none = 進場條件不成立（沒突破或MA不符）
#                      LONG/SHORT = 訊號觸發，明天開盤會嘗試進場
#
# ② 策略結果
#    欄位說明：
#      entered    → True = 確實進場；False = 沒進場
#      entry_date → 實際進場日（= 信號日 + 1）
#      exit_date  → 出場日
#      hold_days  → 持有天數
#      pnl_net    → 淨損益（扣手續費、稅、滑價）
#
# ③ 診斷（signal 有觸發但 entered = False 的原因）
#    常見原因：
#      持倉已滿  → 當時持倉數已達 max_positions（預設10）
#      張數 = 0  → initial_equity × risk_pct ÷ ATR ÷ 1000 < 1
#                  代表資金太小，連一張都買不起
#      單筆超限  → 股價 × 1000 > max_trade_cost
#
for date in SAMPLE_DATES:
    sample_period(date, results, data, cfg)

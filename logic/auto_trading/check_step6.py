import sys
import pandas as pd
sys.path.insert(0, '.')
from data_loader import DataLoader
from indicators import Indicators
from config import TradingConfig
from universe_filter import UniverseFilter

cfg = TradingConfig()
uf  = UniverseFilter(cfg)

print("=== 篩選參數 ===")
print(f"最低日均成交額 : {cfg.min_avg_amount:,.0f} 元")
print(f"最高可負擔股價 : {cfg.initial_equity/1000:.0f} 元")
print(f"52W高點門檻   : 90%")

print("\n載入資料中...")
data     = DataLoader.load_folder(r'C:\Users\USER\Desktop\jupyter\TW-stock\stocks_full')
data_ind = {t: Indicators.add_all(df, cfg) for t, df in data.items()}

# 取所有股票共同有資料的交易日（每月最後一個交易日）
all_dates = sorted({d for df in data_ind.values() for d in df.index})
# 每月取最後一個交易日，從資料最後一年開始
last_date = all_dates[-1]
start_sample = pd.Timestamp(last_date.year - 1, last_date.month, 1)
monthly_dates = []
for d in all_dates:
    if d >= start_sample:
        if not monthly_dates or d.month != monthly_dates[-1].month:
            monthly_dates.append(d)
        else:
            monthly_dates[-1] = d  # 同月份取最後一天

print(f"\n資料範圍 : {all_dates[0].date()} ~ {all_dates[-1].date()}")
print(f"總股票數 : {len(data_ind)} 支")
print(f"\n{'日期':<12} {'通過支數':>8}  通過的股票代號")
print("-" * 70)

for date in monthly_dates:
    candidates = uf.filter(data_ind, date)
    tickers_str = ", ".join(sorted(candidates)[:15])
    suffix = f"... 共{len(candidates)}支" if len(candidates) > 15 else ""
    print(f"{str(date.date()):<12} {len(candidates):>8}  {tickers_str}{suffix}")

# 最後一個交易日詳細資訊
print(f"\n=== {last_date.date()} 通過股票詳細資訊 ===")
candidates = uf.filter(data_ind, last_date)
if candidates:
    print(f"{'代號':<8} {'收盤':>6} {'52W高':>7} {'Close/52W%':>11} {'日均成交(億)':>12}")
    print("-" * 50)
    for t in sorted(candidates):
        row    = data_ind[t].loc[last_date]
        ratio  = row['Close'] / row['High_52W'] * 100
        amount = row['Avg_Amount_20'] / 1e8
        print(f"{t:<8} {row['Close']:>6.1f} {row['High_52W']:>7.1f} {ratio:>11.1f} {amount:>12.2f}")
else:
    print("無符合條件的股票")

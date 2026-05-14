"""
experiment_compare_strategies.py
──────────────────────────────────
2×2 比較：
  策略A（他）：Long only, 10日突破進場, 10日低點出場, 等權
  策略B（我）：Long+Short, 10日突破進退場, ATR 2% 風控部位

  × 標的 S&P 500 個股 vs 0050 成分股

結論：分清楚「策略問題」還是「市場問題」

輸出：output/compare/summary.txt
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
from yahoo_fin import stock_info as si

OUTPUT_DIR = Path("output/compare")
START      = "2009-01-01"
END        = "2024-05-31"
WINDOW     = 10
ATR_PERIOD = 20
ATR_MULT   = 2.0
RISK_PCT   = 0.02
INIT_EQ    = 1_000_000

# 0050 成分股（主要成分）
TAIEX50 = [
    "2330.TW","2317.TW","2454.TW","2412.TW","2308.TW","2303.TW",
    "3711.TW","2881.TW","2882.TW","2886.TW","2891.TW","2892.TW",
    "1301.TW","1303.TW","2002.TW","1216.TW","2382.TW","2357.TW",
    "2379.TW","2395.TW","3008.TW","2345.TW","2207.TW","2327.TW",
    "4938.TW","2408.TW","3034.TW","6505.TW","2474.TW","5880.TW",
]


# ── 資料下載 ─────────────────────────────────────────────────────────

def download(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)
    close = raw["Close"] if "Close" in raw else raw
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = [c[0] if isinstance(c, tuple) else c for c in close.columns]
    return close.dropna(how="all")


# ── 策略 A：Long only, 等權 ───────────────────────────────────────────

def strategy_a(close: pd.DataFrame) -> pd.Series:
    high = close.rolling(WINDOW).max()
    low  = close.rolling(WINDOW).min()

    long_sig = close > high.shift(1)
    exit_sig = close < low.shift(1)

    pos = pd.DataFrame(0, index=close.index, columns=close.columns)
    for col in pos.columns:
        holding = False
        for i in range(len(pos)):
            if holding:
                pos.iloc[i][col] = 1
                if exit_sig.iloc[i][col]:
                    holding = False
            elif long_sig.iloc[i][col]:
                holding = True
                pos.iloc[i][col] = 1

    daily_ret = close.pct_change().fillna(0)
    port_ret  = (pos.shift().fillna(0) * daily_ret).mean(axis=1)
    return (1 + port_ret).cumprod() * INIT_EQ


# ── 策略 B：Long+Short, ATR 風控 ─────────────────────────────────────

def calc_atr_df(close: pd.DataFrame, high: pd.DataFrame,
                low: pd.DataFrame, period: int) -> pd.DataFrame:
    tr = pd.concat([high - low,
                    (high - close.shift(1)).abs(),
                    (low  - close.shift(1)).abs()], axis=1)
    # tr is wide; need per-ticker
    result = {}
    for col in close.columns:
        h = high[col]; l = low[col]; c = close[col]
        t = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        result[col] = t.rolling(period).mean()
    return pd.DataFrame(result)


def strategy_b(close: pd.DataFrame,
               high_df: pd.DataFrame,
               low_df: pd.DataFrame) -> pd.Series:
    roll_high = close.rolling(WINDOW).max()
    roll_low  = close.rolling(WINDOW).min()
    atr_df    = calc_atr_df(close, high_df, low_df, ATR_PERIOD)

    equity   = INIT_EQ
    positions: dict = {}   # ticker -> {direction, entry, units}
    curve    = []

    for i in range(1, len(close)):
        date     = close.index[i]
        prev_i   = i - 1

        # 出場判斷
        to_close = []
        for ticker, pos in positions.items():
            c = close.iloc[i][ticker]
            if pd.isna(c):
                continue
            if pos["direction"] == "long" and c < roll_low.iloc[prev_i][ticker]:
                to_close.append(ticker)
            elif pos["direction"] == "short" and c > roll_high.iloc[prev_i][ticker]:
                to_close.append(ticker)

        for ticker in to_close:
            pos   = positions.pop(ticker)
            c     = close.iloc[i][ticker]
            pnl   = ((c - pos["entry"]) * pos["units"]
                     if pos["direction"] == "long"
                     else (pos["entry"] - c) * pos["units"])
            equity += pnl

        # 進場判斷
        for ticker in close.columns:
            if ticker in positions:
                continue
            c    = close.iloc[i][ticker]
            atr  = atr_df.iloc[i][ticker]
            prev_high = roll_high.iloc[prev_i][ticker]
            prev_low  = roll_low.iloc[prev_i][ticker]
            if pd.isna(c) or pd.isna(atr) or atr <= 0:
                continue

            direction = None
            if c > prev_high:
                direction = "long"
            elif c < prev_low:
                direction = "short"

            if direction:
                units = (equity * RISK_PCT) / (ATR_MULT * atr)
                if units > 0:
                    positions[ticker] = {
                        "direction": direction,
                        "entry":     c,
                        "units":     units,
                    }

        curve.append({"date": date, "equity": equity})

    eq = pd.DataFrame(curve).set_index("date")["equity"]
    eq.index = pd.to_datetime(eq.index)
    return eq


# ── 績效計算 ──────────────────────────────────────────────────────────

def metrics(eq: pd.Series, label: str) -> dict:
    ret   = eq.pct_change().dropna()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)
    cagr  = ((eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1) * 100
    mdd   = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    sharpe = ret.mean() / ret.std() * 252**0.5 if ret.std() > 0 else 0
    return {"標籤": label, "CAGR%": round(cagr,1),
            "MDD%": round(mdd,1), "Sharpe": round(sharpe,2)}


# ── 主程式 ────────────────────────────────────────────────────────────

def run_universe(name: str, tickers: list[str]) -> list[dict]:
    print(f"\n{'='*55}")
    print(f"  下載 {name}（{len(tickers)} 檔）...")

    close = download(tickers, START, END)
    valid = [c for c in close.columns if close[c].notna().sum() > 200]
    close = close[valid]
    print(f"  有效標的：{len(valid)} 檔")

    # 取得 High / Low（策略B需要）
    raw   = yf.download(valid, start=START, end=END,
                        auto_adjust=True, progress=False)
    high_df = raw["High"]
    low_df  = raw["Low"]
    if isinstance(high_df.columns, pd.MultiIndex):
        high_df.columns = [c[0] for c in high_df.columns]
        low_df.columns  = [c[0] for c in low_df.columns]

    results = []

    print(f"  跑策略A（Long only 等權）...")
    eq_a = strategy_a(close)
    m_a  = metrics(eq_a, f"策略A × {name}")
    print(f"    CAGR={m_a['CAGR%']}%  MDD={m_a['MDD%']}%  Sharpe={m_a['Sharpe']}")
    results.append(m_a)

    print(f"  跑策略B（Long+Short ATR）...")
    eq_b = strategy_b(close, high_df, low_df)
    m_b  = metrics(eq_b, f"策略B × {name}")
    print(f"    CAGR={m_b['CAGR%']}%  MDD={m_b['MDD%']}%  Sharpe={m_b['Sharpe']}")
    results.append(m_b)

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # S&P 500 代表性個股（各產業均衡取樣）
    sp500 = [
        # 科技
        "AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","AMD","INTC","CRM",
        "ORCL","CSCO","QCOM","TXN","AVGO","MU","AMAT","LRCX","KLAC","ADI",
        # 金融
        "JPM","BAC","WFC","GS","MS","BLK","AXP","V","MA","COF",
        # 醫療
        "JNJ","UNH","PFE","ABBV","MRK","LLY","TMO","ABT","MDT","BMY",
        # 工業
        "CAT","DE","HON","GE","MMM","BA","LMT","RTX","UPS","FDX",
        # 消費
        "WMT","HD","COST","MCD","NKE","SBUX","TGT","LOW","DG","AMGN",
        # 能源
        "XOM","CVX","COP","SLB","EOG","PSX","VLO","MPC","HAL","OXY",
        # 公用/REITs
        "NEE","DUK","SO","AMT","PLD","SPG","CCI","EQIX","PSA","O",
    ]

    all_results = []
    all_results += run_universe("S&P500", sp500)
    all_results += run_universe("0050成分股", TAIEX50)

    # 加入 Buy & Hold 基準
    print("\n  計算 Buy & Hold 基準...")
    for ticker, label in [("SPY", "SPY Buy&Hold"), ("0050.TW", "0050 Buy&Hold")]:
        df = yf.download(ticker, start=START, end=END,
                         auto_adjust=True, progress=False)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        eq = df["Close"] / df["Close"].iloc[0] * INIT_EQ
        all_results.append(metrics(eq, label))

    # 輸出
    summary = pd.DataFrame(all_results)
    print("\n\n" + "="*55)
    print("  最終比較")
    print("="*55)
    print(summary.to_string(index=False))

    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    with open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write(summary.to_string(index=False))
    print(f"\n結果存至：{OUTPUT_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()

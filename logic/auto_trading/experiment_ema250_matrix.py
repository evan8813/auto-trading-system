"""
experiment_ema250_matrix.py
────────────────────────────
250日EMA牛熊分界線策略 × 多標的 × 多時間段 系統性比較。
策略：收盤 > 250日EMA → Long；收盤 < 250日EMA → Short

輸出：
  output/ema250_matrix/summary.csv
  output/ema250_matrix/alpha_heatmap.png
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf

OUTPUT_DIR = Path("output/ema250_matrix")

TICKERS = {
    "SPY":     "美股大盤",
    "QQQ":     "美股科技",
    "GLD":     "黃金",
    "DBC":     "商品",
    "TLT":     "長期公債",
    "0050.TW": "台股ETF",
}

PERIODS = {
    "2000-2008": ("2000-01-01", "2008-12-31"),
    "2008-2012": ("2008-01-01", "2012-12-31"),
    "2012-2020": ("2012-01-01", "2020-12-31"),
    "2020-2024": ("2020-01-01", "2024-12-31"),
    "全期":       ("2000-01-01", "2024-12-31"),
}

EMA_WINDOW = 250


def load(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    df = yf.download(ticker, start=start, end=end,
                     auto_adjust=True, progress=False)
    if df.empty:
        return None
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Close"]].dropna()
    if len(df) < EMA_WINDOW + 10:
        return None
    return df


def run_strategy(df: pd.DataFrame) -> pd.Series:
    close  = df["Close"]
    ema250 = close.ewm(span=EMA_WINDOW, adjust=False).mean()

    signal = pd.Series(0, index=close.index)
    signal[close > ema250] =  1
    signal[close < ema250] = -1

    ret       = close.pct_change().fillna(0)
    strat_ret = (ret * signal.shift(1).fillna(0)).fillna(0)

    equity = (1 + strat_ret).cumprod() * 100_000
    return equity


def metrics(eq: pd.Series, df_raw: pd.DataFrame) -> dict:
    ret   = eq.pct_change().dropna()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)
    cagr  = ((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1) * 100
    mdd   = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    sharpe = ret.mean() / ret.std() * (252 ** 0.5) if ret.std() > 0 else 0

    bh_ret  = (df_raw["Close"].iloc[-1] / df_raw["Close"].iloc[0] - 1)
    bh_cagr = ((1 + bh_ret) ** (1 / years) - 1) * 100

    return {
        "CAGR%":    round(cagr, 1),
        "MDD%":     round(mdd, 1),
        "Sharpe":   round(sharpe, 2),
        "BH_CAGR%": round(bh_cagr, 1),
        "Alpha%":   round(cagr - bh_cagr, 1),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for ticker, name in TICKERS.items():
        print(f"\n{'='*50}")
        print(f"  {ticker}（{name}）")
        for period, (start, end) in PERIODS.items():
            df = load(ticker, start, end)
            if df is None:
                print(f"    {period}: 無資料")
                rows.append({"標的": f"{ticker}({name})", "期間": period,
                             "CAGR%": "N/A", "MDD%": "N/A",
                             "Sharpe": "N/A", "BH_CAGR%": "N/A", "Alpha%": "N/A"})
                continue

            eq = run_strategy(df)
            m  = metrics(eq, df)
            print(f"    {period}: CAGR={m['CAGR%']:>6}%  MDD={m['MDD%']:>7}%  "
                  f"Sharpe={m['Sharpe']:>5}  BH={m['BH_CAGR%']:>6}%  "
                  f"Alpha={m['Alpha%']:>+6}%")
            rows.append({"標的": f"{ticker}({name})", "期間": period, **m})

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")

    # ── Alpha 熱力圖 ──
    pivot = summary[summary["Alpha%"] != "N/A"].copy()
    pivot["Alpha%"] = pd.to_numeric(pivot["Alpha%"])
    pivot = pivot.pivot(index="標的", columns="期間", values="Alpha%")
    period_order = list(PERIODS.keys())
    pivot = pivot.reindex(columns=[p for p in period_order if p in pivot.columns])

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot.values.astype(float), cmap="RdYlGn", aspect="auto",
                   vmin=-20, vmax=20)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not pd.isna(val):
                ax.text(j, i, f"{val:+.1f}%", ha="center", va="center",
                        fontsize=9, color="black")
    plt.colorbar(im, ax=ax, label="Alpha（EMA250策略 CAGR − Buy&Hold CAGR）")
    ax.set_title("250日EMA牛熊策略 Alpha 熱力圖（綠=跑贏，紅=跑輸）")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "alpha_heatmap.png", dpi=150)
    plt.close()
    print(f"\n結果：{OUTPUT_DIR / 'summary.csv'}")
    print(f"熱力圖：{OUTPUT_DIR / 'alpha_heatmap.png'}")


if __name__ == "__main__":
    main()

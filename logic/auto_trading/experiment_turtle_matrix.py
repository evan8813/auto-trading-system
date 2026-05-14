"""
experiment_turtle_matrix.py
────────────────────────────
海龜策略 × 多標的 × 多時間段 系統性比較。

輸出：
  output/turtle_matrix/summary.csv   ← 完整對比表
  output/turtle_matrix/summary.txt   ← 可讀版本
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

OUTPUT_DIR = Path("output/turtle_matrix")

# ── 測試矩陣設定 ──────────────────────────────────────────────────────
TICKERS = {
    "SPY":    "美股大盤",
    "QQQ":    "美股科技",
    "GLD":    "黃金",
    "DBC":    "商品",
    "TLT":    "長期公債",
    "0050.TW":"台股ETF",
}

PERIODS = {
    "2000-2008": ("2000-01-01", "2008-12-31"),
    "2008-2012": ("2008-01-01", "2012-12-31"),
    "2012-2020": ("2012-01-01", "2020-12-31"),
    "2020-2024": ("2020-01-01", "2024-12-31"),
    "全期":       ("2000-01-01", "2024-12-31"),
}

# 海龜參數
ENTRY_WINDOW = 20
EXIT_WINDOW  = 10
ATR_PERIOD   = 20
ATR_MULT     = 2.0
RISK_PCT     = 0.02
# ─────────────────────────────────────────────────────────────────────


def calc_atr(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l,
                    (h - c.shift(1)).abs(),
                    (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def load(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    df = yf.download(ticker, start=start, end=end,
                     auto_adjust=True, progress=False)
    if df.empty:
        return None
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Open", "High", "Low", "Close"]].dropna()
    if len(df) < 60:
        return None

    df["ATR"]     = calc_atr(df, ATR_PERIOD)
    df["High_20"] = df["Close"].rolling(ENTRY_WINDOW).max()
    df["Low_20"]  = df["Close"].rolling(ENTRY_WINDOW).min()
    df["High_10"] = df["High"].rolling(EXIT_WINDOW).max()
    df["Low_10"]  = df["Low"].rolling(EXIT_WINDOW).min()
    return df.dropna()


@dataclass
class Pos:
    direction:   str
    entry_price: float
    units:       float


def run_turtle(df: pd.DataFrame) -> pd.Series:
    dates    = df.index.tolist()
    equity   = 100_000.0
    position: Pos | None = None
    curve    = []
    pending_exit  = False
    pending_entry: tuple | None = None

    for i, date in enumerate(dates):
        if i == 0:
            curve.append({"date": date, "equity": equity})
            continue

        row  = df.loc[date]
        prev = df.iloc[i - 1]
        open_px = float(row["Open"])

        if pending_exit and position is not None:
            pnl = ((open_px - position.entry_price) * position.units
                   if position.direction == "long"
                   else (position.entry_price - open_px) * position.units)
            equity   += pnl
            position  = None
            pending_exit = False

        if pending_entry is not None and position is None:
            direction, atr, _ = pending_entry
            units = (equity * RISK_PCT) / (ATR_MULT * atr) if atr > 0 else 0
            if units > 0:
                position = Pos(direction, open_px, units)
            pending_entry = None

        if position is not None:
            close_px = float(row["Close"])
            if position.direction == "long":
                if close_px < float(prev["Low_10"]):
                    pending_exit = True
            else:
                if close_px > float(prev["High_10"]):
                    pending_exit = True

        if position is None and pending_entry is None and not pending_exit:
            close_px = float(row["Close"])
            atr      = float(row["ATR"])
            if pd.notna(atr) and atr > 0:
                if close_px > float(prev["High_20"]):
                    pending_entry = ("long", atr, close_px)
                elif close_px < float(prev["Low_20"]):
                    pending_entry = ("short", atr, close_px)

        curve.append({"date": date, "equity": equity})

    eq = pd.DataFrame(curve).set_index("date")["equity"]
    eq.index = pd.to_datetime(eq.index)
    return eq


def metrics(eq: pd.Series, df_raw: pd.DataFrame) -> dict:
    if len(eq) < 2:
        return {}
    ret   = eq.pct_change().dropna()
    total = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)
    cagr  = ((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1) * 100
    mdd   = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    sharpe = ret.mean() / ret.std() * (252 ** 0.5) if ret.std() > 0 else 0

    # Buy & Hold CAGR
    bh_ret = (df_raw["Close"].iloc[-1] / df_raw["Close"].iloc[0] - 1)
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

            eq = run_turtle(df)
            m  = metrics(eq, df)
            print(f"    {period}: 海龜={m['CAGR%']:>6}%  BH={m['BH_CAGR%']:>6}%  "
                  f"Alpha={m['Alpha%']:>+6}%  MDD={m['MDD%']:>7}%  Sharpe={m['Sharpe']:>5}")
            rows.append({"標的": f"{ticker}({name})", "期間": period, **m})

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")

    # ── 可讀版輸出 ──
    with open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write("海龜策略 × 多標的 × 多時間段 比較\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'標的':<16} {'期間':<12} {'CAGR%':>8} {'MDD%':>8} "
                f"{'Sharpe':>8} {'BH%':>8} {'Alpha%':>8}\n")
        f.write("-" * 80 + "\n")
        prev = ""
        for _, r in summary.iterrows():
            if r["標的"] != prev:
                f.write("\n")
                prev = r["標的"]
            f.write(f"{r['標的']:<16} {r['期間']:<12} "
                    f"{str(r['CAGR%']):>8} {str(r['MDD%']):>8} "
                    f"{str(r['Sharpe']):>8} {str(r['BH_CAGR%']):>8} "
                    f"{str(r['Alpha%']):>8}\n")

    print(f"\n\n結果已存：{OUTPUT_DIR / 'summary.csv'}")
    print(f"          {OUTPUT_DIR / 'summary.txt'}")

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
    plt.colorbar(im, ax=ax, label="Alpha (海龜 CAGR − Buy&Hold CAGR)")
    ax.set_title("海龜策略 Alpha 熱力圖（綠=跑贏 Buy&Hold，紅=跑輸）")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "alpha_heatmap.png", dpi=150)
    plt.close()
    print(f"          {OUTPUT_DIR / 'alpha_heatmap.png'}")


if __name__ == "__main__":
    main()

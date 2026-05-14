"""
experiment_turtle_cl.py
───────────────────────
海龜交易法則（原版規則）跑原油期貨 CL=F。

規則：
  進場：收盤突破 20 日高點（做多）/ 跌破 20 日低點（做空）
  出場：收盤跌破 10 日低點（做多）/ 突破 10 日高點（做空）
  停損：每筆風險 = 帳戶 2%
  部位：帳戶風險 / (ATR × N倍)
  不加碼（簡化版）

執行方式：
  python experiment_turtle_cl.py

輸出：
  output/turtle_cl/equity_curve.png
  output/turtle_cl/trade_log.csv
  output/turtle_cl/metrics.txt
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

matplotlib.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 設定 ──────────────────────────────────────────────────────────────
TICKER         = "SPY"
START_DATE     = "2010-01-01"
INITIAL_EQUITY = 100_000     # 10萬美元
RISK_PCT       = 0.02         # 每筆風險 2%
ATR_PERIOD     = 20           # ATR 週期
ENTRY_WINDOW   = 20           # 進場：20 日突破
EXIT_WINDOW    = 10           # 出場：10 日反向突破
ATR_MULT       = 2.0          # 停損倍數（2 ATR）
OUTPUT_DIR     = Path("output/turtle_cl")
# ─────────────────────────────────────────────────────────────────────


def load_data() -> pd.DataFrame:
    print(f"下載 {TICKER} 資料...")
    df = yf.download(TICKER, start=START_DATE, auto_adjust=True, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    # 指標
    df["ATR"] = calc_atr(df, ATR_PERIOD)
    df["High_20"] = df["Close"].rolling(ENTRY_WINDOW).max()  # 進場用收盤
    df["Low_20"]  = df["Close"].rolling(ENTRY_WINDOW).min()
    df["High_10"] = df["High"].rolling(EXIT_WINDOW).max()    # 出場用最高最低
    df["Low_10"]  = df["Low"].rolling(EXIT_WINDOW).min()

    return df.dropna()


def calc_atr(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l,
                    (h - c.shift(1)).abs(),
                    (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


@dataclass
class Position:
    direction:   str
    entry_price: float
    entry_date:  pd.Timestamp
    units:       float
    atr:         float


def run() -> dict:
    df = load_data()
    dates = df.index.tolist()

    equity   = INITIAL_EQUITY
    position: Position | None = None
    trades:   list[dict] = []
    curve:    list[dict] = []

    pending_exit  = False
    pending_entry: tuple | None = None  # (direction, atr, entry_close)

    for i, date in enumerate(dates):
        if i == 0:
            curve.append({"date": date, "equity": equity})
            continue

        row  = df.loc[date]
        prev = df.iloc[i - 1]
        open_px = float(row["Open"])

        # ── (A) T+1 執行前一日訊號 ──
        if pending_exit and position is not None:
            pnl = _close(position, open_px, date)
            equity += pnl["pnl"]
            trades.append(pnl)
            position = None
            pending_exit = False

        if pending_entry is not None and position is None:
            direction, atr, _ = pending_entry
            risk   = equity * RISK_PCT
            units  = risk / (ATR_MULT * atr) if atr > 0 else 0
            if units > 0:
                position = Position(direction, open_px, date, units, atr)
            pending_entry = None

        # ── (B) 更新出場訊號 ──
        if position is not None:
            close_px = float(row["Close"])
            if position.direction == "long":
                # 出場：收盤跌破 10 日低點
                if close_px < float(prev["Low_10"]):
                    pending_exit = True
            else:
                # 出場：收盤突破 10 日高點
                if close_px > float(prev["High_10"]):
                    pending_exit = True

        # ── (C) 產生進場訊號 ──
        if position is None and pending_entry is None and not pending_exit:
            close_px = float(row["Close"])
            atr      = float(row["ATR"])
            if pd.notna(atr) and atr > 0:
                # 做多：收盤突破前日 20 日高點
                if close_px > float(prev["High_20"]):
                    pending_entry = ("long", atr, close_px)
                # 做空：收盤跌破前日 20 日低點
                elif close_px < float(prev["Low_20"]):
                    pending_entry = ("short", atr, close_px)

        curve.append({"date": date, "equity": equity})

    return {
        "equity_curve": pd.DataFrame(curve).set_index("date"),
        "trades":       pd.DataFrame(trades),
    }


def _close(pos: Position, exit_price: float, exit_date: pd.Timestamp) -> dict:
    if pos.direction == "long":
        pnl = (exit_price - pos.entry_price) * pos.units
    else:
        pnl = (pos.entry_price - exit_price) * pos.units
    return {
        "direction":   pos.direction,
        "entry_date":  pos.entry_date,
        "exit_date":   exit_date,
        "hold_days":   (exit_date - pos.entry_date).days,
        "entry_price": round(pos.entry_price, 2),
        "exit_price":  round(exit_price, 2),
        "units":       round(pos.units, 4),
        "pnl":         round(pnl, 2),
    }


def calc_metrics(equity: pd.Series) -> dict:
    ret    = equity.pct_change().dropna()
    total  = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    years  = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr   = ((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) * 100
    mdd    = ((equity - equity.cummax()) / equity.cummax()).min() * 100
    sharpe = (ret.mean() / ret.std() * (252 ** 0.5)) if ret.std() > 0 else 0
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    return {
        "累積報酬 (%)": round(total, 2),
        "年化報酬 (%)": round(cagr, 2),
        "最大回撤 (%)": round(mdd, 2),
        "Sharpe":       round(sharpe, 3),
        "Calmar":       round(calmar, 3),
    }


def main() -> None:
    results   = run()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    equity  = results["equity_curve"]["equity"]
    trades  = results["trades"]
    metrics = calc_metrics(equity)

    print("\n═══ 海龜策略績效（CL=F 原油期貨）═══")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    if not trades.empty:
        wins = trades[trades["pnl"] > 0]
        print(f"  交易筆數: {len(trades)}")
        print(f"  勝率: {round(len(wins)/len(trades)*100, 1)}%")
        print(f"  平均獲利: {round(wins['pnl'].mean(), 0) if len(wins) else 0}")
        print(f"  平均虧損: {round(trades[trades['pnl']<=0]['pnl'].mean(), 0) if len(trades[trades['pnl']<=0]) else 0}")

    # 圖表
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(equity.index, equity.values, linewidth=1.5)
    ax.set_title(f"海龜策略 - {TICKER}（20日突破進場 / 10日反向出場 / 2ATR停損）")
    ax.set_ylabel("淨值（美元）")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "equity_curve.png", dpi=150)
    plt.close()
    print(f"\n圖表：{OUTPUT_DIR / 'equity_curve.png'}")

    if not trades.empty:
        trades.to_csv(OUTPUT_DIR / "trade_log.csv", index=False, encoding="utf-8-sig")
        print(f"交易紀錄：{OUTPUT_DIR / 'trade_log.csv'}")

    with open(OUTPUT_DIR / "metrics.txt", "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")


if __name__ == "__main__":
    main()

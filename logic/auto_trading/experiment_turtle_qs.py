"""
experiment_turtle_qs.py
────────────────────────
執行海龜策略（SPY）並用 QuantStats 輸出詳細績效報告。

執行：
  python experiment_turtle_qs.py

輸出：
  output/turtle_qs/tearsheet.html   ← 完整 HTML 報告
  output/turtle_qs/metrics.txt      ← 文字版指標
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # 非互動模式，避免彈視窗

import pandas as pd
import yfinance as yf
import quantstats as qs

# ── 設定 ──────────────────────────────────────────────────────────────
TICKER         = "SPY"
BENCHMARK      = "SPY"       # 基準：買進持有 SPY
START_DATE     = "2010-01-01"
INITIAL_EQUITY = 100_000
RISK_PCT       = 0.02
ATR_PERIOD     = 20
ENTRY_WINDOW   = 20
EXIT_WINDOW    = 10
ATR_MULT       = 2.0
OUTPUT_DIR     = Path("output/turtle_qs")
# ─────────────────────────────────────────────────────────────────────


def calc_atr(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l,
                    (h - c.shift(1)).abs(),
                    (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def load_data() -> pd.DataFrame:
    print(f"下載 {TICKER} 資料...")
    df = yf.download(TICKER, start=START_DATE, auto_adjust=True, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df["ATR"]    = calc_atr(df, ATR_PERIOD)
    df["High_20"] = df["Close"].rolling(ENTRY_WINDOW).max()
    df["Low_20"]  = df["Close"].rolling(ENTRY_WINDOW).min()
    df["High_10"] = df["High"].rolling(EXIT_WINDOW).max()
    df["Low_10"]  = df["Low"].rolling(EXIT_WINDOW).min()
    return df.dropna()


def run() -> pd.Series:
    """回傳每日 equity Series（index = DatetimeIndex）"""
    df     = load_data()
    dates  = df.index.tolist()
    equity = INITIAL_EQUITY

    from dataclasses import dataclass

    @dataclass
    class Pos:
        direction:   str
        entry_price: float
        units:       float

    position: Pos | None = None
    curve: list[dict]    = []
    pending_exit         = False
    pending_entry: tuple | None = None

    for i, date in enumerate(dates):
        if i == 0:
            curve.append({"date": date, "equity": equity})
            continue

        row  = df.loc[date]
        prev = df.iloc[i - 1]
        open_px = float(row["Open"])

        # T+1 執行前一日訊號
        if pending_exit and position is not None:
            if position.direction == "long":
                pnl = (open_px - position.entry_price) * position.units
            else:
                pnl = (position.entry_price - open_px) * position.units
            equity   += pnl
            position  = None
            pending_exit = False

        if pending_entry is not None and position is None:
            direction, atr, _ = pending_entry
            risk  = equity * RISK_PCT
            units = risk / (ATR_MULT * atr) if atr > 0 else 0
            if units > 0:
                position = Pos(direction, open_px, units)
            pending_entry = None

        # 出場訊號
        if position is not None:
            close_px = float(row["Close"])
            if position.direction == "long":
                if close_px < float(prev["Low_10"]):
                    pending_exit = True
            else:
                if close_px > float(prev["High_10"]):
                    pending_exit = True

        # 進場訊號
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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    equity = run()

    # 轉成每日報酬率（QuantStats 需要）
    returns = equity.pct_change().dropna()
    returns.name = f"Turtle_{TICKER}"

    # ── HTML 完整報告 ──
    html_path = OUTPUT_DIR / "tearsheet.html"
    qs.reports.html(
        returns,
        benchmark=BENCHMARK,
        output=str(html_path),
        title=f"海龜策略 vs SPY Buy & Hold（{START_DATE}～今）",
    )
    print(f"\nHTML 報告：{html_path}")

    # ── 文字版指標 ──
    print("\n═══ QuantStats 核心指標 ═══")
    metrics_dict = {
        "累積報酬":        qs.stats.comp(returns),
        "年化報酬 (CAGR)": qs.stats.cagr(returns),
        "Sharpe":          qs.stats.sharpe(returns),
        "Sortino":         qs.stats.sortino(returns),
        "最大回撤":        qs.stats.max_drawdown(returns),
        "Calmar":          qs.stats.calmar(returns),
        "勝率":            qs.stats.win_rate(returns),
        "獲利因子":        qs.stats.profit_factor(returns),
        "平均獲利/虧損比": qs.stats.profit_ratio(returns),
        "Omega":           qs.stats.omega(returns),
        "Skew":            qs.stats.skew(returns),
        "Kurtosis":        qs.stats.kurtosis(returns),
        "VaR (95%)":       qs.stats.var(returns),
        "CVaR (95%)":      qs.stats.cvar(returns),
    }
    with open(OUTPUT_DIR / "metrics.txt", "w", encoding="utf-8") as f:
        for k, v in metrics_dict.items():
            line = f"  {k:<22}: {round(float(v), 4)}"
            print(line)
            f.write(line + "\n")

    # ── 與 SPY Buy & Hold 對比 ──
    print("\n═══ 與 SPY Buy & Hold 對比 ═══")
    spy_raw = yf.download(BENCHMARK, start=START_DATE, auto_adjust=True, progress=False)
    spy_raw.columns = [c[0] if isinstance(c, tuple) else c for c in spy_raw.columns]
    spy_ret = spy_raw["Close"].pct_change().dropna()
    spy_ret.index = pd.to_datetime(spy_ret.index)
    cmp = {
        "策略 CAGR":   round(float(qs.stats.cagr(returns))      * 100, 2),
        "SPY   CAGR":  round(float(qs.stats.cagr(spy_ret))       * 100, 2),
        "策略 MDD":    round(float(qs.stats.max_drawdown(returns))* 100, 2),
        "SPY   MDD":   round(float(qs.stats.max_drawdown(spy_ret))* 100, 2),
        "策略 Sharpe": round(float(qs.stats.sharpe(returns)), 3),
        "SPY   Sharpe":round(float(qs.stats.sharpe(spy_ret)), 3),
    }
    for k, v in cmp.items():
        print(f"  {k}: {v}{'%' if 'CAGR' in k or 'MDD' in k else ''}")


if __name__ == "__main__":
    main()

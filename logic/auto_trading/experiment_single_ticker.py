"""
experiment_single_ticker.py
────────────────────────────
單一標的（0050.TW）四組變數系統測試。
每次只改一個變數，其他固定。

基準（Baseline）：
  標的   : 0050.TW
  進場   : V2（High_N突破 + 量能）
  出場   : 50日低點
  部位   : 等權（持倉時全資金追蹤報酬）
  大盤   : 無過濾

第一組：只換出場條件
第二組：只換部位大小方式
第三組：只換進場條件（V1→V5）
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

OUTPUT_DIR = Path("output/single_ticker")

# ── 固定參數 ──────────────────────────────────────────────────────────
N        = 50
VOL_MULT = 1.5
MA_FAST  = 50
MA_SLOW  = 100
ATR_P    = 100
ATR_MULT = 5.0
RISK_PCT = 0.01
INIT_EQ  = 1_000_000
START    = "2005-01-01"
END      = "2024-12-31"


# ── 資料下載 ──────────────────────────────────────────────────────────

def download(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, start="2000-01-01", end=END,
                     auto_adjust=True, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    # 指標
    df["High_N"]   = df["Close"].rolling(N).max().shift(1)
    df["Low_N"]    = df["Close"].rolling(N).min().shift(1)
    df["Vol_MA20"] = df["Volume"].rolling(20).mean()
    df["MA_fast"]  = df["Close"].rolling(MA_FAST).mean()
    df["MA_slow"]  = df["Close"].rolling(MA_SLOW).mean()
    df["High_52W"] = df["High"].rolling(252).max().shift(1)
    df["Low_52W"]  = df["Low"].rolling(252).min().shift(1)

    # ATR
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"]  - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(ATR_P).mean()

    return df.dropna()


def download_taiex_bull() -> pd.Series:
    df = yf.download("^TWII", start="2000-01-01", end=END,
                     auto_adjust=True, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    ema = df["Close"].ewm(span=200, adjust=False).mean()
    return (df["Close"] > ema)


# ── 進場條件判斷 ──────────────────────────────────────────────────────

def entry_ok(row, prev_row, version: int, bull: bool) -> bool:
    c   = row["Close"];  h = prev_row["High_N"]
    v   = row["Volume"]; vm = row["Vol_MA20"]
    mf  = row["MA_fast"]; ms = row["MA_slow"]
    h52 = prev_row["High_52W"]

    if pd.isna(c) or pd.isna(h):
        return False
    if not (c > h):                          # V1: 突破
        return False
    if version >= 2:
        if pd.isna(v) or pd.isna(vm) or not (v > vm * VOL_MULT):
            return False
    if version >= 3:
        if pd.isna(mf) or pd.isna(ms) or not (mf > ms):
            return False
    if version >= 4:
        if not bull:
            return False
    if version >= 5:
        if pd.isna(h52) or not (c > h52):
            return False
    return True


# ── 回測核心 ──────────────────────────────────────────────────────────

def backtest(
    df:          pd.DataFrame,
    bull_series: pd.Series,
    exit_mode:   str,    # "low_n" | "atr"
    sizing_mode: str,    # "equal" | "atr"
    version:     int,    # 1-5
) -> dict:
    """
    單一標的回測。

    equal sizing：持倉時 equity 完整追蹤收盤報酬率（等於全資金投入）
    atr sizing  ：進場時算出 units，每日 equity += price_change × units
    """
    idx = (df.index >= START) & (df.index <= END)
    d   = df.loc[idx].copy()
    bull = bull_series.reindex(d.index, method="ffill").fillna(False).astype(bool)

    equity   = float(INIT_EQ)
    holding  = False
    entry_px = 0.0
    units    = 0.0
    trail_h  = 0.0
    curve    = []
    n_trades = 0

    for i in range(1, len(d)):
        row      = d.iloc[i]
        prev_row = d.iloc[i - 1]
        c = float(row["Close"])

        if holding:
            # 更新追蹤高點（ATR出場用）
            trail_h = max(trail_h, c)

            # 出場判斷
            exit_triggered = False
            if exit_mode == "low_n":
                ln = prev_row["Low_N"]
                if not pd.isna(ln) and c < float(ln):
                    exit_triggered = True
            else:  # atr
                atr = float(row["ATR"])
                if not pd.isna(atr) and c < trail_h - ATR_MULT * atr:
                    exit_triggered = True

            if exit_triggered:
                if sizing_mode == "atr":
                    equity += (c - entry_px) * units
                holding  = False
                n_trades += 1
                continue

            # 持倉中更新 equity
            if sizing_mode == "equal":
                prev_c  = float(prev_row["Close"])
                if prev_c > 0:
                    equity *= (c / prev_c)
            else:
                equity += (c - float(prev_row["Close"])) * units

        else:
            # 進場判斷
            b = bool(bull.iloc[i])
            if entry_ok(row, prev_row, version, b):
                holding  = True
                entry_px = c
                trail_h  = c
                if sizing_mode == "atr":
                    atr = float(row["ATR"])
                    if pd.isna(atr) or atr <= 0:
                        holding = False
                    else:
                        risk  = min(equity * RISK_PCT, 2_500)
                        units = risk / (ATR_MULT * atr)
                        if units <= 0:
                            holding = False

        curve.append({"date": d.index[i], "equity": equity})

    if not curve:
        return {}

    eq     = pd.DataFrame(curve).set_index("date")["equity"]
    ret    = eq.pct_change().dropna()
    years  = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)
    cagr   = ((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1) * 100
    mdd    = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    sharpe = ret.mean() / ret.std() * 252**0.5 if ret.std() > 0 else 0

    return {
        "CAGR%":  round(cagr,   1),
        "MDD%":   round(mdd,    1),
        "Sharpe": round(sharpe, 2),
        "筆數":   n_trades,
        "equity": eq,
    }


# ── 主程式 ────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("下載 0050.TW 資料...")
    df   = download("0050.TW")
    bull = download_taiex_bull()
    print(f"資料筆數：{len(df)}，期間：{df.index[0].date()} ~ {df.index[-1].date()}")

    results = []

    def run(label, exit_mode, sizing_mode, version):
        m = backtest(df, bull, exit_mode, sizing_mode, version)
        if not m:
            return
        results.append({
            "測試": label,
            "出場": exit_mode,
            "部位": sizing_mode,
            "版本": f"V{version}",
            **{k: v for k, v in m.items() if k != "equity"},
        })
        print(f"  {label:<30} CAGR={m['CAGR%']:>6}%  MDD={m['MDD%']:>7}%  "
              f"Sharpe={m['Sharpe']:>5}  筆數={m['筆數']}")
        return m["equity"]

    # ── 第一組：只換出場條件 ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  第一組：只換出場條件（進場V2，等權，無大盤過濾）")
    print(f"{'='*60}")
    eq_1a = run("T1a Baseline（50日低點）",      "low_n", "equal", 2)
    eq_1b = run("T1b ATR追蹤停損（5×ATR100）",   "atr",   "equal", 2)

    # ── 第二組：只換部位大小（出場固定50日低點）────────────────────────
    print(f"\n{'='*60}")
    print("  第二組：只換部位大小（進場V2，50日低點出場）")
    print(f"{'='*60}")
    run("T2a Baseline（等權）",          "low_n", "equal", 2)
    run("T2b ATR sizing",                "low_n", "atr",   2)

    # ── 第三組：只換進場條件（出場50日低點，等權）──────────────────────
    print(f"\n{'='*60}")
    print("  第三組：只換進場條件（50日低點出場，等權）")
    print(f"{'='*60}")
    eqs_3 = []
    labels_3 = [
        "T3a V1（High_N突破）",
        "T3b V2（+量能）",
        "T3c V3（+MA趨勢）",
        "T3d V4（+大盤EMA）",
        "T3e V5（+52W突破）",
    ]
    for i, lb in enumerate(labels_3, start=1):
        eq = run(lb, "low_n", "equal", i)
        eqs_3.append(eq)

    # ── 報表 ──────────────────────────────────────────────────────────
    df_res = pd.DataFrame([{k: v for k, v in r.items()} for r in results])
    print(f"\n\n{'='*60}")
    print("  完整結果")
    print(f"{'='*60}")
    print(df_res[["測試","CAGR%","MDD%","Sharpe","筆數"]].to_string(index=False))
    df_res.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")

    # ── 走勢圖 ──────────────────────────────────────────────────────
    # Buy & Hold 基準
    idx  = (df.index >= START) & (df.index <= END)
    bh   = df.loc[idx, "Close"] / df.loc[idx, "Close"].iloc[0] * INIT_EQ

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 圖1：出場條件
    ax = axes[0]
    ax.set_title("第一組：出場條件")
    bh.plot(ax=ax, color="gray", linestyle="--", linewidth=1, label="Buy&Hold")
    if eq_1a is not None: eq_1a.plot(ax=ax, label="T1a 50日低點", linewidth=1.5)
    if eq_1b is not None: eq_1b.plot(ax=ax, label="T1b ATR停損",  linewidth=1.5)
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylabel("淨值")

    # 圖2：部位大小（從 results 撈）
    ax = axes[1]
    ax.set_title("第二組：部位大小")
    bh.plot(ax=ax, color="gray", linestyle="--", linewidth=1, label="Buy&Hold")
    for r in results:
        if r["測試"].startswith("T2"):
            eq_tmp = backtest(df, bull, r["出場"], r["部位"], int(r["版本"][1:]))
            if eq_tmp and eq_tmp.get("equity") is not None:
                pass
    # 直接重跑取 equity（已在 run() 裡列印但沒保存，補畫）
    for label, em, sm in [("T2a 等權", "low_n","equal"), ("T2b ATR sizing","low_n","atr")]:
        m = backtest(df, bull, em, sm, 2)
        if m and m.get("equity") is not None:
            m["equity"].plot(ax=ax, label=label, linewidth=1.5)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 圖3：進場條件
    ax = axes[2]
    ax.set_title("第三組：進場條件（V1→V5）")
    bh.plot(ax=ax, color="gray", linestyle="--", linewidth=1, label="Buy&Hold")
    colors3 = ["#aaaaaa","#3498db","#f39c12","#9b59b6","#e74c3c"]
    for i, (lb, eq) in enumerate(zip(labels_3, eqs_3)):
        if eq is not None:
            eq.plot(ax=ax, label=lb.split(" ")[0]+" "+lb.split(" ")[1],
                    linewidth=1.5, color=colors3[i])
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle("0050.TW 單標的 變數隔離測試", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "equity_curves.png", dpi=150)
    plt.close()

    print(f"\n圖表：{OUTPUT_DIR / 'equity_curves.png'}")
    print(f"報表：{OUTPUT_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()

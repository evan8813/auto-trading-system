"""
experiment_sp500_incremental.py
────────────────────────────────
S&P 500 × 多空各版本 × 多時間段 矩陣測試。

條件逐步疊加（多空各一套）：
  V1  High/Low_N 突破
  V2  + 量能（Volume > Vol_MA20 × 1.5）
  V3  + MA 趨勢（MA_fast > MA_slow / MA_fast < MA_slow）
  V4  + 大盤 SPY EMA200（多方需 SPY > EMA200；空方需 SPY < EMA200）
  V5  + 52W 突破（High > High_252W / Low < Low_252W）

設計決定：
  - 等權分散（每日持倉均攤，不做 ATR 風控）
  - N = 50 日（High_N / Low_N）
  - 出場：持倉期間收盤跌破 N 日低（多方）/ 突破 N 日高（空方）
  - 大盤基準：SPY 200 日 EMA

輸出：
  output/sp500_incremental/long_heatmap.png
  output/sp500_incremental/short_heatmap.png
  output/sp500_incremental/summary.csv
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

OUTPUT_DIR = Path("output/sp500_incremental")

# ── 設定 ──────────────────────────────────────────────────────────────
TICKERS = [
    "AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","AMD","INTC","CRM",
    "ORCL","CSCO","QCOM","TXN","AVGO","MU","AMAT","LRCX","ADI",
    "JPM","BAC","WFC","GS","MS","BLK","AXP","V","MA","COF",
    "JNJ","UNH","PFE","ABBV","MRK","LLY","TMO","ABT","MDT","BMY",
    "CAT","DE","HON","GE","MMM","BA","LMT","RTX","UPS","FDX",
    "WMT","HD","COST","MCD","NKE","SBUX","TGT","LOW","DG","AMGN",
    "XOM","CVX","COP","SLB","EOG","PSX","VLO","MPC","HAL","OXY",
    "NEE","DUK","SO","AMT","PLD","SPG","CCI","EQIX","PSA","O",
]

PERIODS = {
    "2000-2008": ("2000-01-01", "2008-12-31"),
    "2008-2012": ("2008-01-01", "2012-12-31"),
    "2012-2020": ("2012-01-01", "2020-12-31"),
    "2020-2024": ("2020-01-01", "2024-12-31"),
    "全期":       ("2000-01-01", "2024-12-31"),
}

VERSIONS = ["V1 突破", "V2 +量能", "V3 +MA", "V4 +大盤EMA", "V5 +52W"]

N        = 50     # 突破 / 出場窗口
VOL_MULT = 1.5
MA_FAST  = 50
MA_SLOW  = 100
EMA_SPY  = 200
W52      = 252
FULL_START = "1998-01-01"   # 多下一些歷史，讓指標暖機充足
FULL_END   = "2024-12-31"


# ── 資料下載 ──────────────────────────────────────────────────────────

def download_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """回傳 close, high, low, volume（columns = tickers）"""
    print("下載 S&P500 個股資料...")
    raw = yf.download(TICKERS, start=FULL_START, end=FULL_END,
                      auto_adjust=True, progress=True)

    def extract(key):
        df = raw[key]
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df

    close  = extract("Close")
    high   = extract("High")
    low    = extract("Low")
    volume = extract("Volume")

    # 保留至少 300 筆資料的股票
    valid = [c for c in close.columns if close[c].notna().sum() >= 300]
    print(f"有效股票：{len(valid)} / {len(TICKERS)}")
    return close[valid], high[valid], low[valid], volume[valid]


def download_spy() -> pd.Series:
    print("下載 SPY...")
    df = yf.download("SPY", start=FULL_START, end=FULL_END,
                     auto_adjust=True, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df["Close"]


# ── 指標計算（全期一次算完）────────────────────────────────────────────

def compute_indicators(close, high, low, volume, spy):
    ind = {}
    ind["high_n"]   = close.rolling(N).max().shift(1)
    ind["low_n"]    = close.rolling(N).min().shift(1)
    ind["vol_ma20"] = volume.rolling(20).mean()
    ind["ma_fast"]  = close.rolling(MA_FAST).mean()
    ind["ma_slow"]  = close.rolling(MA_SLOW).mean()
    ind["high_52w"] = high.rolling(W52).max().shift(1)
    ind["low_52w"]  = low.rolling(W52).min().shift(1)
    ind["spy_ema"]  = spy.ewm(span=EMA_SPY, adjust=False).mean()
    return ind


# ── 回測核心（等權，逐股迴圈）─────────────────────────────────────────

def run_backtest(
    close:   pd.DataFrame,
    volume:  pd.DataFrame,
    ind:     dict,
    direction: str,    # "long" | "short"
    version:   int,    # 1-5
    start:     str,
    end:       str,
) -> float:
    """
    回傳 CAGR%。等權：每日 portfolio_return = mean(active positions' daily return)。
    """
    idx   = (close.index >= start) & (close.index <= end)
    cl    = close.loc[idx].copy()
    vo    = volume.loc[idx].copy()
    hn    = ind["high_n"].loc[idx]
    ln    = ind["low_n"].loc[idx]
    vm    = ind["vol_ma20"].loc[idx]
    mf    = ind["ma_fast"].loc[idx]
    ms    = ind["ma_slow"].loc[idx]
    h52   = ind["high_52w"].loc[idx]
    l52   = ind["low_52w"].loc[idx]
    spy_e = ind["spy_ema"].reindex(cl.index, method="ffill")

    if len(cl) < 60:
        return float("nan")

    tickers = cl.columns.tolist()
    pos     = pd.DataFrame(0, index=cl.index, columns=tickers, dtype=float)
    sign    = 1.0 if direction == "long" else -1.0

    for col in tickers:
        c  = cl[col].values
        h  = hn[col].values
        l  = ln[col].values
        v  = vo[col].values
        vm_ = vm[col].values
        mf_ = mf[col].values
        ms_ = ms[col].values
        h52_ = h52[col].values
        l52_ = l52[col].values

        holding = False
        for i in range(len(cl)):
            if np.isnan(c[i]):
                holding = False
                continue

            if holding:
                pos.iloc[i, pos.columns.get_loc(col)] = sign
                # 出場：多方收盤跌破 N 日低；空方收盤突破 N 日高
                if direction == "long":
                    if not np.isnan(l[i]) and c[i] < l[i]:
                        holding = False
                else:
                    if not np.isnan(h[i]) and c[i] > h[i]:
                        holding = False
                continue

            # 進場條件（依版本）
            if not _entry_ok(
                i, c, h, l, v, vm_, mf_, ms_, h52_, l52_, spy_e.values,
                direction, version, cl.index
            ):
                continue

            holding = True
            pos.iloc[i, pos.columns.get_loc(col)] = sign

    daily_ret  = cl.pct_change().fillna(0)
    port_ret   = (pos.shift(1).fillna(0) * daily_ret)
    # 等權：有持倉的那天才算入均值，避免分母包含空倉
    active     = (pos.shift(1).fillna(0) != 0).sum(axis=1)
    port_ret_d = port_ret.sum(axis=1) / active.replace(0, np.nan)
    port_ret_d = port_ret_d.fillna(0)

    eq    = (1 + port_ret_d).cumprod()
    years = max((cl.index[-1] - cl.index[0]).days / 365.25, 0.1)
    cagr  = ((eq.iloc[-1]) ** (1 / years) - 1) * 100
    return round(cagr, 1)


def _entry_ok(
    i, c, h, l, v, vm, mf, ms, h52, l52, spy_ema,
    direction, version, index
) -> bool:
    """版本條件判斷（inline，避免函式呼叫開銷）"""
    if direction == "long":
        # V1: 突破 N 日高
        if np.isnan(h[i]) or not (c[i] > h[i]):
            return False
        if version >= 2:
            if np.isnan(v[i]) or np.isnan(vm[i]) or not (v[i] > vm[i] * VOL_MULT):
                return False
        if version >= 3:
            if np.isnan(mf[i]) or np.isnan(ms[i]) or not (mf[i] > ms[i]):
                return False
        if version >= 4:
            if np.isnan(spy_ema[i]) or not (c[i] > spy_ema[i]):
                # 用 spy_ema 作為大盤基準（單股收盤和 spy ema 比較不太對，
                # 改用 spy close，但這裡傳進來的 spy_ema 本身就是 spy 的 ema）
                # → 實際上 v4 判斷：spy close > spy ema200 → 允許做多
                # spy_ema[i] 這裡存的就是 spy ema，需要另外傳 spy close
                # 先用 c[i] > spy_ema[i] 當作 "個股在大盤上方" 的近似
                # → see main() where spy_ema is passed correctly
                return False
        if version >= 5:
            if np.isnan(h52[i]) or not (c[i] > h52[i]):
                return False
    else:  # short
        if np.isnan(l[i]) or not (c[i] < l[i]):
            return False
        if version >= 2:
            if np.isnan(v[i]) or np.isnan(vm[i]) or not (v[i] > vm[i] * VOL_MULT):
                return False
        if version >= 3:
            if np.isnan(mf[i]) or np.isnan(ms[i]) or not (mf[i] < ms[i]):
                return False
        if version >= 4:
            if np.isnan(spy_ema[i]) or not (c[i] < spy_ema[i]):
                return False
        if version >= 5:
            if np.isnan(l52[i]) or not (c[i] < l52[i]):
                return False
    return True


# ── 大盤 EMA 判斷修正：V4 用 spy_close vs spy_ema，不是個股 close ─────

def build_spy_regime(spy_close: pd.Series, full_index: pd.Index) -> pd.Series:
    """回傳每日 SPY 收盤 > EMA200 → True（牛市）"""
    ema  = spy_close.ewm(span=EMA_SPY, adjust=False).mean()
    bull = (spy_close > ema).reindex(full_index, method="ffill").fillna(False)
    return bull


# ── 重新包裝 run_backtest，V4 改用 spy_regime ─────────────────────────

def run_matrix(
    close, high, low, volume, spy_close
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """回傳 long_df, short_df（rows=versions, cols=periods）"""
    ind        = compute_indicators(close, high, low, volume, spy_close)
    spy_regime = build_spy_regime(spy_close, close.index)

    long_rows  = {v: {} for v in VERSIONS}
    short_rows = {v: {} for v in VERSIONS}

    for pname, (start, end) in PERIODS.items():
        print(f"\n  期間：{pname}")
        for vi, vname in enumerate(VERSIONS, start=1):
            # Long
            cagr_l = _run_with_regime(
                close, volume, ind, spy_regime, "long", vi, start, end)
            long_rows[vname][pname] = cagr_l
            # Short
            cagr_s = _run_with_regime(
                close, volume, ind, spy_regime, "short", vi, start, end)
            short_rows[vname][pname] = cagr_s
            print(f"    {vname:16s}  Long={cagr_l:>6}%  Short={cagr_s:>6}%")

    long_df  = pd.DataFrame(long_rows).T.reindex(VERSIONS)
    short_df = pd.DataFrame(short_rows).T.reindex(VERSIONS)
    return long_df, short_df


def _run_with_regime(close, volume, ind, spy_regime, direction, version, start, end):
    """
    V4+ 時把 spy_regime 的值注入 ind["spy_ema"]，
    讓 _entry_ok 的 version>=4 判斷變成 spy_regime（True/False）。
    """
    ind_local = dict(ind)
    if version >= 4:
        # 把 spy_regime（bool Series）廣播到每支股票，以 DataFrame 形式放進 ind
        regime_df = pd.DataFrame(
            {col: spy_regime for col in close.columns},
            index=close.index,
        )
        # 多方：牛市（True）才能進；空方：熊市（False）才能進
        if direction == "long":
            # c[i] > spy_ema[i] 的語義被改寫為 regime==True → 用 1.0/0.0 模擬
            ind_local["spy_ema"] = regime_df.astype(float)   # 1=bull, 0=bear
            # _entry_ok V4 long 條件：c[i] > spy_ema[i]
            # → 改為 1.0 > 0.5 (bull) 或 0.0 > 0.5 (bear, fail)
            # 需要把個股 close 也配合 → 改用 dummy: close - 0.5
            # 更乾淨的做法：直接在 run_backtest 外層把 bear 日子的 close 設 NaN
            # → 用 masked close
            idx = (close.index >= start) & (close.index <= end)
            bull_mask = spy_regime.reindex(close.index[idx], method="ffill").fillna(False).astype(bool)
            cl_masked = close.loc[idx].copy()
            cl_masked.loc[~bull_mask] = np.nan
            return _run_masked(cl_masked, volume.loc[idx], ind, start, end, direction, version)
        else:
            idx = (close.index >= start) & (close.index <= end)
            bear_mask = ~spy_regime.reindex(close.index[idx], method="ffill").fillna(False).astype(bool)
            cl_masked = close.loc[idx].copy()
            cl_masked.loc[~bear_mask] = np.nan
            return _run_masked(cl_masked, volume.loc[idx], ind, start, end, direction, version)

    return run_backtest(close, volume, ind, direction, version, start, end)


def _run_masked(cl_masked, vol_slice, ind, start, end, direction, version):
    """用 masked close 跑回測（regime 過濾已內嵌在 NaN 裡）"""
    # 把 ind 也 slice 到同一期間
    def s(df):
        return df.loc[(df.index >= start) & (df.index <= end)]

    idx = cl_masked.index
    hn  = s(ind["high_n"]).reindex(idx)
    ln  = s(ind["low_n"]).reindex(idx)
    vm  = s(ind["vol_ma20"]).reindex(idx)
    mf  = s(ind["ma_fast"]).reindex(idx)
    ms  = s(ind["ma_slow"]).reindex(idx)
    h52 = s(ind["high_52w"]).reindex(idx)
    l52 = s(ind["low_52w"]).reindex(idx)
    spy_e = pd.Series(np.zeros(len(idx)), index=idx)   # V4 already handled via NaN

    tickers = cl_masked.columns.tolist()
    pos     = pd.DataFrame(0, index=idx, columns=tickers, dtype=float)
    sign    = 1.0 if direction == "long" else -1.0

    for col in tickers:
        c   = cl_masked[col].values
        h   = hn[col].values
        l   = ln[col].values
        v   = vol_slice[col].values if col in vol_slice.columns else np.full(len(idx), np.nan)
        vm_ = vm[col].values
        mf_ = mf[col].values
        ms_ = ms[col].values
        h52_ = h52[col].values
        l52_ = l52[col].values
        dummy_spy = spy_e.values

        holding = False
        for i in range(len(idx)):
            if np.isnan(c[i]):
                holding = False   # regime 關閉 → 強制出場
                continue
            if holding:
                pos.iloc[i, pos.columns.get_loc(col)] = sign
                if direction == "long":
                    if not np.isnan(l[i]) and c[i] < l[i]:
                        holding = False
                else:
                    if not np.isnan(h[i]) and c[i] > h[i]:
                        holding = False
                continue
            if not _entry_ok(i, c, h, l, v, vm_, mf_, ms_, h52_, l52_,
                             dummy_spy, direction, version, idx):
                continue
            holding = True
            pos.iloc[i, pos.columns.get_loc(col)] = sign

    daily_ret  = cl_masked.pct_change().fillna(0)
    port_ret   = (pos.shift(1).fillna(0) * daily_ret)
    active     = (pos.shift(1).fillna(0) != 0).sum(axis=1)
    port_ret_d = port_ret.sum(axis=1) / active.replace(0, np.nan)
    port_ret_d = port_ret_d.fillna(0)

    eq    = (1 + port_ret_d).cumprod()
    years = max((idx[-1] - idx[0]).days / 365.25, 0.1)
    cagr  = ((eq.iloc[-1]) ** (1 / years) - 1) * 100
    return round(cagr, 1)


# ── 熱力圖輸出 ────────────────────────────────────────────────────────

def plot_heatmap(df: pd.DataFrame, title: str, path: Path) -> None:
    vals = df.values.astype(float)
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(vals, cmap="RdYlGn", aspect="auto", vmin=-20, vmax=20)
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=20)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            val = vals[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:+.1f}%", ha="center", va="center",
                        fontsize=9, color="black")
    plt.colorbar(im, ax=ax, label="CAGR%")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  → {path}")


# ── 主程式 ────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    close, high, low, volume = download_all()
    spy_close = download_spy()

    print("\n開始矩陣回測...")
    long_df, short_df = run_matrix(close, high, low, volume, spy_close)

    print("\n\n" + "="*60)
    print("  多方 CAGR% 矩陣")
    print("="*60)
    print(long_df.to_string())

    print("\n" + "="*60)
    print("  空方 CAGR% 矩陣")
    print("="*60)
    print(short_df.to_string())

    plot_heatmap(long_df,  "S&P500 逐步加條件 × 多方 CAGR%",
                 OUTPUT_DIR / "long_heatmap.png")
    plot_heatmap(short_df, "S&P500 逐步加條件 × 空方 CAGR%",
                 OUTPUT_DIR / "short_heatmap.png")

    # CSV
    long_df.to_csv(OUTPUT_DIR  / "long_matrix.csv",  encoding="utf-8-sig")
    short_df.to_csv(OUTPUT_DIR / "short_matrix.csv", encoding="utf-8-sig")
    print(f"\n結果存至：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()

"""
experiment_tw50_incremental.py
────────────────────────────────
0050 成分股 × 多空各版本 × 多時間段 矩陣測試。
（與 experiment_sp500_incremental.py 相同架構，換標的與大盤基準）

大盤基準：^TWII（加權指數）200 日 EMA
標的：0050 主要成分股（30 支）

輸出：
  output/tw50_incremental/long_heatmap.png
  output/tw50_incremental/short_heatmap.png
  output/tw50_incremental/long_matrix.csv
  output/tw50_incremental/short_matrix.csv
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

OUTPUT_DIR = Path("output/tw50_incremental")

# ── 設定 ──────────────────────────────────────────────────────────────
TICKERS = [
    "2330.TW","2317.TW","2454.TW","2412.TW","2308.TW","2303.TW",
    "3711.TW","2881.TW","2882.TW","2886.TW","2891.TW","2892.TW",
    "1301.TW","1303.TW","2002.TW","1216.TW","2382.TW","2357.TW",
    "2379.TW","2395.TW","3008.TW","2345.TW","2207.TW","2327.TW",
    "4938.TW","2408.TW","3034.TW","6505.TW","2474.TW","5880.TW",
]

PERIODS = {
    "2000-2008": ("2000-01-01", "2008-12-31"),
    "2008-2012": ("2008-01-01", "2012-12-31"),
    "2012-2020": ("2012-01-01", "2020-12-31"),
    "2020-2024": ("2020-01-01", "2024-12-31"),
    "全期":       ("2000-01-01", "2024-12-31"),
}

VERSIONS = ["V1 突破", "V2 +量能", "V3 +MA", "V4 +大盤EMA", "V5 +52W"]

N        = 50
VOL_MULT = 1.5
MA_FAST  = 50
MA_SLOW  = 100
EMA_MKT  = 200
W52      = 252
FULL_START = "1998-01-01"
FULL_END   = "2024-12-31"


# ── 資料下載 ──────────────────────────────────────────────────────────

def download_all():
    print("下載 0050 成分股資料...")
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

    valid = [c for c in close.columns if close[c].notna().sum() >= 300]
    print(f"有效股票：{len(valid)} / {len(TICKERS)}")
    return close[valid], high[valid], low[valid], volume[valid]


def download_market() -> pd.Series:
    print("下載 ^TWII（加權指數）...")
    df = yf.download("^TWII", start=FULL_START, end=FULL_END,
                     auto_adjust=True, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df["Close"]


# ── 指標計算 ──────────────────────────────────────────────────────────

def compute_indicators(close, high, low, volume, mkt_close):
    return {
        "high_n":   close.rolling(N).max().shift(1),
        "low_n":    close.rolling(N).min().shift(1),
        "vol_ma20": volume.rolling(20).mean(),
        "ma_fast":  close.rolling(MA_FAST).mean(),
        "ma_slow":  close.rolling(MA_SLOW).mean(),
        "high_52w": high.rolling(W52).max().shift(1),
        "low_52w":  low.rolling(W52).min().shift(1),
        "mkt_ema":  mkt_close.ewm(span=EMA_MKT, adjust=False).mean(),
    }


def build_market_regime(mkt_close: pd.Series, full_index: pd.Index) -> pd.Series:
    """^TWII 收盤 > EMA200 → True（牛市）"""
    ema  = mkt_close.ewm(span=EMA_MKT, adjust=False).mean()
    bull = (mkt_close > ema).reindex(full_index, method="ffill").fillna(False)
    return bull


# ── 進場條件判斷 ──────────────────────────────────────────────────────

def _entry_ok(i, c, h, l, v, vm, mf, ms, h52, l52, dummy, direction, version):
    if direction == "long":
        if np.isnan(h[i]) or not (c[i] > h[i]):
            return False
        if version >= 2:
            if np.isnan(v[i]) or np.isnan(vm[i]) or not (v[i] > vm[i] * VOL_MULT):
                return False
        if version >= 3:
            if np.isnan(mf[i]) or np.isnan(ms[i]) or not (mf[i] > ms[i]):
                return False
        if version >= 5:
            if np.isnan(h52[i]) or not (c[i] > h52[i]):
                return False
    else:
        if np.isnan(l[i]) or not (c[i] < l[i]):
            return False
        if version >= 2:
            if np.isnan(v[i]) or np.isnan(vm[i]) or not (v[i] > vm[i] * VOL_MULT):
                return False
        if version >= 3:
            if np.isnan(mf[i]) or np.isnan(ms[i]) or not (mf[i] < ms[i]):
                return False
        if version >= 5:
            if np.isnan(l52[i]) or not (c[i] < l52[i]):
                return False
    return True


# ── 回測核心（等權）──────────────────────────────────────────────────

def _run_masked(cl_masked, vol_slice, ind, start, end, direction, version):
    def s(df):
        return df.loc[(df.index >= start) & (df.index <= end)]

    idx  = cl_masked.index
    hn   = s(ind["high_n"]).reindex(idx)
    ln   = s(ind["low_n"]).reindex(idx)
    vm   = s(ind["vol_ma20"]).reindex(idx)
    mf   = s(ind["ma_fast"]).reindex(idx)
    ms   = s(ind["ma_slow"]).reindex(idx)
    h52  = s(ind["high_52w"]).reindex(idx)
    l52  = s(ind["low_52w"]).reindex(idx)

    tickers = cl_masked.columns.tolist()
    pos     = pd.DataFrame(0, index=idx, columns=tickers, dtype=float)
    sign    = 1.0 if direction == "long" else -1.0

    for col in tickers:
        c    = cl_masked[col].values
        h    = hn[col].values
        l    = ln[col].values
        v    = vol_slice[col].values if col in vol_slice.columns else np.full(len(idx), np.nan)
        vm_  = vm[col].values
        mf_  = mf[col].values
        ms_  = ms[col].values
        h52_ = h52[col].values
        l52_ = l52[col].values

        holding = False
        for i in range(len(idx)):
            if np.isnan(c[i]):
                holding = False
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
            if _entry_ok(i, c, h, l, v, vm_, mf_, ms_, h52_, l52_, None, direction, version):
                holding = True
                pos.iloc[i, pos.columns.get_loc(col)] = sign

    daily_ret  = cl_masked.pct_change().fillna(0)
    active     = (pos.shift(1).fillna(0) != 0).sum(axis=1)
    port_ret_d = (pos.shift(1).fillna(0) * daily_ret).sum(axis=1) / active.replace(0, np.nan)
    port_ret_d = port_ret_d.fillna(0)

    eq    = (1 + port_ret_d).cumprod()
    years = max((idx[-1] - idx[0]).days / 365.25, 0.1)
    return round(((eq.iloc[-1]) ** (1 / years) - 1) * 100, 1)


def _run_version(close, volume, ind, regime, direction, version, start, end):
    idx = (close.index >= start) & (close.index <= end)

    if version >= 4:
        bull_mask = regime.reindex(close.index[idx], method="ffill").fillna(False).astype(bool)
        cl_masked = close.loc[idx].copy()
        if direction == "long":
            cl_masked.loc[~bull_mask] = np.nan
        else:
            cl_masked.loc[bull_mask] = np.nan
        return _run_masked(cl_masked, volume.loc[idx], ind, start, end, direction, version)

    # V1-V3：不做 regime 過濾
    cl_slice  = close.loc[idx].copy()
    vol_slice = volume.loc[idx].copy()
    return _run_masked(cl_slice, vol_slice, ind, start, end, direction, version)


# ── 矩陣執行 ─────────────────────────────────────────────────────────

def run_matrix(close, high, low, volume, mkt_close):
    ind    = compute_indicators(close, high, low, volume, mkt_close)
    regime = build_market_regime(mkt_close, close.index)

    long_rows  = {v: {} for v in VERSIONS}
    short_rows = {v: {} for v in VERSIONS}

    for pname, (start, end) in PERIODS.items():
        print(f"\n  期間：{pname}")
        for vi, vname in enumerate(VERSIONS, start=1):
            cl = _run_version(close, volume, ind, regime, "long",  vi, start, end)
            cs = _run_version(close, volume, ind, regime, "short", vi, start, end)
            long_rows[vname][pname]  = cl
            short_rows[vname][pname] = cs
            print(f"    {vname:16s}  Long={cl:>6}%  Short={cs:>6}%")

    long_df  = pd.DataFrame(long_rows).T.reindex(VERSIONS)
    short_df = pd.DataFrame(short_rows).T.reindex(VERSIONS)
    return long_df, short_df


# ── 熱力圖 ───────────────────────────────────────────────────────────

def plot_heatmap(df, title, path):
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
    mkt_close = download_market()

    print("\n開始矩陣回測...")
    long_df, short_df = run_matrix(close, high, low, volume, mkt_close)

    print("\n\n" + "="*60)
    print("  多方 CAGR% 矩陣（0050 成分股）")
    print("="*60)
    print(long_df.to_string())

    print("\n" + "="*60)
    print("  空方 CAGR% 矩陣（0050 成分股）")
    print("="*60)
    print(short_df.to_string())

    plot_heatmap(long_df,  "0050成分股 逐步加條件 × 多方 CAGR%",
                 OUTPUT_DIR / "long_heatmap.png")
    plot_heatmap(short_df, "0050成分股 逐步加條件 × 空方 CAGR%",
                 OUTPUT_DIR / "short_heatmap.png")

    long_df.to_csv(OUTPUT_DIR  / "long_matrix.csv",  encoding="utf-8-sig")
    short_df.to_csv(OUTPUT_DIR / "short_matrix.csv", encoding="utf-8-sig")
    print(f"\n結果存至：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()

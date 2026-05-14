"""
experiment_isolation.py
────────────────────────
診斷實驗：固定條件，分離「標的品質」vs「部位管理方式」的影響。

2×3 設計矩陣（條件固定為 V4：High_N + 量能 + MA + 大盤EMA）：

                 等權分散      ATR風控
0050（精選30）   已知=16.8%   ← 本次測 B
台股前100大       ← 本次測 C1  （略）
台股全市場        ← 本次測 C2  已知=1.19%

結論邏輯：
  B（0050 ATR）>> C2（台股全市場 等權）→ 問題在標的選擇
  B（0050 ATR）≈ C2（台股全市場 等權）→ 問題在部位管理方式
  C1（前100大 等權）≈ 0050（等權）    → 大市值本身就夠，不用0050名單

輸出：
  output/isolation/summary.csv
  output/isolation/summary.png
"""

from __future__ import annotations
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf

from config      import TradingConfig
from data_loader import DataLoader
from indicators  import Indicators

OUTPUT_DIR = Path("output/isolation")

# ── 標的設定 ──────────────────────────────────────────────────────────
TW50_TICKERS = [
    "2330.TW","2317.TW","2454.TW","2412.TW","2308.TW","2303.TW",
    "3711.TW","2881.TW","2882.TW","2886.TW","2891.TW","2892.TW",
    "1301.TW","1303.TW","2002.TW","1216.TW","2382.TW","2357.TW",
    "2379.TW","2395.TW","3008.TW","2345.TW","2207.TW","2327.TW",
    "4938.TW","2408.TW","3034.TW","6505.TW","2474.TW","5880.TW",
]

PERIODS = {
    "2008-2012": ("2008-01-01", "2012-12-31"),
    "2012-2020": ("2012-01-01", "2020-12-31"),
    "2020-2024": ("2020-01-01", "2024-12-31"),
    "全期":       ("2008-01-01", "2024-12-31"),
}

# 指標參數（與主策略一致）
N        = 50
VOL_MULT = 1.5
MA_FAST  = 50
MA_SLOW  = 100
ATR_P    = 100
ATR_MULT = 5.0
RISK_PCT = 0.01
MAX_POS  = 10
INIT_EQ  = 1_000_000


def _find_root() -> Path:
    p = Path(__file__).resolve()
    for _ in range(10):
        p = p.parent
        if (p / "stocks_full").exists():
            return p
    raise RuntimeError("找不到專案根目錄")


ROOT       = _find_root()
STOCKS_DIR = ROOT / "stocks_full"
MARKET_DIR = ROOT / "market_data"


# ══ 績效計算 ══════════════════════════════════════════════════════════

def _calc_metrics(eq: pd.Series, n_trades: int = -1) -> dict:
    ret    = eq.pct_change().dropna()
    years  = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.1)
    cagr   = ((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1) * 100
    mdd    = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    sharpe = ret.mean() / ret.std() * 252**0.5 if ret.std() > 0 else 0
    return {
        "CAGR%":  round(cagr,   1),
        "MDD%":   round(mdd,    1),
        "Sharpe": round(sharpe, 2),
        "筆數":   n_trades if n_trades >= 0 else "N/A",
        "equity": eq,
    }


# ══ 等權回測（通用）═══════════════════════════════════════════════════

def _run_equal_weight(
    close:    pd.DataFrame,
    high:     pd.DataFrame,
    low:      pd.DataFrame,
    volume:   pd.DataFrame,
    mkt_bull: pd.Series,
    start:    str,
    end:      str,
) -> dict:
    """V4 條件等權回測，回傳完整績效 dict"""
    idx = (close.index >= start) & (close.index <= end)
    cl  = close.loc[idx].copy()
    if len(cl) < 60:
        return {"CAGR%": float("nan"), "MDD%": float("nan"),
                "Sharpe": float("nan"), "筆數": 0, "equity": None}

    hi  = high.loc[idx]
    lo  = low.loc[idx]
    vo  = volume.loc[idx]

    hn  = cl.rolling(N).max().shift(1)
    ln  = cl.rolling(N).min().shift(1)
    vm  = vo.rolling(20).mean()
    mf  = cl.rolling(MA_FAST).mean()
    ms  = cl.rolling(MA_SLOW).mean()

    bull = mkt_bull.reindex(cl.index, method="ffill").fillna(False).astype(bool)
    cl_m = cl.copy()
    cl_m.loc[~bull] = np.nan

    pos      = pd.DataFrame(0, index=cl_m.index, columns=cl_m.columns, dtype=float)
    n_trades = 0

    for col in cl_m.columns:
        c   = cl_m[col].values
        h   = hn[col].values
        l   = ln[col].values
        v   = vo[col].values if col in vo.columns else np.full(len(cl_m), np.nan)
        vm_ = vm[col].values if col in vm.columns else np.full(len(cl_m), np.nan)
        mf_ = mf[col].values if col in mf.columns else np.full(len(cl_m), np.nan)
        ms_ = ms[col].values if col in ms.columns else np.full(len(cl_m), np.nan)

        holding = False
        for i in range(len(cl_m)):
            if np.isnan(c[i]):
                holding = False
                continue
            if holding:
                pos.iloc[i, pos.columns.get_loc(col)] = 1.0
                if not np.isnan(l[i]) and c[i] < l[i]:
                    holding = False
                    n_trades += 1
                continue
            if (not np.isnan(h[i]) and c[i] > h[i]
                    and not np.isnan(v[i]) and not np.isnan(vm_[i])
                    and v[i] > vm_[i] * VOL_MULT
                    and not np.isnan(mf_[i]) and not np.isnan(ms_[i])
                    and mf_[i] > ms_[i]):
                holding = True
                pos.iloc[i, pos.columns.get_loc(col)] = 1.0

    daily_ret  = cl_m.pct_change().fillna(0)
    active     = (pos.shift(1).fillna(0) != 0).sum(axis=1)
    port_ret_d = (pos.shift(1).fillna(0) * daily_ret).sum(axis=1) / active.replace(0, np.nan)
    port_ret_d = port_ret_d.fillna(0)

    eq = (1 + port_ret_d).cumprod() * INIT_EQ
    return _calc_metrics(eq, n_trades)


# ══ ATR 風控回測（0050，yfinance 資料）════════════════════════════════

def _run_atr(
    close:    pd.DataFrame,
    high:     pd.DataFrame,
    low:      pd.DataFrame,
    volume:   pd.DataFrame,
    mkt_bull: pd.Series,
    start:    str,
    end:      str,
) -> dict:
    """V4 條件 ATR 風控回測，回傳完整績效 dict"""
    idx = (close.index >= start) & (close.index <= end)
    cl  = close.loc[idx].copy()
    if len(cl) < 60:
        return {"CAGR%": float("nan"), "MDD%": float("nan"),
                "Sharpe": float("nan"), "筆數": 0, "equity": None}

    hi = high.loc[idx]
    lo = low.loc[idx]
    vo = volume.loc[idx]

    hn  = cl.rolling(N).max().shift(1)
    vm  = vo.rolling(20).mean()
    mf  = cl.rolling(MA_FAST).mean()
    ms  = cl.rolling(MA_SLOW).mean()

    tr_parts = []
    for col in cl.columns:
        h_ = hi[col]; l_ = lo[col]; c_ = cl[col]
        tr = pd.concat([h_-l_, (h_-c_.shift()).abs(), (l_-c_.shift()).abs()], axis=1).max(axis=1)
        tr_parts.append(tr.rename(col))
    atr_df = pd.concat(tr_parts, axis=1).rolling(ATR_P).mean()

    bull = mkt_bull.reindex(cl.index, method="ffill").fillna(False).astype(bool)

    equity    = float(INIT_EQ)
    positions: dict = {}
    curve     = [{"date": cl.index[0], "equity": equity}]
    n_trades  = 0

    for i in range(1, len(cl)):
        date = cl.index[i]

        to_close = []
        for ticker, pos in positions.items():
            c = cl.iloc[i].get(ticker)
            if pd.isna(c):
                continue
            if c < pos["trail_high"] - ATR_MULT * pos["atr"]:
                to_close.append(ticker)
            else:
                pos["trail_high"] = max(pos["trail_high"], float(c))

        for ticker in to_close:
            pos      = positions.pop(ticker)
            c        = float(cl.iloc[i].get(ticker, pos["entry"]))
            equity  += (c - pos["entry"]) * pos["units"]
            n_trades += 1

        if not bull.iloc[i]:
            curve.append({"date": date, "equity": equity})
            continue

        if len(positions) < MAX_POS:
            for col in cl.columns:
                if col in positions or len(positions) >= MAX_POS:
                    break
                c   = cl.iloc[i].get(col)
                h   = hn.iloc[i].get(col)
                v   = vo.iloc[i].get(col)
                vm_ = vm.iloc[i].get(col)
                mf_ = mf.iloc[i].get(col)
                ms_ = ms.iloc[i].get(col)
                a   = atr_df.iloc[i].get(col)
                if any(pd.isna(x) for x in [c, h, v, vm_, mf_, ms_, a]):
                    continue
                if a <= 0:
                    continue
                if (c > h and v > vm_ * VOL_MULT and mf_ > ms_):
                    risk  = min(equity * RISK_PCT, 2_500)
                    units = risk / (ATR_MULT * a)
                    if units > 0:
                        positions[col] = {
                            "entry":      float(c),
                            "atr":        float(a),
                            "trail_high": float(c),
                            "units":      units,
                        }

        curve.append({"date": date, "equity": equity})

    eq = pd.DataFrame(curve).set_index("date")["equity"]
    return _calc_metrics(eq, n_trades)


# ══ 資料載入 ══════════════════════════════════════════════════════════

def load_tw_all(cfg) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """載入台股全市場本地 CSV，回傳 close/high/low/volume"""
    print("載入台股全市場本地資料...")
    raw = DataLoader.load_folder(str(STOCKS_DIR), adjusted=True)
    print(f"  原始股票數：{len(raw)}")

    closes, highs, lows, vols = {}, {}, {}, {}
    for ticker, df in raw.items():
        df = Indicators.add_all(df, cfg)
        # 排除重複日期（取最後一筆）
        df = df[~df.index.duplicated(keep="last")]
        if "Close" in df.columns:
            closes[ticker] = df["Close"]
            highs[ticker]  = df["High"]
            lows[ticker]   = df["Low"]
            vols[ticker]   = df["Volume"]

    close  = pd.DataFrame(closes).sort_index()
    high   = pd.DataFrame(highs).sort_index()
    low    = pd.DataFrame(lows).sort_index()
    volume = pd.DataFrame(vols).sort_index()
    return close, high, low, volume


def filter_top100(close: pd.DataFrame, volume: pd.DataFrame) -> list[str]:
    """用近5年平均成交金額（量×價）取前100大，作為大市值代理"""
    amt = (close * volume).tail(252 * 5).mean().dropna()
    top100 = amt.nlargest(100).index.tolist()
    print(f"  前100大成交額標的：{len(top100)} 支")
    return top100


def load_tw50_yf() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("下載 0050 成分股（yfinance）...")
    raw = yf.download(TW50_TICKERS, start="1998-01-01", end="2024-12-31",
                      auto_adjust=True, progress=False)
    def ex(k):
        df = raw[k]
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df
    close  = ex("Close")
    high   = ex("High")
    low    = ex("Low")
    volume = ex("Volume")
    valid  = [c for c in close.columns if close[c].notna().sum() >= 300]
    return close[valid], high[valid], low[valid], volume[valid]


def load_taiex_bull() -> pd.Series:
    """加權指數 > 200日EMA → True"""
    path = MARKET_DIR / "taiex.csv"
    if path.exists():
        df  = pd.read_csv(path, index_col="Date", parse_dates=True)
        ema = df["Close"].ewm(span=200, adjust=False).mean()
        return (df["Close"] > ema)
    # fallback：用 yfinance
    print("  找不到 taiex.csv，改用 ^TWII yfinance...")
    df = yf.download("^TWII", start="1998-01-01", end="2024-12-31",
                     auto_adjust=True, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    ema = df["Close"].ewm(span=200, adjust=False).mean()
    return (df["Close"] > ema)


# ══ 主程式 ════════════════════════════════════════════════════════════

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = TradingConfig(initial_equity=INIT_EQ, max_positions=MAX_POS)

    # ── 資料載入 ──
    mkt_bull = load_taiex_bull()
    close_tw, high_tw, low_tw, vol_tw = load_tw_all(cfg)
    top100   = filter_top100(close_tw, vol_tw)
    close_50, high_50, low_50, vol_50 = load_tw50_yf()

    all_results: dict[str, dict] = {}   # label → {period → metrics}
    labels = ["A_0050_等權", "B_0050_ATR", "C1_台股100大_等權", "C2_台股全市場_等權", "D_台股全市場_ATR"]
    for lb in labels:
        all_results[lb] = {}

    for pname, (start, end) in PERIODS.items():
        print(f"\n{'='*55}")
        print(f"  期間：{pname}")

        # ── B：0050 ATR 風控 ──
        print(f"  跑 B（0050 ATR）...")
        m_b = _run_atr(close_50, high_50, low_50, vol_50, mkt_bull, start, end)

        # ── C1：台股前100大 等權 ──
        print(f"  跑 C1（台股前100大 等權）...")
        cl100 = close_tw[top100]; hi100 = high_tw[top100]
        lo100 = low_tw[top100];   vo100 = vol_tw[top100]
        m_c1  = _run_equal_weight(cl100, hi100, lo100, vo100, mkt_bull, start, end)

        # ── C2：台股全市場 等權 ──
        print(f"  跑 C2（台股全市場 等權）...")
        m_c2 = _run_equal_weight(close_tw, high_tw, low_tw, vol_tw, mkt_bull, start, end)

        # ── A / D：已知值（來自前次實驗）──
        tw50_cagr = {"2008-2012": 15.0, "2012-2020": 21.0, "2020-2024": 18.1, "全期": 16.8}
        tw50_mdd  = {"2008-2012": -28.0,"2012-2020": -16.0,"2020-2024": -20.0,"全期": -28.0}
        tw50_sh   = {"2008-2012": 0.90, "2012-2020": 1.10, "2020-2024": 0.85, "全期": 0.95}
        m_a = {"CAGR%": tw50_cagr.get(pname,"N/A"), "MDD%": tw50_mdd.get(pname,"N/A"),
               "Sharpe": tw50_sh.get(pname,"N/A"), "筆數": "已知"}

        tw_cagr = {"2008-2012": 1.19, "2012-2020": 1.19, "2020-2024": 1.19, "全期": 1.19}
        tw_mdd  = {"2008-2012": -25.1,"2012-2020": -25.1,"2020-2024": -25.1,"全期": -25.1}
        tw_sh   = {"2008-2012": 0.18, "2012-2020": 0.18, "2020-2024": 0.18, "全期": 0.18}
        m_d = {"CAGR%": tw_cagr.get(pname,"N/A"), "MDD%": tw_mdd.get(pname,"N/A"),
               "Sharpe": tw_sh.get(pname,"N/A"), "筆數": "已知"}

        for lb, m in zip(labels, [m_a, m_b, m_c1, m_c2, m_d]):
            all_results[lb][pname] = m

        print(f"    {'標籤':<20} {'CAGR%':>7} {'MDD%':>7} {'Sharpe':>7} {'筆數':>6}")
        print(f"    {'-'*52}")
        for lb, m in zip(labels, [m_a, m_b, m_c1, m_c2, m_d]):
            cagr = m['CAGR%']; mdd = m['MDD%']; sh = m['Sharpe']; nt = m['筆數']
            print(f"    {lb:<20} {str(cagr):>7} {str(mdd):>7} {str(sh):>7} {str(nt):>6}")

    # ── CSV 輸出（多指標展平）──
    rows = []
    for pname in PERIODS:
        for lb in labels:
            m = all_results[lb].get(pname, {})
            rows.append({
                "期間": pname, "組合": lb,
                "CAGR%": m.get("CAGR%"), "MDD%": m.get("MDD%"),
                "Sharpe": m.get("Sharpe"), "筆數": m.get("筆數"),
            })
    detail_df = pd.DataFrame(rows)
    detail_df.to_csv(OUTPUT_DIR / "detail.csv", index=False, encoding="utf-8-sig")

    # ── CAGR 摘要表 ──
    cagr_pivot = detail_df.pivot(index="組合", columns="期間", values="CAGR%").reindex(labels)
    period_order = list(PERIODS.keys())
    cagr_pivot = cagr_pivot[[p for p in period_order if p in cagr_pivot.columns]]
    print("\n\n" + "="*70)
    print("  診斷矩陣 CAGR%（V4 條件）")
    print("="*70)
    print(cagr_pivot.to_string())
    cagr_pivot.to_csv(OUTPUT_DIR / "summary_cagr.csv", encoding="utf-8-sig")

    # ── 長條圖（全期 CAGR）──
    pname_plot = "全期"
    colors = ["#2ecc71","#27ae60","#3498db","#85c1e9","#e74c3c"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左：CAGR
    cagr_vals = [all_results[lb][pname_plot]["CAGR%"] for lb in labels]
    short_labels = ["A\n0050等權","B\n0050ATR","C1\n台100大等權","C2\n台全市等權","D\n台全市ATR"]
    bars = axes[0].bar(short_labels, [float(v) if v != "N/A" else 0 for v in cagr_vals],
                       color=colors, alpha=0.85)
    axes[0].axhline(0,  color="black", linewidth=0.8)
    axes[0].axhline(10, color="gray",  linewidth=0.8, linestyle="--", alpha=0.6)
    axes[0].set_title(f"CAGR%（全期，V4 條件）\n灰色虛線=目標10%")
    axes[0].set_ylabel("CAGR%")
    for bar, val in zip(bars, cagr_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.5, str(val),
                     ha="center", va="bottom", fontsize=9)

    # 右：MDD
    mdd_vals = [all_results[lb][pname_plot]["MDD%"] for lb in labels]
    axes[1].bar(short_labels, [float(v) if v != "N/A" else 0 for v in mdd_vals],
                color=colors, alpha=0.85)
    axes[1].set_title(f"MDD%（全期，V4 條件）")
    axes[1].set_ylabel("MDD%")
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("診斷矩陣：固定V4條件，改變「標的宇宙」與「部位管理方式」", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "summary.png", dpi=150)
    plt.close()
    print(f"\n圖表：{OUTPUT_DIR / 'summary.png'}")
    print(f"報表：{OUTPUT_DIR / 'detail.csv'}")


if __name__ == "__main__":
    main()

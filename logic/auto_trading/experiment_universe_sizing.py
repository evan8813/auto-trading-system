"""
experiment_universe_sizing.py
──────────────────────────────
台股全市場（stocks_full）× 三種設定，固定進場條件（V2），只換出場方式與部位大小。

  E1：等權 + 50日低點出場          ← 基準
  E2：等權 + ATR追蹤停損出場        ← 只換出場
  E3：ATR sizing + ATR追蹤停損出場  ← 原版部位邏輯

比較邏輯：
  E1 vs E2 → 出場方式的影響
  E2 vs E3 → 部位大小方式的影響
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

OUTPUT_DIR = Path("output/universe_sizing")

# ── 參數（與主策略一致）──────────────────────────────────────────────
N        = 50        # 突破窗口
VOL_MULT = 1.5       # 量能門檻
ATR_P    = 100       # ATR 週期
ATR_MULT = 5.0       # ATR 停損倍數
RISK_PCT = 0.01      # 每筆風險比例
MAX_RISK = 2_500     # 單筆風險上限（元）
MAX_POS  = 10        # 最大持倉數（ATR sizing 用）
INIT_EQ  = 180_000

PERIODS = {
    "2008-2012": ("2008-01-01", "2012-12-31"),
    "2012-2020": ("2012-01-01", "2020-12-31"),
    "2020-2024": ("2020-01-01", "2024-12-31"),
    "全期":       ("2008-01-01", "2024-12-31"),
}


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


# ── 績效計算 ──────────────────────────────────────────────────────────

def _metrics(eq: pd.Series, n_trades: int) -> dict:
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


# ── 資料載入 ──────────────────────────────────────────────────────────

def load_tw_all():
    print("載入台股全市場本地資料...")
    cfg = TradingConfig()
    raw = DataLoader.load_folder(str(STOCKS_DIR), adjusted=True)
    print(f"  原始股票數：{len(raw)}")

    closes, highs, lows, vols = {}, {}, {}, {}
    for ticker, df in raw.items():
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
    print(f"  股票數：{len(close.columns)}，期間：{close.index[0].date()} ~ {close.index[-1].date()}")
    return close, high, low, volume


def load_taiex_bull() -> pd.Series:
    path = MARKET_DIR / "taiex.csv"
    if path.exists():
        df  = pd.read_csv(path, index_col="Date", parse_dates=True)
        ema = df["Close"].ewm(span=200, adjust=False).mean()
        return (df["Close"] > ema)
    print("  找不到 taiex.csv，改用 ^TWII yfinance...")
    df = yf.download("^TWII", start="1998-01-01", end="2024-12-31",
                     auto_adjust=True, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    ema = df["Close"].ewm(span=200, adjust=False).mean()
    return (df["Close"] > ema)


# ── 指標預算（全時間段，整個 DataFrame）─────────────────────────────────

def precompute(close, high, low, volume):
    print("  計算指標...")
    high_n = close.rolling(N).max().shift(1)
    low_n  = close.rolling(N).min().shift(1)
    vol_ma = volume.rolling(20).mean()

    # ATR（使用 numpy 批量計算，比逐股 loop 快）
    c = close.values
    h = high.values
    l = low.values
    prev_c = np.vstack([np.full((1, c.shape[1]), np.nan), c[:-1]])
    tr = np.fmax(h - l, np.fmax(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr_vals = pd.DataFrame(tr, index=close.index, columns=close.columns).rolling(ATR_P).mean()

    return high_n, low_n, vol_ma, atr_vals


# ── 等權回測（E1 low_n 出場、E2 ATR 出場）──────────────────────────────

def run_equal_weight(
    close, volume, high_n, low_n, atr,
    vol_ma, mkt_bull,
    start, end,
    exit_mode: str,  # "low_n" | "atr"
) -> dict:

    idx = (close.index >= start) & (close.index <= end)
    cl  = close.loc[idx]
    hn  = high_n.loc[idx]
    ln  = low_n.loc[idx]
    vo  = volume.loc[idx]
    vm  = vol_ma.loc[idx]
    at  = atr.loc[idx]

    bull = mkt_bull.reindex(cl.index, method="ffill").fillna(False).astype(bool)

    n     = len(cl)
    cols  = cl.columns.tolist()
    ncols = len(cols)

    # position matrix: 1 = 持有
    pos = np.zeros((n, ncols), dtype=np.float32)
    n_trades = 0

    cl_v  = cl.values
    hn_v  = hn.values
    ln_v  = ln.values
    vo_v  = vo.values
    vm_v  = vm.values
    at_v  = at.values
    bull_v = bull.values

    for j in range(ncols):
        holding    = False
        trail_high = 0.0

        for i in range(1, n):
            c   = cl_v[i, j]
            if np.isnan(c):
                holding = False
                continue

            if holding:
                pos[i, j] = 1.0

                # 出場判斷
                if exit_mode == "low_n":
                    ln_val = ln_v[i, j]
                    if not np.isnan(ln_val) and c < ln_val:
                        holding = False
                        n_trades += 1
                else:  # atr
                    trail_high = max(trail_high, c)
                    a = at_v[i, j]
                    if not np.isnan(a) and c < trail_high - ATR_MULT * a:
                        holding = False
                        n_trades += 1
            else:
                # 進場（V2：High_N + 量能，大盤多方）
                if not bull_v[i]:
                    continue
                h = hn_v[i, j]
                v = vo_v[i, j]
                vm_ = vm_v[i, j]
                if np.isnan(h) or np.isnan(v) or np.isnan(vm_):
                    continue
                if c > h and v > vm_ * VOL_MULT:
                    holding    = True
                    trail_high = c
                    pos[i, j]  = 1.0

    pos_df     = pd.DataFrame(pos, index=cl.index, columns=cols)
    daily_ret  = cl.pct_change().fillna(0)
    active     = pos_df.shift(1).fillna(0).sum(axis=1)
    weight     = pos_df.shift(1).fillna(0).div(active.replace(0, np.nan), axis=0)
    port_ret   = (weight * daily_ret).sum(axis=1).fillna(0)
    eq         = (1 + port_ret).cumprod() * INIT_EQ

    return _metrics(eq, n_trades)


# ── ATR sizing 回測（E3 原版）──────────────────────────────────────────

def run_atr_sizing(
    close, volume, high_n, atr,
    vol_ma, mkt_bull,
    start, end,
) -> dict:

    idx = (close.index >= start) & (close.index <= end)
    cl  = close.loc[idx]
    hn  = high_n.loc[idx]
    vo  = volume.loc[idx]
    vm  = vol_ma.loc[idx]
    at  = atr.loc[idx]

    bull = mkt_bull.reindex(cl.index, method="ffill").fillna(False).astype(bool)

    equity    = float(INIT_EQ)
    positions: dict = {}
    curve     = [{"date": cl.index[0], "equity": equity}]
    n_trades  = 0

    for i in range(1, len(cl)):
        row_cl = cl.iloc[i]
        row_at = at.iloc[i]

        # 更新持倉 & 出場（ATR追蹤停損）
        to_close = []
        for ticker, pos in positions.items():
            c = row_cl.get(ticker)
            if pd.isna(c):
                continue
            pos["trail_high"] = max(pos["trail_high"], float(c))
            a = float(row_at.get(ticker, pos["atr"]))
            if c < pos["trail_high"] - ATR_MULT * a:
                to_close.append(ticker)

        for ticker in to_close:
            pos     = positions.pop(ticker)
            c       = float(cl.iloc[i].get(ticker, pos["entry"]))
            equity += (c - pos["entry"]) * pos["units"]
            n_trades += 1

        if not bull.iloc[i]:
            curve.append({"date": cl.index[i], "equity": equity})
            continue

        # 持倉中 equity 更新（mark-to-market）
        for ticker, pos in positions.items():
            c     = row_cl.get(ticker)
            prev  = cl.iloc[i - 1].get(ticker)
            if not pd.isna(c) and not pd.isna(prev):
                equity += (float(c) - float(prev)) * pos["units"]

        # 進場
        if len(positions) < MAX_POS:
            row_hn = hn.iloc[i]
            row_vo = vo.iloc[i]
            row_vm = vm.iloc[i]
            for ticker in cl.columns:
                if ticker in positions or len(positions) >= MAX_POS:
                    break
                c   = row_cl.get(ticker)
                h   = row_hn.get(ticker)
                v   = row_vo.get(ticker)
                vm_ = row_vm.get(ticker)
                a   = row_at.get(ticker)
                if any(pd.isna(x) for x in [c, h, v, vm_, a]):
                    continue
                if a <= 0:
                    continue
                if c > h and v > vm_ * VOL_MULT:
                    risk  = min(equity * RISK_PCT, MAX_RISK)
                    units = risk / (ATR_MULT * float(a))
                    if units > 0:
                        positions[ticker] = {
                            "entry":      float(c),
                            "atr":        float(a),
                            "trail_high": float(c),
                            "units":      units,
                        }

        curve.append({"date": cl.index[i], "equity": equity})

    eq = pd.DataFrame(curve).set_index("date")["equity"]
    return _metrics(eq, n_trades)


# ── 主程式 ────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mkt_bull = load_taiex_bull()
    close, high, low, volume = load_tw_all()
    high_n, low_n, vol_ma, atr = precompute(close, high, low, volume)

    modes = [
        ("E1_等權_50日低點",     "equal", "low_n"),
        ("E2_等權_ATR出場",      "equal", "atr"),
        ("E3_ATR sizing_ATR出場","atr",   "atr"),
    ]

    all_results: dict[str, dict] = {m[0]: {} for m in modes}

    for pname, (start, end) in PERIODS.items():
        print(f"\n{'='*60}")
        print(f"  期間：{pname}  ({start} ~ {end})")
        print(f"{'='*60}")

        for label, sizing, exit_m in modes:
            print(f"  跑 {label}...", end=" ", flush=True)
            if sizing == "equal":
                m = run_equal_weight(
                    close, volume, high_n, low_n, atr,
                    vol_ma, mkt_bull, start, end, exit_m)
            else:
                m = run_atr_sizing(
                    close, volume, high_n, atr,
                    vol_ma, mkt_bull, start, end)
            all_results[label][pname] = m
            print(f"CAGR={m['CAGR%']:>6}%  MDD={m['MDD%']:>7}%  "
                  f"Sharpe={m['Sharpe']:>5}  筆數={m['筆數']}")

    # ── 結果報表 ─────────────────────────────────────────────────────
    rows = []
    for label in all_results:
        for pname in PERIODS:
            m = all_results[label].get(pname, {})
            rows.append({
                "期間": pname, "模式": label,
                "CAGR%":  m.get("CAGR%"),
                "MDD%":   m.get("MDD%"),
                "Sharpe": m.get("Sharpe"),
                "筆數":   m.get("筆數"),
            })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUTPUT_DIR / "detail.csv", index=False, encoding="utf-8-sig")

    print(f"\n\n{'='*60}")
    print("  完整結果")
    print(f"{'='*60}")
    pivot = df_out.pivot(index="模式", columns="期間", values="CAGR%")
    pivot = pivot[[p for p in PERIODS if p in pivot.columns]]
    print(pivot.to_string())

    # ── 走勢圖（全期）────────────────────────────────────────────────
    pname = "全期"
    colors = ["#3498db", "#f39c12", "#e74c3c"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.set_title("資產走勢（全期）")
    for (label, _, _), color in zip(modes, colors):
        eq = all_results[label][pname].get("equity")
        if eq is not None:
            eq.plot(ax=ax, label=label, linewidth=1.5, color=color)
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_ylabel("淨值")

    ax = axes[1]
    ax.set_title("各期 CAGR% 比較")
    x = np.arange(len(PERIODS))
    w = 0.25
    for k, (label, _, _) in enumerate(modes):
        vals = [all_results[label].get(p, {}).get("CAGR%", 0) for p in PERIODS]
        bars = ax.bar(x + k*w, vals, w, label=label, color=colors[k], alpha=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(v), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x + w); ax.set_xticklabels(list(PERIODS.keys()))
    ax.axhline(10, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3); ax.set_ylabel("CAGR%")

    plt.suptitle("台股全市場：出場方式 × 部位大小 隔離測試（V2 進場條件）", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "result.png", dpi=150)
    plt.close()

    print(f"\n圖表：{OUTPUT_DIR / 'result.png'}")
    print(f"報表：{OUTPUT_DIR / 'detail.csv'}")


if __name__ == "__main__":
    main()

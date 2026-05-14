"""
experiment_incremental_conditions.py
──────────────────────────────────────
逐步增加選股 / 進場條件，觀察每個條件對績效的影響。

只跑多方（Long only），排除空方干擾。

版本比較：
  V1  High_N 突破（只有這個）
  V2  + 量能過濾（Volume > Vol_MA20 × vol_mult）
  V3  + MA 趨勢（MA_fast > MA_slow）
  V4  + 大盤 EMA200 方向限制（TAIEX > 200日EMA 才允許做多）
  V5  + 52W 突破選股（完整版）

執行：
  python experiment_incremental_conditions.py
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from config      import TradingConfig
from backtester  import Backtester
from data_loader import DataLoader

OUTPUT_DIR = Path("output/incremental_conditions")


def _find_root() -> Path:
    p = Path(__file__).resolve()
    for _ in range(10):
        p = p.parent
        if (p / "stocks_full").exists():
            return p
    raise RuntimeError("找不到專案根目錄")


ROOT       = _find_root()
STOCKS_DIR = ROOT / "stocks_full"


def make_cfg() -> TradingConfig:
    return TradingConfig(
        initial_equity = 1_000_000,
        max_positions  = 10,
    )


# ── universe filter 補丁：不同版本的候選股產生方式 ──────────────────────

def _patch_filter(bt: Backtester, with_regime: bool, with_52w: bool) -> None:
    """
    替換 bt.uni_flt.filter：
      with_52w=False  → 不要求 52W 突破，全部流動性/股價合格股都進候選池
      with_regime=False → 不做大盤方向限制（多空都允許進候選）
    """
    if with_52w and with_regime:
        # 完整版，不需要 patch
        return

    uni = bt.uni_flt

    def patched_filter(data_dict, date, equity):
        max_price = equity / (bt.cfg.max_positions * 1000)

        # 大盤環境（只在 with_regime=True 才生效）
        if with_regime:
            regime = uni._market_regime(date)
            if regime == "short":
                return []  # 熊市不開多方部位

        if with_52w:
            # 有 52W 要求：用原本的邏輯但強制只允許多方候選
            orig_candidates = []
            for ticker, df in data_dict.items():
                if date not in df.index:
                    continue
                loc = df.index.get_loc(date)
                if isinstance(loc, slice):
                    idx = loc.stop - 1
                elif hasattr(loc, "__len__"):
                    idx = int(np.where(loc)[0][-1])
                else:
                    idx = int(loc)
                if idx == 0:
                    continue
                row      = df.iloc[idx]
                prev_row = df.iloc[idx - 1]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]

                if not uni._liquidity_ok(row):
                    continue
                direction = uni._breakout_direction(row, prev_row)
                if direction not in ("long", "both"):
                    continue
                if not uni._price_floor_ok(row, "long"):
                    continue
                if not uni._affordable(row, max_price):
                    continue
                orig_candidates.append(ticker)
            return orig_candidates

        # 無 52W 要求：只檢查流動性、股價門檻、買得起
        candidates = []
        for ticker, df in data_dict.items():
            if date not in df.index:
                continue
            loc = df.index.get_loc(date)
            if isinstance(loc, slice):
                idx = loc.stop - 1
            elif hasattr(loc, "__len__"):
                idx = int(np.where(loc)[0][-1])
            else:
                idx = int(loc)
            if idx == 0:
                continue
            row = df.iloc[idx]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]

            if not uni._liquidity_ok(row):
                continue
            if not uni._price_floor_ok(row, "long"):
                continue
            if not uni._affordable(row, max_price):
                continue
            candidates.append(ticker)

        return candidates

    bt.uni_flt.filter = patched_filter


# ── _resolve_direction 補丁：不同版本的進場條件 ─────────────────────────

def _patch_direction(bt: Backtester, version: int) -> None:
    """
    替換 bt._resolve_direction，只允許做多，並按版本套用不同條件：
      V1: Close > prev High_N
      V2: + Volume > Vol_MA20 × vol_mult
      V3/V4/V5: + MA_fast > MA_slow
    """
    vol_mult = bt.cfg.vol_mult

    if version == 1:
        def patched(row, prev_row):
            if pd.isna(row["Close"]) or pd.isna(prev_row.get("High_N")):
                return None
            return "long" if row["Close"] > prev_row["High_N"] else None

    elif version == 2:
        def patched(row, prev_row):
            if any(pd.isna([row["Close"], prev_row.get("High_N"),
                            row.get("Volume"), row.get("Vol_MA20")])):
                return None
            breakout   = row["Close"]  > prev_row["High_N"]
            vol_surge  = row["Volume"] > row["Vol_MA20"] * vol_mult
            return "long" if breakout and vol_surge else None

    else:  # version 3, 4, 5
        def patched(row, prev_row):
            if any(pd.isna([row["Close"], prev_row.get("High_N"),
                            row.get("Volume"), row.get("Vol_MA20"),
                            row.get("MA_fast"), row.get("MA_slow")])):
                return None
            breakout    = row["Close"]   > prev_row["High_N"]
            vol_surge   = row["Volume"]  > row["Vol_MA20"] * vol_mult
            ma_trending = row["MA_fast"] > row["MA_slow"]
            return "long" if breakout and vol_surge and ma_trending else None

    bt._resolve_direction = patched


# ── 版本定義 ────────────────────────────────────────────────────────────

VERSIONS = [
    dict(label="V1 High_N 突破",         version=1, with_regime=False, with_52w=False),
    dict(label="V2 + 量能",               version=2, with_regime=False, with_52w=False),
    dict(label="V3 + MA 趨勢",            version=3, with_regime=False, with_52w=False),
    dict(label="V4 + 大盤EMA200",         version=4, with_regime=True,  with_52w=False),
    dict(label="V5 + 52W突破（完整版）",  version=5, with_regime=True,  with_52w=True),
]


# ── 執行單版本 ───────────────────────────────────────────────────────────

def run_version(v: dict, data: dict) -> dict:
    print(f"\n{'─'*55}")
    print(f"  跑：{v['label']}")

    cfg = make_cfg()
    bt  = Backtester(cfg)

    _patch_filter(bt,    with_regime=v["with_regime"], with_52w=v["with_52w"])
    _patch_direction(bt, version=v["version"])

    results = bt.run(data)
    equity  = results["equity_curve"]["equity"]
    trades  = results.get("trades", pd.DataFrame())

    ret    = equity.pct_change().dropna()
    years  = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr   = ((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) * 100
    mdd    = ((equity - equity.cummax()) / equity.cummax()).min() * 100
    sharpe = ret.mean() / ret.std() * 252**0.5 if ret.std() > 0 else 0

    n_trades = len(trades) if not trades.empty else 0
    win_rate = 0.0
    if n_trades > 0 and "pnl_net" in trades.columns:
        win_rate = (trades["pnl_net"] > 0).mean() * 100

    print(f"    CAGR    : {cagr:+.2f}%")
    print(f"    MDD     : {mdd:.2f}%")
    print(f"    Sharpe  : {sharpe:.3f}")
    print(f"    交易筆數 : {n_trades}")
    print(f"    勝率    : {win_rate:.1f}%")

    return {
        "版本":   v["label"],
        "CAGR%":  round(cagr,    2),
        "MDD%":   round(mdd,     2),
        "Sharpe": round(sharpe,  3),
        "筆數":   n_trades,
        "勝率%":  round(win_rate, 1),
        "equity": equity,
    }


# ── 主程式 ───────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("載入資料中...")
    cfg  = make_cfg()
    data = DataLoader.load_folder(str(STOCKS_DIR), adjusted=True)
    print(f"載入完成：{len(data)} 支股票")

    results = [run_version(v, data) for v in VERSIONS]

    # ── 報表 ──
    print(f"\n\n{'='*60}")
    print("  逐步增加條件比較（Long only）")
    print(f"{'='*60}")
    cols = ["版本", "CAGR%", "MDD%", "Sharpe", "筆數", "勝率%"]
    df   = pd.DataFrame([{c: r[c] for c in cols} for r in results])
    print(df.to_string(index=False))
    df.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")

    # ── 走勢圖 ──
    colors = ["#aaaaaa", "#3498db", "#f39c12", "#9b59b6", "#e74c3c"]
    fig, ax = plt.subplots(figsize=(14, 7))
    for r, color in zip(results, colors):
        eq = r["equity"]
        ax.plot(eq.index, eq.values, label=r["版本"], linewidth=1.5, color=color)
    ax.set_title("逐步增加條件比較（Long only）\n"
                 "V1=High_N  V2=+量能  V3=+MA  V4=+大盤EMA  V5=+52W")
    ax.set_ylabel("淨值（元）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "equity_curve.png", dpi=150)
    plt.close()

    print(f"\n圖表：{OUTPUT_DIR / 'equity_curve.png'}")
    print(f"報表：{OUTPUT_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()

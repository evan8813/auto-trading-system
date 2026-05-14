"""
experiment_long_short_split.py
────────────────────────────────
用現有主策略分別跑：
  1. 純多方（只做多，完全不做空）
  2. 純空方（只做空，完全不做多）
  3. 原版（多空都做，作為基準）

不動任何主程式，用 monkey-patch 在執行前替換方法。

執行：
  python experiment_long_short_split.py
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config      import TradingConfig
from backtester  import Backtester
from data_loader import DataLoader

OUTPUT_DIR = Path("output/long_short_split")


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


def make_cfg() -> TradingConfig:
    return TradingConfig(
        initial_equity = 1_000_000,
        max_positions  = 5,
    )


def run_mode(label: str, mode: str, data: dict) -> dict:
    """
    mode: 'both' | 'long_only' | 'short_only'
    """
    print(f"\n{'─'*50}")
    print(f"  跑：{label}")

    cfg = make_cfg()
    bt  = Backtester(cfg)

    # ── monkey-patch _resolve_direction ──
    original_resolve = bt._resolve_direction

    if mode == "long_only":
        def patched(row, prev_row):
            d = original_resolve(row, prev_row)
            return d if d == "long" else None
        bt._resolve_direction = patched

    elif mode == "short_only":
        def patched(row, prev_row):
            d = original_resolve(row, prev_row)
            return d if d == "short" else None
        bt._resolve_direction = patched

    results = bt.run(data)
    equity  = results["equity_curve"]["equity"]
    trades  = results.get("trades", pd.DataFrame())

    # 績效
    ret    = equity.pct_change().dropna()
    years  = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr   = ((equity.iloc[-1] / equity.iloc[0]) ** (1/years) - 1) * 100
    mdd    = ((equity - equity.cummax()) / equity.cummax()).min() * 100
    sharpe = ret.mean() / ret.std() * 252**0.5 if ret.std() > 0 else 0

    n_trades = len(trades) if not trades.empty else 0
    win_rate = 0.0
    if n_trades > 0 and "pnl_net" in trades.columns:
        win_rate = (trades["pnl_net"] > 0).mean() * 100

    print(f"    CAGR   : {cagr:+.2f}%")
    print(f"    MDD    : {mdd:.2f}%")
    print(f"    Sharpe : {sharpe:.3f}")
    print(f"    交易筆數: {n_trades}")
    print(f"    勝率   : {win_rate:.1f}%")

    return {
        "模式":   label,
        "CAGR%":  round(cagr, 2),
        "MDD%":   round(mdd, 2),
        "Sharpe": round(sharpe, 3),
        "筆數":   n_trades,
        "勝率%":  round(win_rate, 1),
        "equity": equity,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("載入資料中...")
    cfg  = make_cfg()
    data = DataLoader.load_folder(str(STOCKS_DIR), adjusted=True)
    print(f"載入完成：{len(data)} 支股票")

    results = []
    results.append(run_mode("原版（多空都做）",    "both",         data))
    results.append(run_mode("純多方（Long only）", "long_only",    data))
    results.append(run_mode("純空方（Short only）","short_only",   data))

    # ── 報表 ──
    print(f"\n\n{'='*55}")
    print("  最終比較")
    print(f"{'='*55}")
    cols = ["模式", "CAGR%", "MDD%", "Sharpe", "筆數", "勝率%"]
    df   = pd.DataFrame([{c: r[c] for c in cols} for r in results])
    print(df.to_string(index=False))
    df.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")

    # ── 走勢圖 ──
    fig, ax = plt.subplots(figsize=(14, 6))
    colors  = ["#555555", "#2ecc71", "#e74c3c"]
    for r, color in zip(results, colors):
        eq = r["equity"]
        ax.plot(eq.index, eq.values, label=r["模式"], linewidth=1.5, color=color)
    ax.set_title("主策略：原版 vs 純多方 vs 純空方")
    ax.set_ylabel("淨值（元）")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "equity_curve.png", dpi=150)
    plt.close()

    print(f"\n圖表：{OUTPUT_DIR / 'equity_curve.png'}")
    print(f"報表：{OUTPUT_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()

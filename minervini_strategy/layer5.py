"""
Layer 5：動能排名選股
======================
在 Layer 4（52週突破 + 股價下限 + 均線多頭排列 + ATR追蹤停損）基礎上：
  從每日符合條件的股票中，依 ROC(60) 動能排名，只選前 max_positions 強。

解決的問題：
  Layer 1~4 當多支股票同時有訊號，進場順序由股票代號決定（偏差）。
  Layer 5 改為「選動能最強的那幾支」，進場更有依據。

進場：ROC(60) 排名前 5 且符合 Layer 3 所有條件
出場：跌出前 5 名（訊號消失）OR ATR 追蹤停損觸發
部位：海龜法 — risk_amount / (ATR × atr_multiplier × 1000)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from data_loader import load_data
from layer4 import compute_atr
from sim import sim


def build_position(close, cfg: Config) -> pd.DataFrame:
    # ── Layer 1：52 週新高突破 ──────────────────
    high_52w = close.shift(1).rolling(cfg.breakout_n, min_periods=126).max()
    breakout  = close > high_52w

    # ── Layer 2：股價下限 ───────────────────────
    above_floor = close >= cfg.min_price_long

    # ── Layer 3：均線多頭排列 ───────────────────
    ma_fast    = close.rolling(cfg.ma_fast).mean()
    ma_slow    = close.rolling(cfg.ma_slow).mean()
    ma_aligned = (close > ma_fast) & (ma_fast > ma_slow)

    condition = breakout & above_floor & ma_aligned

    # ── Layer 5 新增：ROC(60) 動能排名 ──────────
    roc    = close / close.shift(cfg.roc_period) - 1
    ranked = roc.where(condition)                           # 不符合條件的設為 NaN
    top_n  = ranked.rank(axis=1, ascending=False, na_option='bottom') <= cfg.max_positions
    position = top_n & condition                            # 排名前 N 且條件成立

    return position.fillna(False)


if __name__ == "__main__":
    cfg = Config()
    print("── Layer 5：動能排名選股（ROC Top 5）+ ATR 追蹤停損 ──")

    print("\n[1/4] 載入資料...")
    data  = load_data()
    close = data["close"]
    high  = data["high"]
    low   = data["low"]

    print("[2/4] 計算 ATR...")
    atr = compute_atr(high, low, close, cfg.atr_period)

    print("[3/4] 計算訊號...")
    position  = build_position(close, cfg)
    n_signals = position.sum(axis=1)
    print(f"      平均每日有 {n_signals.mean():.1f} 檔入選（上限 {cfg.max_positions} 檔）")

    print("[4/4] 開始回測...")
    report = sim(
        position            = position,
        close               = close,
        atr                 = atr,
        atr_multiplier      = cfg.atr_multiplier,
        risk_pct            = cfg.risk_pct,
        max_risk_amount     = cfg.max_risk_amount,
        exit_on_signal_off  = True,
        stop_loss           = None,
        max_positions       = cfg.max_positions,
        fee_ratio           = cfg.fee_ratio,
        tax_ratio           = cfg.tax_ratio,
        initial_equity      = cfg.initial_equity,
    )

    print("\n── 回測結果 ──")
    report.print_stats()
    report.plot(title="Layer 5：ROC 動能排名 + ATR 追蹤停損")

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    report.trades.to_csv(os.path.join(out_dir, "layer5_trades.csv"), index=False, encoding="utf-8-sig")
    report.equity.to_csv(os.path.join(out_dir, "layer5_equity.csv"), header=["equity"], encoding="utf-8-sig")
    print(f"\n結果已儲存至 {out_dir}/")

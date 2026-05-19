"""
minervini.py — Mark Minervini 動能選股策略
============================================
完整實作 Minervini 的 8 個選股條件：

  1. 收盤 > MA150 且 收盤 > MA200
  2. MA150 > MA200
  3. MA200 呈上升趨勢至少 1 個月（20 日）
  4. MA50 > MA150 且 MA50 > MA200
  5. 收盤 > MA50
  6. 收盤 > 52 週低點 × 1.30（距低點至少 +30%）
  7. 收盤 > 52 週高點 × 0.75（在高點 25% 以內）
  8. 自製 RS-Rating > 70（過去252日漲幅贏過市場70%以上）

進場：上述 8 條件同時成立，且為首次觸發（False → True）
出場：ATR 追蹤停損（trail_stop = max(trail_stop, close - atr_mult × ATR)）
      OR 訊號消失（exit_on_signal_off=True）
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
    # ── 條件 1：收盤 > MA150 且 收盤 > MA200 ────
    ma150 = close.rolling(150).mean()
    ma200 = close.rolling(200).mean()
    above_ma150_200 = (close > ma150) & (close > ma200)

    # ── 條件 2：MA150 > MA200 ───────────────────
    ma150_above_ma200 = ma150 > ma200

    # ── 條件 3：MA200 上升趨勢至少 1 個月（20日）─
    ma200_trending = ma200 > ma200.shift(20)

    # ── 條件 4：MA50 > MA150 且 MA50 > MA200 ────
    ma50 = close.rolling(50).mean()
    ma50_above = (ma50 > ma150) & (ma50 > ma200)

    # ── 條件 5：收盤 > MA50 ─────────────────────
    above_ma50 = close > ma50

    # ── 條件 6：收盤 > 52 週低點 × 1.30 ─────────
    low_52w  = close.rolling(252, min_periods=126).min()
    above_low = close > low_52w * 1.30

    # ── 條件 7：收盤在 52 週高點 25% 以內 ─────────
    high_52w   = close.rolling(252, min_periods=126).max()
    near_high  = close >= high_52w * 0.75

    # ── 條件 8：自製 RS-Rating > 70 ──────────────
    # 個股252日漲幅在全市場的百分位排名
    rs_raw    = close / close.shift(252) - 1
    rs_rating = rs_raw.rank(axis=1, pct=True) * 100   # 0~100
    rs_ok     = rs_rating > 70

    position = (
        above_ma150_200
        & ma150_above_ma200
        & ma200_trending
        & ma50_above
        & above_ma50
        & above_low
        & near_high
        & rs_ok
    )

    return position.fillna(False)


if __name__ == "__main__":
    cfg = Config()
    print("── Minervini 策略：8 條件選股 + ATR 追蹤停損 ──")

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
    print(f"      平均每日有 {n_signals.mean():.1f} 檔符合條件（上限 {cfg.max_positions} 檔）")

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
    report.plot(title="Minervini 策略：8 條件選股 + ATR 追蹤停損")

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    report.trades.to_csv(os.path.join(out_dir, "minervini_trades.csv"), index=False, encoding="utf-8-sig")
    report.equity.to_csv(os.path.join(out_dir, "minervini_equity.csv"), header=["equity"], encoding="utf-8-sig")
    print(f"\n結果已儲存至 {out_dir}/")

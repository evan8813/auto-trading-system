"""
Layer 4：固定停損 → ATR 追蹤停損
===================================
在 Layer 3（52週突破 + 股價下限 + 均線多頭排列）基礎上：
  將固定停損 8% 換成 ATR 追蹤停損。

ATR 追蹤停損邏輯：
  - 進場時：trail_stop = entry_price − atr_multiplier × ATR
  - 每日更新：trail_stop = max(trail_stop, close − atr_multiplier × ATR)
              （只升不降，讓獲利奔跑）
  - 出場條件：close < trail_stop

進出場訊號：
  - 進場：Layer 3 條件觸發（52週突破 & 股價>=10 & 均線多頭排列）
  - 出場：ATR 追蹤停損（不再依賴訊號消失出場）
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from data_loader import load_data
from sim import sim


def compute_atr(high: pd.DataFrame, low: pd.DataFrame,
                close: pd.DataFrame, period: int) -> pd.DataFrame:
    """計算每支股票的 ATR（寬格式 DataFrame）。"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low  - prev_close).abs()
    tr  = np.maximum(np.maximum(tr1.values, tr2.values), tr3.values)
    tr  = pd.DataFrame(tr, index=close.index, columns=close.columns)
    return tr.rolling(period).mean()


def build_position(close, cfg: Config):
    # ── Layer 1：52 週新高突破 ──────────────────
    high_52w = close.shift(1).rolling(cfg.breakout_n, min_periods=126).max()
    breakout  = close > high_52w

    # ── Layer 2：股價下限 ───────────────────────
    above_floor = close >= cfg.min_price_long

    # ── Layer 3：均線多頭排列 ───────────────────
    ma_fast    = close.rolling(cfg.ma_fast).mean()
    ma_slow    = close.rolling(cfg.ma_slow).mean()
    ma_aligned = (close > ma_fast) & (ma_fast > ma_slow)

    return breakout & above_floor & ma_aligned


if __name__ == "__main__":
    cfg = Config()
    print("── Layer 4：ATR 追蹤停損（取代固定停損 8%）──")

    print("\n[1/4] 載入資料...")
    data  = load_data()
    close = data["close"]
    high  = data["high"]
    low   = data["low"]

    print("[2/4] 計算 ATR...")
    atr = compute_atr(high, low, close, cfg.atr_period)
    print(f"      ATR 週期：{cfg.atr_period} 日，倍數：{cfg.atr_multiplier}")

    print("[3/4] 計算訊號...")
    position  = build_position(close, cfg)
    n_signals = position.sum(axis=1)
    print(f"      平均每日有 {n_signals.mean():.0f} 檔符合條件")

    print("[4/4] 開始回測...")
    report = sim(
        position            = position,
        close               = close,
        atr                 = atr,
        atr_multiplier      = cfg.atr_multiplier,
        risk_pct            = cfg.risk_pct,
        max_risk_amount     = cfg.max_risk_amount,
        exit_on_signal_off  = True,    # 訊號消失也出場（避免鎖倉）
        stop_loss           = None,
        max_positions       = cfg.max_positions,
        fee_ratio           = cfg.fee_ratio,
        tax_ratio           = cfg.tax_ratio,
        initial_equity      = cfg.initial_equity,
    )

    print("\n── 回測結果 ──")
    report.print_stats()
    report.plot(title="Layer 4：ATR 追蹤停損")

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    report.trades.to_csv(os.path.join(out_dir, "layer4_trades.csv"), index=False, encoding="utf-8-sig")
    report.equity.to_csv(os.path.join(out_dir, "layer4_equity.csv"), header=["equity"], encoding="utf-8-sig")
    print(f"\n結果已儲存至 {out_dir}/")

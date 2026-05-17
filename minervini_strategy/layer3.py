"""
Layer 3：加均線多頭排列
========================
在 Layer 2（52週突破 + 股價 >= 10元）基礎上新增：
  3. 均線多頭排列：收盤 > MA50 且 MA50 > MA100

用意：確保進場時個股本身處於上升趨勢，避免在弱勢股上面追高突破。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from config import Config
from data_loader import load_data
from sim import sim


def build_position(close, cfg: Config):
    # ── Layer 1：52 週新高突破 ──────────────────
    high_52w = close.shift(1).rolling(cfg.breakout_n, min_periods=126).max()
    breakout  = close > high_52w

    # ── Layer 2：股價下限 ───────────────────────
    above_floor = close >= cfg.min_price_long

    # ── Layer 3 新增：均線多頭排列 ──────────────
    ma_fast    = close.rolling(cfg.ma_fast).mean()   # MA50
    ma_slow    = close.rolling(cfg.ma_slow).mean()   # MA100
    ma_aligned = (close > ma_fast) & (ma_fast > ma_slow)

    position = breakout & above_floor & ma_aligned
    return position


if __name__ == "__main__":
    cfg = Config()
    print("── Layer 3：52週突破 + 股價下限 + 均線多頭排列 ──")

    print("\n[1/3] 載入資料...")
    data  = load_data()
    close = data["close"]

    print("[2/3] 計算訊號...")
    position  = build_position(close, cfg)
    n_signals = position.sum(axis=1)
    print(f"      平均每日有 {n_signals.mean():.0f} 檔符合條件（Layer 2 約 18 檔）")

    print("[3/3] 開始回測...")
    report = sim(
        position       = position,
        close          = close,
        stop_loss      = cfg.stop_loss,
        max_positions  = cfg.max_positions,
        fee_ratio      = cfg.fee_ratio,
        tax_ratio      = cfg.tax_ratio,
        initial_equity = cfg.initial_equity,
    )

    print("\n── 回測結果 ──")
    report.print_stats()
    report.plot(title="Layer 3：52週突破 + 股價下限 + 均線多頭排列")

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    report.trades.to_csv(os.path.join(out_dir, "layer3_trades.csv"), index=False, encoding="utf-8-sig")
    report.equity.to_csv(os.path.join(out_dir, "layer3_equity.csv"), header=["equity"], encoding="utf-8-sig")
    print(f"\n結果已儲存至 {out_dir}/")

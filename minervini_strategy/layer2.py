"""
Layer 2：加股價下限
====================
在 Layer 1（52週新高突破）基礎上新增：
  1. 收盤價 >= 10 元（過濾雞蛋水餃股）

進場：兩個條件同時成立
出場：任一條件不再成立 OR 固定停損 8%

注意：20日均成交金額過濾測試後對績效有負面影響（門檻 500萬過嚴），
      暫不納入，待後期層次調整後再評估。
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
    breakout = close > high_52w

    # ── Layer 2 新增：股價下限 ──────────────────
    above_price_floor = close >= cfg.min_price_long  # 收盤 >= 10 元

    position = breakout & above_price_floor
    return position


if __name__ == "__main__":
    cfg = Config()
    print("── Layer 2：52週突破 + 流動性 + 股價下限 ──")

    print("\n[1/3] 載入資料...")
    data  = load_data()
    close = data["close"]

    print("[2/3] 計算訊號...")
    position  = build_position(close, cfg)
    n_signals = position.sum(axis=1)
    print(f"      平均每日有 {n_signals.mean():.0f} 檔符合條件（Layer 1 約 18 檔）")

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
    report.plot(title="Layer 2：52週突破 + 流動性 + 股價下限")

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    report.trades.to_csv(os.path.join(out_dir, "layer2_trades.csv"), index=False, encoding="utf-8-sig")
    report.equity.to_csv(os.path.join(out_dir, "layer2_equity.csv"), header=["equity"], encoding="utf-8-sig")
    print(f"\n結果已儲存至 {out_dir}/")

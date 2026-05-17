"""
Layer 1：52 週新高突破（最純粹版本）
======================================
洋蔥第一層，只驗證一件事：
  「收盤價突破過去 252 個交易日的最高收盤價，就買進。」

進場：今日收盤 > 昨日為止的 252 日滾動最高收盤（shift(1) 避免未來資訊）
出場：訊號消失（收盤跌回 252 日高點以下）OR 固定停損 8%
持倉：最多 5 檔，等權重（各占總資產 1/5）
資金：100 萬元
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_data
from sim import sim


def build_position(close):
    """
    52 週新高突破訊號。
    shift(1)：用昨天為止的歷史最高，避免當日未來資訊偏差。
    """
    high_52w = close.shift(1).rolling(252, min_periods=126).max()
    position = close > high_52w
    return position


if __name__ == "__main__":
    print("── Layer 1：52 週新高突破 ──")

    print("\n[1/3] 載入資料...")
    data  = load_data()
    close = data["close"]

    print("[2/3] 計算訊號...")
    position = build_position(close)
    n_signals = position.sum(axis=1)
    print(f"      平均每日有 {n_signals.mean():.0f} 檔符合條件")

    print("[3/3] 開始回測...")
    report = sim(
        position       = position,
        close          = close,
        stop_loss      = 0.08,
        max_positions  = 5,
        initial_equity = 1_000_000,
    )

    print("\n── 回測結果 ──")
    report.print_stats()
    report.plot(title="Layer 1：52週新高突破")

    # 儲存 trade log
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    report.trades.to_csv(os.path.join(out_dir, "layer1_trades.csv"), index=False, encoding="utf-8-sig")
    report.equity.to_csv(os.path.join(out_dir, "layer1_equity.csv"), header=["equity"], encoding="utf-8-sig")
    print(f"\n結果已儲存至 {out_dir}/")

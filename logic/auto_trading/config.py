"""
config.py
─────────
交易系統的所有可調參數集中在這裡。
修改此檔即可調整策略邏輯，無需碰其他模組。
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class TradingConfig:
    """使用者可調整的交易系統參數"""

    # ── 資金設定 ──
    initial_equity: float = 180_000.0         # 初始資金（新台幣）
    risk_pct: float = 0.01                   # 每筆風險比例（1% → 每筆最多虧 1,800 元）
    max_risk_amount: float = 2_500.0          # 單筆風險金額硬上限（元），防止資產暴增後風險失控

    # ── 交易成本 ──
    commission_rate: float = 0.001425        # 手續費率（買賣各）
    transaction_tax: float = 0.003           # 證交稅（賣出時）
    slippage: float = 0.001                  # 滑價比例

    # ── 指標參數 ──
    breakout_window: int = 20                # 突破 N 日高 / 低（進場）
    stop_window: int = 10                    # N 日低 / 高（初始停損）
    macd_fast: int = 12                      # MACD 快線 EMA 週期
    macd_slow: int = 26                      # MACD 慢線 EMA 週期
    macd_signal: int = 9                     # MACD Signal EMA 週期
    atr_period: int = 100                    # ATR 週期
    atr_multiplier: float = 3.0              # ATR 停損倍數
    week52: int = 252                        # 52 週交易日

    # ── 篩選條件 ──
    min_avg_amount:  float = 5_000_000.0     # 最低 20 日平均成交金額（元）
    min_long_price:  float = 10.0            # 做多股價下限（元）
    min_short_price: float = 20.0            # 做空股價下限（元）

    # ── 大盤環境（TAIEX 200 日 EMA）──
    taiex_csv_path:  str = ""                # 加權指數 CSV 路徑（空字串 = 停用環境過濾）
    taiex_ema_period: int = 200              # 大盤 EMA 週期

    # ── 回測區間 ──
    backtest_start: str = "2010-01-01"
    backtest_end: str = "2023-12-29"

    # ── 部位限制 ──
    max_positions: int = 5                   # 最大同時持倉數（180,000 ÷ 5 = 36,000/倉）
    max_trade_cost: float = 36_000.0         # 單筆最高買入金額（= 總資金 ÷ max_positions）
    point_value: float = 1.0                 # 每點價值（股票 = 1）

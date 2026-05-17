"""
config.py — 策略全域設定
所有參數集中在這裡，每層只需改這份。
"""
from dataclasses import dataclass


@dataclass
class Config:
    # ── 資金 ──────────────────────────────────
    initial_equity: float = 1_000_000     # 初始資金（元）

    # ── 風控 ──────────────────────────────────
    risk_pct:        float = 0.02         # 每筆最大虧損佔淨值 2%
    max_risk_amount: float = 15_000       # 單筆風險硬上限（元）
    max_positions:   int   = 5            # 最大同時持倉數

    # ── 技術指標 ──────────────────────────────
    atr_period:      int   = 100          # ATR 計算週期（Layer 5 用）
    atr_multiplier:  float = 3.0          # ATR 停損倍數（Layer 5 用）
    ma_fast:         int   = 50           # 快速均線日數（Layer 3 用）
    ma_slow:         int   = 100          # 慢速均線日數（Layer 3 用）
    breakout_n:      int   = 252          # 52 週突破回看天數（Layer 1 核心）
    taiex_ema:       int   = 200          # 大盤 EMA 天數（Layer 3 用）

    # ── 選股門檻 ──────────────────────────────
    min_amount_20d:  float = 5_000_000    # 20 日均成交金額下限（Layer 2 用）
    min_price_long:  float = 10.0         # 做多股價下限（Layer 2 用）
    min_price_short: float = 20.0         # 做空股價下限（Layer 2 用）

    # ── 執行規則 ──────────────────────────────
    trade_at: str = "next_open"           # "next_open" = T+1 隔日開盤

    # ── 費用 ──────────────────────────────────
    fee_ratio: float = 1.425 / 1000 / 3  # 手續費（折扣後，買賣各收）
    tax_ratio: float = 3 / 1000          # 交易稅（賣出時收）

    # ── Layer 1 固定停損（Layer 5 換成 ATR 追蹤停損後移除）──
    stop_loss: float = 0.08              # 固定停損 8%

"""
Minervini-Style Momentum Breakout Strategy (Taiwan Market)
==========================================================
Inspired by Mark Minervini's SEPA methodology.

Entry logic (all conditions must be met):
  1. Price making 52-week new high
  2. Bullish MA alignment: MA20 > MA60 > MA120
  3. Volume expansion: today > 1.5x 20-day average
  4. Positive EPS (profitable company)

Ranking: top 15 by 3-month momentum
Rebalance: monthly
Risk: 8% stop-loss, 25% take-profit, equal-weight (1/15 per stock)
"""

from finlab import data
from finlab.backtest import sim


def build_position():
    # --- Data ---
    close  = data.get("price:收盤價")
    volume = data.get("price:成交股數")
    eps    = data.get("fundamental_features:EPS")

    # --- Technical conditions ---

    # 1. 52-week new high (strength only)
    new_high = close >= close.rolling(252, min_periods=126).max()

    # 2. Bullish MA alignment: price > MA20 > MA60 > MA120
    ma20  = close.average(20)
    ma60  = close.average(60)
    ma120 = close.average(120)
    ma_aligned = (close > ma20) & (ma20 > ma60) & (ma60 > ma120)

    # 3. Volume expansion: today > 1.5x 20-day average
    vol_avg = volume.average(20)
    vol_expansion = volume > vol_avg * 1.5

    # --- Fundamental condition ---

    # 4. Positive EPS (profitable company)
    profitable = eps > 0

    # --- Combine all conditions ---
    condition = new_high & ma_aligned & vol_expansion & profitable

    # --- Rank survivors by 3-month momentum, pick top 15 ---
    momentum_3m = close / close.shift(63) - 1
    position = momentum_3m[condition].is_largest(15)

    return position


if __name__ == "__main__":
    position = build_position()

    report = sim(
        position,
        resample="M",
        stop_loss=0.08,
        take_profit=0.25,
        position_limit=1/15,
        upload=False,
    )

    stats = report.get_stats()
    print(f"CAGR:         {stats['cagr']:.2%}")
    print(f"Max Drawdown: {stats['max_drawdown']:.2%}")
    print(f"Sharpe:       {stats['monthly_sharpe']:.2f}")

    report

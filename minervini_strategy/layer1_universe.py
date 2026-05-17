"""
海龜趨勢策略 — Layer 1：基礎動能股票池（只做多）
=================================================
洋蔥法第一層，驗證基礎動能是否有效。

條件：
  1. 52 週新高（只選強勢股，不混空方）
  2. 成交量過濾（20 日均量 > 1,000 張，確保流動性）
  3. MA50 > MA100（多頭趨勢確認）
  4. 依 ROC(20) 動能排名，選出前 20 檔

出場：
  - 固定停損 8%（Layer 4 再換成 ATR 追蹤停損）

換倉：每月
"""

from finlab import data
from finlab.backtest import sim


def build_position():
    close  = data.get("price:收盤價")
    high   = data.get("price:最高價")
    volume = data.get("price:成交股數")

    # --- 條件 1：52 週新高（只做多，不混空方）---
    high_52w = high.rolling(252, min_periods=126).max()
    new_high  = close >= high_52w

    # --- 條件 2：流動性過濾（20 日均量 > 1,000 張）---
    vol_avg    = volume.average(20) / 1000  # 換算成張
    liquid     = vol_avg > 1000

    # --- 條件 3：均線多頭（MA50 > MA100）---
    ma50  = close.average(50)
    ma100 = close.average(100)
    trend = ma50 > ma100

    # --- 組合條件 ---
    condition = new_high & liquid & trend

    # --- 依 ROC(20) 排名，從符合條件的股票裡選前 20 ---
    roc20    = close / close.shift(20) - 1
    position = roc20[condition].is_largest(20)

    return position


if __name__ == "__main__":
    position = build_position()

    report = sim(
        position,
        resample="M",
        stop_loss=0.08,           # 固定停損 8%，ATR 版本留到 Layer 4
        trade_at_price="open",    # 隔日開盤進場（符合海龜規則）
        position_limit=1/20,      # 等權重
        fee_ratio=1.425/1000/3,   # 台股手續費（折扣後）
        tax_ratio=3/1000,
        upload=False,
    )

    stats = report.get_stats()
    print("=" * 40)
    print("Layer 1 回測結果")
    print("=" * 40)
    print(f"CAGR:         {stats['cagr']:.2%}")
    print(f"Max Drawdown: {stats['max_drawdown']:.2%}")
    print(f"Sharpe:       {stats['monthly_sharpe']:.2f}")
    print(f"勝率:         {stats.get('win_rate', 'N/A')}")
    print("=" * 40)

    # Jupyter 裡直接顯示互動圖表
    report

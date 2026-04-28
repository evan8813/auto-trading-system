"""
test_universe_filter.py
───────────────────────
步驟 6：驗證每日股票篩選邏輯。

測試項目：
  1. 同時符合流動性、近52週高點、股價可負擔 → 納入候選
  2. 流動性不足 → 排除
  3. 距52週高點過遠 → 排除
  4. 股價超出資金可負擔範圍 → 排除
  5. 兩個條件都不符合 → 排除
  6. 指定日期不在 DataFrame → 跳過
  7. Avg_Amount_20 為 NaN → 排除
  8. High_52W 為 NaN 或 0 → 排除
  9. 多支股票篩選結果正確
"""

import pytest
import numpy as np
import pandas as pd

from config import TradingConfig
from universe_filter import UniverseFilter


def make_filter(
    min_avg_amount: float = 5_000_000,
    initial_equity: float = 180_000,
) -> UniverseFilter:
    cfg = TradingConfig(min_avg_amount=min_avg_amount, initial_equity=initial_equity)
    return UniverseFilter(cfg)


def make_df_with_row(
    date: str,
    close: float,
    high_52w: float,
    avg_amount_20: float,
) -> pd.DataFrame:
    """建立單日 DataFrame，index 為指定日期"""
    ts = pd.Timestamp(date)
    return pd.DataFrame({
        "Close":         [close],
        "High_52W":      [high_52w],
        "Avg_Amount_20": [avg_amount_20],
    }, index=[ts])


class TestUniverseFilterPassConditions:
    """通過篩選的情境"""

    def test_passes_all_conditions(self):
        # close=95 <= 180(max_price) ✓；close=95 >= 100×0.90=90 ✓；avg=10M >= 5M ✓
        uf  = make_filter(min_avg_amount=5_000_000, initial_equity=180_000)
        df  = make_df_with_row("2023-06-01", close=95.0, high_52w=100.0, avg_amount_20=10_000_000)
        result = uf.filter({"2330": df}, pd.Timestamp("2023-06-01"))
        assert "2330" in result

    def test_passes_at_exact_90pct_threshold(self):
        uf  = make_filter()
        df  = make_df_with_row("2023-06-01", close=90.0, high_52w=100.0, avg_amount_20=6_000_000)
        # close=90 >= 100×0.90=90 ✓（邊界值）
        result = uf.filter({"2330": df}, pd.Timestamp("2023-06-01"))
        assert "2330" in result

    def test_passes_at_exact_min_avg_amount(self):
        uf  = make_filter(min_avg_amount=5_000_000)
        df  = make_df_with_row("2023-06-01", close=95.0, high_52w=100.0, avg_amount_20=5_000_000)
        # avg_amount=5M >= 5M ✓（邊界值）
        result = uf.filter({"2330": df}, pd.Timestamp("2023-06-01"))
        assert "2330" in result

    def test_passes_at_exact_max_price(self):
        # close=180 <= 180,000/1000=180 ✓（邊界值）
        uf  = make_filter(initial_equity=180_000)
        df  = make_df_with_row("2023-06-01", close=180.0, high_52w=185.0, avg_amount_20=10_000_000)
        result = uf.filter({"2330": df}, pd.Timestamp("2023-06-01"))
        assert "2330" in result


class TestUniverseFilterFailConditions:
    """不通過篩選的情境"""

    def test_fails_liquidity(self):
        uf  = make_filter(min_avg_amount=5_000_000)
        df  = make_df_with_row("2023-06-01", close=95.0, high_52w=100.0, avg_amount_20=4_000_000)
        result = uf.filter({"2330": df}, pd.Timestamp("2023-06-01"))
        assert "2330" not in result

    def test_fails_near_52w_high(self):
        uf  = make_filter()
        df  = make_df_with_row("2023-06-01", close=80.0, high_52w=100.0, avg_amount_20=10_000_000)
        # close=80 < 100×0.90=90 ✗
        result = uf.filter({"2330": df}, pd.Timestamp("2023-06-01"))
        assert "2330" not in result

    def test_fails_too_expensive(self):
        # close=200 > 180,000/1000=180 ✗（超出資金可負擔範圍）
        uf  = make_filter(initial_equity=180_000)
        df  = make_df_with_row("2023-06-01", close=200.0, high_52w=210.0, avg_amount_20=10_000_000)
        result = uf.filter({"2330": df}, pd.Timestamp("2023-06-01"))
        assert "2330" not in result

    def test_fails_all_conditions(self):
        uf  = make_filter(min_avg_amount=5_000_000, initial_equity=180_000)
        df  = make_df_with_row("2023-06-01", close=300.0, high_52w=100.0, avg_amount_20=1_000_000)
        result = uf.filter({"2330": df}, pd.Timestamp("2023-06-01"))
        assert "2330" not in result


class TestUniverseFilterEdgeCases:
    """邊界條件"""

    def test_date_not_in_df_skipped(self):
        uf  = make_filter()
        df  = make_df_with_row("2023-06-01", close=95.0, high_52w=100.0, avg_amount_20=10_000_000)
        # 查詢不存在的日期
        result = uf.filter({"2330": df}, pd.Timestamp("2023-07-01"))
        assert "2330" not in result

    def test_nan_avg_amount_excluded(self):
        uf  = make_filter()
        ts  = pd.Timestamp("2023-06-01")
        df  = pd.DataFrame({
            "Close":         [95.0],
            "High_52W":      [100.0],
            "Avg_Amount_20": [np.nan],
        }, index=[ts])
        result = uf.filter({"2330": df}, ts)
        assert "2330" not in result

    def test_nan_high_52w_excluded(self):
        uf  = make_filter()
        ts  = pd.Timestamp("2023-06-01")
        df  = pd.DataFrame({
            "Close":         [95.0],
            "High_52W":      [np.nan],
            "Avg_Amount_20": [10_000_000],
        }, index=[ts])
        result = uf.filter({"2330": df}, ts)
        assert "2330" not in result

    def test_zero_high_52w_excluded(self):
        uf  = make_filter()
        df  = make_df_with_row("2023-06-01", close=0.01, high_52w=0.0, avg_amount_20=10_000_000)
        result = uf.filter({"2330": df}, pd.Timestamp("2023-06-01"))
        assert "2330" not in result

    def test_empty_data_dict(self):
        uf     = make_filter()
        result = uf.filter({}, pd.Timestamp("2023-06-01"))
        assert result == []


class TestUniverseFilterMultipleStocks:
    """多支股票同時篩選"""

    def test_filters_correct_subset(self):
        uf  = make_filter(min_avg_amount=5_000_000, initial_equity=180_000)
        ts  = pd.Timestamp("2023-06-01")

        # 2330：三個條件全過（close=95 <= 180, 近高點, 流動性足）
        df_pass = make_df_with_row("2023-06-01", close=95.0, high_52w=100.0, avg_amount_20=10_000_000)
        # 2317：流動性不足
        df_low_liq = make_df_with_row("2023-06-01", close=95.0, high_52w=100.0, avg_amount_20=3_000_000)
        # 2454：距高點太遠
        df_far_high = make_df_with_row("2023-06-01", close=80.0, high_52w=100.0, avg_amount_20=10_000_000)
        # 2412：股價超出資金範圍（close=200 > 180）
        df_expensive = make_df_with_row("2023-06-01", close=200.0, high_52w=210.0, avg_amount_20=10_000_000)

        data = {"2330": df_pass, "2317": df_low_liq, "2454": df_far_high, "2412": df_expensive}
        result = uf.filter(data, ts)

        assert "2330" in result
        assert "2317" not in result
        assert "2454" not in result
        assert "2412" not in result
        assert len(result) == 1

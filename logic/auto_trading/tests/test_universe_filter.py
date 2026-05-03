"""
test_universe_filter.py
───────────────────────
步驟 6：驗證每日股票篩選邏輯。

篩選條件（依 CLAUDE.md 規格）：
  1. 20 日平均成交金額 >= min_avg_amount（流動性）
  2. 今日 High > 昨日 High_52W 或 今日 Low < 昨日 Low_52W（52 週突破）
  3. 收盤 × 1000 <= equity / max_positions（單倉動態上限）
"""

import pytest
import numpy as np
import pandas as pd

from config import TradingConfig
from universe_filter import UniverseFilter

DATE_PREV  = pd.Timestamp("2023-06-01")
DATE_TODAY = pd.Timestamp("2023-06-02")
DEFAULT_EQUITY = 180_000.0


def make_filter(
    min_avg_amount: float = 5_000_000,
    max_positions:  int   = 5,
) -> UniverseFilter:
    cfg = TradingConfig(min_avg_amount=min_avg_amount, max_positions=max_positions)
    return UniverseFilter(cfg)


def make_df(
    high: float = 110.0,
    low: float = 90.0,
    close: float = 30.0,
    avg_amount: float = 10_000_000,
    prev_high_52w: float = 105.0,
    prev_low_52w: float = 95.0,
) -> pd.DataFrame:
    """建立兩列測試 DataFrame：index[0]=昨日, index[1]=今日"""
    return pd.DataFrame({
        "High":          [100.0,      high],
        "Low":           [100.0,      low],
        "Close":         [100.0,      close],
        "Avg_Amount_20": [avg_amount, avg_amount],
        "High_52W":      [prev_high_52w, prev_high_52w],
        "Low_52W":       [prev_low_52w,  prev_low_52w],
    }, index=[DATE_PREV, DATE_TODAY])


class TestUniverseFilterPassConditions:
    """應該通過篩選的情境"""

    def test_passes_all_conditions(self):
        """三條件全符合：突破高點、均量足、股價可負擔"""
        # equity=180K, max_positions=5 → max_price=36
        # high=110 > prev_high_52w=105 ✓, avg=10M >= 5M ✓, close=30 <= 36 ✓
        uf     = make_filter()
        df     = make_df(high=110.0, low=96.0, close=30.0)
        result = uf.filter({"2330": df}, DATE_TODAY, equity=DEFAULT_EQUITY)
        assert "2330" in result

    def test_passes_when_low_breaks_52w_low(self):
        """今日 Low < 昨日 Low_52W → 空方突破也入選"""
        uf     = make_filter()
        df     = make_df(high=104.0, low=88.0, close=30.0,
                         prev_high_52w=105.0, prev_low_52w=92.0)
        result = uf.filter({"2330": df}, DATE_TODAY, equity=DEFAULT_EQUITY)
        assert "2330" in result

    def test_passes_at_exact_min_avg_amount(self):
        """均量剛好等於門檻（>= 邊界值應通過）"""
        uf     = make_filter(min_avg_amount=5_000_000)
        df     = make_df(avg_amount=5_000_000)
        result = uf.filter({"2330": df}, DATE_TODAY, equity=DEFAULT_EQUITY)
        assert "2330" in result

    def test_passes_at_exact_max_price(self):
        """股價剛好等於動態單倉上限（邊界值應通過）"""
        # equity=180,000, max_positions=5 → max_price=180,000/(5×1000)=36
        uf     = make_filter(max_positions=5)
        df     = make_df(close=36.0)
        result = uf.filter({"2330": df}, DATE_TODAY, equity=180_000.0)
        assert "2330" in result

    def test_max_price_grows_with_equity(self):
        """equity 成長後，原本太貴的股票可以入選"""
        uf  = make_filter(max_positions=5)
        df  = make_df(close=72.0)
        # equity=180K → max_price=36，close=72 > 36 → 不通過
        assert "2330" not in uf.filter({"2330": df}, DATE_TODAY, equity=180_000.0)
        # equity=360K → max_price=72，close=72 = 72 → 通過
        assert "2330" in uf.filter({"2330": df}, DATE_TODAY, equity=360_000.0)


class TestUniverseFilterFailConditions:
    """應該被排除的情境"""

    def test_fails_liquidity(self):
        """均量不足"""
        uf     = make_filter(min_avg_amount=5_000_000)
        df     = make_df(avg_amount=4_000_000)
        result = uf.filter({"2330": df}, DATE_TODAY, equity=DEFAULT_EQUITY)
        assert "2330" not in result

    def test_fails_no_52w_breakout(self):
        """High <= High_52W 且 Low >= Low_52W → 沒有突破"""
        uf     = make_filter()
        df     = make_df(high=104.0, low=96.0,
                         prev_high_52w=105.0, prev_low_52w=95.0)
        result = uf.filter({"2330": df}, DATE_TODAY, equity=DEFAULT_EQUITY)
        assert "2330" not in result

    def test_fails_too_expensive(self):
        """股價超出動態單倉上限"""
        # equity=180K → max_price=36；close=37 > 36
        uf     = make_filter(max_positions=5)
        df     = make_df(close=37.0)
        result = uf.filter({"2330": df}, DATE_TODAY, equity=180_000.0)
        assert "2330" not in result

    def test_fails_all_conditions(self):
        """三條件全不符"""
        uf     = make_filter()
        df     = make_df(high=104.0, low=96.0, close=37.0, avg_amount=1_000_000,
                         prev_high_52w=105.0, prev_low_52w=95.0)
        result = uf.filter({"2330": df}, DATE_TODAY, equity=DEFAULT_EQUITY)
        assert "2330" not in result


class TestUniverseFilterEdgeCases:
    """邊界條件與異常情境"""

    def test_no_breakout_when_high_equals_52w_high(self):
        """High == High_52W（非嚴格大於）→ 不算突破"""
        uf     = make_filter()
        df     = make_df(high=105.0, low=96.0,
                         prev_high_52w=105.0, prev_low_52w=95.0)
        result = uf.filter({"2330": df}, DATE_TODAY, equity=DEFAULT_EQUITY)
        assert "2330" not in result

    def test_no_breakout_when_low_equals_52w_low(self):
        """Low == Low_52W（非嚴格小於）→ 不算突破"""
        uf     = make_filter()
        df     = make_df(high=104.0, low=95.0,
                         prev_high_52w=105.0, prev_low_52w=95.0)
        result = uf.filter({"2330": df}, DATE_TODAY, equity=DEFAULT_EQUITY)
        assert "2330" not in result

    def test_no_previous_row_skipped(self):
        """只有一列資料（沒有昨日）→ 無法判斷突破，跳過"""
        uf  = make_filter()
        df  = pd.DataFrame({
            "High": [110.0], "Low": [90.0], "Close": [30.0],
            "Avg_Amount_20": [10_000_000],
            "High_52W": [105.0], "Low_52W": [95.0],
        }, index=[DATE_TODAY])
        result = uf.filter({"2330": df}, DATE_TODAY, equity=DEFAULT_EQUITY)
        assert "2330" not in result

    def test_date_not_in_df_skipped(self):
        """查詢日期不在資料中 → 跳過，不報錯"""
        uf     = make_filter()
        df     = make_df()
        result = uf.filter({"2330": df}, pd.Timestamp("2023-07-01"), equity=DEFAULT_EQUITY)
        assert "2330" not in result

    def test_nan_avg_amount_excluded(self):
        uf     = make_filter()
        df     = make_df(avg_amount=np.nan)
        result = uf.filter({"2330": df}, DATE_TODAY, equity=DEFAULT_EQUITY)
        assert "2330" not in result

    def test_nan_high_52w_and_low_52w_excluded(self):
        """昨日 High_52W 與 Low_52W 都是 NaN → 無法判斷突破 → 排除"""
        uf     = make_filter()
        df     = make_df(prev_high_52w=np.nan, prev_low_52w=np.nan)
        result = uf.filter({"2330": df}, DATE_TODAY, equity=DEFAULT_EQUITY)
        assert "2330" not in result

    def test_empty_data_dict(self):
        uf     = make_filter()
        result = uf.filter({}, DATE_TODAY, equity=DEFAULT_EQUITY)
        assert result == []


class TestUniverseFilterMultipleStocks:
    """多支股票同時篩選"""

    def test_filters_correct_subset(self):
        uf = make_filter()

        df_pass   = make_df(high=110.0, low=96.0, close=30.0)
        df_liq    = make_df(avg_amount=4_000_000)
        df_no_brk = make_df(high=104.0, low=96.0,
                            prev_high_52w=105.0, prev_low_52w=95.0)
        df_exp    = make_df(close=37.0)

        result = uf.filter({
            "2330": df_pass,
            "2317": df_liq,
            "2454": df_no_brk,
            "2412": df_exp,
        }, DATE_TODAY, equity=DEFAULT_EQUITY)

        assert "2330" in result
        assert "2317" not in result
        assert "2454" not in result
        assert "2412" not in result
        assert len(result) == 1

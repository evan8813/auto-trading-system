"""
test_universe_filter.py
───────────────────────
步驟 6：驗證每日股票篩選邏輯。

篩選條件（依 CLAUDE.md 規格）：
  1. 20 日平均成交金額 >= min_avg_amount（流動性）
  2. 今日 High > 昨日 High_52W 或 今日 Low < 昨日 Low_52W（52 週突破）
  3. 收盤 × 1000 <= equity / max_positions（單倉動態上限）
  4. 股價下限：做多 >= min_long_price；做空 >= min_short_price
  5. 大盤環境：加權指數 > 200 日 EMA → 只做多；< 200 日 EMA → 只做空
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
    min_avg_amount:  float = 5_000_000,
    max_positions:   int   = 5,
    min_long_price:  float = 0.0,   # 0 = 不限制（向下相容）
    min_short_price: float = 0.0,
    taiex_csv_path:  str   = "",
) -> UniverseFilter:
    cfg = TradingConfig(
        min_avg_amount=min_avg_amount,
        max_positions=max_positions,
        min_long_price=min_long_price,
        min_short_price=min_short_price,
        taiex_csv_path=taiex_csv_path,
    )
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


# ── 輔助：只觸發多方突破（High 突破，Low 不突破）──────────────────────────

def make_df_long_only(close: float = 30.0) -> pd.DataFrame:
    """只有高點突破（做多候選），低點未跌破"""
    return make_df(high=110.0, low=96.0, close=close,
                   prev_high_52w=105.0, prev_low_52w=95.0)


def make_df_short_only(close: float = 30.0) -> pd.DataFrame:
    """只有低點跌破（做空候選），高點未突破"""
    return make_df(high=104.0, low=88.0, close=close,
                   prev_high_52w=105.0, prev_low_52w=95.0)


class TestPriceFloor:
    """條件 4：股價下限過濾"""

    def test_long_breakout_excluded_when_close_below_min_long_price(self):
        """做多候選，收盤 < min_long_price=10 → 排除"""
        uf = make_filter(min_long_price=10.0)
        df = make_df_long_only(close=9.0)
        assert "2330" not in uf.filter({"2330": df}, DATE_TODAY, equity=DEFAULT_EQUITY)

    def test_long_breakout_passes_at_exactly_min_long_price(self):
        """做多候選，收盤 == min_long_price=10 → 通過（>= 邊界值）"""
        uf = make_filter(min_long_price=10.0)
        df = make_df_long_only(close=10.0)
        assert "2330" in uf.filter({"2330": df}, DATE_TODAY, equity=DEFAULT_EQUITY)

    def test_short_breakout_excluded_when_close_below_min_short_price(self):
        """做空候選，收盤 < min_short_price=20 → 排除"""
        uf = make_filter(min_short_price=20.0)
        df = make_df_short_only(close=15.0)
        assert "2330" not in uf.filter({"2330": df}, DATE_TODAY, equity=DEFAULT_EQUITY)

    def test_short_breakout_passes_at_exactly_min_short_price(self):
        """做空候選，收盤 == min_short_price=20 → 通過（>= 邊界值）"""
        uf = make_filter(min_short_price=20.0)
        df = make_df_short_only(close=20.0)
        assert "2330" in uf.filter({"2330": df}, DATE_TODAY, equity=DEFAULT_EQUITY)

    def test_both_breakout_passes_when_only_long_floor_met(self):
        """雙向突破，收盤介於 min_long_price 和 min_short_price 之間 → 符合做多下限，仍入選"""
        uf = make_filter(min_long_price=10.0, min_short_price=20.0)
        # make_df 預設 high=110>105 且 low=90<95，雙向突破；close=15 >= 10 but < 20
        df = make_df(close=15.0)
        assert "2330" in uf.filter({"2330": df}, DATE_TODAY, equity=DEFAULT_EQUITY)


class TestMarketRegime:
    """條件 5：大盤 200 日 EMA 環境過濾"""

    def _make_taiex_csv(self, tmp_path, close: float, ema: float) -> str:
        """建立一個 TAIEX CSV，讓最後一列的 Close=close 且 EMA_200 可被計算出對應值"""
        # 直接建出 200 列相同數值，EMA(200) 收斂到 close 本身
        n = 250
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        df = pd.DataFrame({
            "Date":   dates,
            "Open":   [ema] * n,
            "High":   [ema] * n,
            "Low":    [ema] * n,
            "Close":  [ema] * (n - 1) + [close],  # 最後一日設為 close
            "Volume": [1_000_000] * n,
        })
        path = tmp_path / "taiex.csv"
        df.to_csv(path, index=False)
        return str(path)

    def test_market_regime_long_only_excludes_short_breakout(self, tmp_path):
        """TAIEX > 200 EMA（多頭）→ 只允許做多，空方突破被排除"""
        csv = self._make_taiex_csv(tmp_path, close=20000.0, ema=18000.0)
        uf  = make_filter(taiex_csv_path=csv)
        # 只有空方突破的股票
        df  = make_df_short_only(close=30.0)
        date = pd.Timestamp("2020-12-31")
        df.index = [date - pd.Timedelta(days=1), date]
        assert "2330" not in uf.filter({"2330": df}, date, equity=DEFAULT_EQUITY)

    def test_market_regime_short_only_excludes_long_breakout(self, tmp_path):
        """TAIEX < 200 EMA（空頭）→ 只允許做空，多方突破被排除"""
        csv = self._make_taiex_csv(tmp_path, close=14000.0, ema=18000.0)
        uf  = make_filter(taiex_csv_path=csv)
        # 只有多方突破的股票
        df  = make_df_long_only(close=30.0)
        date = pd.Timestamp("2020-12-31")
        df.index = [date - pd.Timedelta(days=1), date]
        assert "2330" not in uf.filter({"2330": df}, date, equity=DEFAULT_EQUITY)

    def test_market_regime_long_passes_long_breakout(self, tmp_path):
        """TAIEX > 200 EMA（多頭）→ 多方突破正常入選"""
        csv = self._make_taiex_csv(tmp_path, close=20000.0, ema=18000.0)
        uf  = make_filter(taiex_csv_path=csv)
        df  = make_df_long_only(close=30.0)
        date = pd.Timestamp("2020-12-31")
        df.index = [date - pd.Timedelta(days=1), date]
        assert "2330" in uf.filter({"2330": df}, date, equity=DEFAULT_EQUITY)

    def test_market_regime_disabled_when_no_taiex_path(self):
        """未設定 taiex_csv_path → 大盤過濾停用，多空皆可入選"""
        uf = make_filter(taiex_csv_path="")
        # 空方突破
        df = make_df_short_only(close=30.0)
        assert "2330" in uf.filter({"2330": df}, DATE_TODAY, equity=DEFAULT_EQUITY)

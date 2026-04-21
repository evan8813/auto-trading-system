"""
test_indicators.py
──────────────────
步驟 3：驗證技術指標計算邏輯。

測試項目：
  1. sma()   ── 簡單移動平均數值正確
  2. atr()   ── ATR 計算使用正確的 True Range 公式
  3. rolling_max() / rolling_min() ── 滾動最大/最小值
  4. add_all() ── 附加所有指標後欄位是否齊全
  5. 邊界條件：資料長度不足時出現 NaN
"""

import pytest
import numpy as np
import pandas as pd

from config import TradingConfig
from indicators import Indicators


# ── 共用輔助：建立測試用 OHLCV DataFrame ──

def make_ohlcv(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """建立確定性 OHLCV 資料（用 random seed 保持可重複）"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high  = close + np.abs(rng.normal(0, 0.5, n))
    low   = close - np.abs(rng.normal(0, 0.5, n))
    open_ = close + rng.normal(0, 0.3, n)
    vol   = rng.integers(1000, 10000, n).astype(float)
    amount = vol * close

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low,
         "Close": close, "Volume": vol, "Amount": amount},
        index=dates,
    )


class TestSMA:
    """Simple Moving Average"""

    def test_sma_basic_calculation(self):
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = Indicators.sma(series, window=3)
        # index 2: (1+2+3)/3 = 2.0
        assert result.iloc[2] == pytest.approx(2.0)
        # index 4: (3+4+5)/3 = 4.0
        assert result.iloc[4] == pytest.approx(4.0)

    def test_sma_first_window_minus_1_is_nan(self):
        series = pd.Series([10.0, 20.0, 30.0, 40.0])
        result = Indicators.sma(series, window=3)
        assert pd.isna(result.iloc[0])  # 前 window-1 個應為 NaN
        assert pd.isna(result.iloc[1])

    def test_sma_window_1(self):
        series = pd.Series([5.0, 10.0, 15.0])
        result = Indicators.sma(series, window=1)
        assert list(result) == pytest.approx([5.0, 10.0, 15.0])

    def test_sma_constant_series(self):
        series = pd.Series([7.0] * 10)
        result = Indicators.sma(series, window=5)
        # 常數序列的移動平均仍是常數
        valid = result.dropna()
        assert all(v == pytest.approx(7.0) for v in valid)


class TestATR:
    """Average True Range"""

    def test_atr_returns_series(self):
        df = make_ohlcv(50)
        result = Indicators.atr(df, period=14)
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)

    def test_atr_first_period_minus_1_is_nan(self):
        df = make_ohlcv(50)
        result = Indicators.atr(df, period=14)
        # 前 13 個必定是 NaN（rolling(14).mean() 需要 14 個點）
        assert result.iloc[:13].isna().all()

    def test_atr_non_negative(self):
        df = make_ohlcv(100)
        result = Indicators.atr(df, period=14)
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_atr_true_range_formula(self):
        """手動計算單一 TR 值驗證公式：max(H-L, |H-prev_C|, |L-prev_C|)"""
        df = pd.DataFrame({
            "High":  [105.0, 108.0],
            "Low":   [100.0, 103.0],
            "Close": [104.0, 107.0],
        }, index=pd.bdate_range("2023-01-01", periods=2))

        # 第 2 日 TR = max(108-103, |108-104|, |103-104|) = max(5, 4, 1) = 5
        # ATR with period=1 just equals TR (rolling(1).mean() = TR itself)
        result = Indicators.atr(df, period=1)
        assert result.iloc[1] == pytest.approx(5.0)

    def test_atr_greater_spread_means_higher_atr(self):
        """波動大的資料其 ATR 應高於波動小的資料"""
        n = 50
        dates = pd.bdate_range("2020-01-01", periods=n)
        base = 100.0

        low_vol = pd.DataFrame({
            "High": [base + 0.5] * n,
            "Low":  [base - 0.5] * n,
            "Close": [base] * n,
        }, index=dates)

        high_vol = pd.DataFrame({
            "High": [base + 5.0] * n,
            "Low":  [base - 5.0] * n,
            "Close": [base] * n,
        }, index=dates)

        atr_low  = Indicators.atr(low_vol,  period=14).dropna().mean()
        atr_high = Indicators.atr(high_vol, period=14).dropna().mean()
        assert atr_high > atr_low


class TestRollingMaxMin:
    """rolling_max / rolling_min"""

    def test_rolling_max_basic(self):
        s = pd.Series([1.0, 3.0, 2.0, 5.0, 4.0])
        result = Indicators.rolling_max(s, window=3)
        assert result.iloc[2] == pytest.approx(3.0)  # max(1,3,2)
        assert result.iloc[4] == pytest.approx(5.0)  # max(2,5,4)

    def test_rolling_min_basic(self):
        s = pd.Series([5.0, 3.0, 4.0, 1.0, 2.0])
        result = Indicators.rolling_min(s, window=3)
        assert result.iloc[2] == pytest.approx(3.0)  # min(5,3,4)
        assert result.iloc[4] == pytest.approx(1.0)  # min(4,1,2)

    def test_rolling_max_nan_before_window(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0])
        result = Indicators.rolling_max(s, window=3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])

    def test_rolling_min_nan_before_window(self):
        s = pd.Series([4.0, 3.0, 2.0, 1.0])
        result = Indicators.rolling_min(s, window=3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])

    def test_rolling_max_window_1(self):
        s = pd.Series([9.0, 1.0, 5.0])
        result = Indicators.rolling_max(s, window=1)
        assert list(result) == pytest.approx([9.0, 1.0, 5.0])


class TestAddAll:
    """Indicators.add_all() 整合測試"""

    def setup_method(self):
        self.cfg = TradingConfig()
        self.df  = make_ohlcv(300)

    def test_returns_dataframe(self):
        result = Indicators.add_all(self.df, self.cfg)
        assert isinstance(result, pd.DataFrame)

    def test_original_df_not_modified(self):
        original_cols = list(self.df.columns)
        Indicators.add_all(self.df, self.cfg)
        assert list(self.df.columns) == original_cols

    def test_all_indicator_columns_present(self):
        result = Indicators.add_all(self.df, self.cfg)
        expected = ["ATR", "MA_Fast", "MA_Slow",
                    "High_N", "Low_N", "High_52W", "Low_52W", "Avg_Amount_20"]
        for col in expected:
            assert col in result.columns, f"缺少欄位：{col}"

    def test_row_count_unchanged(self):
        result = Indicators.add_all(self.df, self.cfg)
        assert len(result) == len(self.df)

    def test_ma_fast_uses_correct_window(self):
        result = Indicators.add_all(self.df, self.cfg)
        expected_sma = self.df["Close"].rolling(self.cfg.ma_fast).mean()
        pd.testing.assert_series_equal(
            result["MA_Fast"], expected_sma, check_names=False
        )

    def test_avg_amount_20_uses_amount_column(self):
        """有 Amount 欄時應使用 Amount 計算，而非 Volume × Close"""
        result = Indicators.add_all(self.df, self.cfg)
        expected = self.df["Amount"].rolling(20).mean()
        pd.testing.assert_series_equal(
            result["Avg_Amount_20"], expected, check_names=False
        )

    def test_avg_amount_20_fallback_to_volume_times_close(self):
        """無 Amount 欄時應 fallback 到 Volume × Close"""
        df_no_amount = self.df.drop(columns=["Amount"])
        result = Indicators.add_all(df_no_amount, self.cfg)
        expected = (df_no_amount["Volume"] * df_no_amount["Close"]).rolling(20).mean()
        pd.testing.assert_series_equal(
            result["Avg_Amount_20"], expected, check_names=False
        )

    def test_high_n_equals_rolling_max_close(self):
        result = Indicators.add_all(self.df, self.cfg)
        expected = self.df["Close"].rolling(self.cfg.breakout_window).max()
        pd.testing.assert_series_equal(
            result["High_N"], expected, check_names=False
        )

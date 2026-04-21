"""
test_models.py
──────────────
步驟 2：驗證 Position 與 CorporateEvent 資料結構。

測試項目：
  1. Position 能正確建立並儲存欄位
  2. update_trail() 做多方向只更新 trail_high
  3. update_trail() 做空方向只更新 trail_low
  4. CorporateEvent 能正確建立並儲存欄位
  5. 預設值（dividend_received=0, split_ratio=1）
"""

import pytest
import pandas as pd
from models import Position, CorporateEvent


def make_long_position(**kwargs) -> Position:
    """建立一個做多測試持倉（提供合理預設值）"""
    defaults = dict(
        ticker="2330",
        direction="long",
        entry_date=pd.Timestamp("2023-01-10"),
        lots=2,
        shares=2000,
        adj_entry_price=500.0,
        raw_entry_price=500.0,
        trail_high=510.0,
        trail_low=490.0,
        atr_at_entry=8.0,
    )
    defaults.update(kwargs)
    return Position(**defaults)


def make_short_position(**kwargs) -> Position:
    defaults = dict(
        ticker="2317",
        direction="short",
        entry_date=pd.Timestamp("2023-02-01"),
        lots=1,
        shares=1000,
        adj_entry_price=100.0,
        raw_entry_price=100.0,
        trail_high=105.0,
        trail_low=95.0,
        atr_at_entry=3.0,
    )
    defaults.update(kwargs)
    return Position(**defaults)


class TestPositionCreation:
    """Position 建立與欄位儲存"""

    def test_basic_fields(self):
        pos = make_long_position()
        assert pos.ticker == "2330"
        assert pos.direction == "long"
        assert pos.lots == 2
        assert pos.shares == 2000
        assert pos.adj_entry_price == 500.0
        assert pos.raw_entry_price == 500.0
        assert pos.atr_at_entry == 8.0

    def test_default_dividend_received(self):
        pos = make_long_position()
        assert pos.dividend_received == 0.0

    def test_default_split_ratio(self):
        pos = make_long_position()
        assert pos.split_ratio == 1.0

    def test_entry_date_is_timestamp(self):
        pos = make_long_position()
        assert isinstance(pos.entry_date, pd.Timestamp)


class TestPositionUpdateTrailLong:
    """做多方向的追蹤停損更新"""

    def test_trail_high_updates_when_new_high(self):
        pos = make_long_position(trail_high=510.0)
        pos.update_trail(high=530.0, low=500.0)
        assert pos.trail_high == 530.0

    def test_trail_high_unchanged_when_lower(self):
        pos = make_long_position(trail_high=510.0)
        pos.update_trail(high=505.0, low=490.0)
        assert pos.trail_high == 510.0  # 不應更新

    def test_trail_low_unchanged_for_long(self):
        """做多方向不應更新 trail_low"""
        pos = make_long_position(trail_low=490.0)
        pos.update_trail(high=520.0, low=480.0)
        assert pos.trail_low == 490.0  # 做多不動 trail_low

    def test_multiple_updates_long(self):
        pos = make_long_position(trail_high=500.0)
        pos.update_trail(high=510.0, low=490.0)
        pos.update_trail(high=520.0, low=500.0)
        pos.update_trail(high=515.0, low=505.0)
        assert pos.trail_high == 520.0  # 取歷史最高


class TestPositionUpdateTrailShort:
    """做空方向的追蹤停損更新"""

    def test_trail_low_updates_when_new_low(self):
        pos = make_short_position(trail_low=95.0)
        pos.update_trail(high=100.0, low=88.0)
        assert pos.trail_low == 88.0

    def test_trail_low_unchanged_when_higher(self):
        pos = make_short_position(trail_low=95.0)
        pos.update_trail(high=102.0, low=97.0)
        assert pos.trail_low == 95.0  # 不應更新

    def test_trail_high_unchanged_for_short(self):
        """做空方向不應更新 trail_high"""
        pos = make_short_position(trail_high=105.0)
        pos.update_trail(high=110.0, low=90.0)
        assert pos.trail_high == 105.0  # 做空不動 trail_high

    def test_multiple_updates_short(self):
        pos = make_short_position(trail_low=100.0)
        pos.update_trail(high=102.0, low=92.0)
        pos.update_trail(high=96.0,  low=85.0)
        pos.update_trail(high=90.0,  low=88.0)
        assert pos.trail_low == 85.0  # 取歷史最低


class TestCorporateEventCreation:
    """CorporateEvent 建立與欄位儲存"""

    def test_basic_dividend_event(self):
        e = CorporateEvent(
            ticker="2330",
            event_date=pd.Timestamp("2023-08-28"),
            event_type="dividend",
            cash_dividend=3.5,
        )
        assert e.ticker == "2330"
        assert e.event_type == "dividend"
        assert e.cash_dividend == 3.5

    def test_default_values(self):
        e = CorporateEvent(
            ticker="0050",
            event_date=pd.Timestamp("2023-01-01"),
            event_type="split",
        )
        assert e.cash_dividend == 0.0
        assert e.stock_ratio == 0.0
        assert e.split_ratio == 1.0
        assert e.note == ""

    def test_split_event(self):
        e = CorporateEvent(
            ticker="2454",
            event_date=pd.Timestamp("2023-03-01"),
            event_type="split",
            split_ratio=2.0,
        )
        assert e.split_ratio == 2.0

    def test_stock_dividend_event(self):
        e = CorporateEvent(
            ticker="2412",
            event_date=pd.Timestamp("2023-07-01"),
            event_type="dividend",
            stock_ratio=0.05,
            cash_dividend=1.5,
        )
        assert e.stock_ratio == 0.05
        assert e.cash_dividend == 1.5

"""
test_corporate_action.py
────────────────────────
步驟 7：驗證除權息事件的載入、查詢、套用。

測試項目：
  1. add() 手動新增後可查詢到
  2. get_events() 依日期區間正確篩選
  3. apply_to_position() ── 現金股息累加 dividend_received
  4. apply_to_position() ── 股票分割更新 split_ratio 與 shares
  5. apply_to_position() ── 股票股利增加 shares
  6. 同一日多個事件全部套用
  7. load_csv() 能正確讀取 CSV（使用 tmp_path）
"""

import csv
import pytest
import pandas as pd

from models import CorporateEvent, Position
from corporate_action import CorporateActionLog


# ── 共用輔助 ──────────────────────────────────

def make_position(ticker="2330", shares=1000) -> Position:
    return Position(
        ticker=ticker,
        direction="long",
        entry_date=pd.Timestamp("2023-01-01"),
        lots=1,
        shares=shares,
        adj_entry_price=100.0,
        raw_entry_price=100.0,
        trail_high=110.0,
        trail_low=90.0,
        atr_at_entry=3.0,
    )


def make_dividend_event(ticker="2330", date_str="2023-08-01", cash=2.0) -> CorporateEvent:
    return CorporateEvent(
        ticker=ticker,
        event_date=pd.Timestamp(date_str),
        event_type="dividend",
        cash_dividend=cash,
    )


def make_split_event(ticker="2330", date_str="2023-08-01", ratio=2.0) -> CorporateEvent:
    return CorporateEvent(
        ticker=ticker,
        event_date=pd.Timestamp(date_str),
        event_type="split",
        split_ratio=ratio,
    )


def make_stock_dividend_event(ticker="2330", date_str="2023-08-01", ratio=0.1) -> CorporateEvent:
    return CorporateEvent(
        ticker=ticker,
        event_date=pd.Timestamp(date_str),
        event_type="dividend",
        stock_ratio=ratio,
    )


class TestCorporateActionLogAdd:
    """add() 與 get_events()"""

    def test_add_single_event(self):
        log = CorporateActionLog()
        e   = make_dividend_event()
        log.add(e)
        events = log.get_events("2330", pd.Timestamp("2023-07-01"), pd.Timestamp("2023-09-01"))
        assert len(events) == 1
        assert events[0].cash_dividend == 2.0

    def test_add_multiple_events_same_ticker(self):
        log = CorporateActionLog()
        log.add(make_dividend_event("2330", "2023-06-01", cash=1.0))
        log.add(make_dividend_event("2330", "2023-09-01", cash=1.5))
        events = log.get_events("2330", pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31"))
        assert len(events) == 2

    def test_get_events_date_filter(self):
        log = CorporateActionLog()
        log.add(make_dividend_event("2330", "2023-01-01"))
        log.add(make_dividend_event("2330", "2023-12-01"))
        # 只查詢前半年
        events = log.get_events("2330", pd.Timestamp("2023-01-01"), pd.Timestamp("2023-06-30"))
        assert len(events) == 1

    def test_get_events_different_ticker_not_returned(self):
        log = CorporateActionLog()
        log.add(make_dividend_event("2317"))
        events = log.get_events("2330", pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31"))
        assert len(events) == 0

    def test_empty_log_returns_empty_list(self):
        log    = CorporateActionLog()
        events = log.get_events("2330", pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31"))
        assert events == []


class TestApplyToPositionDividend:
    """現金股息套用"""

    def test_cash_dividend_accumulates(self):
        log = CorporateActionLog()
        log.add(make_dividend_event("2330", "2023-08-01", cash=2.0))
        pos = make_position("2330")
        assert pos.dividend_received == 0.0
        log.apply_to_position(pos, pd.Timestamp("2023-08-01"))
        assert pos.dividend_received == pytest.approx(2.0)

    def test_multiple_dividends_sum_up(self):
        log = CorporateActionLog()
        log.add(make_dividend_event("2330", "2023-08-01", cash=2.0))
        log.add(make_dividend_event("2330", "2023-08-01", cash=1.5))
        pos = make_position("2330")
        log.apply_to_position(pos, pd.Timestamp("2023-08-01"))
        assert pos.dividend_received == pytest.approx(3.5)

    def test_dividend_not_applied_on_wrong_date(self):
        log = CorporateActionLog()
        log.add(make_dividend_event("2330", "2023-08-01", cash=2.0))
        pos = make_position("2330")
        log.apply_to_position(pos, pd.Timestamp("2023-07-31"))  # 前一天
        assert pos.dividend_received == 0.0


class TestApplyToPositionSplit:
    """股票分割套用"""

    def test_split_updates_shares(self):
        log = CorporateActionLog()
        log.add(make_split_event("2330", "2023-08-01", ratio=2.0))
        pos = make_position("2330", shares=1000)
        log.apply_to_position(pos, pd.Timestamp("2023-08-01"))
        assert pos.shares == 2000

    def test_split_updates_split_ratio(self):
        log = CorporateActionLog()
        log.add(make_split_event("2330", "2023-08-01", ratio=2.0))
        pos = make_position("2330")
        log.apply_to_position(pos, pd.Timestamp("2023-08-01"))
        assert pos.split_ratio == pytest.approx(2.0)

    def test_split_ratio_1_no_change(self):
        """split_ratio=1.0 時不應改變 shares"""
        log = CorporateActionLog()
        log.add(CorporateEvent(
            ticker="2330",
            event_date=pd.Timestamp("2023-08-01"),
            event_type="split",
            split_ratio=1.0,
        ))
        pos = make_position("2330", shares=1000)
        log.apply_to_position(pos, pd.Timestamp("2023-08-01"))
        assert pos.shares == 1000  # 未改變


class TestApplyToPositionStockDividend:
    """股票股利套用"""

    def test_stock_dividend_increases_shares(self):
        """stock_ratio=0.1 → 每股配 0.1 股，1000 股 → 1100 股"""
        log = CorporateActionLog()
        log.add(make_stock_dividend_event("2330", "2023-08-01", ratio=0.1))
        pos = make_position("2330", shares=1000)
        log.apply_to_position(pos, pd.Timestamp("2023-08-01"))
        assert pos.shares == 1100

    def test_stock_dividend_zero_ratio_no_change(self):
        log = CorporateActionLog()
        log.add(CorporateEvent(
            ticker="2330",
            event_date=pd.Timestamp("2023-08-01"),
            event_type="dividend",
            stock_ratio=0.0,
            cash_dividend=0.0,
        ))
        pos = make_position("2330", shares=1000)
        log.apply_to_position(pos, pd.Timestamp("2023-08-01"))
        assert pos.shares == 1000


class TestLoadCSV:
    """load_csv() 使用 pytest tmp_path"""

    def test_load_csv_basic(self, tmp_path):
        csv_file = tmp_path / "events.csv"
        csv_file.write_text(
            "ticker,event_date,event_type,cash_dividend,stock_ratio,split_ratio,note\n"
            "2330,2023-08-28,dividend,1.5,0.0,1.0,1H23除息\n"
            "0050,2023-09-01,dividend,2.0,0.0,1.0,半年配\n",
            encoding="utf-8",
        )
        log = CorporateActionLog()
        log.load_csv(str(csv_file))
        events_2330 = log.get_events("2330", pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31"))
        assert len(events_2330) == 1
        assert events_2330[0].cash_dividend == pytest.approx(1.5)

    def test_load_csv_multiple_rows(self, tmp_path):
        csv_file = tmp_path / "events.csv"
        rows = "\n".join([
            "ticker,event_date,event_type,cash_dividend,stock_ratio,split_ratio,note",
            "2330,2023-01-01,dividend,1.0,0.0,1.0,",
            "2330,2023-07-01,dividend,1.5,0.0,1.0,",
            "2317,2023-06-01,split,0.0,0.0,2.0,",
        ])
        csv_file.write_text(rows, encoding="utf-8")
        log = CorporateActionLog()
        log.load_csv(str(csv_file))

        events_2330 = log.get_events("2330", pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31"))
        events_2317 = log.get_events("2317", pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31"))
        assert len(events_2330) == 2
        assert len(events_2317) == 1
        assert events_2317[0].split_ratio == pytest.approx(2.0)

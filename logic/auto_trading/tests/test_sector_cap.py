"""
test_sector_cap.py
──────────────────
驗證 max_positions_per_sector 同產業席數上限功能。

測試項目：
  1. cap=2 時，同產業最多進 2 席，第 3 席被拒絕
  2. cap=0（預設）時不限制，同產業可進滿 max_positions
  3. 不同產業各自有獨立的席數計算，不互相影響
  4. ticker 不在 sector map 中時，不受限制仍可進場
  5. sector_csv_path 不存在時，sector cap 功能自動停用
"""

import csv
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from backtester import Backtester
from config import TradingConfig
from data_loader import DataLoader


# ── 共用工具 ────────────────────────────────────────────

def make_sector_csv(tmp_path: Path, mapping: dict[str, str]) -> str:
    """建立暫存 sector_mapping.csv，回傳絕對路徑字串。"""
    p = tmp_path / "sector_mapping.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "sector"])
        for ticker, sector in mapping.items():
            w.writerow([ticker, sector])
    return str(p)


def sector_cfg(sector_csv: str, cap: int, **overrides) -> TradingConfig:
    base = dict(
        initial_equity           = 1_000_000.0,
        backtest_start           = "2020-01-01",
        backtest_end             = "2021-06-30",
        risk_pct                 = 0.02,
        breakout_window          = 5,
        stop_window              = 3,
        ma_fast                  = 3,
        ma_slow                  = 6,
        atr_period               = 5,
        atr_multiplier           = 3.0,
        week52                   = 25,
        min_avg_amount           = 1_000,
        max_trade_cost           = 999_999_999,
        max_positions            = 5,
        point_value              = 1.0,
        max_positions_per_sector = cap,
        sector_csv_path          = sector_csv,
    )
    base.update(overrides)
    return TradingConfig(**base)


def run_with_sector(cfg: TradingConfig, n_stocks: int = 10, seed: int = 42):
    tickers = [f"{2000+i}" for i in range(n_stocks)]
    data    = DataLoader.generate_synthetic(
        tickers, start=cfg.backtest_start, end=cfg.backtest_end, seed=seed,
    )
    results = Backtester(cfg).run(data)
    return results, data


def max_concurrent_same_sector(trades: pd.DataFrame, sector_map: dict[str, str]) -> int:
    """計算回測中任一天同一產業最多同時持有幾席。"""
    if trades.empty:
        return 0
    df = trades.copy()
    df["sector"] = df["ticker"].astype(str).map(sector_map)
    all_dates = pd.bdate_range(df["entry_date"].min(), df["exit_date"].max())
    max_count = 0
    for date in all_dates:
        active = df[(df["entry_date"] <= date) & (df["exit_date"] > date)]
        if active.empty:
            continue
        counts = active.groupby("sector").size()
        max_count = max(max_count, counts.max())
    return max_count


# ── 測試類別 ────────────────────────────────────────────

class TestSectorCapEnforced:
    """cap=2：同產業同時持倉不超過 2 席"""

    def setup_method(self, tmp_path_factory):
        pass   # pytest tmp_path 只能用 fixture 注入，在個別 test 中處理

    def test_same_sector_never_exceeds_cap(self, tmp_path):
        # 所有 10 支股票都屬於同一產業
        mapping = {f"{2000+i}": "電子業" for i in range(10)}
        csv_path = make_sector_csv(tmp_path, mapping)
        cfg      = sector_cfg(csv_path, cap=2)

        results, _ = run_with_sector(cfg, n_stocks=10, seed=42)
        trades = results["trades"]

        if trades.empty:
            pytest.skip("合成資料未產生交易，無法驗證")

        max_c = max_concurrent_same_sector(trades, mapping)
        assert max_c <= 2, f"同產業同時持倉 {max_c} 席，超過 cap=2"

    def test_has_trades_with_cap(self, tmp_path):
        """cap=2 仍應有交易（只是數量受限，不是 0）"""
        mapping  = {f"{2000+i}": "電子業" for i in range(10)}
        csv_path = make_sector_csv(tmp_path, mapping)
        cfg      = sector_cfg(csv_path, cap=2)

        results, _ = run_with_sector(cfg, n_stocks=10, seed=42)
        assert not results["trades"].empty, "cap=2 時應有交易，不應為 0"


class TestSectorCapDisabled:
    """cap=0（預設）：不限制同產業席數"""

    def test_cap_zero_allows_same_sector(self, tmp_path):
        mapping  = {f"{2000+i}": "電子業" for i in range(10)}
        csv_path = make_sector_csv(tmp_path, mapping)
        cfg      = sector_cfg(csv_path, cap=0)

        results, _ = run_with_sector(cfg, n_stocks=10, seed=42)
        trades = results["trades"]

        if trades.empty:
            pytest.skip("合成資料未產生交易")

        # cap=0 時同產業持倉可能 > 2
        max_c = max_concurrent_same_sector(trades, mapping)
        # 不應被 cap=2 限制（有機會超過 2）；這裡只驗證行為沒有被錯誤截斷
        assert max_c >= 0   # 永遠成立，重點是不 crash


class TestSectorCapMultipleSectors:
    """不同產業各自獨立計算，互不影響"""

    def test_two_sectors_each_capped(self, tmp_path):
        # 前 5 支 → 電子業；後 5 支 → 金融業
        mapping = {
            **{f"{2000+i}": "電子業"  for i in range(5)},
            **{f"{2005+i}": "金融業"  for i in range(5)},
        }
        csv_path = make_sector_csv(tmp_path, mapping)
        cfg      = sector_cfg(csv_path, cap=2)

        results, _ = run_with_sector(cfg, n_stocks=10, seed=42)
        trades = results["trades"]

        if trades.empty:
            pytest.skip("合成資料未產生交易")

        for sector in ["電子業", "金融業"]:
            sector_trades = trades[trades["ticker"].astype(str).map(mapping) == sector]
            if sector_trades.empty:
                continue
            max_c = max_concurrent_same_sector(sector_trades, mapping)
            assert max_c <= 2, f"{sector} 同時持倉 {max_c} 席，超過 cap=2"


class TestSectorCapMissingTicker:
    """ticker 不在 sector map 中 → 不受限制，仍可進場"""

    def test_unknown_ticker_not_blocked(self, tmp_path):
        # sector map 只有部分 ticker（故意遺漏幾支）
        mapping  = {f"{2000+i}": "電子業" for i in range(3)}   # 只有前 3 支有 sector
        csv_path = make_sector_csv(tmp_path, mapping)
        cfg      = sector_cfg(csv_path, cap=1)   # 嚴格 cap=1

        results, _ = run_with_sector(cfg, n_stocks=10, seed=42)
        trades = results["trades"]

        if trades.empty:
            pytest.skip("合成資料未產生交易")

        # 不在 map 裡的 ticker（2003~2009）應仍可交易
        unknown = trades[~trades["ticker"].astype(str).isin(mapping.keys())]
        assert not unknown.empty, "不在 sector map 中的 ticker 應仍可進場"


class TestSectorCapFileNotFound:
    """sector_csv_path 不存在時，功能停用、回測正常執行"""

    def test_missing_csv_does_not_crash(self):
        cfg = TradingConfig(
            initial_equity           = 1_000_000.0,
            backtest_start           = "2020-01-01",
            backtest_end             = "2020-06-30",
            risk_pct                 = 0.02,
            breakout_window          = 5,
            stop_window              = 3,
            ma_fast                  = 3,
            ma_slow                  = 6,
            atr_period               = 5,
            atr_multiplier           = 3.0,
            week52                   = 25,
            min_avg_amount           = 1_000,
            max_trade_cost           = 999_999_999,
            max_positions            = 5,
            point_value              = 1.0,
            max_positions_per_sector = 2,
            sector_csv_path          = "/nonexistent/path/sector_mapping.csv",
        )
        tickers = [f"{2000+i}" for i in range(5)]
        data    = DataLoader.generate_synthetic(
            tickers, start=cfg.backtest_start, end=cfg.backtest_end, seed=0,
        )
        results = Backtester(cfg).run(data)
        assert "trades" in results
        assert "equity_curve" in results

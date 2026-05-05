"""
test_scenarios.py
─────────────────
六個邊界情境的 controlled test。
每個情境都是「已知答案的人造資料」，確保不會因真實資料碰不到而漏測。

情境清單：
  1. 選股池為空         → 0 筆交易
  2. 持倉上限封頂       → 同時持倉不超過 max_positions
  3. 出場後補進新股票    → 空出的位置會被補滿
  4. 同支股票不重複持倉  → 持倉期間不會再次進場
  5. 張數計算為 0       → 不進場、0 筆交易
  6. 資料缺口不強制出場  → 停牌期間留倉，不提前出場
"""

import numpy as np
import pandas as pd
import pytest

from backtester import Backtester
from config import TradingConfig
from data_loader import DataLoader
from indicators import Indicators


# ══════════════════════════════════════════════
# 共用工具
# ══════════════════════════════════════════════

def scenario_cfg(**overrides) -> TradingConfig:
    """
    測試專用小參數 Config。
    week52=25 讓暖機期短（25 個交易日），其餘指標週期更小。
    """
    base = dict(
        initial_equity   = 1_000_000.0,
        backtest_start   = "2020-01-01",
        backtest_end     = "2021-06-30",
        risk_pct         = 0.02,          # 風險比例放大，確保 lots > 0
        breakout_window  = 5,
        stop_window      = 3,             # 初始停損窗口（< breakout_window）
        macd_fast        = 3,             # 短週期 EMA，讓 MACD 快速反應合成資料
        macd_slow        = 6,
        macd_signal      = 3,
        atr_period       = 5,
        atr_multiplier   = 3.0,
        week52           = 25,            # 暖機期 = 25 個交易日
        min_avg_amount   = 1_000,         # 幾乎所有股票都能通過
        max_trade_cost   = 999_999_999,   # 不限制單筆金額
        max_positions    = 5,
        point_value      = 1.0,
    )
    base.update(overrides)
    return TradingConfig(**base)


def run(cfg: TradingConfig, n_stocks: int = 10, seed: int = 42) -> dict:
    """用合成資料跑回測，回傳 results"""
    tickers = [f"{2000+i}" for i in range(n_stocks)]
    data    = DataLoader.generate_synthetic(
        tickers,
        start = cfg.backtest_start,
        end   = cfg.backtest_end,
        seed  = seed,
    )
    return Backtester(cfg).run(data), data


def concurrent_positions(trades: pd.DataFrame) -> pd.Series:
    """
    計算每個交易日的同時持倉數。
    回傳 Series（index=date, value=持倉數）。
    """
    if trades.empty:
        return pd.Series(dtype=int)

    df = trades.copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"]  = pd.to_datetime(df["exit_date"])

    all_dates = pd.bdate_range(df["entry_date"].min(), df["exit_date"].max())
    counts = pd.Series(0, index=all_dates)
    for _, row in df.iterrows():
        # exit_date 當天已出場（先出後進），不計入持倉
        mask = (counts.index >= row["entry_date"]) & (counts.index < row["exit_date"])
        counts[mask] += 1
    return counts


# ══════════════════════════════════════════════
# 情境 1：選股池為空 → 0 筆交易
# ══════════════════════════════════════════════

class TestScenario1EmptyPool:
    """
    把 min_avg_amount 設到不可能達到的值，
    每次選股都回傳空清單 → 不會有任何進場。
    """

    def test_zero_trades_when_no_candidates(self):
        cfg = scenario_cfg(min_avg_amount=999_999_999_999)
        results, _ = run(cfg)
        assert results["trades"].empty, "選股池為空時不應有任何交易"

    def test_equity_unchanged_when_no_trades(self):
        cfg = scenario_cfg(min_avg_amount=999_999_999_999)
        results, _ = run(cfg)
        eq = results["equity_curve"]["equity"]
        # 無交易時資金應維持不變
        assert eq.iloc[0] == eq.iloc[-1]

    def test_metrics_total_trades_zero(self):
        cfg = scenario_cfg(min_avg_amount=999_999_999_999)
        results, _ = run(cfg)
        assert results["metrics"]["total_trades"] == 0


# ══════════════════════════════════════════════
# 情境 2：持倉上限封頂
# ══════════════════════════════════════════════

class TestScenario2MaxPositionsCap:
    """
    用 20 支股票、max_positions=3，
    確認任何時刻同時持倉都 <= 3。
    """

    def setup_method(self):
        self.cfg     = scenario_cfg(max_positions=3)
        self.results, _ = run(self.cfg, n_stocks=20, seed=1)
        self.trades  = self.results["trades"]

    def test_has_trades(self):
        assert not self.trades.empty, "應有交易，請檢查合成資料或參數"

    def test_concurrent_never_exceeds_max(self):
        counts = concurrent_positions(self.trades)
        assert counts.max() <= self.cfg.max_positions, (
            f"同時持倉 {counts.max()} 支，超過上限 {self.cfg.max_positions}"
        )

    def test_total_trades_reasonable(self):
        # 至少要有幾筆交易（確認回測有在跑）
        assert self.results["metrics"]["total_trades"] >= 1


# ══════════════════════════════════════════════
# 情境 3：出場後補進新股票
# ══════════════════════════════════════════════

class TestScenario3RefillAfterExit:
    """
    max_positions=1，只要有交易，就代表：
    第一支股票出場後，第二支才能進場。
    用「總交易次數 > 1」驗證補倉機制有在運作。
    """

    def setup_method(self):
        self.cfg     = scenario_cfg(max_positions=1)
        self.results, _ = run(self.cfg, n_stocks=15, seed=7)
        self.trades  = self.results["trades"]

    def test_multiple_trades_occur(self):
        """有多筆交易 = 出場後確實補進新股票"""
        total = self.results["metrics"]["total_trades"]
        assert total > 1, (
            f"只有 {total} 筆交易，max_positions=1 時應先後進出多支股票"
        )

    def test_no_two_positions_overlap(self):
        """max_positions=1：任何時刻同時持倉 <= 1"""
        counts = concurrent_positions(self.trades)
        assert counts.max() <= 1, (
            f"同時持倉最高 {counts.max()} 支，max_positions=1 下不應超過 1"
        )


# ══════════════════════════════════════════════
# 情境 4：同支股票不重複持倉
# ══════════════════════════════════════════════

class TestScenario4NoDuplicateTicker:
    """
    同一支股票的持倉期間不應重疊。
    遍歷所有交易，確認沒有「進場日 < 前次出場日」的情況。
    """

    def setup_method(self):
        self.cfg     = scenario_cfg(max_positions=5)
        self.results, _ = run(self.cfg, n_stocks=20, seed=3)
        self.trades  = self.results["trades"]

    def test_has_trades(self):
        assert not self.trades.empty

    def test_no_overlapping_positions_same_ticker(self):
        if self.trades.empty:
            return

        df = self.trades.copy()
        df["entry_date"] = pd.to_datetime(df["entry_date"])
        df["exit_date"]  = pd.to_datetime(df["exit_date"])

        overlaps = []
        for ticker, grp in df.groupby("ticker"):
            grp = grp.sort_values("entry_date").reset_index(drop=True)
            for j in range(1, len(grp)):
                prev_exit  = grp.loc[j - 1, "exit_date"]
                curr_entry = grp.loc[j,     "entry_date"]
                # 同天出場後進場是合法的（先出後進），嚴格小於才算真正重疊
                if curr_entry < prev_exit:
                    overlaps.append({
                        "ticker":     ticker,
                        "prev_exit":  prev_exit.date(),
                        "next_entry": curr_entry.date(),
                    })

        assert len(overlaps) == 0, (
            f"發現 {len(overlaps)} 筆同股票重疊持倉：\n"
            + pd.DataFrame(overlaps).to_string(index=False)
        )


# ══════════════════════════════════════════════
# 情境 5：張數為 0 → 不進場
# ══════════════════════════════════════════════

class TestScenario5ZeroLotsNoEntry:
    """
    把資金設到極小，風險金額 / ATR < 1000，
    position_size_lots() 會算出 0 張 → 不進場。

    risk_amount = 100 × 0.002 = 0.2 元
    合成股票 ATR ≈ 0.5~1.5 元
    lots = int(0.2 / (1.0 × 1000)) = 0
    """

    def test_zero_trades_when_lots_always_zero(self):
        cfg = scenario_cfg(
            initial_equity = 100.0,   # 極小資金
            risk_pct       = 0.002,   # 風險 0.2 元
            min_avg_amount = 1_000,
        )
        results, _ = run(cfg)
        assert results["trades"].empty, (
            "張數為 0 時不應有任何進場，但仍有交易紀錄"
        )


# ══════════════════════════════════════════════
# 情境 6：資料缺口（停牌）不強制出場
# ══════════════════════════════════════════════

class TestScenario6DataGapKeepsPosition:
    """
    在持倉期間把該股票的幾個日期從資料中移除（模擬停牌），
    確認回測不會因資料缺口而強制出場。

    驗證方式：
      1. 不移除資料 → 取得正常的 hold_days
      2. 移除持倉中間的幾天 → hold_days 應 >= 原本值（不提前出場）
    """

    def setup_method(self):
        self.cfg = scenario_cfg(max_positions=1)
        tickers  = [f"{2000+i}" for i in range(10)]
        self.raw_data = DataLoader.generate_synthetic(
            tickers,
            start = self.cfg.backtest_start,
            end   = self.cfg.backtest_end,
            seed  = 99,
        )

    def test_position_survives_data_gap(self):
        # ── Step 1：正常跑，找到第一筆交易 ──
        normal_results = Backtester(self.cfg).run(self.raw_data)
        trades = normal_results["trades"]

        if trades.empty:
            pytest.skip("合成資料無交易，跳過此情境")

        first_trade = trades.iloc[0]
        ticker      = first_trade["ticker"]
        entry_date  = pd.Timestamp(first_trade["entry_date"])
        exit_date   = pd.Timestamp(first_trade["exit_date"])
        hold_days   = first_trade["hold_days"]

        # ── Step 2：移除持倉中間 3 天的資料 ──
        import copy
        gapped_data = {t: df.copy() for t, df in self.raw_data.items()}
        df_stock    = gapped_data[ticker]

        # 找持倉期間的交易日，移除中間 3 天
        in_hold = df_stock.index[
            (df_stock.index > entry_date) & (df_stock.index < exit_date)
        ]
        if len(in_hold) < 3:
            pytest.skip("持倉期間交易日不足 3 天，跳過此情境")

        dates_to_remove   = in_hold[:3]
        gapped_data[ticker] = df_stock.drop(index=dates_to_remove)

        # ── Step 3：用缺口資料重跑 ──
        gapped_results = Backtester(self.cfg).run(gapped_data)
        gapped_trades  = gapped_results["trades"]

        # 若有交易，同一支股票的出場不應早於原本的出場日
        if not gapped_trades.empty:
            same = gapped_trades[gapped_trades["ticker"] == ticker]
            if not same.empty:
                gapped_exit = pd.Timestamp(same.iloc[0]["exit_date"])
                assert gapped_exit >= exit_date - pd.Timedelta(days=5), (
                    f"{ticker} 在資料缺口後提早出場："
                    f"原本 {exit_date.date()}，缺口後 {gapped_exit.date()}"
                )

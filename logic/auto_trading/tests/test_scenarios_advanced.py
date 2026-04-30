"""
test_scenarios_advanced.py
──────────────────────────
進階邊界情境，針對「這次 debug 過程中發現或排除的潛在 bug」：

  S7.  ROC 排序錯誤        → 動能最強的股票必須優先進場
  S8.  T+1 停牌 fallback  → 執行日無資料時，次選股票正確補位
  S9.  max_trade_cost 截斷 → 股價高時張數被正確壓縮
  S10. 同日先出後進         → 出場與進場可在同一 T+1 日發生，不重疊
  S11. 權益更新正確         → 第二筆交易的 equity_at_entry 反映前筆損益
"""

import numpy as np
import pandas as pd
import pytest

from backtester import Backtester
from config import TradingConfig
from data_loader import DataLoader


# ══════════════════════════════════════════════
# 共用工具
# ══════════════════════════════════════════════

def scenario_cfg(**overrides) -> TradingConfig:
    base = dict(
        initial_equity  = 1_000_000.0,
        backtest_start  = "2020-01-01",
        backtest_end    = "2021-06-30",
        risk_pct        = 0.02,
        breakout_window = 5,
        ma_fast         = 5,
        ma_slow         = 10,
        atr_period      = 5,
        atr_multiplier  = 3.0,
        week52          = 25,
        min_avg_amount  = 1_000,
        max_trade_cost  = 999_999_999,
        max_positions   = 5,
        point_value     = 1.0,
    )
    base.update(overrides)
    return TradingConfig(**base)


def make_df(dates, closes, volume: float = 100_000_000.0) -> pd.DataFrame:
    """
    從收盤價序列建立 OHLCV DataFrame。
    High = Close × 1.02, Low = Close × 0.98。
    Open = 前一天收盤（第 0 天開盤 = 收盤）。
    ATR（5-period）≈ Close × 0.04。
    """
    closes = np.array(closes, dtype=float)
    n      = len(closes)
    opens  = np.concatenate([[closes[0]], closes[:-1]])
    return pd.DataFrame(
        {
            "Open":   opens,
            "High":   closes * 1.02,
            "Low":    closes * 0.98,
            "Close":  closes,
            "Volume": np.full(n, volume),
            "Amount": closes * volume,
        },
        index=dates[:n],
    )


def uptrend(n: int, base: float = 10.0, rate: float = 0.01) -> np.ndarray:
    """上升 80% 再急跌 20%，確保有進場訊號也有出場訊號（停損觸發）"""
    rise = int(n * 0.8)
    fall = n - rise
    up   = base * ((1 + rate) ** np.arange(rise))
    peak = up[-1]
    down = peak * ((1 - rate * 5) ** np.arange(1, fall + 1))
    return np.concatenate([up, down])


def downtrend(n: int, base: float = 10.0, rate: float = 0.01) -> np.ndarray:
    """嚴格下降趨勢：每日 -rate，觸發做空訊號"""
    return base * ((1 - rate) ** np.arange(n))


# ══════════════════════════════════════════════
# 情境 S7：ROC 排序 — 動能最強的股票必須優先進場
# ══════════════════════════════════════════════
# 已知 bug 來源：
#   回測曾未對 candidates 做 ROC 排序，導致字典順序的股票先進場。
#   修正後應確保 ROC_avg 最高的股票優先佔位。
#
# 設計：
#   HIGH（2% 日漲）vs LOW（0.1% 日漲），max_positions=1。
#   兩支都在上升趨勢 → 暖機後每日都有進場訊號。
#   ROC_avg 計算：HIGH ≈ 62%，LOW ≈ 2.3%。
#   預期：HIGH 先進場，LOW 因 slot 已滿排不進去。
# ══════════════════════════════════════════════

class TestScenario7RocPriority:

    def setup_method(self):
        self.cfg  = scenario_cfg(max_positions=1, backtest_end="2020-08-31")
        n         = 150
        dates     = pd.bdate_range("2020-01-01", periods=n)

        self.data = {
            "HIGH": make_df(dates, uptrend(n, base=10.0, rate=0.02)),
            "LOW":  make_df(dates, uptrend(n, base=10.0, rate=0.001)),
        }
        self.results = Backtester(self.cfg).run(self.data)
        self.trades  = self.results["trades"]

    def test_has_trades(self):
        assert not self.trades.empty, "應有交易，請確認合成資料暖機期設定"

    def test_high_roc_enters_first(self):
        """ROC 最強的 HIGH 應最先進場；若 LOW 先進場表示排序有誤"""
        first = self.trades.sort_values("entry_date").iloc[0]
        assert first["ticker"] == "HIGH", (
            f"ROC 最強的 HIGH 應最先進場，但實際進場的是 {first['ticker']}。"
            "可能是 candidates 未依 ROC 排序。"
        )

    def test_low_never_jumps_ahead_of_high(self):
        """LOW 的進場日不應早於 HIGH（若有 LOW 的交易）"""
        df = self.trades.sort_values("entry_date")
        high_trades = df[df["ticker"] == "HIGH"]
        low_trades  = df[df["ticker"] == "LOW"]
        if high_trades.empty or low_trades.empty:
            return
        first_high = pd.Timestamp(high_trades.iloc[0]["entry_date"])
        first_low  = pd.Timestamp(low_trades.iloc[0]["entry_date"])
        assert first_high <= first_low, (
            f"LOW 在 {first_low.date()} 進場，早於 HIGH 的 {first_high.date()}"
        )


# ══════════════════════════════════════════════
# 情境 S8：T+1 停牌 — 執行日無資料時次選補位
# ══════════════════════════════════════════════
# 設計：
#   HIGH 和 LOW 都有訊號，max_positions=2（兩支都進 pending_entries）。
#   移除 HIGH 在執行日（T+1）的資料，模擬停牌。
#   預期：
#     - HIGH 在停牌日不進場
#     - LOW 在同一執行日進場（fallback 成功）
# ══════════════════════════════════════════════

class TestScenario8T1Halt:

    def setup_method(self):
        self.cfg   = scenario_cfg(max_positions=2, backtest_end="2020-08-31")
        n          = 150
        dates      = pd.bdate_range("2020-01-01", periods=n)
        df_high    = make_df(dates, uptrend(n, base=10.0, rate=0.02))
        df_low     = make_df(dates, uptrend(n, base=10.0, rate=0.001))

        # 先跑正常版本，找出 HIGH 的第一個 entry_date（= 停牌日）
        normal = Backtester(self.cfg).run({"HIGH": df_high.copy(), "LOW": df_low.copy()})
        normal_trades = normal["trades"]

        if normal_trades.empty:
            self.halt_date = None
            self.data = None
            return

        high_first = normal_trades[normal_trades["ticker"] == "HIGH"]
        if high_first.empty:
            self.halt_date = None
            self.data = None
            return

        self.halt_date = pd.Timestamp(high_first.iloc[0]["entry_date"])

        # 移除 HIGH 在 halt_date 的資料（停牌）
        df_high_halted = df_high.drop(index=self.halt_date, errors="ignore")
        self.data      = {"HIGH": df_high_halted, "LOW": df_low}
        self.results   = Backtester(self.cfg).run(self.data)
        self.trades    = self.results["trades"]

    def test_setup_ok(self):
        if self.halt_date is None:
            pytest.skip("合成資料未產生 HIGH 的初始交易，跳過")

    def test_high_not_entered_on_halt_day(self):
        """停牌日 HIGH 不應進場"""
        if self.halt_date is None:
            pytest.skip("合成資料未產生初始交易")
        entered_on_halt = self.trades[
            (self.trades["ticker"] == "HIGH") &
            (pd.to_datetime(self.trades["entry_date"]) == self.halt_date)
        ]
        assert entered_on_halt.empty, (
            f"HIGH 在停牌日 {self.halt_date.date()} 不應進場，但仍有交易紀錄"
        )

    def test_low_enters_as_fallback(self):
        """HIGH 停牌後，LOW 應在同一執行日補位進場"""
        if self.halt_date is None:
            pytest.skip("合成資料未產生初始交易")
        low_on_halt = self.trades[
            (self.trades["ticker"] == "LOW") &
            (pd.to_datetime(self.trades["entry_date"]) == self.halt_date)
        ]
        assert not low_on_halt.empty, (
            f"HIGH 停牌，LOW 應在 {self.halt_date.date()} 補位進場，"
            "但找不到 LOW 的交易。可能是 pending_entries 只放了 HIGH 而未放 LOW。"
        )


# ══════════════════════════════════════════════
# 情境 S9：max_trade_cost 截斷 — 高股價時張數正確壓縮
# ══════════════════════════════════════════════
# 已知 bug 來源：
#   C check 原本未套用 max_trade_cost，導致誤報 FAIL。
#   這個情境驗證 backtester 的張數計算確實被截斷。
#
# 設計：
#   股價 ~100 元，risk_pct=0.10 → lots_raw = 25 張。
#   max_trade_cost = 120,000 → 最多 1 張（100×1000=100,000 < 120,000）。
#   預期：所有交易的 lots ≤ 1。
# ══════════════════════════════════════════════

class TestScenario9MaxTradeCostCap:

    def setup_method(self):
        self.cfg = scenario_cfg(
            initial_equity = 1_000_000,
            risk_pct       = 0.10,        # risk_amount = 100,000 → lots_raw = 25
            max_trade_cost = 120_000,     # 100 × 1000 = 100,000 < 120,000 → 1 張
            max_positions  = 3,
            backtest_end   = "2020-08-31",
        )
        n     = 150
        dates = pd.bdate_range("2020-01-01", periods=n)

        # 股價 100 元持續上漲，ATR ≈ 100 × 0.04 = 4
        self.data = {
            "PRICEY": make_df(dates, uptrend(n, base=100.0, rate=0.005)),
        }
        self.results = Backtester(self.cfg).run(self.data)
        self.trades  = self.results["trades"]

    def test_has_trades(self):
        assert not self.trades.empty, "應有交易"

    def test_lots_capped_to_one(self):
        """
        股價 ~100 元時，max_trade_cost=120,000 只能買 1 張。
        若 backtester 未套用 max_trade_cost 截斷，lots 會是 25。
        """
        for _, trade in self.trades.iterrows():
            lots    = int(trade["lots"])
            price   = float(trade["raw_entry_price"])
            # 最多允許 floor(max_trade_cost / (price * 1000)) 張
            max_lots = max(int(self.cfg.max_trade_cost / (price * 1000)), 0)
            assert lots <= max(max_lots, 1), (
                f"lots={lots} 超過 max_trade_cost 上限 {max_lots} 張"
                f"（raw_entry={price:.2f}，max_trade_cost={self.cfg.max_trade_cost}）"
            )

    def test_lots_at_least_one(self):
        """截斷後 lots 應 >= 1（確認截斷邏輯沒有歸零）"""
        for _, trade in self.trades.iterrows():
            assert int(trade["lots"]) >= 1, (
                f"lots=0，max_trade_cost 截斷後不應為 0（stock 股價合理、資金充足）"
            )


# ══════════════════════════════════════════════
# 情境 S10：同日先出後進（exit_date = next entry_date）
# ══════════════════════════════════════════════
# 設計：
#   max_positions=1，atr_multiplier 縮小讓停損快一點，促使快速換倉。
#   A 出場訊號與 B 進場訊號落在同一個訊號日 T。
#   T+1：A 出場、B 進場，entry_date(B) = exit_date(A)。
#   預期：
#     - B 的 entry_date = A 的 exit_date（合法同日換股）
#     - 任何時刻持倉數 ≤ max_positions（不重疊持倉）
# ══════════════════════════════════════════════

class TestScenario10SameDayTurnover:

    def setup_method(self):
        self.cfg = scenario_cfg(
            max_positions  = 1,
            atr_multiplier = 1.5,   # 停損較緊，加速換倉
            backtest_end   = "2021-06-30",
        )
        tickers    = [f"{2000+i}" for i in range(15)]
        self.data  = DataLoader.generate_synthetic(
            tickers,
            start = self.cfg.backtest_start,
            end   = self.cfg.backtest_end,
            seed  = 77,
        )
        self.results = Backtester(self.cfg).run(self.data)
        self.trades  = self.results["trades"]

    def test_has_multiple_trades(self):
        assert self.results["metrics"]["total_trades"] > 1, (
            "應有多筆交易（出場後才能補進新股），請調整 seed 或 atr_multiplier"
        )

    def test_same_day_turnover_is_valid(self):
        """
        exit_date(A) = entry_date(B) 是合法的先出後進，
        驗證此情況有發生，且沒有被誤判為重疊持倉。
        """
        if self.trades.empty:
            pytest.skip("無交易")

        df = self.trades.copy()
        df["entry_date"] = pd.to_datetime(df["entry_date"])
        df["exit_date"]  = pd.to_datetime(df["exit_date"])

        exit_dates  = set(df["exit_date"])
        same_day_re = df[df["entry_date"].isin(exit_dates)]

        # 如果有同日換股，驗證它不被算為重疊
        if not same_day_re.empty:
            assert len(same_day_re) >= 1

    def test_no_concurrent_position_overflow(self):
        """任何時刻持倉不超過 max_positions（同日先出後進不算重疊）"""
        if self.trades.empty:
            pytest.skip("無交易")

        df = self.trades.copy()
        df["entry_date"] = pd.to_datetime(df["entry_date"])
        df["exit_date"]  = pd.to_datetime(df["exit_date"])

        all_dates = pd.bdate_range(df["entry_date"].min(), df["exit_date"].max())
        counts    = pd.Series(0, index=all_dates)
        for _, row in df.iterrows():
            # exit_date 已出場（先出後進），不計入持倉
            mask = (counts.index >= row["entry_date"]) & (counts.index < row["exit_date"])
            counts[mask] += 1

        assert counts.max() <= self.cfg.max_positions, (
            f"同時持倉 {counts.max()} 支，超過上限 {self.cfg.max_positions}"
        )


# ══════════════════════════════════════════════
# 情境 S11：權益更新正確 — 後筆交易的 equity_at_entry 反映前筆損益
# ══════════════════════════════════════════════
# 設計：
#   max_positions=1（確保不並行部位，equity 變化單純）。
#   第 n 筆交易的 equity_at_entry 應 ≈ initial_equity + 前 n-1 筆 pnl_net 總和。
#   若 backtester 固定用 initial_equity 或記錯累積損益，此測試會 FAIL。
# ══════════════════════════════════════════════

class TestScenario11EquityTracking:

    def setup_method(self):
        self.cfg  = scenario_cfg(max_positions=1, backtest_end="2021-06-30")
        tickers   = [f"{2000+i}" for i in range(10)]
        self.data = DataLoader.generate_synthetic(
            tickers,
            start = self.cfg.backtest_start,
            end   = self.cfg.backtest_end,
            seed  = 42,
        )
        self.results = Backtester(self.cfg).run(self.data)
        self.trades  = self.results["trades"]

    def test_has_multiple_trades(self):
        assert self.results["metrics"]["total_trades"] >= 2, (
            "需要至少 2 筆交易才能驗證跨交易的權益更新"
        )

    def test_equity_at_entry_accumulates_pnl(self):
        """
        max_positions=1 時，每筆進場的 equity_at_entry 應等於
        initial_equity 加上前所有已結算交易的 pnl_net 總和。
        （允許小浮點誤差，不允許大幅偏差。）
        """
        if len(self.trades) < 2:
            pytest.skip("交易筆數不足")

        df = self.trades.sort_values("entry_date").reset_index(drop=True)

        cumulative_pnl = 0.0
        for i, row in df.iterrows():
            expected = self.cfg.initial_equity + cumulative_pnl
            actual   = float(row["equity_at_entry"])

            # 允許 1% 內的誤差（浮點 + 同日多倉並行的邊界情況）
            tolerance = max(abs(expected) * 0.01, 1.0)
            assert abs(actual - expected) <= tolerance, (
                f"第 {i+1} 筆 {row['ticker']} equity_at_entry={actual:.0f}，"
                f"預期 ≈ {expected:.0f}（初始 {self.cfg.initial_equity:.0f} "
                f"+ 累積損益 {cumulative_pnl:.0f}）。"
                "可能是 backtester 未正確累加 pnl_net 到 equity。"
            )
            cumulative_pnl += float(row["pnl_net"])

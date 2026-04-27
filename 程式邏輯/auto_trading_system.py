"""
自動化交易系統 (Auto Trading System)
依據 auto_trading_system_spec_v2.md 實作

策略邏輯：
- 股票篩選：52週新高附近 / 成交量門檻
- 進場：收盤突破50日高 + MA50 > MA100（做多）
- 出場：收盤 < 最高價 - 3×ATR（做多停損）
- 部位大小：Risk Amount / (ATR × 每點價值)

資料格式（TWSE CSV）：
  欄位：日期,成交股數,成交金額,開盤價,最高價,最低價,收盤價,漲跌價差,成交筆數
  檔名：{代碼}.csv，例如 2330.csv

除權息處理說明：
  - 回測：使用「還原股價（調整後收盤價）」計算所有指標與報酬，
          避免因除權息造成假突破 / 假跌破。
  - 實盤：Position 同時記錄「原始進場價」與「調整後進場價」，
          出場時以「原始出場價」計算實際損益，
          並透過 CorporateActionLog 記錄每筆除息 / 除權事件，
          讓你可以在損益報表中還原真實的經濟報酬。
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # 無 GUI 環境也能存圖
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 可選：SHIOAJI (永豐金 API) 用於實盤
# ─────────────────────────────────────────────
try:
    import shioaji as sj
    SHIOAJI_AVAILABLE = True
except ImportError:
    SHIOAJI_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# 1. 參數設定
# ══════════════════════════════════════════════

@dataclass
class TradingConfig:
    """使用者可調整的交易系統參數"""

    # ── 資金設定 ──
    initial_equity: float = 1_000_000.0      # 初始資金（新台幣）
    risk_pct: float = 0.002                  # 每筆風險比例（預設 0.2%）

    # ── 交易成本 ──
    commission_rate: float = 0.001425        # 手續費率（買賣各）
    transaction_tax: float = 0.003           # 證交稅（賣出時）
    slippage: float = 0.001                  # 滑價比例

    # ── 指標參數 ──
    breakout_window: int = 50               # 突破 N 日高 / 低
    ma_fast: int = 50                       # 快線均線
    ma_slow: int = 100                      # 慢線均線
    atr_period: int = 14                    # ATR 週期
    atr_multiplier: float = 3.0             # ATR 停損倍數
    week52: int = 252                       # 52 週交易日

    # ── 篩選條件 ──
    min_avg_amount: float = 5_000_000       # 最低 20 日平均成交金額（元）

    # ── 回測區間 ──
    backtest_start: str = "2010-01-01"
    backtest_end: str = "2023-12-29"

    # ── 部位限制 ──
    max_positions: int = 10                 # 最大同時持倉數
    point_value: float = 1.0               # 每點價值（股票 = 1）


# ══════════════════════════════════════════════
# 2. 資料載入（TWSE CSV 格式）
# ══════════════════════════════════════════════

class DataLoader:
    """
    載入 TWSE 歷史 OHLCV 資料

    CSV 欄位格式（與你的資料相符）：
      日期, 成交股數, 成交金額, 開盤價, 最高價, 最低價, 收盤價, 漲跌價差, 成交筆數

    輸出標準化欄位：
      Open, High, Low, Close, Volume（成交股數）, Amount（成交金額）

    關於「還原股價」：
      - 若資料夾內有同名的 {ticker}_adj.csv（還原後），優先載入。
      - 否則直接使用原始資料（無除權息調整）。
      - 建議先用外部工具（tquant / ffn / 自行計算）產生 *_adj.csv 以提升回測準確度。
    """

    # TWSE CSV 原始欄位 → 標準化名稱
    COL_MAP = {
        "日期":    "Date",
        "成交股數": "Volume",
        "成交金額": "Amount",
        "開盤價":  "Open",
        "最高價":  "High",
        "最低價":  "Low",
        "收盤價":  "Close",
        "漲跌價差": "Change",
        "成交筆數": "Transactions",
    }

    @classmethod
    def load_folder(
        cls,
        folder: str,
        tickers: Optional[list[str]] = None,
        adjusted: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        從資料夾批次載入所有（或指定）股票的 CSV。

        Parameters
        ----------
        folder   : CSV 資料夾路徑
        tickers  : 若為 None，自動掃描資料夾內所有 *.csv
        adjusted : True = 優先使用 {ticker}_adj.csv（調整後）；
                   若不存在則 fallback 到原始 CSV。
                   False = 強制使用原始 CSV。
        """
        folder_path = Path(folder)
        if not folder_path.exists():
            raise FileNotFoundError(f"資料夾不存在：{folder}")

        if tickers is None:
            # 自動掃描，排除 _adj 檔案
            tickers = [
                p.stem for p in sorted(folder_path.glob("*.csv"))
                if not p.stem.endswith("_adj")
            ]
            logger.info(f"自動掃描到 {len(tickers)} 支股票。")

        data: dict[str, pd.DataFrame] = {}
        skipped = 0

        for ticker in tickers:
            df = cls._load_one(folder_path, ticker, adjusted)
            if df is not None and len(df) >= 120:   # 至少要有足夠資料計算指標
                data[ticker] = df
            else:
                skipped += 1

        logger.info(
            f"成功載入 {len(data)} 支股票資料"
            f"（跳過 {skipped} 支資料不足或格式錯誤）。"
        )
        return data

    @classmethod
    def _load_one(
        cls,
        folder: Path,
        ticker: str,
        prefer_adjusted: bool,
    ) -> Optional[pd.DataFrame]:
        """載入單一股票 CSV，回傳標準化 DataFrame 或 None。"""

        # 決定要讀的檔案（優先 adjusted）
        adj_path = folder / f"{ticker}_adj.csv"
        raw_path = folder / f"{ticker}.csv"

        if prefer_adjusted and adj_path.exists():
            path   = adj_path
            is_adj = True
        elif raw_path.exists():
            path   = raw_path
            is_adj = False
        else:
            logger.debug(f"找不到 {ticker}.csv，略過。")
            return None

        try:
            df = pd.read_csv(path, dtype=str)

            # ── 欄位重命名 ──
            df.rename(columns=cls.COL_MAP, inplace=True)

            # ── 必要欄位檢查 ──
            required = ["Date", "Open", "High", "Low", "Close", "Volume"]
            missing  = [c for c in required if c not in df.columns]
            if missing:
                logger.warning(f"{ticker}: 缺少欄位 {missing}，略過。")
                return None

            # ── 日期解析 ──
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df.dropna(subset=["Date"], inplace=True)
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)

            # ── 數值欄位轉型（含去除千分位逗號 / 特殊值） ──
            INVALID = {"--", "除息", "除權", "除權息", ""}
            num_cols = ["Open", "High", "Low", "Close", "Volume", "Amount"]
            for col in num_cols:
                if col not in df.columns:
                    continue
                df[col] = (
                    df[col]
                    .str.replace(",", "", regex=False)
                    .str.strip()
                    .apply(lambda x: np.nan if x in INVALID else x)
                    .pipe(pd.to_numeric, errors="coerce")
                )

            # ── 過濾停牌 / 下市資料行 ──
            df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)

            # ── 確保 High >= Low（資料品質保護）──
            df = df[df["High"] >= df["Low"]]

            df.attrs["ticker"]      = ticker
            df.attrs["is_adjusted"] = is_adj
            return df

        except Exception as e:
            logger.warning(f"{ticker}: 載入失敗（{e}），略過。")
            return None

    @staticmethod
    def generate_synthetic(
        tickers: list[str],
        start: str = "2010-01-01",
        end:   str = "2023-12-29",
        seed:  int = 42,
    ) -> dict[str, pd.DataFrame]:
        """
        產生合成資料（用於單元測試 / CI，不代表真實市場）。
        無 CSV 時可用此函式驗證策略邏輯。
        """
        dates = pd.bdate_range(start, end)
        data: dict[str, pd.DataFrame] = {}

        for i, ticker in enumerate(tickers):
            rng     = np.random.default_rng(seed + i)
            n       = len(dates)
            log_ret = rng.normal(0.0003, 0.015, n)
            close   = 50.0 * np.exp(np.cumsum(log_ret))
            high    = close * (1 + np.abs(rng.normal(0, 0.006, n)))
            low     = close * (1 - np.abs(rng.normal(0, 0.006, n)))
            open_   = close * (1 + rng.normal(0, 0.004, n))
            amount  = rng.integers(5_000_000, 100_000_000, n).astype(float)
            volume  = (amount / close).astype(int).astype(float)

            df = pd.DataFrame(
                {"Open": open_, "High": high, "Low": low,
                 "Close": close, "Volume": volume, "Amount": amount},
                index=dates,
            )
            df.attrs["ticker"]      = ticker
            df.attrs["is_adjusted"] = True
            data[ticker] = df

        logger.info(f"已產生 {len(data)} 支合成股票資料（{start} ~ {end}）。")
        return data


# ══════════════════════════════════════════════
# 3. 技術指標計算
# ══════════════════════════════════════════════

class Indicators:
    """在 DataFrame 上計算並附加所有技術指標"""

    @staticmethod
    def atr(df: pd.DataFrame, period: int) -> pd.Series:
        high, low, close = df["High"], df["Low"], df["Close"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def add_all(df: pd.DataFrame, cfg: TradingConfig) -> pd.DataFrame:
        df = df.copy()
        c  = df["Close"]

        df["ATR"]      = Indicators.atr(df, cfg.atr_period)
        df["MA_Fast"]  = c.rolling(cfg.ma_fast).mean()
        df["MA_Slow"]  = c.rolling(cfg.ma_slow).mean()
        df["High_N"]   = c.rolling(cfg.breakout_window).max()
        df["Low_N"]    = c.rolling(cfg.breakout_window).min()
        df["High_52W"] = df["High"].rolling(cfg.week52).max()
        df["Low_52W"]  = df["Low"].rolling(cfg.week52).min()

        # 成交金額（若有 Amount 欄使用之，否則估算）
        if "Amount" in df.columns:
            df["Avg_Amount_20"] = df["Amount"].rolling(20).mean()
        else:
            df["Avg_Amount_20"] = (df["Volume"] * df["Close"]).rolling(20).mean()

        return df


# ══════════════════════════════════════════════
# 4. 股票篩選
# ══════════════════════════════════════════════

class UniverseFilter:
    def __init__(self, cfg: TradingConfig):
        self.cfg = cfg

    def filter(
        self,
        data_dict: dict[str, pd.DataFrame],
        date:      pd.Timestamp,
    ) -> list[str]:
        """
        回傳在指定日期符合篩選條件的股票代號。

        條件：
        1. 20 日平均成交金額 >= min_avg_amount
        2. 今日最高價 > 52 週最高價（突破新高）
           或 今日最低價 < 52 週最低價（突破新低）
        """
        candidates = []
        for ticker, df in data_dict.items():
            if date not in df.index:
                continue
            idx = df.index.get_loc(date)
            if idx == 0:
                continue
            row      = df.iloc[idx]
            prev_row = df.iloc[idx - 1]
            if any(pd.isna([row.get("Avg_Amount_20"), prev_row.get("High_52W"), prev_row.get("Low_52W")])):
                continue
            amount_ok    = row["Avg_Amount_20"] >= self.cfg.min_avg_amount
            new_52w_high = row["High"] > prev_row["High_52W"]
            new_52w_low  = row["Low"]  < prev_row["Low_52W"]
            if amount_ok and (new_52w_high or new_52w_low):
                candidates.append(ticker)
        return candidates


# ══════════════════════════════════════════════
# 5. 訊號產生
# ══════════════════════════════════════════════

class SignalGenerator:

    @staticmethod
    def long_entry(row: pd.Series, prev_row: pd.Series) -> bool:
        """做多進場：當日收盤突破前日 N 日高，且 MA_Fast > MA_Slow"""
        if any(pd.isna([row["Close"], prev_row["High_N"],
                        row["MA_Fast"], row["MA_Slow"]])):
            return False
        return (row["Close"] > prev_row["High_N"]) and (row["MA_Fast"] > row["MA_Slow"])

    @staticmethod
    def long_exit(row: pd.Series, trail_high: float, atr_mult: float) -> bool:
        """做多出場：收盤 < 追蹤最高價 - atr_mult × ATR"""
        if pd.isna(row["ATR"]) or pd.isna(row["Close"]):
            return False
        return row["Close"] < trail_high - atr_mult * row["ATR"]

    @staticmethod
    def short_entry(row: pd.Series, prev_row: pd.Series) -> bool:
        """做空進場：當日收盤跌破前日 N 日低，且 MA_Fast < MA_Slow"""
        if any(pd.isna([row["Close"], prev_row["Low_N"],
                        row["MA_Fast"], row["MA_Slow"]])):
            return False
        return (row["Close"] < prev_row["Low_N"]) and (row["MA_Fast"] < row["MA_Slow"])

    @staticmethod
    def short_exit(row: pd.Series, trail_low: float, atr_mult: float) -> bool:
        """做空出場：收盤 > 追蹤最低價 + atr_mult × ATR"""
        if pd.isna(row["ATR"]) or pd.isna(row["Close"]):
            return False
        return row["Close"] > trail_low + atr_mult * row["ATR"]


# ══════════════════════════════════════════════
# 6. 風險控管
# ══════════════════════════════════════════════

class RiskManager:
    def __init__(self, cfg: TradingConfig):
        self.cfg = cfg

    def risk_amount(self, equity: float) -> float:
        return equity * self.cfg.risk_pct

    def position_size_lots(self, equity: float, atr: float) -> int:
        """回傳「張數」（1 張 = 1000 股），最小 0 張"""
        if not (atr > 0):
            return 0
        raw  = self.risk_amount(equity) / (atr * self.cfg.point_value)
        return max(int(raw / 1000), 0)

    def transaction_cost(self, price: float, shares: int, side: str) -> float:
        """計算單邊手續費 + 稅 + 滑價"""
        notional      = price * shares
        commission    = notional * self.cfg.commission_rate
        slippage_cost = notional * self.cfg.slippage
        tax           = notional * self.cfg.transaction_tax if side == "sell" else 0.0
        return commission + slippage_cost + tax


# ══════════════════════════════════════════════
# 7. 持倉記錄
# ══════════════════════════════════════════════

@dataclass
class Position:
    """
    一筆持倉，同時記錄調整後價格（回測用）與原始市場價格（實盤對帳用）。
    """
    ticker:          str
    direction:       str             # "long" | "short"
    entry_date:      pd.Timestamp
    lots:            int             # 張數
    shares:          int             # 股數 = lots × 1000

    # ── 回測使用（還原股價）──
    adj_entry_price: float           # 調整後進場價（回測損益計算基準）

    # ── 實盤使用（原始市場價格）──
    raw_entry_price: float           # 原始進場價（Shioaji 成交均價）

    # ── 追蹤停損 ──
    trail_high:      float           # 持倉中累計最高價（做多停損基準）
    trail_low:       float           # 持倉中累計最低價（做空停損基準）
    atr_at_entry:    float

    # ── 除權息事件累計（實盤用）──
    dividend_received: float = 0.0   # 持倉期間累計現金股息（元 / 股）
    split_ratio:       float = 1.0   # 累計股票分割比（> 1 = 股本膨脹）

    def update_trail(self, high: float, low: float) -> None:
        if self.direction == "long":
            self.trail_high = max(self.trail_high, high)
        else:
            self.trail_low  = min(self.trail_low, low)


# ══════════════════════════════════════════════
# 8. 除權息事件記錄（實盤用）
# ══════════════════════════════════════════════

@dataclass
class CorporateEvent:
    """
    記錄一筆除權息事件。

    cash_dividend : 現金股息（元 / 股），除息日後入帳。
    stock_ratio   : 股票股利比（例如 0.1 = 每股配 0.1 股）。
    split_ratio   : 股票分割比（例如 2.0 = 1 股變 2 股）。

    真實損益計算：
      total_pnl = 價差損益 + cash_dividend × shares_held + 股利市值
    """
    ticker:        str
    event_date:    pd.Timestamp
    event_type:    str              # "dividend" | "split" | "rights"
    cash_dividend: float = 0.0     # 元 / 股
    stock_ratio:   float = 0.0     # 股 / 股
    split_ratio:   float = 1.0     # 分割比
    note:          str  = ""


class CorporateActionLog:
    """
    管理所有除權息事件，在實盤持倉更新時自動套用。

    CSV 格式（corporate_actions.csv）：
      ticker, event_date, event_type, cash_dividend, stock_ratio, split_ratio, note

    使用方式：
      ca_log = CorporateActionLog()
      ca_log.load_csv("corporate_actions.csv")
      # 每日盤前呼叫：
      for pos in positions:
          ca_log.apply_to_position(pos, today)
    """

    def __init__(self) -> None:
        self._events: list[CorporateEvent] = []

    def add(self, event: CorporateEvent) -> None:
        self._events.append(event)

    def load_csv(self, path: str) -> None:
        """從 CSV 批次載入除權息事件"""
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._events.append(CorporateEvent(
                    ticker        = row["ticker"].strip(),
                    event_date    = pd.Timestamp(row["event_date"].strip()),
                    event_type    = row.get("event_type", "dividend").strip(),
                    cash_dividend = float(row.get("cash_dividend") or 0),
                    stock_ratio   = float(row.get("stock_ratio")   or 0),
                    split_ratio   = float(row.get("split_ratio")   or 1),
                    note          = row.get("note", "").strip(),
                ))
        logger.info(f"載入 {len(self._events)} 筆除權息事件（{path}）。")

    def get_events(
        self,
        ticker:    str,
        from_date: pd.Timestamp,
        to_date:   pd.Timestamp,
    ) -> list[CorporateEvent]:
        return [
            e for e in self._events
            if e.ticker == ticker and from_date <= e.event_date <= to_date
        ]

    def apply_to_position(
        self,
        pos:   Position,
        as_of: pd.Timestamp,
    ) -> None:
        """
        將 as_of 日期當天的除權息事件套用到持倉。
          - 現金股息 → 累計到 pos.dividend_received（元 / 股）
          - 股票分割 → 更新 pos.split_ratio 與 pos.shares
          - 股票股利 → 累加持股數（市場慣例，股東取得股票而非現金）
        """
        for e in self.get_events(pos.ticker, as_of, as_of):
            if e.cash_dividend > 0:
                pos.dividend_received += e.cash_dividend
                logger.info(
                    f"[除息] {pos.ticker} {as_of.date()}"
                    f"  現金股息 +{e.cash_dividend:.4f} 元/股"
                )
            if e.split_ratio != 1.0:
                pos.split_ratio *= e.split_ratio
                pos.shares       = int(pos.shares * e.split_ratio)
                logger.info(
                    f"[分割] {pos.ticker} {as_of.date()}"
                    f"  分割比 {e.split_ratio}  持股 → {pos.shares} 股"
                )
            if e.stock_ratio > 0:
                extra      = int(pos.shares * e.stock_ratio)
                pos.shares += extra
                logger.info(
                    f"[股利] {pos.ticker} {as_of.date()}"
                    f"  股票股利 +{extra} 股  持股 → {pos.shares} 股"
                )


# ══════════════════════════════════════════════
# 9. 回測引擎
# ══════════════════════════════════════════════

class Backtester:
    """
    逐日回測引擎（使用調整後股價）

    時序規則：
      T   日收盤 → 判斷訊號
      T+1 日開盤 → 執行進出場
    """

    def __init__(self, cfg: TradingConfig) -> None:
        self.cfg      = cfg
        self.risk_mgr = RiskManager(cfg)
        self.sig_gen  = SignalGenerator()
        self.uni_flt  = UniverseFilter(cfg)

    # ── 主入口 ────────────────────────────────

    def run(self, raw_data: dict[str, pd.DataFrame]) -> dict:
        """
        Parameters
        ----------
        raw_data : dict  key = 股票代號, value = OHLCV DataFrame

        Returns
        -------
        dict with keys: equity_curve, trades, metrics
        """
        data = {t: Indicators.add_all(df, self.cfg) for t, df in raw_data.items()}

        start     = pd.Timestamp(self.cfg.backtest_start)
        end       = pd.Timestamp(self.cfg.backtest_end)
        all_dates = sorted({d for df in data.values() for d in df.index
                            if start <= d <= end})

        equity         = self.cfg.initial_equity
        positions:     list[Position] = []
        closed_trades: list[dict]    = []
        equity_curve:  list[dict]    = []

        for i, date in enumerate(all_dates):
            if i == 0:
                equity_curve.append({"date": date, "equity": equity})
                continue

            prev_date = all_dates[i - 1]

            # ── (A) 出場檢查 ──
            positions, exited = self._check_exits(positions, data, date)
            for t in exited:
                equity += t["pnl_net"]
                closed_trades.append(t)

            # ── (B) 篩選 + 進場 ──
            candidates   = self.uni_flt.filter(data, date)
            held_tickers = {p.ticker for p in positions}

            if len(positions) < self.cfg.max_positions:
                new_pos = self._check_entries(
                    candidates, data, date, prev_date, equity, held_tickers
                )
                positions.extend(new_pos)

            equity_curve.append({"date": date, "equity": equity})

        metrics = self._compute_metrics(equity_curve, closed_trades)
        trades_df = pd.DataFrame(closed_trades) if closed_trades else pd.DataFrame()

        return {
            "equity_curve": pd.DataFrame(equity_curve).set_index("date"),
            "trades":       trades_df,
            "metrics":      metrics,
        }

    # ── 出場 ──────────────────────────────────

    def _check_exits(
        self,
        positions: list[Position],
        data:      dict[str, pd.DataFrame],
        date:      pd.Timestamp,
    ) -> tuple[list[Position], list[dict]]:
        remaining, exited = [], []

        for pos in positions:
            df = data.get(pos.ticker)
            if df is None or date not in df.index:
                remaining.append(pos)
                continue

            row = df.loc[date]
            pos.update_trail(row["High"], row["Low"])

            if pos.direction == "long":
                should_exit = self.sig_gen.long_exit(
                    row, pos.trail_high, self.cfg.atr_multiplier)
            else:
                should_exit = self.sig_gen.short_exit(
                    row, pos.trail_low, self.cfg.atr_multiplier)

            if should_exit:
                exit_px = row["Open"] if not pd.isna(row["Open"]) else row["Close"]
                trade   = self._close_position(pos, date, exit_px, "signal")
                exited.append(trade)
                logger.debug(
                    f"EXIT  {pos.ticker} {pos.direction} @ {exit_px:.2f}"
                    f"  PnL={trade['pnl_net']:+.0f}"
                )
            else:
                remaining.append(pos)

        return remaining, exited

    def _close_position(
        self,
        pos:        Position,
        exit_date:  pd.Timestamp,
        exit_price: float,
        reason:     str,
    ) -> dict:
        """計算一筆平倉的損益並回傳 trade dict（回測版，使用調整後價格）"""
        cfg = self.cfg

        # 滑價調整
        if pos.direction == "long":
            actual_exit = exit_price * (1 - cfg.slippage)
        else:
            actual_exit = exit_price * (1 + cfg.slippage)

        # 毛利（調整後）
        if pos.direction == "long":
            gross_pnl = (actual_exit - pos.adj_entry_price) * pos.shares
        else:
            gross_pnl = (pos.adj_entry_price - actual_exit) * pos.shares

        # 交易成本（進場 + 出場）
        cost_entry = self.risk_mgr.transaction_cost(pos.adj_entry_price, pos.shares, "buy")
        cost_exit  = self.risk_mgr.transaction_cost(actual_exit, pos.shares, "sell")
        total_cost = cost_entry + cost_exit
        pnl_net    = gross_pnl - total_cost

        return {
            # ── 基本資訊 ──
            "ticker":          pos.ticker,
            "direction":       pos.direction,
            "lots":            pos.lots,
            "shares":          pos.shares,
            # ── 時間 ──
            "entry_date":      pos.entry_date,
            "exit_date":       exit_date,
            "hold_days":       (exit_date - pos.entry_date).days,
            # ── 調整後價格（回測損益依此計算）──
            "adj_entry_price": round(pos.adj_entry_price, 4),
            "adj_exit_price":  round(actual_exit, 4),
            # ── 原始市場價格（實盤對帳 / 報稅用）──
            #    回測中 raw = adj（沒有調整差異），實盤由 LiveTrader 覆寫
            "raw_entry_price": round(pos.raw_entry_price, 4),
            "raw_exit_price":  round(exit_price, 4),
            # ── 損益 ──
            "gross_pnl":       round(gross_pnl, 2),
            "total_cost":      round(total_cost, 2),
            "pnl_net":         round(pnl_net, 2),
            # ── ATR / 風控 ──
            "atr_at_entry":    round(pos.atr_at_entry, 4),
            # ── 出場原因 ──
            "exit_reason":     reason,
        }

    # ── 進場 ──────────────────────────────────

    def _check_entries(
        self,
        candidates:   list[str],
        data:         dict[str, pd.DataFrame],
        date:         pd.Timestamp,
        prev_date:    pd.Timestamp,
        equity:       float,
        held_tickers: set[str],
    ) -> list[Position]:
        new_positions = []

        for ticker in candidates:
            if ticker in held_tickers:
                continue
            if len(held_tickers) + len(new_positions) >= self.cfg.max_positions:
                break

            df = data.get(ticker)
            if df is None or date not in df.index or prev_date not in df.index:
                continue

            row      = df.loc[date]
            prev_row = df.loc[prev_date]
            atr      = row["ATR"]

            if pd.isna(atr) or atr <= 0:
                continue

            direction = None
            if self.sig_gen.long_entry(row, prev_row):
                direction = "long"
            elif self.sig_gen.short_entry(row, prev_row):
                direction = "short"

            if direction is None:
                continue

            lots = self.risk_mgr.position_size_lots(equity, atr)
            if lots == 0:
                continue

            shares      = lots * 1000
            entry_price = row["Open"]
            if pd.isna(entry_price):
                continue

            if direction == "long":
                adj_entry = entry_price * (1 + self.cfg.slippage)
            else:
                adj_entry = entry_price * (1 - self.cfg.slippage)

            pos = Position(
                ticker          = ticker,
                direction       = direction,
                entry_date      = date,
                lots            = lots,
                shares          = shares,
                adj_entry_price = adj_entry,
                raw_entry_price = entry_price,   # 回測中 raw = adj
                trail_high      = row["High"],
                trail_low       = row["Low"],
                atr_at_entry    = atr,
            )
            new_positions.append(pos)
            held_tickers.add(ticker)
            logger.debug(
                f"ENTRY {ticker} {direction.upper()} @ {adj_entry:.2f}"
                f"  lots={lots}  ATR={atr:.2f}"
            )

        return new_positions

    # ── 績效指標 ──────────────────────────────

    def _compute_metrics(
        self,
        equity_curve: list[dict],
        trades:       list[dict],
    ) -> dict:
        ec = pd.DataFrame(equity_curve).set_index("date")["equity"]

        total_return = (ec.iloc[-1] / ec.iloc[0] - 1) * 100

        roll_max = ec.cummax()
        drawdown = (ec - roll_max) / roll_max
        mdd      = drawdown.min() * 100

        n_years = (ec.index[-1] - ec.index[0]).days / 365.25
        cagr    = ((ec.iloc[-1] / ec.iloc[0]) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

        win_rate = avg_win = avg_loss = profit_factor = 0.0

        if trades:
            df_t          = pd.DataFrame(trades)
            wins          = df_t.loc[df_t["pnl_net"] > 0,  "pnl_net"]
            losses        = df_t.loc[df_t["pnl_net"] <= 0, "pnl_net"]
            win_rate      = len(wins) / len(df_t) * 100
            avg_win       = wins.mean()   if len(wins)   else 0.0
            avg_loss      = losses.mean() if len(losses) else 0.0
            profit_factor = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")

        return {
            "total_return_pct": round(total_return, 2),
            "cagr_pct":         round(cagr, 2),
            "max_drawdown_pct": round(mdd, 2),
            "win_rate_pct":     round(win_rate, 2),
            "total_trades":     len(trades),
            "avg_win":          round(avg_win, 0),
            "avg_loss":         round(avg_loss, 0),
            "profit_factor":    round(profit_factor, 3),
        }


# ══════════════════════════════════════════════
# 10. 報表與視覺化
# ══════════════════════════════════════════════

class Reporter:

    @staticmethod
    def print_metrics(metrics: dict) -> None:
        print("\n" + "═" * 48)
        print("  📊  回測績效報告")
        print("═" * 48)
        print(f"  總報酬率        : {metrics['total_return_pct']:>9.2f} %")
        print(f"  年化報酬 (CAGR) : {metrics['cagr_pct']:>9.2f} %")
        print(f"  最大回撤 (MDD)  : {metrics['max_drawdown_pct']:>9.2f} %")
        print(f"  勝率            : {metrics['win_rate_pct']:>9.2f} %")
        print(f"  總交易次數      : {metrics['total_trades']:>9d}")
        print(f"  平均獲利 / 筆   : {metrics['avg_win']:>9.0f} 元")
        print(f"  平均虧損 / 筆   : {metrics['avg_loss']:>9.0f} 元")
        print(f"  獲利因子        : {metrics['profit_factor']:>9.3f}")
        print("═" * 48 + "\n")

    @staticmethod
    def save_trade_log(trades: pd.DataFrame, path: str = "trade_log.csv") -> None:
        """
        輸出每筆交易紀錄 CSV。

        欄位說明：
          adj_entry/exit_price  → 調整後股價（回測損益依此計算）
          raw_entry/exit_price  → 原始市場成交價（實盤對帳 / 報稅用）
          pnl_net               → 扣除手續費、稅、滑價後的淨損益
          dividend_received     → 實盤持倉期間累計現金股息（元/股）[僅實盤有值]
          dividend_income       → 實盤現金股息總收入（元）[僅實盤有值]
        """
        if trades.empty:
            logger.warning("無交易紀錄可輸出。")
            return
        trades.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  交易紀錄已儲存：{path}  （共 {len(trades)} 筆）")

    @staticmethod
    def plot_equity_curve(
        equity_curve: pd.DataFrame,
        save_path: Optional[str] = "equity_curve.png",
    ) -> None:
        fig, axes = plt.subplots(
            2, 1, figsize=(14, 8), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        fig.suptitle("自動化交易系統 — Equity Curve", fontsize=14, fontweight="bold")

        ec       = equity_curve["equity"]
        roll_max = ec.cummax()
        drawdown = (ec - roll_max) / roll_max * 100

        axes[0].plot(ec.index, ec.values, color="#1565C0", linewidth=1.5, label="Equity")
        axes[0].fill_between(ec.index, ec.values, alpha=0.08, color="#1565C0")
        axes[0].set_ylabel("資產淨值（元）")
        axes[0].yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x / 1_000_000:.1f}M"))
        axes[0].legend(loc="upper left")
        axes[0].grid(alpha=0.3)

        axes[1].fill_between(
            ec.index, drawdown.values, color="#C62828", alpha=0.6, label="Drawdown")
        axes[1].set_ylabel("回撤 (%)")
        axes[1].set_xlabel("日期")
        axes[1].legend(loc="lower left")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Equity Curve 已儲存：{save_path}")
        else:
            plt.show()
        plt.close()


# ══════════════════════════════════════════════
# 11. 實盤自動交易（Shioaji）
# ══════════════════════════════════════════════

class LiveTrader:
    """
    實盤交易介面，搭配永豐金 Shioaji API。

    ┌─────────────────────────────────────────────────────┐
    │  除權息在實盤的處理方式                                │
    ├─────────────────────────────────────────────────────┤
    │  問題：                                               │
    │    台股除息日，股價會向下跳空（理論上等於股息金額），     │
    │    若以原始股價計算停損，除息當天很容易誤觸停損出場。     │
    │                                                       │
    │  建議做法（本系統採用）：                               │
    │    1. 每日盤前呼叫 apply_corporate_actions()，         │
    │       自動更新 pos.dividend_received（現金股息）        │
    │       與 pos.shares（股票股利 / 分割）。               │
    │    2. 停損判斷使用「還原後股價」（已扣除股息影響），      │
    │       避免除息日誤觸停損。                             │
    │    3. 出場紀錄同時保留：                               │
    │         raw_entry_price / raw_exit_price（真實成交價） │
    │         dividend_income（持倉期間現金股息收入）         │
    │       → 合計才是你的真實經濟報酬與申報稅務的依據。       │
    │                                                       │
    │  補充：若你的歷史資料已是還原後股價，可直接用調整後的      │
    │  停損水位，則不需特別處理 trail_high 的除息跳空問題。    │
    └─────────────────────────────────────────────────────┘

    安裝：pip install shioaji
    """

    def __init__(
        self,
        cfg:        TradingConfig,
        api_key:    str,
        secret_key: str,
        ca_path:    str = "",
        ca_passwd:  str = "",
        sim:        bool = True,
        ca_log:     Optional[CorporateActionLog] = None,
    ) -> None:
        if not SHIOAJI_AVAILABLE:
            raise ImportError("請先安裝 shioaji：pip install shioaji")

        self.cfg      = cfg
        self.risk_mgr = RiskManager(cfg)
        self.ca_log   = ca_log or CorporateActionLog()

        self.api = sj.Shioaji(simulation=sim)
        accounts = self.api.login(api_key=api_key, secret_key=secret_key)

        if ca_path:
            self.api.activate_ca(
                ca_path    = ca_path,
                ca_passwd  = ca_passwd,
                person_id  = accounts[0].person_id,
            )

        logger.info(f"Shioaji 登入成功（simulation={sim}）")

    # ── 下單 ──────────────────────────────────

    def place_order(
        self,
        ticker:    str,
        direction: str,   # "long"（買進）| "short"（賣出）
        lots:      int,
        price:     float,
    ) -> object:
        """送出市價 ROD 委託，回傳 Shioaji Trade 物件"""
        action   = sj.constant.Action.Buy if direction == "long" else sj.constant.Action.Sell
        contract = self.api.Contracts.Stocks[ticker]
        order    = self.api.Order(
            price      = price,
            quantity   = lots,
            action     = action,
            price_type = sj.constant.StockPriceType.MKT,
            order_type = sj.constant.OrderType.ROD,
        )
        trade = self.api.place_order(contract, order)
        logger.info(
            f"委託送出：{ticker} {direction.upper()} {lots}張"
            f" @ {price:.2f}  (sim={self.api.simulation})"
        )
        return trade

    # ── 除權息套用 ─────────────────────────────

    def apply_corporate_actions(
        self,
        positions: list[Position],
        as_of:     pd.Timestamp,
    ) -> None:
        """
        每日開盤前呼叫，自動套用 as_of 當日的除權息事件到所有持倉。
        副作用：更新 pos.dividend_received / pos.shares / pos.split_ratio。
        """
        for pos in positions:
            self.ca_log.apply_to_position(pos, as_of)

    # ── 監控停損 ───────────────────────────────

    def monitor_and_exit(
        self,
        positions:   list[Position],
        latest_data: dict[str, pd.DataFrame],
        today:       pd.Timestamp,
    ) -> list[Position]:
        """
        掃描所有持倉，觸發停損時自動送出出場委託。
        回傳剩餘持倉（已出場者移除）。

        注意：實際成交價由 Shioaji 回報（callback），
              請在 on_order_cb 中呼叫 build_live_trade_record() 補上 raw_exit_price。
        """
        sig_gen   = SignalGenerator()
        remaining = []

        for pos in positions:
            df = latest_data.get(pos.ticker)
            if df is None or today not in df.index:
                remaining.append(pos)
                continue

            row = df.loc[today]
            pos.update_trail(row["High"], row["Low"])

            if pos.direction == "long":
                should_exit = sig_gen.long_exit(
                    row, pos.trail_high, self.cfg.atr_multiplier)
            else:
                should_exit = sig_gen.short_exit(
                    row, pos.trail_low, self.cfg.atr_multiplier)

            if should_exit:
                logger.warning(
                    f"⚠️  停損觸發：{pos.ticker} {pos.direction}"
                    f"  trail_high={pos.trail_high:.2f} / trail_low={pos.trail_low:.2f}"
                )
                exit_dir = "short" if pos.direction == "long" else "long"
                self.place_order(pos.ticker, exit_dir, pos.lots, row["Close"])
                # 不加入 remaining → 視為已出場
            else:
                remaining.append(pos)

        return remaining

    # ── 建立實盤交易紀錄 ──────────────────────

    def build_live_trade_record(
        self,
        pos:            Position,
        exit_date:      pd.Timestamp,
        raw_exit_price: float,         # 從 Shioaji 成交回報取得的實際成交均價
        exit_reason:    str = "signal",
    ) -> dict:
        """
        建立完整的實盤交易紀錄。

        損益計算邏輯：
          raw_pnl       = 價差損益（原始成交價差 × 股數）
          dividend_income = 持倉期間現金股息收入（元/股 × 股數）
          pnl_net       = raw_pnl + dividend_income − 手續費 − 稅 − 滑價

        注意：股票股利的市值（已更新到 pos.shares 中）不計入此欄，
              但 split_ratio / shares 欄位已反映分割後的實際股數。
        """
        cfg = self.cfg

        if pos.direction == "long":
            raw_pnl = (raw_exit_price - pos.raw_entry_price) * pos.shares
        else:
            raw_pnl = (pos.raw_entry_price - raw_exit_price) * pos.shares

        cost = (
            self.risk_mgr.transaction_cost(pos.raw_entry_price, pos.shares, "buy")
            + self.risk_mgr.transaction_cost(raw_exit_price, pos.shares, "sell")
        )

        dividend_income = pos.dividend_received * pos.shares
        pnl_net         = raw_pnl - cost + dividend_income

        return {
            # ── 基本 ──
            "ticker":                      pos.ticker,
            "direction":                   pos.direction,
            "lots":                        pos.lots,
            "shares":                      pos.shares,
            # ── 時間 ──
            "entry_date":                  pos.entry_date,
            "exit_date":                   exit_date,
            "hold_days":                   (exit_date - pos.entry_date).days,
            # ── 原始市場價格（報稅 / 對帳用）──
            "raw_entry_price":             round(pos.raw_entry_price, 4),
            "raw_exit_price":              round(raw_exit_price, 4),
            # ── 回測參考價（僅供比對）──
            "adj_entry_price":             round(pos.adj_entry_price, 4),
            # ── 除權息明細 ──
            "dividend_received_per_share": round(pos.dividend_received, 4),
            "dividend_income":             round(dividend_income, 2),
            "split_ratio":                 round(pos.split_ratio, 6),
            # ── 損益 ──
            "raw_pnl":                     round(raw_pnl, 2),
            "total_cost":                  round(cost, 2),
            "pnl_net":                     round(pnl_net, 2),
            # ── 其他 ──
            "atr_at_entry":                round(pos.atr_at_entry, 4),
            "exit_reason":                 exit_reason,
        }

    def logout(self) -> None:
        self.api.logout()
        logger.info("Shioaji 已登出。")


# ══════════════════════════════════════════════
# 12. 主程式入口
# ══════════════════════════════════════════════

def run_backtest(
    data_folder: Optional[str] = None,
    tickers:     Optional[list[str]] = None,
    cfg:         Optional[TradingConfig] = None,
    output_dir:  str = "output",
) -> dict:
    """
    執行回測的主入口。

    Parameters
    ----------
    data_folder : 存放 TWSE CSV 的資料夾路徑。
                  None = 使用合成資料（測試模式）。
    tickers     : 指定股票代號清單；None = 自動掃描資料夾。
    cfg         : TradingConfig；None = 使用預設值。
    output_dir  : 輸出 equity_curve.png 與 trade_log.csv 的目錄。

    Returns
    -------
    dict  包含 equity_curve, trades, metrics
    """
    cfg = cfg or TradingConfig()

    if data_folder is not None:
        logger.info(f"從 CSV 資料夾載入：{data_folder}")
        data = DataLoader.load_folder(data_folder, tickers=tickers, adjusted=True)
    else:
        logger.info("未指定資料夾，使用合成資料（測試模式）。")
        synthetic_tickers = tickers or [f"{2000 + i}" for i in range(50)]
        data = DataLoader.generate_synthetic(
            synthetic_tickers, cfg.backtest_start, cfg.backtest_end
        )

    if not data:
        logger.error("無可用資料，請確認資料夾路徑與 CSV 格式。")
        return {}

    engine  = Backtester(cfg)
    logger.info(f"開始回測（{cfg.backtest_start} ~ {cfg.backtest_end}），"
                f"共 {len(data)} 支股票…")
    results = engine.run(data)

    # ── 輸出 ──
    Reporter.print_metrics(results["metrics"])

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    Reporter.plot_equity_curve(
        results["equity_curve"],
        save_path=str(out / "equity_curve.png"),
    )
    Reporter.save_trade_log(
        results["trades"],
        path=str(out / "trade_log.csv"),
    )

    # 印出最後 10 筆交易（選取存在的欄位）
    if not results["trades"].empty:
        display_cols = [
            "ticker", "direction", "lots",
            "entry_date", "exit_date", "hold_days",
            "adj_entry_price", "adj_exit_price",
            "raw_entry_price", "raw_exit_price",
            "pnl_net", "exit_reason",
        ]
        cols = [c for c in display_cols if c in results["trades"].columns]
        print("\n  最近 10 筆交易：")
        print(results["trades"][cols].tail(10).to_string(index=False))

    return results


# ── 直接執行 ──────────────────────────────────

if __name__ == "__main__":
    import sys

    # 用法：
    #   python auto_trading_system.py                          → 合成資料測試
    #   python auto_trading_system.py /path/to/csv_folder     → 載入所有股票
    #   python auto_trading_system.py /path/to/csv_folder 2330 2317 2454

    folder  = sys.argv[1] if len(sys.argv) > 1 else None
    symbols = sys.argv[2:] if len(sys.argv) > 2 else None

    cfg = TradingConfig(
        initial_equity  = 1_000_000,
        risk_pct        = 0.002,
        commission_rate = 0.001425,
        transaction_tax = 0.003,
        slippage        = 0.001,
        max_positions   = 10,
        min_avg_amount  = 5_000_000,
        backtest_start  = "2010-01-01",
        backtest_end    = "2023-12-29",
    )

    run_backtest(
        data_folder = folder,
        tickers     = symbols,
        cfg         = cfg,
        output_dir  = "output",
    )

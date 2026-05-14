"""
experiment_etf_strategy.py
──────────────────────────
用現有策略邏輯（High_N 突破進場 + ATR 追蹤停損出場）直接跑 ETF。
跳過選股條件（universe filter），只保留進出場 + 風控邏輯。

執行方式：
  python experiment_etf_strategy.py

輸出：
  output/etf_strategy/equity_curve.png
  output/etf_strategy/trade_log.csv
  output/etf_strategy/metrics.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

# 讓 import 找到 logic/auto_trading 下的模組
sys.path.insert(0, str(Path(__file__).parent))

from config import TradingConfig
from data_loader import DataLoader
from indicators import Indicators
from models import Position
from risk_manager import RiskManager
from signal_generator import SignalGenerator

# ── 設定 ──────────────────────────────────────────────────────────────
ETF_TICKER  = "taiex"    # 改成 taiex 跑加權指數
OUTPUT_DIR  = Path("output/etf_strategy")

def _find_project_root() -> Path:
    """往上找到含有 stocks_full 或 market_data 的專案根目錄"""
    p = Path(__file__).resolve()
    for _ in range(10):
        p = p.parent
        if (p / "market_data").exists():
            return p
    raise RuntimeError("找不到專案根目錄，請確認 market_data 資料夾存在")

MARKET_DATA = _find_project_root() / "market_data"
# ─────────────────────────────────────────────────────────────────────


def _close_position(pos: Position, exit_date: pd.Timestamp,
                    exit_price: float, cfg: TradingConfig,
                    risk_mgr: RiskManager) -> dict:
    actual_exit = (exit_price * (1 - cfg.slippage) if pos.direction == "long"
                   else exit_price * (1 + cfg.slippage))
    if pos.direction == "long":
        gross_pnl = (actual_exit - pos.adj_entry_price) * pos.shares
    else:
        gross_pnl = (pos.adj_entry_price - actual_exit) * pos.shares
    # 指數實驗不計算交易成本（無法對應台股手續費結構）
    cost = 0.0
    return {
        "ticker": pos.ticker, "direction": pos.direction,
        "lots": pos.lots, "entry_date": pos.entry_date, "exit_date": exit_date,
        "hold_days": (exit_date - pos.entry_date).days,
        "adj_entry_price": round(pos.adj_entry_price, 4),
        "adj_exit_price": round(actual_exit, 4),
        "gross_pnl": round(gross_pnl, 2),
        "total_cost": round(cost, 2),
        "pnl_net": round(gross_pnl - cost, 2),
    }


def load_etf(ticker: str, cfg: TradingConfig) -> pd.DataFrame:
    path = MARKET_DATA / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}")

    # taiex.csv 是標準 OHLCV 格式，直接讀；其他走 DataLoader
    if ticker == "taiex":
        df = pd.read_csv(path, parse_dates=["Date"])
        df = df.set_index("Date").sort_index()
        df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        # 補 Amount 欄（指標計算需要）
        df["Amount"] = df["Volume"] * df["Close"]
    else:
        raw = DataLoader._load_one(MARKET_DATA, ticker, prefer_adjusted=False)
        if raw is None:
            raise ValueError(f"無法載入 {ticker} 資料")
        df = raw

    return Indicators.add_all(df, cfg)


def run(cfg: TradingConfig) -> dict:
    print(f"載入 {ETF_TICKER}...")
    df = load_etf(ETF_TICKER, cfg)

    start = pd.Timestamp(cfg.backtest_start)
    end   = pd.Timestamp(cfg.backtest_end)
    dates = [d for d in df.index if start <= d <= end]

    risk_mgr = RiskManager(cfg)
    sig_gen  = SignalGenerator()

    equity        = cfg.initial_equity
    position: Position | None = None
    closed_trades: list[dict] = []
    equity_curve:  list[dict] = []

    pending_exit  = False
    pending_entry: tuple | None = None  # (direction, atr)

    for i, date in enumerate(dates):
        row = df.loc[date]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]

        # ── (A) 執行前一日訊號（T+1 開盤成交）──
        open_px = float(row["Open"])

        if pending_exit and position is not None:
            pnl = _close_position(position, date, open_px, cfg, risk_mgr)
            equity += pnl["pnl_net"]
            closed_trades.append(pnl)
            position = None
            pending_exit = False

        if pending_entry is not None and position is None:
            direction, atr = pending_entry
            # 指數/ETF 不用「張」的概念，直接用 equity × risk_pct 換算單位數
            risk_amount = min(equity * cfg.risk_pct, cfg.max_risk_amount)
            units = risk_amount / (cfg.atr_multiplier * atr) if atr > 0 else 0
            if units > 0:
                position = Position(
                    ticker=ETF_TICKER,
                    direction=direction,
                    adj_entry_price=open_px,
                    raw_entry_price=open_px,
                    shares=units,   # 用 units 代替 shares
                    lots=1,
                    entry_date=date,
                    trail_high=open_px,
                    trail_low=open_px,
                    atr_at_entry=atr,
                    equity_at_entry=equity,
                )
            pending_entry = None

        # ── (B) 更新 trail ──
        if position is not None:
            position.update_trail(float(row["High"]), float(row["Low"]))

        # ── (C) 出場訊號 ──
        if position is not None:
            if position.direction == "long":
                if sig_gen.long_exit(row, position.trail_high, cfg.atr_multiplier):
                    pending_exit = True
            else:
                if sig_gen.short_exit(row, position.trail_low, cfg.atr_multiplier):
                    pending_exit = True

        # ── (D) 進場訊號（無持倉且無待進場才判斷）──
        # 實驗版：只用突破 + 量能，去掉 MA 條件
        if position is None and pending_entry is None and not pending_exit:
            prev_row = df.iloc[df.index.get_loc(date) - 1] if i > 0 else None
            if prev_row is not None:
                atr = float(row.get("ATR", float("nan")))
                if pd.notna(atr) and atr > 0:
                    vol_ok = row["Volume"] > row["Vol_MA20"] * cfg.vol_mult
                    long_ok  = row["Close"] > prev_row["High_N"] and vol_ok
                    short_ok = row["Close"] < prev_row["Low_N"]  and vol_ok
                    if long_ok:
                        pending_entry = ("long", atr)
                    elif short_ok:
                        pending_entry = ("short", atr)

        equity_curve.append({"date": date, "equity": equity})

    return {
        "equity_curve": pd.DataFrame(equity_curve).set_index("date"),
        "trades":       pd.DataFrame(closed_trades),
    }


def calc_metrics(equity: pd.Series) -> dict:
    ret   = equity.pct_change().dropna()
    total = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr  = ((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) * 100
    mdd   = ((equity - equity.cummax()) / equity.cummax()).min() * 100
    sharpe = (ret.mean() / ret.std() * (252 ** 0.5)) if ret.std() > 0 else 0
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    return {
        "累積報酬 (%)": round(total, 2),
        "年化報酬 (%)": round(cagr, 2),
        "最大回撤 (%)": round(mdd, 2),
        "Sharpe":       round(sharpe, 3),
        "Calmar":       round(calmar, 3),
    }


def main() -> None:
    cfg = TradingConfig(
        min_avg_amount   = 0,          # 指數不做成交金額篩選
        min_long_price   = 0,
        min_short_price  = 0,
        initial_equity   = 10_000_000,
        max_risk_amount  = 100_000,
        breakout_window  = 20,         # 改成 20 日突破
    )

    results = run(cfg)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    equity = results["equity_curve"]["equity"]
    metrics = calc_metrics(equity)

    print("\n═══ 策略績效 ═══")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # 儲存圖表
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(equity.index, equity.values, linewidth=1.5)
    ax.set_title(f"{ETF_TICKER} 策略回測（High_N 突破 + ATR 追蹤停損）")
    ax.set_ylabel("淨值（元）")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "equity_curve.png", dpi=150)
    plt.close()
    print(f"\n圖表：{OUTPUT_DIR / 'equity_curve.png'}")

    # 儲存交易紀錄
    if not results["trades"].empty:
        results["trades"].to_csv(OUTPUT_DIR / "trade_log.csv", index=False, encoding="utf-8-sig")
        print(f"交易紀錄：{OUTPUT_DIR / 'trade_log.csv'}")
        print(f"共 {len(results['trades'])} 筆交易")

    # 儲存績效
    with open(OUTPUT_DIR / "metrics.txt", "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")


if __name__ == "__main__":
    main()

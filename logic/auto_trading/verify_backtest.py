"""
verify_backtest.py
──────────────────
回測驗證腳本。

Method B：合成資料，已知正確答案，自動 pass / fail
Method C：逐筆交易圖表，人工目視判讀

用法：
  python verify_backtest.py
    -> Method B (合成資料自動驗證) + Method C 圖表

  python verify_backtest.py C:/path/to/stocks_folder
    -> Method B + 真實資料 Method C

  python verify_backtest.py C:/path/to/stocks_folder 2330 2317
    -> Method B + 指定股票 Method C
"""
from __future__ import annotations

import sys
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from backtester import Backtester
from config import TradingConfig
from data_loader import DataLoader
from indicators import Indicators

# ──────────────────────────────────────────────
# 輸出工具
# ──────────────────────────────────────────────

PASS_TAG = "[PASS]"
FAIL_TAG = "[FAIL]"

def check(cond: bool, msg: str) -> bool:
    tag = PASS_TAG if cond else FAIL_TAG
    print(f"  {tag}  {msg}")
    return cond


# ══════════════════════════════════════════════
# Method B：合成資料
# ══════════════════════════════════════════════

def make_controlled_data(seed: int = 0):
    """
    建立可預測答案的價格序列。

    Phase 1 (200日): Close 從 90 線性漲至 100  -> MA50 > MA100
    Phase 2 ( 55日): Close = 100.00             -> 50日收盤高 = 100
    Phase 3 (  1日): Close = 101               -> 突破訊號日
    Phase 4 ( 20日): Close 每日 +0.7 -> ~115   -> trail_high 持續更新
    Phase 5 ( 10日): Close = 105               -> 停損觸發
    """
    rng = np.random.default_rng(seed)
    N1, N2, N3, N4, N5 = 200, 55, 1, 20, 10
    total  = N1 + N2 + N3 + N4 + N5
    dates  = pd.bdate_range("2015-01-01", periods=total)

    close = np.zeros(total)
    close[:N1]       = np.linspace(90, 100, N1)
    close[N1:N1+N2]  = 100.0
    sig_idx          = N1 + N2
    close[sig_idx]   = 101.0
    for k in range(N4):
        close[sig_idx + 1 + k] = 101.0 + (k + 1) * 0.7
    drop_idx         = sig_idx + 1 + N4
    close[drop_idx:] = 105.0

    noise  = rng.uniform(0.10, 0.35, total)
    high   = close + noise
    low    = close - noise
    open_  = close + rng.uniform(-0.05, 0.05, total)
    amount = np.full(total, 50_000_000.0)
    volume = (amount / close).astype(int).astype(float)

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low,
         "Close": close, "Volume": volume, "Amount": amount},
        index=dates,
    )
    df.attrs["ticker"]      = "TEST"
    df.attrs["is_adjusted"] = True

    return (
        {"TEST": df},
        dates[sig_idx],        # signal_date
        dates[sig_idx + 1],    # expected_entry  (T+1)
        dates[drop_idx],       # drop_date
        dates[drop_idx + 1],   # expected_exit   (T+1)
    )


def run_method_b():
    print("\n" + "=" * 58)
    print("  Method B: Synthetic Data Verification")
    print("=" * 58)

    raw_data, signal_date, expected_entry, drop_date, expected_exit = \
        make_controlled_data()

    cfg = TradingConfig(
        initial_equity  = 1_000_000,
        risk_pct        = 0.01,
        atr_period      = 14,       # 縮短暖機期
        backtest_start  = "2015-01-01",
        backtest_end    = "2016-06-30",
        max_positions   = 1,
        max_trade_cost  = 500_000,  # 測試用，不限制
        min_avg_amount  = 1_000_000,
    )

    engine  = Backtester(cfg)
    results = engine.run(raw_data)
    trades  = results["trades"]

    print(f"\n  {'':4} {'Expected':>22}  {'Actual':>22}")
    print(f"  {'-'*50}")
    if trades.empty:
        print(f"  {FAIL_TAG}  No trades generated! Check filter conditions.")
        return None, raw_data, cfg

    t = trades.iloc[0]
    print(f"  Signal Date   : {signal_date.date()!s:>22}")
    print(f"  Entry Date    : {expected_entry.date()!s:>22}  "
          f"{pd.Timestamp(t['entry_date']).date()!s:>22}")
    print(f"  Stop Triggered: {drop_date.date()!s:>22}")
    print(f"  Exit Date     : {expected_exit.date()!s:>22}  "
          f"{pd.Timestamp(t['exit_date']).date()!s:>22}")
    print(f"  Direction     : {'long':>22}  {t['direction']:>22}")
    print()

    df_ind   = Indicators.add_all(raw_data["TEST"], cfg)
    sig_row  = df_ind.loc[signal_date]
    sig_iloc = df_ind.index.get_loc(signal_date)
    prev_row = df_ind.iloc[sig_iloc - 1]

    entry_open    = df_ind.loc[expected_entry, "Open"]
    exit_open     = df_ind.loc[expected_exit,  "Open"]
    exp_adj_entry = entry_open * (1 + cfg.slippage)
    exp_adj_exit  = exit_open  * (1 - cfg.slippage)

    results_list = [
        check(pd.Timestamp(t["entry_date"]).date() == expected_entry.date(),
              f"[Timing] Entry = Signal+1 ({expected_entry.date()})"),

        check(pd.Timestamp(t["exit_date"]).date() == expected_exit.date(),
              f"[Timing] Exit = StopSignal+1 ({expected_exit.date()})"),

        check(t["direction"] == "long",
              "[Signal] Direction = long"),

        check(abs(t["adj_entry_price"] - exp_adj_entry) < 0.02,
              f"[Price] Entry({t['adj_entry_price']:.4f}) ~= NextDayOpen x(1+slip)({exp_adj_entry:.4f})"),

        check(abs(t["adj_exit_price"] - exp_adj_exit) < 0.02,
              f"[Price] Exit({t['adj_exit_price']:.4f}) ~= NextDayOpen x(1-slip)({exp_adj_exit:.4f})"),

        check(sig_row["Close"] > prev_row["High_N"],
              f"[Entry Cond] Close({sig_row['Close']:.2f}) > Prev 50D High({prev_row['High_N']:.2f})"),

        check(sig_row["MA_Fast"] > sig_row["MA_Slow"],
              f"[Entry Cond] MA50({sig_row['MA_Fast']:.3f}) > MA100({sig_row['MA_Slow']:.3f})"),

        check(sig_row["Avg_Amount_20"] >= cfg.min_avg_amount,
              f"[Filter] 20D AvgAmt({sig_row['Avg_Amount_20']:.0f}) >= {cfg.min_avg_amount:.0f}"),

        check(
            sig_row["High"] > float(prev_row.get("High_52W", float("nan"))) or
            sig_row["Low"]  < float(prev_row.get("Low_52W",  float("nan"))),
            f"[Filter] 52W breakout: High({sig_row['High']:.2f}) > prev High_52W"
            f"({float(prev_row.get('High_52W', float('nan'))):.2f})"
            f" OR Low({sig_row['Low']:.2f}) < prev Low_52W"
            f"({float(prev_row.get('Low_52W', float('nan'))):.2f})",
        ),
    ]

    passed = sum(results_list)
    total  = len(results_list)
    print()
    if passed == total:
        print(f"  [OK]  All {total} checks passed!")
    else:
        print(f"  [NG]  {total - passed}/{total} checks FAILED. Please review logic.")

    return results, raw_data, cfg


# ══════════════════════════════════════════════
# Method C：逐筆交易圖表
# ══════════════════════════════════════════════

def _trail_stop_series(
    df:          pd.DataFrame,
    entry_date:  pd.Timestamp,
    exit_date:   pd.Timestamp,
    direction:   str,
    entry_price: float,
    atr_mult:    float,
) -> pd.Series:
    """重建持倉期間每日停損水位，供圖表標示。"""
    mask   = (df.index >= entry_date) & (df.index <= exit_date)
    period = df.loc[mask].copy()

    if direction == "long":
        trail_high = period["High"].copy()
        trail_high.iloc[0] = max(entry_price, trail_high.iloc[0])
        trail_high = trail_high.cummax()
        stop = trail_high - atr_mult * period["ATR"]
    else:
        trail_low = period["Low"].copy()
        trail_low.iloc[0] = min(entry_price, trail_low.iloc[0])
        trail_low = trail_low.cummin()
        stop = trail_low + atr_mult * period["ATR"]

    return stop


def plot_trade(
    trade:    pd.Series,
    data_all: dict[str, pd.DataFrame],
    cfg:      TradingConfig,
    out_dir:  str = "output/trade_charts",
    idx:      int = 0,
) -> None:
    ticker      = trade["ticker"]
    entry_date  = pd.Timestamp(trade["entry_date"])
    exit_date   = pd.Timestamp(trade["exit_date"])
    direction   = trade["direction"]
    adj_entry   = trade["adj_entry_price"]
    adj_exit    = trade["adj_exit_price"]
    pnl         = trade["pnl_net"]

    df_raw = data_all.get(ticker)
    if df_raw is None:
        print(f"  [WARN] No data for {ticker}, skipping.")
        return

    df = Indicators.add_all(df_raw, cfg)

    view_start = entry_date - pd.tseries.offsets.BDay(60)
    view_end   = exit_date  + pd.tseries.offsets.BDay(20)
    dv = df.loc[(df.index >= view_start) & (df.index <= view_end)].copy()
    if dv.empty:
        return

    stop_series = _trail_stop_series(
        df, entry_date, exit_date, direction, adj_entry, cfg.atr_multiplier)
    stop_full = stop_series.reindex(dv.index)

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.suptitle(
        f"#{idx+1} {ticker}  {direction.upper()}  "
        f"Entry {entry_date.date()} -> Exit {exit_date.date()}  "
        f"Hold {trade['hold_days']}d  PnL={pnl:+,.0f}",
        fontsize=11, fontweight="bold",
    )

    ax = axes[0]
    ax.plot(dv.index, dv["Close"],   color="#1565C0", lw=1.3, label="Close")
    ax.plot(dv.index, dv["MA_Fast"], color="#FB8C00", lw=1.0, ls="--",
            label=f"MA{cfg.ma_fast}")
    ax.plot(dv.index, dv["MA_Slow"], color="#7B1FA2", lw=1.0, ls="--",
            label=f"MA{cfg.ma_slow}")
    ax.plot(dv.index, dv["High_N"],  color="#388E3C", lw=0.8, ls=":",
            label=f"{cfg.breakout_window}D High (entry line)")
    ax.plot(dv.index, stop_full,     color="#C62828", lw=1.2, ls="-.",
            label="Trailing Stop")

    if entry_date in dv.index:
        ax.axvline(entry_date, color="#00897B", lw=1.5, ls="--", alpha=0.7)
        ax.scatter([entry_date], [adj_entry],
                   color="#00897B", marker="^", s=150, zorder=6, label="Entry")
        ax.annotate(f"Entry\n{adj_entry:.2f}",
                    xy=(entry_date, adj_entry),
                    xytext=(5, 10), textcoords="offset points", fontsize=7)

    if exit_date in dv.index:
        ax.axvline(exit_date, color="#E53935", lw=1.5, ls="--", alpha=0.7)
        ax.scatter([exit_date], [adj_exit],
                   color="#E53935", marker="v", s=150, zorder=6, label="Exit")
        ax.annotate(f"Exit\n{adj_exit:.2f}",
                    xy=(exit_date, adj_exit),
                    xytext=(5, -20), textcoords="offset points", fontsize=7)

    ax.set_ylabel("Price")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    ax.grid(alpha=0.3)

    axes[1].bar(dv.index, dv["ATR"], color="#546E7A", alpha=0.6, width=1)
    if entry_date in dv.index:
        axes[1].axvline(entry_date, color="#00897B", lw=1.0, ls="--", alpha=0.7)
    if exit_date in dv.index:
        axes[1].axvline(exit_date, color="#E53935", lw=1.0, ls="--", alpha=0.7)
    axes[1].set_ylabel(f"ATR{cfg.atr_period}")
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fname = out / f"{idx+1:03d}_{ticker}_{entry_date.strftime('%Y%m%d')}_{direction}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart -> {fname.name}")


def run_method_c(
    results:  dict,
    data_all: dict[str, pd.DataFrame],
    cfg:      TradingConfig,
    out_dir:  str = "output/trade_charts",
) -> None:
    print("\n" + "=" * 58)
    print("  Method C: Trade-by-Trade Charts")
    print("=" * 58)

    trades = results.get("trades")
    if trades is None or trades.empty:
        print("  No trades to plot.")
        return

    print(f"  {len(trades)} trades -> charts saved to {out_dir}/\n")
    for i, (_, trade) in enumerate(trades.iterrows()):
        plot_trade(trade, data_all, cfg, out_dir=out_dir, idx=i)


# ══════════════════════════════════════════════
# 5-Step Audit Tables
# ══════════════════════════════════════════════

def _prev_bday(date: pd.Timestamp, index: pd.DatetimeIndex) -> pd.Timestamp | None:
    """回傳 date 在 index 中的前一個交易日，找不到回傳 None。"""
    loc = index.get_loc(date)
    return index[loc - 1] if loc > 0 else None


def audit_step1_filter(
    data: dict[str, pd.DataFrame],
    cfg:  TradingConfig,
    sample_dates: list[str] | None = None,
) -> None:
    """
    Step 1：篩選驗證
    隨機抽取幾個日期，列出所有股票的篩選數值，讓你手動比對原始 CSV。
    """
    print("\n" + "=" * 70)
    print("  Step 1 | Filter Verification")
    print("=" * 70)

    # 合併所有交易日，取樣
    all_dates = sorted({d for df in data.values() for d in df.index})
    if sample_dates:
        check_dates = [pd.Timestamp(d) for d in sample_dates if pd.Timestamp(d) in all_dates]
    else:
        # 自動抽 3 個：前段、中段、後段
        n = len(all_dates)
        check_dates = [all_dates[n // 6], all_dates[n // 2], all_dates[5 * n // 6]]

    for check_date in check_dates:
        print(f"\n  Date: {check_date.date()}")
        print(f"  {'Ticker':<8} {'Close':>8} {'20D AvgAmt':>14} {'prevHi52W':>10} "
              f"{'Breakout':>8} {'AmtOK':>6} {'BrkOK':>7} {'Selected':>9}")
        print(f"  {'-'*75}")

        for ticker, df_raw in sorted(data.items()):
            df = Indicators.add_all(df_raw, cfg)
            if check_date not in df.index:
                continue
            row = df.loc[check_date]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]

            close     = row["Close"]
            avg_amt   = row.get("Avg_Amount_20", float("nan"))
            # 52W 突破：用昨日的 High_52W / Low_52W（需要前一列）
            idx = df.index.get_loc(check_date)
            if idx > 0:
                prev = df.iloc[idx - 1]
                high52w_prev = float(prev.get("High_52W", float("nan")))
                low52w_prev  = float(prev.get("Low_52W",  float("nan")))
                brk_high = (not pd.isna(high52w_prev)) and row["High"] > high52w_prev
                brk_low  = (not pd.isna(low52w_prev))  and row["Low"]  < low52w_prev
                near_ok  = brk_high or brk_low
            else:
                high52w_prev = float("nan")
                near_ok = False
            amt_ok   = avg_amt >= cfg.min_avg_amount
            selected = amt_ok and near_ok

            print(f"  {ticker:<8} {close:>8.2f} {avg_amt:>14,.0f} {high52w_prev:>10.2f} "
                  f"{'BRK' if near_ok else '-':>7} {str(amt_ok):>6} {str(near_ok):>7} {str(selected):>9}")


def audit_step2_entries(
    trades: pd.DataFrame,
    data:   dict[str, pd.DataFrame],
    cfg:    TradingConfig,
) -> None:
    """
    Step 2：進場訊號驗證
    對每筆交易，顯示訊號日的指標值，讓你確認進場條件是否成立。
    """
    print("\n" + "=" * 90)
    print("  Step 2 | Entry Signal Verification")
    print("=" * 90)
    print(f"  {'Ticker':<8} {'SigDate':>12} {'SigClose':>9} {'Prev50Hi':>9} "
          f"{'MA50':>8} {'MA100':>8} {'EntryDate':>12} {'EntryOpen':>10}")
    print(f"  {'-'*82}")

    for _, t in trades.iterrows():
        ticker     = t["ticker"]
        entry_date = pd.Timestamp(t["entry_date"])
        df_raw = data.get(ticker)
        if df_raw is None:
            continue
        df = Indicators.add_all(df_raw, cfg)

        sig_date = _prev_bday(entry_date, df.index)
        if sig_date is None:
            continue
        prev_sig = _prev_bday(sig_date, df.index)
        if prev_sig is None:
            continue

        sig_row  = df.loc[sig_date]
        prev_row = df.loc[prev_sig]
        if isinstance(sig_row,  pd.DataFrame): sig_row  = sig_row.iloc[-1]
        if isinstance(prev_row, pd.DataFrame): prev_row = prev_row.iloc[-1]

        entry_open = df.loc[entry_date, "Open"] if entry_date in df.index else float("nan")

        print(f"  {ticker:<8} {str(sig_date.date()):>12} {sig_row['Close']:>9.2f} "
              f"{prev_row['High_N']:>9.2f} {sig_row['MA_Fast']:>8.2f} "
              f"{sig_row['MA_Slow']:>8.2f} {str(entry_date.date()):>12} {entry_open:>10.2f}")

    print(f"\n  Verify: SigClose > Prev50Hi, MA50 > MA100, EntryDate = SigDate+1")


def audit_step3_exits(
    trades: pd.DataFrame,
    data:   dict[str, pd.DataFrame],
    cfg:    TradingConfig,
) -> None:
    """
    Step 3：出場訊號驗證
    對每筆交易，重建出場訊號當日的 trail_high 與停損水位。
    """
    print("\n" + "=" * 100)
    print("  Step 3 | Exit Signal Verification")
    print("=" * 100)
    print(f"  {'Ticker':<8} {'HoldDays':>9} {'SigDate':>12} {'TrailHigh':>10} "
          f"{'ATR':>7} {'StopLvl':>9} {'SigClose':>9} {'ExitDate':>12} {'ExitOpen':>10}")
    print(f"  {'-'*92}")

    for _, t in trades.iterrows():
        ticker     = t["ticker"]
        entry_date = pd.Timestamp(t["entry_date"])
        exit_date  = pd.Timestamp(t["exit_date"])
        direction  = t["direction"]
        df_raw = data.get(ticker)
        if df_raw is None:
            continue
        df = Indicators.add_all(df_raw, cfg)

        exit_sig_date = _prev_bday(exit_date, df.index)
        if exit_sig_date is None:
            continue

        # 重建 trail_high 從進場到出場訊號日
        hold_mask = (df.index >= entry_date) & (df.index <= exit_sig_date)
        period    = df.loc[hold_mask]
        if period.empty:
            continue

        if direction == "long":
            trail = period["High"].cummax().iloc[-1]
            atr   = period["ATR"].iloc[-1]
            stop  = trail - cfg.atr_multiplier * atr
            sig_close = period["Close"].iloc[-1]
        else:
            trail = period["Low"].cummin().iloc[-1]
            atr   = period["ATR"].iloc[-1]
            stop  = trail + cfg.atr_multiplier * atr
            sig_close = period["Close"].iloc[-1]

        exit_open = df.loc[exit_date, "Open"] if exit_date in df.index else float("nan")
        hold_days = t["hold_days"]

        print(f"  {ticker:<8} {hold_days:>9} {str(exit_sig_date.date()):>12} "
              f"{trail:>10.2f} {atr:>7.2f} {stop:>9.2f} {sig_close:>9.2f} "
              f"{str(exit_date.date()):>12} {exit_open:>10.2f}")

    print(f"\n  Verify: SigClose < StopLvl (long), ExitDate = SigDate+1")


def audit_step4_position_size(
    trades: pd.DataFrame,
    cfg:    TradingConfig,
) -> None:
    """
    Step 4：部位大小驗證
    顯示每筆進場的部位計算過程，讓你手算確認。
    """
    print("\n" + "=" * 82)
    print("  Step 4 | Position Size Verification")
    print("=" * 82)
    print(f"  {'Ticker':<8} {'Equity':>12} {'ATR':>7} {'RiskAmt':>10} "
          f"{'TheoLots':>9} {'ActualLots':>11} {'Shares':>8}")
    print(f"  {'-'*72}")

    for _, t in trades.iterrows():
        equity    = t.get("equity_at_entry", float("nan"))
        atr       = t["atr_at_entry"]
        # 新公式：risk = min(equity × risk_pct, max_risk_amount)
        # lots = int(risk / (atr_multiplier × ATR × 1000))
        risk_amt  = min(equity * cfg.risk_pct, cfg.max_risk_amount)
        stop_dist = cfg.atr_multiplier * atr
        theo_lots = risk_amt / (stop_dist * 1000) if stop_dist > 0 else float("nan")
        actual    = t["lots"]
        shares    = t["shares"]

        print(f"  {t['ticker']:<8} {equity:>12,.0f} {atr:>7.4f} {risk_amt:>10.2f} "
              f"{theo_lots:>9.2f} {actual:>11} {shares:>8}")

    print(f"\n  Formula: RiskAmt = min(Equity × {cfg.risk_pct*100:.1f}%, {cfg.max_risk_amount:,.0f}),  "
          f"TheoLots = RiskAmt / ({cfg.atr_multiplier} × ATR × 1000),  ActualLots = int(TheoLots)")


def audit_step5_costs(
    trades: pd.DataFrame,
    cfg:    TradingConfig,
) -> None:
    """
    Step 5：交易成本驗證
    逐筆列出費用明細，讓你手動確認計算是否正確。
    """
    print("\n" + "=" * 110)
    print("  Step 5 | Transaction Cost Verification")
    print("=" * 110)
    print(f"  {'Ticker':<8} {'EntryPx':>8} {'ExitPx':>8} {'Shares':>7} "
          f"{'BuyComm':>9} {'SellComm':>9} {'Tax':>9} "
          f"{'BuySlip':>8} {'SellSlip':>9} {'TotalCost':>10} {'GrossPnL':>10} {'NetPnL':>10}")
    print(f"  {'-'*102}")

    for _, t in trades.iterrows():
        ep    = t["adj_entry_price"]
        xp    = t["adj_exit_price"]
        sh    = t["shares"]
        buy_comm  = ep * sh * cfg.commission_rate
        sell_comm = xp * sh * cfg.commission_rate
        tax       = xp * sh * cfg.transaction_tax
        buy_slip  = ep * sh * cfg.slippage
        sell_slip = xp * sh * cfg.slippage
        total     = buy_comm + sell_comm + tax + buy_slip + sell_slip
        gross     = t["gross_pnl"]
        net       = t["pnl_net"]

        print(f"  {t['ticker']:<8} {ep:>8.2f} {xp:>8.2f} {sh:>7} "
              f"{buy_comm:>9.1f} {sell_comm:>9.1f} {tax:>9.1f} "
              f"{buy_slip:>8.1f} {sell_slip:>9.1f} {total:>10.1f} {gross:>10.1f} {net:>10.1f}")

    print(f"\n  Rates: Comm={cfg.commission_rate*100:.4f}%, "
          f"Tax={cfg.transaction_tax*100:.3f}%(sell only), "
          f"Slip={cfg.slippage*100:.3f}%(both sides)")


def run_audit_tables(
    results:  dict,
    data_all: dict[str, pd.DataFrame],
    cfg:      TradingConfig,
    sample_dates: list[str] | None = None,
) -> None:
    trades = results.get("trades")
    if trades is None or trades.empty:
        print("  No trades — cannot run audit tables.")
        return

    audit_step1_filter(data_all, cfg, sample_dates)
    audit_step2_entries(trades, data_all, cfg)
    audit_step3_exits(trades, data_all, cfg)
    audit_step4_position_size(trades, cfg)
    audit_step5_costs(trades, cfg)


# ══════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Backtest Verification Tool",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Synthetic data only (Method B + audit)
  python verify_backtest.py

  # Real data, all stocks
  python verify_backtest.py --folder C:/path/to/stocks

  # Real data, specific tickers
  python verify_backtest.py --folder C:/path/to/stocks --tickers 2330 2317 2454

  # Real data + check filter on specific dates (Step 1)
  python verify_backtest.py --folder C:/path/to/stocks --dates 2020-04-01 2020-04-15

  # All options combined
  python verify_backtest.py --folder C:/path/to/stocks --tickers 2330 2317 --dates 2020-04-01 2020-07-01
        """,
    )
    parser.add_argument("--folder",  "-f", default=None,
                        help="CSV folder path (omit = use synthetic data)")
    parser.add_argument("--tickers", "-t", nargs="*", default=None,
                        help="Stock tickers to load (default: all in folder)")
    parser.add_argument("--dates",   "-d", nargs="*", default=None,
                        help="Dates to check in Step 1 filter audit (YYYY-MM-DD)")
    args = parser.parse_args()

    # ── Method B：固定跑合成資料驗證 ──
    result_b, synth_data, synth_cfg = run_method_b()

    if args.folder is not None:
        # ── 真實資料：Method C + 5-Step Audit ──
        print(f"\n  Loading real data: {args.folder}")
        real_cfg  = TradingConfig()
        real_data = DataLoader.load_folder(
            args.folder, tickers=args.tickers, adjusted=True)
        if real_data:
            engine  = Backtester(real_cfg)
            results = engine.run(real_data)
            run_method_c(results, real_data, real_cfg)
            run_audit_tables(results, real_data, real_cfg, sample_dates=args.dates)
        else:
            print("  [WARN] No data loaded. Check folder path and CSV format.")
    else:
        # ── 合成資料：Method C + 5-Step Audit ──
        if result_b is not None:
            run_method_c(result_b, synth_data, synth_cfg)
            run_audit_tables(result_b, synth_data, synth_cfg, sample_dates=args.dates)

    print("\nDone. Charts saved to output/trade_charts/\n")

"""
checkpoint.py
─────────────
回測邏輯驗證：在兩個關鍵節點確認程式行為符合預期。

Checkpoint 1 — 選股池驗證
  對任意一天的候選清單，逐支確認是否符合：
    - 流動性：Avg_Amount_20 >= min_avg_amount
    - 近 52 週新高：Close >= High_52W * 0.90
    - 買得起：Close * 1000 <= initial_equity

Checkpoint 2 — 進出場驗證
  對每筆成交，回頭確認：
    - 進場時訊號有觸發（突破 + MA 條件）
    - 進場當天該股票在選股池內
"""

from __future__ import annotations

import pandas as pd

from config import TradingConfig
from indicators import Indicators
from risk_manager import RiskManager
from signal_generator import SignalGenerator
from universe_filter import UniverseFilter



# ══════════════════════════════════════════════
# Checkpoint 1：選股池驗證
# ══════════════════════════════════════════════

def check_universe(
    data:       dict[str, pd.DataFrame],
    date:       pd.Timestamp,
    candidates: list[str],
    cfg:        TradingConfig,
) -> pd.DataFrame:
    """
    對指定日期的候選清單，逐支列出關鍵指標並標記是否通過各條件。

    Returns
    -------
    pd.DataFrame  每列一支股票，欄位包含指標值與通過狀態
    """
    records = []
    max_price = cfg.initial_equity / 1000

    for ticker in candidates:
        df = data.get(ticker)
        if df is None or date not in df.index:
            continue

        row = df.loc[date]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]

        close       = float(row.get("Close", float("nan")))
        high_52w    = float(row.get("High_52W", float("nan")))
        low_52w     = float(row.get("Low_52W", float("nan")))
        avg_amount  = float(row.get("Avg_Amount_20", float("nan")))
        roc_avg      = float(row.get("ROC_avg", float("nan")))

        liquidity_ok  = avg_amount >= cfg.min_avg_amount
        near_52w_high = (close >= high_52w * 0.90) if high_52w > 0 else False
        affordable    = close <= max_price

        records.append({
            "ticker":        ticker,
            "close":         round(close, 2),
            "close/52w_pct": round(close / high_52w * 100, 1) if high_52w > 0 else None,
            "avg_amount_20": round(avg_amount / 1_000_000, 2),
            "roc_avg":        round(roc_avg, 1) if not pd.isna(roc_avg) else None,
            "liquidity_ok":  liquidity_ok,
            "near_52w_high": near_52w_high,
            "affordable":    affordable,
            "ALL_PASS":      liquidity_ok and near_52w_high and affordable,
        })

    return pd.DataFrame(records)


# ══════════════════════════════════════════════
# Checkpoint 2：進出場驗證
# ══════════════════════════════════════════════

def check_trades(
    trades:   pd.DataFrame,
    raw_data: dict[str, pd.DataFrame],
    cfg:      TradingConfig,
) -> pd.DataFrame:
    """
    對每筆成交紀錄，回頭確認進場條件是否真的成立。

    Returns
    -------
    pd.DataFrame  每列一筆交易，含各條件的通過狀態
    """
    if trades.empty:
        return pd.DataFrame()

    data    = {t: Indicators.add_all(df, cfg) for t, df in raw_data.items()}
    uni_flt = UniverseFilter(cfg)
    sig_gen = SignalGenerator()

    all_dates  = sorted({d for df in data.values() for d in df.index})
    prev_of    = {d: all_dates[i-1] for i, d in enumerate(all_dates) if i > 0}

    records = []
    for _, trade in trades.iterrows():
        ticker     = trade["ticker"]
        entry_date = pd.Timestamp(trade["entry_date"])  # T+1（執行日）
        direction  = trade["direction"]

        # 時序：entry_date = T+1（執行），signal_date = T（訊號日）
        signal_date      = prev_of.get(entry_date)        # T
        prev_signal_date = prev_of.get(signal_date) if signal_date else None  # T-1

        df = data.get(ticker)

        # A. 在選股池內？（訊號日 T 的選股池）
        candidates = uni_flt.filter(data, signal_date) if signal_date else []
        in_pool    = ticker in candidates

        # B. 進場訊號觸發？（T 日收盤 vs T-1 日指標）
        signal_ok = False
        if (df is not None and signal_date and prev_signal_date
                and signal_date in df.index and prev_signal_date in df.index):
            row      = df.loc[signal_date]
            prev_row = df.loc[prev_signal_date]
            if isinstance(row,      pd.DataFrame): row      = row.iloc[-1]
            if isinstance(prev_row, pd.DataFrame): prev_row = prev_row.iloc[-1]
            if direction == "long":
                signal_ok = sig_gen.long_entry(row, prev_row)
            else:
                signal_ok = sig_gen.short_entry(row, prev_row)

        # C. 比這支 ROC 更高的股票，若有訊號，必須有正當理由沒進場
        #    正當理由：已持倉 / 張數=0 / 當時持倉已滿
        #    若沒有正當理由卻沒進場 → 違規（代表排序沒有照 ROC）
        higher_violations = []
        if signal_date and prev_signal_date:
            risk_mgr = RiskManager(cfg)

            # 取得這支股票的 roc_avg
            own_roc = float("nan")
            if df is not None and signal_date in df.index:
                r = df.loc[signal_date]
                if isinstance(r, pd.DataFrame): r = r.iloc[-1]
                own_roc = float(r.get("ROC_avg", float("nan")))

            # 當時已持倉的股票（進場日 < entry_date，且尚未出場）
            already_held = set(
                trades.loc[
                    (pd.to_datetime(trades["entry_date"]) < entry_date) &
                    (pd.to_datetime(trades["exit_date"])  > entry_date),
                    "ticker"
                ]
            )
            # 同一批次進場的股票（同一 T+1 執行日），屬於合法同批選入，不算違規
            same_batch = set(
                trades.loc[pd.to_datetime(trades["entry_date"]) == entry_date, "ticker"]
            )

            for other in candidates:
                if other == ticker:
                    continue
                other_df = data.get(other)
                if other_df is None or signal_date not in other_df.index:
                    continue
                other_row = other_df.loc[signal_date]
                if isinstance(other_row, pd.DataFrame): other_row = other_row.iloc[-1]
                other_roc = float(other_row.get("ROC_avg", float("nan")))
                if pd.isna(other_roc) or other_roc <= own_roc:
                    continue

                # ROC 更高，確認有沒有訊號
                if prev_signal_date not in other_df.index:
                    continue
                other_prev = other_df.loc[prev_signal_date]
                if isinstance(other_prev, pd.DataFrame): other_prev = other_prev.iloc[-1]
                has_signal = (sig_gen.long_entry(other_row, other_prev) or
                              sig_gen.short_entry(other_row, other_prev))
                if not has_signal:
                    continue

                # 有訊號 → 記錄原因（不管是否有正當理由，都列出）
                atr  = float(other_row.get("ATR", 0) or 0)
                lots = risk_mgr.position_size_lots(cfg.initial_equity, atr) if atr > 0 else 0

                # 確認 T+1（執行日）是否有可交易的開盤資料
                exec_open = float("nan")
                if entry_date in other_df.index:
                    exec_row = other_df.loc[entry_date]
                    if isinstance(exec_row, pd.DataFrame): exec_row = exec_row.iloc[-1]
                    exec_open = float(exec_row.get("Open", float("nan")))

                # 以 T+1 開盤價重算 max_trade_cost 後是否 lots=0
                exec_lots = lots
                if not pd.isna(exec_open) and exec_lots > 0:
                    adj_open = exec_open * (1 + cfg.slippage)
                    if adj_open * exec_lots * 1000 > cfg.max_trade_cost:
                        exec_lots = max(int(cfg.max_trade_cost / (adj_open * 1000)), 0)

                # 判斷 entry_date 當天實際可用槽位
                slots_used = len(already_held) + len(same_batch)
                # same_batch 中 ROC 比 6120 更高的股票數（排在它前面）
                higher_in_batch = sum(
                    1 for t in same_batch
                    if t != ticker and t != other and data.get(t) is not None
                    and signal_date in data[t].index
                    and float(data[t].loc[signal_date].get("ROC_avg", float("-inf"))
                              if not isinstance(data[t].loc[signal_date], pd.DataFrame)
                              else data[t].loc[signal_date].iloc[-1].get("ROC_avg", float("-inf"))) > other_roc
                )
                slots_before_other = len(already_held) + higher_in_batch

                if other in already_held or other in same_batch:
                    reason = "已持倉/同批"
                elif pd.isna(exec_open):
                    reason = "T+1停牌"
                elif lots == 0:
                    reason = "張數=0"
                elif exec_lots == 0:
                    reason = "單張超限"
                elif slots_before_other >= cfg.max_positions:
                    reason = "持倉已滿"
                else:
                    reason = "無正當理由"

                higher_violations.append(f"{other}(roc={other_roc:.1f},{reason})")
                if reason == "無正當理由":
                    pass   # 這才是真正違規，c_pass 會設為 False

        c_pass = not any("無正當理由" in v for v in higher_violations)

        records.append({
            "ticker":              ticker,
            "entry_date":          entry_date.date(),
            "direction":           direction,
            "A_in_pool":           in_pool,
            "B_signal_ok":         signal_ok,
            "C_higher_no_signal":  c_pass,
            "C_violators":         ", ".join(higher_violations) if higher_violations else "-",
            "ALL_PASS":            in_pool and signal_ok and c_pass,
        })

    return pd.DataFrame(records)


# ══════════════════════════════════════════════
# Checkpoint 3：進出場價格 / 部位大小 / 停損驗證
# ══════════════════════════════════════════════

def check_execution(
    trades:   pd.DataFrame,
    raw_data: dict[str, pd.DataFrame],
    cfg:      TradingConfig,
) -> pd.DataFrame:
    """
    逐筆驗證進出場執行細節：

      D. 進場條件  : 訊號日 T 的進場條件確實成立
                     做多：Close > prev_High_N（突破）且 MA_Fast > MA_Slow（趨勢）
                     做空：Close < prev_Low_N（跌破）且 MA_Fast < MA_Slow（趨勢）
      E. 進場價格  : raw_entry_price == T+1 開盤價（允許浮點誤差）
      F. 部位大小  : lots 符合 equity × risk_pct ÷ ATR ÷ 1000，
                     並套用 max_trade_cost / 可用資金上限
      G. 出場價格  : raw_exit_price == 出場日 T+1 開盤價
      H. 停損觸發  : 出場訊號日 T 滿足追蹤停損條件
                     做多：Close < trail_high − atr_mult × ATR
                     做空：Close > trail_low  + atr_mult × ATR
    """
    if trades.empty:
        return pd.DataFrame()

    data      = {t: Indicators.add_all(df, cfg) for t, df in raw_data.items()}
    risk_mgr  = RiskManager(cfg)

    all_dates = sorted({d for df in data.values() for d in df.index})
    prev_of   = {d: all_dates[i-1] for i, d in enumerate(all_dates) if i > 0}

    records = []
    for _, trade in trades.iterrows():
        ticker     = trade["ticker"]
        direction  = trade["direction"]
        entry_date = pd.Timestamp(trade["entry_date"])
        exit_date  = pd.Timestamp(trade["exit_date"])
        raw_entry  = float(trade["raw_entry_price"])
        raw_exit   = float(trade["raw_exit_price"])
        adj_entry  = float(trade["adj_entry_price"])
        lots_actual = int(trade["lots"])
        atr_entry  = float(trade["atr_at_entry"])
        equity_ent = float(trade["equity_at_entry"])

        df = data.get(ticker)
        signal_date      = prev_of.get(entry_date)        # T（訊號日）
        prev_signal_date = prev_of.get(signal_date) if signal_date else None  # T-1

        # ── D. 進場條件：突破 + MA 趨勢 ──────────────
        d_ok   = False
        d_note = ""
        if (df is not None and signal_date and prev_signal_date
                and signal_date in df.index and prev_signal_date in df.index):
            row      = df.loc[signal_date]
            prev_row = df.loc[prev_signal_date]
            if isinstance(row,      pd.DataFrame): row      = row.iloc[-1]
            if isinstance(prev_row, pd.DataFrame): prev_row = prev_row.iloc[-1]
            close     = float(row.get("Close",   float("nan")))
            ma_fast   = float(row.get("MA_Fast", float("nan")))
            ma_slow   = float(row.get("MA_Slow", float("nan")))
            if direction == "long":
                high_n    = float(prev_row.get("High_N", float("nan")))
                breakout  = close > high_n
                trend     = ma_fast > ma_slow
                d_ok      = breakout and trend
                d_note    = (f"Close={close:.2f} > prev_High_N={high_n:.2f}({breakout}) "
                             f"MA_Fast={ma_fast:.2f} > MA_Slow={ma_slow:.2f}({trend})")
            else:
                low_n     = float(prev_row.get("Low_N",  float("nan")))
                breakdown = close < low_n
                trend     = ma_fast < ma_slow
                d_ok      = breakdown and trend
                d_note    = (f"Close={close:.2f} < prev_Low_N={low_n:.2f}({breakdown}) "
                             f"MA_Fast={ma_fast:.2f} < MA_Slow={ma_slow:.2f}({trend})")
        else:
            d_note = "no signal date data"

        # ── E. 進場價格 = T+1 開盤價 ──────────────────
        e_ok   = False
        e_note = ""
        if df is not None and entry_date in df.index:
            t1_open = df.loc[entry_date]
            if isinstance(t1_open, pd.DataFrame): t1_open = t1_open.iloc[-1]
            t1_open = float(t1_open.get("Open", float("nan")))
            e_ok   = abs(raw_entry - t1_open) < 1e-4
            e_note = f"raw={raw_entry} T+1open={t1_open:.4f}"
        else:
            e_note = "no data"

        # ── F. 部位大小 ───────────────────────────────
        f_ok   = False
        f_note = ""
        lots_exp = risk_mgr.position_size_lots(equity_ent, atr_entry)
        if lots_exp > 0:
            adj_e = adj_entry  # 已含滑價，直接使用
            cost  = adj_e * lots_exp * 1000
            if cost > cfg.max_trade_cost:
                lots_exp = max(int(cfg.max_trade_cost / (adj_e * 1000)), 0)
            if lots_exp > 0 and adj_e * lots_exp * 1000 > equity_ent:
                lots_exp = max(int(equity_ent / (adj_e * 1000)), 0)
        f_ok   = (lots_exp == lots_actual)
        f_note = f"expected={lots_exp} actual={lots_actual}"

        # ── G. 出場價格 = 出場日 T+1 開盤價 ──────────
        g_ok   = False
        g_note = ""
        if df is not None and exit_date in df.index:
            exit_open = df.loc[exit_date]
            if isinstance(exit_open, pd.DataFrame): exit_open = exit_open.iloc[-1]
            exit_open = float(exit_open.get("Open", float("nan")))
            g_ok   = abs(raw_exit - exit_open) < 1e-4
            g_note = f"raw={raw_exit} exit_open={exit_open:.4f}"
        else:
            g_note = "no data"

        # ── H. 停損訊號觸發（重建 trail） ────────────
        h_ok   = False
        h_note = ""
        signal_exit_date = prev_of.get(exit_date)
        if df is not None and signal_exit_date and signal_exit_date in df.index:
            period_dates = [d for d in all_dates
                            if entry_date <= d <= signal_exit_date]
            if direction == "long":
                trail = raw_entry
                for d in period_dates:
                    if d not in df.index: continue
                    r = df.loc[d]
                    if isinstance(r, pd.DataFrame): r = r.iloc[-1]
                    trail = max(trail, float(r.get("High", trail)))
                exit_row = df.loc[signal_exit_date]
                if isinstance(exit_row, pd.DataFrame): exit_row = exit_row.iloc[-1]
                close = float(exit_row.get("Close", float("nan")))
                atr   = float(exit_row.get("ATR",   float("nan")))
                stop  = trail - cfg.atr_multiplier * atr
                h_ok   = (close < stop)
                h_note = (f"Close={close:.2f} trail_high={trail:.2f} "
                          f"stop={stop:.2f} triggered={h_ok}")
            else:
                trail = raw_entry
                for d in period_dates:
                    if d not in df.index: continue
                    r = df.loc[d]
                    if isinstance(r, pd.DataFrame): r = r.iloc[-1]
                    trail = min(trail, float(r.get("Low", trail)))
                exit_row = df.loc[signal_exit_date]
                if isinstance(exit_row, pd.DataFrame): exit_row = exit_row.iloc[-1]
                close = float(exit_row.get("Close", float("nan")))
                atr   = float(exit_row.get("ATR",   float("nan")))
                stop  = trail + cfg.atr_multiplier * atr
                h_ok   = (close > stop)
                h_note = (f"Close={close:.2f} trail_low={trail:.2f} "
                          f"stop={stop:.2f} triggered={h_ok}")
        else:
            h_note = "no exit signal date"

        all_pass = d_ok and e_ok and f_ok and g_ok and h_ok
        records.append({
            "ticker":     ticker,
            "direction":  direction,
            "entry_date": entry_date.date(),
            "exit_date":  exit_date.date(),
            "D_entry_cond":   d_ok,
            "E_entry_price":  e_ok,
            "F_lots":         f_ok,
            "G_exit_price":   g_ok,
            "H_stop_trigger": h_ok,
            "ALL_PASS":       all_pass,
            "D_note":         d_note,
            "E_note":         e_note,
            "F_note":         f_note,
            "G_note":         g_note,
            "H_note":         h_note,
        })

    return pd.DataFrame(records)


# ══════════════════════════════════════════════
# 主驗證入口
# ══════════════════════════════════════════════

def run_checkpoint(
    results:      dict,
    raw_data:     dict[str, pd.DataFrame],
    cfg:          TradingConfig,
    sample_dates: list[str] | None = None,
) -> None:
    """
    執行兩個 checkpoint 並印出結果。

    Parameters
    ----------
    results      : run_backtest() 的回傳值
    raw_data     : 原始 OHLCV dict（與回測相同）
    cfg          : TradingConfig
    sample_dates : 要抽查的選股日期清單（字串），None = 自動取每季第一天
    """
    trades = results.get("trades", pd.DataFrame())
    data   = {t: Indicators.add_all(df, cfg) for t, df in raw_data.items()}

    print("\n" + "=" * 60)
    print("  CHECKPOINT 1 — 選股池驗證")
    print("=" * 60)

    # 自動取樣：每季第一筆交易的進場日
    if sample_dates is None:
        if not trades.empty:
            df_t = trades.copy()
            df_t["entry_date"] = pd.to_datetime(df_t["entry_date"])
            df_t["quarter"]    = df_t["entry_date"].dt.to_period("Q")
            sample_dates = (
                df_t.groupby("quarter")["entry_date"]
                .min()
                .dt.strftime("%Y-%m-%d")
                .tolist()[:4]   # 最多抽4季
            )
        else:
            sample_dates = []

    uni_flt = UniverseFilter(cfg)
    for date_str in sample_dates:
        date       = pd.Timestamp(date_str)
        candidates = uni_flt.filter(data, date)
        cp1        = check_universe(data, date, candidates, cfg)

        fail_count = (~cp1["ALL_PASS"]).sum() if not cp1.empty else 0
        print(f"\n  [{date_str}] 選股池 {len(candidates)} 支，"
              f"驗證通過 {len(cp1) - fail_count}/{len(cp1)}")

        if fail_count > 0:
            print("  FAIL 項目：")
            print(cp1[~cp1["ALL_PASS"]].to_string(index=False))
        else:
            display = (
                cp1[["ticker", "close", "close/52w_pct", "avg_amount_20", "roc_avg", "ALL_PASS"]]
                .sort_values("roc_avg", ascending=False)
                .reset_index(drop=True)
            )
            display.index += 1   # ROC 排名從 1 開始
            display.index.name = "rank"
            print(display.to_string())

    print("\n" + "=" * 60)
    print("  CHECKPOINT 2 — 進出場驗證")
    print("=" * 60)

    if trades.empty:
        print("  無交易紀錄。")
        return

    cp2   = check_trades(trades, raw_data, cfg)
    total = len(cp2)
    passed = cp2["ALL_PASS"].sum()
    failed = total - passed

    print(f"\n  共 {total} 筆交易  PASS {passed}  FAIL {failed}")
    print(f"  A 在選股池內          : {cp2['A_in_pool'].sum()}/{total}")
    print(f"  B 訊號確實觸發        : {cp2['B_signal_ok'].sum()}/{total}")
    print(f"  C 更高ROC股票無訊號   : {cp2['C_higher_no_signal'].sum()}/{total}")

    if failed > 0:
        print("\n  FAIL 項目：")
        print(cp2[~cp2["ALL_PASS"]].to_string(index=False))
    else:
        print("\n  抽樣 5 筆（全部通過）：")
        print(cp2.sample(min(5, total), random_state=42).to_string(index=False))

    print("\n" + "=" * 60)
    print("  CHECKPOINT 3 — 進出場執行驗證")
    print("=" * 60)

    cp3   = check_execution(trades, raw_data, cfg)
    total3 = len(cp3)
    passed3 = cp3["ALL_PASS"].sum()
    failed3 = total3 - passed3

    print(f"\n  共 {total3} 筆交易  PASS {passed3}  FAIL {failed3}")
    print(f"  D 進場條件正確        : {cp3['D_entry_cond'].sum()}/{total3}")
    print(f"  E 進場價格正確        : {cp3['E_entry_price'].sum()}/{total3}")
    print(f"  F 部位大小正確        : {cp3['F_lots'].sum()}/{total3}")
    print(f"  G 出場價格正確        : {cp3['G_exit_price'].sum()}/{total3}")
    print(f"  H 停損確實觸發        : {cp3['H_stop_trigger'].sum()}/{total3}")

    if failed3 > 0:
        print("\n  FAIL 明細：")
        fail_rows = cp3[~cp3["ALL_PASS"]]
        for _, row in fail_rows.iterrows():
            print(f"\n  {row['ticker']} {row['direction']} "
                  f"{row['entry_date']} → {row['exit_date']}")
            if not row["D_entry_cond"]:
                print(f"    D FAIL: {row['D_note']}")
            if not row["E_entry_price"]:
                print(f"    E FAIL: {row['E_note']}")
            if not row["F_lots"]:
                print(f"    F FAIL: {row['F_note']}")
            if not row["G_exit_price"]:
                print(f"    G FAIL: {row['G_note']}")
            if not row["H_stop_trigger"]:
                print(f"    H FAIL: {row['H_note']}")
    else:
        print("\n  所有執行細節驗證通過，詳細數值：")
        print(cp3[["ticker","direction","entry_date","exit_date",
                   "D_note","E_note","F_note","G_note","H_note"]].to_string(index=False))

    print("\n" + "=" * 60 + "\n")


# ══════════════════════════════════════════════
# 抽樣檢視：指定時間點的選股池 + 策略結果
# ══════════════════════════════════════════════

def sample_period(
    date_str: str,
    results:  dict,
    raw_data: dict[str, pd.DataFrame],
    cfg:      TradingConfig,
) -> None:
    """
    給定一個日期，顯示：
      1. 當天選股池（依 ROC 排名）
      2. 每支股票後來的策略結果（有沒有進場、進出場日、損益）

    Parameters
    ----------
    date_str : 要抽查的日期，格式 "YYYY-MM-DD"
    results  : run_backtest() 的回傳值
    raw_data : 原始 OHLCV dict
    cfg      : TradingConfig
    """
    exec_date = pd.Timestamp(date_str)   # 你傳入的日期（entry_date 或 signal_date 皆可）
    data      = {t: Indicators.add_all(df, cfg) for t, df in raw_data.items()}
    trades    = results.get("trades", pd.DataFrame())

    all_dates = sorted({d for df in data.values() for d in df.index})

    # 自動判斷：若傳入的是 entry_date（T+1），往前一個交易日找 signal_date（T）
    # 若傳入的本來就是 signal_date，prev_date 就是 T-1
    if exec_date in all_dates:
        # 傳入的是有交易資料的日期 → 當作 entry_date，往前找 signal_date
        date      = next((d for d in reversed(all_dates) if d < exec_date), exec_date)
    else:
        date      = exec_date   # 傳入的不是交易日，直接用（容錯）

    prev_date = next((d for d in reversed(all_dates) if d < date), None)

    # ── 1. 選股池 ──────────────────────────────
    uni_flt    = UniverseFilter(cfg)
    sig_gen    = SignalGenerator()
    candidates = uni_flt.filter(data, date)
    pool_df    = check_universe(data, date, candidates, cfg)

    if pool_df.empty:
        print(f"\n[{date_str}] 選股池為空，無候選股票。")
        return

    # 每支股票加上當日訊號狀態
    def get_signal(ticker):
        df = data.get(ticker)
        if df is None or date not in df.index or prev_date not in df.index:
            return "-"
        row      = df.loc[date]
        prev_row = df.loc[prev_date]
        if isinstance(row,      pd.DataFrame): row      = row.iloc[-1]
        if isinstance(prev_row, pd.DataFrame): prev_row = prev_row.iloc[-1]
        if sig_gen.long_entry(row, prev_row):
            return "LONG"
        if sig_gen.short_entry(row, prev_row):
            return "SHORT"
        return "none"

    pool_df = (
        pool_df[["ticker", "close", "close/52w_pct", "avg_amount_20", "roc_avg"]]
        .sort_values("roc_avg", ascending=False)
        .reset_index(drop=True)
    )
    pool_df["signal"] = pool_df["ticker"].apply(get_signal)
    pool_df.index += 1
    pool_df.index.name = "rank"

    print(f"\n{'=' * 60}")
    print(f"  你傳入：{date_str}  →  選股池日期：{date.date()}（訊號日）")
    print(f"  選股池共 {len(pool_df)} 支（依 ROC 排名）")
    print(f"{'=' * 60}")
    print(pool_df.to_string())

    # ── 2. 策略結果 ────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"  策略結果（訊號觸發者的進出場明細）")
    print(f"{'─' * 60}")

    if trades.empty:
        print("  無任何交易紀錄。")
        return

    df_t = trades.copy()
    df_t["entry_date"] = pd.to_datetime(df_t["entry_date"])
    df_t["exit_date"]  = pd.to_datetime(df_t["exit_date"])

    rows = []
    for rank, row in pool_df.iterrows():
        ticker = row["ticker"]

        t = df_t[
            (df_t["ticker"] == ticker) &
            (df_t["entry_date"] >= date)
        ].sort_values("entry_date")

        if t.empty:
            rows.append({
                "rank":       rank,
                "ticker":     ticker,
                "roc_avg":    row["roc_avg"],
                "signal":     row["signal"],
                "entered":    False,
                "entry_date": "-",
                "exit_date":  "-",
                "hold_days":  "-",
                "pnl_net":    "-",
            })
        else:
            tr = t.iloc[0]
            rows.append({
                "rank":       rank,
                "ticker":     ticker,
                "roc_avg":    row["roc_avg"],
                "signal":     row["signal"],
                "entered":    True,
                "entry_date": str(tr["entry_date"].date()),
                "exit_date":  str(tr["exit_date"].date()),
                "hold_days":  int(tr["hold_days"]),
                "pnl_net":    round(tr["pnl_net"], 0),
            })

    result_df = pd.DataFrame(rows).set_index("rank")
    print(result_df.to_string())

    # ── 3. 診斷：訊號觸發但沒進場的原因 ──────────
    missed = [r for r in rows if r["signal"] != "none" and not r["entered"]]
    if missed:
        from risk_manager import RiskManager
        risk_mgr = RiskManager(cfg)

        # 計算該日期當下已有幾個持倉（進場日 <= date < 出場日）
        active_count = len(df_t[
            (df_t["entry_date"] <= date) & (df_t["exit_date"] > date)
        ])

        print(f"\n{'─' * 60}")
        print(f"  診斷：訊號觸發但未進場的股票")
        print(f"  當時已有持倉：{active_count} / {cfg.max_positions}")
        print(f"{'─' * 60}")

        for r in missed:
            ticker = r["ticker"]
            df_s   = data.get(ticker)
            reason = []

            if active_count >= cfg.max_positions:
                reason.append(f"持倉已滿（{active_count}/{cfg.max_positions}）")

            if df_s is not None and date in df_s.index:
                row_s = df_s.loc[date]
                if isinstance(row_s, pd.DataFrame): row_s = row_s.iloc[-1]
                atr   = float(row_s.get("ATR", 0) or 0)
                close = float(row_s.get("Close", 0) or 0)
                lots  = risk_mgr.position_size_lots(cfg.initial_equity, atr) if atr > 0 else 0
                cost  = close * 1000 * lots if lots > 0 else close * 1000

                if atr <= 0:
                    reason.append("ATR = 0，無法計算張數")
                elif lots == 0:
                    risk_amt = cfg.initial_equity * cfg.risk_pct
                    reason.append(
                        f"張數 = 0（風險金額 {risk_amt:.0f} / ATR {atr:.2f} / 1000 < 1）"
                    )
                elif cost > cfg.max_trade_cost:
                    reason.append(
                        f"單筆成本 {cost:.0f} > max_trade_cost {cfg.max_trade_cost:.0f}"
                    )

            print(f"  [{r['rank']}] {ticker}  ->  {'、'.join(reason) if reason else '原因不明'}")

    print()

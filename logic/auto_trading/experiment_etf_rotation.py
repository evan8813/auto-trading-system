"""
experiment_etf_rotation.py
──────────────────────────
ETF 輪換策略實驗：測試「熊市轉進台灣債券 ETF」是否優於原本的空倉。

策略邏輯：
  大盤 > 200 EMA  →  沿用主策略的每日 equity（個股選股）
  大盤 < 200 EMA  →  把 equity 改投入債券 ETF，跟著 ETF 每日漲跌

執行方式：
  python experiment_etf_rotation.py <equity_csv> <data_folder>

  equity_csv  : 主策略跑出的 output/equity_curve.csv（每日淨值）
  data_folder : 存放 CSV 的資料夾（需包含 taiex.csv 和 bond ETF 的 CSV）

輸出：
  output/etf_rotation/comparison.png   ── 兩條 equity 曲線對比
  output/etf_rotation/metrics.txt      ── 兩者績效數字比較

注意：
  - 不修改 main.py / backtester.py，完全獨立
  - 債券 ETF 代號預設 00679B，可在下方 BOND_TICKER 修改
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

matplotlib.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 設定 ──────────────────────────────────────────────────────────────
BOND_TICKER  = "00679B"   # 債券 ETF 代號（改這裡換別的 ETF）
EMA_PERIOD   = 200        # 大盤 EMA 週期
TAIEX_FILE   = "taiex.csv"
# ─────────────────────────────────────────────────────────────────────


def load_csv(path: Path) -> pd.DataFrame:
    """讀取 TWSE 格式 CSV，回傳含 Date index 的 DataFrame。"""
    col_map = {
        "日期": "Date", "成交股數": "Volume", "成交金額": "Amount",
        "開盤價": "Open", "最高價": "High", "最低價": "Low",
        "收盤價": "Close", "漲跌價差": "Change", "成交筆數": "Transactions",
    }
    invalid = {"--", "除息", "除權", "除權息", ""}

    for enc in ("utf-8-sig", "cp950", "utf-8"):
        try:
            df = pd.read_csv(path, encoding=enc, dtype=str)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"無法讀取 {path}")

    df.columns = [col_map.get(c.strip(), c.strip()) for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"].str.strip(), errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].str.replace(",", "").replace(invalid, np.nan),
                errors="coerce",
            )
    return df


def load_equity_curve(path: str) -> pd.Series:
    """讀取 equity_curve.csv，回傳 pd.Series（index=Date, values=equity）。"""
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    col = "equity" if "equity" in df.columns else df.columns[0]
    return df[col].astype(float)


def apply_etf_rotation(
    equity: pd.Series,
    taiex_close: pd.Series,
    bond_close: pd.Series,
) -> pd.Series:
    """
    熊市期間將 equity 改為跟隨債券 ETF 漲跌。

    Parameters
    ----------
    equity      : 主策略逐日 equity（空倉期間 equity 不變）
    taiex_close : 加權指數收盤
    bond_close  : 債券 ETF 收盤
    """
    ema = taiex_close.ewm(span=EMA_PERIOD, adjust=False).mean()
    bull = taiex_close > ema   # True = 牛市

    bond_ret = bond_close.pct_change().fillna(0)

    rotated = equity.copy()
    dates = equity.index

    for i in range(1, len(dates)):
        today = dates[i]
        prev  = dates[i - 1]

        in_bull = bull.get(prev, True)   # 前一日判斷（避免 look-ahead）

        if in_bull:
            # 牛市：equity 維持主策略結果（不動）
            pass
        else:
            # 熊市：equity 跟著債券 ETF 的日報酬走
            bond_daily = bond_ret.get(today, 0.0)
            rotated.iloc[i] = rotated.iloc[i - 1] * (1 + bond_daily)

    return rotated


def calc_metrics(equity: pd.Series, label: str) -> dict:
    ret = equity.pct_change().dropna()
    total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = ((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) * 100

    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    mdd = dd.min() * 100

    sharpe = (ret.mean() / ret.std() * (252 ** 0.5)) if ret.std() > 0 else 0
    calmar = cagr / abs(mdd) if mdd != 0 else 0

    return {
        "標籤": label,
        "累積報酬 (%)": round(total_return, 2),
        "年化報酬 (%)": round(cagr, 2),
        "最大回撤 (%)": round(mdd, 2),
        "Sharpe": round(sharpe, 3),
        "Calmar": round(calmar, 3),
    }


def plot_comparison(original: pd.Series, rotated: pd.Series, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))

    norm_orig    = original / original.iloc[0] * 100
    norm_rotated = rotated  / rotated.iloc[0]  * 100

    ax.plot(norm_orig.index,    norm_orig.values,    label="原策略（熊市空倉）", linewidth=1.5)
    ax.plot(norm_rotated.index, norm_rotated.values, label=f"ETF 輪換（熊市持 {BOND_TICKER}）",
            linewidth=1.5, linestyle="--")

    ax.set_title("ETF 輪換策略 vs 原策略")
    ax.set_ylabel("淨值（基準 = 100）")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = out_dir / "comparison.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"圖表已儲存：{save_path}")


def main() -> None:
    if len(sys.argv) < 3:
        print("用法：python experiment_etf_rotation.py <equity_csv> <data_folder>")
        print("範例：python experiment_etf_rotation.py output/equity_curve.csv /path/to/stocks")
        sys.exit(1)

    equity_path  = sys.argv[1]
    data_folder  = Path(sys.argv[2])
    out_dir      = Path("output/etf_rotation")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 載入資料 ──────────────────────────────────────────────────────
    print("載入 equity curve...")
    equity = load_equity_curve(equity_path)

    print("載入加權指數...")
    taiex_path = data_folder / TAIEX_FILE
    if not taiex_path.exists():
        taiex_path = data_folder.parent / TAIEX_FILE
    taiex = load_csv(taiex_path)["Close"].dropna()

    print(f"載入債券 ETF ({BOND_TICKER})...")
    bond_candidates = [
        data_folder / f"{BOND_TICKER}.csv",
        data_folder / f"{BOND_TICKER}_adj.csv",
    ]
    bond_path = next((p for p in bond_candidates if p.exists()), None)
    if bond_path is None:
        print(f"找不到 {BOND_TICKER}.csv，請確認 data_folder 中有此檔案。")
        sys.exit(1)
    bond = load_csv(bond_path)["Close"].dropna()

    # ── 對齊日期 ──────────────────────────────────────────────────────
    common = equity.index.intersection(taiex.index).intersection(bond.index)
    equity = equity.loc[common]
    taiex  = taiex.loc[common]
    bond   = bond.loc[common]

    if len(common) == 0:
        print("日期無交集，請確認三份資料的時間範圍有重疊。")
        sys.exit(1)

    print(f"有效回測區間：{common[0].date()} ~ {common[-1].date()}（{len(common)} 天）")

    # ── 執行輪換 ──────────────────────────────────────────────────────
    print("計算 ETF 輪換 equity...")
    rotated = apply_etf_rotation(equity, taiex, bond)

    # ── 輸出 ─────────────────────────────────────────────────────────
    plot_comparison(equity, rotated, out_dir)

    m1 = calc_metrics(equity,  "原策略（空倉）")
    m2 = calc_metrics(rotated, f"ETF 輪換（{BOND_TICKER}）")

    print("\n═══ 績效比較 ═══")
    df_metrics = pd.DataFrame([m1, m2]).set_index("標籤")
    print(df_metrics.to_string())

    metrics_path = out_dir / "metrics.txt"
    df_metrics.to_string(open(metrics_path, "w", encoding="utf-8"))
    print(f"\n績效已儲存：{metrics_path}")


if __name__ == "__main__":
    main()

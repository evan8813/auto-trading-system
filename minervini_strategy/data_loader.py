"""
data_loader.py
載入 stocks_full/ 所有 CSV，輸出寬格式 DataFrame（日期 × 股票代號）。
"""
from __future__ import annotations

import os
import pandas as pd

_DEFAULT_DIR = os.path.join(os.path.dirname(__file__), "..", "stocks_full")

_COL_MAP = {
    "日期":   "date",
    "開盤價":  "open",
    "最高價":  "high",
    "最低價":  "low",
    "收盤價":  "close",
    "成交股數": "volume",
    "成交金額": "amount",
}


def load_data(csv_dir: str = _DEFAULT_DIR) -> dict[str, pd.DataFrame]:
    """
    讀取資料夾內所有 *.csv，回傳：
        data["close"]  — 收盤價（日期 × 股票代號）
        data["high"]   — 最高價
        data["low"]    — 最低價
        data["open"]   — 開盤價
        data["volume"] — 成交股數
        data["amount"] — 成交金額
    """
    buckets: dict[str, dict] = {k: {} for k in _COL_MAP.values() if k != "date"}

    for fname in sorted(os.listdir(csv_dir)):
        if not fname.endswith(".csv"):
            continue
        sid = fname[:-4]
        path = os.path.join(csv_dir, fname)
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
            df.rename(columns=_COL_MAP, inplace=True)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df.dropna(subset=["date"], inplace=True)
            df.set_index("date", inplace=True)
            df = df[~df.index.duplicated(keep="last")]
            df.sort_index(inplace=True)

            for col in buckets:
                if col in df.columns:
                    series = pd.to_numeric(
                        df[col].astype(str)
                        .str.replace(",", "", regex=False)
                        .str.strip()
                        .replace({"--": None, "除息": None, "除權": None, "除權息": None, "": None}),
                        errors="coerce",
                    )
                    buckets[col][sid] = series
        except Exception:
            continue

    data = {col: pd.DataFrame(d).sort_index() for col, d in buckets.items()}

    n_stocks = len(data["close"].columns)
    n_days   = len(data["close"])
    print(f"[data_loader] 載入完成：{n_stocks} 檔股票，{n_days} 個交易日")
    return data

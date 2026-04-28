"""
data_loader.py
──────────────
職責：將 TWSE CSV 原始檔案讀入並轉換成標準化 OHLCV DataFrame。
      只負責「讀檔 + 清洗」，不做任何指標計算或策略邏輯。

擴充指引：
  - 新增資料來源（例如 Yahoo Finance / TEJ）→ 繼承 DataLoader 並覆寫 _load_one
  - 新增欄位對應 → 修改 COL_MAP
  - 調整最少資料列數門檻 → 修改 load_folder 中的 MIN_ROWS
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 最少需要多少列才視為有效資料（需足夠計算 MA100、52W High 等指標）
_MIN_ROWS = 120


class DataLoader:
    """
    載入 TWSE 歷史 OHLCV 資料。

    CSV 欄位格式：
      日期, 成交股數, 成交金額, 開盤價, 最高價, 最低價, 收盤價, 漲跌價差, 成交筆數

    輸出標準化欄位：
      Open, High, Low, Close, Volume（成交股數）, Amount（成交金額）

    關於「還原股價」：
      - 若資料夾內有同名的 {ticker}_adj.csv，優先載入（調整後股價）。
      - 否則 fallback 到 {ticker}.csv（原始股價）。
      - 建議先用外部工具產生 *_adj.csv 以提升回測準確度。
    """

    # TWSE CSV 原始欄位 → 標準化名稱
    COL_MAP: dict[str, str] = {
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

    # 數值欄位中代表「無效 / 停牌」的字串集合
    _INVALID_VALUES: frozenset[str] = frozenset(
        {"--", "除息", "除權", "除權息", ""}
    )

    # ── 公開 API ──────────────────────────────

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
        adjusted : True = 優先使用 {ticker}_adj.csv；
                   False = 強制使用原始 CSV。
        """
        folder_path = Path(folder)
        if not folder_path.exists():
            raise FileNotFoundError(f"資料夾不存在：{folder}")

        if tickers is None:
            tickers = cls._scan_tickers(folder_path)
            logger.info(f"自動掃描到 {len(tickers)} 支股票。")

        data: dict[str, pd.DataFrame] = {}
        skipped = 0

        for ticker in tickers:
            df = cls._load_one(folder_path, ticker, adjusted)
            if df is not None and len(df) >= _MIN_ROWS:
                data[ticker] = df
            else:
                skipped += 1

        logger.info(
            f"成功載入 {len(data)} 支股票資料"
            f"（跳過 {skipped} 支資料不足或格式錯誤）。"
        )
        return data

    @staticmethod
    def generate_synthetic(
        tickers: list[str],
        start: str = "2010-01-01",
        end:   str = "2023-12-29",
        seed:  int = 42,
    ) -> dict[str, pd.DataFrame]:
        """
        產生合成隨機資料（用於單元測試 / CI，不代表真實市場）。
        無 CSV 時可用此函式驗證策略流程。
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

    # ── 私有輔助 ──────────────────────────────

    @staticmethod
    def _scan_tickers(folder: Path) -> list[str]:
        """掃描資料夾，排除 _adj 結尾的調整後檔案"""
        return [
            p.stem for p in sorted(folder.glob("*.csv"))
            if not p.stem.endswith("_adj")
        ]

    @classmethod
    def _load_one(
        cls,
        folder: Path,
        ticker: str,
        prefer_adjusted: bool,
    ) -> Optional[pd.DataFrame]:
        """載入單一股票 CSV，回傳標準化 DataFrame 或 None。"""
        path, is_adj = cls._resolve_path(folder, ticker, prefer_adjusted)
        if path is None:
            logger.debug(f"找不到 {ticker}.csv，略過。")
            return None

        try:
            encoding = cls._detect_encoding(path)
            df = pd.read_csv(path, dtype=str, encoding=encoding)
            df = cls._rename_columns(df, ticker)
            if df is None:
                return None
            df = cls._parse_dates(df)
            df = cls._cast_numerics(df)
            df = cls._filter_invalid_rows(df)
            df.attrs["ticker"]      = ticker
            df.attrs["is_adjusted"] = is_adj
            return df

        except Exception as e:
            logger.warning(f"{ticker}: 載入失敗（{e}），略過。")
            return None

    @staticmethod
    def _resolve_path(
        folder: Path,
        ticker: str,
        prefer_adjusted: bool,
    ) -> tuple[Optional[Path], bool]:
        """決定要讀哪個 CSV 檔（adj 優先或強制原始）"""
        adj_path = folder / f"{ticker}_adj.csv"
        raw_path = folder / f"{ticker}.csv"

        if prefer_adjusted and adj_path.exists():
            return adj_path, True
        if raw_path.exists():
            return raw_path, False
        return None, False

    @classmethod
    def _rename_columns(
        cls,
        df: pd.DataFrame,
        ticker: str,
    ) -> Optional[pd.DataFrame]:
        """重命名欄位並檢查必要欄位是否存在"""
        df.rename(columns=cls.COL_MAP, inplace=True)
        required = ["Date", "Open", "High", "Low", "Close", "Volume"]
        missing  = [c for c in required if c not in df.columns]
        if missing:
            logger.warning(f"{ticker}: 缺少欄位 {missing}，略過。")
            return None
        return df

    @staticmethod
    def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
        """解析日期欄並設為 index"""
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df.dropna(subset=["Date"], inplace=True)
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)
        return df

    @classmethod
    def _cast_numerics(cls, df: pd.DataFrame) -> pd.DataFrame:
        """去除千分位逗號、特殊字串，轉換為 float"""
        for col in ["Open", "High", "Low", "Close", "Volume", "Amount"]:
            if col not in df.columns:
                continue
            df[col] = (
                df[col]
                .str.replace(",", "", regex=False)
                .str.strip()
                .apply(lambda x: np.nan if x in cls._INVALID_VALUES else x)
                .pipe(pd.to_numeric, errors="coerce")
            )
        return df

    @staticmethod
    def _filter_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
        """移除停牌 / 下市行，並確保 High >= Low"""
        df.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
        df = df[df["High"] >= df["Low"]]
        return df

    @staticmethod
    def _detect_encoding(path: Path) -> str:
        """
        自動偵測 CSV 編碼。
        偵測順序：utf-8-sig → cp950（Big5）→ utf-8
        """
        try:
            raw = path.read_bytes()[:4096]
        except OSError:
            return "utf-8"

        for enc in ("utf-8-sig", "cp950", "utf-8"):
            try:
                raw.decode(enc)
                return enc
            except (UnicodeDecodeError, LookupError):
                continue

        return "utf-8"

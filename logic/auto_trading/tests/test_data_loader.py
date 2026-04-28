"""
test_data_loader.py
───────────────────
步驟 8：驗證 DataLoader 的 CSV 讀取與資料清洗流程。

測試項目：
  1. generate_synthetic() ── 回傳正確欄位、日期範圍、形狀
  2. _scan_tickers()      ── 掃描資料夾排除 _adj 結尾
  3. _rename_columns()    ── 中文欄位對應正確
  4. _detect_encoding()   ── 正確偵測 utf-8-sig / cp950
  5. _filter_invalid_rows() ── 移除 Open/High/Low/Close 為 NaN 的列
  6. _filter_invalid_rows() ── 移除 High < Low 的列
  7. load_folder() ── 整合測試：讀取模擬 CSV 檔
  8. load_folder() ── 資料不足時（< 120列）略過該股票
"""

import io
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from data_loader import DataLoader


# ── 共用輔助 ──────────────────────────────────

def make_twse_csv_content(rows: int = 150) -> str:
    """產生模擬 TWSE CSV 內容（繁體中文欄位名稱）"""
    dates = pd.bdate_range("2020-01-01", periods=rows)
    lines = ["日期,成交股數,成交金額,開盤價,最高價,最低價,收盤價,漲跌價差,成交筆數"]
    for d in dates:
        close = 100.0
        lines.append(
            f"{d.strftime('%Y-%m-%d')},"
            f"10000,1000000,{close-0.5:.2f},{close+1:.2f},{close-1:.2f},{close:.2f},0.5,500"
        )
    return "\n".join(lines)


class TestGenerateSynthetic:
    """generate_synthetic()"""

    def test_returns_dict(self):
        data = DataLoader.generate_synthetic(["2330", "0050"])
        assert isinstance(data, dict)

    def test_correct_tickers(self):
        data = DataLoader.generate_synthetic(["2330", "0050"])
        assert set(data.keys()) == {"2330", "0050"}

    def test_has_ohlcv_columns(self):
        data = DataLoader.generate_synthetic(["2330"])
        df   = data["2330"]
        for col in ["Open", "High", "Low", "Close", "Volume", "Amount"]:
            assert col in df.columns, f"缺少欄位：{col}"

    def test_date_range(self):
        data  = DataLoader.generate_synthetic(["2330"], start="2020-01-01", end="2020-12-31")
        df    = data["2330"]
        assert df.index[0]  >= pd.Timestamp("2020-01-01")
        assert df.index[-1] <= pd.Timestamp("2020-12-31")

    def test_reproducible_with_same_seed(self):
        d1 = DataLoader.generate_synthetic(["2330"], seed=42)
        d2 = DataLoader.generate_synthetic(["2330"], seed=42)
        pd.testing.assert_frame_equal(d1["2330"], d2["2330"])

    def test_different_seed_different_data(self):
        d1 = DataLoader.generate_synthetic(["2330"], seed=1)
        d2 = DataLoader.generate_synthetic(["2330"], seed=2)
        assert not d1["2330"]["Close"].equals(d2["2330"]["Close"])

    def test_high_ge_low(self):
        data = DataLoader.generate_synthetic(["2330"])
        df   = data["2330"]
        assert (df["High"] >= df["Low"]).all()

    def test_attrs_ticker_set(self):
        data = DataLoader.generate_synthetic(["2330"])
        assert data["2330"].attrs.get("ticker") == "2330"

    def test_attrs_is_adjusted(self):
        data = DataLoader.generate_synthetic(["2330"])
        assert data["2330"].attrs.get("is_adjusted") is True


class TestScanTickers:
    """_scan_tickers() — 使用 tmp_path"""

    def test_returns_ticker_names(self, tmp_path):
        (tmp_path / "2330.csv").touch()
        (tmp_path / "0050.csv").touch()
        tickers = DataLoader._scan_tickers(tmp_path)
        assert "2330" in tickers
        assert "0050" in tickers

    def test_excludes_adj_files(self, tmp_path):
        (tmp_path / "2330.csv").touch()
        (tmp_path / "2330_adj.csv").touch()
        tickers = DataLoader._scan_tickers(tmp_path)
        assert "2330_adj" not in tickers
        assert "2330" in tickers

    def test_empty_folder(self, tmp_path):
        tickers = DataLoader._scan_tickers(tmp_path)
        assert tickers == []


class TestRenameColumns:
    """_rename_columns()"""

    def test_renames_chinese_to_english(self):
        df = pd.DataFrame(columns=["日期", "成交股數", "成交金額",
                                    "開盤價", "最高價", "最低價", "收盤價"])
        result = DataLoader._rename_columns(df, "2330")
        assert "Open" in result.columns
        assert "High" in result.columns
        assert "Low"  in result.columns
        assert "Close" in result.columns

    def test_missing_required_column_returns_none(self):
        # 缺少 Close（收盤價）
        df = pd.DataFrame(columns=["日期", "成交股數", "成交金額", "開盤價", "最高價", "最低價"])
        result = DataLoader._rename_columns(df, "2330")
        assert result is None

    def test_all_required_columns_present(self):
        df = pd.DataFrame(columns=["日期", "成交股數", "成交金額",
                                    "開盤價", "最高價", "最低價", "收盤價"])
        result = DataLoader._rename_columns(df, "2330")
        assert result is not None
        for col in ["Date", "Open", "High", "Low", "Close", "Volume"]:
            assert col in result.columns


class TestFilterInvalidRows:
    """_filter_invalid_rows()"""

    def test_removes_nan_close_rows(self):
        df = pd.DataFrame({
            "Open":  [100.0, np.nan],
            "High":  [105.0, 110.0],
            "Low":   [95.0,  100.0],
            "Close": [102.0, np.nan],
        }, index=pd.bdate_range("2023-01-01", periods=2))
        result = DataLoader._filter_invalid_rows(df)
        assert len(result) == 1

    def test_removes_rows_where_high_less_than_low(self):
        df = pd.DataFrame({
            "Open":  [100.0, 100.0],
            "High":  [105.0, 90.0],   # 第 2 行 High < Low → 無效
            "Low":   [95.0,  95.0],
            "Close": [102.0, 92.0],
        }, index=pd.bdate_range("2023-01-01", periods=2))
        result = DataLoader._filter_invalid_rows(df)
        assert len(result) == 1
        assert result.iloc[0]["High"] == 105.0

    def test_keeps_valid_rows(self):
        df = pd.DataFrame({
            "Open":  [100.0, 101.0],
            "High":  [105.0, 106.0],
            "Low":   [95.0,  96.0],
            "Close": [102.0, 103.0],
        }, index=pd.bdate_range("2023-01-01", periods=2))
        result = DataLoader._filter_invalid_rows(df)
        assert len(result) == 2


class TestDetectEncoding:
    """_detect_encoding()"""

    def test_detects_utf8_sig(self, tmp_path):
        f = tmp_path / "test_utf8sig.csv"
        f.write_bytes("日期,收盤價\n".encode("utf-8-sig"))
        enc = DataLoader._detect_encoding(f)
        assert enc == "utf-8-sig"

    def test_detects_cp950(self, tmp_path):
        f = tmp_path / "test_cp950.csv"
        f.write_bytes("日期,收盤價\n".encode("cp950"))
        enc = DataLoader._detect_encoding(f)
        assert enc == "cp950"

    def test_file_not_exist_returns_utf8(self, tmp_path):
        enc = DataLoader._detect_encoding(tmp_path / "nonexistent.csv")
        assert enc == "utf-8"


class TestLoadFolder:
    """load_folder() 整合測試"""

    def test_load_folder_not_exist_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DataLoader.load_folder(str(tmp_path / "nonexistent"))

    def test_load_folder_returns_dict(self, tmp_path):
        csv_content = make_twse_csv_content(200)
        (tmp_path / "2330.csv").write_text(csv_content, encoding="utf-8-sig")
        data = DataLoader.load_folder(str(tmp_path))
        assert isinstance(data, dict)

    def test_load_folder_correct_ticker(self, tmp_path):
        csv_content = make_twse_csv_content(200)
        (tmp_path / "2330.csv").write_text(csv_content, encoding="utf-8-sig")
        data = DataLoader.load_folder(str(tmp_path))
        assert "2330" in data

    def test_load_folder_skips_insufficient_rows(self, tmp_path):
        """資料列數 < 120 的股票應被略過"""
        csv_content = make_twse_csv_content(50)  # 只有 50 列
        (tmp_path / "9999.csv").write_text(csv_content, encoding="utf-8-sig")
        data = DataLoader.load_folder(str(tmp_path))
        assert "9999" not in data

    def test_load_folder_has_ohlcv_columns(self, tmp_path):
        csv_content = make_twse_csv_content(200)
        (tmp_path / "2330.csv").write_text(csv_content, encoding="utf-8-sig")
        data = DataLoader.load_folder(str(tmp_path))
        df   = data["2330"]
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in df.columns

    def test_load_folder_specific_tickers(self, tmp_path):
        """指定 tickers 只載入指定股票"""
        for ticker in ["2330", "0050", "2317"]:
            (tmp_path / f"{ticker}.csv").write_text(
                make_twse_csv_content(200), encoding="utf-8-sig"
            )
        data = DataLoader.load_folder(str(tmp_path), tickers=["2330", "0050"])
        assert "2330" in data
        assert "0050" in data
        assert "2317" not in data

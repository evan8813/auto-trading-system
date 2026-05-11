# TW-stock
+ 提供完整台股上市股逾10年資料，及爬蟲程式。不支援上櫃股。
+ 2025年持續更新

+ 贊助我 (另供2支程式: 回測與get_ids)
https://www.buymeacoffee.com/k66inthesky/e/216935

+ 教學影片: https://www.youtube.com/watch?v=p3U6V91wWlg

+ Log:
  - v1.0 台股上市股 

---

# 自動化交易系統 — 操作說明

## 目錄結構

```
TW-stock/
├── taiex.csv                          # 加權指數歷史資料
├── get_taiex.py                       # 下載加權指數資料的腳本
├── trade_analysis.ipynb               # 回測結果分析 Notebook
├── stocks_full/                       # 個股歷史 CSV 資料夾
└── logic/auto_trading/
    ├── config.py                      # ★ 所有策略參數（唯一設定檔）
    ├── main.py                        # 主程式入口
    ├── experiment.py                  # 測試用執行檔（改這裡做實驗）
    └── output/
        ├── trade_log.csv              # 交易紀錄
        ├── equity_curve.png           # 資產曲線圖
        └── experiment/                # experiment.py 的輸出
```

---

## 日常執行流程

### 1. 執行主回測（正式）

```bash
cd logic/auto_trading
python main.py C:/Users/USER/Desktop/jupyter/TW-stock/stocks_full
```

結果輸出到 `logic/auto_trading/output/`。

### 2. 執行測試（改參數實驗）

1. 打開 `logic/auto_trading/experiment.py`
2. 修改 `TradingConfig(...)` 裡的參數
3. 執行：

```bash
cd logic/auto_trading
python experiment.py C:/Users/USER/Desktop/jupyter/TW-stock/stocks_full
```

結果輸出到 `logic/auto_trading/output/experiment/`，不會蓋掉正式結果。

### 3. 指定股票回測

```bash
python main.py C:/path/to/stocks_full 2330 2317 2454
```

### 4. 合成資料快速測試（不需要 CSV）

```bash
python main.py
```

---

## 修改策略參數

所有參數集中在 `logic/auto_trading/config.py`，**只要改這一個檔案**，不需要動其他地方。

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `initial_equity` | 180,000 | 初始資金（元） |
| `risk_pct` | 1% | 每筆最大風險比例 |
| `max_positions` | 5 | 最大同時持倉數 |
| `max_trade_cost` | 36,000 | 單筆買入上限（元） |
| `atr_multiplier` | 3.0 | ATR 停損倍數 |
| `breakout_window` | 20 | 突破 N 日高低進場 |
| `taiex_csv_path` | 自動偵測 | 加權指數 CSV 路徑 |

> 臨時測試不想改 config.py → 改 `experiment.py` 即可

---

## 更新加權指數資料（taiex.csv）

```bash
cd C:/Users/USER/Desktop/jupyter/TW-stock
python get_taiex.py 2010-01-01 2023-12-31
```

---

## 分析回測結果

打開 Jupyter，執行 `trade_analysis.ipynb`，可以看到：
- 多空方向各自的勝率、PF
- 出場原因（phase1_stop / phase2_trail）比例
- 持有天數分佈
- 年度損益
- 大盤環境（多頭 / 空頭）vs 損益
- 贏家 vs 輸家持有天數比較
- 最大虧損前 20 筆
- 長期持有卻虧損的交易

### 個股交易分析（Section 10）

針對特定幾筆交易，繪製進出場圖表：

1. 執行 **Cell 1（cell-setup）** 載入 `trade_log.csv`
2. 執行 **Cell 21（plot_trade 函式）** 定義繪圖工具
3. 修改 **Cell 22** 裡的清單，填入要查看的交易：

```python
trades_to_analyze = [
    ('1905', '2021-04-22'),   # (股票代碼, 進場日期)
    ('2330', '2023-06-01'),
]
```

4. 執行 Cell 22，每筆交易各出一張圖

每張圖顯示：進出場前後 30 個交易日的價格走勢、進出場標記、成交量，以及損益／出場原因。

> 不想自己填 → 告訴 Claude「我要看哪幾筆」，它會直接更新 Cell 22

---

## 執行單元測試

```bash
cd logic/auto_trading
pytest tests/ -v
```

> PR 前必須全部通過（目前 253 個測試）

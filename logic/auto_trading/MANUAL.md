# Auto Trading 系統使用手冊

> 台灣股市自動交易策略系統 — 回測 & 實盤操作指南

---

## 目錄

1. [系統架構概觀](#1-系統架構概觀)
2. [環境安裝](#2-環境安裝)
3. [執行流程步驟](#3-執行流程步驟)
4. [單元測試執行](#4-單元測試執行)
5. [參數設定說明](#5-參數設定說明)
6. [資料格式規範](#6-資料格式規範)
7. [除權息事件設定](#7-除權息事件設定)
8. [實盤交易（Shioaji）](#8-實盤交易shioaji)
9. [輸出結果說明](#9-輸出結果說明)
10. [常見問題](#10-常見問題)

---

## 1. 系統架構概觀

```
auto_trading/
├── config.py           # 步驟1：策略參數設定
├── data_loader.py      # 步驟2：讀入 OHLCV 資料
├── indicators.py       # 步驟3：計算技術指標
├── universe_filter.py  # 步驟4：每日股票篩選
├── signal_generator.py # 步驟5：進出場訊號判斷
├── risk_manager.py     # 步驟6：部位大小與成本計算
├── backtester.py       # 步驟7：回測執行引擎
├── corporate_action.py # （選用）除權息事件管理
├── reporter.py         # 步驟8：輸出報表與圖表
├── live_trader.py      # （選用）實盤下單（需 Shioaji）
├── main.py             # 主程式入口
└── tests/              # 單元測試（見第4節）
```

### 資料流動方向

```
CSV 資料 → DataLoader → Indicators → UniverseFilter
                                          ↓
                                   SignalGenerator
                                          ↓
                                    RiskManager
                                          ↓
                                     Backtester
                                          ↓
                                      Reporter → equity_curve.png
                                                 trades_log.csv
```

---

## 2. 環境安裝

### 必要套件

```bash
pip install pandas numpy matplotlib
```

### 選用套件（實盤交易）

```bash
pip install shioaji
```

### 執行測試所需

```bash
pip install pytest
```

---

## 3. 執行流程步驟

### 步驟 1：設定策略參數（config.py）

開啟 `config.py`，依需求調整 `TradingConfig`：

```python
from config import TradingConfig

cfg = TradingConfig(
    initial_equity  = 1_000_000,   # 初始資金（元）
    risk_pct        = 0.002,       # 每筆最大損失 0.2%
    breakout_window = 50,          # 突破 N 日高低點
    ma_fast         = 50,          # 快線均線週期
    ma_slow         = 100,         # 慢線均線週期
    atr_period      = 14,          # ATR 週期
    atr_multiplier  = 3.0,         # ATR 停損倍數
    min_avg_amount  = 5_000_000,   # 最低日均成交金額
    max_positions   = 10,          # 最大持倉數
    backtest_start  = "2015-01-01",
    backtest_end    = "2023-12-29",
)
```

> **重要**：`ma_fast` 必須小於 `ma_slow`，否則趨勢判斷邏輯永遠不會觸發。

---

### 步驟 2：準備資料（data_loader.py）

#### 方式 A：使用真實 CSV（TWSE 格式）

CSV 欄位順序（中文表頭）：
```
日期,成交股數,成交金額,開盤價,最高價,最低價,收盤價,漲跌價差,成交筆數
```

資料夾結構：
```
data/
├── 2330.csv         # 台積電原始股價
├── 2330_adj.csv     # 台積電還原股價（優先使用）
├── 2317.csv
└── ...
```

載入範例：
```python
from data_loader import DataLoader

# 載入所有股票
data = DataLoader.load_folder("data/", adjusted=True)

# 只載入特定股票
data = DataLoader.load_folder("data/", tickers=["2330", "2317", "0050"])
```

#### 方式 B：使用合成測試資料

```python
data = DataLoader.generate_synthetic(
    tickers=["2330", "0050", "2317"],
    start="2015-01-01",
    end="2023-12-31",
    seed=42,       # 固定 seed 確保可重現
)
```

> 合成資料僅用於**驗證策略流程**，不代表真實市場行為。

---

### 步驟 3：指標計算（indicators.py）

指標計算由 `Backtester` 自動呼叫，**無需手動操作**。
如需獨立計算（例如除錯），可：

```python
from indicators import Indicators
from config import TradingConfig

cfg = TradingConfig()
df_with_indicators = Indicators.add_all(df, cfg)

# 查看附加的欄位：
# ATR, MA_Fast, MA_Slow, High_N, Low_N, High_52W, Low_52W, Avg_Amount_20
print(df_with_indicators.tail())
```

---

### 步驟 4：執行回測（backtester.py）

```python
from backtester import Backtester
from config import TradingConfig

cfg    = TradingConfig(backtest_start="2015-01-01", backtest_end="2023-12-31")
bt     = Backtester(cfg)
result = bt.run(data)

# result 包含：
# result["equity_curve"]  → pd.DataFrame（index=date, col=equity）
# result["trades"]        → pd.DataFrame（每筆平倉紀錄）
# result["metrics"]       → dict（績效指標）
```

---

### 步驟 5：查看結果（reporter.py）

```python
from reporter import Reporter

# 印出績效摘要
Reporter.print_metrics(result["metrics"])

# 儲存交易紀錄
Reporter.save_trade_log(result["trades"], "output/trades_log.csv")

# 繪製權益曲線
Reporter.plot_equity_curve(result["equity_curve"], save_path="output/equity_curve.png")
```

---

### 步驟 6：一鍵執行（main.py）

```bash
# 使用合成資料（無需 CSV）
python main.py

# 指定 CSV 資料夾（載入所有股票）
python main.py C:/data/twse_csv

# 指定資料夾 + 特定股票
python main.py C:/data/twse_csv 2330 2317 0050
```

輸出結果會存放在 `output/` 資料夾（自動建立）。

---

## 4. 單元測試執行

### 安裝 pytest

```bash
pip install pytest
```

### 執行所有測試

```bash
# 在 auto_trading 目錄下執行
cd logic/auto_trading
pytest tests/ -v
```

### 執行單一模組測試

```bash
# 只測試某個步驟
pytest tests/test_config.py -v           # 步驟1：參數設定
pytest tests/test_models.py -v           # 步驟2：資料結構
pytest tests/test_indicators.py -v      # 步驟3：指標計算
pytest tests/test_signal_generator.py -v # 步驟4：訊號邏輯
pytest tests/test_risk_manager.py -v    # 步驟5：風控計算
pytest tests/test_universe_filter.py -v # 步驟6：股票篩選
pytest tests/test_corporate_action.py -v # 步驟7：除權息
pytest tests/test_data_loader.py -v     # 步驟8：資料載入
pytest tests/test_backtester.py -v      # 步驟9：回測引擎
```

### 各測試檔案對應的驗證重點

| 測試檔案 | 驗證模組 | 驗證重點 |
|---------|---------|---------|
| `test_config.py` | `config.py` | 預設值、自訂參數、型別 |
| `test_models.py` | `models.py` | Position/CorporateEvent 欄位、update_trail 邏輯 |
| `test_indicators.py` | `indicators.py` | ATR/SMA/rolling 數值正確性、NaN 邊界 |
| `test_signal_generator.py` | `signal_generator.py` | 進出場條件 True/False、NaN 保護 |
| `test_risk_manager.py` | `risk_manager.py` | 風險金額、張數計算、交易成本含稅 |
| `test_universe_filter.py` | `universe_filter.py` | 篩選條件通過/不通過、邊界值 |
| `test_corporate_action.py` | `corporate_action.py` | 股息累加、分割更新、CSV 載入 |
| `test_data_loader.py` | `data_loader.py` | CSV 讀取、欄位對應、無效資料過濾 |
| `test_backtester.py` | `backtester.py` | 回測流程、績效計算、損益驗算 |

### 執行輸出範例

```
tests/test_config.py::TestTradingConfigDefaults::test_initial_equity_default PASSED
tests/test_config.py::TestTradingConfigDefaults::test_risk_pct_default PASSED
...
========================= 87 passed in 4.21s =========================
```

---

## 5. 參數設定說明

| 參數 | 預設值 | 說明 | 調整建議 |
|-----|-------|------|---------|
| `initial_equity` | 1,000,000 | 初始資金（元） | 依實際資金設定 |
| `risk_pct` | 0.002 | 每筆風險 0.2% | 保守：0.001；積極：0.005 |
| `commission_rate` | 0.001425 | 手續費 0.1425% | 依券商折扣調整 |
| `transaction_tax` | 0.003 | 證交稅 0.3%（賣出） | 固定，勿修改 |
| `slippage` | 0.001 | 滑價 0.1% | 流動性好可調低至 0.0005 |
| `breakout_window` | 50 | 突破 N 日高低 | 20（短線）~ 100（長線） |
| `ma_fast` | 50 | 快線 | 需 < ma_slow |
| `ma_slow` | 100 | 慢線 | 需 > ma_fast |
| `atr_period` | 14 | ATR 週期 | 通常 10~20 |
| `atr_multiplier` | 3.0 | ATR 停損倍數 | 寬鬆：4.0；緊縮：2.0 |
| `min_avg_amount` | 5,000,000 | 最低日均成交額（元） | 小型股可降至 2,000,000 |
| `max_positions` | 10 | 最大持倉數 | 依資金分散程度調整 |

---

## 6. 資料格式規範

### TWSE CSV 格式（標準）

```csv
日期,成交股數,成交金額,開盤價,最高價,最低價,收盤價,漲跌價差,成交筆數
2023-01-02,5000000,500000000,500.0,510.0,495.0,505.0,5.0,12345
2023-01-03,4800000,490000000,505.0,515.0,500.0,512.0,7.0,11200
```

### 停牌日處理

停牌或除權息日，欄位值可為 `--`、`除息`、`除權`、`除權息` 或空白，`DataLoader` 會自動過濾。

### 還原股價（推薦）

- 有 `{ticker}_adj.csv` → 優先使用（較準確）
- 無 → fallback 到 `{ticker}.csv`

---

## 7. 除權息事件設定

### CSV 格式

```csv
ticker,event_date,event_type,cash_dividend,stock_ratio,split_ratio,note
2330,2023-08-28,dividend,3.5,0.0,1.0,2H23除息
2454,2023-09-01,dividend,1.0,0.05,1.0,股息+股利
2330,2021-07-01,split,0.0,0.0,2.0,股票分割1轉2
```

### 使用方式（實盤）

```python
from corporate_action import CorporateActionLog

ca_log = CorporateActionLog()
ca_log.load_csv("corporate_actions.csv")

# 每日盤前執行
for pos in positions:
    ca_log.apply_to_position(pos, today)
```

> 回測模式**不使用**除權息調整（已透過還原股價處理）。

---

## 8. 實盤交易（Shioaji）

> **注意**：需先向永豐金證券申請 API 金鑰與 CA 憑證。

```python
from live_trader import LiveTrader
from config import TradingConfig
from corporate_action import CorporateActionLog

cfg    = TradingConfig()
ca_log = CorporateActionLog()
ca_log.load_csv("corporate_actions.csv")

# 初始化（模擬模式 sim=True 不實際下單）
lt = LiveTrader(
    cfg        = cfg,
    api_key    = "YOUR_API_KEY",
    secret_key = "YOUR_SECRET_KEY",
    ca_path    = "C:/path/to/Sinopac.pfx",
    ca_passwd  = "YOUR_CA_PASSWORD",
    sim        = True,  # ← 測試時保持 True
    ca_log     = ca_log,
)

# 下單
trade = lt.place_order("2330", direction="long", lots=1, price=580.0)

# 檢查出場訊號
remaining_positions = lt.monitor_and_exit(positions, latest_data, today)

# 登出
lt.logout()
```

---

## 9. 輸出結果說明

### metrics 字典

```python
{
    "total_return_pct": 42.5,    # 回測期間總報酬率（%）
    "cagr_pct":          5.8,    # 年化複合成長率（%）
    "max_drawdown_pct": -18.3,   # 最大回撤（%，負值）
    "win_rate_pct":     52.1,    # 勝率（%）
    "total_trades":      312,    # 總交易筆數
    "avg_win":          8500.0,  # 平均獲利（元）
    "avg_loss":        -4200.0,  # 平均虧損（元）
    "profit_factor":     2.02,   # 獲利因子（>1 為正期望值）
}
```

### trades_log.csv 欄位說明

| 欄位 | 說明 |
|-----|------|
| `ticker` | 股票代號 |
| `direction` | long / short |
| `entry_date` | 進場日 |
| `exit_date` | 出場日 |
| `hold_days` | 持倉天數 |
| `adj_entry_price` | 調整後進場價（回測用） |
| `adj_exit_price` | 調整後出場價（回測用） |
| `raw_entry_price` | 原始進場價（對帳用） |
| `raw_exit_price` | 原始出場價（對帳用） |
| `gross_pnl` | 毛利（元） |
| `total_cost` | 手續費+稅+滑價（元） |
| `pnl_net` | 淨損益（元） |
| `exit_reason` | 出場原因（signal） |

---

## 10. 常見問題

### Q1：執行後沒有交易產生

**可能原因：**
- `backtest_start` / `backtest_end` 超出資料範圍
- 資料列數不足（< 120 列），`ma_slow=100` 需要至少 100 筆資料才有有效值
- `min_avg_amount` 設定過高，所有股票都被篩除

**排查方式：**
```python
# 確認資料長度
for ticker, df in data.items():
    print(ticker, len(df), df.index[0], df.index[-1])

# 測試篩選結果
from universe_filter import UniverseFilter
uf = UniverseFilter(cfg)
candidates = uf.filter(data_with_indicators, some_date)
print("候選股票數：", len(candidates))
```

### Q2：ATR 全部為 NaN

**原因：** 資料前幾列的 `High`/`Low`/`Close` 欄位有 NaN（停牌日未過濾）。  
**解法：** 確認 `DataLoader._filter_invalid_rows()` 已執行（`load_folder` 自動呼叫）。

### Q3：import 錯誤（ModuleNotFoundError）

**原因：** Python 未在 `auto_trading/` 目錄執行。  
**解法：**
```bash
cd logic/auto_trading
python main.py
# 或
python -m pytest tests/ -v
```

### Q4：CSV 讀取出現亂碼

**原因：** CSV 編碼非 UTF-8-SIG（台灣常見 Big5）。  
`DataLoader` 會自動偵測，若仍失敗可手動指定：

```python
# 手動指定 Big5 編碼後另存為 UTF-8-SIG
import pandas as pd
df = pd.read_csv("2330.csv", encoding="cp950")
df.to_csv("2330_utf8.csv", encoding="utf-8-sig", index=False)
```

### Q5：如何在不同時間段比較策略

```python
results = {}
for period in [("2015-01-01","2019-12-31"), ("2020-01-01","2023-12-31")]:
    cfg = TradingConfig(backtest_start=period[0], backtest_end=period[1])
    bt  = Backtester(cfg)
    r   = bt.run(data)
    results[period] = r["metrics"]

for period, m in results.items():
    print(f"{period}: CAGR={m['cagr_pct']}%  MDD={m['max_drawdown_pct']}%")
```

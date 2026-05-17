# FinLab 使用手冊

> 適用版本：finlab >= 2.0.0 | Python >= 3.10

---

## 目錄

1. [安裝與登入](#1-安裝與登入)
2. [取得資料](#2-取得資料)
3. [建立選股條件](#3-建立選股條件)
4. [建構持倉](#4-建構持倉)
5. [回測](#5-回測)
6. [解讀回測結果](#6-解讀回測結果)
7. [重要注意事項](#7-重要注意事項)
8. [常見錯誤](#8-常見錯誤)

---

## 1. 安裝與登入

### 安裝

```bash
pip install "finlab>=2.0.0"
```

### 登入（第一次使用）

```python
import finlab
finlab.login()  # 開啟瀏覽器，用 Google 帳號登入，自動儲存 token
```

登入後 token 會自動儲存，之後不需要再重複登入。

### 確認用量

```python
import finlab
print(finlab.data.get_role())   # 查看帳號等級
print(finlab.data.is_vip())     # True = VIP，False = 免費
```

| 方案 | 每日限額 | 重置時間 |
|------|----------|----------|
| 免費 | 500 MB | 每天 08:00（台灣時間）|
| VIP | 5,000 MB | 同上 |

---

## 2. 取得資料

### 基本語法

```python
from finlab import data

df = data.get("表格名稱:欄位名稱")
```

### 常用資料表

```python
from finlab import data

# 價格類
close  = data.get("price:收盤價")
open_  = data.get("price:開盤價")
high   = data.get("price:最高價")
low    = data.get("price:最低價")
volume = data.get("price:成交股數")

# 估值類
pe  = data.get("price_earning_ratio:本益比")
pb  = data.get("price_earning_ratio:股價淨值比")
div = data.get("price_earning_ratio:殖利率(%)")

# 基本面
roe = data.get("fundamental_features:ROE稅後")
eps = data.get("fundamental_features:EPS")
gpm = data.get("fundamental_features:營業毛利率")

# 月營收
rev        = data.get("monthly_revenue:當月營收")
rev_growth = data.get("monthly_revenue:去年同月增減(%)")

# 技術指標（不需要傳入資料，自動使用收盤價）
rsi  = data.indicator("RSI", timeperiod=14)
macd, macd_signal, macd_hist = data.indicator(
    "MACD", fastperiod=12, slowperiod=26, signalperiod=9
)
```

### 搜尋可用資料

```python
# 用關鍵字搜尋（台股用中文）
data.search("營收", market="tw")
data.search("ROE", market="tw")
```

### 資料格式說明

所有 `data.get()` 回傳的都是 **FinLabDataFrame**：
- **列（index）**：日期
- **欄（columns）**：股票代號（如 `'2330'`, `'0050'`）
- **值**：對應數值

```python
close.head()
#            2330   2317   2454  ...
# 2020-01-02  330.0  81.7  410.0
# 2020-01-03  331.5  82.0  412.5
```

---

## 3. 建立選股條件

### 比較運算 → 產生布林 DataFrame

```python
close = data.get("price:收盤價")

# 股價 > 60 日均線
above_ma60 = close > close.average(60)

# 股價創 52 週新高
new_high = close >= close.rolling(252, min_periods=126).max()

# 成交量放大（今日 > 20日均量 × 1.5）
volume  = data.get("price:成交股數")
vol_avg = volume.average(20)
vol_exp = volume > vol_avg * 1.5
```

### 常用 FinLabDataFrame 方法

| 方法 | 說明 | 範例 |
|------|------|------|
| `.average(n)` | n 日移動平均 | `close.average(60)` |
| `.rise(n)` | 比 n 天前上漲 → 布林 | `close.rise(10)` |
| `.fall(n)` | 比 n 天前下跌 → 布林 | `close.fall(10)` |
| `.sustain(n)` | 連續 n 天為 True | `cond.sustain(3)` |
| `.is_largest(n)` | 每列最大的 n 個 → 布林 | `momentum.is_largest(15)` |
| `.is_smallest(n)` | 每列最小的 n 個 → 布林 | `pe.is_smallest(10)` |
| `.rank(axis=1, pct=True)` | 橫向百分位排名 | `pb.rank(axis=1, pct=True)` |
| `.shift(n)` | 往前移 n 期（取前值）| `close.shift(1)` |
| `.industry_rank()` | 同產業內排名（百分位）| `roe.industry_rank()` |

### 組合條件

```python
# AND：兩個條件都要符合
position = cond1 & cond2

# OR：任一條件符合
position = cond1 | cond2

# NOT：反轉條件
position = ~cond1
```

---

## 4. 建構持倉

### 方法一：直接用布林條件

```python
# 所有符合條件的股票都持有（等權重）
position = cond1 & cond2 & cond3
```

### 方法二：用因子篩選前 N 名

```python
# 先用條件篩選，再從中選動能最強的 15 檔
momentum_3m = close / close.shift(63) - 1
position = momentum_3m[condition].is_largest(15)

# 選 P/B 最低的 10 檔
pb = data.get("price_earning_ratio:股價淨值比")
position = pb[condition].is_smallest(10)
```

### 方法三：進出場訊號（hold_until）

```python
# 進場：站上 20 日均線
entries = close > close.average(20)

# 出場：跌破 60 日均線
exits = close < close.average(60)

# 持有直到出場，最多同時持有 10 檔，優先持有低 P/B
position = entries.hold_until(exits, nstocks_limit=10, rank=-pb)
```

---

## 5. 回測

### 基本語法

```python
from finlab.backtest import sim

report = sim(position, resample="M", upload=False)
```

### 常用參數

| 參數 | 說明 | 建議值 |
|------|------|--------|
| `resample` | 換倉頻率 | `"M"`月 / `"W"`週 / `"Q"`季 |
| `stop_loss` | 停損（跌多少出場）| `0.08`（8%）|
| `take_profit` | 停利（漲多少出場）| `0.25`（25%）|
| `position_limit` | 單檔最大持倉比例 | `1/15`（等權重）|
| `fee_ratio` | 手續費 | `1.425/1000/3`（台股折扣後）|
| `tax_ratio` | 交易稅 | `3/1000`（台股）|
| `trade_at_price` | 成交價格 | `'open'`開盤 / `'close'`收盤 |
| `upload` | 上傳到 FinLab 雲端 | 開發期間用 `False` |

### 完整範例

```python
from finlab.backtest import sim

report = sim(
    position,
    resample="M",
    stop_loss=0.08,
    take_profit=0.25,
    position_limit=1/15,
    fee_ratio=1.425/1000/3,
    tax_ratio=3/1000,
    trade_at_price="open",
    upload=False,
)
```

---

## 6. 解讀回測結果

### 取得績效數字

```python
# 方法 A：metrics 物件
print(f"年化報酬: {report.metrics.annual_return():.2%}")
print(f"夏普比率: {report.metrics.sharpe_ratio():.2f}")
print(f"最大回撤: {report.metrics.max_drawdown():.2%}")

# 方法 B：get_stats() 字典
stats = report.get_stats()
print(f"CAGR:    {stats['cagr']:.2%}")
print(f"Sharpe:  {stats['monthly_sharpe']:.2f}")
print(f"MDD:     {stats['max_drawdown']:.2%}")
```

### 指標對照表

| metrics 方法 | get_stats() key | 說明 |
|-------------|-----------------|------|
| `annual_return()` | `cagr` | 年化報酬率 |
| `sharpe_ratio()` | `monthly_sharpe` | 夏普比率 |
| `max_drawdown()` | `max_drawdown` | 最大回撤 |

### 在 Jupyter 顯示完整報告

```python
report  # 最後一行直接輸出，顯示互動圖表
```

### 在終端機顯示

```python
report.to_terminal()  # ASCII 格式，不需要 Jupyter
```

---

## 7. 重要注意事項

### ❌ Lookahead Bias（未來資料污染）

最常見的致命錯誤，讓回測虛高。

```python
# ❌ 錯誤：iloc[-2] 可能造成 lookahead
prev = close.iloc[-2]

# ✅ 正確：用 shift(1) 取前一期
prev = close.shift(1)

# ❌ 錯誤：用 bfill 往回填值（看到未來）
df.reindex(target_index, method='bfill')

# ✅ 正確：只能用 ffill（往前填）
df.reindex(target_index, method='ffill')

# ❌ 錯誤：手動改 index（會破壞自動對齊）
df.index = new_index

# ✅ 正確：保持 index 不動
```

### ❌ 不要用 for loop

```python
# ❌ 慢，且容易出錯
for date in close.index:
    for stock in close.columns:
        if close.loc[date, stock] > ...:
            ...

# ✅ 向量化，快且正確
position = close > close.average(60)
```

### ❌ 不要用 == 比較浮點數

```python
# ❌ 浮點誤差
condition = close == 100.0

# ✅ 用範圍
condition = (close > 99.9) & (close < 100.1)
```

### ✅ 開發期間一律用 upload=False

```python
# 開發 / 測試
report = sim(position, resample="M", upload=False)

# 確定上線才改 True
report = sim(position, resample="M", upload=True)
```

---

## 8. 常見錯誤

### 用量超限

```
quota exceeded / daily limit reached
```

**解法：**
- 等到台灣時間早上 8 點自動重置
- 把常用資料存成變數，不要重複 `data.get()`

```python
# ❌ 重複取同一份資料，浪費用量
if close.average(20) > ...:
    x = close.average(20)  # 又取一次

# ✅ 存成變數
ma20 = close.average(20)
cond = close > ma20
```

### 記憶體錯誤（`_ArrayMemoryError`）

```python
# 重新啟動 Python kernel 再試
```

### 技術指標語法錯誤

```python
# ❌ 不要傳入收盤價
rsi = data.indicator("RSI", close, timeperiod=14)

# ✅ 直接呼叫，自動使用收盤價
rsi = data.indicator("RSI", timeperiod=14)
```

---

## 快速參考

```python
from finlab import data
from finlab.backtest import sim

# 1. 取資料
close  = data.get("price:收盤價")
volume = data.get("price:成交股數")
eps    = data.get("fundamental_features:EPS")

# 2. 建條件
cond1 = close >= close.rolling(252, min_periods=126).max()  # 新高
cond2 = close > close.average(20)                           # 站上均線
cond3 = eps > 0                                              # 獲利

# 3. 選股
momentum = close / close.shift(63) - 1
position = momentum[cond1 & cond2 & cond3].is_largest(15)

# 4. 回測
report = sim(position, resample="M", stop_loss=0.08, upload=False)

# 5. 看結果
stats = report.get_stats()
print(f"CAGR: {stats['cagr']:.2%}")
print(f"MDD:  {stats['max_drawdown']:.2%}")
```

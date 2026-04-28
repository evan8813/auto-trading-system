# 驗證指南

## 指令速查

| 目的 | 指令 |
|---|---|
| 跑全部單元測試 | `pytest tests/ -v` |
| 跑回測 + checkpoint | `python main.py C:\...\stocks` |
| 一鍵驗證（抽樣） | `python verify.py` |

---

## 驗證工具說明

### 1. 單元測試 `tests/`

```bash
pytest tests/ -v
```

| 檔案 | 測試內容 |
|---|---|
| `test_config.py` | 參數預設值、型別、邏輯關係 |
| `test_data_loader.py` | CSV 載入、欄位格式、合成資料 |
| `test_indicators.py` | ATR、MA、High_N 等指標計算 |
| `test_universe_filter.py` | 選股篩選條件（流動性、52週高點） |
| `test_signal_generator.py` | 進出場訊號邏輯 |
| `test_risk_manager.py` | 張數計算、交易成本 |
| `test_models.py` | Position 資料結構 |
| `test_corporate_action.py` | 除權息事件套用 |
| `test_backtester.py` | 回測引擎整合、損益計算 |
| `test_scenarios.py` | 六個邊界情境（見下方） |

**六個邊界情境**

| 情境 | 驗證內容 | 預期結果 |
|---|---|---|
| 1. 選股池為空 | min_avg_amount 極高 | 0 筆交易 |
| 2. 持倉上限封頂 | 20支股票、max=3 | 同時持倉 ≤ 3 |
| 3. 出場後補進 | max=1，多支股票 | 出場後有新股補入 |
| 4. 不重複持倉 | 同股票前後持倉 | 無重疊期間 |
| 5. 張數為 0 | 資金極小 | 0 筆交易 |
| 6. 停牌留倉 | 移除持倉期間資料 | 不提前出場 |

---

### 2. Checkpoint `checkpoint.py`

回測跑完後自動執行，驗證兩個節點：

**Checkpoint 1 — 選股池驗證**
每季抽一個日期，確認選股池內每支股票都符合：
- 流動性：Avg_Amount_20 >= min_avg_amount
- 近 52 週新高：Close >= High_52W × 90%
- 買得起：Close × 1000 <= initial_equity

**Checkpoint 2 — 進出場驗證**
每筆交易逐一確認：
- A：進場當天該股票在選股池內
- B：進場訊號確實觸發（突破 + MA 條件）

---

### 3. 抽樣診斷 `sample_period()`

```python
sample_period("2013-01-14", results, data, cfg)
```

**注意：填的是信號日（進場日的前一個交易日）**

輸出三段：

**① 選股池（依 ROC 排名）**
```
rank  ticker  close  close/52w_pct  avg_amount_20  roc_avg  signal
1     9939    72.8   100.0          203.61         16.3     none
5     2614    4.3    90.5           10.34           9.2     LONG
```

**② 策略結果**
- `signal = LONG/SHORT`：進場訊號有觸發
- `signal = none`：條件不成立，直接排除
- `entered = True`：確實進場，附進出場日與損益
- `entered = False`：訊號觸發但沒進場 → 看診斷區

**③ 診斷（訊號觸發但未進場）**
```
當時已有持倉：8 / 10
[3] 5871  ->  張數 = 0（風險金額 360 / ATR 1.53 / 1000 < 1）
[7] 2330  ->  持倉已滿（10/10）
```

常見未進場原因：
| 原因 | 說明 |
|---|---|
| 持倉已滿 | 當時持倉數 = max_positions |
| 張數 = 0 | initial_equity × risk_pct ÷ ATR ÷ 1000 < 1 |
| 單筆超限 | 股價 × 1000張 > max_trade_cost |

---

### 4. ROC 計算方式

```
ROC_avg = (ROC_10 + ROC_25 + ROC_35) / 3
```
- ROC_10 = 2 週動能（10 個交易日）
- ROC_25 = 5 週動能（25 個交易日）
- ROC_35 = 7 週動能（35 個交易日）

三週期平均，避免單一週期的偶發偏差。

---

## 常見問題

**Q：為什麼 sample_period 要填前一天的日期？**
回測的時序是 T日收盤判斷訊號 → T+1日開盤執行。
trade log 裡的 entry_date 是 T+1，所以要查 T 的選股池。

**Q：signal=LONG 但 entered=False 是程式 bug 嗎？**
不是。最常見原因是張數=0（資金太小）或持倉已滿，診斷區會告訴你確切原因。

**Q：同一天 exit_date = next entry_date 是重疊持倉嗎？**
不是。回測執行順序是先出場再進場，同天換股是合法的。

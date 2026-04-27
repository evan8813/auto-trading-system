# 開發規範

## 1. PR 規範

**所有程式碼變更都必須透過 Pull Request，禁止直接 merge 到 main。**

- 每次變更完成後，push 到 feature branch 並開 PR
- 等待確認後才可 merge
- PR 說明需包含：變更內容、原因、測試結果

---

## 2. 單元測試規範

**每次修改程式邏輯，測試檔必須同步更新。**

- 測試檔：`test_auto_trading_system.py`
- 執行方式：`pytest test_auto_trading_system.py -v`
- PR 前必須確認全部測試通過

### 測試設計原則

- 每個測試使用**自製假資料**，讓預期答案完全可控
- 每個測試只驗證**一件事**
- 測試名稱需清楚說明「在什麼情況下，預期什麼結果」

### 必須涵蓋的情境

| 情境 | 說明 |
|------|------|
| 正常入選 | 所有條件都符合時，確認有入選 |
| 正常排除 | 任一條件不符時，確認沒有入選 |
| 邊界值 | 剛好等於門檻時的行為（`>` vs `>=`）|
| NaN 保護 | 資料缺失或不足時，程式不誤動作 |

---

## 3. 修改其他模組前需確認

**單元測試過程中，若需修改非測試檔的程式，必須先詢問確認再動手。**

---

## 4. 目前測試涵蓋範圍

| 元件 | 測試數 | 說明 |
|------|--------|------|
| `Indicators` | 5 | ATR、MA、High_N、High_52W、Low_52W |
| `UniverseFilter` | 5 | 52週突破新高/新低、成交金額、NaN 保護 |
| `SignalGenerator` | 8 | 做多/做空進出場、NaN 保護 |
| `RiskManager` | 6 | 風險金額、張數計算、交易成本 |
| `Position` | 4 | trail_high/trail_low 追蹤停損更新 |

---

## 5. 選股條件規格

| 條件 | 說明 |
|------|------|
| 52週突破 | 今日 `High > 昨日 High_52W` 或 今日 `Low < 昨日 Low_52W` |
| 成交金額 | 20日平均成交金額 >= 5,000,000 元 |
| High_52W | 取 `High`（最高價）欄位的252日 rolling max |
| Low_52W | 取 `Low`（最低價）欄位的252日 rolling min |

---

## 6. 進出場條件規格

| 條件 | 說明 |
|------|------|
| 做多進場 | 收盤 > 前日 High_N 且 MA_Fast > MA_Slow |
| 做多出場 | 收盤 < trail_high - atr_multiplier × ATR |
| 做空進場 | 收盤 < 前日 Low_N 且 MA_Fast < MA_Slow |
| 做空出場 | 收盤 > trail_low + atr_multiplier × ATR |

# 開發規範

## 1. PR 規範

**所有程式碼變更都必須透過 Pull Request，禁止直接 merge 到 main。**

- 每次變更完成後，push 到 feature branch 並開 PR
- 等待確認後才可 merge
- PR 說明需包含：變更內容、原因、測試結果

---

## 2. 單元測試規範

**每次修改程式邏輯，測試檔必須同步更新。**

- 測試資料夾：`logic/auto_trading/tests/`
- 執行方式：`pytest tests/ -v`
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

| 測試檔 | 測試數 | 說明 |
|--------|--------|------|
| `test_config.py` | 43 | 預設值、自訂值、型別、邏輯關係 |
| `test_indicators.py` | 22 | ATR、SMA、rolling max/min、add_all |
| `test_signal_generator.py` | 25 | 做多/做空進出場、NaN 保護 |
| `test_universe_filter.py` | 17 | 52週突破、成交金額、動態股價上限、NaN 保護 |
| `test_risk_manager.py` | 22 | 風險金額上限、ATR 張數計算、容忍區、交易成本 |
| `test_models.py` | 16 | Position 欄位、trail 追蹤停損、CorporateEvent |
| `test_corporate_action.py` | 15 | 除權息載入、查詢、套用到持倉 |
| `test_data_loader.py` | 27 | CSV 讀取、欄位對應、編碼偵測、資料清洗 |
| `test_backtester.py` | 30 | 整合流程、績效計算、損益、equity_at_entry |
| `test_scenarios.py` | 12 | 空池、持倉上限、補倉、不重複、零張數、停牌 |
| `test_scenarios_advanced.py` | 15 | ROC 排序、T+1 停牌補位、動態成本截斷、換倉、權益追蹤 |
| **合計** | **244** | |

---

## 5. 選股條件規格

| 條件 | 說明 |
|------|------|
| 52週突破 | 今日 `High > 昨日 High_52W`（多方候選）或 今日 `Low < 昨日 Low_52W`（空方候選） |
| 成交金額 | 20日平均成交金額 >= 5,000,000 元 |
| 股價上限 | 收盤 × 1,000 ≤ equity / max_positions（動態，隨資產成長） |
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

---

## 7. 風控規格

### 每筆風險金額

```
risk_amount = min(equity × risk_pct, max_risk_amount)
```

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `risk_pct` | 1% | 每筆最大虧損佔總淨值比例 |
| `max_risk_amount` | 2,500 元 | 單筆風險硬上限（equity > 250,000 後觸發） |

### 部位大小（張數）

```
主公式：lots = int(risk_amount / (atr_multiplier × ATR × 1,000))
```

- 主公式確保打到停損時實際虧損 ≈ risk_amount
- **容忍區**：主公式給 0，但 1 張風險 ≤ equity × 2% → 進 1 張
- equity 成長後主公式自然給出 ≥ 1 張，容忍區逐漸不觸發
- ATR ≤ 0 或 1 張風險超出容忍區 → 回傳 0（不進場）

### 動態單倉上限

```
dynamic_max_cost = equity / max_positions
```

- 單筆進場成本超過此上限時，張數向下截斷
- max_positions 預設 5，隨資產成長自動擴張可買股價範圍

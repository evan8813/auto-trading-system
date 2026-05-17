# Minervini-Style Momentum Breakout Strategy

Taiwan market momentum breakout strategy inspired by Mark Minervini's SEPA methodology.

## Entry Conditions

| Layer | Condition | Purpose |
|-------|-----------|---------|
| Breakout | Price ≥ 52-week high | Only buy strength |
| Trend | MA20 > MA60 > MA120 | Avoid false breakouts |
| Volume | Today > 1.5× 20-day avg | Confirm participation |
| Fundamental | EPS > 0 | Avoid loss-making stocks |
| Ranking | Top 15 by 3-month return | Concentrate in strongest |

## Risk Management

- Stop-loss: 8%
- Take-profit: 25%
- Position size: equal-weight (1/15 per stock)
- Rebalance: monthly

## How to Run

```bash
python minervini_strategy/strategy.py
```

Requires FinLab >= 2.0.0 and a valid API token (`finlab.login()`).

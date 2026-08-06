# Evaluation Report

Test games: 20

| Metric | LSTM | Logistic Regression baseline |
|---|---|---|
| Brier score (lower better) | 0.2177 | 0.1594 |
| Log-loss (lower better) | 0.6169 | 0.4743 |

## Accuracy by time remaining
| Time remaining | LSTM accuracy | Baseline accuracy | N |
|---|---|---|---|
| >36 min | 0.547 | 0.664 | 2357 |
| 12-36 min | 0.547 | 0.731 | 4735 |
| 3-12 min | 0.553 | 0.847 | 1698 |
| <3 min | 0.582 | 0.938 | 660 |

# Evaluation Report

Test games: 20

| Metric | LSTM | Logistic Regression baseline |
|---|---|---|
| Brier score (lower better) | 0.2476 | 0.2282 |
| Log-loss (lower better) | 0.6882 | 0.6401 |

## Accuracy by time remaining
| Time remaining | LSTM accuracy | Baseline accuracy | N |
|---|---|---|---|
| >36 min | 0.547 | 0.575 | 2357 |
| 12-36 min | 0.539 | 0.588 | 4735 |
| 3-12 min | 0.541 | 0.614 | 1698 |
| <3 min | 0.561 | 0.658 | 660 |

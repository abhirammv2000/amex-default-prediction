# Credit-risk evaluation report

```
========================================================
CREDIT-RISK EVALUATION (out-of-fold, 458,913 customers)
========================================================
Amex metric        : 0.79266
AUC / Gini         : 0.96151 / 0.92302
KS statistic       : 0.79523  (separation of goods vs bads)
Default rate       : 0.2589
Brier  (raw)       : 0.06786
Brier  (isotonic)  : 0.06787  (worse)

Score bands (decile 1 = highest predicted risk):
      customers  defaults  avg_pred  bad_rate  cum_defaults_%
band                                                         
1         45892     44214      96.0      96.3            37.2
2         45891     36623      80.6      79.8            68.0
3         45891     24020      52.8      52.3            88.2
4         45891     10231      21.0      22.3            96.9
5         45891      2653       5.2       5.8            99.1
6         45892       671       1.5       1.5            99.6
7         45891       249       0.6       0.5            99.9
8         45891        95       0.3       0.2            99.9
9         45891        43       0.2       0.1           100.0
10        45892        29       0.1       0.1           100.0

Top decile captures 37.2% of all defaults.

Saved figures: calibration.png, score_bands.png, approval_tradeoff.png
```

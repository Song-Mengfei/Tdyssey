# Fixed C base/delta development comparison

Both methods use the same training sampler with std floor 0.01. Differences are delta minus baseline.

| Shape | Base successes/10 | Delta successes/10 | Difference (pp) | Success win/tie/loss | Mean final coverage difference | Mean peak coverage difference |
|---|---:|---:|---:|---:|---:|---:|
| T | 6/10 | 5/10 | -10 | 0/9/1 | -0.0123956 | 0.00330306 |
| L | 0/10 | 0/10 | 0 | 0/10/0 | 0.022792 | -0.00817852 |

Coverage and IoU CSVs include method mean/median, paired difference mean/median and win/tie/loss at tolerance 1e-9.
Success step differences use only pairs where both succeed; negative means delta succeeded earlier. Other pairs remain censored and blank, never assigned 300.
There are five scene blocks per shape. Two policy seeds within a scene are correlated; these are not ten independent scenes.
This is a fixed development comparison, not an untouched test or original-sampler baseline. No training or new evaluation is performed by this export.

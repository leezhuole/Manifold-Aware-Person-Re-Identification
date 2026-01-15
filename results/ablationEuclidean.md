| Method | M+MS+CS -> C3 mAP | M+MS+CS -> C3 Rank-1 | M+CS+C3 -> MS mAP | M+CS+C3 -> MS Rank-1 | MS+CS+C3 -> M mAP | MS+CS+C3 -> M Rank-1 | Average mAP | Average Rank-1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline (w/o augmented images) | 31.5 | 30.2 | 19.8 | 42.3 | - | - | 25.6 | 36.2 |
| $\mathcal{L}_{ce}$ | 24.3 | 22.9 | 17.4 | 40.6 | 62.2 | 81.9 | 34.6 | 48.5 |
| $\mathcal{L}_{align}$ | 40.2 | 40.1 | 21.5 | 47.0 | 76.7 | 89.0 | 46.1 | 58.7 |
| $\mathcal{L}_{align}+\mathcal{L}_{uniform}$ | 40.7 | 41.1 | 21.3 | 45.5 | 77.0 | 89.3 | 46.3 | 58.6 |
| $\mathcal{L}_{align}+\mathcal{L}_{uniform}+\mathcal{L}_{domain}$ | 42.1 | 42.0 | 22.5 | 48.0 | 76.9 | 90.3 | 47.2 | 60.1 |

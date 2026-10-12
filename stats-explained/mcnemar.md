# McNemar's exact test

**Used for:** success rate, when the comparison is paired (both policy versions ran the same seeds, and the setup is deterministic).

## What it represents

Each seed is a small experiment run twice: once with the baseline policy, once with the candidate. There are four possible outcomes per seed:

| | Candidate succeeded | Candidate failed |
|---|---|---|
| **Baseline succeeded** | both succeed | **b**: flipped to failure |
| **Baseline failed** | **c**: flipped to success | both fail |

Seeds where both versions did the same thing say nothing about which version is better. All the evidence is in the seeds that **flipped**: `b` got worse, `c` got better.

McNemar's question is: *if the two versions were really equally good, a flipped seed would be as likely to flip one way as the other, like a fair coin. Is the split between `b` and `c` more lopsided than a fair coin would give?*

## How it is computed

Under "no real change", `b` follows a binomial distribution with `b + c` coin flips and probability 0.5. The exact two-sided p-value is the probability of a split at least as lopsided as the one observed.

**Worked example.** 100 seeds; 9 flipped to failure and 1 flipped to success.

```text
flips = b + c = 10
P(1 or fewer heads in 10 fair flips) = (1 + 10) / 1024 = 0.0107
two-sided p = 2 x 0.0107 = 0.0215
```

A split this lopsided would happen about 2% of the time by chance, so it is unlikely to be noise.

More splits, for a feel of the scale:

| Flipped to failure (b) | Flipped to success (c) | p-value |
|---|---|---|
| 3 | 3 | 1.00 |
| 4 | 1 | 0.375 |
| 5 | 0 | 0.0625 |
| 6 | 0 | 0.031 |
| 12 | 2 | 0.013 |
| 20 | 1 | 0.00002 |

## What polcheck reports alongside it

- **Estimate:** `(b − c) / n`, the net drop in success rate. With 9 and 1 out of 100 seeds, that is 0.08: eight points worse.
- **Confidence interval:** from a bootstrap over seeds. Seeds are resampled with replacement 10,000 times and the estimate is recomputed each time.
- **Method label in results:** `McNemar exact; paired bootstrap CI`.

## Things worth knowing

- **Why "exact".** The textbook McNemar test uses a chi-square approximation that breaks down when `b + c` is small, which is the normal case for a good policy (few seeds flip at all). The exact binomial version has no such problem.
- **You need at least 6 flips to reach p < 0.05.** Five flips all in one direction give p = 0.0625. So a regression that flips only a handful of seeds cannot be confirmed in one family, however many seeds agree. This is also why per-family tests are backed by an "all families" test.
- **No flips means no evidence.** If the two versions agree on every seed, p = 1 and the estimate is exactly 0: a clean `PASS`.
- **It depends on real pairing.** The test assumes seed 17 is the same scene for both versions. If the simulator is not deterministic, "the same seed" is not the same experiment, and polcheck falls back to the unpaired [Fisher test](fisher.md). Determinism is checked from repeated seeds inside each batch.
- **Why it beats the unpaired test.** Hard seeds tend to be hard for both versions. Pairing cancels that shared difficulty, so a small real drop stands out. With 96% vs 90% success on 50 seeds, an unpaired test sees two similar-looking rates; a paired test sees that every one of the three extra failures is a seed the baseline used to solve.

## In the code

`mcnemar_exact(b, c)` and `paired_success(...)` in `src/polcheck/stats/success.py`. Tested against `statsmodels` for nine `(b, c)` pairs and against the hand calculation above.

## Reference

McNemar, Q. (1947). "Note on the sampling error of the difference between correlated proportions or percentages." *Psychometrika* 12, 153–157. The exact (binomial) form is standard; see Agresti, *Categorical Data Analysis*, section on matched pairs.

# Newcombe's score interval

**Used for:** the confidence interval on the difference between two success rates, in unpaired comparisons. It accompanies the [Fisher test](fisher.md).

## What it represents

A p-value says whether a difference is real. It does not say how big the difference might be. The confidence interval does: it is the range of true differences the data is compatible with.

For a baseline of 48/50 (96%) and a candidate of 41/50 (82%), the observed drop is 14 points. Newcombe's 95% interval is **1.5 to 27.1 points**: the true drop could be barely anything or very large, but it is very unlikely to be zero or negative.

polcheck's verdict rule uses both ends of this interval:

- The **lower end** above zero is one of the three conditions for `REGRESSED`.
- The **upper end** below the threshold is what allows `PASS`: "any drop is smaller than what you care about".

## How it is computed

It is built from two single-rate intervals:

1. Compute a **Wilson score interval** for each batch's success rate. Wilson's interval is the standard well-behaved interval for one proportion; unlike the textbook formula it stays inside 0–100% and does not collapse at 100%.
2. Combine them. With rates `p1` (baseline) and `p2` (candidate) and Wilson limits `[l1, u1]` and `[l2, u2]`:

```text
lower = (p1 - p2) - sqrt( (p1 - l1)^2 + (u2 - p2)^2 )
upper = (p1 - p2) + sqrt( (u1 - p1)^2 + (p2 - l2)^2 )
```

Each limit of the difference takes the distance to the relevant limit of each rate and combines the two like sides of a right triangle.

**Worked example** (Newcombe's own, from the 1998 paper): 56/70 vs 48/80.

```text
p1 = 0.800, Wilson interval [0.692, 0.877]
p2 = 0.600, Wilson interval [0.490, 0.700]
difference = 0.200
lower = 0.200 - sqrt(0.108^2 + 0.100^2) = 0.0524
upper = 0.200 + sqrt(0.077^2 + 0.110^2) = 0.3339
```

The intermediate values shown are rounded to three places; the results are from the unrounded calculation. polcheck reproduces the published `0.0524 to 0.3339` exactly.

## Why not the textbook interval

The textbook ("Wald") interval is `difference ± 1.96 × standard error`. It fails exactly where robot policies live, near 100% success:

| Baseline | Candidate | Wald interval | Newcombe interval |
|---|---|---|---|
| 48/50 | 41/50 | 2.1 to 26.0 points | 1.5 to 27.1 points |
| 50/50 | 50/50 | **0 to 0** | −7.1 to +7.1 points |
| 50/50 | 47/50 | lower limit ignores that 50/50 is uncertain | −2.2 to +16.2 points |

Two perfect batches of 50 do not prove the policies are identical: a policy that fails 5% of the time would still score 50/50 about 8% of the time. Wald claims certainty; Newcombe gives an honest ±7 points.

## A consequence you will run into

Because the interval is honest about small samples, **an unpaired comparison needs many runs to earn a `PASS`**. The upper limit must be below the threshold (3 points by default). Even in the best case, two flawless batches:

| Runs per side (both 100%) | Upper limit of the interval | Verdict at a 3-point threshold |
|---|---|---|
| 50 | 7.1 points | `INCONCLUSIVE` |
| 100 | 3.7 points | `INCONCLUSIVE` |
| 200 | 1.9 points | `PASS` |
| 500 | 0.8 points | `PASS` |

So with 50 unpaired runs per family, success rate can show `REGRESSED` or `INCONCLUSIVE` but essentially never `PASS`. The remedies are more runs, a larger threshold, or (best) a paired comparison: when both versions agree on every seed, the paired interval is exactly zero and the test passes at any sample size above `min_n`.

## Things worth knowing

- **No continuity correction.** polcheck uses Newcombe's "method 10" (plain Wilson limits), which his paper recommends as the best balance of accuracy and simplicity.
- **It assumes independent batches.** For paired data polcheck does not use it; the paired interval comes from a bootstrap over seeds.
- **It is an interval, not a test.** The p-value still comes from Fisher's test. The two usually agree, but near the boundary the interval can exclude zero while Fisher's p is just above 0.05 (as in the 48/50 vs 41/50 example, p = 0.051). `REGRESSED` needs both, so that case is `INCONCLUSIVE`.

## In the code

`wilson_interval(k, n, alpha)` and `newcombe_interval(k1, n1, k2, n2, alpha)` in `src/polcheck/stats/success.py`. Tested against the published example and eight reference cases from `statsmodels`, including rates of exactly 0% and 100%.

## Reference

Newcombe, R. G. (1998). "Interval estimation for the difference between independent proportions: comparison of eleven methods." *Statistics in Medicine* 17, 873–890. Wilson, E. B. (1927). "Probable inference, the law of succession, and statistical inference." *Journal of the American Statistical Association* 22, 209–212.

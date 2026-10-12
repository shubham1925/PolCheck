# Benjamini–Hochberg correction

**Used for:** adjusting the p-values of all tests in one comparison, so that running many tests does not manufacture false alarms.

## What it represents

A single test at the usual 0.05 level raises a false alarm 5% of the time when nothing has changed. A polcheck comparison runs about 35 tests (families × measures, plus tail tests). If each had its own 5% chance of a false alarm, a comparison of two identical policies would flag *something* most of the time, and people would stop trusting the tool.

There are two common ways to fix this:

- **Bonferroni:** demand that each p-value be below 0.05 ÷ 35. This almost never raises a false alarm, but it is so strict that real regressions get missed.
- **Benjamini–Hochberg (BH):** control the *false discovery rate*. Of all the tests you end up flagging, at most about 5% (on average) are false alarms. It is much less strict when several tests show real effects, which is the usual situation: a genuinely worse policy is worse in several families at once.

polcheck uses BH with a rate `q` of 0.05 (`fdr_q` in `polcheck.toml`).

## How it is computed

1. Sort the p-values from smallest to largest.
2. Multiply each by `m / rank`, where `m` is the number of tests and `rank` is its position (1 for the smallest).
3. Working from the largest back to the smallest, make sure no adjusted value is larger than the one after it (take the running minimum), and cap at 1.

A test is flagged when its adjusted p-value is below `q`.

**Worked example** with 8 tests:

| Rank | p-value | × 8 / rank | BH adjusted | Bonferroni (× 8) |
|---|---|---|---|---|
| 1 | 0.001 | 0.008 | **0.008** | 0.008 |
| 2 | 0.008 | 0.032 | **0.032** | 0.064 |
| 3 | 0.012 | 0.032 | **0.032** | 0.096 |
| 4 | 0.030 | 0.060 | 0.060 | 0.240 |
| 5 | 0.040 | 0.064 | 0.064 | 0.320 |
| 6 | 0.200 | 0.267 | 0.267 | 1 |
| 7 | 0.500 | 0.571 | 0.571 | 1 |
| 8 | 0.900 | 0.900 | 0.900 | 1 |

BH flags three tests at 0.05; Bonferroni flags one. Tests 4 and 5 had raw p-values below 0.05 and would have been flagged without any correction; after correction they are not.

## How it shows up in polcheck

- Results carry both `p_value` (raw) and `p_adjusted`. The verdict rule uses `p_adjusted`.
- The adjusted value is never smaller than the raw one.
- **A lone small p-value is not enough.** One test with p = 0.004 among 34 unremarkable ones adjusts to 0.14 and is not flagged. A change has to be clear (very small p) or consistent across tests to count.
- **Every test you add costs a little power.** This is one reason per-family tail tests only run with at least 100 runs per side: at smaller sizes they are mostly noise, and each one would dilute the tests that matter.
- **Info-only tests are corrected separately.** When two batches were logged at different rates, the rate-sensitive tests are shown but cannot affect the verdict. They are adjusted among themselves, so they neither weaken nor strengthen the tests that count.

## Things worth knowing

- **It limits the share of false alarms among flagged tests, not the chance of any false alarm at all.** If a comparison flags 20 tests, expect about one of them to be noise. For the *overall* verdict, the false-alarm rate is measured directly by comparing a policy against itself many times (the A/A experiment in the validation report).
- **It assumes the tests are independent or positively related.** polcheck's tests are positively related (the same runs feed several measures, and families share a policy), which is the case the method is proven for.
- **It is not the only safeguard.** `REGRESSED` also requires the confidence interval to exclude zero and the estimated change to reach the team's threshold, so statistical significance alone never fails a build.
- **Whole-number data makes it a little conservative.** Exact tests such as McNemar and Fisher produce p-values in jumps, which errs toward fewer flags.

## In the code

`benjamini_hochberg(p_values)` in `src/polcheck/stats/multiple.py`; applied in `_finish` in `src/polcheck/compare.py`. Tested against `statsmodels` on seven sets of p-values, a hand-worked example, and property tests (adjusted values are never smaller than raw ones, never above 1, and keep the original ordering).

## Reference

Benjamini, Y. and Hochberg, Y. (1995). "Controlling the false discovery rate: a practical and powerful approach to multiple testing." *Journal of the Royal Statistical Society, Series B* 57, 289–300.

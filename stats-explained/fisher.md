# Fisher's exact test

**Used for:** success rate, when the comparison is unpaired (the two batches used different seeds, or determinism could not be confirmed).

## What it represents

Two batches, each with a count of successes and failures:

| | Succeeded | Failed | Total |
|---|---|---|---|
| **Baseline** | 48 | 2 | 50 |
| **Candidate** | 41 | 9 | 50 |
| **Total** | 89 | 11 | 100 |

Fisher's question is: *there were 11 failures in total. If the policy version made no difference, those 11 failures would be scattered between the two batches at random. How often would random scattering put them as unevenly as 2 and 9, or more unevenly?*

It treats the totals as fixed and counts the ways the failures could have been dealt out, like dealing cards. That makes it exact: there is no bell-curve approximation, so it stays valid at the small counts and near-100% rates typical of a good policy.

## How it is computed

With the row and column totals fixed, the number of baseline failures follows a hypergeometric distribution. The two-sided p-value adds up the probability of every table that is as unlikely as the observed one, or less likely.

**Worked example: the classic.** Fisher's original illustration was a lady who claimed she could tell whether milk or tea was poured first. Given 8 cups, 4 of each kind, suppose she identifies 3 of the 4 milk-first cups correctly.

```text
Ways to pick 4 cups out of 8: 70
Getting all 4 right:  1 way   -> probability 1/70  = 0.014
Getting 3 right:     16 ways  -> probability 16/70 = 0.229
Getting 2 right:     36 ways  -> probability 36/70 = 0.514
Getting 1 right:     16 ways  -> 0.229
Getting 0 right:      1 way   -> 0.014

Two-sided p = tables as unlikely as "3 right" = (16 + 1 + 16 + 1) / 70 = 34/70 = 0.486
```

Three out of four is entirely consistent with guessing.

**Worked example: two policy versions.** For the table at the top (48/50 vs 41/50), p = 0.051. A 14-point drop on 50 runs per side is right at the edge of what this test can confirm. With 200 runs per side and the same rates (190/200 vs 162/200), p = 0.00002.

## What polcheck reports alongside it

- **Estimate:** baseline rate minus candidate rate (0.96 − 0.82 = 0.14).
- **Confidence interval:** the [Newcombe score interval](newcombe.md) for the difference.
- **Method label in results:** `Fisher exact; Newcombe score CI`.

## Things worth knowing

- **It is conservative.** Because counts are whole numbers, the p-value moves in jumps and tends to err on the large side. Fisher's test rarely raises a false alarm, at the cost of missing some real drops. That matches polcheck's principle that a false alarm costs more trust than a miss.
- **It is much weaker than the paired test.** It cannot tell that seed 17 failed in both batches for the same reason. If your simulator is deterministic, record a few repeated seeds so polcheck can verify it and use [McNemar](mcnemar.md) instead.
- **It needs a lot of runs for small drops.** To detect a 3-point drop from 95% with 80% power takes roughly 1,060 unpaired runs per side; a 5-point drop takes about 435; a 10-point drop about 140. polcheck prints this estimate next to inconclusive success-rate tests.
- **The two rows must be independent.** Runs in one batch must not influence the other. Repeat-check copies inside a deterministic batch are exact duplicates, so polcheck counts each seed once before testing.

## In the code

`unpaired_success(base_k, base_n, cand_k, cand_n, alpha=...)` in `src/polcheck/stats/success.py`, which calls `scipy.stats.fisher_exact`. Tested against the tea-tasting hand calculation and five reference tables.

## Reference

Fisher, R. A. (1935). *The Design of Experiments*, chapter 2 (the lady tasting tea). For the two-sided definition used here, see the `scipy.stats.fisher_exact` documentation.

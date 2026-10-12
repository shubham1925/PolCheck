# The statistics behind a polcheck verdict

Plain-language notes on the statistical methods polcheck uses, what each one answers, and how to read its output. Every number in the worked examples was computed with polcheck's own code, which is tested against independent reference implementations (`tests/fixtures/stats/reference.json`).

| Page | Method | Question it answers |
|---|---|---|
| [mcnemar.md](mcnemar.md) | McNemar exact test | Same seeds on both sides: did more seeds flip to failure than chance explains? |
| [fisher.md](fisher.md) | Fisher exact test | Different or unmatched runs: are two success rates really different? |
| [newcombe.md](newcombe.md) | Newcombe score interval | How big could the difference between two success rates plausibly be? |
| [benjamini-hochberg.md](benjamini-hochberg.md) | Benjamini–Hochberg correction | With dozens of tests in one comparison, which results can be trusted? |

## How they fit together

A comparison runs one test per family per measure (about 35 on the demo). Each test produces three things:

1. **An estimate** of the change, oriented so that positive means the candidate is worse.
2. **A 95% confidence interval** around it: the range of changes the data is compatible with.
3. **A p-value**: how surprising the data would be if nothing had changed.

The p-values of all tests are then adjusted together (Benjamini–Hochberg), and each test gets a verdict against its threshold `T`, the smallest change the team cares about:

| Verdict | Needs all of |
|---|---|
| `REGRESSED` | adjusted p below 0.05, **and** the interval excludes zero, **and** the estimate is at least `T` |
| `PASS` | the whole interval is below `T` |
| `INCONCLUSIVE` | anything else, or fewer than `min_n` runs |

The p-value guards against calling noise a regression. The threshold guards against flagging changes too small to matter. The interval is what allows a confident `PASS`: "we looked, and any drop is smaller than what you care about". Without enough runs the interval is wide, and the honest answer is `INCONCLUSIVE`.

## Which method is used when

| Data | Success rate | Continuous measures (smoothness, time, …) |
|---|---|---|
| **Paired** (both batches deterministic, same seeds) | McNemar exact test; bootstrap interval over seeds | Wilcoxon signed-rank test; BCa bootstrap interval of the mean per-seed difference |
| **Unpaired** | Fisher exact test; Newcombe interval | Mann–Whitney U test; BCa bootstrap interval of the difference in means |
| Either | | Tail test: bootstrap of the difference in 95th percentiles |

Paired comparisons are far more sensitive, because each seed acts as its own control: scene difficulty cancels out. That is why polcheck checks determinism and matches seeds before choosing.

## The other methods, briefly

- **Bootstrap.** Resample the runs with replacement thousands of times and recompute the statistic each time; the spread of the results is the uncertainty. The *percentile* method reads the interval straight off that spread. *BCa* (bias-corrected and accelerated) adjusts it for skew, which matters for measures such as contact force. polcheck seeds every bootstrap, so the same two batches always give the same verdict.
- **Wilcoxon signed-rank test** (paired). Ranks the per-seed differences by size and asks whether positive and negative ones balance. It does not assume a bell curve, so one wild run cannot dominate.
- **Mann–Whitney U test** (unpaired). Asks whether values from one batch tend to be larger than values from the other, again using ranks.
- **Tail test.** Averages can hide a worse worst case. polcheck bootstraps the difference in the 95th percentile (the bad end of each measure). One pooled test per measure always runs; per-family tail tests need 100 runs per side, because a 95th percentile of 50 runs rests on two or three of them.

One consequence of mixing a rank test (for p) with a mean (for the interval): on measures that are zero in most runs and huge in a few, the interval can exclude zero while the rank test stays unconvinced. polcheck then reports `INCONCLUSIVE`, which is the cautious reading.

## Where this lives in the code

- `src/polcheck/stats/success.py`: McNemar, Wilson, Newcombe, Fisher.
- `src/polcheck/stats/continuous.py`: Wilcoxon, Mann–Whitney, bootstrap intervals, tail test.
- `src/polcheck/stats/multiple.py`: Benjamini–Hochberg.
- `src/polcheck/stats/verdict.py`: the verdict rule and the overall rollup.
- `src/polcheck/compare.py`: decides which test applies and runs them all.

# Research Notes: Benford Conformity vs. Convergence in Generalized Collatz Space

Exploratory findings from `bridge_export.py` and `convergence_scan.py`, scanning the
generalized Collatz parameter cube `(a, b, c)` shared with
[collatz-box-universes](https://github.com/numberwonderman/Collatz-box-universes):

```
n -> n/a      if n = 0 (mod a)
n -> b*n + c  otherwise, then maximally divided by a (the "shortcut")
```

The classical Collatz conjecture is `(a, b, c) = (2, 3, 1)`.

## Summary

Searching a large region of `(a, b, c)` space for parameter sets with unusually strong
Benford's-Law digit conformity turned up a real, reproducible structure - but not new
mathematics. Every strong-conformity point found is *explained* by the same classical
argument (Benford conformity of near-geometric growth), and every one of them is almost
entirely divergent. Separately, a convergence-rate scan found that genuine "returns to 1"
behavior is essentially isolated at the classical `(2, 3, 1)` point: every immediate
neighbor in parameter space collapses from 100% convergence to under 0.4%. That's a clean,
quantifiable confirmation of something usually only argued heuristically: the classical
Collatz rule sits on a knife's edge, not in a basin of "nearby easier versions."

## Finding 1: the `c=a, b=2a` ridge is a textbook case, not a discovery

A cube scan centered near `(7, 14, 8)` (`bridge_export.py --center 7 14 8 --side 5`) found
that every point with `c == a` and `b == 2a` - e.g. `(6,12,6)`, `(7,14,7)`, `(8,16,8)`,
`(15,30,15)` - shows exceptional Benford conformity (MAD as low as 0.00043) *at every
scale of `a` tested, from 4 to 20*.

Tracing actual sequences explains why: for this family, the odd-branch formula is
`b*n + c = a(2n+1)`, so the shortcut's "maximal division by `a`" almost always divides
out exactly once, collapsing the whole rule to **n -> 2n+1** - the classical doubling map.
A sequence that grows by a constant ratio obeys Benford's Law because `log10(ratio)` is
irrational, so leading digits equidistribute mod 1 (the standard proof that powers of 2
obey Benford's Law). This family is a clean, provable special case, not a mystery.

## Finding 2: the broader non-degenerate scan is the same phenomenon at other ratios

A 729-point scan (`a,b,c` in `10-18, 14-22, 8-16`) found points outside the `c=a,b=2a`
family with even better conformity, e.g. `(12,21,9)` at MAD 0.00041. Tracing these showed
the same mechanism: >99% of sampled starting values diverge, growing by a roughly constant
ratio `b/a` per step. This is the same equidistribution argument generalized to `r = b/a`
instead of `r = 2`. See `convergence_scan.py`'s `is_degenerate()` for the residue-based
filters (`b % a == 0`, `c % a == 0`) that catch the most extreme related cases.

## Finding 3: convergence and Benford conformity trade off

Within the `a,b,c in 2-10 x 2-10 x 1-9` cube (`convergence_scan.py`, 729 points,
sampled over `n in [2, 3000]`), **only one non-degenerate point reaches >=90%
convergence: `(2, 3, 1)`, at 100%.** Every other 100%-convergent point in the cube is
part of the trivial `a=b=c` family (odd-branch formula `a(n+1)`, collapsing to
`n -> n+1` between divisions - convergent but not chaotic, hence its weak MAD ~0.02-0.03).

Classical Collatz's own MAD (0.00898) is far weaker than the divergent ridge's
(0.0004-0.0009) - about 20x worse. That's a real, coherent tradeoff: **bounded,
"returns to 1" dynamics mixes leading digits far less cleanly than unbounded geometric
growth does.** The two things people find dramatic here - "wow, extreme Benford
conformity!" and "wow, always converges!" - do not co-occur at strength in this search.

## Finding 4: `(2, 3, 1)` is a cliff, not a slope

Reusing the 729-point cube's local neighborhood around `(2, 3, 1)`:

| Manhattan distance | points | avg convergence |
|---|---|---|
| 0 | 1 | 100% |
| 1 | 4 | 0.33% |
| 2 | 8 | 13.8% (only because of the unrelated trivial `(2,2,2)`) |

All four immediate neighbors - `(2,2,1)`, `(2,3,2)`, `(2,4,1)`, `(3,3,1)` - collapse to
under 0.4% convergence. Changing any single classical parameter by exactly 1 destroys
convergence almost completely. There is no smooth basin of attraction around the
classical rule in this parameter family.

## Caveats

- All results are empirical, from finite samples (`n` up to a few thousand,
  `max_iterations` up to 3000) - not proofs, and not exhaustive.
- "Convergence" here means "reached 1 within `max_iterations`, for sampled `n`"; a point
  reported as divergent could in principle converge for larger `n` or more iterations
  than sampled.
- The `is_degenerate()` filters catch the mechanisms found so far, not necessarily every
  way the shortcut can trivially collapse the dynamics.

## Reproducing

```bash
# Finding 1
python3 bridge_export.py --center 7 14 8 --side 5 --range 1 5000 --max-iterations 2000 --output research_notes_data/cube_export_7_14_7_fine.json

# Finding 2
python3 bridge_export.py --center 14 18 12 --side 9 --range 1 5000 --max-iterations 2000 --output research_notes_data/cube_export_broad_scan.json

# Findings 3 and 4 (same run - the neighborhood in Finding 4 is read straight out of this scan's output)
python3 convergence_scan.py --a-range 2:10 --b-range 2:10 --c-range 1:9 --range 2 3000 --max-iterations 3000 --output research_notes_data/convergence_scan_results.json
```

## Open questions worth scanning for next

- Is there a *second* isolated high-convergence, non-degenerate point anywhere in a much
  larger `(a, b, c)` region, or is `(2, 3, 1)` unique within reach of this search?
- Does the "cliff" shape (Finding 4) hold at other scales - e.g. is there an analogous
  isolated point near `(2k, 3k, k)` for some `k`, or does scaling `a` uniformly break the
  convergence property entirely (distinct from the divergent `c=a,b=2a` ridge, which
  scales cleanly but never converges)?
- Among points with substantial but non-dominant convergence (10-50%, of which there are
  14 in the base cube), does Benford conformity of *only the converging trajectories*
  (as opposed to all sampled trajectories) show a different pattern?

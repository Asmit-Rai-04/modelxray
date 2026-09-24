# Failure clustering validation checkpoint

## Change
Replaced transitive connected-component clustering with complete-linkage style grouping.
A hypothesis can join a failure cluster only when its region overlaps every existing cluster member at or above the Jaccard threshold.

## Why
Connected components can chain unrelated regions:
A overlaps B, B overlaps C, while A and C do not overlap. The old algorithm would merge all three.

## Validation
- Full backend suite: 51 passed
- Added regression test for chain-merge prevention
- Existing overlapping-hypothesis consolidation remains covered

## Scope
This is geometric consolidation only. It does not claim causal equivalence between hypotheses.

## Remaining work
Add a benchmark with known same-mechanism and different-mechanism hypothesis groups and measure pairwise cluster precision/recall before calling clustering reliable.

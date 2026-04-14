# ORS Benchmark Summary: v1 vs v2 vs v3


## 1. Scope

This summary reports benchmarking results for charging sessions of length 30 to 60, comparing:

- **v1**: vanilla Stage-1 DP in `src/mt4xai/ors.py` via `stage1_dp` (ORS paper-exact)
- **v2**: legacy DP-prefix in `src/mt4xai/ors.py` via `stage1_dp_prefix` (MT4XAI master thesis paper-exact)
- **v3**: DP-prefix in `src/mt4xai/ors_v3.py` via `stage1_dp_prefix` (new addition - Github only)

It also reports end-to-end ORS timing for:

- **full ORS v1 baseline**: `ors.py` with `stage1_mode="dp"`
- **full ORS v3**: `ors.py` with `stage1_mode="dp_prefix_v3"`

## 2. Algorithmic Differences and Runtime Complexity

### 2.1 Stage-1 error table construction

All variants build four tables over all pairs `(i, j)` with `i < j`:

- `err[i, j]`
- `errL[i, j]`
- `errR[i, j]`
- `errA[i, j]`

Differences:

- **v1 (DP)**: direct summation for all four tables  
  Reference: `build_error_tables_no_prefix` in `ors.py`
- **v2 (legacy DP-prefix)**: prefix sums only for `err`, direct summation for `errL`, `errR`, `errA`  
  Reference: `build_error_tables_with_prefix` in `ors.py`
- **v3 (DP-prefix v3)**: prefix sums for all four tables  
  Reference: `build_error_tables_with_prefix` in `ors_v3.py`

Big O for the table builder:

- **v1**: `O(n^3)`
- **v2**: `O(n^3)` (dominant terms remain `errL`, `errR`, `errA`)
- **v3**: `O(n^2)`

### 2.2 Stage-1 DP candidate generation

All variants then run heap based DP ranking to produce top `q` candidates.

- This phase is structurally similar across versions.
- Practical complexity is driven by heap operations over many `(i, j, rank)` states.
- A reasonable upper bound is `O(q * n^2 * log(q*n))` for the ranking phase.

So total Stage-1 complexity is approximately:

- **v1**: `O(n^3 + q*n^2*log(q*n))`
- **v2**: `O(n^3 + q*n^2*log(q*n))`
- **v3**: `O(n^2 + q*n^2*log(q*n))`

### 2.3 Candidate representation difference in v3

`v3` introduces optional explicit pivot values in Stage-1 candidates for both-ends single-line candidates.

- `v2`: candidates are represented as `(cost, pivots)`
- `v3`: candidates are represented as `(cost, pivots, optional_pivot_values)`

This improves cost consistency with reconstruction for both-ends candidates.

### 2.4 Meaning of the four error tables

For each pivot pair \((i, j)\), Stage-1 defines a line through \((i, y_i)\) and \((j, y_j)\).  
The tables store **SSE** (sum of squared errors), where:

\[
\mathrm{SSE} = \sum_{t \in \mathcal{I}} (y_t - \hat{y}_t)^2
\]

and \(\hat{y}_t\) is the value on that line at index \(t\).

- `err[i,j]`: segment-only SSE on \([i..j]\).  
  Uses only the local segment between the two pivots.
- `errL[i,j]`: left-extended SSE on \([0..j]\).  
  Used when evaluating first-segment style candidates.
- `errR[i,j]`: right-extended SSE on \([i..T-1]\).  
  Used when evaluating last-segment style candidates.
- `errA[i,j]`: all-points SSE on \([0..T-1]\).  
  Used for both-ends single-line candidates.

## 3. Benchmark Setup

### 3.1 Data and sampling

- Dataset: `Data/etron55-charging-sessions-public-processed.parquet`
- Session length filter: `30 <= T <= 60`
- Total sessions in range: `18,208`
- Benchmarked sessions for Stage-1: `400` (random sample, seed `42`)
- Benchmarked sessions for full ORS: `25` from test split in range `30..60` (seed `42`)

### 3.2 Stage-1 timing parameters

- `alpha = 0.001`
- `beta = 3.0`
- `q = 250`
- Repetitions per session: `2` (median used per session)

### 3.3 End-to-end ORS timing context

- Model: `Models/final/final_model.pth`
- Device: `cuda:0`
- Shared parameters unless noted:
  - `dp_alpha=0.001`, `beta=3.0`, `gamma=0.05`
  - `t_min_eval=1`, `anchor_endpoints="last"`
  - `min_k=1`, `max_k=15`
  - `epsilon_mode="fraction"`, `epsilon_value=0.3`
  - `soc_stage1_mode=None`
  - `random_seed=42`

## 4. Stage-1 Results (30 to 60)

Sample characteristics:

- `n_min=30`
- `n_median=36`
- `n_mean=37.8925`
- `n_max=60`

Aggregate runtime totals over 400 sessions:

- **v1 total**: `18.5683 s`
- **v2 total**: `17.8014 s`
- **v3 total**: `15.1456 s`

Speedups:

- **v2 vs v1**: `1.0431x` = **`+4.31%`**
- **v3 vs v1**: `1.2260x` = **`+22.60%`**
- **v3 vs v2**: `1.1753x` = **`+17.53%`**

Per-length-bin speedups:

- `30..39`
  - v2 vs v1: `+3.41%`
  - v3 vs v1: `+20.58%`
  - v3 vs v2: `+16.61%`
- `40..49`
  - v2 vs v1: `+5.66%`
  - v3 vs v1: `+24.51%`
  - v3 vs v2: `+17.84%`
- `50..60`
  - v2 vs v1: `+4.82%`
  - v3 vs v1: `+26.53%`
  - v3 vs v2: `+20.72%`

## 5. Stage-1 Output Consistency

Top candidate checks across the 400-session sample:

- **v1 vs v2**
  - exact top pivot match: `400 / 400`
  - top cost close match: `400 / 400`
- **v2 vs v3**
  - exact top pivot match: `214 / 400`
  - top cost close match: `400 / 400`
  - mismatches with explicit pivot values in v3: `186 / 186` mismatches

Interpretation:

- v1 and v2 are output-equivalent for top candidate on this sample.
- v2 and v3 preserve top cost, while pivot representation differs in the expected explicit-value cases.

## 6. Full ORS Results: v3 vs v1

### 6.1 Configuration A: robustness-heavy

Parameters:

- `stage1_mode="dp"` for v1, `stage1_mode="dp_prefix_v3"` for v3
- `dp_q=120`
- `R=120`

Results across 25 sessions:

- v1 total: `65.7866 s`
- v3 total: `67.1143 s`
- v3 vs v1 speedup: `0.9802x` = **`-1.98%`**
- label agreement: `25 / 25`
- `k` agreement: `16 / 25`
- objective close agreement: `20 / 25`

### 6.2 Configuration B: stage1-heavier

Parameters:

- `stage1_mode="dp"` for v1, `stage1_mode="dp_prefix_v3"` for v3
- `dp_q=220`
- `R=20`

Results across 25 sessions:

- v1 total: `39.9782 s`
- v3 total: `38.6330 s`
- v3 vs v1 speedup: `1.0348x` = **`+3.48%`**

## 7. Analysis of Full ORS Speedup

The full ORS speedup of v3 vs v1 is sensitive to where runtime is spent:

- When robustness sampling is high (`R=120`), Stage-2 cost dominates, so Stage-1 improvements have little impact and may be masked by candidate-path differences.
- When Stage-1 has relatively higher weight (`R=20`, higher `dp_q`), v3 shows a measurable end-to-end gain.

This behaviour is consistent with the complexity decomposition:

- Stage-1 improvement in v3 reduces the table component from `O(n^3)` to `O(n^2)`.
- Full ORS still includes substantial Stage-2 work that scales with candidate count and robustness sampling budget.

## 8. Practical Conclusion

- For Stage-1 candidate generation, v3 is clearly faster than both v1 and v2 on session lengths 30 to 60.
- v2 gives only a modest gain over v1 because its dominant table terms remain cubic.
- v3 provides the strongest Stage-1 improvement by making all error-table terms prefix based.
- End-to-end ORS speedup depends on workload balance between Stage-1 and Stage-2.

## 9. Executive Summary

| Transition | Main technical contribution | Complexity impact | Measured Stage-1 contribution (30..60) | Output impact |
|---|---|---|---:|---|
| **v1 -> v2** | Introduces prefix sums for `err[i,j]` only | Stage-1 remains \(O(n^3 + q n^2 \log(qn))\) | \(\Delta = +4.31\%\) | Top candidate is output-equivalent in benchmark (\(400/400\) exact pivot and cost match) |
| **v2 -> v3** | Extends prefix-sum SSE to `errL`, `errR`, `errA` and adds explicit pivot-value candidates for both-ends lines | Stage-1 shifts \(O(n^3 + q n^2 \log(qn)) \rightarrow O(n^2 + q n^2 \log(qn))\) | \(\Delta = +17.53\%\) | Top cost is preserved (\(400/400\)), while pivot representation changes for explicit-value cases (\(186/400\)) (no pseudo-pivots for \(k=1\)) |

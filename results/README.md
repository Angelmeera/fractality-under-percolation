# Regenerated results

The JSONs behind the tables and figures of the paper. Each carries its own `argv`, so a result is
never separated from the command that produced it.

All covering-derived numbers here were regenerated under a single seeding order (see
`docs/METHODS_NOTES.md`, pitfall 1). Output of ours that predates that fix is
superseded and is not included.

`docs/REPRODUCING.md` maps each file to its table and explains the keys. The two worth
knowing before you open anything:

- **`direct`** is the joint OLS of `ln m` on `(ln L, ln k)`. It uses neither γ nor δ, so
  `α + β d_k = d_B` is a testable statement rather than an identity.
- **`alpha_macroscopic` / `beta_macroscopic`** are derived from `d_B`, γ and δ through
  the inversion formulae. With `d_B` and γ fixed both are functions of δ alone, so
  `closure_macroscopic_pct` is `0.0` by construction. It is not a test of anything.
- **`table3_nulls.json`**: the `shuffled` and `independent` entries are means over
  `ndraw` draws with the stated `seed`, and `beta_sd` is the spread across draws, not a
  standard error. `edge_centred` uses the box definition in `table3_nulls.edge_boxes`.
- **`dblp60_percolation.json`**: one row per `p`, with the per-realization values under
  `realizations`. `delta`, `alpha_macroscopic` and `beta_macroscopic` are `NaN` for a
  realization in which no `l_B` level supplies a per-level δ; there is deliberately no
  fallback estimator (see the docstring of `src/dblp_percolation.py`). `<key>_n` counts
  the realizations with a finite value.
- **`results_large.json`** and **`results_yf.json`** carry no `argv`; they are the
  outputs of `run_large.py` and `run_yf.py` with no arguments. `run_yf.py` reproduces
  its file exactly.
- **`extended_covers.json`**: one record per network with `N_B` at every `l_B` from 2 to
  `l_max`, every candidate grid (`family` = geometric, arithmetic or consecutive), the best
  grid overall and the best geometric one, and `dAIC_range_all_sizes` over every
  subgrid of those sizes (not computed for the FSFN). Table 8's range column is over the
  sizes of the geometric scan only; this file holds the wider ranges quoted in Sec. 4.9.
- **`www_scan.json`** is scored offline by `combine_grids.py` from `www_comm.json`
  (`l_B = 2, 4, 6`) and `www_l35.json` (`l_B = 3, 5`).
- **`covering_memb.json`**: the MEMB rows of Table 9. `commensurate` holds `t = 4, 5` at
  `r_B = 1, 4, 13`, where MEMB returns exactly `N_{t-1}, N_{t-2}, N_{t-3}`;
  `integer_ladder_t5` holds `t = 5` at `r_B = 1..6, 9`.
- **`tiebreak_spread.json`**: intact `t = 4` covers of `G^A` and `G^B` under the `str(node)`,
  numeric and NumPy-argsort tie-breaks, and over 30 random orders within degree classes
  (`seed` 1).
- **`results_large.json`**: only `B_pc` (the simulated `p_c(N)` of Sec. 4.2) is used in
  the paper. `A_intact` (the old edge-unit estimator) and `C_dyn` are superseded.
- **`results_fss.json`** and **`dblp_sweep_canonical.json`** have a top-level `argv`
  and `note`: the finite-size-scaling run depends on `--jobs` through its seeds, and
  both files were reproduced bit-for-bit from the current code.

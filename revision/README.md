# Analyses added in the revision (October 2026)

The referee report on the submitted manuscript asked for four things that the original
pipeline in `src/` did not do:

1. measure `d_B`, `d_k`, `α` and `β` on the **same** percolated graphs;
2. explain why `β(p)` depends on the definition of a fragmented box's mass;
3. test the box-size grid against alternative selection criteria;
4. shorten the paper.

This directory holds the code, result files and logs behind the first three. The
revised paper is *Mass-law scaling of fractal scale-free networks under bond percolation*
(main text and Supplementary Material, SM).

Everything here is new; nothing in `src/` or `results/` was changed by these analyses.
The scripts import the paper's own builder and coverings from `src/`
(`fast_cover.cover_one`, `covering.cover_memb`, `snap_networks`, `yakubo_fujiki`), so the
construction, node labels and seeding tie-break are the ones documented in
[`docs/METHODS_NOTES.md`](../docs/METHODS_NOTES.md).

## Running

Run from the repository root with the packages in `requirements.txt`. Every script
finds `lib.py` and `../src` relative to its own location, so the working directory only
matters for the output path you pass.

```bash
# one example: the instrumental-variable fits of Table 4 and Fig. 2(a,c)
python revision/a13_iv_all.py GB 5 20 1,0.95,0.9,0.85,0.8,0.75,0.7,0.65 revision/results/a13_GB_t5.json
python revision/a13_analyze.py revision/results/a13_GB_t5.json
```

The model runs need no data. `a12_grid_criteria.py` reads the WWW box counts from
`results/www_scan.json` and `results/www_l7.json`. Its AS and DBLP box counts are cached
in `revision/results/a12_NB_cache.json`, so `data/` is needed only if you delete the
cache. The FSFN generators are `G^B` (`GB`, triangle-free) and `G^A` (`GA`, clustered).

## The three box-mass conventions

`lib.py` builds the construction-aware hub-centred boxes of a percolated FSFN under the
three conventions compared in the paper (Sec. 2.6):

| key | convention |
|---|---|
| `pub` | the **largest connected component** of a box's surviving members (the reference convention) |
| `mass` | **total surviving mass**, with `L` and `k` from the largest component |
| `retile` | **re-tiled**: every other surviving fragment is merged into the box that a breadth-first search on the fragment graph reaches first, so the boxes are connected and tile the giant component |

`L` is the box diameter plus one, and a box enters a fit only if its largest component
holds at least 3 nodes. The fits are:

- **OLS** on the damaged hub degree `k`;
- **IV** (two-stage least squares) with the pre-damage degree `k0` as the instrument;
- **FE IV**: IV with a separate intercept per level `τ ≥ 2`.

## Scripts, results and where they appear in the paper

Numbers in the "paper" column are those of the **revised** manuscript ("S" = Supplementary
Material). Commands are the ones that produced the files; seeds are fixed in each script
(the `SEED` environment variable, where a script reads it, defaults to 99).

| script | result file(s) in `results/` | paper | command |
|---|---|---|---|
| `a13_iv_all.py` | `a13_GB_t5.json` | Table 4, Fig. 2(a,c), Table S10; η, first stage, hub-out fraction (Secs. 2.6, 4.2) | `a13_iv_all.py GB 5 20 1,0.95,0.9,0.85,0.8,0.75,0.7,0.65 …` |
| | `a13_GB_t5_s7.json` | the forty further configurations (Sec. 4.2) | `SEED=7 a13_iv_all.py GB 5 40 0.9,0.8,0.7,0.65 …` |
| | `a13_GA_t5.json` | `G^A` values quoted in Sec. 4.2 | `a13_iv_all.py GA 5 20 1,0.9,0.8,0.75,0.72 …` |
| `a10_t7.py` | `a10_GB_t7.json`, `a10_GA_t7.json` | Fig. 2(b), Fig. 3, Sec. 4.3, Table S11, Table S12 (lower block) | `a10_t7.py GB 7 6 1,0.95,0.9,0.85,0.8,0.75,0.7,0.65 …`; `GA 7 6 1,0.95,0.9,0.85,0.8,0.75,0.72` |
| `a11_closure_iv.py` | `a11_GB_t5.json`, `a11_GA_t5.json` | greedy `d_B(p)`, `d_k(p)` (Sec. 4.1); Table 5, OLS, IV and frozen columns; mixed test on `G^A` (Sec. S10) | `a11_closure_iv.py GB 5 20 1,0.95,…,0.65 …`; `GA 5 20 1,0.95,0.9,0.85,0.8,0.75,0.72` |
| `a24_construction.py` | `a24_GB_t5.json`, `a24_GA_t5.json`, `a24_GB_t6.json`, `a24_GA_t6.json` (+ `*_summary.json` from `a24_analyze.py`) | Table 5, shuffled column; Tables S7 and S9; re-tiled and total-mass closures (Secs. 4.4, S9) | `a24_construction.py GB 5 20 …`; `GA 5 20 …`; `GB 6 8 1,0.9,0.8,0.7,0.65 … nogreedy`; `GA 6 8 1,0.9,0.8,0.75,0.72 … nogreedy` |
| `a15_cut_frames.py` | `a15_GB_t5.json`, `a15_GB_t6.json` | Table S8 (mass cut, Sec. 2.5); Table S12 upper block (construction frame) | `a15_cut_frames.py GB 5 20 1,0.95,…,0.65 …`; `GB 6 6 1,0.9,0.8,0.7,0.65 …` |
| `a8_iv.py` (+ `a8_lib.py`) | `a8_GB_t5.json`, `a8_GB_t6.json`, `a8_GA_t6.json` | first IV analysis; `t = 6` values in Sec. 4.2 (superseded at `t = 5` by `a13`, which reuses its configurations) | `a8_iv.py GB 5 20 …`; `GB 6 8 1,0.9,0.8,0.7,0.65`; `GA 6 8 1,0.9,0.8,0.75,0.72` |
| `ivL.py` | `logs/ivL.log` | `L` instrumented by `λ^τ` (Sec. 4.2) | `ivL.py` |
| `ivFE.py` | `logs/ivFE.log` | first level-fixed-effects IV check, superseded by the `fe` key of `a13` | `ivFE.py` |
| `a9_memb_all.py` | `a9_GB_t5.json` | MEMB over all radii `r_B = 1..16` (Sec. 4.1, Sec. S10) | `a9_memb_all.py GB 5 10 1,0.95,0.9,0.8,0.7,0.65 16 …` |
| `a1_memb.py` (+ `a1_analyze.py`) | `a1_GB_t5.json`, `a1_GA_t5.json`, `a1p_*.json` (+ `*_summary.json`) | MEMB at commensurate radii `1,4,13` and shifted radii `2,5,14` / `2,6,18` as a single covering (Table S13) | `a1_memb.py GB 5 30 1,0.95,…,0.65 1,4,13 …` (and `2,5,14`, `2,6,18`; `GA` down to `0.72`) |
| `a12_grid_criteria.py` | `a12_grid_criteria.json`, `a12_NB_cache.json` | Table S2: `d_B` under alternative grid-selection criteria (Sec. 2.3, Sec. S1) | `a12_grid_criteria.py` |
| `a6_grid.py`, `a6b_periodic.py` | `a6_grid.json`, `a6b_periodic.json` | open and periodic lattices and critical percolation clusters with known `d_B`, the calibration rows of Table S2 | `a6_grid.py`; `a6b_periodic.py` |
| `make_figs_rev.py` | — | draws `figures/fig2_mass_law_iv` (Fig. 2) and `figures/fig3_restoration` (Fig. 3) | `make_figs_rev.py` |

`*_table_rows.json` and `a15_tables.json` are the rows as typeset, kept for checking.
They are derived from the per-configuration files above.

**Exploratory checks**, kept because the text quotes some of their output:

- `rg_collapse.py`: exact RG flow `p_τ` and the per-level `β`;
- `stair.py`: the MEMB staircase `N_B(r_B)`;
- `stretch.py` and `atten.py`: first versions of the `t = 7` level-by-level and IV analyses;
- `t_memb.py`: MEMB timing.

`logs/` holds the console output of each run, including the per-configuration lines.

## Reproduction checks

Checks run in October 2026 on this directory as committed:

- **`a11_closure_iv.py`, `GB 5 1 1.0,0.95`:** reproduces the first two records of
  `a11_GB_t5.json` to `3e-13`. This also confirms `a8_lib.py`. The original
  `a8_lib.py` was not kept with the run, so it was restored from `a8_iv.py`.
- **`ivL.py`:** reproduces `logs/ivL.log` digit for digit.
- **`SEED=7 a13_iv_all.py`:** reproduces the first line of `logs/a13_GB_t5_s7.log`.
- **`a12_grid_criteria.py`:** run from the cache reproduces `a12_grid_criteria.json`
  to floating-point rounding (`1e-16`).
- **`make_figs_rev.py` and `src/make_fss_fig.py`:** reproduce the paper's Figs. 2 and 3
  and SM Fig. S5 pixel for pixel with matplotlib 3.10/3.11. Rasters made on a machine
  with different fonts differ slightly.
- **Every other script:** runs to completion on a `t = 4` network.

## Sample sizes

These are the bond configurations per value of `p` (`p = 1` is always one exact
evaluation):

- **20 at `t = 5`:** `a8`, `a11`, `a13`, `a15`, `a24`;
- **40:** the further `a13` set;
- **30:** `a1`;
- **10:** `a9`;
- **6–8 at `t = 6`;**
- **6 at `t = 7`** (`N = 1,198,374`).

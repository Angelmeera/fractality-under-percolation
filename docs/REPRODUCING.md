# Reproducing the numbers

Every command is run **from the repository root**. Each driver writes its own `argv`
into the JSON it produces, so a result file always carries a record of how it was made.

Wall-clock times below are for two cores; the FSFN runs need no data.

---

## Model sections (no data required)

```bash
# Table: the direct joint fit, FSFN rows
python src/run_snap.py --fsfn 4 --l-b 2 3 4 6 9 13 --jobs 2 --out results/fsfn4_hand.json   # ~2 s
python src/run_snap.py --fsfn 4 --l-b 2 4 10 28    --jobs 2 --out results/fsfn4_comm.json   # ~2 s
python src/run_snap.py --fsfn 5 --l-b 2 4 10 28    --jobs 2 --out results/fsfn5_comm.json   # ~5 s
python src/run_snap.py --fsfn 6 --l-b 2 4 10 28    --jobs 2 --out results/fsfn6_comm.json   # ~2 min

# Box-size grid selection (the scan also gives the competing-grid rows)
python src/run_snap.py --fsfn 5 --scan --lambdas 1.6 2.0 2.5 3.0 --l-max 30 --jobs 2 \
    --out results/fsfn5_scan.json                                                           # ~1 min
python src/run_snap.py --fsfn 6 --scan --lambdas 1.6 2.0 2.5 3.0 --l-max 30 --jobs 2 \
    --out results/fsfn6_scan.json                                                           # ~4 min

# Exact quantities, fig1 and fig4 (paper Figs 2 and 4); writes results/results_yf.json
python src/run_yf.py                                                                        # ~2 min

# Large generations: intact exponents t = 4..6 and the simulated p_c(N) of paper Fig. 1
python src/run_large.py                                  # writes results/results_large.json

# Table 3: correct fit, both nulls (1000 seeded draws), edge-centred boxes
python src/table3_nulls.py                                                                  # ~1 min

# Percolation critical exponents by finite-size scaling
python src/fss.py --n-p 33 --jobs 2 --out results/results_fss.json                        # ~15 min
# --jobs sets how the seeds are split into RNG streams: keep it at 2 to reproduce exactly

# The four covering algorithms compared
python src/run_covering.py --fsfn 4 5 --dblp 60 --out results/covering.json                 # ~8 min

# Non-fractal controls
CONTROLS_OUT=results/controls.json python src/controls.py                                   # ~20 s
```

## Empirical sections (data required)

```bash
python data/get_data.py     # SNAP graphs; see data/README.md for DBLP

python src/run_snap.py --www data/web-NotreDame.txt.gz --l-b 2 4 6 --jobs 2 \
    --out results/www_comm.json          # ~45 min; the l_B = 6 cover is the cost
python src/run_snap.py --as  data/as20000102.txt.gz --l-b 2 3 7 --jobs 1 \
    --out results/as_comm.json           # ~10 s
python src/run_snap.py --dblp data/dblp_weighted.tsv.gz --min-joint 60 --l-b 2 3 5 9 17 \
    --jobs 2 --out results/dblp60_comm.json     # ~1 min, mostly parsing the 51 MB file
python src/run_snap.py --dblp data/dblp_weighted.tsv.gz --min-joint 25 --l-b 2 3 4 5 8 11 \
    --jobs 2 --out results/dblp25_comm.json     # ~3 min

# Weak-tie threshold sweep
SWEEP_EDGELIST=data/dblp_weighted.tsv.gz SWEEP_THRESHOLDS="60 50 45 40 35 30 25" \
    SWEEP_OUT=results/dblp_sweep_canonical.json python src/dblp_sweep2.py   # ~5 min

# Percolation of the DBLP w>=60 backbone (paper Fig. 7); seeded, independent of PYTHONHASHSEED
python src/dblp_percolation.py                  # ~3 min, writes results/dblp60_percolation.json

# Candidate-grid scans for Fig. 6(a) (paper numbering)
python src/run_snap.py --as data/as20000102.txt.gz --scan --lambdas 1.4 1.6 2.0 2.5 \
    --min-points 3 --l-max 9 --jobs 1 --out results/as_scan.json                  # ~10 s
python src/run_snap.py --dblp data/dblp_weighted.tsv.gz --min-joint 60 --scan --l-max 40 \
    --jobs 1 --out results/dblp60_scan.json                                         # ~1 min
python src/run_snap.py --dblp data/dblp_weighted.tsv.gz --min-joint 25 --scan \
    --lambdas 1.6 2.0 --l-max 12 --jobs 1 --out results/dblp25_scan.json           # ~3 min
# WWW: the odd l_B covers, then every grid scored offline from l_B = 2..6 (also Fig. 6(b))
python src/run_snap.py --www data/web-NotreDame.txt.gz --l-b 3 5 --jobs 1 \
    --out results/www_l35.json                                                      # ~20 min
python src/combine_grids.py results/www_comm.json results/www_l35.json --name WWW \
    --lambdas 1.3 1.5 2.0 --min-points 3 --out results/www_scan.json

# Every l_B up to the pipeline's largest, all four grid families (Sec. 2.3, Sec. 4.9)
python src/extended_covers.py                  # ~10 min, writes results/extended_covers.json
```

## Audits

```bash
# Disconnected-box audit: per-l_B fractions and the connected-only refit, t = 4,5,6
python src/conn_audit.py                                   # ~3 min, writes results/conn_audit.json

# Mass-convention control: three definitions on identical bond configurations
NREAL=30 python src/m5_control_t5.py                       # ~4 min, writes results/m5_control_t5.json

# Delta-AIC over every >=3-point subgrid of every result file
python src/subgrid_daic.py                                 # ~2 min over results/
```

## Figures

File names predate the paper's figure numbering; the paper number is in brackets.
Every figure is drawn at its printed size, 5.0 in (12.7 cm) wide with 6-7.5 pt
lettering, by the shared style in `src/paper_style.py`, and is written twice:
a 600-dpi PNG (what pdflatex includes) and a vector EPS. Both go to `figures/`
unless `FIGS_DIR` and `FIGS_EPS_DIR` say otherwise; to write straight into the
manuscript's folders, run e.g.
`FIGS_DIR=../Corrected_Analysis/draft_v2/figs FIGS_EPS_DIR=../Corrected_Analysis/draft_v2/figs_eps python src/make_fss_fig.py`.

```bash
python src/make_fig1_convergence.py    # figures/fig5_convergence [Fig. 1]  (table3_nulls.json, results_large.json)
python src/make_fig1_fig4.py           # figures/fig1_pc_dB [Fig. 2], fig4_clustering [Fig. 4]  (results_yf.json)
python src/run_yf.py                   # recomputes results_yf.json, then draws the same two figures
python src/make_fig3_conventions.py    # figures/fig3_alpha_beta_corrected [Fig. 3]  (m5_control_t5.json)
python src/redo_fig2.py                # the single-convention version of the same figure
python src/make_fss_fig.py             # figures/fig9_fss [Fig. 5]  (results_fss.json)
python src/make_fig6_dblp.py           # figures/fig6_dblp [Fig. 7]  (dblp60_percolation.json)
python src/make_snap_figs.py           # figures/fig8_grid_and_size [Fig. 6], fig7_fractality [Fig. 8]
                                       #   (*_comm.json, *_scan.json, controls.json, dblp_sweep_canonical.json)
```

The plotting scripts only read `results/`; none recomputes anything.

---

## Which file holds which result

| result | file in `results/` | produced by |
|---|---|---|
| Table 3: intact-network fit, nulls and edge-centred row; Fig. 1 left | `table3_nulls.json` | `table3_nulls.py` |
| Intact exponents at `t = 4..6`; simulated `p_c(N)` (Fig. 1 right) | `results_large.json` | `run_large.py` |
| Susceptibility and `d_B(p)`, `G^A` and `G^B` (Figs 2, 4) | `results_yf.json` | `run_yf.py` |
| Direct joint fit, FSFN `t=4` hand grid | `fsfn4_hand.json` | `run_snap.py --fsfn 4 --l-b 2 3 4 6 9 13` |
| Direct joint fit, FSFN `t=4,5,6` commensurate | `fsfn4_comm`, `fsfn5_comm`, `fsfn6_comm` | `run_snap.py --fsfn N --l-b 2 4 10 28` |
| Grid-selection table, competing grids | `fsfn5_scan.json`, `fsfn6_scan.json` | `run_snap.py --scan` |
| Direct joint fit, WWW | `www_comm.json` | `run_snap.py --www … --l-b 2 4 6` |
| Direct joint fit, Internet AS | `as_comm.json` | `run_snap.py --as …` |
| Direct joint fit, DBLP `w ≥ 25`, `w ≥ 60` | `dblp25_comm.json`, `dblp60_comm.json` | `run_snap.py --dblp …` |
| Fractality test and controls | `controls.json` | `controls.py` |
| Percolation critical exponents (finite-size scaling) | `results_fss.json` | `fss.py --n-p 33 --jobs 2` |
| DBLP weak-tie sweep, grid (2,3,4,6,9) | `dblp_sweep_canonical.json` | `dblp_sweep2.py` |
| DBLP `w ≥ 60` percolation (Fig. 7) | `dblp60_percolation.json` | `dblp_percolation.py` |
| Grid rule with all four families; ΔAIC ranges over every size (controls, AS, DBLP `w ≥ 25`, FSFN `t = 4`) | `extended_covers.json` | `extended_covers.py` |
| Candidate grids, Fig. 6(a); WWW per-`l_B` counts, Fig. 6(b) | `as_scan`, `dblp25_scan`, `dblp60_scan`, `www_scan` (+ `www_l35`) | `run_snap.py --scan`, `combine_grids.py` |
| Four-algorithm covering comparison | `covering.json` | `run_covering.py` |
| Mass-convention control (three conventions) | `m5_control_t5.json` | `m5_control_t5.py` |
| Disconnected-box audit | `conn_audit.json` | `conn_audit.py` |

## Reading a result file

The interesting keys under `main`:

| key | meaning |
|---|---|
| `l_B`, `NB` | the box-size grid and the box counts |
| `dB`, `dB_R2` | box dimension and the fit quality |
| `fractality` | power-law vs exponential fit, including `dAIC_pl_minus_exp` |
| `direct` | the joint OLS of `ln m` on `(ln L, ln k)` — **uses neither γ nor δ** |
| `direct_connected` | the same fit restricted to internally connected boxes |
| `disconnected_by_lB` | per `l_B`: box count, disconnected count and fraction, and the same among boxes surviving the mass cut |
| `gamma`, `delta`, `delta_per_lB` | CCDF estimates; `delta` is the **median** of `delta_per_lB` |
| `n_delta_levels`, `delta_support_points` | how many levels and how many distinct CCDF points each δ rests on |
| `gamma_mle`, `delta_mle` | continuous MLE + bootstrap interval, and the discrete (zeta) MLE, for comparison |
| `alpha_macroscopic`, `beta_macroscopic` | derived from `d_B`, `γ`, `δ` — functions of δ alone once `d_B` and `γ` are fixed |
| `closure_direct_pct`, `closure_macroscopic_pct` | the latter is **0.0 by construction**; only the former is a test |

That last row is worth dwelling on. On the DBLP `w ≥ 60` backbone the macroscopic route
closes `d_B = α + β d_k` to machine zero while the direct route misses by 44 %. The
macroscopic closure is an identity, not evidence.

## If your numbers differ from ours

In rough order of likelihood:

1. **A different DBLP snapshot.** Check `data/get_data.py --check`. DBLP grows; a fixed
   weight threshold does not give a fixed backbone.
2. **A different seeding tie-break**, if you have modified `fast_cover.csr_from_graph`.
   See pitfall 1 in [`METHODS_NOTES.md`](METHODS_NOTES.md) — it moves `d_B` in the third
   decimal and `α` in the second.
3. **A different box-size grid.** Grid choice moves `d_B` by 13–16 % (full range) on these
   networks, comparable to the spread between covering algorithms (8–15 %).
4. **Library versions.** `requirements.txt` pins what we used. The greedy cover is
   order-sensitive, so a different `networkx` iteration order can shift box counts
   slightly even with the tie-break fixed.

Differences within these bounds are the subject of the paper, not a failure to
reproduce it. What should be stable to the digit is any FSFN run under the same
tie-break and library versions, and every exact model quantity.

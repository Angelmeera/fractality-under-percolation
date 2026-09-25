# Fractal scaling theory under percolation

Code, regenerated results and figures for a study of what happens to the two-variable
mass law of fractal complex networks when the network is progressively damaged.

The scaling theory of Fronczak *et al.* (2024) says the mass of a covering box depends
jointly on the box diameter and on the degree of its best-connected node,

<p align="center"><code>m(L, k) = B · L<sup>α</sup> k<sup>β</sup></code>, &nbsp; with &nbsp; <code>d<sub>B</sub> = α + β d<sub>k</sub></code></p>

and it is formulated for a *static* network. This work asks whether it survives bond
percolation, using hierarchical fractal scale-free networks (Yakubo & Fujiki 2022)
whose threshold and critical exponents are known in closed form — so every estimator
can be calibrated against an exact answer before being pointed at real data.

**The short version of the result is negative, and deliberately so.** The mass law
holds well on the intact model. Under damage the two exponents decouple, but *which*
one looks fragile turns out to be set by the convention for the mass of a fragmented
box rather than by the network. And the only route to α and β that does not presuppose
the scaling relation — a direct joint fit over greedy boxes — fails on a network where
the answer is known, for reasons this repository documents precisely.

If you take one thing from this repo, make it
**[`docs/METHODS_NOTES.md`](docs/METHODS_NOTES.md)**: four box-covering pitfalls, each
calibrated against a network with exact exponents, each of which moves `d_B` or `β` by
more than the effects usually being reported.

---

## Layout

```
src/         all analysis code (flat package; scripts import their siblings)
data/        get_data.py + instructions — no data files are committed
results/     the 28 JSONs behind the tables and figures of the paper
figures/     the figures (seven in the manuscript; fig5_convergence is an extra)
docs/        REPRODUCING.md (command → output → table) and METHODS_NOTES.md
```

## Quick start

```bash
pip install -r requirements.txt

# The model sections need no data at all:
python src/run_snap.py --fsfn 4 --l-b 2 4 10 28 --jobs 2 --out results/fsfn4_comm.json

# For the empirical sections:
python data/get_data.py            # two SNAP graphs; DBLP is built locally, see data/README.md
```

Run everything from the repository root. Scripts resolve their own imports, and the
figure/results directories default to `../figures` and `../results` relative to `src/`,
so those work from any working directory; paths you pass on the command line are
relative to where you are.

## What is in `src/`

**Model and exact quantities**

| file | what it does |
|---|---|
| `yakubo_fujiki.py` | builds the FSFNs by iterated edge replacement; exact `p_c`, `ν`, `ν̃`, `D_f`, hub-centred exponents |
| `fractal_dynamics.py` | bond percolation, giant component, susceptibility over finite clusters |
| `large_lib.py` | the sparse machinery that makes generation `t = 6` (`N = 149,798`) routine |

**Box covering** — four algorithms, because the choice matters and the paper quantifies
how much

| file | what it does |
|---|---|
| `fast_cover.py` | `csr_from_graph` (the canonical seeding order — read the note below) and the in-memory greedy cover |
| `snap_networks.py` | the main analysis module: streaming cover, box records, the joint mass-law fit, the grid scan, the exponent estimators |
| `covering.py` | CBB, MEMB and graph-colouring covers, and the comparison harness |
| `combine_grids.py` | scores every candidate box-size grid offline from per-`l_B` output |
| `extended_covers.py` | covers every `l_B` up to the pipeline's largest and applies the grid rule with all four candidate families (the controls, AS, DBLP `w ≥ 25`, FSFN `t = 4`) |

**Drivers** — each writes a JSON containing its own `argv`, so no number in the paper is
quoted without a record of how it was produced

| file | produces |
|---|---|
| `run_snap.py` | the main per-network analysis (FSFN, WWW, Internet AS, DBLP) |
| `run_yf.py` | the exact-model checks and `results_yf.json`, then draws `fig1_pc_dB` and `fig4_clustering` via `make_fig1_fig4.py` |
| `table3_nulls.py` | Table 3 in full: the intact-network fit, the shuffled and random-regressor nulls (1000 seeded draws) and the edge-centred estimator |
| `run_dblp.py` | builds the DBLP co-authorship graph from a dump (its `--percolate` option is the old estimator with a pooled-δ fallback; the paper's percolation run is `dblp_percolation.py`) |
| `dblp_percolation.py` | bond percolation of the DBLP `w ≥ 60` backbone on the Table 6 grid and cover: `χ(p)` and `d_B`, γ, δ, α, β of the largest component |
| `dblp_sweep2.py` | the weak-tie threshold sweep |
| `run_covering.py` | the four-algorithm comparison |
| `memb_fsfn.py` | MEMB-only covers of the FSFN, for Table 9's MEMB rows |
| `controls.py` | the non-fractal controls (Barabási–Albert `m = 1, 2`, Erdős–Rényi) |
| `fss.py` | finite-size scaling for the percolation critical exponents |
| `run_large.py` | the large-generation runs, including the simulated `p_c(N)` of Sec. 4.2 |
| `real_network.py` | DBLP parsing, weighting and backbone extraction |

**Audits** — checks that any covering can be run against

| file | what it checks |
|---|---|
| `conn_audit.py` | how many greedy boxes are internally *disconnected*, and what the mass-law fit looks like without them |
| `m5_control_t5.py` | the same percolation run under three different definitions of "the mass of a fragmented box" |
| `subgrid_daic.py` | Δ AIC over *every* ≥3-point subgrid, which is what shows a single Δ AIC is not a verdict |
| `tiebreak_spread.py` | how much the intact FSFN `d_B` depends on the seeding tie-break, over random orders within degree classes |

**Figures**: `make_fig1_convergence.py`, `make_fig1_fig4.py`, `redo_fig2.py`,
`make_fig3_conventions.py`, `make_fig6_dblp.py`, `make_snap_figs.py`, `make_fss_fig.py`,
all sharing the print style in `paper_style.py`; `run_yf.py` also draws Figs 1 and 3.
Every figure in `figures/` is drawn by one of these from a file in `results/`, as a
600-dpi PNG and a vector EPS at the printed width of 5.0 in.

## Results

`results/` holds the 28 JSONs behind the tables and figures. Each carries its own `argv`, the
box-size grid, `N_B(l_B)`, the fits and the diagnostics.
[`docs/REPRODUCING.md`](docs/REPRODUCING.md) maps each table and figure to the file and
the command that made it.

Every covering-derived number was regenerated under a single seeding order for this
version. That mattered more than it sounds — see below.

## The thing that will bite you

The greedy box cover seeds boxes in decreasing-degree order, and **degree ties have to
be broken by a rule you state**. We break them on `str(node)`. On the FSFN, whose nodes
are integers, that is lexicographic ordering (`"10"` before `"2"`), and switching to
numeric ordering — a choice with no principled claim on it either way — moves things by
more than several published disagreements about `d_B`:

| tie-break | FSFN `t=4`, grid (2,4,10,28) | `d_B` | direct `α` |
|---|---|---|---|
| `str(node)` (this repo) | `N_B = (1303, 383, 71, 10)` | 1.847 | 1.529 |
| numeric | `N_B = (1312, 390, 69, 10)` | 1.854 | 1.607 |
| insertion order, no tie-break | `N_B = (1252, 277, 38, 6)` | — | — |

Under the canonical order the three independent cover implementations in this repo
(`fast_cover`, `snap_networks`, `covering`) agree exactly, across process counts and
across `PYTHONHASHSEED`, and `run_yf.py` (Figs 1 and 3) uses the same order. On the
clustered generator `G^A` the tie-break matters far more than on `G^B`: its intact `d_B`
is 1.612 under `str(node)` and `1.70 ± 0.07` over random orders within degree classes
(`results/tiebreak_spread.json`). Earlier output of ours that predates the fixed order
is superseded and is not included here.

`docs/METHODS_NOTES.md` covers this and three more: about half of all greedy boxes are
internally disconnected; the box-mass cut silently removes the smallest box size from
the fit; and the maximum-likelihood exponent estimator usually recommended is, on this
network, considerably *worse* than the CCDF fit it is meant to replace.

## Data

Nothing is redistributed here. `python data/get_data.py` downloads the two SNAP graphs
and verifies them against recorded SHA-256 checksums; the DBLP co-authorship network is
a derived product that you build locally from a DBLP or AMiner dump, and
`data/README.md` gives the commands. DBLP has roughly doubled since the snapshot used
by Fronczak *et al.*, so a fixed weight threshold does **not** reproduce their backbone
size on a current dump — which is why the paper sweeps the threshold rather than fixing
it. The FSFN sections need no data at all.

## Licence and citing

Released under the [MIT licence](LICENSE) — code, results, figures and documentation.
The datasets are not covered and are not redistributed here; see
[`data/README.md`](data/README.md) for their sources and terms.

The manuscript is not yet published. If you use this code or these results, please cite
the paper once it appears; in the meantime a link to this repository is fine. Questions
and corrections are welcome through the repository's issue tracker.

## Key references

- A. Fronczak, P. Fronczak, M. J. Samsel, K. Makulski, M. Łepek and M. J. Mrowinski,
  *Scientific Reports* **14**, 9079 (2024) — the scaling theory under test.
- K. Yakubo and Y. Fujiki, *PLoS ONE* **17**, e0264589 (2022) — the FSFN
  construction and its exact percolation quantities.
- C. Song, L. K. Gallos, S. Havlin and H. A. Makse, *J. Stat. Mech.* (2007) P03006 —
  the box-covering algorithms.
- S. Furuya and K. Yakubo, *Phys. Rev. E* **84**, 036118 (2011) — the multifractal
  relation that is the `β = 1` case of the mass law, thirteen years earlier.
- A. Clauset, C. R. Shalizi and M. E. J. Newman, *SIAM Review* **51**, 661 (2009) —
  power-law estimation, and the autocorrelation warning about CCDF fits.

## Contact

Angel Meera

Supported by the INSPIRE programme, Department of Science and Technology, Government of
India.

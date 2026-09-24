# Four box-covering pitfalls, calibrated

Everything here is measured on a network whose exponents are known exactly — the
hierarchical FSFN with `d_B = ln 8 / ln 3 = 1.8928`, `d_k = ln 2 / ln 3 = 0.6309`,
`α = ln 4 / ln 3 = 1.2619`, `β = 1`, `γ = δ = 4` — so each effect below is an error, not
a disagreement. Each one is larger than the effects box-counting studies typically
report.

Reproduce them with `src/conn_audit.py`, `src/subgrid_daic.py` and
`src/m5_control_t5.py`.

---

## 1. The seeding tie-break is part of the measurement

The greedy cover visits nodes in decreasing degree order and lets each join the oldest
box that can still accept it. Degrees tie constantly on a hierarchical network, so the
tie-break rule determines the cover.

| tie-break | `N_B` at `l_B = 2,4,10,28` (`t=4`) | `d_B` | direct `α` | direct `β` |
|---|---|---|---|---|
| `str(node)` — this repo | (1303, 383, 71, 10) | 1.847 | 1.529 | −0.557 |
| numeric | (1312, 390, 69, 10) | 1.854 | 1.607 | −0.563 |
| insertion order, none | (1252, 277, 38, 6) | — | — | — |

Both of the first two are defensible; neither is "right". `str(node)` is what this repo
uses, chosen because the string form is well defined for author-name labels as well as
integer ones, and because ties broken on a language's iteration order are not
reproducible at all — we once saw the same script return `N_B(l_B=2) = 24,935` and
`24,962` on the same graph in two processes, because Python's `set` ordering depends on
`PYTHONHASHSEED`.

Under the fixed order the three cover implementations here (`fast_cover.cover_one`,
`snap_networks.cover_stream`, `covering.cover_greedy_degree`) agree exactly, across
worker counts and across hash seeds. The dense-matrix cover in
`fractal_dynamics.greedy_box_cover_D`, used by `run_yf.py` for Figs 2 and 4 only, does
not apply the `str(node)` tie-break and gives `d_B = 1.849` against 1.847 on the `t = 4`
FSFN.

**Report the tie-break rule with the algorithm.** It moves `d_B` by about as much as
some of the published disagreements it might be invoked to explain.

## 2. About half of all greedy boxes are internally disconnected

A greedy box is a set of nodes pairwise within `l_B − 1` **in the whole graph**. Nothing
requires the induced subgraph to be connected, and on this network it usually is not:

| `l_B` | `t=4` | `t=5` | `t=6` |
|---|---|---|---|
| 2 | 0.0 % | 0.0 % | 0.0 % |
| 4 | 57.2 % | 57.8 % | 57.6 % |
| 10 | 71.8 % | 70.7 % | 70.1 % |
| 28 | 60.0 % | 74.4 % | 73.1 % |

Pooled over the boxes that actually enter the regression at `t = 5`: **53.8 %**.

This matters twice. "The diameter of the box" is undefined for such a box, and what the
pipeline computes is the largest distance between a *reachable* pair, which is a
different quantity. And a disconnected box draws its mass from parts of the network its
diameter does not span — precisely the point that breaks a mass law.

Refitting on the same covers with disconnected boxes removed:

| `t` | all boxes (`α` / `β` / `R²` / `n`) | connected only |
|---|---|---|
| 4 | 1.529 / −0.557 / 0.713 / 410 | 1.746 / **−0.121** / 0.967 / 188 |
| 5 | 1.539 / −0.527 / 0.691 / 3229 | 1.750 / **−0.113** / 0.955 / 1491 |
| 6 | 1.546 / −0.511 / 0.683 / 25773 | 1.756 / **−0.117** / 0.953 / 11990 |

So `β ≈ −0.5` is not a property of greedy covering. It is an artefact of fitting a
two-variable mass law to objects with no single diameter. The residual failure is real
and is concentrated in `α ≈ 1.75` against an exact 1.2619 — but the *sign* of `β` is not
a finding.

A related collapse: with all boxes, swapping the measured box diameter for the nominal
`l_B` moves `β` from −0.53 to **+0.13**, a sign change. With disconnected boxes removed
the same swap gives −0.113 against −0.092. One cause, not two.

## 3. The mass cut silently removes the smallest box size

`joint_mass_law` drops boxes of mass below 3. At `l_B = 2` on the `t = 5` FSFN, **all
10,487 boxes** have mass below 3 (8,239 of mass exactly 2, 2,248 of mass 1), so the
entire `l_B = 2` column disappears from the fit. Same at `t = 4`:
`N_B = (1312, 390, 69, 10)` but only 407 boxes are fitted, which is `390 + 69 + 10`
after the cut.

A grid advertised as `{2, 4, 10, 28}` therefore contributes **three** levels to the
direct route, not four. `l_B = 2` also yields no `δ` estimate — fewer than five distinct
CCDF support points.

At `l_B = 4`, `t = 5` the cut removes exactly **504 boxes, every one of mass exactly 2
and every one disconnected**: non-adjacent pairs, not fragments. That single shape is
the whole of the difference between "57.8 % of boxes are disconnected" and "49.7 % of
fitted boxes are disconnected".

## 4. The recommended MLE is worse than the CCDF fit here

Clauset, Shalizi and Newman recommend maximum likelihood over fitting a line to the
empirical CCDF, and warn specifically that CCDF points are not independent — "adjacent
values of the CDF are strongly correlated", so the least-squares standard error "is
typically a gross underestimate because of the failure to account for the
correlations". That warning is right and the standard error should not be quoted.

The remedy does not transfer. With `γ = 4` exactly:

| estimator | `t = 4` | `t = 5` | `t = 6` | behaviour |
|---|---|---|---|---|
| CCDF least squares | 4.219 (+5.5 %) | 4.164 (+4.1 %) | 4.126 (+3.1 %) | converges |
| continuous MLE | 2.980 | 2.982 | 2.982 | −25 %, **does not improve** |
| discrete (zeta) MLE | 2.362 | 2.363 | 2.363 | −41 %, **does not improve** |

The reason is that this degree distribution is not a zeta law but a hierarchy with
atoms at `κⁿ` — five distinct degrees above `k = 2` at `t = 4`, seven at `t = 6`. The
likelihood form misdescribes it; the CCDF slope across log-spaced atoms does not.

Practical conclusion: use the CCDF estimate, and quote the **disagreement between
estimators** — here approaching a factor of two — as the honest uncertainty, rather
than any single estimator's standard error. Report how many distinct support points
each slope rests on; on the shortest grids in this study it is as few as one.

## 5. (Bonus) A single Δ AIC is not a verdict

Evaluated over every subgrid of three or more of the box sizes covered, **every**
network here admits both signs:

| network | Δ AIC on the selected grid | range over all ≥3-point subgrids |
|---|---|---|
| FSFN `t=6` (fractal by construction) | −22.3 | −40.0 … **+35.7** |
| DBLP `w ≥ 60` | −19.5 | −39.0 … +17.0 |
| DBLP `w ≥ 25` | −14.0 | −27.3 … +15.8 |
| WWW | −28.2 | −28.2 … +15.5 |
| Internet AS | −5.4 | −23.4 … +18.6 |
| Barabási–Albert tree | −7.5 | −19.4 … +16.4 |
| Barabási–Albert `m = 2` | +0.4 | **−15.4** … +4.4 |
| Erdős–Rényi | +7.7 | **−10.9** … +30.1 |

The selected grid is not a neutral draw either: on the WWW the `R²` rule picks
`{2, 4, 6}`, which is the most power-law-favouring of the sixteen available subgrids.

What survives is a statement *conditional on the grid rule*, quoted alongside the range.
The discriminator with real teeth in this study is the **sign** of the renormalisation
exponent `d_k` — negative is impossible for a self-similar network and cannot be
produced by a fitting artefact.

---

## Two cheap checks worth adopting

**Admissibility.** Since mass grows with diameter at fixed degree, and
`d_B = α + β d_k` with `β, d_k ≥ 0` for a self-similar network, any estimate must
satisfy `0 ≤ α ≤ d_B`. It costs nothing and it flags exactly the fits that should not be
trusted. In this study the violations are the two Barabási–Albert controls and a WWW fit
taken at a single box size — and that last one is instructive: a fit at one `l_B` has
`ln L` varying only through within-level diameter scatter, and returns `α = −0.104`. On
the proper three-point grid the same graph gives `α = 1.095`, which is admissible.

**A null estimator.** Replace the degree regressor with random numbers. On this network
that still closes `d_B = α + β d_k` to 0.42 % at `t = 6`, against 0.36 % for the correct
estimator, and still improves monotonically with system size. So the closure residual is
a consistency check on the tiling, not evidence for the theory — and a null estimator
costs nothing to report next to it.

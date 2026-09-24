"""Bond percolation of the DBLP w>=60 backbone (paper Fig. 7(b,c) and the
bullet list after Table 7), on the same pipeline as the Table 7 sweep.

This replaces the `percolation_w60` block of the old results_dblp_final.json,
which was produced by `run_dblp.py --percolate` through
real_network.percolate_real(): a different cover (fronczak_protocol), a
different grid (l_B up to 13) and three realizations per p. Its intact d_B
(1.824) therefore did not match Table 7 (1.952) for the same graph.

Here every quantity comes from snap_networks.analyse() -- the canonical
str(node)-tie-broken streaming cover used for Table 7 -- on the grid
l_B in {2,3,4,6,9}, so the p = 1 row reproduces the w>=60 row of Table 7 exactly.

For each p in 0.35, 0.40, ..., 1.00 and each of N_REAL realizations the
backbone is bond-percolated (each edge kept with probability p), the largest
component is kept if it has at least MIN_GIANT nodes, and d_B, gamma, delta and
the macroscopic alpha = (delta-2)/(delta-1) d_B, beta = (gamma-1)/(delta-1)
are measured on it. delta is the median of the per-l_B CCDF slopes, and a
level contributes one only if at least 25 boxes have mu >= 1 (Sec. 2 of the
paper); when no level qualifies, delta, alpha and beta are left undefined
(NaN) for that realization and the count of finite values is stored as
<key>_n. There is deliberately no fallback: the old fronczak_protocol()
fell back to one CCDF slope over the pooled masses of all levels, a different
estimator that returns delta ~ 2.1-2.2 at every p, and the switch to it on
small giants is what produced the apparent fall of delta in the old run. The finite-cluster susceptibility chi(p) is computed on
the grid p = 0.30, 0.31, ..., 0.99, 1.00 (standard errors stored).

Usage (from the repository root, after data/get_data.py):
    python src/dblp_percolation.py
Environment overrides: PERC_EDGELIST, PERC_OUT, PERC_NREAL, PERC_NCHI, PERC_SEED.
Writes results/dblp60_percolation.json (about three minutes on one core).
"""
import json
import os
import sys

import numpy as np
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from real_network import load_weighted_edgelist, backbone      # noqa: E402
from snap_networks import giant, analyse                        # noqa: E402

EDGELIST = os.environ.get("PERC_EDGELIST", "data/dblp_weighted.tsv.gz")
OUT = os.environ.get("PERC_OUT", "results/dblp60_percolation.json")
N_REAL = int(os.environ.get("PERC_NREAL", "20"))
N_CHI = int(os.environ.get("PERC_NCHI", "1000"))
SEED = int(os.environ.get("PERC_SEED", "20260923"))
THRESHOLD = 60
GRID = (2, 3, 4, 6, 9)
MIN_GIANT = 60
PS = [round(x, 2) for x in np.arange(0.35, 1.0001, 0.05)]
PS_CHI = [round(x, 2) for x in np.arange(0.30, 0.9901, 0.01)] + [1.0]
KEYS = ("dB", "dB_R2", "gamma", "delta", "alpha_macroscopic",
        "beta_macroscopic", "N")


def percolate(B, edges, p, rng):
    H = nx.Graph()
    H.add_nodes_from(B.nodes())
    H.add_edges_from(e for e, r in zip(edges, rng.random(len(edges))) if r < p)
    return H


def chi_finite(H):
    sizes = sorted((len(c) for c in nx.connected_components(H)),
                   reverse=True)[1:]
    if not sizes:
        return 0.0
    a = np.asarray(sizes, float)
    return float((a ** 2).sum() / a.sum())


def main():
    Gw = load_weighted_edgelist(EDGELIST)
    B = giant(backbone(Gw, THRESHOLD))
    # Canonical edge list: each edge oriented and the list sorted on str(node),
    # so the i-th random number always goes to the same edge. B.edges() alone
    # depends on set iteration order, i.e. on PYTHONHASHSEED.
    edges = sorted((tuple(sorted(e, key=str)) for e in B.edges()),
                   key=lambda e: (str(e[0]), str(e[1])))
    print(f"w>={THRESHOLD} backbone: N={B.number_of_nodes()} "
          f"M={B.number_of_edges()}", flush=True)
    rng = np.random.default_rng(SEED)

    chi, chi_se = [], []
    for p in PS_CHI:
        v = np.array([chi_finite(percolate(B, edges, p, rng))
                      for _ in range(N_CHI)])
        chi.append(float(v.mean()))
        chi_se.append(float(v.std(ddof=1) / np.sqrt(v.size)))
    ipk = int(np.argmax(chi))
    print(f"chi peak at p={PS_CHI[ipk]:.2f} ({chi[ipk]:.2f} +- {chi_se[ipk]:.2f})")

    rows = []
    for p in PS:
        reals = []
        n_try = 1 if p == 1.0 else N_REAL
        for _ in range(n_try):
            H = B if p == 1.0 else giant(percolate(B, edges, p, rng))
            if H.number_of_nodes() < MIN_GIANT:
                continue
            r = analyse(H, f"p={p}", GRID, n_jobs=1, verbose=False)
            reals.append({k: float(r[k]) for k in KEYS})
            reals[-1]["n_delta_levels"] = int(r["n_delta_levels"])
        row = {"p": p, "n_used": len(reals), "realizations": reals}
        for k in KEYS + ("n_delta_levels",):
            v = np.array([x[k] for x in reals], float)
            v = v[np.isfinite(v)]
            row[k + "_n"] = int(v.size)
            row[k] = float(v.mean()) if v.size else float("nan")
            row[k + "_sd"] = float(v.std()) if v.size > 1 else 0.0
        rows.append(row)
        print(f"p={p:.2f} n={len(reals):2d} N_gc={row['N']:.0f} "
              f"dB={row['dB']:.3f}+-{row['dB_sd']:.3f} "
              f"gamma={row['gamma']:.2f} delta={row['delta']:.2f} "
              f"alpha={row['alpha_macroscopic']:.3f} "
              f"beta={row['beta_macroscopic']:.3f}+-{row['beta_macroscopic_sd']:.3f}",
              flush=True)

    out = {"argv": sys.argv, "threshold": THRESHOLD, "grid": list(GRID),
           "seed": SEED, "n_realizations": N_REAL, "n_chi": N_CHI,
           "min_giant": MIN_GIANT,
           "note": "Paper Fig. 7(b,c) and the DBLP percolation paragraph. "
                   "p = 1 is the intact backbone (one cover, = Table 7 row w>=60).",
           "ps_chi": PS_CHI, "chi": chi, "chi_se": chi_se,
           "chi_peak_p": PS_CHI[ipk], "rows": rows}
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

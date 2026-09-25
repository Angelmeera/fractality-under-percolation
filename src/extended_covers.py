"""Covers at EVERY box size up to the pipeline's largest, and the grid rule
applied with all four candidate families (paper Sec. 2.3, Sec. 4.9, Table 8).

The grid scans behind Tables 4, 8 and 9 (run_snap.py --scan, controls.py)
cover only the union of the geometric candidate grids {lambda^tau + 1}. The
arithmetic grids quoted in the text need further sizes (l_B = 6, 7, 10, ...),
so this script covers every l_B from 2 to l_max and scores, on those counts:

  geometric      {lambda^tau + 1} over the same lambdas as the original scan,
                 at least 4 points (3 on the Internet AS graph, as in as_scan)
  arithmetic     constant step 2 or 3: every residue class over the covered
                 sizes, at least 3 points (the construction of combine_grids.py)
  consecutive    all covered sizes, and all but l_B = 2

and it records the dAIC range over every subgrid of >= 3 covered sizes (the
Table 8 definition, here over the full size set; skipped for the FSFN, whose
27 sizes give 2^27 subgrids).

Networks and l_max:
  BA m=2, BA m=1 tree, ER   as controls.py (same seeds), l_max = max(6, diam//2 + 1)
  Internet AS               l_B = 2..9   (diameter 9)
  DBLP w>=25                l_B = 2..12  (the l_max of dblp25_scan.json)
  FSFN G^B t=4              l_B = 2..28  (the l_max of the FSFN scans), no range

Usage (from the repository root, after data/get_data.py):
    python src/extended_covers.py            # ~10 min on two cores
Writes results/extended_covers.json. EXT_OUT overrides the output path and
EXT_ONLY (comma-separated keys: bam2,tree,er,as,dblp25,fsfn4) runs a subset.
"""
import itertools
import json
import os
import sys

import numpy as np
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from snap_networks import (load_snap_edgelist, giant, cover_all,       # noqa: E402
                           powerlaw_vs_exponential, geometric_grid,
                           _approx_diameter)

OUT = os.environ.get("EXT_OUT", "results/extended_covers.json")
ONLY = [k for k in os.environ.get("EXT_ONLY", "").split(",") if k]
JOBS = int(os.environ.get("EXT_JOBS", "2"))
MAX_RANGE_SIZES = 14          # 2^14 subgrids at most for the dAIC range


def _controls(label):
    if label == "bam2":
        return "BA m=2 N=6474", giant(nx.barabasi_albert_graph(6474, 2, seed=11))
    if label == "tree":
        return "BA m=1 N=6474 (tree)", giant(nx.barabasi_albert_graph(6474, 1, seed=12))
    return "ER <k>=4 N=6474", giant(nx.gnm_random_graph(6474, 12948, seed=13))


def build(key):
    """(name, graph, l_max, lambdas, geometric min points, want range)"""
    if key in ("bam2", "tree", "er"):
        name, G = _controls(key)
        d = _approx_diameter(G)
        return name, G, max(6, d // 2 + 1), (1.6, 2.0, 2.5, 3.0), 4, True
    if key == "as":
        G = giant(load_snap_edgelist("data/as20000102.txt.gz", verbose=False))
        return "Internet-AS", G, 9, (1.4, 1.6, 2.0, 2.5), 3, True
    if key == "dblp25":
        from real_network import load_weighted_edgelist, backbone
        G = giant(backbone(load_weighted_edgelist("data/dblp_weighted.tsv.gz"), 25))
        return "DBLP(w>=25)", G, 12, (1.6, 2.0), 4, True
    if key == "fsfn4":
        from yakubo_fujiki import generate_fsfn
        return "FSFN-GB-t4", generate_fsfn("GB", 4), 28, (1.6, 2.0, 2.5, 3.0), 4, False
    raise KeyError(key)


def score(NB, N, grid, family, label):
    g = [l for l in grid if NB.get(l, 0) > 1]
    if len(g) < 3:
        return None
    f = powerlaw_vs_exponential(np.array(g, float),
                                np.array([NB[l] for l in g], float) / N)
    if "R2_powerlaw" not in f:
        return None
    return dict(family=family, label=label, grid=g, dB=f["exponent"],
                dB_R2=f["R2_powerlaw"], dAIC=f["dAIC_pl_minus_exp"])


def candidates(NB, N, diam, l_max, lambdas, geo_min):
    have = sorted(l for l in NB if NB[l] > 1)
    out = []
    for lam in lambdas:
        g = [l for l in geometric_grid(lam, diam, l_max=l_max) if l in NB]
        if len(g) >= geo_min:
            out.append(score(NB, N, g, "geometric", f"lambda={lam}"))
    for step in (2, 3):
        for r in range(step):
            g = [l for l in have if (l - have[0]) % step == r]
            if len(g) >= 3:
                out.append(score(NB, N, g, "arithmetic", f"step {step} from l_B={g[0]}"))
    out.append(score(NB, N, have, "consecutive", "all"))
    out.append(score(NB, N, [l for l in have if l != 2], "consecutive", "all except 2"))
    return [c for c in out if c]


def daic_range(NB, N):
    ls = sorted(l for l in NB if NB[l] > 1)
    vals = []
    for k in range(3, len(ls) + 1):
        for sub in itertools.combinations(ls, k):
            a = powerlaw_vs_exponential(np.array(sub, float),
                                        np.array([NB[l] for l in sub], float) / N
                                        ).get("dAIC_pl_minus_exp")
            if a is not None and np.isfinite(a):
                vals.append((a, sub))
    vals.sort()
    return dict(n_subgrids=len(vals), dAIC_min=vals[0][0], grid_min=list(vals[0][1]),
                dAIC_max=vals[-1][0], grid_max=list(vals[-1][1]))


def main():
    keys = ONLY or ["bam2", "tree", "er", "as", "dblp25", "fsfn4"]
    res = {"argv": sys.argv, "note": "Every l_B from 2 to l_max covered with the "
           "canonical cover; grid rule scored over all four families.", "networks": {}}
    for key in keys:
        name, G, l_max, lambdas, geo_min, want_range = build(key)
        N = G.number_of_nodes()
        diam = _approx_diameter(G)
        ls = list(range(2, l_max + 1))
        print(f"== {name}: N={N} diam~{diam} covering l_B=2..{l_max}", flush=True)
        per, _, _ = cover_all(G, ls, n_jobs=JOBS, verbose=False)
        NB = {l: int(per[l][0]) for l in ls}
        cands = candidates(NB, N, diam, l_max, lambdas, geo_min)
        best = max(cands, key=lambda c: c["dB_R2"])
        best_geo = max((c for c in cands if c["family"] == "geometric"),
                       key=lambda c: c["dB_R2"])
        rec = dict(key=key, N=N, diameter_lb=diam, l_max=l_max,
                   l_B=ls, NB=[NB[l] for l in ls], lambdas=list(lambdas),
                   geometric_min_points=geo_min, candidates=cands,
                   best=best, best_geometric=best_geo)
        if want_range and len([l for l in ls if NB[l] > 1]) <= MAX_RANGE_SIZES:
            rec["dAIC_range_all_sizes"] = daic_range(NB, N)
        res["networks"][name] = rec
        print(f"   best overall  {best['grid']} ({best['family']}) R2={best['dB_R2']:.5f} "
              f"d_B={best['dB']:.3f} dAIC={best['dAIC']:+.1f}")
        print(f"   best geometric {best_geo['grid']} R2={best_geo['dB_R2']:.5f} "
              f"d_B={best_geo['dB']:.3f} dAIC={best_geo['dAIC']:+.1f}")
        if "dAIC_range_all_sizes" in rec:
            r = rec["dAIC_range_all_sizes"]
            print(f"   dAIC range {r['dAIC_min']:+.1f} {r['grid_min']} .. "
                  f"{r['dAIC_max']:+.1f} {r['grid_max']} over {r['n_subgrids']} subgrids")
        os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
        with open(OUT, "w") as fh:
            json.dump(res, fh, indent=1, default=float)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

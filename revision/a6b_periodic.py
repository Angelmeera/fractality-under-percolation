"""A6b: same calibration on PERIODIC lattices (no boundary), with both the paper's
greedy-degree cover and MEMB; selection rule applied to each."""
import sys, json, time
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, _os.path.join(HERE, "..", "src"))
import numpy as np, networkx as nx, lib
from fast_cover import csr_from_graph, cover_one
from snap_networks import geometric_grid, powerlaw_vs_exponential

def select(NB, n, have, lmax):
    cands = {}
    for lam in (1.3, 1.4, 1.5, 1.7, 2.0, 2.5, 3.0):
        g = tuple(l for l in geometric_grid(lam, 10 ** 6, l_max=lmax) if l in NB)
        if len(g) >= 4: cands[f"geo {lam}"] = g
    for step in (2, 3):
        for start in range(step):
            g = tuple(l for l in have if (l - have[0]) % step == start % step)
            if len(g) >= 3: cands[f"step{step}/{g[0]}"] = g
    cands["consecutive"] = tuple(have); cands["consecutive-no2"] = tuple(l for l in have if l != have[0])
    rows = []
    for lab, g in cands.items():
        f = powerlaw_vs_exponential(np.array(g, float), np.array([NB[l] for l in g], float) / n)
        rows.append(dict(label=lab, grid=list(g), dB=f["exponent"], R2=f["R2_powerlaw"]))
    best = max(rows, key=lambda r: r["R2"])
    allv = [r["dB"] for r in rows]
    return best, rows, (float(np.median(allv)), float(min(allv)), float(max(allv)))

res = []
for name, dims, truth, lmax, rmax in (("square torus 200x200", (200, 200), 2.0, 22, 10), ("cubic torus 34^3", (34, 34, 34), 3.0, 13, 6)):
    G = nx.convert_node_labels_to_integers(nx.grid_graph(dim=list(dims), periodic=True), ordering="sorted")
    nodes, idx, A = csr_from_graph(G); n = A.shape[0]; t0 = time.time()
    have = list(range(2, lmax + 1)); NBg = {l: len(cover_one(A, l)) for l in have}
    NBm = {2 * r + 1: len(lib.memb_cover(A, r)[0]) for r in range(1, rmax + 1)}
    bg, rg, sg = select(NBg, n, have, lmax)
    bm, rm, sm = select(NBm, n, sorted(NBm), 2 * rmax + 1)
    print(f"{name}: truth {truth} | greedy: max-R2 pick {bg['label']} {bg['grid']} dB={bg['dB']:.3f} R2={bg['R2']:.5f}; candidates median {sg[0]:.3f} range {sg[1]:.3f}-{sg[2]:.3f}"
          f" | MEMB: pick {bm['label']} {bm['grid']} dB={bm['dB']:.3f} R2={bm['R2']:.5f}; candidates median {sm[0]:.3f} range {sm[1]:.3f}-{sm[2]:.3f} [{time.time()-t0:.0f}s]", flush=True)
    loc = [round(float(-np.log(NBg[l + 1] / NBg[l]) / np.log((l + 1) / l)), 2) for l in have[:-1]]
    locm = [round(float(-np.log(NBm[l + 2] / NBm[l]) / np.log((l + 2) / l)), 2) for l in sorted(NBm)[:-1]]
    print("   greedy local slopes:", loc, "\n   MEMB local slopes:", locm, flush=True)
    res.append(dict(name=name, truth=truth, NB_greedy=NBg, NB_memb=NBm, greedy=dict(best=bg, rows=rg, spread=sg), memb=dict(best=bm, rows=rm, spread=sm)))
json.dump(res, open(_os.path.join(HERE, "results", "a6b_periodic.json"), "w"), indent=1, default=str)

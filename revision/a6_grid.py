"""A6: calibrate the max-R^2 grid-selection rule on graphs with known d_B and no
built-in scale factor: 2D square lattice (d_B = 2), 3D cubic lattice (d_B = 3) and
the largest cluster of 2D critical bond percolation (chemical-distance box dimension
d_f/d_min = (91/48)/1.1306 = 1.677). Greedy-degree covering at every l_B, then the
paper's candidate set (combine_grids.py): geometric {round(lam^tau)+1} with >=4
points, constant step 2 and 3, consecutive, consecutive without l_B=2."""
import sys, json, time
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, _os.path.join(HERE, "..", "src"))
import numpy as np, networkx as nx
from fast_cover import csr_from_graph, cover_one
from snap_networks import geometric_grid, powerlaw_vs_exponential
from scipy.sparse.csgraph import connected_components

def lattice(dims):
    G = nx.grid_graph(dim=list(dims))
    return nx.convert_node_labels_to_integers(G, ordering="sorted")

def perc_cluster(L, seed):
    rng = np.random.default_rng(seed)
    G = nx.grid_2d_graph(L, L)
    H = nx.Graph(); H.add_nodes_from(G.nodes())
    H.add_edges_from([e for e in G.edges() if rng.random() < 0.5])
    c = max(nx.connected_components(H), key=len)
    return nx.convert_node_labels_to_integers(H.subgraph(c).copy(), ordering="sorted")

def scan(G, lmax, name, truth):
    nodes, idx, A = csr_from_graph(G); n = A.shape[0]
    have = list(range(2, lmax + 1)); NB = {}
    t0 = time.time()
    for l in have:
        NB[l] = len(cover_one(A, l))
    cands = {}
    for lam in (1.3, 1.4, 1.5, 1.7, 2.0, 2.5, 3.0):
        g = tuple(l for l in geometric_grid(lam, 10 ** 6, l_max=lmax) if l in NB)
        if len(g) >= 4: cands[f"geo {lam}"] = g
    for step in (2, 3):
        for start in range(step):
            g = tuple(l for l in have if (l - have[0]) % step == start % step)
            if len(g) >= 3: cands[f"step{step}/{g[0]}"] = g
    cands["consecutive"] = tuple(have)
    cands["consecutive-no2"] = tuple(l for l in have if l != 2)
    rows = []
    for lab, g in cands.items():
        f = powerlaw_vs_exponential(np.array(g, float), np.array([NB[l] for l in g], float) / n)
        rows.append(dict(label=lab, grid=list(g), dB=f["exponent"], R2=f["R2_powerlaw"], dAIC=f["dAIC_pl_minus_exp"]))
    best = max(rows, key=lambda r: r["R2"])
    # all subgrids of >=4 consecutive-or-not sizes would be huge; use all contiguous windows of >=4
    win = []
    for i in range(len(have)):
        for j in range(i + 4, len(have) + 1):
            g = have[i:j]
            f = powerlaw_vs_exponential(np.array(g, float), np.array([NB[l] for l in g], float) / n)
            win.append(f["exponent"])
    out = dict(name=name, n=n, truth=truth, NB={str(k): v for k, v in NB.items()}, cands=rows, best=best,
               windows=dict(median=float(np.median(win)), lo=float(np.min(win)), hi=float(np.max(win)), n=len(win)),
               even=[r for r in rows if r["label"] == "step2/2"][0], secs=time.time() - t0)
    print(f"{name}: n={n} truth={truth:.3f} | max-R2 pick {best['label']} {best['grid']} dB={best['dB']:.3f} R2={best['R2']:.5f} | "
          f"even ladder dB={out['even']['dB']:.3f} | contiguous windows>=4: median {out['windows']['median']:.3f} "
          f"range {out['windows']['lo']:.3f}-{out['windows']['hi']:.3f} [{out['secs']:.0f}s]", flush=True)
    for r in sorted(rows, key=lambda r: -r["R2"]):
        print(f"   {r['label']:>16} {str(r['grid']):>40} dB={r['dB']:.3f} R2={r['R2']:.5f}")
    return out

if __name__ == "__main__":
    res = []
    Gsq = lattice((200, 200))
    res.append(scan(Gsq, 22, "square 200x200", 2.0))
    # A7: parity. Diameter-rule local slopes alternate; MEMB (radius rule) local slopes do not.
    import lib
    from fast_cover import csr_from_graph
    _, _, Asq = csr_from_graph(Gsq)
    nb_memb = {r: len(lib.memb_cover(Asq, r)[0]) for r in range(1, 11)}
    NBd = res[0]["NB"]
    loc_d = [float(-np.log(NBd[str(l + 1)] / NBd[str(l)]) / np.log((l + 1) / l)) for l in range(2, 21)]
    loc_m = [float(-np.log(nb_memb[r + 1] / nb_memb[r]) / np.log((2 * r + 3) / (2 * r + 1))) for r in range(1, 10)]
    print("A7 square lattice: diameter-rule local slopes l=2..21:", [round(x, 2) for x in loc_d], flush=True)
    print("A7 square lattice: MEMB local slopes (l=2r+1, r=1..10):", [round(x, 2) for x in loc_m], flush=True)
    res.append(dict(name="A7 parity square", memb_NB=nb_memb, local_diameter_rule=loc_d, local_memb=loc_m))
    res.append(scan(lattice((34, 34, 34)), 13, "cubic 34^3", 3.0))
    for s in (1, 2, 3):
        res.append(scan(perc_cluster(500, s), 22, f"2D critical perc cluster seed{s}", 91 / 48 / 1.1306))
    json.dump(res, open(_os.path.join(HERE, "results", "a6_grid.json"), "w"), indent=1)

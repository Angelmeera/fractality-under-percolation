"""A9: single covering with MEMB over ALL radii r = 1..RMAX (no commensurate 3-point
grid). Per config: N_B(r), s(r); per box m, diameter, centre degree (damaged and
intact, for IV). usage: python a9_memb_all.py WHICH T NREAL p1,.. RMAX OUT.json"""
import sys, json, time
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, _os.path.join(HERE, "..", "src"))
import numpy as np, networkx as nx, lib
from fast_cover import csr_from_graph

which, t, nreal = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ps = [float(x) for x in sys.argv[4].split(",")]; rmax = int(sys.argv[5]); outp = sys.argv[6]
E0, N, par, g = lib.build(which, t); deg0 = np.bincount(E0.ravel(), minlength=N)
rng = np.random.default_rng(4242); recs = []; t0 = time.time()
for p in ps:
    for rep in range(1 if p >= 1 else nreal):
        E = lib.percolate_gc(E0, N, p, rng)
        G = nx.Graph(); G.add_edges_from(map(tuple, E)); nodes, idx, A = csr_from_graph(G)
        nodes = np.array(nodes); n = A.shape[0]
        rec = dict(p=p, rep=rep, Ngc=int(n), levels={})
        for r in range(1, rmax + 1):
            bx, cen = lib.memb_cover(A, r); st = lib.box_stats(A, bx, cen)
            keep = st["M"] >= 3
            rec["levels"][str(r)] = dict(NB=len(bx), s=float(st["kB"].mean() / st["khub"].mean()),
                                         M=st["M"][keep].tolist(), D=st["D"][keep].tolist(),
                                         kc=st["kc"][keep].tolist(), k0=deg0[nodes[np.array(cen)[keep]]].tolist())
        recs.append(rec)
        print(f"{which} t={t} p={p} rep={rep} NB(r=1..)=" + ",".join(str(rec['levels'][str(r)]['NB']) for r in range(1, rmax + 1)) + f" [{time.time()-t0:.0f}s]", flush=True)
        json.dump(dict(which=which, t=t, rmax=rmax, recs=recs), open(outp, "w"))
print("done", flush=True)

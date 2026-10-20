import sys
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, lib
from scipy.sparse.csgraph import shortest_path
E0,N,par,g = lib.build("GB",5); rng=np.random.default_rng(1)
radii=list(range(1,17))
for p in (1.0,0.95,0.8):
    E=lib.percolate_gc(E0,N,p,rng); A=lib.gc_csr(E); n=A.shape[0]
    NB=[len(lib.memb_cover(A,r)[0]) for r in radii]
    print(f"p={p} n={n} NB(r=1..16)=", NB, flush=True)
# chemical stretch between level-tau hubs and their children far ends: mean distance from root 0 to root 1 and between G_{t-tau} adjacent nodes
seeds=lib.seeds_for(par,g,5)
for p in (1.0,0.95,0.8):
    rng=np.random.default_rng(2); ratios={}
    for rep in range(3):
        E=lib.percolate_gc(E0,N,p,rng); A=lib.adj(E,N); gcn=set(np.unique(E))
        for tau in (1,2,3,4):
            # nodes born at generation <= 5-tau that are adjacent in G_{5-tau}: build that graph
            Et,Nt,_,_=lib.build("GB",5-tau)
            pairs=[(u,v) for u,v in Et[:200] if u in gcn and v in gcn]
            src=np.array(sorted(set(u for u,v in pairs)))
            if len(src)==0: continue
            D=shortest_path(A,unweighted=True,indices=src)
            idx={s:i for i,s in enumerate(src)}
            ds=[D[idx[u],v] for u,v in pairs if np.isfinite(D[idx[u],v])]
            ratios.setdefault(tau,[]).extend(ds)
    print(f"p={p} mean distance between nodes adjacent in G_(5-tau) (intact = 3^tau):", {t:round(float(np.mean(v)),2) for t,v in ratios.items()}, flush=True)

"""Per level tau: mean ln(m/m0), ln((D+1)/(D0+1)), ln(k/k0) for the hub-containing largest fragment.
If these converge to constants as tau grows, the large-scale exponents (and slope differences) are the intact ones."""
import sys
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, lib
from scipy.sparse.csgraph import connected_components
which, t = sys.argv[1], int(sys.argv[2]); ps=[float(x) for x in sys.argv[3].split(",")]; nrep=int(sys.argv[4])
E0,N,par,g = lib.build(which,t); seeds=lib.seeds_for(par,g,t); deg0=np.bincount(E0.ravel(),minlength=N)
rng=np.random.default_rng(9)
def boxes(E):
    A=lib.adj(E,N); deg=np.bincount(E.ravel(),minlength=N); gc=np.unique(E.ravel()); out={}
    for tau in range(1,t):
        seed=seeds[tau]; same=seed[E[:,0]]==seed[E[:,1]]
        _,lab=connected_components(lib.adj(E[same],N),directed=False)
        fs=np.bincount(lab[gc],minlength=lab.max()+1)
        best={}
        for v in gc:
            s=seed[v]; f=lab[v]
            if s not in best or fs[f]>fs[best[s]]: best[s]=f
        hubs=[h for h in np.unique(seed) if h in best and lab[h]==best[h] and deg[h]>0 and fs[best[h]]>=3]
        hubs=[h for h in hubs if (tau>2 or h%7==0)]
        d={}
        for h in hubs:
            mem=gc[lab[gc]==lab[h]]
            d[h]=(len(mem), lib.diam_of(A,mem)+1, deg[h])
        out[tau]=d
    return out
B0=boxes(E0)
for p in ps:
    acc={}
    for rep in range(nrep):
        B=boxes(lib.percolate_gc(E0,N,p,rng))
        for tau,d in B.items():
            for h,(m,L,k) in d.items():
                m0,L0,k0=B0[tau][h]
                acc.setdefault(tau,[]).append((np.log(m/m0),np.log(L/L0),np.log(k/k0)))
    print(f"{which} t={t} p={p}  tau: <ln m/m0>  <ln L/L0>  <ln k/k0>   (ln p={np.log(p):.3f})")
    for tau,v in sorted(acc.items()):
        v=np.array(v); print(f"   {tau}: {v[:,0].mean():+.3f}  {v[:,1].mean():+.3f}  {v[:,2].mean():+.3f}   n={len(v)}")

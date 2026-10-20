"""Is the per-level beta_tau < 1 under damage regression dilution from binomial thinning of hub degree?
Regress ln m (largest fragment of each level-tau group, 'pub') on ln k_damaged, on ln k_intact (instrument),
and IV estimate beta_IV = cov(ln m, ln k0)/cov(ln k, ln k0)."""
import sys
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, lib
from scipy.sparse.csgraph import connected_components
which, t = sys.argv[1], int(sys.argv[2]); ps=[float(x) for x in sys.argv[3].split(",")]; nrep=int(sys.argv[4])
E0,N,par,g = lib.build(which,t); seeds=lib.seeds_for(par,g,t)
deg0=np.bincount(E0.ravel(),minlength=N)
rng=np.random.default_rng(5)
for p in ps:
    acc={}
    for rep in range(nrep):
        E=lib.percolate_gc(E0,N,p,rng); deg=np.bincount(E.ravel(),minlength=N)
        for tau in range(1,t):
            seed=seeds[tau]; same=seed[E[:,0]]==seed[E[:,1]]
            _,lab=connected_components(lib.adj(E[same],N),directed=False)
            hubs=np.unique(seed)
            # fragment containing the hub, and require it to be the largest fragment of the group
            gc=np.unique(E.ravel())
            fs=np.bincount(lab[gc],minlength=lab.max()+1)
            M,Kd,K0=[],[],[]
            # largest fragment per group
            best={}
            for v in gc:
                s=seed[v]; f=lab[v]
                if s not in best or fs[f]>fs[best[s]]: best[s]=f
            for h in hubs:
                if h not in best or deg[h]==0 or lab[h]!=best[h]: continue
                m=fs[best[h]]
                if m<3: continue
                M.append(m); Kd.append(deg[h]); K0.append(deg0[h])
            M,Kd,K0=(np.log(np.array(x,float)) for x in (M,Kd,K0))
            if len(M)<5 or np.var(K0)==0: continue
            ols=np.polyfit(Kd,M,1)[0]; red=np.polyfit(K0,M,1)[0]
            iv=np.cov(M,K0)[0,1]/np.cov(Kd,K0)[0,1]
            acc.setdefault(tau,[]).append((ols,red,iv,len(M)))
    print(f"{which} t={t} p={p}: tau: OLS(ln m|ln k) / OLS(ln m|ln k_intact) / IV(k_intact) [n boxes]")
    for tau,v in sorted(acc.items()):
        v=np.array(v); print(f"   tau={tau}: {v[:,0].mean():.3f}  {v[:,1].mean():.3f}  {v[:,2].mean():.3f}  [{v[:,3].mean():.0f}]")

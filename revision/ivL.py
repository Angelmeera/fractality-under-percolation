import sys
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, lib
from a8_lib import boxes, tsls
for which,t,ps in (("GB",5,(0.9,0.8,0.7,0.65)),("GB",6,(0.8,0.65)),("GA",6,(0.8,0.72))):
    E0,N,par,g=lib.build(which,t); seeds=lib.seeds_for(par,g,t); deg0=np.bincount(E0.ravel(),minlength=N)
    rng=np.random.default_rng(5)
    for p in ps:
        res=[]
        for r in range(10 if t==5 else 4):
            E=lib.percolate_gc(E0,N,p,rng); b=boxes(E,N,seeds,t,deg0)["pub"]
            M,L,K,K0,lev=(np.array(b[k],float) for k in ("M","L","K","K0","lev"))
            s=lev>=2; y=np.log(M[s]); X=np.column_stack([np.ones(s.sum()),np.log(L[s]),np.log(K[s])])
            Z1=np.column_stack([np.ones(s.sum()),np.log(L[s]),np.log(K0[s])])
            Z2=np.column_stack([np.ones(s.sum()),lev[s]*np.log(3),np.log(K0[s])])
            v1=tsls(y,X,Z1); v2=tsls(y,X,Z2); o=np.linalg.lstsq(X,y,rcond=None)[0]
            res.append((o[1],o[2],v1[1],v1[2],v2[1],v2[2]))
        r=np.mean(res,0); sd=np.std(res,0)
        print(f"{which} t={t} p={p}: OLS a,b={r[0]:.3f},{r[1]:.3f} | IV(k0) a,b={r[2]:.3f},{r[3]:.3f} | IV(k0, L0=lambda^tau) a,b={r[4]:.3f},{r[5]:.3f} (sd b {sd[5]:.3f})",flush=True)

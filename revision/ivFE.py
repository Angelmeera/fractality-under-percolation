import sys
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, lib
from a8_lib import boxes, tsls
for which,t,ps,nr in (("GB",5,(0.9,0.8,0.7,0.65),10),("GA",5,(0.8,0.72),10),("GB",6,(0.8,0.65),4)):
    E0,N,par,g=lib.build(which,t); seeds=lib.seeds_for(par,g,t); deg0=np.bincount(E0.ravel(),minlength=N)
    rng=np.random.default_rng(5)
    for p in ps:
        res=[]
        for r in range(nr):
            E=lib.percolate_gc(E0,N,p,rng)
            for conv in ("pub","mass"):
                b=boxes(E,N,seeds,t,deg0)[conv] if conv=="pub" else bb
                if conv=="pub": bb=boxes(E,N,seeds,t,deg0)["mass"]
                M,L,K,K0,lev=(np.array(b[k],float) for k in ("M","L","K","K0","lev"))
                s=lev>=2; y=np.log(M[s]); levs=sorted(set(lev[s]))
                D=np.column_stack([(lev[s]==l).astype(float) for l in levs])
                X=np.column_stack([D,np.log(K[s])]); Z=np.column_stack([D,np.log(K0[s])])
                v=tsls(y,X,Z); o=np.linalg.lstsq(X,y,rcond=None)[0]
                res.append((conv,o[-1],v[-1]))
        for conv in ("pub","mass"):
            a=np.array([(x[1],x[2]) for x in res if x[0]==conv])
            print(f"{which} t={t} p={p} {conv}: level-fixed-effects beta OLS={a[:,0].mean():.3f} IV={a[:,1].mean():.3f}±{a[:,1].std():.3f}",flush=True)

import sys, time
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, lib
for t in (4,5):
    E,N,par,g = lib.build("GB",t); A=lib.gc_csr(E)
    for r in (1,4,13,40):
        t0=time.time(); boxes,cen=lib.memb_cover(A,r); st=lib.box_stats(A,boxes,cen)
        print(t, r, len(boxes), "maxD",st["D"].max(), "kc==khub", np.mean(st["kc"]==st["khub"]), f"{time.time()-t0:.1f}s", flush=True)

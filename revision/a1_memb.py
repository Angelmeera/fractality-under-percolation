"""A1: single-covering test. On each percolated giant component, ONE covering
(commensurate MEMB, r_B = (lambda^tau - 1)/2) supplies d_B, d_k, alpha and beta.
Per configuration we store everything needed for the closure, the nulls, the
per-level betas and the s_m / s_k decomposition.
usage: python a1_memb.py WHICH T NREAL p1,p2,... r1,r2,... OUT.json"""
import sys, json, time
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import lib

which, t, nreal = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ps = [float(x) for x in sys.argv[4].split(",")]
radii = [int(x) for x in sys.argv[5].split(",")]
out_path = sys.argv[6]
E0, N, par, g = lib.build(which, t)
rng = np.random.default_rng(20261019)
recs = []; t0 = time.time()
for p in ps:
    for r in range(1 if p >= 1 else nreal):
        E = lib.percolate_gc(E0, N, p, rng)
        A = lib.gc_csr(E); n = A.shape[0]
        rec = dict(p=p, rep=r, Ngc=int(n), levels={})
        for lev, rb in enumerate(radii, start=1):
            boxes, cen = lib.memb_cover(A, rb)
            st = lib.box_stats(A, boxes, cen)
            rec["levels"][str(rb)] = dict(NB=len(boxes), M=st["M"].tolist(), D=st["D"].tolist(),
                                          kc=st["kc"].tolist(), khub=st["khub"].tolist(),
                                          s=float(st["kB"].mean() / st["khub"].mean()))
        recs.append(rec)
        print(f"{which} t={t} p={p} rep={r} Ngc={n} NB=" +
              ",".join(str(rec['levels'][str(rb)]['NB']) for rb in radii) + f" [{time.time()-t0:.0f}s]", flush=True)
    json.dump(dict(which=which, t=t, radii=radii, recs=recs), open(out_path, "w"))
print("done", flush=True)

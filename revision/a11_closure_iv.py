"""A11: closure of d_B = alpha + beta d_k on the SAME configurations with greedy d_B, d_k
and construction-box alpha, beta fitted by OLS (damaged degree) and by IV (pre-damage
degree), both conventions, all levels and levels>=2. usage: WHICH T NREAL ps OUT"""
import sys, json, time
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, lib
from a8_lib import boxes, tsls
which, t, nreal = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); ps = [float(x) for x in sys.argv[4].split(",")]; outp = sys.argv[5]
E0, N, par, g = lib.build(which, t); seeds = lib.seeds_for(par, g, t); deg0 = np.bincount(E0.ravel(), minlength=N)
rng = np.random.default_rng(31337); recs = []; t0 = time.time(); GRID = [2, 4, 10, 28]
for p in ps:
    for rep in range(1 if p >= 1 else nreal):
        E = lib.percolate_gc(E0, N, p, rng)
        A = lib.gc_csr(E); n = A.shape[0]; NB, s = [], []
        for lb in GRID:
            st = lib.box_stats(A, lib.greedy_cover(A, lb), diam=False); NB.append(len(st["M"])); s.append(float(st["kB"].mean() / st["khub"].mean()))
        dB = -lib.slope(GRID, np.array(NB) / n); dk = -lib.slope(GRID, s)
        rec = dict(p=p, rep=rep, dB=dB, dk=dk)
        for c, b in boxes(E, N, seeds, t, deg0).items():
            M, L, K, K0, lev = (np.array(b[k], float) for k in ("M", "L", "K", "K0", "lev"))
            y, lL, lK, lK0 = np.log(M), np.log(L), np.log(K), np.log(K0)
            for mode in ("all", "ge2"):
                sel = np.ones(len(y), bool) if mode == "all" else lev >= 2
                X = np.column_stack([np.ones(sel.sum()), lL[sel], lK[sel]]); Z = np.column_stack([np.ones(sel.sum()), lL[sel], lK0[sel]])
                o = np.linalg.lstsq(X, y[sel], rcond=None)[0]; v = tsls(y[sel], X, Z)
                for est, sol in (("ols", o), ("iv", v)):
                    rec[f"{c}_{mode}_{est}"] = dict(alpha=float(sol[1]), beta=float(sol[2]), closure=float(100 * (dB - (sol[1] + sol[2] * dk)) / dB))
        recs.append(rec)
        print(f"{which} p={p} rep={rep} dB={dB:.3f} dk={dk:.3f} pub all OLS clo={rec['pub_all_ols']['closure']:+.2f} IV clo={rec['pub_all_iv']['closure']:+.2f} | mass all OLS {rec['mass_all_ols']['closure']:+.2f} IV {rec['mass_all_iv']['closure']:+.2f} [{time.time()-t0:.0f}s]", flush=True)
        json.dump(dict(which=which, t=t, recs=recs), open(outp, "w"))
print("done", flush=True)

"""A2/A3/A4/A5 on the same configurations:
 - construction-aware boxes under pub / mass / retile conventions,
   L = diam+1 / nominal / diam, fits over all levels, levels>=2 and level-weighted,
   per-level beta, mass cuts 2,3,4,6 (A2, A3, A5);
 - greedy-degree cover of the same giant component on {2,4,10,28} for the mixed
   test d_B(p) vs alpha+beta d_k (A4).
usage: python a24_construction.py WHICH T NREAL p1,p2,.. OUT.json [nogreedy]"""
import sys, json, time
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import lib

which, t, nreal = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ps = [float(x) for x in sys.argv[4].split(",")]
out_path = sys.argv[5]
do_greedy = not (len(sys.argv) > 6 and sys.argv[6] == "nogreedy")
GRID = [2, 4, 10, 28]
E0, N, par, g = lib.build(which, t)
seeds = lib.seeds_for(par, g, t)
rng = np.random.default_rng(777)
recs = []; t0 = time.time()
for p in ps:
    for r in range(1 if p >= 1 else nreal):
        E = lib.percolate_gc(E0, N, p, rng)
        rec = dict(p=p, rep=r, Ngc=int(len(np.unique(E))))
        for lconv in ("plus1", "nominal", "diam"):
            cb = lib.construction_boxes(E, N, seeds, t, lconv=lconv)
            for conv, B in cb.items():
                key = f"{conv}_{lconv}"
                res = {}
                for mode in ("all", "ge2", "weighted"):
                    res[mode] = lib.joint_fit(B["M"], B["L"], B["K"], B["lev"], mode=mode)
                res["levels"] = lib.per_level_beta(B["M"], B["K"], B["lev"])
                res["sm"], res["sk"] = lib.level_slopes(B["M"], B["L"], B["K"], B["lev"])
                if lconv == "plus1":
                    res["cuts"] = {c: lib.joint_fit(B["M"], B["L"], B["K"], B["lev"], mode="all", cut=c) for c in (2, 3, 4, 6)}
                    res["cuts_ge2"] = {c: lib.joint_fit(B["M"], B["L"], B["K"], B["lev"], mode="ge2", cut=c) for c in (2, 3, 4, 6)}
                    Kp = np.array(B["K"], float); rng.shuffle(Kp)
                    res["null"] = lib.joint_fit(B["M"], B["L"], Kp, B["lev"], mode="all")
                    res["massfrac"] = float(np.sum(B["M"]) / rec["Ngc"])
                rec[key] = res
        if do_greedy:
            A = lib.gc_csr(E); n = A.shape[0]
            grid = [l for l in GRID]
            NB, s = [], []
            for lb in grid:
                st = lib.box_stats(A, lib.greedy_cover(A, lb), diam=False)
                NB.append(len(st["M"])); s.append(float(st["kB"].mean() / st["khub"].mean()))
            rec["greedy"] = dict(grid=grid, NB=NB, s=s, dB=-lib.slope(grid, np.array(NB) / n), dk=-lib.slope(grid, s))
        recs.append(rec)
        pb = rec["pub_plus1"]["all"]
        print(f"{which} t={t} p={p} rep={r} pub a,b={pb['alpha']:.3f},{pb['beta']:.3f} ge2 b={rec['pub_plus1']['ge2']['beta']:.3f} "
              + (f"greedy dB,dk={rec['greedy']['dB']:.3f},{rec['greedy']['dk']:.3f}" if do_greedy else "") + f" [{time.time()-t0:.0f}s]", flush=True)
        json.dump(dict(which=which, t=t, recs=recs), open(out_path, "w"))
print("done", flush=True)

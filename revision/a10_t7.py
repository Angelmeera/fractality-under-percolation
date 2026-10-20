"""A10: large-system (t=7, N = 1,198,372) check of the two Phase-1 claims.
 (i)  beta_tau with the intact degree as instrument stays ~1 under damage (F2);
 (ii) at large scales the giant component's units return to the intact scaling:
      hub degree thinned by a constant factor, surviving mass fraction -> const,
      chemical stretch ln(L/L0) -> const, so the unit 'local dimension'
      d_tau = [ln<m>_{tau+1} - ln<m>_tau] / [ln<L>_{tau+1} - ln<L>_tau] -> D_f.
Construction boxes only (no covering), 'pub' = largest connected fragment of each
level-tau group (>=3 nodes), 'mass' = all GC members of the group; L = diam+1.
usage: python a10_t7.py WHICH T NREAL p1,p2,.. OUT.json"""
import sys, json, time
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, scipy.sparse as sp, lib
from scipy.sparse.csgraph import connected_components, shortest_path

def frag_diams(Ei, lab, frags, members):
    """Exact-ish (double sweep) diameter of each fragment in `frags` using only the
    intra-group edges Ei (fragments = connected components of Ei)."""
    fl = lab[Ei[:, 0]]
    order = np.argsort(fl, kind="stable"); Es = Ei[order]; fls = fl[order]
    starts = np.searchsorted(fls, frags, side="left"); ends = np.searchsorted(fls, frags, side="right")
    out = np.zeros(len(frags), int)
    for i, (f, a, b) in enumerate(zip(frags, starts, ends)):
        mem = members[i]
        if len(mem) < 3:
            out[i] = len(mem) - 1; continue
        e = Es[a:b]; loc = np.searchsorted(mem, e)
        n = len(mem)
        A = sp.coo_matrix((np.ones(2 * len(e)), (np.r_[loc[:, 0], loc[:, 1]], np.r_[loc[:, 1], loc[:, 0]])), shape=(n, n)).tocsr()
        d1 = shortest_path(A, unweighted=True, indices=0)
        d2 = shortest_path(A, unweighted=True, indices=int(np.argmax(d1)))
        out[i] = int(d2.max())
    return out

which, t, nreal = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ps = [float(x) for x in sys.argv[4].split(",")]; outp = sys.argv[5]
t0 = time.time()
E0, N, par, g = lib.build(which, t); seeds = lib.seeds_for(par, g, t)
deg0 = np.bincount(E0.ravel(), minlength=N)
m0_all = {tau: np.bincount(seeds[tau], minlength=N) for tau in range(1, t)}
print(f"built {which} t={t} N={N} M={len(E0)} [{time.time()-t0:.0f}s]", flush=True)
rng = np.random.default_rng(777000); recs = []
for p in ps:
    for rep in range(1 if p >= 1 else nreal):
        E = lib.percolate_gc(E0, N, p, rng)
        gc = np.unique(E.ravel()); deg = np.bincount(E.ravel(), minlength=N)
        rec = dict(p=p, rep=rep, Ngc=int(len(gc)), levels={})
        for tau in range(1, t):
            seed = seeds[tau]; same = seed[E[:, 0]] == seed[E[:, 1]]; Ei = E[same]
            _, lab = connected_components(lib.adj(Ei, N), directed=False)
            fids, inv, fsize = np.unique(lab[gc], return_inverse=True, return_counts=True)
            fseed = np.empty(len(fids), int); fseed[inv] = seed[gc]
            maxdeg = np.zeros(len(fids)); np.maximum.at(maxdeg, inv, deg[gc])
            kfrag = maxdeg.copy(); ishub = seed[gc] == gc; kfrag[inv[ishub]] = deg[gc[ishub]]
            gs, ginv = np.unique(fseed, return_inverse=True); gmass = np.bincount(ginv, weights=fsize)
            big = np.full(len(gs), -1)
            for fi in np.argsort(fsize, kind="stable"):
                big[ginv[fi]] = fi
            keep = fsize[big] >= 3
            bigk = big[keep]
            order = np.argsort(inv, kind="stable")
            allmem = np.split(gc[order], np.flatnonzero(np.diff(inv[order])) + 1)
            mems = [np.sort(allmem[fi]) for fi in bigk]
            D = frag_diams(Ei, lab, fids[bigk], mems)
            M = fsize[bigk].astype(float); Mm = gmass[keep]; L = D + 1.0
            K = kfrag[bigk]; hubs = gs[keep]; K0 = deg0[hubs].astype(float); M0 = m0_all[tau][hubs].astype(float)
            y, ym, lL, lK, lK0 = np.log(M), np.log(Mm), np.log(L), np.log(K), np.log(K0)
            lv = dict(n=int(len(M)), mean_lnm=float(y.mean()), mean_lnm_mass=float(ym.mean()), mean_lnL=float(lL.mean()),
                      mean_lnk=float(lK.mean()), mean_lnk0=float(lK0.mean()),
                      thin=float(np.mean(lK - lK0)), massfrac_pub=float(np.mean(y - np.log(M0))),
                      massfrac_mass=float(np.mean(ym - np.log(M0))), stretch=float(np.mean(lL - tau * np.log(3))))
            if len(M) >= 5 and np.var(lK) > 0:
                for nm, yy in (("pub", y), ("mass", ym)):
                    lv[f"{nm}_ols"] = float(np.polyfit(lK, yy, 1)[0])
                    c = np.cov(lK, lK0)[0, 1]
                    lv[f"{nm}_iv"] = float(np.cov(yy, lK0)[0, 1] / c) if c != 0 else float("nan")
            rec["levels"][str(tau)] = lv
            print(f"   {which} p={p} rep={rep} tau={tau} n={len(M)} pub OLS/IV={lv.get('pub_ols',np.nan):.3f}/{lv.get('pub_iv',np.nan):.3f} "
                  f"mass OLS/IV={lv.get('mass_ols',np.nan):.3f}/{lv.get('mass_iv',np.nan):.3f} thin={lv['thin']:+.3f} "
                  f"mfrac={lv['massfrac_pub']:+.3f} stretch={lv['stretch']:+.3f} [{time.time()-t0:.0f}s]", flush=True)
        recs.append(rec)
        json.dump(dict(which=which, t=t, recs=recs), open(outp, "w"))
print("done", flush=True)

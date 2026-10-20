"""A8: is the drop of beta under damage regression dilution from binomial thinning of
the hub degree? Construction boxes ('pub' and 'mass' conventions, L = diam+1).
Per level and jointly: OLS on the damaged degree k, IV with the intact degree k0 as
instrument (2SLS for the joint fit), and a method-of-moments correction that needs
only p (no k0): Var(ln k | k0) ~= (1-p)/(p k0), estimated per box by (1-p)/(p k).
usage: python a8_iv.py WHICH T NREAL p1,p2,.. OUT.json"""
import sys, json, time
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, lib
from scipy.sparse.csgraph import connected_components

def tsls(y, X, Z):
    """2SLS: y on X with instruments Z (both include the constant)."""
    P = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    return np.linalg.lstsq(P, y, rcond=None)[0]

def boxes(E, N, seeds, t, deg0):
    gc = np.unique(E.ravel()); deg = np.bincount(E.ravel(), minlength=N); A = lib.adj(E, N)
    out = {c: dict(M=[], L=[], K=[], K0=[], lev=[]) for c in ("pub", "mass")}
    for tau in range(1, t):
        seed = seeds[tau]; same = seed[E[:, 0]] == seed[E[:, 1]]
        _, lab = connected_components(lib.adj(E[same], N), directed=False)
        fids, inv, fsize = np.unique(lab[gc], return_inverse=True, return_counts=True)
        fseed = np.empty(len(fids), int); fseed[inv] = seed[gc]
        maxdeg = np.zeros(len(fids)); np.maximum.at(maxdeg, inv, deg[gc])
        kfrag = maxdeg.copy(); ishub = seed[gc] == gc; kfrag[inv[ishub]] = deg[gc[ishub]]
        order = np.argsort(inv, kind="stable"); members = np.split(gc[order], np.flatnonzero(np.diff(inv[order])) + 1)
        gs, ginv = np.unique(fseed, return_inverse=True); gmass = np.bincount(ginv, weights=fsize)
        big = np.full(len(gs), -1)
        for fi in np.argsort(fsize, kind="stable"):
            big[ginv[fi]] = fi
        for gi, fi in enumerate(big):
            if fsize[fi] < 3:
                continue
            L = lib.diam_of(A, members[fi]) + 1
            for c, m in (("pub", fsize[fi]), ("mass", gmass[gi])):
                out[c]["M"].append(m); out[c]["L"].append(L); out[c]["K"].append(kfrag[fi])
                out[c]["K0"].append(deg0[gs[gi]]); out[c]["lev"].append(tau)
    return out

which, t, nreal = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ps = [float(x) for x in sys.argv[4].split(",")]; outp = sys.argv[5]
E0, N, par, g = lib.build(which, t); seeds = lib.seeds_for(par, g, t)
deg0 = np.bincount(E0.ravel(), minlength=N)
rng = np.random.default_rng(99); recs = []; t0 = time.time()
for p in ps:
    for r in range(1 if p >= 1 else nreal):
        E = lib.percolate_gc(E0, N, p, rng)
        B = boxes(E, N, seeds, t, deg0); rec = dict(p=p, rep=r)
        for c, b in B.items():
            M, L, K, K0, lev = (np.array(b[k], float) for k in ("M", "L", "K", "K0", "lev"))
            y, lL, lK, lK0 = np.log(M), np.log(L), np.log(K), np.log(K0)
            per = {}
            for tau in range(1, t):
                s = lev == tau
                if s.sum() < 5 or np.var(lK[s]) == 0:
                    continue
                ols = np.polyfit(lK[s], y[s], 1)[0]
                iv = np.cov(y[s], lK0[s])[0, 1] / np.cov(lK[s], lK0[s])[0, 1]
                noise = np.mean((1 - p) / (p * K[s])) if p < 1 else 0.0
                mm = ols * np.var(lK[s]) / (np.var(lK[s]) - noise) if np.var(lK[s]) > noise else np.nan
                per[tau] = dict(ols=float(ols), iv=float(iv), mm=float(mm), n=int(s.sum()))
            joint = {}
            for mode in ("all", "ge2"):
                s = np.ones(len(y), bool) if mode == "all" else lev >= 2
                X = np.column_stack([np.ones(s.sum()), lL[s], lK[s]]); Z = np.column_stack([np.ones(s.sum()), lL[s], lK0[s]])
                o = np.linalg.lstsq(X, y[s], rcond=None)[0]; v = tsls(y[s], X, Z)
                joint[mode] = dict(ols_alpha=float(o[1]), ols_beta=float(o[2]), iv_alpha=float(v[1]), iv_beta=float(v[2]))
            rec[c] = dict(per=per, joint=joint)
        recs.append(rec)
        print(f"{which} t={t} p={p} rep={r} pub joint OLS b={rec['pub']['joint']['all']['ols_beta']:.3f} IV b={rec['pub']['joint']['all']['iv_beta']:.3f} "
              f"| mass OLS b={rec['mass']['joint']['all']['ols_beta']:.3f} IV b={rec['mass']['joint']['all']['iv_beta']:.3f} [{time.time()-t0:.0f}s]", flush=True)
        json.dump(dict(which=which, t=t, recs=recs), open(outp, "w"))
print("done", flush=True)

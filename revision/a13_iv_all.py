"""A13 (Major 3; cross-check items C, D, E, C2): all three box-mass conventions
(largest piece 'pub', total surviving mass 'mass', re-tiled 'retile') with OLS and
IV (instrument: pre-damage hub degree k0), per level, joint, and with level fixed
effects; plus the diagnostics that bound what the IV means:
  - within-(level, k0) OLS slope of ln m on ln k (0 under pure classical noise in ln k)
  - reduced form (ln m on ln k0) and first stage (ln k on ln k0) per level
  - fraction of boxes whose hub is not in the largest piece
Same bond configurations as a8_iv.py for the first NREAL_A8 configurations (rng 99),
then further configurations to reach NREAL.
usage: python a13_iv_all.py WHICH T NREAL p1,p2,.. OUT.json"""
import sys, json, time
from collections import deque
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, scipy.sparse as sp, lib
from scipy.sparse.csgraph import connected_components


def tsls(y, X, Z):
    P = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    return np.linalg.lstsq(P, y, rcond=None)[0]


def boxes3(E, N, seeds, t, deg0):
    gc = np.unique(E.ravel()); deg = np.bincount(E.ravel(), minlength=N); A = lib.adj(E, N)
    out = {c: dict(M=[], L=[], K=[], K0=[], lev=[], hubin=[]) for c in ("pub", "mass", "retile")}
    for tau in range(1, t):
        seed = seeds[tau]; same = seed[E[:, 0]] == seed[E[:, 1]]
        _, lab = connected_components(lib.adj(E[same], N), directed=False)
        fids, inv, fsize = np.unique(lab[gc], return_inverse=True, return_counts=True); nf = len(fids)
        fseed = np.empty(nf, int); fseed[inv] = seed[gc]
        maxdeg = np.zeros(nf); np.maximum.at(maxdeg, inv, deg[gc])
        kfrag = maxdeg.copy(); ishub = seed[gc] == gc; kfrag[inv[ishub]] = deg[gc[ishub]]
        hasHub = np.zeros(nf, bool); hasHub[inv[ishub]] = True
        order = np.argsort(inv, kind="stable"); members = np.split(gc[order], np.flatnonzero(np.diff(inv[order])) + 1)
        gs, ginv = np.unique(fseed, return_inverse=True); gmass = np.bincount(ginv, weights=fsize)
        big = np.full(len(gs), -1)
        for fi in np.argsort(fsize, kind="stable"):
            big[ginv[fi]] = fi
        lab2f = -np.ones(lab.max() + 1, int); lab2f[fids] = np.arange(nf)
        cross = ~same; fu, fv = lab2f[lab[E[cross, 0]]], lab2f[lab[E[cross, 1]]]
        FG = sp.coo_matrix((np.ones(2 * len(fu)), (np.r_[fu, fv], np.r_[fv, fu])), shape=(nf, nf)).tocsr()
        owner = -np.ones(nf, int); dq = deque()
        for fi in big:
            owner[fi] = fi; dq.append(fi)
        ip, ix = FG.indptr, FG.indices
        while dq:
            u = dq.popleft()
            for v in ix[ip[u]:ip[u + 1]]:
                if owner[v] < 0:
                    owner[v] = owner[u]; dq.append(v)
        ret = {}
        for fi in range(nf):
            ret.setdefault(owner[fi], []).append(fi)
        for gi, fi in enumerate(big):
            mem = members[fi]
            if len(mem) < 3:
                continue
            L = lib.diam_of(A, mem) + 1
            fl = ret[fi]; rmem = np.concatenate([members[x] for x in fl])
            Lr = lib.diam_of(A, rmem) + 1 if len(fl) > 1 else L
            for c, m, l in (("pub", len(mem), L), ("mass", gmass[gi], L), ("retile", len(rmem), Lr)):
                o = out[c]; o["M"].append(m); o["L"].append(l); o["K"].append(kfrag[fi])
                o["K0"].append(deg0[gs[gi]]); o["lev"].append(tau); o["hubin"].append(bool(hasHub[fi]))
    return out


def analyse(b, t):
    M, L, K, K0, lev = (np.array(b[k], float) for k in ("M", "L", "K", "K0", "lev"))
    hubin = np.array(b["hubin"], bool)
    y, lL, lK, lK0 = np.log(M), np.log(L), np.log(K), np.log(K0)
    per = {}
    for tau in range(1, t):
        s = lev == tau
        if s.sum() < 5 or np.var(lK0[s]) == 0:
            continue
        ols = np.polyfit(lK[s], y[s], 1)[0] if np.var(lK[s]) > 0 else np.nan
        cyk0 = np.cov(y[s], lK0[s])[0, 1]; ckk0 = np.cov(lK[s], lK0[s])[0, 1]; vk0 = np.var(lK0[s], ddof=1)
        # within-(level,k0) slope of ln m on ln k: demean by k0 cell
        yy, kk = y[s].copy(), lK[s].copy()
        for v in np.unique(K0[s]):
            c = K0[s] == v; yy[c] -= yy[c].mean(); kk[c] -= kk[c].mean()
        within = float((yy @ kk) / (kk @ kk)) if (kk @ kk) > 0 else np.nan
        per[tau] = dict(ols=float(ols), iv=float(cyk0 / ckk0), reduced=float(cyk0 / vk0), first=float(ckk0 / vk0),
                        within_k0=within, n=int(s.sum()), n_k0=int(len(np.unique(K0[s]))))
    joint = {}
    for mode in ("all", "ge2"):
        s = np.ones(len(y), bool) if mode == "all" else lev >= 2
        X = np.column_stack([np.ones(s.sum()), lL[s], lK[s]]); Z = np.column_stack([np.ones(s.sum()), lL[s], lK0[s]])
        o = np.linalg.lstsq(X, y[s], rcond=None)[0]; v = tsls(y[s], X, Z)
        joint[mode] = dict(ols_alpha=float(o[1]), ols_beta=float(o[2]), iv_alpha=float(v[1]), iv_beta=float(v[2]))
    s = lev >= 2; levs = sorted(set(lev[s]))
    D = np.column_stack([(lev[s] == l).astype(float) for l in levs])
    X = np.column_stack([D, lK[s]]); Z = np.column_stack([D, lK0[s]])
    fe = dict(ols=float(np.linalg.lstsq(X, y[s], rcond=None)[0][-1]), iv=float(tsls(y[s], X, Z)[-1]))
    hub = dict(all=float(1 - hubin.mean()), ge2=float(1 - hubin[lev >= 2].mean()))
    return dict(per=per, joint=joint, fe=fe, hub_out=hub)


if __name__ == "__main__":
    which, t, nreal = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    ps = [float(x) for x in sys.argv[4].split(",")]; outp = sys.argv[5]
    E0, N, par, g = lib.build(which, t); seeds = lib.seeds_for(par, g, t)
    deg0 = np.bincount(E0.ravel(), minlength=N)
    import os; rng = np.random.default_rng(int(os.environ.get("SEED", 99))); recs = []; t0 = time.time()
    for p in ps:
        for r in range(1 if p >= 1 else nreal):
            E = lib.percolate_gc(E0, N, p, rng)
            B = boxes3(E, N, seeds, t, deg0)
            rec = dict(p=p, rep=r, **{c: analyse(b, t) for c, b in B.items()})
            recs.append(rec)
            print(f"{which} t={t} p={p} rep={r} " + " | ".join(
                f"{c}: OLS {rec[c]['joint']['all']['ols_beta']:.3f} IV {rec[c]['joint']['all']['iv_beta']:.3f} FE-IV {rec[c]['fe']['iv']:.3f}"
                for c in ("pub", "mass", "retile")) + f" [{time.time()-t0:.0f}s]", flush=True)
            json.dump(dict(which=which, t=t, recs=recs), open(outp, "w"))
    print("done", flush=True)

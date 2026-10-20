"""Phase-1 analysis library for the referee reply (scaling paper).

Everything needed to measure d_B, d_k, alpha and beta on bond-percolated
Yakubo-Fujiki FSFNs, with:
  * construction-aware hub-centred boxes under three mass conventions
      pub    : largest connected fragment of each level-tau group
      mass   : all surviving members of the group (L, k from the largest fragment)
      retile : largest fragment of each group, with every other fragment merged into
               the box it first reaches through the giant component (multi-source
               BFS on the fragment graph). Tiles the giant component with connected
               boxes, so every box has a diameter.
  * greedy-degree covering (the paper's fast_cover.cover_one, string tie-break)
  * MEMB (the paper's covering.cover_memb), whose boxes are connected and tile the GC.
FSFN builder and node labels are identical to yakubo_fujiki.build_parented.
"""
import sys
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, _os.path.join(HERE, "..", "src"))
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, shortest_path

GEN = {"GB": [(0, 2), (0, 3), (1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5)],
       "GA": [(0, 2), (0, 3), (1, 4), (1, 5), (2, 3), (2, 4), (3, 5), (4, 5)]}
PC = {"GB": 0.6288, "GA": 0.6961}
LAM, KAPPA, MGEN = 3, 2, 8
DF = np.log(8) / np.log(3)
DK = np.log(2) / np.log(3)
ALPHA_EX = np.log(4) / np.log(3)


# ------------------------------------------------------------------ model
def build(which, t):
    ge = GEN[which]
    edges = [(0, 1)]; ctr = 2
    parent = {0: -1, 1: -1}; gen = {0: 0, 1: 0}
    for s in range(1, t + 1):
        new = []
        for (u, v) in edges:
            f = {2: ctr, 3: ctr + 1, 4: ctr + 2, 5: ctr + 3}; ctr += 4
            parent[f[2]] = u; parent[f[3]] = u; parent[f[4]] = v; parent[f[5]] = v
            for r in (2, 3, 4, 5):
                gen[f[r]] = s
            mp = {0: u, 1: v, **f}
            new += [(mp[a], mp[b]) for a, b in ge]
        edges = new
    par = np.array([parent[i] for i in range(ctr)])
    g = np.array([gen[i] for i in range(ctr)])
    return np.array(edges, dtype=np.int64), ctr, par, g


def seeds_for(par, g, t):
    out = {}
    for tau in range(1, t):
        cut = t - tau
        s = np.arange(len(par))
        for _ in range(t + 1):
            m = g[s] > cut
            if not m.any():
                break
            s[m] = par[s[m]]
        out[tau] = s
    return out


def percolate_gc(edges, N, p, rng):
    if p >= 1:
        return edges
    E = edges[rng.random(len(edges)) < p]
    A = sp.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(N, N))
    _, lab = connected_components(A, directed=False)
    return E[lab[E[:, 0]] == np.bincount(lab).argmax()]


def adj(E, N):
    return sp.coo_matrix((np.ones(2 * len(E)), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])),
                         shape=(N, N)).tocsr()


def diam_of(A, mem):
    """Double-sweep diameter of the subgraph of A induced on mem (connected)."""
    if len(mem) == 1:
        return 0
    sub = A[mem][:, mem]
    d1 = shortest_path(sub, unweighted=True, indices=0)
    d2 = shortest_path(sub, unweighted=True, indices=int(np.argmax(np.where(np.isfinite(d1), d1, -1))))
    return int(np.max(d2[np.isfinite(d2)]))


# ------------------------------------------------------------------ fits
def joint_fit(M, L, K, levels=None, mode="all", cut=3):
    """ln m = c + alpha ln L + beta ln K. mode: all | ge2 (levels>=2) | weighted
    (each level gets equal total weight). Returns dict(alpha, beta, r2, n)."""
    M, L, K = (np.asarray(x, float) for x in (M, L, K))
    lev = np.asarray(levels) if levels is not None else np.ones(len(M), int)
    ok = (M >= cut) & (L > 0) & (K > 0)
    if mode == "ge2":
        ok &= lev >= 2
    if ok.sum() < 8 or len(set(np.round(L[ok], 6))) < 2 or len(set(K[ok])) < 2:
        return dict(alpha=np.nan, beta=np.nan, r2=np.nan, n=int(ok.sum()))
    y = np.log(M[ok]); X = np.column_stack([np.ones(ok.sum()), np.log(L[ok]), np.log(K[ok])])
    w = np.ones(ok.sum())
    if mode == "weighted":
        lv = lev[ok]
        for l in np.unique(lv):
            w[lv == l] = 1.0 / (lv == l).sum()
    sw = np.sqrt(w)
    sol, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    res = y - X @ sol
    ybar = np.average(y, weights=w)
    r2 = 1 - np.sum(w * res ** 2) / np.sum(w * (y - ybar) ** 2)
    return dict(alpha=float(sol[1]), beta=float(sol[2]), r2=float(r2), n=int(ok.sum()))


def per_level_beta(M, K, levels, cut=3):
    M, K, lev = (np.asarray(x, float) for x in (M, K, levels))
    out = {}
    for l in np.unique(lev):
        ok = (lev == l) & (M >= cut) & (K > 0)
        if ok.sum() >= 5 and len(set(K[ok])) > 1:
            out[int(l)] = float(np.polyfit(np.log(K[ok]), np.log(M[ok]), 1)[0])
    return out


def level_slopes(M, L, K, levels, cut=3):
    """s_m, s_k: slopes of ln m and ln k on ln L over the fitted boxes (identity:
    alpha + beta*s_k = s_m for the unweighted all-level fit)."""
    M, L, K = (np.asarray(x, float) for x in (M, L, K))
    ok = (M >= cut) & (L > 0) & (K > 0)
    x = np.log(L[ok])
    return float(np.polyfit(x, np.log(M[ok]), 1)[0]), float(np.polyfit(x, np.log(K[ok]), 1)[0])


# ------------------------------------------------------------------ construction boxes
def construction_boxes(E, N, seeds, t, lconv="plus1"):
    """Return dict conv -> dict(M, L, K, lev) for pub, mass, retile.
    lconv: 'diam' (measured diameter), 'plus1' (diameter+1), 'nominal' (lambda^tau)."""
    gc = np.unique(E.ravel())
    deg = np.bincount(E.ravel(), minlength=N)
    A = adj(E, N)
    out = {c: dict(M=[], L=[], K=[], lev=[]) for c in ("pub", "mass", "retile")}
    for tau in range(1, t):
        seed = seeds[tau]
        same = seed[E[:, 0]] == seed[E[:, 1]]
        Ai = adj(E[same], N)
        _, lab = connected_components(Ai, directed=False)
        lab_gc = lab[gc]
        fids, inv, fsize = np.unique(lab_gc, return_inverse=True, return_counts=True)
        nf = len(fids)
        fseed = np.empty(nf, int); fseed[inv] = seed[gc]
        maxdeg = np.zeros(nf); np.maximum.at(maxdeg, inv, deg[gc])
        kfrag = maxdeg.copy()
        ishub = seed[gc] == gc
        kfrag[inv[ishub]] = deg[gc[ishub]]
        order = np.argsort(inv, kind="stable")
        members = np.split(gc[order], np.flatnonzero(np.diff(inv[order])) + 1)
        gs, ginv = np.unique(fseed, return_inverse=True)
        big = np.full(len(gs), -1)
        for fi in np.argsort(fsize, kind="stable"):
            big[ginv[fi]] = fi
        isbig = np.zeros(nf, bool); isbig[big] = True
        gmass = np.bincount(ginv, weights=fsize)
        # retile: multi-source BFS on fragment graph from the big fragments
        lab2f = -np.ones(lab.max() + 1, int); lab2f[fids] = np.arange(nf)
        cross = ~same
        fu, fv = lab2f[lab[E[cross, 0]]], lab2f[lab[E[cross, 1]]]
        FG = sp.coo_matrix((np.ones(2 * len(fu)), (np.r_[fu, fv], np.r_[fv, fu])), shape=(nf, nf)).tocsr()
        owner = -np.ones(nf, int)
        from collections import deque
        dq = deque()
        for fi in big:  # big fragments in group order -> deterministic
            owner[fi] = fi; dq.append(fi)
        ip, ix = FG.indptr, FG.indices
        while dq:
            u = dq.popleft()
            for v in ix[ip[u]:ip[u + 1]]:
                if owner[v] < 0:
                    owner[v] = owner[u]; dq.append(v)
        ret_members = {}
        for fi in range(nf):
            ret_members.setdefault(owner[fi], []).append(fi)

        def Lval(mem):
            d = diam_of(A, mem)
            return {"diam": max(d, 1), "plus1": d + 1, "nominal": float(LAM ** tau)}[lconv]

        for gi, fi in enumerate(big):
            mem = members[fi]
            if len(mem) < 3:
                continue
            L = Lval(mem)
            out["pub"]["M"].append(len(mem)); out["pub"]["L"].append(L); out["pub"]["K"].append(kfrag[fi]); out["pub"]["lev"].append(tau)
            out["mass"]["M"].append(gmass[gi]); out["mass"]["L"].append(L); out["mass"]["K"].append(kfrag[fi]); out["mass"]["lev"].append(tau)
            fl = ret_members[fi]
            rmem = np.concatenate([members[x] for x in fl])
            Lr = Lval(rmem) if len(fl) > 1 else L
            out["retile"]["M"].append(len(rmem)); out["retile"]["L"].append(Lr); out["retile"]["K"].append(kfrag[fi]); out["retile"]["lev"].append(tau)
    return out


# ------------------------------------------------------------------ coverings
def gc_csr(E):
    from fast_cover import csr_from_graph
    G = nx.Graph(); G.add_edges_from(map(tuple, E))
    nodes, idx, A = csr_from_graph(G)
    return A


def box_stats(A, boxes, centres=None, diam=True):
    """Per box: m, diameter, k_hub (max degree), k_centre, k_B (renormalized degree)."""
    n = A.shape[0]
    deg = np.asarray(A.sum(1)).ravel()
    lab = np.empty(n, int)
    for bi, b in enumerate(boxes):
        lab[b] = bi
    Ac = A.tocoo()
    u, v = lab[Ac.row], lab[Ac.col]; m = u != v
    pairs = np.unique(np.sort(np.c_[u[m], v[m]], axis=1), axis=0)
    kB = np.bincount(pairs.ravel(), minlength=len(boxes))
    M = np.array([len(b) for b in boxes]); khub = np.zeros(len(boxes)); np.maximum.at(khub, lab, deg)
    kc = np.array([deg[c] for c in centres]) if centres is not None else khub
    D = np.array([diam_of(A, np.asarray(b)) if len(b) >= 3 else len(b) - 1 for b in boxes]) if diam else None
    return dict(M=M, D=D, khub=khub, kc=kc, kB=kB)


def greedy_cover(A, lb):
    from fast_cover import cover_one
    return cover_one(A, lb)


def memb_cover(A, r):
    from covering import cover_memb, cover_memb_lean
    if r <= 13 and A.shape[0] <= 30000:
        boxes, centres = cover_memb(A, r)
    else:
        boxes, centres = cover_memb_lean(A, r)
    # cover_memb returns boxes in centre order; singleton tail boxes have no centre
    cen = list(centres) + [b[0] for b in boxes[len(centres):]]
    return boxes, cen[:len(boxes)]


def slope(x, y):
    return float(np.polyfit(np.log(np.asarray(x, float)), np.log(np.asarray(y, float)), 1)[0])

"""Table 3 of the paper: the intact-network mass-law fit on hub-centred boxes, the two
null estimators (hub degrees shuffled, and an independent random draw), and the
edge-centred estimator that the paper rejects.

Both nulls keep the boxes, masses and diameters of the correct fit and replace only
the degree regressor:
  shuffled    : a random permutation of the hub degrees across all boxes
  independent : hub degrees drawn with replacement from the same set, independently of the box
Each null is repeated NDRAW times with a fixed seed; the table reports the mean over
draws, with the standard deviation across draws for beta.

Edge-centred boxes: the level-tau box of an edge (u, v) of G_{t-tau} is the copy of
G_tau that replaced it, i.e. every node created inside that edge's expansion together
with u and v. Its hub is its best-connected node, with the current degree, and
L = lambda^tau. Every such box holds exactly N_tau nodes, so mass is constant within a
level while hub degree varies, and the boxes overlap at their endpoints (they do not
tile the network, so the closure argument of Sec. 4.1 does not apply to them).

    python src/table3_nulls.py            # writes results/table3_nulls.json
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from yakubo_fujiki import build_parented, hub_seed, GEN, LAM, KAPPA, DF_THEORY

NDRAW, SEED = 1000, 20260923
DK = np.log(KAPPA) / np.log(LAM)


def boxes(which, t):
    edges, N, parent, gen = build_parented(which, t)
    deg = np.zeros(N, dtype=np.int64)
    for u, v in edges:
        deg[u] += 1; deg[v] += 1
    M, L, K, lev = [], [], [], []
    for tau in range(1, t):
        seed = hub_seed(parent, gen, N, t, tau)
        mass = np.bincount(seed, minlength=N); hubs = np.where(mass > 0)[0]
        m, k = mass[hubs].astype(float), deg[hubs].astype(float)
        keep = (m > 1) & (k > 0)
        M.append(m[keep]); K.append(k[keep]); L.append(np.full(keep.sum(), float(LAM) ** tau))
        lev.append(np.full(keep.sum(), tau))
    return M, L, K, lev


def edge_boxes(which, t):
    """Level-tau edge-centred boxes (mass, L, hub degree) for tau = 1..t-1."""
    ge = GEN[which]
    edges = np.array([[0, 1]], dtype=np.int64); ctr = 2
    elists, eanc = [edges], [np.zeros(1, dtype=np.int64)]
    born_gen, born_edge = [0, 0], [-1, -1]
    for s in range(1, t + 1):
        M = len(edges)
        fresh = np.arange(ctr, ctr + 4 * M, dtype=np.int64).reshape(M, 4); ctr += 4 * M
        born_gen += [s] * (4 * M); born_edge += list(np.repeat(np.arange(M), 4))
        node = np.column_stack([edges, fresh])
        new = np.empty((8 * M, 2), dtype=np.int64); anc = np.empty(8 * M, dtype=np.int64)
        for i, (a, b) in enumerate(ge):
            new[i * M:(i + 1) * M] = node[:, [a, b]]; anc[i * M:(i + 1) * M] = np.arange(M)
        edges = new; elists.append(edges); eanc.append(anc)
    N = ctr
    deg = np.bincount(edges.ravel(), minlength=N).astype(float)
    born_gen, born_edge = np.array(born_gen), np.array(born_edge)
    Ms, Ls, Ks = [], [], []
    for tau in range(1, t):
        cut = t - tau
        idx = np.where(born_gen > cut)[0]
        e = born_edge[idx].copy(); lvl = born_gen[idx] - 1      # index into elists[lvl]
        for l in range(t - 1, cut, -1):                         # walk ancestry down to list `cut`
            m = lvl == l
            e[m] = eanc[l][e[m]]; lvl[m] = l - 1
        E = elists[cut]
        mass = np.bincount(e, minlength=len(E)).astype(float) + 2.0
        k = np.maximum(deg[E[:, 0]], deg[E[:, 1]])
        np.maximum.at(k, e, deg[idx])
        Ms.append(mass); Ks.append(k); Ls.append(np.full(len(E), float(LAM) ** tau))
    return np.concatenate(Ms), np.concatenate(Ls), np.concatenate(Ks)


def ols(M, L, K):
    A = np.column_stack([np.ones(len(M)), np.log(L), np.log(K)])
    y = np.log(M)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ c
    cov = (r @ r) / (len(y) - 3) * np.linalg.inv(A.T @ A)
    return dict(alpha=float(c[1]), beta=float(c[2]), alpha_se=float(np.sqrt(cov[1, 1])),
                beta_se=float(np.sqrt(cov[2, 2])),
                R2=float(1 - (r @ r) / ((y - y.mean()) ** 2).sum()))


def closure(a, b):
    return abs(DF_THEORY - (a + b * DK)) / DF_THEORY * 100


def main(which="GB"):
    rng = np.random.default_rng(SEED)
    out = dict(which=which, ndraw=NDRAW, seed=SEED, dB=float(DF_THEORY), dk=float(DK))
    for t in (4, 5, 6):
        Ml, Ll, Kl, _ = boxes(which, t)
        M, L, K = (np.concatenate(x) for x in (Ml, Ll, Kl))
        f = ols(M, L, K)
        f["beta_within"] = float(np.mean([np.polyfit(np.log(k), np.log(m), 1)[0]
                                          for m, k in zip(Ml, Kl) if len(m) >= 5 and len(set(k)) > 1]))
        f["rhs"] = f["alpha"] + f["beta"] * DK; f["closure_pct"] = closure(f["alpha"], f["beta"])
        row = dict(n_boxes=int(len(M)), correct=f)
        for name, draw in (("shuffled", lambda: rng.permutation(K)),
                           ("independent", lambda: rng.choice(K, len(K), replace=True))):
            ab = np.array([[g["alpha"], g["beta"]] for g in (ols(M, L, draw()) for _ in range(NDRAW))])
            rhs = ab[:, 0] + ab[:, 1] * DK
            row[name] = dict(alpha=float(ab[:, 0].mean()), alpha_sd=float(ab[:, 0].std()),
                             beta=float(ab[:, 1].mean()), beta_sd=float(ab[:, 1].std()),
                             rhs=float(rhs.mean()), closure_pct=float(closure(ab[:, 0].mean(), ab[:, 1].mean())))
        e = ols(*edge_boxes(which, t))
        e["rhs"] = e["alpha"] + e["beta"] * DK; e["closure_pct"] = closure(e["alpha"], e["beta"])
        row["edge_centred"] = e
        out[str(t)] = row
        c, s, i = row["correct"], row["shuffled"], row["independent"]
        print(f"t={t} n={row['n_boxes']}: correct a={c['alpha']:.4f} b={c['beta']:.4f} bw={c['beta_within']:.4f} "
              f"cl={c['closure_pct']:.2f}% | shuffled a={s['alpha']:.4f} b={s['beta']:+.4f}+-{s['beta_sd']:.4f} "
              f"cl={s['closure_pct']:.2f}% | indep a={i['alpha']:.4f} b={i['beta']:+.4f}+-{i['beta_sd']:.4f} "
              f"cl={i['closure_pct']:.2f}% | edge a={e['alpha']:.4f} b={e['beta']:+.4f} R2={e['R2']:.4f} "
              f"cl={e['closure_pct']:.2f}%", flush=True)
    dest = os.environ.get("T3_OUT", os.path.join(HERE, "..", "results", "table3_nulls.json"))
    json.dump(out, open(dest, "w"), indent=1)
    print("wrote", dest)


if __name__ == "__main__":
    main()

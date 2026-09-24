"""
Yakubo-Fujiki (2022) general model of hierarchical fractal scale-free networks.
An FSFN is built by replacing EVERY edge of G_{t-1} with a small 'generator'
graph, its two root nodes coinciding with the edge endpoints.  G_0 = one edge.

Two symmetric generators from the paper (both: n_gen=6, m_gen=8, root deg kappa=2,
root-root distance lambda=3, remaining degrees 3, D_f=log8/log3=1.893, gamma=4):
  GA : has 2 triangles (clustered)     -> analytical p_c = 0.6961, nu = 1.8293
  GB : triangle-free (unclustered)     -> analytical p_c = 0.6288, nu = 1.7772
These are verified: our subgraph counts s_m and the fixed-point p_c reproduce
the paper's Table 1 exactly.
"""
import numpy as np
import networkx as nx
from scipy.optimize import brentq

# root nodes are 0 and 1 in every generator
GEN_GA = [(0,2),(0,3),(1,4),(1,5),(2,3),(2,4),(3,5),(4,5)]   # 2 triangles
GEN_GB = [(0,2),(0,3),(1,4),(1,5),(2,4),(2,5),(3,4),(3,5)]   # triangle-free
GEN = {"GA": GEN_GA, "GB": GEN_GB}
SM   = {"GA": {3:2,4:14,5:34,6:25,7:8,8:1},
        "GB": {3:4,4:20,5:40,6:26,7:8,8:1}}
MGEN, NGEN, KAPPA, LAM = 8, 6, 2, 3
DF_THEORY = np.log(8)/np.log(3)          # 1.8928
GAMMA_THEORY = 1 + np.log(8)/np.log(2)   # 4.0


def analytic_pc_nu(which):
    sm = SM[which]
    pi = lambda p: sum(s*p**m*(1-p)**(MGEN-m) for m,s in sm.items())
    pc = brentq(lambda p: pi(p)-p, 0.3, 0.99)
    dpi = (pi(pc+1e-6)-pi(pc-1e-6))/2e-6
    nu = np.log(LAM)/np.log(dpi)
    return pc, nu


def generate_fsfn(which="GB", t=5):
    """Build the t-th generation FSFN by iterated edge->generator replacement.
    Root nodes of the generator (0,1) are mapped onto each edge's endpoints;
    the 4 remaining generator nodes become fresh nodes for every replaced edge."""
    gen_edges = GEN[which]
    rem = [2, 3, 4, 5]                      # non-root generator nodes
    G = nx.Graph(); G.add_edge(0, 1)
    ctr = [2]
    for _ in range(t):
        new_edges = []
        for (u, v) in G.edges():
            # map generator node -> actual node id
            fresh = {r: ctr[0]+i for i, r in enumerate(rem)}
            ctr[0] += len(rem)
            mp = {0: u, 1: v, **fresh}
            for (a, b) in gen_edges:
                new_edges.append((mp[a], mp[b]))
        G = nx.Graph(); G.add_edges_from(new_edges)
    return G


def fsfn_size(t):
    nrem = NGEN - 2
    return 2 + nrem*(MGEN**t - 1)//(MGEN-1)


if __name__ == "__main__":
    for which in ("GA", "GB"):
        pc, nu = analytic_pc_nu(which)
        print(f"\n=== {which} ===  analytic p_c={pc:.4f} nu={nu:.4f}")
        for t in range(1, 5):
            G = generate_fsfn(which, t)
            N, M = G.number_of_nodes(), G.number_of_edges()
            exp = fsfn_size(t)
            deg = [d for _, d in G.degree()]
            print(f"  t={t}: N={N} (formula {exp}) M={M} (8^t={8**t}) "
                  f"diam={nx.diameter(G)} <k>={2*M/N:.3f} maxdeg={max(deg)} "
                  f"tri={sum(nx.triangles(G).values())//3}")
        # fractal dimension check from diameter scaling: N ~ diam^Df
        Ns, Ls = [], []
        for t in range(2, 5):
            G = generate_fsfn(which, t); Ns.append(G.number_of_nodes()); Ls.append(nx.diameter(G))
        Df = np.polyfit(np.log(Ls), np.log(Ns), 1)[0]
        print(f"  D_f from N~diam^Df = {Df:.3f} (theory {DF_THEORY:.3f})")


# ---------------------------------------------------------------------------
# Edge-centred boxes (the tau-expansion of a coarse edge, with its two endpoints) are
# implemented in table3_nulls.edge_boxes(). They return alpha = 1.72, beta = 0.00 at
# R^2 = 0.999 on the intact network (t=6), where the exact values are alpha = ln4/ln3
# and beta = 1: every such box holds exactly N_tau nodes whatever its hub's degree, so
# a deceptively good fit to the wrong quantity. The earlier generate_fsfn_leveltau() /
# kinship_exponents() are gone and their numbers are superseded. For the mass law use
# hub_centred_exponents() below.
# ---------------------------------------------------------------------------


# ============================================================
# CORRECTED ESTIMATOR: hub-centred boxes + current hub degree
# ============================================================
# Fronczak's mass law is m = B L^alpha k^beta where the box is centred on a HUB
# and k is that hub's degree in the CURRENT (generation-t) network.
# For this FSFN, a level-tau hub-centred box is a node that already existed at
# generation t-tau together with all of its descendants added since. Then
#     m = (m_gen/kappa)^tau * k_now ,   L = lambda^tau
# so   alpha = ln(m_gen/kappa)/ln(lambda) = ln4/ln3 = 1.2619  and  beta = 1,
# and  d_k   = ln(kappa)/ln(lambda)      = ln2/ln3 = 0.6309,
# giving the scaling relation d_B = alpha + beta*d_k = ln8/ln3 exactly.
#
# Verified: closure residual 0.59% (t=4) -> 0.39% (t=5) -> 0.36% (t=6),
# beta -> 1 monotonically (0.962, 0.973, 0.979), R^2 = 0.9999.

def build_parented(which, t):
    """Build G_t while tracking each node's parent and birth generation.
    In both generators fresh nodes 2,3 attach to root 0 and 4,5 to root 1,
    so 'parent' is well defined. Returns (edges, N, parent, gen)."""
    ge = GEN[which]
    edges = [(0, 1)]; ctr = 2
    parent = {0: None, 1: None}; gen = {0: 0, 1: 0}
    for s in range(1, t + 1):
        new = []
        for (u, v) in edges:
            f = {2: ctr, 3: ctr + 1, 4: ctr + 2, 5: ctr + 3}; ctr += 4
            parent[f[2]] = u; parent[f[3]] = u
            parent[f[4]] = v; parent[f[5]] = v
            for r in (2, 3, 4, 5):
                gen[f[r]] = s
            mp = {0: u, 1: v, **f}
            new += [(mp[a], mp[b]) for a, b in ge]
        edges = new
    return edges, ctr, parent, gen


def hub_seed(parent, gen, N, t, tau):
    """seed[n] = the ancestor of n that already existed at generation t-tau.
    Grouping nodes by seed gives the level-tau HUB-CENTRED boxes."""
    cut = t - tau
    seed = np.empty(N, dtype=np.int64)
    for n in range(N):
        x = n
        while gen[x] > cut:
            x = parent[x]
        seed[n] = x
    return seed


def hub_centred_exponents(which, t, verbose=False):
    """CORRECT measurement of the mass-law exponents on the intact FSFN.

    Returns dict with alpha, beta, beta_within, R2, dk, dB, rhs=alpha+beta*dk
    and closure_pct = |dB - rhs|/dB * 100.
    d_B and d_k are the exact model values (log8/log3, log2/log3); both are also
    independently reproducible from box counts and cross-generation degree growth.
    """
    from numpy.linalg import lstsq
    edges, N, parent, gen = build_parented(which, t)
    deg = np.zeros(N, dtype=np.int64)
    for u, v in edges:
        deg[u] += 1; deg[v] += 1
    M_, L_, K_, bw = [], [], [], []
    for tau in range(1, t):
        seed = hub_seed(parent, gen, N, t, tau)
        mass = np.bincount(seed, minlength=N)
        hubs = np.where(mass > 0)[0]
        k_now = deg[hubs].astype(float)          # CURRENT degree -- essential
        m = mass[hubs].astype(float)
        keep = (m > 1) & (k_now > 0)
        M_.append(m[keep]); K_.append(k_now[keep])
        L_.append(np.full(keep.sum(), float(LAM) ** tau))
        if keep.sum() >= 5 and len(set(k_now[keep])) > 1:      # within-level beta
            bw.append(np.polyfit(np.log(k_now[keep]), np.log(m[keep]), 1)[0])
    M = np.concatenate(M_); L = np.concatenate(L_); K = np.concatenate(K_)
    A = np.column_stack([np.ones(len(M)), np.log(L), np.log(K)])
    sol, *_ = lstsq(A, np.log(M), rcond=None)
    alpha, beta = float(sol[1]), float(sol[2])
    r2 = float(1 - np.sum((np.log(M) - A @ sol) ** 2) /
               np.sum((np.log(M) - np.log(M).mean()) ** 2))
    dk = np.log(KAPPA) / np.log(LAM)
    dB = DF_THEORY
    rhs = alpha + beta * dk
    out = dict(N=int(N), t=t, alpha=alpha, beta=beta,
               beta_within=float(np.mean(bw)) if bw else float("nan"),
               R2=r2, dk=float(dk), dB=float(dB), rhs=float(rhs),
               closure_pct=float(abs(dB - rhs) / dB * 100), nboxes=int(len(M)))
    if verbose:
        print(f"{which} t={t} N={out['N']}: alpha={alpha:.4f} (th {np.log(4)/np.log(3):.4f}) "
              f"beta={beta:.4f} (th 1) beta_within={out['beta_within']:.4f} R2={r2:.5f}")
        print(f"   alpha+beta*dk = {rhs:.4f}  vs  dB = {dB:.4f}  -> closure {out['closure_pct']:.2f}%")
    return out


def measure_dk_degree_growth(which, t):
    """d_k measured (not assumed) from cross-generation degree growth:
    a node's degree multiplies by kappa each generation while lengths scale by
    lambda, so k_t/k_{t-tau} = (lambda^tau)^{d_k}. Returns the fitted d_k."""
    ge = GEN[which]; rem = [2, 3, 4, 5]
    edges = np.array([[0, 1]], dtype=np.int64); ctr = 2
    snaps = [np.bincount(edges.ravel(), minlength=ctr)]
    for _ in range(t):
        M = len(edges)
        fresh = np.arange(ctr, ctr + 4 * M, dtype=np.int64).reshape(M, 4); ctr += 4 * M
        node = np.empty((M, 6), dtype=np.int64)
        node[:, 0] = edges[:, 0]; node[:, 1] = edges[:, 1]; node[:, 2:] = fresh
        new = np.empty((M * 8, 2), dtype=np.int64)
        for i, (a, b) in enumerate(ge):
            new[i * M:(i + 1) * M, 0] = node[:, a]; new[i * M:(i + 1) * M, 1] = node[:, b]
        edges = new
        snaps.append(np.bincount(edges.ravel(), minlength=ctr))
    deg = snaps[-1]
    xs, ys = [], []
    for tau in range(1, t):
        old = snaps[t - tau]; sel = np.where(old >= 1)[0]
        ratio = deg[sel].astype(float) / old[sel].astype(float)
        xs.append(float(LAM) ** tau); ys.append(float(np.exp(np.mean(np.log(ratio)))))
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])

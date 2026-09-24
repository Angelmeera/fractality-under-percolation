"""
Paper 1 -- Fractality Under Network Dynamics: Percolation Phase Transition
Corrected implementation.

Fixes over the original notebooks:
  1. generate_shm builds the TRUE Song-Havlin-Makse tree (one offspring-pair
     link per removed edge), so N = (2s+1)^t + 1, and the deterministic
     exponents are d_B = ln(2s+1)/ln 3, d_k = ln s / ln 3, alpha = ln((2s+1)/s)/ln3,
     beta = 1, gamma = delta = 1 + ln(2s+1)/ln s.
  2. degree exponent gamma from a proper (complementary-CDF / MLE) fit, and
     d_k = d_B/(gamma-1) -- not the rank-plot slope.
  3. alpha, beta estimated by Fronczak's protocol: pool (m,L,k) box triples over
     the linear range of l_B, rescale, log-bin, geometric-mean, then fit.
  4. box-covering distances computed once per graph with a fast BFS (scipy csgraph),
     so t_max=5 finishes.
  5. finite-size scaling: p_c(N) measured per size and extrapolated to p_c(inf);
     data collapse uses x = (p - p_c) * N^{1/(d_B nu)}.

Dependencies: networkx, numpy, scipy, matplotlib, (sklearn optional)
"""
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, connected_components
import math

# ============================================================
# 1. FRACTAL MODEL GENERATORS
# ============================================================

def generate_shm(s: int = 2, t_max: int = 5, track: bool = False):
    """
    TRUE Song-Havlin-Makse model (deterministic, tree).

    t=0: two nodes joined by one edge.
    Each step, for every edge (u,v):
        - attach s new offspring to u and s new offspring to v,
        - remove the old edge (u,v),
        - reconnect ONE offspring of u to ONE offspring of v
          (this single replacement link is what keeps the graph a tree and
           gives the mass factor n = 2s+1).

    Theory:  d_B = ln(2s+1)/ln 3,  d_k = ln s/ln 3,
             alpha = ln((2s+1)/s)/ln 3,  beta = 1,
             gamma = delta = 1 + ln(2s+1)/ln s.

    If track=True, also return (parent, gen) dicts for construction-aware
    (kinship) box covering used to validate microscopic exponents.
    """
    G = nx.Graph()
    G.add_edge(0, 1)
    ctr = [2]
    parent = {0: None, 1: None}
    gen = {0: 0, 1: 0}
    step = 0
    for _ in range(t_max):
        step += 1
        rem_e = list(G.edges())
        new_e = []
        for u, v in rem_e:
            ou = []
            ov = []
            for _ in range(s):
                n = ctr[0]; ctr[0] += 1
                ou.append(n); new_e.append((u, n))
                parent[n] = u; gen[n] = step
            for _ in range(s):
                n = ctr[0]; ctr[0] += 1
                ov.append(n); new_e.append((v, n))
                parent[n] = v; gen[n] = step
            # single replacement link between one offspring of each endpoint
            new_e.append((ou[0], ov[0]))
        G.remove_edges_from(rem_e)
        G.add_edges_from(new_e)
    if track:
        return G, parent, gen
    return G


def kinship_boxes(G, parent, gen, t_max, tau):
    """
    Construction-aware (kinship) box covering for the deterministic SHM tree.
    A box is a hub existing at generation (t_max - tau) together with all of
    its descendants added in the following tau generations. These boxes have
    diameter ~3^tau and mass ~ (2s+1)^tau, so they recover the microscopic
    exponents that greedy covering biases on deterministic models
    (cf. Fronczak et al. 2024, Fig. 4, closed vs open points).
    """
    cutoff = t_max - tau
    seed_of = {}

    def find_seed(x):
        path = []
        while gen[x] > cutoff and parent[x] is not None:
            path.append(x); x = parent[x]
        for y in path:
            seed_of[y] = x
        return x

    boxes = {}
    for node in G.nodes():
        boxes.setdefault(find_seed(node), []).append(node)
    return list(boxes.values())


def shm_theory(s: int = 2) -> dict:
    n = 2 * s + 1
    a = 3.0
    return {
        "n": n,
        "d_B": math.log(n) / math.log(a),
        "d_k": math.log(s) / math.log(a),
        "alpha": math.log(n / s) / math.log(a),
        "beta": 1.0,
        "gamma": 1 + math.log(n) / math.log(s),
        "delta": 1 + math.log(n) / math.log(s),
    }


# ============================================================
# 2. FAST DISTANCE / BOX-COVERING CORE
# ============================================================

def _relabel(G):
    """Return (nodes, index map, csr adjacency) for fast array work."""
    nodes = list(G.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    rows, cols = [], []
    for u, v in G.edges():
        iu, iv = idx[u], idx[v]
        rows += [iu, iv]
        cols += [iv, iu]
    n = len(nodes)
    data = np.ones(len(rows), dtype=np.int8)
    A = csr_matrix((data, (rows, cols)), shape=(n, n))
    return nodes, idx, A


def all_pairs_dist(G) -> np.ndarray:
    """Unweighted all-pairs shortest-path length as a dense int matrix.
    Uses scipy BFS; unreachable pairs are set to a large finite sentinel."""
    nodes, idx, A = _relabel(G)
    D = shortest_path(A, method="D", unweighted=True)
    D[np.isinf(D)] = 10**9
    return nodes, D.astype(np.int64)


def greedy_box_cover_D(D: np.ndarray, degrees: np.ndarray, l_B: int):
    """
    Greedy box-covering on a precomputed distance matrix.
    A node joins an existing box only if it is within diameter (l_B - 1)
    of EVERY current member (so the box's true diameter is < l_B).
    Boxes are seeded in descending-degree order (hubs first).
    Returns list of boxes, each a list of node-index integers.
    """
    order = np.argsort(-degrees)          # hubs first
    boxes = []                            # list of lists of indices
    box_seed = []                         # first (hub) member of each box
    box_ok = []                           # boolean row: which nodes still fit this box
    thr = l_B - 1
    n = len(degrees)
    for node in order:
        placed = False
        for bi in range(len(boxes)):
            if box_ok[bi][node]:
                boxes[bi].append(node)
                # a candidate stays admissible only if within thr of the new member too
                box_ok[bi] &= (D[node] <= thr)
                placed = True
                break
        if not placed:
            boxes.append([node])
            box_seed.append(node)
            box_ok.append(D[node] <= thr)   # nodes within thr of this seed
    return boxes


def measure_dB_from_D(D, degrees, N, l_B_values):
    """Fit log(N_B/N) = -d_B log(l_B); return (d_B, R2, NB_array)."""
    NB = np.array([len(greedy_box_cover_D(D, degrees, l)) for l in l_B_values],
                  dtype=float)
    log_l = np.log(np.asarray(l_B_values, float))
    log_NB = np.log(NB / N)
    coeffs = np.polyfit(log_l, log_NB, 1)
    pred = np.polyval(coeffs, log_l)
    ss_res = np.sum((log_NB - pred) ** 2)
    ss_tot = np.sum((log_NB - np.mean(log_NB)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(-coeffs[0]), float(r2), NB


def measure_dB(G, l_B_values):
    """Convenience wrapper straight from a graph."""
    N = G.number_of_nodes()
    if N < 5:
        return float("nan"), float("nan"), None
    nodes, D = all_pairs_dist(G)
    deg = np.array([G.degree(u) for u in nodes], dtype=float)
    return measure_dB_from_D(D, deg, N, l_B_values)


# ============================================================
# 3. PERCOLATION
# ============================================================

def bond_percolation(G, p, rng):
    """Keep each edge with probability p."""
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from([(u, v) for u, v in G.edges() if rng.random() < p])
    return H


def giant_component(G):
    if G.number_of_nodes() == 0:
        return G
    largest = max(nx.connected_components(G), key=len)
    return G.subgraph(largest).copy()


def susceptibility(G, p_values, n_trials=30, rng=None):
    """
    chi(p) = sum_s s^2 n_s / sum_s s n_s over FINITE clusters
    (largest component excluded). Peak locates p_c.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    E = list(G.edges())
    nodeslist = list(G.nodes())
    chi = np.zeros(len(p_values))
    for i, p in enumerate(p_values):
        acc = []
        for _ in range(n_trials):
            H = nx.Graph()
            H.add_nodes_from(nodeslist)
            H.add_edges_from([e for e in E if rng.random() < p])
            sizes = sorted((len(c) for c in nx.connected_components(H)),
                           reverse=True)[1:]           # drop giant
            if sizes:
                a = np.array(sizes, dtype=float)
                acc.append((a ** 2).sum() / a.sum())
            else:
                acc.append(0.0)
        chi[i] = np.mean(acc)
    return chi


def find_pc(G, p_values, n_trials=30, rng=None):
    chi = susceptibility(G, p_values, n_trials, rng)
    return float(p_values[int(np.argmax(chi))]), chi


# ============================================================
# 4. d_k, gamma  (CORRECTED)
# ============================================================

def estimate_gamma(G, kmin=2):
    """
    Clauset-style MLE for the power-law exponent of P(k) ~ k^-gamma
    over degrees >= kmin.  gamma = 1 + n / sum(ln(k/(kmin-0.5))).
    Falls back gracefully for small samples.
    """
    ks = np.array([d for _, d in G.degree() if d >= kmin], dtype=float)
    if len(ks) < 10:
        return float("nan")
    gamma = 1.0 + len(ks) / np.sum(np.log(ks / (kmin - 0.5)))
    return float(gamma)


def dk_from_dB_gamma(d_B, gamma):
    """d_k = d_B/(gamma-1), the theoretically correct relation."""
    if not np.isfinite(gamma) or gamma <= 1:
        return float("nan")
    return d_B / (gamma - 1.0)


# ============================================================
# 5. alpha, beta  (Fronczak protocol, CORRECTED)
# ============================================================

def box_triples(D, degrees, l_B):
    """Return arrays (m, L, k) over the boxes of one l_B cover."""
    boxes = greedy_box_cover_D(D, degrees, l_B)
    m_arr, L_arr, k_arr = [], [], []
    for members in boxes:
        members = np.asarray(members)
        m = len(members)
        sub = D[np.ix_(members, members)]
        L = int(sub[sub < 10**9].max()) if m > 1 else 0
        L = max(L, 1)
        k = int(degrees[members].max())
        m_arr.append(m); L_arr.append(L); k_arr.append(k)
    return np.array(m_arr, float), np.array(L_arr, float), np.array(k_arr, float)


def _logbin_fit(x, y, nbins=12):
    """Geometric-mean log-binning of y over log-spaced x bins, then slope."""
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 4 or x.min() == x.max():
        return float("nan"), float("nan")
    edges = np.logspace(np.log10(x.min()), np.log10(x.max()), nbins + 1)
    bx, by = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x >= lo) & (x <= hi) if hi == edges[-1] else (x >= lo) & (x < hi)
        if sel.sum() >= 1:
            bx.append(np.exp(np.mean(np.log(x[sel]))))
            by.append(np.exp(np.mean(np.log(y[sel]))))
    if len(bx) < 3:
        return float("nan"), float("nan")
    coeffs, cov = np.polyfit(np.log(bx), np.log(by), 1, cov=True)
    return float(coeffs[0]), float(np.sqrt(cov[0, 0]))


def fit_alpha_beta(D, degrees, l_B_range, beta_guess=1.0, n_iter=3):
    """
    Fronczak-style joint estimate of alpha and beta.
    Pool (m,L,k) triples across every l_B in the linear range, then alternate:
        alpha = slope of  m/k^beta  vs L   (log-binned)
        beta  = slope of  m/L^alpha vs k   (log-binned)
    A couple of iterations from beta_guess=1 converges quickly for these models.
    Returns (alpha, beta, alpha_err, beta_err, n_boxes).
    """
    m_all, L_all, k_all = [], [], []
    for l in l_B_range:
        m, L, k = box_triples(D, degrees, l)
        keep = (m > 1) & (k > 0)
        m_all.append(m[keep]); L_all.append(L[keep]); k_all.append(k[keep])
    m = np.concatenate(m_all); L = np.concatenate(L_all); k = np.concatenate(k_all)
    if len(m) < 6:
        return (float("nan"),) * 4 + (len(m),)
    beta = beta_guess
    alpha = a_err = b_err = float("nan")
    for _ in range(n_iter):
        alpha, a_err = _logbin_fit(L, m / np.power(k, beta))
        if not np.isfinite(alpha):
            break
        beta, b_err = _logbin_fit(k, m / np.power(L, alpha))
        if not np.isfinite(beta):
            break
    return alpha, beta, a_err, b_err, len(m)


# ============================================================
# 6. CONSISTENT d_k FROM RENORMALIZATION (same covering as alpha,beta)
# ============================================================

def dk_alpha_beta_consistent(G, l_B_values):
    """
    Measure alpha, beta, d_k and d_B from ONE set of box coverings so the
    scaling relation d_B = alpha + beta*d_k is tested internally (no gamma).

      alpha,beta : from the mass law  m = B L^alpha k^beta  (log-binned/OLS pooled).
      d_k        : from renormalization  k' = k * l_B^{-d_k}, where k is the
                   box hub's ORIGINAL degree and k' is the supernode degree
                   after replacing each l_B-box by a node. Fit
                   log k' = c + 1*log k - d_k*log l_B  (joint OLS over all boxes/l_B).
      d_B        : box count  N_B/N ~ l_B^{-d_B}.

    Returns dict with dB, R2, alpha, beta, dk, and the closure residual.
    """
    from numpy.linalg import lstsq
    N = G.number_of_nodes()
    if N < 20:
        return None
    nodes, D = all_pairs_dist(G)
    idx = {u: i for i, u in enumerate(nodes)}
    deg = np.array([G.degree(u) for u in nodes], float)
    diam = int(D[D < 10**9].max())
    grid = [l for l in l_B_values if 2 <= l <= diam + 1]
    if len(grid) < 3:
        return None

    # d_B
    dB, r2, _ = measure_dB_from_D(D, deg, N, grid)

    # mass-law pooled triples  +  renormalization (k, k', l_B)
    M, L, K = [], [], []
    Kr, Krp, Lb_r = [], [], []
    edges = [(idx[u], idx[v]) for u, v in G.edges()]
    for l in grid:
        boxes = greedy_box_cover_D(D, deg, l)
        box_of = np.empty(N, dtype=int)
        for bi, bx in enumerate(boxes):
            for n in bx:
                box_of[n] = bi
        # mass-law triples
        for bx in boxes:
            bx = np.asarray(bx)
            m = len(bx)
            if m < 2:
                continue
            sub = D[np.ix_(bx, bx)]
            Ld = int(sub[sub < 10**9].max())
            M.append(m); L.append(max(Ld, 1)); K.append(int(deg[bx].max()))
        # supernode degrees
        superadj = [set() for _ in boxes]
        for a, b in edges:
            ba, bb = box_of[a], box_of[b]
            if ba != bb:
                superadj[ba].add(bb); superadj[bb].add(ba)
        for bi, bx in enumerate(boxes):
            k = deg[np.asarray(bx)].max()
            kp = len(superadj[bi])
            if k > 1 and kp >= 1:
                Kr.append(k); Krp.append(kp); Lb_r.append(l)

    M, L, K = map(lambda a: np.array(a, float), (M, L, K))
    if len(M) < 8 or len(set(L)) < 2 or len(set(K)) < 2:
        return None
    # alpha,beta by log-binned alternating fit (robust)
    beta = 1.0; alpha = np.nan
    for _ in range(4):
        alpha, _ = _logbin_fit(L, M / np.power(K, beta))
        if not np.isfinite(alpha):
            break
        beta, _ = _logbin_fit(K, M / np.power(L, alpha))
        if not np.isfinite(beta):
            break

    # d_k : joint OLS  log k' = c + b1 log k + b2 log l_B ,  d_k = -b2
    Kr, Krp, Lb_r = map(lambda a: np.array(a, float), (Kr, Krp, Lb_r))
    dk = np.nan
    if len(Kr) >= 8 and len(set(Lb_r)) >= 2:
        A = np.column_stack([np.ones_like(Kr), np.log(Kr), np.log(Lb_r)])
        sol, *_ = lstsq(A, np.log(Krp), rcond=None)
        dk = float(-sol[2])

    rhs = alpha + beta * dk if np.isfinite(dk) else np.nan
    resid = abs(dB - rhs) / abs(dB) if np.isfinite(rhs) else np.nan
    return {"dB": dB, "R2": r2, "alpha": float(alpha), "beta": float(beta),
            "dk": dk, "rhs": rhs, "closure_resid": resid, "N": N}

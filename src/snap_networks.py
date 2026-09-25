"""
Second and third empirical networks for the fractal-scaling paper.

  * WWW  -- the nd.edu web graph (SNAP `web-NotreDame`, 325,729 nodes,
    1,497,134 directed hyperlinks; Albert, Jeong & Barabasi 1999). This is the
    network in Fronczak et al. (2024) Table 1 (their row: N = 325,728,
    <k> = 4.6, diameter 46, d_B = 4.8, gamma = 2.4, delta = 2.2,
    alpha = 0.68 (0.63), beta = 1.22 (1.22)), so it is a direct external check of
    our implementation on a second real network -- and one whose exponents sit on
    the opposite side of alpha ~ beta ~ 1 from the DBLP backbone.

    Note on <k>: dropping 27,455 self-loops and collapsing reciprocal links gives
    325,729 nodes and 1,090,108 distinct undirected edges, i.e. <k> = 6.69. Their
    4.6 is 1,497,134/325,728 -- the directed edge count per node, i.e. the edge
    list taken as undirected without de-duplication. The two conventions differ,
    but the graph is the same one: the node count and the diameter (46) agree
    exactly, and neither self-loops nor duplicate links change any shortest path,
    so d_B is unaffected.

  * Internet AS-level (SNAP `as20000102`, 6,474 nodes) -- the NEGATIVE CONTROL.
    Song, Havlin & Makse, Nature Physics 2, 275 (2006) report the Internet as
    NON-fractal: the box quantities decay exponentially in l_B rather than as a
    power law, so d_B -> infinity. If our pipeline returns a clean fractal
    dimension here then it is manufacturing power laws and every d_B in the
    paper is suspect. This is the cheapest available check that the method can
    say "no".

WHAT IS NEW HERE RELATIVE TO THE DBLP ANALYSIS
----------------------------------------------
On DBLP we took alpha and beta through the MACROSCOPIC route,

    alpha = ((delta-2)/(delta-1)) d_B ,      beta = (gamma-1)/(delta-1) ,

which is how Fronczak et al. quote their "theoretical" column. On that route
the closure d_B = alpha + beta d_k is an algebraic identity, so printing it
proves nothing (we said so in the paper). Here we add two measurements that do
not go through gamma or delta at all, so that the relation can actually be
TESTED on an empirical network:

  d_k   from Song's renormalisation: cover at l_B, contract each box to a
        supernode, and fit s(l_B) = <k_B>/<k_hub> ~ l_B^(-d_k), where k_B is a
        box's degree in the renormalised network and k_hub the largest degree
        inside it. Uses no degree-distribution exponent.

  alpha, beta  from a JOINT least-squares fit of log m on (log L, log k) over
        every box of every l_B. Uses no box-mass-distribution exponent.

alpha + beta*d_k vs d_B is then a genuine comparison rather than a tautology.
Standard errors and the design condition number are reported alongside, because
the whole point of the alpha/beta story in this paper is that a good-looking
fit can still be a fit to the wrong quantity.

MEMORY NOTE
-----------
fast_cover.ball_index() precomputes the (l_B-1)-ball of EVERY node. On a graph
with d_B ~ 4.8 a radius-8 ball already holds ~2x10^4 nodes, so at N = 3x10^5
that table would need ~10^10 entries. cover_stream() below computes balls in
row blocks and throws each block away, keeping only the surviving `allowed`
sets of the boxes -- memory becomes O(block x ball + sum of allowed), which is
bounded and small. Box output is identical to fast_cover.cover_one(); see
_selftest().
"""
import gzip
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse import csgraph as sp_csgraph, identity
from scipy.sparse.csgraph import dijkstra, connected_components


# ------------------------------------------------------------------ loading
def load_snap_edgelist(path, verbose=True):
    """Read a SNAP '# comment' + 'u<TAB>v' edge list (optionally .gz).

    SNAP web graphs are DIRECTED. Box counting is defined on the undirected
    graph, so we symmetrise; duplicate reciprocal links collapse to one edge,
    which is why 1,497,134 directed hyperlinks (27,455 of them self-loops) give
    1,090,108 undirected edges and <k> = 6.69 rather than 9.2. Self-loops are dropped (they are meaningless
    for shortest-path distance and inflate degree).
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    G = nx.Graph()
    n_lines = n_self = 0
    with opener(path, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line or line[0] == "#":
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = parts[0], parts[1]
            n_lines += 1
            if u == v:
                n_self += 1
                continue
            G.add_edge(u, v)
    if verbose:
        print(f"  read {n_lines} directed lines ({n_self} self-loops dropped) "
              f"-> undirected N={G.number_of_nodes()} M={G.number_of_edges()} "
              f"<k>={2 * G.number_of_edges() / max(G.number_of_nodes(), 1):.2f}")
    return G


def giant(G):
    """Largest connected component, as a fresh graph."""
    comps = list(nx.connected_components(G))
    if not comps:
        return G
    return G.subgraph(max(comps, key=len)).copy()


def kcore_series(G, ks):
    """k-core reductions. For an unweighted graph this plays the role the
    weak-tie threshold plays for weighted DBLP: a principled family of nested
    subgraphs of decreasing size, so d_B can be checked for stability against
    the reduction instead of being quoted at one arbitrary cut."""
    out = []
    for k in ks:
        H = nx.k_core(G, k)
        if H.number_of_nodes() >= 60:
            out.append((k, giant(H)))
    return out


# --------------------------------------------------- streaming box covering
_SPMM_MAX_RADIUS = 10

# Target ball entries per batch (~50 MB at 5 bytes/entry). Keeps a worker's
# peak memory bounded independently of how large the balls turn out to be.
_TARGET_BALL_NNZ = int(__import__('os').environ.get('BALL_NNZ_TARGET', 10_000_000))


def _ai_matrix(A):
    """A + I, boolean, csr -- one multiply by this expands a frontier by one hop
    while keeping what it already had, so k multiplies give the radius-k ball."""
    n = A.shape[0]
    AI = A + identity(n, dtype=np.int8, format="csr")
    AI.data[:] = 1
    return AI.tocsr()


def _balls_for(A, AI, idxs, thr, spmm_max_radius=_SPMM_MAX_RADIUS):
    """Sorted int32 ball arrays for an arbitrary list of source indices.

    Two engines, as in fast_cover.ball_index but on a small batch of sources so
    memory stays O(batch x ball) rather than O(N x ball):
      * boolean frontier expansion (sparse matrix products) at small radius --
        cheapest while balls are still small;
      * scipy BFS with a distance limit at large radius, where the frontier
        expansion keeps re-touching a ball that already spans the graph.
    """
    idxs = np.asarray(idxs, dtype=np.int64)
    rows = idxs.size
    n = A.shape[0]
    if thr <= 0:
        return [np.array([i], dtype=np.int32) for i in idxs]
    if thr <= spmm_max_radius:
        F = csr_matrix((np.ones(rows, np.int8), (np.arange(rows), idxs)),
                       shape=(rows, n))
        for _ in range(thr):
            F = F @ AI
            F.data[:] = 1
        F = F.tocsr()
        F.sort_indices()
        ip, ix = F.indptr, F.indices.astype(np.int32)
        return [ix[ip[i]:ip[i + 1]] for i in range(rows)]
    D = dijkstra(A, directed=False, unweighted=True, indices=idxs, limit=thr)
    near = D <= thr
    return [np.flatnonzero(near[i]).astype(np.int32) for i in range(rows)]


def cover_stream(A, l_B, chunk=2048, verbose=False):
    """Greedy diameter-based box cover with balls computed in row blocks.

    Semantics are identical to fast_cover.cover_one(): nodes are visited in
    decreasing-degree order (which csr_from_graph has already made index order),
    a node joins the OLDEST box that can still accept it, and a box accepts a
    node only if the node lies within l_B - 1 of every current member -- kept as
    the running intersection `allowed` of the members' balls, so every box has
    true diameter < l_B.

    Balls are computed LAZILY and only for nodes still unassigned. That matters
    at large l_B, where a single box can swallow a large fraction of the graph:
    the eager version computed a ball for every node whether or not it was
    already covered, which is most of the cost on a 3x10^5-node graph.
    """
    n = A.shape[0]
    thr = max(l_B - 1, 0)
    # The batch size is ADAPTIVE, and it has to be: ball size varies by orders of
    # magnitude both between networks and within one. On the WWW, nodes are
    # visited in decreasing-degree order, so the first batch is the biggest hubs,
    # whose radius-4 balls hold ~10^5 nodes each -- a fixed 2048-source batch
    # asked for ~10^9 entries and the worker was OOM-killed. We therefore aim at
    # a fixed number of ball ENTRIES per batch (~10^7, i.e. ~50 MB at 5 bytes)
    # and re-derive the batch size from the mean ball size actually observed.
    target_nnz = _TARGET_BALL_NNZ
    batch = 64
    if thr > _SPMM_MAX_RADIUS:
        # scipy BFS returns a dense (batch x N) float64 block; cap it at ~200 MB
        batch = int(max(16, min(2048, 200_000_000 // (8 * max(n, 1)))))
    AI = _ai_matrix(A) if 0 < thr <= _SPMM_MAX_RADIUS else None
    assigned = np.zeros(n, dtype=bool)
    is_seed = np.zeros(n, dtype=bool)
    box_of_seed = np.full(n, -1, dtype=np.int64)
    boxes, allowed = [], []
    cache = {}
    for node in range(n):
        if assigned[node]:
            continue
        ball = cache.pop(node, None)
        if ball is None:
            cache = {}                                # drop stale entries
            idxs = [node]
            j = node + 1
            while len(idxs) < batch and j < n:
                if not assigned[j]:
                    idxs.append(j)
                j += 1
            balls = _balls_for(A, AI, idxs, thr)
            for i, b in zip(idxs, balls):
                cache[i] = b
            mean_ball = max(1.0, float(np.mean([b.size for b in balls])))
            nxt = int(target_nnz // mean_ball)
            if thr > _SPMM_MAX_RADIUS:
                nxt = min(nxt, 200_000_000 // (8 * max(n, 1)))
            # Hard cap on top of the adaptive estimate. Ball size varies by
            # orders of magnitude WITHIN one graph -- on the WWW a peripheral
            # node's radius-5 ball is tiny while a core node's is most of the
            # network. Sizing the next batch from the last batch's mean therefore
            # over-reaches: one batch of peripheral nodes pushed the batch to
            # 4096, the next batch happened to be core nodes, and 4096 x 3x10^5
            # entries OOM-killed the worker at 5.6 GB. The cap assumes the worst
            # case ball is ~N/8, so peak memory is bounded whatever the graph.
            hard_cap = max(16, target_nnz // max(n // 8, 1))
            batch = int(min(max(nxt, 16), hard_cap, 4096))
            ball = cache.pop(node)
            if verbose:
                print(f"      l_B={l_B}: at node {node}/{n}, "
                      f"{len(boxes)} boxes, mean ball {mean_ball:.0f}, "
                      f"next batch {batch}", flush=True)
        seeds_here = ball[is_seed[ball]]
        placed = False
        if seeds_here.size:
            for bi in np.unique(box_of_seed[seeds_here]):
                arr = allowed[bi]
                j = np.searchsorted(arr, node)
                if j < arr.size and arr[j] == node:
                    boxes[bi].append(node)
                    assigned[node] = True
                    new = np.intersect1d(arr, ball, assume_unique=True)
                    allowed[bi] = new[~assigned[new]]
                    placed = True
                    break
        if not placed:
            boxes.append([node])
            assigned[node] = True
            is_seed[node] = True
            box_of_seed[node] = len(boxes) - 1
            # Store only the still-unassigned part of the ball. An assigned node
            # can never join another box, so dropping it changes nothing -- but it
            # is the difference between running and being OOM-killed: on the WWW
            # at l_B = 5 there are ~10^4 boxes whose balls hold ~10^5 nodes each,
            # and keeping them whole asks for >10^9 int32 entries.
            allowed.append(ball[~assigned[ball]].astype(np.int32, copy=False))
    return boxes


def _induced(A, bidx, pos):
    """Local CSR-ish adjacency (list of neighbour arrays) of the box's induced
    subgraph, built by scanning the box's own rows of A.

    Why not `A[bidx][:, bidx]`: that is two fancy-index operations on an
    N x N csr matrix and allocates two intermediates PER BOX. At small l_B a
    3x10^5-node graph has ~10^5 boxes, and this was the dominant cost of the
    whole cover. Scanning rows with a scatter array is O(sum of degrees in the
    box) with no large intermediates.

    `pos` is a reusable int32 scratch array of length N, filled with -1.
    """
    ip, ix = A.indptr, A.indices
    pos[bidx] = np.arange(bidx.size, dtype=np.int32)
    nbrs = []
    for u in bidx:
        loc = pos[ix[ip[u]:ip[u + 1]]]
        nbrs.append(loc[loc >= 0])
    pos[bidx] = -1
    return nbrs


def _bfs_ecc(nbrs, src):
    """(eccentricity of src, farthest vertex) in the local adjacency."""
    n = len(nbrs)
    dist = np.full(n, -1, dtype=np.int32)
    dist[src] = 0
    frontier = [src]
    d = 0
    far = src
    while frontier:
        nxt = []
        d += 1
        for u in frontier:
            for v in nbrs[u]:
                if dist[v] < 0:
                    dist[v] = d
                    far = v
                    nxt.append(v)
        frontier = nxt
    return int(dist.max()), int(far)


def _box_diameter(A, bidx, exact_max=64, n_sweeps=3, pos=None):
    """Diameter of the box's induced subgraph.

    Exact (BFS from every member) for small boxes. For large ones -- a radius-9
    box of the WWW holds ~10^4 nodes, where all-pairs would be a 10^8-entry
    dense matrix -- the standard double-sweep lower bound, which is exact on
    trees and off by at most one on essentially every real graph.

    A greedy box is guaranteed to have diameter < l_B in the FULL graph, but the
    induced subgraph can be disconnected, because the connecting paths may leave
    the box. For boxes of up to exact_max nodes we take the maximum over
    reachable pairs, the same convention as box_triples_greedy(); above that the
    double sweep starts from the seed and so only explores the seed's own
    component (12 of 3,229 fitted boxes at t = 5 on the FSFN; it moves the
    direct alpha by about 0.02).
    """
    m = bidx.size
    if m < 2:
        return 0
    if m == 2:
        return 1
    if pos is None:
        pos = np.full(A.shape[0], -1, dtype=np.int32)
    nbrs = _induced(A, bidx, pos)
    if m <= exact_max:
        best = 1
        for s in range(m):
            e, _ = _bfs_ecc(nbrs, s)
            if e > best:
                best = e
        return best
    best, src = 1, 0
    for _ in range(n_sweeps):
        e, far = _bfs_ecc(nbrs, src)
        best = max(best, e)
        src = far
    return best


def box_records(A, deg, boxes, exact_diam_max=64):
    """Per box: (mass m, diameter L, hub degree k_hub, renormalised degree k_B).

    k_B is the box's degree in the RENORMALISED network -- boxes are supernodes,
    linked if any edge runs between them. It is what Song's renormalisation
    needs, and it gives d_k without touching the degree distribution. Computed
    as P^T A P with P the box-indicator matrix, so it costs one sparse product
    rather than a Python loop over edges.
    """
    n = A.shape[0]
    NB = len(boxes)
    label = np.empty(n, dtype=np.int64)
    for bi, b in enumerate(boxes):
        label[np.asarray(b, dtype=np.int64)] = bi
    P = csr_matrix((np.ones(n, np.int8), (np.arange(n), label)), shape=(n, NB))
    # Association matters, and by a lot. Python evaluates P.T @ A @ P as
    # (P.T @ A) @ P, and P.T @ A has one entry per (box, neighbour-of-a-member)
    # pair: at l_B = 7 on the WWW a single box holds ~2x10^5 nodes whose
    # neighbourhood is most of the graph, so that intermediate approached
    # NB x N entries and the worker was OOM-killed at 5.8 GB. Multiplying
    # A @ P first bounds the intermediate by nnz(A).
    R = (P.T.tocsr() @ (A @ P)).tocsr()
    R.setdiag(0)
    R.eliminate_zeros()
    R.data[:] = 1
    kB = np.asarray(R.sum(axis=1)).ravel().astype(float)
    pos = np.full(n, -1, dtype=np.int32)
    recs = []
    for bi, b in enumerate(boxes):
        bidx = np.asarray(b, dtype=np.int32)
        recs.append((int(bidx.size),
                     max(_box_diameter(A, bidx, exact_diam_max, pos=pos), 1),
                     float(deg[bidx].max()),
                     float(kB[bi]),
                     int(_box_connected(A, bidx))))
    return recs


def _box_connected(A, bidx):
    """Is the subgraph induced on this box connected?

    A greedy box is a set of nodes pairwise within l_B - 1 of one another IN THE
    WHOLE GRAPH; nothing in the covering constraint makes the induced subgraph
    connected, and on hierarchical networks most boxes above l_B = 2 are not.
    That matters because `_box_diameter` returns the largest distance between a
    REACHABLE pair, which is not a diameter, and because a disconnected box has
    mass drawn from parts of the network its diameter does not span. Callers can
    use this flag to refit on connected boxes only.
    """
    if bidx.size <= 1:
        return True
    sub = A[np.ix_(bidx, bidx)]
    ncomp, _ = sp_csgraph.connected_components(sub, directed=False)
    return ncomp == 1


def _worker(args):
    A, l_B, deg, chunk, exact_diam_max = args
    boxes = cover_stream(A, l_B, chunk=chunk)
    return l_B, len(boxes), box_records(A, deg, boxes, exact_diam_max)


def cover_all(G, l_B_values, n_jobs=None, chunk=512, exact_diam_max=64,
              verbose=True):
    """Cover G at every l_B and return per-l_B box records.

    Returns (per_l, N, deg) where per_l[l] = (N_B, [(m, L, k_hub, k_B), ...]).
    """
    import os
    from concurrent.futures import ProcessPoolExecutor
    from fast_cover import csr_from_graph
    nodes, idx, A = csr_from_graph(G)
    N = len(nodes)
    deg = np.asarray(A.getnnz(axis=1), dtype=float)
    l_B_values = list(l_B_values)
    if n_jobs is None:
        n_jobs = max(1, min(len(l_B_values), (os.cpu_count() or 2)))
    payload = [(A, l, deg, chunk, exact_diam_max) for l in l_B_values]
    per_l = {}
    if n_jobs == 1:
        for p in payload:
            l, NB, rec = _worker(p)
            per_l[l] = (NB, rec)
            if verbose:
                print(f"    l_B={l:3d}: N_B={NB:7d}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for l, NB, rec in ex.map(_worker, payload):
                per_l[l] = (NB, rec)
                if verbose:
                    print(f"    l_B={l:3d}: N_B={NB:7d}", flush=True)
    return per_l, N, deg


# ------------------------------------------------------------------- fitting
def _linfit(x, y):
    """Least squares y = a x + b; returns (a, b, R2, sse)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    a, b = np.polyfit(x, y, 1)
    pred = a * x + b
    sse = float(np.sum((y - pred) ** 2))
    ss = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - sse / ss) if ss > 0 else float("nan")
    return float(a), float(b), r2, sse


def powerlaw_vs_exponential(x, y):
    """Discriminate y ~ x^-e (fractal) from y ~ exp(-x/l0) (non-fractal).

    This is Song, Havlin & Makse's own criterion (Nature 2005 Fig. 1; Nature
    Physics 2006 Fig. 2a,b): fractal networks give a straight line for
    N_B(l_B)/N on log-log axes, non-fractal ones on log-linear axes. Both fits
    have two parameters, so comparing residual sums of squares is a fair
    comparison; we report dAIC = n ln(SSE_pl / SSE_exp), negative meaning the
    power law wins.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    keep = (x > 0) & (y > 0)
    x, y = x[keep], y[keep]
    if x.size < 3:
        return dict(n=int(x.size), verdict="too few points")
    e_pl, c_pl, r2_pl, sse_pl = _linfit(np.log(x), np.log(y))
    e_ex, c_ex, r2_ex, sse_ex = _linfit(x, np.log(y))
    n = x.size
    daic = float(n * np.log(max(sse_pl, 1e-300) / max(sse_ex, 1e-300)))
    return dict(n=int(n),
                exponent=float(-e_pl), R2_powerlaw=r2_pl, sse_powerlaw=sse_pl,
                decay_length=float(-1.0 / e_ex) if e_ex != 0 else float("inf"),
                R2_exponential=r2_ex, sse_exponential=sse_ex,
                dAIC_pl_minus_exp=daic,
                verdict=("power law" if sse_pl < sse_ex else "exponential"))


def joint_mass_law(M, L, K, min_mass=3, mask=None):
    """Joint OLS of log m = log B + alpha log L + beta log k over all boxes.

    This is the DIRECT route: it uses neither gamma nor delta, so
    alpha + beta*d_k = d_B becomes a testable statement on a real network
    instead of an identity. Standard errors come from the usual OLS covariance,
    and we report the condition number of the design because the recurring
    lesson of this project is that a high R^2 does not mean the right quantity
    was fitted.
    """
    M = np.asarray(M, float); L = np.asarray(L, float); K = np.asarray(K, float)
    keep = (M >= min_mass) & (L >= 1) & (K >= 1)
    if mask is not None:
        keep &= np.asarray(mask, bool)
    M, L, K = M[keep], L[keep], K[keep]
    if M.size < 20:
        return dict(n=int(M.size), alpha=float("nan"), beta=float("nan"))
    X = np.column_stack([np.ones(M.size), np.log(L), np.log(K)])
    y = np.log(M)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = max(M.size - 3, 1)
    s2 = float(resid @ resid) / dof
    XtXinv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(XtXinv) * s2)
    ss = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - float(resid @ resid) / ss) if ss > 0 else float("nan")
    return dict(n=int(M.size), logB=float(coef[0]),
                alpha=float(coef[1]), beta=float(coef[2]),
                alpha_se=float(se[1]), beta_se=float(se[2]),
                R2=r2, cond=float(np.linalg.cond(X)))


def _ccdf_exponent(vals, vmin=None):
    """Exponent q of P(X >= x) ~ x^-(q-1), i.e. the index of P(x) ~ x^-q.
    Same estimator as real_network._ccdf_exponent (CCDF rather than a histogram
    or a naive MLE, because the box-mass distributions here are spiked)."""
    v = np.sort(np.asarray(vals, float))
    v = v[v > 0]
    if vmin is not None:
        v = v[v >= vmin]
    if len(v) < 20:
        return float("nan")
    u = np.unique(v)
    cc = np.array([(v >= x).mean() for x in u])
    m = cc > 0
    if m.sum() < 5:
        return float("nan")
    return float(1 - np.polyfit(np.log(u[m]), np.log(cc[m]), 1)[0])


def _tail_exponent_mle(vals, vmin=1.0, n_boot=200, seed=0):
    """Hill/Clauset maximum-likelihood tail exponent, with a bootstrap interval.

    `_ccdf_exponent` fits a straight line to the empirical CCDF by unweighted
    least squares. Clauset, Shalizi and Newman warn against exactly that: the
    points of a CCDF are not independent -- "adjacent values of the CDF are
    strongly correlated" -- so the regression standard error "is typically a
    gross underestimate because of the failure to account for the
    correlations". On the short grids used here the fit can also rest on very
    few distinct support points (one, at l_B = 2 on the FSFN).

    This is the continuous MLE, exponent = 1 + n / sum(ln(x/vmin)), with a
    non-parametric bootstrap over boxes for the interval. It is reported
    alongside the CCDF estimate rather than replacing it, so that the two can
    be compared on the networks where the answer is known exactly.
    """
    v = np.asarray(vals, float)
    v = v[np.isfinite(v) & (v >= vmin) & (v > 0)]
    n = v.size
    if n < 20:
        return dict(n=int(n), exponent=float("nan"),
                    ci=(float("nan"), float("nan")), n_distinct=int(np.unique(v).size))
    def _mle(x):
        ssum = float(np.log(x / vmin).sum())
        return float("nan") if ssum <= 0 else 1.0 + x.size / ssum
    est = _mle(v)
    rng = np.random.default_rng(seed)
    disc = _discrete_mle(v, vmin)
    boot = np.array([_mle(v[rng.integers(0, n, n)]) for _ in range(n_boot)], float)
    boot = boot[np.isfinite(boot)]
    lo, hi = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) \
        if boot.size else (float("nan"), float("nan"))
    return dict(n=int(n), exponent=float(est), ci=(lo, hi),
                exponent_discrete=disc,
                n_distinct=int(np.unique(v).size),
                sd_boot=float(np.std(boot)) if boot.size else float("nan"))


def _discrete_mle(v, vmin):
    """MLE for a DISCRETE power law, argmax of -n ln zeta(a, vmin) - a sum ln x.

    The continuous MLE is badly biased when the support is coarse, which is
    exactly the FSFN case: degrees take the values 3 kappa^n (and kappa^t for the
    two initial nodes), so a t = 4 network has five distinct degrees above
    vmin = 2. The paper quotes this estimator with k_min = vmin = 2; with
    k_min = 3 it overshoots instead. Clauset, Shalizi and Newman give
    the discrete likelihood; we maximise it on a grid and refine.
    """
    try:
        from scipy.special import zeta as _zeta
        from scipy.optimize import minimize_scalar as _mins
    except Exception:
        return float("nan")
    x = np.asarray(v, float)
    x = x[x >= vmin]
    if x.size < 20 or vmin < 1:
        return float("nan")
    slog = float(np.log(x).sum()); n = x.size
    def nll(a):
        z = _zeta(a, vmin)
        if not np.isfinite(z) or z <= 0:
            return 1e18
        return n * np.log(z) + a * slog
    r = _mins(nll, bounds=(1.01, 12.0), method="bounded")
    return float(r.x) if r.success else float("nan")


# -------------------------------------------------- choosing the l_B grid
def geometric_grid(lam, diameter, tau_max=12, l_max=None):
    """The box sizes commensurate with a scale factor lam: l_B = round(lam^tau)+1.

    A hierarchical fractal network built by replacing each edge with a generator
    whose roots are at distance lam is exactly self-similar under coarse-graining
    at l_B = lam^tau + 1 (on G^B a level-tau edge unit has diameter lam^tau; on
    G^A it is 29 and 89 at tau = 3, 4, and the covering
    rule admits boxes of diameter < l_B). Sampling l_B *between* those values
    mixes two levels and puts a systematic sawtooth into N_B(l_B) -- which is
    why an arbitrary grid loses several per cent of d_B (see grid_scan).
    """
    if l_max is None:
        # Beyond about half the diameter the cover collapses to a handful of
        # boxes, which adds no information to the fit and costs the most to
        # compute, so the grid stops there.
        l_max = max(4, diameter // 2 + 1)
    out = []
    for tau in range(tau_max + 1):
        l = int(round(lam ** tau)) + 1
        if l < 2:
            continue
        if l > min(diameter + 1, l_max):
            break
        if l not in out:
            out.append(l)
    return tuple(out)


def grid_scan(G, lambdas=(1.6, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0), n_jobs=None,
              chunk=512, exact_diam_max=64, min_points=4, l_max=None,
              verbose=True):
    """Pick the l_B grid by fitting quality instead of by hand.

    Motivation, measured on the FSFN (GB, t = 5, N = 18,726, exact d_B = 1.8928,
    d_k = 0.6309):

        grid                               d_B      err     d_k      err   R2(N_B)
        lam = 1.6 (2,3,4,5,8,11,18,28)     1.8075   4.5 %   0.5668  10.2 %  0.9926
        lam = 2.0 (2,3,5,9,17)             1.6867  10.9 %   0.4665  26.1 %  0.9816
        lam = 2.5 (2,3,7,17)               1.6749  11.5 %   0.4297  31.9 %  0.9703
        lam = 3.0 (2,4,10,28)  <- chosen   1.8632   1.6 %   0.6240   1.1 %  0.9998
    (canonical cover; results/fsfn5_scan.json)

    So the grid is not a cosmetic choice: a grid commensurate with the network's
    scale factor recovers both exponents to ~1 %, and an arbitrary one is out by
    5-12 % in d_B, and more in d_k, with a visibly worse fit. On a deterministic model we know lam; on a
    real network we do not, so we scan lam and keep the grid whose N_B(l_B) power
    law fits best. The scan is reported in full so the reader can see how much
    the answer moves -- silently quoting the best grid would be cheating.
    """
    diam = _approx_diameter(G)
    rows = []
    seen = {}
    grids = {}
    for lam in lambdas:
        g = geometric_grid(lam, diam, l_max=l_max)
        if len(g) < min_points:
            if verbose:
                print(f"  lam={lam:4.2f}: grid {g} has <{min_points} points "
                      f"-- skipped")
            continue
        grids[float(lam)] = g
    # Candidate grids overlap heavily, so cover each DISTINCT l_B once and reuse
    # it. On the WWW this turns 15 covers into 6, and the largest l_B is the
    # expensive one.
    union = sorted({l for g in grids.values() for l in g})
    if not union:
        return rows, None
    if verbose:
        print(f"  covering l_B in {tuple(union)} once each "
              f"(grids: {sorted(set(grids.values()))})", flush=True)
    cache, N, _deg = cover_all(G, union, n_jobs=n_jobs, chunk=chunk,
                               exact_diam_max=exact_diam_max, verbose=verbose)
    for lam, grid in grids.items():
        if grid in seen:
            rows.append({**seen[grid], "lam": float(lam)})
            continue
        per_l = {l: cache[l] for l in grid if l in cache}
        ls = sorted(per_l)
        lb = np.array(ls, float)
        NB = np.array([per_l[l][0] for l in ls], float)
        keep = NB > 1
        fB = powerlaw_vs_exponential(lb[keep], NB[keep] / N)
        s = []
        for l in ls:
            recs = per_l[l][1]
            kb = np.array([r[3] for r in recs], float)
            kh = np.array([r[2] for r in recs], float)
            s.append(kb.mean() / kh.mean() if kh.size and kh.mean() > 0 else np.nan)
        s = np.array(s, float)
        okk = keep & np.isfinite(s) & (s > 0)
        fk = powerlaw_vs_exponential(lb[okk], s[okk]) if okk.sum() >= 3 else {}
        rec = dict(lam=float(lam), grid=list(grid), n_points=int(keep.sum()),
                   dB=fB.get("exponent", float("nan")),
                   dB_R2=fB.get("R2_powerlaw", float("nan")),
                   dB_verdict=fB.get("verdict"),
                   dk_renorm=fk.get("exponent", float("nan")),
                   dk_R2=fk.get("R2_powerlaw", float("nan")),
                   NB=NB[keep].tolist(), s=[float(x) for x in s])
        seen[grid] = rec
        rows.append(rec)
        if verbose:
            print(f"  lam={lam:4.2f}: grid={grid}  d_B={rec['dB']:.4f} "
                  f"(R2={rec['dB_R2']:.4f})  d_k={rec['dk_renorm']:.4f} "
                  f"(R2={rec['dk_R2']:.4f})", flush=True)
    best = max((r for r in rows if np.isfinite(r["dB_R2"])),
               key=lambda r: r["dB_R2"], default=None)
    if verbose and best:
        print(f"  -> best R2 at lam={best['lam']}, grid={tuple(best['grid'])}")
    return rows, best, cache


def _approx_diameter(G, n_probe=8, seed=0):
    """Double-sweep diameter lower bound, repeated from a few random starts.
    Exact-enough to decide how far the l_B grid can run."""
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    best = 0
    for _ in range(min(n_probe, len(nodes))):
        src = nodes[int(rng.integers(len(nodes)))]
        for _ in range(2):
            d = nx.single_source_shortest_path_length(G, src)
            src, ecc = max(d.items(), key=lambda kv: kv[1])
            best = max(best, ecc)
    return int(best)


# ------------------------------------------------------------------- report
def analyse(G, name, l_B_values, n_jobs=None, chunk=512, exact_diam_max=64,
            verbose=True, cover=None):
    """Everything for one network: the fractality test, d_B, gamma, delta, the
    renormalisation d_k, the direct (alpha, beta), the macroscopic (alpha, beta)
    for comparison with Fronczak et al.'s table, and both closure figures."""
    if cover is not None:
        per_l = {l: cover[l] for l in l_B_values if l in cover}
        N = G.number_of_nodes()
    else:
        per_l, N, _deg = cover_all(G, l_B_values, n_jobs=n_jobs, chunk=chunk,
                                   exact_diam_max=exact_diam_max,
                                   verbose=verbose)
    ls = sorted(per_l)
    NB = np.array([per_l[l][0] for l in ls], float)
    lb = np.array(ls, float)

    # ---- macroscopic: box dimension, and does a power law even fit?
    keep = NB > 1
    frac = powerlaw_vs_exponential(lb[keep], NB[keep] / N)
    dB = frac.get("exponent", float("nan"))

    # ---- renormalisation: d_k without the degree distribution
    s_of_l, kB_mean, khub_mean = [], [], []
    for l in ls:
        recs = per_l[l][1]
        if not recs:
            s_of_l.append(np.nan); kB_mean.append(np.nan)
            khub_mean.append(np.nan)
            continue
        kb = np.array([r[3] for r in recs], float)
        kh = np.array([r[2] for r in recs], float)
        kB_mean.append(float(kb.mean()))
        khub_mean.append(float(kh.mean()))
        s_of_l.append(float(kb.mean() / kh.mean()) if kh.mean() > 0 else np.nan)
    s_arr = np.array(s_of_l, float)
    ok = np.isfinite(s_arr) & (s_arr > 0) & (NB > 1)
    dk_fit = powerlaw_vs_exponential(lb[ok], s_arr[ok]) if ok.sum() >= 3 else {}
    dk_renorm = dk_fit.get("exponent", float("nan"))

    # ---- pooled box records for the mass law
    M = np.concatenate([[r[0] for r in per_l[l][1]] for l in ls]) if ls else np.array([])
    L = np.concatenate([[r[1] for r in per_l[l][1]] for l in ls]) if ls else np.array([])
    Kh = np.concatenate([[r[2] for r in per_l[l][1]] for l in ls]) if ls else np.array([])
    CN = (np.concatenate([[bool(r[4]) if len(r) > 4 else True
                           for r in per_l[l][1]] for l in ls])
          if ls else np.array([], bool))
    direct = joint_mass_law(M, L, Kh)
    # The same fit restricted to internally CONNECTED boxes. About half of the
    # greedy boxes above l_B = 2 are not connected, and for those "the diameter
    # of the box" is not defined; the restricted fit is the defensible one and
    # the difference between the two is large, so both are reported.
    direct_connected = joint_mass_law(M, L, Kh, mask=CN)
    disconnected = {}
    for l in ls:
        recs = per_l[l][1]
        if not recs:
            continue
        flags = [bool(r[4]) if len(r) > 4 else True for r in recs]
        nd = sum(1 for f in flags if not f)
        fitted = [(r, f) for r, f in zip(recs, flags) if r[0] >= 3]
        ndf = sum(1 for r, f in fitted if not f)
        disconnected[str(int(l))] = dict(
            n_boxes=len(recs), n_disconnected=int(nd),
            frac=float(nd) / len(recs),
            n_fitted=len(fitted),
            frac_fitted=(float(ndf) / len(fitted)) if fitted else float("nan"))

    # ---- gamma, delta and the macroscopic (derived) exponents
    gamma = _ccdf_exponent([d for _, d in G.degree()], vmin=2)
    deltas = []
    delta_support = []
    delta_mle = []
    for l in ls:
        recs = per_l[l][1]
        if not recs:
            continue
        mm = np.array([r[0] for r in recs], float)
        mu = mm / mm.mean()
        tail = mu[mu >= 1.0]
        d_l = _ccdf_exponent(tail) if tail.size >= 25 else float("nan")
        if np.isfinite(d_l):
            deltas.append(float(d_l))
            # how many distinct CCDF support points that slope rests on
            delta_support.append(int(np.unique(tail).size))
            delta_mle.append(_tail_exponent_mle(tail))
    delta = float(np.median(deltas)) if deltas else float("nan")
    dk_gamma = dB / (gamma - 1) if np.isfinite(gamma) and gamma > 1 else float("nan")
    alpha_mac = (((delta - 2) / (delta - 1)) * dB
                 if np.isfinite(delta) and delta > 1 else float("nan"))
    beta_mac = ((gamma - 1) / (delta - 1)
                if np.isfinite(delta) and delta > 1 and np.isfinite(gamma)
                else float("nan"))

    # ---- the two closures: one is a real test, one is an identity
    rhs_direct = direct["alpha"] + direct["beta"] * dk_renorm
    rhs_mac = alpha_mac + beta_mac * dk_gamma
    out = dict(
        name=name, N=N, M_edges=G.number_of_edges(),
        mean_degree=2 * G.number_of_edges() / N,
        l_B=lb[keep].tolist(), NB=NB[keep].tolist(),
        fractality=frac, dB=dB, dB_R2=frac.get("R2_powerlaw", float("nan")),
        gamma=gamma, delta=delta, delta_per_lB=deltas,
        n_delta_levels=len(deltas), delta_support_points=delta_support,
        delta_mle=delta_mle,
        gamma_mle=_tail_exponent_mle(
            np.array([d for _, d in G.degree()], float), vmin=2.0),
        direct_connected=direct_connected, disconnected_by_lB=disconnected,
        s_of_l=[float(x) for x in s_arr], kB_mean=kB_mean, khub_mean=khub_mean,
        dk_renorm=dk_renorm, dk_renorm_fit=dk_fit, dk_from_gamma=dk_gamma,
        direct=direct, alpha_macroscopic=alpha_mac, beta_macroscopic=beta_mac,
        rhs_direct=float(rhs_direct), rhs_macroscopic=float(rhs_mac),
        closure_direct_pct=float(abs(dB - rhs_direct) / abs(dB) * 100)
        if np.isfinite(rhs_direct) and dB else float("nan"),
        closure_macroscopic_pct=float(abs(dB - rhs_mac) / abs(dB) * 100)
        if np.isfinite(rhs_mac) and dB else float("nan"),
        n_boxes_fitted=direct.get("n", 0))
    if verbose:
        print(f"[{name}] N={N} M={out['M_edges']} <k>={out['mean_degree']:.2f}")
        print(f"  fractality: power law R2={frac.get('R2_powerlaw', float('nan')):.4f} "
              f"vs exponential R2={frac.get('R2_exponential', float('nan')):.4f} "
              f"-> {frac.get('verdict')} (dAIC={frac.get('dAIC_pl_minus_exp', float('nan')):.1f})")
        print(f"  d_B={dB:.3f}   gamma={gamma:.2f}   delta={delta:.2f}"
              f"  (per-l_B {['%.2f' % x for x in deltas]})")
        print(f"  d_k: renormalisation={dk_renorm:.3f} "
              f"[{dk_fit.get('verdict')}, R2={dk_fit.get('R2_powerlaw', float('nan')):.3f}]"
              f"   from gamma={dk_gamma:.3f}")
        print(f"  direct  alpha={direct.get('alpha', float('nan')):.3f}"
              f"+-{direct.get('alpha_se', float('nan')):.3f}  "
              f"beta={direct.get('beta', float('nan')):.3f}"
              f"+-{direct.get('beta_se', float('nan')):.3f}  "
              f"R2={direct.get('R2', float('nan')):.3f} cond={direct.get('cond', float('nan')):.1f}"
              f" on {direct.get('n', 0)} boxes")
        print(f"  macroscopic alpha={alpha_mac:.3f}  beta={beta_mac:.3f}")
        print(f"  CLOSURE  direct: {direct.get('alpha', float('nan')):.3f}"
              f"+{direct.get('beta', float('nan')):.3f}*{dk_renorm:.3f}"
              f"={rhs_direct:.3f} vs d_B={dB:.3f} -> "
              f"{out['closure_direct_pct']:.1f}%   [a real test]")
        print(f"  CLOSURE  macroscopic: {rhs_mac:.3f} vs {dB:.3f} -> "
              f"{out['closure_macroscopic_pct']:.1f}%   [identity, not a test]")
    return out


# ------------------------------------------------------------------ selftest
def _selftest():
    """cover_stream must reproduce fast_cover.cover_one box-for-box, and the
    renormalised degree must match a brute-force count."""
    from fast_cover import csr_from_graph, cover_one
    tests = [("BA n=1200", nx.barabasi_albert_graph(1200, 2, seed=3)),
             ("WS n=900", nx.watts_strogatz_graph(900, 4, 0.03, seed=4)),
             ("grid 30x30", nx.convert_node_labels_to_integers(
                 nx.grid_2d_graph(30, 30)))]
    ok = True
    for name, G in tests:
        nodes, idx, A = csr_from_graph(G)
        deg = np.asarray(A.getnnz(axis=1), dtype=float)
        for l in (2, 3, 4, 6):
            b1 = cover_stream(A, l, chunk=64)
            b2 = cover_one(A, l)
            same = ([sorted(b) for b in b1] == [sorted(b) for b in b2])
            if not same:
                ok = False
            print(f"  {'OK ' if same else 'DIFF'} {name:12s} l_B={l}: "
                  f"stream={len(b1):5d} ref={len(b2):5d}")
        # renormalised degree against a brute-force edge scan
        boxes = cover_stream(A, 3, chunk=64)
        recs = box_records(A, deg, boxes)
        label = {}
        for bi, b in enumerate(boxes):
            for u in b:
                label[u] = bi
        brute = [set() for _ in boxes]
        Ac = A.tocoo()
        for u, v in zip(Ac.row, Ac.col):
            bu, bv = label[u], label[v]
            if bu != bv:
                brute[bu].add(bv)
        good = all(abs(recs[i][3] - len(brute[i])) < 1e-9 for i in range(len(boxes)))
        if not good:
            ok = False
        print(f"  {'OK ' if good else 'DIFF'} {name:12s} renormalised degree "
              f"vs brute force")
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    _selftest()

"""
Fast box covering for large real networks.

Why the previous version was slow: the greedy cover issued one pure-Python
NetworkX BFS per admitted node -- roughly N BFS calls per box size, so ~200k
interpreted BFS traversals for a 35k-node network covered at six values of l_B.

Two changes here:
  1. All (l_B-1)-balls are computed in BULK with scipy's C implementation
     (`dijkstra(..., unweighted=True, limit=thr)`) over chunks of source nodes,
     so the traversal cost moves out of Python entirely. Chunking keeps the
     dense intermediate bounded (chunk x N floats), independent of N^2.
  2. Different l_B values are independent, so they are covered in PARALLEL
     across processes.

Box counts are identical to the reference implementation -- verified in
`_selftest()` against greedy_box_cover_sparse on fractal, scale-free and
small-world graphs.
"""
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def csr_from_graph(G):
    """Return (nodes, index_map, csr adjacency) with nodes ordered by DECREASING
    degree, so index order is already the greedy seeding order.

    Ties are broken on str(node), NOT on the graph's iteration order. This is not
    cosmetic: with string node labels (author names, page ids) the iteration order
    of a set depends on PYTHONHASHSEED, which is randomised per process, so
    `sorted(..., key=-degree)` alone made the greedy cover differ between runs of
    the same script (on the DBLP w>=25 backbone, N_B(l_B=2) came out 24,935 in one
    process and 24,962 in another). Every box count in the paper must be
    reproducible, so the order is made canonical here.
    """
    nodes = sorted(G.nodes(), key=lambda n: (-G.degree(n), str(n)))
    idx = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)
    if G.number_of_edges() == 0:
        return nodes, idx, csr_matrix((n, n), dtype=np.int8)
    r, c = [], []
    for u, v in G.edges():
        iu, iv = idx[u], idx[v]
        r += [iu, iv]
        c += [iv, iu]
    A = csr_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(n, n))
    return nodes, idx, A


def ball_index(A, thr, chunk=512, progress=None, spmm_max_radius=12):
    """balls[i] = sorted array of nodes within `thr` hops of i.

    Two engines, picked by radius (benchmarked on an 18.7k-node fractal network):
      * boolean sparse matrix products -- frontier expansion for ALL sources at
        once. Much faster at small radius (0.24 s vs 1.62 s at thr=9) because
        there is no dense intermediate and no per-source Python overhead.
      * chunked scipy BFS -- wins at large radius, where the ball matrix would
        otherwise blow up (at thr=27 the balls already hold 2.1e7 entries).
    """
    n = A.shape[0]
    if thr <= 0:
        return [np.array([i], dtype=np.int32) for i in range(n)]
    if thr <= spmm_max_radius:
        from scipy.sparse import identity
        I = identity(n, dtype=np.int8, format="csr")
        AI = (A + I); AI.data[:] = 1
        B = I.copy(); F = I.copy()
        for _ in range(thr):
            F = F @ AI; F.data[:] = 1
            B = B + F; B.data[:] = 1
        B = B.tocsr(); B.sort_indices()
        ip, ix = B.indptr, B.indices.astype(np.int32)
        return [ix[ip[i]:ip[i + 1]] for i in range(n)]
    balls = [None] * n
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        D = dijkstra(A, directed=False, unweighted=True,
                     indices=np.arange(start, stop), limit=thr)
        near = D <= thr
        for k in range(stop - start):
            balls[start + k] = np.flatnonzero(near[k]).astype(np.int32)
        if progress:
            progress(stop, n)
    return balls


def cover_one(A, l_B, balls=None, chunk=512):
    """Greedy diameter-based cover at a single l_B.

    Nodes are already in decreasing-degree order (see csr_from_graph), so
    range(n) is the seed order and box indices increase with creation time.

    The speed trick: a node can only join a box whose SEED lies within thr of it
    (necessary condition, since the seed is a member). So instead of scanning all
    boxes -- which is O(#boxes) per node and dominates when l_B is small and there
    are ~N boxes -- we look up only the boxes seeded inside the node's own ball,
    then test those in creation order. Taking the first that accepts reproduces
    the reference semantics exactly.

    `allowed` sets are kept as sorted int32 arrays and intersected with
    np.intersect1d, so both the membership test and the update stay in C and the
    memory stays proportional to the surviving candidate sets.
    """
    n = A.shape[0]
    thr = max(l_B - 1, 0)
    if balls is None:
        balls = ball_index(A, thr, chunk=chunk)
    assigned = np.zeros(n, dtype=bool)
    is_seed = np.zeros(n, dtype=bool)
    box_of_seed = np.full(n, -1, dtype=np.int64)
    boxes, allowed = [], []
    for node in range(n):
        if assigned[node]:
            continue
        ball = balls[node]
        # candidate boxes: those seeded within thr of this node
        seeds_here = ball[is_seed[ball]]
        placed = False
        if seeds_here.size:
            cand = np.unique(box_of_seed[seeds_here])
            for bi in cand:                      # ascending == creation order
                arr = allowed[bi]
                j = np.searchsorted(arr, node)
                if j < arr.size and arr[j] == node:
                    boxes[bi].append(node)
                    assigned[node] = True
                    allowed[bi] = np.intersect1d(arr, ball, assume_unique=True)
                    placed = True
                    break
        if not placed:
            boxes.append([node])
            assigned[node] = True
            is_seed[node] = True
            box_of_seed[node] = len(boxes) - 1
            allowed.append(np.asarray(ball, dtype=np.int32))
    return boxes


def _cover_worker(args):
    """Top-level for pickling: cover at one l_B and return its box statistics."""
    A, l_B, deg, chunk = args
    balls = ball_index(A, max(l_B - 1, 0), chunk=chunk)
    boxes = cover_one(A, l_B, balls=balls)
    NB = len(boxes)
    masses = np.array([len(b) for b in boxes], float)
    mean_m = masses.mean()
    rec = []
    for b in boxes:
        m = len(b)
        if m < 2:
            continue
        bi = np.asarray(b, dtype=np.int32)
        if m == 2:
            L = 1                     # two mutually-reachable nodes in one box
        else:
            # scipy call only when the answer is not trivial; this removes the
            # tens of thousands of tiny calls that dominated small-l_B covers
            sub = A[bi][:, bi]
            Dl = dijkstra(sub, directed=False, unweighted=True)
            finite = Dl[np.isfinite(Dl)]
            L = int(finite.max()) if finite.size else 1
        rec.append((m, max(L, 1), float(deg[bi].max()), m / mean_m))
    return l_B, NB, rec


def box_triples_fast(G, l_B_values, n_jobs=None, chunk=512, verbose=True):
    """Parallel replacement for box_triples_greedy. Same return signature:
    (M, L, K, MU, NB, N, MU_BY_L)."""
    import os
    from concurrent.futures import ProcessPoolExecutor
    nodes, idx, A = csr_from_graph(G)
    N = len(nodes)
    deg = np.asarray(A.getnnz(axis=1), dtype=float)
    l_B_values = list(l_B_values)
    if n_jobs is None:
        n_jobs = max(1, min(len(l_B_values), (os.cpu_count() or 2)))
    payload = [(A, l, deg, chunk) for l in l_B_values]
    results = {}
    if n_jobs == 1:
        for p in payload:
            l, NB, rec = _cover_worker(p)
            results[l] = (NB, rec)
            if verbose:
                print(f"    l_B={l:3d}: N_B={NB:7d}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for l, NB, rec in ex.map(_cover_worker, payload):
                results[l] = (NB, rec)
                if verbose:
                    print(f"    l_B={l:3d}: N_B={NB:7d}", flush=True)
    M, L, K, MU, NB_list, MU_BY_L = [], [], [], [], [], {}
    for l in l_B_values:
        NBl, rec = results[l]
        NB_list.append(float(NBl))
        here = []
        for m, Ld, k, mu in rec:
            M.append(m); L.append(Ld); K.append(k); MU.append(mu); here.append(mu)
        MU_BY_L[int(l)] = here
    return (np.array(M, float), np.array(L, float), np.array(K, float),
            np.array(MU, float), np.array(NB_list, float), N, MU_BY_L)


def _selftest():
    """Box counts must match the reference sparse implementation exactly."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from real_network import greedy_box_cover_sparse
    from yakubo_fujiki import generate_fsfn
    tests = [("FSFN-GB t=4", generate_fsfn("GB", 4)),
             ("BA n=1500", nx.barabasi_albert_graph(1500, 2, seed=1)),
             ("WS n=1200", nx.watts_strogatz_graph(1200, 4, 0.02, seed=2))]
    ok = True
    for name, G in tests:
        nodes, idx, A = csr_from_graph(G)
        order = nodes                        # decreasing degree, same as reference
        for l in (2, 3, 4, 6, 9):
            fast = len(cover_one(A, l))
            ref = len(greedy_box_cover_sparse(G, l, order=order))
            flag = "OK " if fast == ref else "DIFF"
            if fast != ref:
                ok = False
            print(f"  {flag} {name:14s} l_B={l}: fast={fast:5d} ref={ref:5d}")
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    _selftest()

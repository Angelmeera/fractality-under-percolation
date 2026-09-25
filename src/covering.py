"""
Four box-covering algorithms, so that d_B can be shown to be (or not to be)
robust to the choice.

Referees of box-counting papers reasonably ask whether the reported dimension is
an artefact of one covering heuristic. The literature offers several, and they are
known to disagree; Kovacs, Nagy & Molontay (Appl. Netw. Sci. 6, 73 (2021))
compare them systematically. We implement:

  greedy-degree  our default: seed boxes in DECREASING DEGREE order and admit a
                 node only if it lies within l_B - 1 of every current member, so
                 the box diameter is < l_B. Deterministic once ties are broken on
                 the node label.
  cbb            Compact Box Burning (Song, Gallos, Havlin & Makse, J. Stat.
                 Mech. P03006 (2007)): the same admission rule but with the seed
                 order RANDOM, averaged over several orders. This isolates how
                 much of our answer comes from the degree ordering.
  memb           Maximum-Excluded-Mass Burning (same reference), which Song et
                 al. recommend for large networks. It is RADIUS based: repeatedly choose as a centre the
                 node whose r_B-ball covers the most still-uncovered nodes, then
                 assign every node to its nearest centre. A box then has radius
                 <= r_B and hence diameter <= 2 r_B, so the comparable box size is
                 l_B = 2 r_B + 1.
  colouring      the graph-colouring formulation (same reference): build H with
                 i ~ j iff d(i,j) >= l_B and colour it greedily; each colour class
                 has all pairwise distances < l_B and is therefore a valid box.
                 Exact-ish but needs the distance matrix, so small graphs only.

Why this matters here specifically: on the nd.edu WWW graph the two published box
dimensions are 4.1 (Song et al. 2005, from the greedy-colouring cover; see Song et
al., J. Stat. Mech. P03006 (2007), Sec. 2) and 4.8 (Fronczak et al. 2024), while our
greedy-degree cover gives 3.86 on the even ladder. MEMB agrees with our cover to
1.5% on l_B = 3 -> 5 (3.275 against 3.225), so on the interval both can cover the
choice of algorithm does not explain the discrepancy; the box-size grid does most of
it (4.096 on l_B = 2..5, paper Sec. 4.8).

LIMIT OF MEMB AT LARGE RADIUS (measured, not conjectured)
---------------------------------------------------------
On the WWW (N = 325,729) the radius-3 ball statistics are: mean 4,582, max 162,309
-- the largest single ball is HALF THE NETWORK. Two consequences, both of which
bound what MEMB can say about this graph:

  * memory: a full ball table at r = 3 is ~1.5x10^9 entries (~7.5 GB), which is
    why the eager cover_memb() was OOM-killed at 6.1 GB;
  * time: the lazy greedy needs ~20 batched rounds of 512 re-evaluations to accept
    each new centre once the first huge ball is taken, because thousands of stale
    upper bounds sit far above their true excluded masses. Measured: 3 centres in
    60 rounds. Extrapolating to the ~1,500 centres r = 3 would need gives ~30
    hours, and no amount of further batching changes the re-evaluation count.

The scientific reading matters more than the engineering one: a box that covers
half the graph is not probing self-similarity at that scale, it is reporting that
the network radius is comparable to 3. The WWW has mean shortest path ~7, so
radius-3 balls around hubs reach most of it. The usable MEMB range on this graph is
therefore r_B in {1, 2}, and that is a property of the network rather than of the
implementation.
"""
import os
import sys
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from fast_cover import csr_from_graph
from snap_networks import _balls_for, _ai_matrix, cover_stream, _SPMM_MAX_RADIUS


# --------------------------------------------------------------- diameter rule
def cover_greedy_degree(A, l_B, **kw):
    """Our default cover (decreasing-degree seeding). Delegates to the streaming
    implementation, which is verified box-for-box against the reference."""
    return cover_stream(A, l_B, **kw)


def cover_cbb(A, l_B, order=None, seed=0, chunk=512):
    """Compact Box Burning with an arbitrary (by default random) seed order.

    Identical admission rule to cover_greedy_degree; only the order differs. Balls
    are computed lazily for the nodes that need them, as there.
    """
    n = A.shape[0]
    thr = max(l_B - 1, 0)
    rng = np.random.default_rng(seed)
    if order is None:
        order = rng.permutation(n)
    order = np.asarray(order)
    AI = _ai_matrix(A) if 0 < thr <= _SPMM_MAX_RADIUS else None
    assigned = np.zeros(n, dtype=bool)
    boxes, allowed = [], []
    cache, pending = {}, list(order)
    pos = 0
    while pos < len(pending):
        node = int(pending[pos]); pos += 1
        if assigned[node]:
            continue
        ball = cache.pop(node, None)
        if ball is None:
            cache = {}
            idxs = [node]
            k = pos
            while len(idxs) < chunk and k < len(pending):
                if not assigned[pending[k]]:
                    idxs.append(int(pending[k]))
                k += 1
            for i, bl in zip(idxs, _balls_for(A, AI, idxs, thr)):
                cache[i] = bl
            ball = cache.pop(node)
        placed = False
        for bi in range(len(boxes)):
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
            allowed.append(ball[~assigned[ball]].astype(np.int32, copy=False))
    return boxes


def cover_colouring(A, l_B, max_n=6000):
    """Greedy graph colouring of the 'far enough apart' graph.

    i and j may NOT share a box iff d(i,j) >= l_B. Colour that conflict graph
    greedily in decreasing conflict-degree order; every colour class is then a set
    with all pairwise distances < l_B, i.e. a valid box. Needs all-pairs
    distances, so it is restricted to small graphs.
    """
    n = A.shape[0]
    if n > max_n:
        raise MemoryError(f"colouring needs all-pairs distances; n={n} > {max_n}")
    D = dijkstra(A, directed=False, unweighted=True)
    far = (D >= l_B)                      # conflicts
    np.fill_diagonal(far, False)
    deg = far.sum(axis=1)
    order = np.argsort(-deg, kind="stable")
    colour = np.full(n, -1, dtype=np.int64)
    for v in order:
        used = set(colour[far[v]][colour[far[v]] >= 0].tolist())
        c = 0
        while c in used:
            c += 1
        colour[v] = c
    boxes = [[] for _ in range(colour.max() + 1)]
    for v in range(n):
        boxes[colour[v]].append(int(v))
    return [b for b in boxes if b]


# ----------------------------------------------------------------- radius rule
def _balls_all(A, r, chunk=2048):
    """All r-balls as sorted int32 arrays. Only used by MEMB, and only at the
    small radii MEMB needs, so the memory is bounded in practice."""
    n = A.shape[0]
    AI = _ai_matrix(A) if 0 < r <= _SPMM_MAX_RADIUS else None
    out = [None] * n
    step = max(16, min(chunk, 200_000_000 // (8 * max(n, 1)))) if r > _SPMM_MAX_RADIUS else chunk
    for s in range(0, n, step):
        idxs = list(range(s, min(s + step, n)))
        for i, b in zip(idxs, _balls_for(A, AI, idxs, r)):
            out[i] = b
    return out


def _ball_sizes(A, r, chunk=1024):
    """|ball_r(i)| for every i, WITHOUT storing the balls.

    The sizes are all the lazy-greedy MEMB needs as initial upper bounds, and they
    are N integers instead of the ~5x10^8 entries a full radius-3 ball table wants
    on the WWW (which is what OOM-killed the eager version). Computed blockwise and
    the block thrown away.
    """
    n = A.shape[0]
    AI = _ai_matrix(A) if 0 < r <= _SPMM_MAX_RADIUS else None
    sizes = np.zeros(n, dtype=np.int64)
    step = max(16, min(chunk, 200_000_000 // (8 * max(n, 1)))) if r > _SPMM_MAX_RADIUS else chunk
    for s in range(0, n, step):
        idxs = list(range(s, min(s + step, n)))
        for i, b in zip(idxs, _balls_for(A, AI, idxs, r)):
            sizes[i] = b.size
    return sizes


def _ball_one(A, src, r, AI=None):
    """One r-ball, computed in C.

    This was originally a pure-Python BFS, which is fine on a few hundred nodes and
    hopeless on the WWW: a radius-3 ball there holds ~10^5 nodes, the lazy-greedy
    heap pops far more often than it accepts a centre, and the run made no
    measurable progress in 90 minutes. Routing it through the same sparse
    frontier-expansion primitive the rest of the module uses moves the traversal
    out of the interpreter entirely.
    """
    return _balls_for(A, AI if AI is not None else _ai_matrix(A), [src], r)[0]


def cover_memb_lean(A, r_B, verbose=False, batch=512, progress_every=20):
    """MEMB without a stored ball table, with BATCHED re-evaluation.

    Same algorithm and same output as cover_memb; the difference is purely how the
    excluded masses are recomputed. Two earlier versions of this function are worth
    recording as cautionary notes:

      * computing each ball with a pure-Python BFS -- fine on a few hundred nodes,
        hopeless here, because a radius-3 ball on the WWW holds ~1.6x10^4 nodes on
        average and 1.6x10^5 at worst;
      * computing them one at a time through the sparse primitive -- correct, but
        the per-call overhead of three sparse matrix products dominates, so a run
        that needs O(10^5) re-evaluations takes tens of minutes.

    The fix is to pop the top `batch` candidates by their current upper bound,
    recompute all of their true excluded masses in ONE batched call, and accept the
    best of them only if it is at least the largest bound still in the heap. That
    acceptance test is what keeps the result exactly equal to the unbatched greedy:
    the largest remaining bound is an upper bound on any unexamined node's true
    mass, so if the best examined candidate beats it, no unexamined node can beat
    the examined one.
    """
    import heapq
    n = A.shape[0]
    AI = _ai_matrix(A) if 0 < r_B <= _SPMM_MAX_RADIUS else None
    sizes = _ball_sizes(A, r_B)
    if verbose:
        print(f"      MEMB(lean) r={r_B}: ball sizes done, "
              f"mean {sizes.mean():.0f}, max {sizes.max()}", flush=True)
    covered = np.zeros(n, dtype=bool)
    heap = [(-int(sizes[i]), i) for i in range(n)]
    heapq.heapify(heap)
    centres, n_covered, rounds = [], 0, 0
    while n_covered < n and heap:
        rounds += 1
        popped = []
        for _ in range(min(batch, len(heap))):
            popped.append(heapq.heappop(heap))
        idxs = [i for _, i in popped]
        balls = _balls_for(A, AI, idxs, r_B)
        masses = np.array([int((~covered[b]).sum()) for b in balls])
        # Tie-break on the smallest node index, which is exactly what the heap's
        # own (-mass, index) ordering does in the unbatched version. Without this
        # the two agree on the box COUNT but not on which centres are chosen,
        # because the masses are recomputed and so no longer follow the pop order.
        tied = np.flatnonzero(masses == masses.max())
        best = int(tied[np.argmin([idxs[k] for k in tied])])
        nxt_bound = -heap[0][0] if heap else -1
        if masses[best] >= nxt_bound:
            i = idxs[best]
            newly = balls[best][~covered[balls[best]]]
            covered[newly] = True
            n_covered += int(newly.size)
            centres.append(i)
            for k, (_, j) in enumerate(popped):
                if k != best and masses[k] > 0:
                    heapq.heappush(heap, (-int(masses[k]), j))
        else:
            for k, (_, j) in enumerate(popped):
                if masses[k] > 0:
                    heapq.heappush(heap, (-int(masses[k]), j))
        if verbose and rounds % progress_every == 0:
            print(f"      MEMB(lean) r={r_B}: round {rounds}, {len(centres)} "
                  f"centres, {n_covered}/{n} covered, heap {len(heap)}",
                  flush=True)
    for i in np.flatnonzero(~covered):
        centres.append(int(i))
        covered[i] = True
    owner, _ = _multi_source_owner(A, centres)
    boxes = [[] for _ in centres]
    for v in range(n):
        if owner[v] >= 0:
            boxes[owner[v]].append(v)
        else:
            boxes.append([v])
    return [b for b in boxes if b], centres


def _multi_source_owner(A, centres):
    """Nearest-centre assignment by one simultaneous BFS from every centre."""
    n = A.shape[0]
    ip, ix = A.indptr, A.indices
    owner = np.full(n, -1, dtype=np.int64)
    dist = np.full(n, -1, dtype=np.int32)
    dq = deque()
    for ci, c in enumerate(centres):
        owner[c] = ci
        dist[c] = 0
        dq.append(c)
    while dq:
        u = dq.popleft()
        du = dist[u]
        for v in ix[ip[u]:ip[u + 1]]:
            if dist[v] < 0:
                dist[v] = du + 1
                owner[v] = owner[u]
                dq.append(v)
    return owner, dist


def cover_memb(A, r_B, balls=None, verbose=False):
    """Maximum-Excluded-Mass Burning at radius r_B.

    Lazy greedy: keep a heap keyed by an upper bound on each node's excluded mass
    (the number of still-uncovered nodes in its r_B-ball), pop the best, recompute
    its true value, and accept it only if it is still the maximum. That is the
    standard trick for greedy max-coverage and it turns an O(N) rescan per centre
    into a few recomputations.

    Returns (boxes, centres). Box radius <= r_B, so diameter <= 2 r_B and the
    comparable diameter-rule box size is l_B = 2 r_B + 1.
    """
    import heapq
    n = A.shape[0]
    if balls is None:
        balls = _balls_all(A, r_B)
    covered = np.zeros(n, dtype=bool)
    heap = [(-int(balls[i].size), i) for i in range(n)]
    heapq.heapify(heap)
    centres = []
    n_covered = 0
    while n_covered < n:
        while heap:
            neg, i = heapq.heappop(heap)
            true_mass = int((~covered[balls[i]]).sum())
            if true_mass == 0:
                continue
            if heap and true_mass < -heap[0][0]:
                heapq.heappush(heap, (-true_mass, i))
                continue
            centres.append(i)
            newly = balls[i][~covered[balls[i]]]
            covered[newly] = True
            n_covered += newly.size
            break
        else:
            # nothing left in the heap but uncovered nodes remain (disconnected
            # pieces smaller than a ball): make each its own centre
            for i in np.flatnonzero(~covered):
                centres.append(int(i))
                covered[i] = True
                n_covered += 1
            break
        if verbose and len(centres) % 500 == 0:
            print(f"      MEMB r={r_B}: {len(centres)} centres, "
                  f"{n_covered}/{n} covered", flush=True)
    owner, _ = _multi_source_owner(A, centres)
    boxes = [[] for _ in centres]
    for v in range(n):
        if owner[v] >= 0:
            boxes[owner[v]].append(v)
        else:                                 # unreachable: singleton box
            boxes.append([v])
    return [b for b in boxes if b], centres


# ------------------------------------------------------------------- reporting
def _fit(lb, NB, N):
    lb = np.asarray(lb, float); NB = np.asarray(NB, float)
    keep = NB > 1
    x = np.log(lb[keep]); y = np.log(NB[keep] / N)
    a, b = np.polyfit(x, y, 1)
    pred = a * x + b
    ss = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - np.sum((y - pred) ** 2) / ss) if ss > 0 else float("nan")
    return float(-a), r2


def compare(G, l_B_values, r_B_values=None, cbb_reps=3, do_colouring=None,
            name="network", verbose=True):
    """Run every applicable algorithm and return d_B from each."""
    nodes, idx, A = csr_from_graph(G)
    N = len(nodes)
    if r_B_values is None:
        r_B_values = [(l - 1) // 2 for l in l_B_values if l >= 3]
        r_B_values = sorted(set(r for r in r_B_values if r >= 1))
    if do_colouring is None:
        do_colouring = N <= 6000
    out = {"name": name, "N": N, "M": G.number_of_edges(),
           "l_B": list(l_B_values), "r_B": list(r_B_values)}

    NB = [len(cover_greedy_degree(A, l)) for l in l_B_values]
    out["greedy_degree"] = {"NB": NB, "dB": None}
    out["greedy_degree"]["dB"], out["greedy_degree"]["R2"] = _fit(l_B_values, NB, N)
    if verbose:
        print(f"  greedy-degree : N_B={NB}  d_B={out['greedy_degree']['dB']:.3f} "
              f"(R2={out['greedy_degree']['R2']:.4f})", flush=True)

    reps = []
    for s in range(cbb_reps):
        reps.append([len(cover_cbb(A, l, seed=100 + s)) for l in l_B_values])
    reps = np.array(reps, float)
    dBs = [_fit(l_B_values, reps[i], N)[0] for i in range(reps.shape[0])]
    out["cbb"] = {"NB_mean": reps.mean(axis=0).tolist(),
                  "NB_sd": reps.std(axis=0, ddof=1).tolist() if reps.shape[0] > 1
                  else [0.0] * reps.shape[1],
                  "dB": float(np.mean(dBs)),
                  "dB_sd": float(np.std(dBs, ddof=1)) if len(dBs) > 1 else 0.0,
                  "reps": reps.tolist()}
    out["cbb"]["R2"] = _fit(l_B_values, reps.mean(axis=0), N)[1]
    if verbose:
        print(f"  cbb (random)  : d_B={out['cbb']['dB']:.3f} "
              f"+- {out['cbb']['dB_sd']:.3f} over {cbb_reps} orders "
              f"(R2={out['cbb']['R2']:.4f})", flush=True)

    if r_B_values:
        NBm, lbm = [], []
        for r in r_B_values:
            boxes, centres = cover_memb(A, r)
            NBm.append(len(boxes))
            lbm.append(2 * r + 1)
            if verbose:
                print(f"    MEMB r_B={r}: N_B={len(boxes)} "
                      f"(centres {len(centres)})", flush=True)
        d, r2 = _fit(lbm, NBm, N)
        out["memb"] = {"r_B": list(r_B_values), "l_B_equiv": lbm, "NB": NBm,
                       "dB": d, "R2": r2}
        if verbose:
            print(f"  memb          : d_B={d:.3f} (R2={r2:.4f}) on "
                  f"l_B=2r+1={lbm}", flush=True)

    if do_colouring:
        try:
            NBc = [len(cover_colouring(A, l)) for l in l_B_values]
            d, r2 = _fit(l_B_values, NBc, N)
            out["colouring"] = {"NB": NBc, "dB": d, "R2": r2}
            if verbose:
                print(f"  colouring     : N_B={NBc}  d_B={d:.3f} (R2={r2:.4f})",
                      flush=True)
        except MemoryError as exc:
            out["colouring"] = {"skipped": str(exc)}
            if verbose:
                print(f"  colouring     : skipped ({exc})", flush=True)
    return out


def _selftest():
    """Every algorithm must produce genuinely valid boxes: diameter < l_B for the
    diameter-rule ones, radius <= r_B for MEMB."""
    import networkx as nx
    G = nx.barabasi_albert_graph(400, 2, seed=5)
    nodes, idx, A = csr_from_graph(G)
    D = dijkstra(A, directed=False, unweighted=True)
    ok = True
    for l in (3, 4, 6):
        for label, boxes in (("greedy", cover_greedy_degree(A, l)),
                             ("cbb", cover_cbb(A, l, seed=1)),
                             ("colouring", cover_colouring(A, l))):
            bad = 0
            for b in boxes:
                bi = np.asarray(b)
                sub = D[np.ix_(bi, bi)]
                fin = sub[np.isfinite(sub)]
                if fin.size and fin.max() >= l:
                    bad += 1
            if bad:
                ok = False
            print(f"  {'OK ' if not bad else 'BAD'} {label:10s} l_B={l}: "
                  f"{len(boxes)} boxes, {bad} violate diameter < {l}")
    for r in (1, 2, 3):
        boxes, centres = cover_memb(A, r)
        bad = 0
        for b, c in zip(boxes, centres):
            if len(b) and D[c, np.asarray(b)].max() > r:
                bad += 1
        if bad:
            ok = False
        print(f"  {'OK ' if not bad else 'BAD'} memb       r_B={r}: "
              f"{len(boxes)} boxes, {bad} violate radius <= {r}")
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    _selftest()

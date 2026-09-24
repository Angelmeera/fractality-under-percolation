"""
Real-network analysis for the Fronczak mass law — works on ANY network
(real or synthetic, intact or percolated), because it needs no knowledge of a
construction hierarchy.

THE IDEA
--------
On a deterministic model we can build Fronczak's HUB-CENTRED boxes exactly from
the construction. On a real network there is no construction to consult. But a
hub-centred box of diameter L around a hub of degree k is, operationally, just a
BALL: take a node of degree k and grow a breadth-first ball of radius r around
it. Its mass m(r,k) should obey the mass law

        m(r, k) = B r^alpha k^beta ,

so alpha and beta come from a two-variable log-log regression over many
(m, r, k) triples sampled across seeds and radii. This is the cluster-growing
counterpart of box-covering, and it is exactly hub-centred by construction.

Validated against the Yakubo-Fujiki FSFN, where alpha = ln4/ln3 = 1.2619 and
beta = 1 are known exactly -- see validate_on_fsfn().

DBLP RECIPE (Fronczak et al. 2024, Methods)
-------------------------------------------
Take the DBLP coauthorship network, weight each edge by the number of joint
papers, and keep only edges with weight >= 25. This removes weak ties and the
remaining backbone (~2.5k nodes, ~3.2k edges in their run) is naturally fractal.
Loaders for the DBLP XML dump, the DBLP-V12 JSON and a plain weighted edge list
are provided below.
"""
import re
import numpy as np
import networkx as nx
from collections import defaultdict


# ---------------------------------------------------------------- estimators
def gamma_ccdf(G, kmin=2):
    """Degree exponent from the complementary CDF (robust to spiked P(k))."""
    ks = np.sort(np.array([d for _, d in G.degree() if d >= kmin], float))
    if len(ks) < 20:
        return float("nan")
    u = np.unique(ks)
    cc = np.array([(ks >= k).mean() for k in u])
    m = cc > 0
    return float(1 - np.polyfit(np.log(u[m]), np.log(cc[m]), 1)[0])


def ball_records(G, radii=(1, 2, 3, 4), n_seeds=400, rng=None, min_mass=3):
    """Grow BFS balls of each radius around sampled seeds.
    Returns arrays (m, r, k) where k is the SEED's degree (the hub degree)."""
    if rng is None:
        rng = np.random.default_rng(0)
    nodes = list(G.nodes())
    deg = dict(G.degree())
    # prefer higher-degree seeds so log k spans a range, but keep a random tail
    nodes_sorted = sorted(nodes, key=lambda n: -deg[n])
    n_top = min(len(nodes), n_seeds // 2)
    seeds = list(nodes_sorted[:n_top])
    rest = [n for n in nodes if n not in set(seeds)]
    if rest:
        seeds += list(rng.choice(rest, size=min(len(rest), n_seeds - n_top),
                                replace=False))
    M, R, K = [], [], []
    for s in seeds:
        if deg[s] < 1:
            continue
        lengths = nx.single_source_shortest_path_length(G, s, cutoff=max(radii))
        for r in radii:
            m = sum(1 for _, d in lengths.items() if d <= r)
            if m >= min_mass:
                M.append(m); R.append(r); K.append(deg[s])
    return np.array(M, float), np.array(R, float), np.array(K, float)


def fit_mass_law_balls(M, R, K):
    """OLS of log m on (log r, log k). Returns alpha, beta, R2, n."""
    ok = (M > 1) & (R > 0) & (K > 0)
    if ok.sum() < 12 or len(set(R[ok])) < 2 or len(set(K[ok])) < 2:
        return dict(alpha=float("nan"), beta=float("nan"), R2=float("nan"),
                    n=int(ok.sum()))
    A = np.column_stack([np.ones(ok.sum()), np.log(R[ok]), np.log(K[ok])])
    sol, *_ = np.linalg.lstsq(A, np.log(M[ok]), rcond=None)
    pred = A @ sol
    y = np.log(M[ok])
    r2 = float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
    return dict(alpha=float(sol[1]), beta=float(sol[2]), R2=r2, n=int(ok.sum()))


def ball_mass_law(G, radii=(1, 2, 3, 4), n_seeds=400, rng=None):
    return fit_mass_law_balls(*ball_records(G, radii, n_seeds, rng))


# ---------------------------------------------------------------- validation
def validate_on_fsfn(t=5, which="GB", radii=(1, 2, 3, 4, 5, 6, 7, 8, 9)):
    """Sanity-check the ball estimator where the answer is known exactly:
    Yakubo-Fujiki FSFN has alpha = ln4/ln3 = 1.2619 and beta = 1."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from yakubo_fujiki import generate_fsfn
    G = generate_fsfn(which, t)
    out = ball_mass_law(G, radii=radii, n_seeds=600)
    out.update(model=f"FSFN-{which} t={t}", N=G.number_of_nodes(),
               alpha_theory=float(np.log(4) / np.log(3)), beta_theory=1.0)
    return out


# ---------------------------------------------------------------- DBLP loaders
def backbone(Gw, min_joint=25):
    """Keep only edges with weight >= min_joint, then the largest component.
    This is Fronczak's weak-tie removal that makes DBLP fractal."""
    H = nx.Graph()
    H.add_edges_from([(u, v) for u, v, w in Gw.edges(data="weight")
                      if (w or 0) >= min_joint])
    if H.number_of_nodes() == 0:
        return H
    return H.subgraph(max(nx.connected_components(H), key=len)).copy()


class _EntitySafeStream:
    """Byte stream wrapper that neutralises DBLP's external-DTD entity refs.

    dblp.xml declares HTML-style entities (&uuml; &ouml; &szlig; ...) in dblp.dtd.
    Without the DTD, ElementTree raises "undefined entity" on the first accented
    author name. We map the common ones to ASCII and replace anything else of the
    form &word; with '?', so the parse never dies on a name.
    """
    _MAP = {
        b"&auml;": b"ae", b"&ouml;": b"oe", b"&uuml;": b"ue", b"&Auml;": b"Ae",
        b"&Ouml;": b"Oe", b"&Uuml;": b"Ue", b"&szlig;": b"ss",
        b"&aacute;": b"a", b"&eacute;": b"e", b"&iacute;": b"i", b"&oacute;": b"o",
        b"&uacute;": b"u", b"&agrave;": b"a", b"&egrave;": b"e", b"&igrave;": b"i",
        b"&ograve;": b"o", b"&ugrave;": b"u", b"&acirc;": b"a", b"&ecirc;": b"e",
        b"&icirc;": b"i", b"&ocirc;": b"o", b"&ucirc;": b"u", b"&ccedil;": b"c",
        b"&ntilde;": b"n", b"&atilde;": b"a", b"&otilde;": b"o", b"&aring;": b"a",
        b"&oslash;": b"o", b"&aelig;": b"ae", b"&yacute;": b"y", b"&thorn;": b"th",
        b"&eth;": b"d", b"&reg;": b"", b"&times;": b"x", b"&micro;": b"u",
    }
    _KEEP = re.compile(rb"&(?:amp|lt|gt|quot|apos|#\d+|#x[0-9A-Fa-f]+);")
    _ANY  = re.compile(rb"&[A-Za-z][A-Za-z0-9]*;")

    def __init__(self, raw, chunk=1 << 20):
        self.raw, self.chunk = raw, chunk
        self.buf = b""      # cleaned bytes ready to hand out
        self.carry = b""    # trailing bytes holding a possibly-split entity
        self.eof = False

    def _clean(self, b):
        for k, v in self._MAP.items():
            if k in b:
                b = b.replace(k, v)
        keep = self._KEEP

        def sub(m):
            return m.group(0) if keep.fullmatch(m.group(0)) else b"?"
        return self._ANY.sub(sub, b)

    def _fill(self):
        piece = self.raw.read(self.chunk)
        if not piece:
            self.eof = True
            if self.carry:
                self.buf += self._clean(self.carry)
                self.carry = b""
            return
        piece = self.carry + piece
        self.carry = b""
        amp = piece.rfind(b"&")
        if amp != -1 and b";" not in piece[amp:]:
            self.carry, piece = piece[amp:], piece[:amp]
        self.buf += self._clean(piece)

    def read(self, n=-1):
        if n is None or n < 0:
            while not self.eof:
                self._fill()
            out, self.buf = self.buf, b""
            return out
        while not self.eof and len(self.buf) < n:
            self._fill()
        out, self.buf = self.buf[:n], self.buf[n:]
        return out


_PUB_TAGS = {"article", "inproceedings", "proceedings", "book",
             "incollection", "phdthesis", "mastersthesis"}


def _iter_author_lists_xml(path, max_records=None, progress=500_000):
    """Stream author lists from dblp.xml[.gz], entity-safe and memory-safe."""
    import gzip
    import xml.etree.ElementTree as ET
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as fh:
        stream = _EntitySafeStream(fh)
        ctx = ET.iterparse(stream, events=("start", "end"))
        _, root = next(ctx)                     # grab the root to prune it
        n = 0
        for event, elem in ctx:
            if event != "end" or elem.tag not in _PUB_TAGS:
                continue
            authors = [a.text for a in elem.findall("author") if a.text]
            if len(authors) > 1:
                yield authors
            n += 1
            root.clear()                        # essential: stop the tree growing
            if progress and n % progress == 0:
                print(f"    ... {n:,} publications parsed", flush=True)
            if max_records and n >= max_records:
                return


def dblp_from_xml(path, max_records=None, min_joint=25, two_pass=True,
                  verbose=True):
    """Parse the official DBLP XML dump into a weighted coauthorship graph.

    Memory note: the full dump has ~7M publications and tens of millions of
    distinct coauthor pairs, which will exhaust a laptop if counted naively.
    With two_pass=True (default) we first count publications per author and keep
    only authors with at least min_joint of them -- a necessary condition for
    sharing min_joint papers with anybody -- then count pairs only among those.
    This shrinks the pair table by orders of magnitude and gives the identical
    thresholded backbone.
    """
    if two_pass:
        if verbose:
            print("  pass 1/2: counting publications per author ...", flush=True)
        pubs = defaultdict(int)
        for authors in _iter_author_lists_xml(path, max_records):
            for a in authors:
                pubs[a] += 1
        keep = {a for a, c in pubs.items() if c >= min_joint}
        if verbose:
            print(f"    {len(pubs):,} authors seen; {len(keep):,} have >= "
                  f"{min_joint} publications", flush=True)
        del pubs
        if verbose:
            print("  pass 2/2: counting joint papers among those authors ...",
                  flush=True)
    else:
        keep = None

    W = defaultdict(int)
    for authors in _iter_author_lists_xml(path, max_records):
        if keep is not None:
            authors = [a for a in authors if a in keep]
        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                a, b = authors[i], authors[j]
                W[(a, b) if a < b else (b, a)] += 1
    G = nx.Graph()
    G.add_edges_from((a, b, {"weight": w}) for (a, b), w in W.items())
    if verbose:
        print(f"    {G.number_of_edges():,} coauthor pairs retained", flush=True)
    return G


def _iter_author_lists_v12(path, max_records=None, progress=500_000):
    import json
    n = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip().rstrip(",")
            if not line or line[0] != "{":
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            authors = [a.get("name") for a in rec.get("authors", []) if a.get("name")]
            if len(authors) > 1:
                yield authors
            n += 1
            if progress and n % progress == 0:
                print(f"    ... {n:,} records parsed", flush=True)
            if max_records and n >= max_records:
                return


def dblp_from_v12_json(path, max_records=None, min_joint=25, two_pass=True,
                       verbose=True):
    """Parse the AMiner DBLP-Citation-network V12 json-lines file (the exact
    dataset Fronczak et al. used) into a weighted coauthorship graph.
    Uses the same two-pass author prefilter as dblp_from_xml -- see its docstring."""
    if two_pass:
        if verbose:
            print("  pass 1/2: counting publications per author ...", flush=True)
        pubs = defaultdict(int)
        for authors in _iter_author_lists_v12(path, max_records):
            for a in authors:
                pubs[a] += 1
        keep = {a for a, c in pubs.items() if c >= min_joint}
        if verbose:
            print(f"    {len(pubs):,} authors seen; {len(keep):,} have >= "
                  f"{min_joint} publications", flush=True)
        del pubs
        if verbose:
            print("  pass 2/2: counting joint papers ...", flush=True)
    else:
        keep = None
    W = defaultdict(int)
    for authors in _iter_author_lists_v12(path, max_records):
        if keep is not None:
            authors = [a for a in authors if a in keep]
        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                a, b = authors[i], authors[j]
                W[(a, b) if a < b else (b, a)] += 1
    G = nx.Graph()
    G.add_edges_from((a, b, {"weight": w}) for (a, b), w in W.items())
    if verbose:
        print(f"    {G.number_of_edges():,} coauthor pairs retained", flush=True)
    return G


def load_weighted_edgelist(path, sep=None):
    """Loader for 'u<sep>v<sep>w' per line (w = number of joint papers).
    Handles .gz and tab-separated author names containing spaces."""
    import gzip
    G = nx.Graph()
    opener = gzip.open if str(path).endswith(".gz") else open
    if sep is None and str(path).endswith((".tsv", ".tsv.gz")):
        sep = "\t"
    with opener(path, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(sep)
            if len(parts) < 2:
                continue
            w = float(parts[2]) if len(parts) > 2 else 1.0
            G.add_edge(parts[0], parts[1], weight=w)
    return G


# ---------------------------------------------------------------- full report
def analyse_real(G, name="network", radii=(1, 2, 3, 4), lb_grid=(2, 3, 4, 6, 9, 13),
                 n_seeds=500, verbose=True):
    """Full intact-network report: d_B, gamma, d_k, mass-law alpha/beta and the
    closure of d_B = alpha + beta*d_k."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fractal_dynamics import all_pairs_dist, measure_dB_from_D
    N = G.number_of_nodes()
    nodes, D = all_pairs_dist(G)
    deg = np.array([G.degree(u) for u in nodes], float)
    diam = int(D[D < 10 ** 9].max())
    grid = [l for l in lb_grid if l <= diam + 1]
    dB, r2, NB = measure_dB_from_D(D, deg, N, grid)
    g = gamma_ccdf(G)
    dk = dB / (g - 1) if np.isfinite(g) and g > 1 else float("nan")
    ml = ball_mass_law(G, radii=radii, n_seeds=n_seeds)
    rhs = ml["alpha"] + ml["beta"] * dk
    out = dict(name=name, N=N, M=G.number_of_edges(), diameter=diam,
               mean_degree=2 * G.number_of_edges() / N,
               dB=dB, dB_R2=r2, l_B_grid=grid, NB=NB.tolist(),
               gamma=g, dk=dk, alpha=ml["alpha"], beta=ml["beta"],
               masslaw_R2=ml["R2"], n_balls=ml["n"], rhs=float(rhs),
               closure_pct=float(abs(dB - rhs) / abs(dB) * 100)
               if np.isfinite(rhs) else float("nan"))
    if verbose:
        print(f"[{name}] N={N} M={out['M']} diam={diam} <k>={out['mean_degree']:.2f}")
        print(f"  d_B={dB:.3f} (R2={r2:.3f})  gamma={g:.2f}  d_k=d_B/(gamma-1)={dk:.3f}")
        print(f"  mass law (balls): alpha={ml['alpha']:.3f} beta={ml['beta']:.3f} "
              f"R2={ml['R2']:.3f} on {ml['n']} balls")
        print(f"  closure: alpha+beta*d_k={rhs:.3f} vs d_B={dB:.3f} "
              f"-> {out['closure_pct']:.1f}%")
    return out


def percolate_real(G, ps, l_B_values=(2, 3, 4, 6, 9, 13), n_trials=4,
                   seed=0, verbose=True, **_ignored):
    """alpha(p), beta(p), d_B(p), gamma(p), delta(p) on the giant component of a
    percolated real network.

    IMPORTANT: this uses the same greedy-box macroscopic route as
    fronczak_protocol(). An earlier version used metric balls; balls are NOT a
    valid substitute (see the note above -- the fitted beta decays with radius),
    so any alpha(p)/beta(p) produced by that version should be discarded.
    """
    rng = np.random.default_rng(seed)
    E = list(G.edges())
    rows = []
    for p in ps:
        acc = {k: [] for k in ("alpha", "beta", "dB", "gamma", "delta")}
        for _ in range(n_trials):
            H = nx.Graph()
            H.add_nodes_from(G.nodes())
            H.add_edges_from([e for e in E if rng.random() < p])
            comps = list(nx.connected_components(H))
            if not comps:
                continue
            gc = H.subgraph(max(comps, key=len)).copy()
            if gc.number_of_nodes() < 60:
                continue
            try:
                r = fronczak_protocol(gc, l_B_values=l_B_values,
                                      name="", verbose=False)
            except Exception:
                continue
            for k, key in (("alpha", "alpha_theory"), ("beta", "beta_theory"),
                           ("dB", "dB"), ("gamma", "gamma"), ("delta", "delta")):
                v = r.get(key)
                if v is not None and np.isfinite(v):
                    acc[k].append(v)
        row = {"p": float(p)}
        for k, v in acc.items():
            row[k] = float(np.mean(v)) if v else float("nan")
            row[k + "_sd"] = float(np.std(v)) if v else float("nan")
        rows.append(row)
        if verbose:
            print(f"  p={p:.2f}: dB={row['dB']:.3f} gamma={row['gamma']:.2f} "
                  f"delta={row['delta']:.2f} alpha={row['alpha']:.3f} "
                  f"beta={row['beta']:.3f}", flush=True)
    return rows


def threshold_sweep(Gw, thresholds=(25, 40, 60, 100, 150, 250),
                    l_B_values=(2, 3, 4, 6, 9, 13), max_nodes=60000,
                    n_jobs=None, verbose=True):
    """Backbone size and exponents as a function of the weak-tie threshold.

    Two uses. (a) Calibration: a fixed joint-paper threshold is NOT invariant
    across DBLP snapshots -- the database roughly doubled between the 2020 V12
    release Fronczak et al. used and the current dump, so threshold 25 now yields
    a much larger backbone. Sweeping lets you report the threshold that matches
    their network size, or state your own. (b) Robustness: if d_B is stable across
    thresholds, the measured fractality is not an artefact of one cut.
    """
    out = []
    for thr in thresholds:
        B = backbone(Gw, thr)
        if B.number_of_nodes() > max_nodes:
            if verbose:
                print(f"  w>={thr:4d}: backbone N={B.number_of_nodes()} exceeds "
                      f"max_nodes={max_nodes} -- skipped (raise max_nodes if you "
                      f"have time; covering cost grows quickly)")
            continue
        if B.number_of_nodes() < 60:
            if verbose:
                print(f"  w>={thr:4d}: backbone too small "
                      f"(N={B.number_of_nodes()}) -- skipped")
            continue
        r = fronczak_protocol(B, l_B_values=l_B_values,
                              name=f"w>={thr}", verbose=False)
        r["threshold"] = thr
        out.append(r)
        if verbose:
            print(f"  w>={thr:4d}: N={r['N']:6d} M={r['M_edges']:7d} "
                  f"<k>={r['mean_degree']:.2f}  d_B={r['dB']:.3f} "
                  f"(R2={r['dB_R2']:.3f})  gamma={r['gamma']:.2f} "
                  f"delta={r['delta']:.2f}", flush=True)
    return out


def save_weighted(Gw, path):
    """Write the weighted coauthorship graph so the ~15-minute XML parse is done
    once. Gzipped 'u<TAB>v<TAB>w' with author names preserved."""
    import gzip
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as fh:
        fh.write("# u\tv\tweight\n")
        for u, v, w in Gw.edges(data="weight"):
            fh.write(f"{u}\t{v}\t{int(w or 1)}\n")




# ============================================================================
# FRONCZAK'S REAL-NETWORK PROTOCOL  (the estimator to use on empirical graphs)
# ============================================================================
# NEGATIVE RESULT, worth stating in the paper: metric BALLS do not measure the
# same object as Fronczak's boxes. On the FSFN the ball estimator gives a beta
# that decays with radius (0.86 at r=1 -> ~0 at r=81), because a ball's mass
# becomes governed by r alone once r exceeds a couple of generations, whereas a
# box's mass keeps its hub-degree dependence. ball_mass_law() is retained only to
# document this; do not use it for results.
#
# What works on an arbitrary (real) network is the macroscopic route:
#   1. d_B   from greedy box-covering,  N_B/N ~ l_B^-d_B
#   2. gamma from P(k)    (CCDF fit)
#   3. delta from P(mu), mu = m/<m> the normalised box mass -- fitted SEPARATELY
#      for each l_B (pooling diameters mixes distributions and biases delta low)
#      and restricted to the mu >= 1 tail, since the small-mass range is exactly
#      where greedy covering is known to misbehave.
#   4. then  d_k = d_B/(gamma-1),  d_m = d_B/(delta-1),
#            alpha = ((delta-2)/(delta-1)) d_B,  beta = (gamma-1)/(delta-1)
#   5. CAUTION -- the relation d_B = alpha + beta*d_k is then an ALGEBRAIC
#      IDENTITY, not a test:
#         alpha + beta*d_k = ((delta-2)/(delta-1))d_B + ((gamma-1)/(delta-1))(d_B/(gamma-1))
#                          = d_B[(delta-2)+1]/(delta-1) = d_B   identically.
#      So closure_pct always reads ~0 on this route. The MEANINGFUL test on a real
#      network is whether the EMPIRICAL alpha,beta (from the rescaled, log-binned
#      mass plots) agree with the derived ones -- exactly what Fronczak's Table 1
#      reports as "empirical (theoretical)". The genuine, non-tautological
#      verification of the relation is the one done on the deterministic FSFN with
#      construction-aware boxes (closure 0.36%).

def _ccdf_exponent(vals, vmin=None):
    """Exponent q of P(X>=x) ~ x^-(q-1); returns q, the index of P(x)~x^-q."""
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


def _ball(G, src, radius):
    """Set of nodes within `radius` hops of src (BFS, no distance matrix)."""
    return set(nx.single_source_shortest_path_length(G, src, cutoff=radius))


def greedy_box_cover_sparse(G, l_B, order=None):
    """Greedy diameter-based box covering WITHOUT an all-pairs distance matrix.

    Same rule as the dense version: seed boxes in decreasing-degree order and admit
    a node only if it is within distance l_B - 1 of EVERY current member, so each
    box has true diameter < l_B. Instead of a dense matrix we keep, per box, the
    running intersection of its members' (l_B-1)-balls; admitting a node is then a
    set-membership test and one BFS.

    Memory is O(sum of ball sizes) rather than O(N^2), which is what makes real
    networks of 10^5 nodes feasible -- a dense matrix at N = 10^5 needs ~74 GiB.
    """
    thr = max(l_B - 1, 0)
    if order is None:
        order = sorted(G.nodes(), key=lambda n: (-G.degree(n), str(n)))
    assigned = set()
    boxes, allowed = [], []
    for node in order:
        if node in assigned:
            continue
        placed = False
        for bi in range(len(boxes)):
            if node in allowed[bi]:
                boxes[bi].append(node)
                assigned.add(node)
                allowed[bi] &= _ball(G, node, thr)
                placed = True
                break
        if not placed:
            boxes.append([node])
            assigned.add(node)
            allowed.append(_ball(G, node, thr))
    return boxes


def _box_diameter(G, box):
    """Exact diameter of the induced subgraph on `box` (boxes are small)."""
    if len(box) < 2:
        return 0
    sub = G.subgraph(box)
    best = 0
    for n in box:
        d = nx.single_source_shortest_path_length(sub, n)
        if d:
            m = max(d.values())
            if m > best:
                best = m
    return best


def box_triples_greedy(G, l_B_values, verbose=False):
    """Cover G at each l_B and return per-box (m, L, k), the normalised masses mu
    (pooled and grouped by l_B), the box counts, N, and the mu-by-l_B mapping.
    Uses the sparse cover, so it scales to real networks."""
    N = G.number_of_nodes()
    deg = dict(G.degree())
    order = sorted(G.nodes(), key=lambda n: (-deg[n], str(n)))
    M, L, K, MU = [], [], [], []
    MU_BY_L, NB = {}, []
    for l in l_B_values:
        boxes = greedy_box_cover_sparse(G, l, order=order)
        NB.append(len(boxes))
        masses = np.array([len(b) for b in boxes], float)
        mean_m = masses.mean()
        here = []
        for b in boxes:
            m = len(b)
            if m < 2:
                continue
            Ld = _box_diameter(G, b)
            M.append(m); L.append(max(Ld, 1)); K.append(float(max(deg[x] for x in b)))
            MU.append(m / mean_m); here.append(m / mean_m)
        MU_BY_L[int(l)] = here
        if verbose:
            print(f"    l_B={l:3d}: N_B={len(boxes):6d}  <m>={mean_m:8.2f}", flush=True)
    return (np.array(M, float), np.array(L, float), np.array(K, float),
            np.array(MU, float), np.array(NB, float), N, MU_BY_L)


def fronczak_protocol(G, l_B_values=(2, 3, 4, 6, 9, 13, 19, 28), name="network",
                      verbose=True, n_jobs=None):
    """Full Fronczak analysis of any network via the macroscopic route.

    Box covering runs through fast_cover.box_triples_fast (bulk sparse-matmul /
    scipy balls, candidate-box pruning, parallel over l_B), falling back to the
    pure reference cover if that module is missing. Both give identical box
    counts -- see fast_cover._selftest().
    """
    try:
        from fast_cover import box_triples_fast
        M, L, K, MU, NB, N, MU_BY_L = box_triples_fast(
            G, l_B_values, n_jobs=n_jobs, verbose=verbose)
    except Exception as exc:
        if verbose:
            print(f"  (fast cover unavailable: {exc}; using reference cover)")
        M, L, K, MU, NB, N, MU_BY_L = box_triples_greedy(G, l_B_values,
                                                        verbose=verbose)
    lb = np.array(l_B_values, float)[:len(NB)]
    keep = NB > 1
    coef = np.polyfit(np.log(lb[keep]), np.log(NB[keep] / N), 1)
    dB = float(-coef[0])
    yy = np.log(NB[keep] / N)
    pred = np.polyval(coef, np.log(lb[keep]))
    r2 = float(1 - np.sum((yy - pred) ** 2) / np.sum((yy - yy.mean()) ** 2))
    gamma = _ccdf_exponent([d for _, d in G.degree()], vmin=2)
    deltas = []
    for l, mus in MU_BY_L.items():
        mus = np.asarray(mus, float)
        tail = mus[mus >= 1.0]
        d_l = _ccdf_exponent(tail) if len(tail) >= 25 else float("nan")
        if np.isfinite(d_l):
            deltas.append(d_l)
    delta = float(np.median(deltas)) if deltas else _ccdf_exponent(MU[MU > 0])
    dk = dB / (gamma - 1) if np.isfinite(gamma) and gamma > 1 else float("nan")
    dm = dB / (delta - 1) if np.isfinite(delta) and delta > 1 else float("nan")
    alpha_th = ((delta - 2) / (delta - 1)) * dB if np.isfinite(delta) and delta > 1 else float("nan")
    beta_th = (gamma - 1) / (delta - 1) if np.isfinite(delta) and delta > 1 else float("nan")
    from fractal_dynamics import _logbin_fit
    alpha_emp = _logbin_fit(L, M / np.power(K, beta_th))[0] if np.isfinite(beta_th) else float("nan")
    beta_emp = _logbin_fit(K, M / np.power(L, alpha_th))[0] if np.isfinite(alpha_th) else float("nan")
    rhs = alpha_th + beta_th * dk if np.isfinite(alpha_th) and np.isfinite(beta_th) else float("nan")
    out = dict(name=name, N=N, M_edges=G.number_of_edges(),
               mean_degree=2 * G.number_of_edges() / N,
               l_B=lb[keep].tolist(), NB=NB[keep].tolist(),
               dB=dB, dB_R2=r2, gamma=gamma, delta=delta,
               delta_per_lB=[float(x) for x in deltas], dk=dk, dm=dm,
               alpha_theory=alpha_th, beta_theory=beta_th,
               alpha_empirical=float(alpha_emp), beta_empirical=float(beta_emp),
               rhs=float(rhs), n_boxes=int(len(M)),
               closure_pct=float(abs(dB - rhs) / abs(dB) * 100) if np.isfinite(rhs) else float("nan"))
    if verbose:
        print(f"[{name}] N={N} M={out['M_edges']} <k>={out['mean_degree']:.2f}")
        print(f"  d_B={dB:.3f} (R2={r2:.3f})   gamma={gamma:.2f}   "
              f"delta={delta:.2f}  (per-l_B: {['%.2f' % x for x in deltas]})")
        print(f"  d_k={dk:.3f}  d_m={dm:.3f}")
        print(f"  alpha={alpha_th:.3f} (empirical {alpha_emp:.3f});  "
              f"beta={beta_th:.3f} (empirical {beta_emp:.3f})")
        print(f"  closure alpha+beta*d_k={rhs:.3f} vs d_B={dB:.3f} -> "
              f"{out['closure_pct']:.1f}%  [identity on this route -- not a test]")
    return out

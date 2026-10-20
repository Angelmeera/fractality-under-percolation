"""A12 (Major 4, Minor 6): robustness of d_B to the box-size-grid selection criterion.

For each network we cover with the paper's greedy-degree algorithm at every integer
l_B in a range, then apply several grid-selection criteria to the same N_B(l_B):

  F  = the paper's candidate family (geometric lam in {1.3..3.0} with >=4 sizes,
       step-2 and step-3 sub-ladders, all consecutive sizes, consecutive without 2)
  W  = all contiguous windows of >= 4 consecutive sizes (a much larger family)

  maxR2_F      the submitted rule (maximum R^2 over F)
  maxadjR2_F   maximum adjusted R^2 over F (penalizes few points)
  minSE_F      minimum standard error of the fitted slope over F
  median_F     median d_B over F (no selection)
  maxR2_W      maximum R^2 over W (shows selection bias when the family is large)
  median_W     median d_B over W (no selection)
  all          one fit through every computed size
  structural   the grid fixed by the network's own scale factor, when one exists

Calibration networks with a known answer: square and cubic lattices (open and
periodic), three 2D critical-percolation clusters (d_f/d_min = 1.677) from A6/A6b,
and the FSFN G^B and G^A at t=5 (D_f = ln8/ln3). Real networks (no known answer):
WWW nd.edu (l_B = 2..7 from the submitted analysis), Internet AS, DBLP backbones.
Output: results/a12_grid_criteria.json, logs/a12.log
"""
import sys, json, gzip, time
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, _os.path.join(HERE, "..", "src"))
import numpy as np, networkx as nx
from fast_cover import csr_from_graph, cover_one
from snap_networks import geometric_grid
import yakubo_fujiki as yf

R = _os.path.join(HERE, "results")
ROOT = _os.path.join(HERE, "..")
DF = np.log(8) / np.log(3)


def fit(g, NB, n):
    x = np.log(np.array(g, float)); y = np.log(np.array([NB[l] for l in g], float) / n)
    k = len(g)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef; sse = float(res @ res); sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - sse / sst if sst > 0 else np.nan
    adj = 1 - (1 - r2) * (k - 1) / (k - 2) if k > 2 else np.nan
    se = np.sqrt(sse / (k - 2) / ((x - x.mean()) ** 2).sum()) if k > 2 else np.inf
    return dict(grid=list(g), dB=float(-coef[0]), R2=float(r2), adjR2=float(adj), se=float(se), n=k)


def family(have, NB, lmax):
    F = {}
    for lam in (1.3, 1.4, 1.5, 1.7, 2.0, 2.5, 3.0):
        g = tuple(l for l in geometric_grid(lam, 10 ** 6, l_max=lmax) if l in NB)
        if len(g) >= 4: F[f"geo {lam}"] = g
    for step in (2, 3):
        for start in range(step):
            g = tuple(l for l in have if (l - have[0]) % step == start % step)
            if len(g) >= 3: F[f"step{step}/{g[0]}"] = g
    F["consecutive"] = tuple(have)
    if len(have) > 4: F["consecutive-no2"] = tuple(l for l in have if l != have[0])
    W = {f"win {have[i]}-{have[j-1]}": tuple(have[i:j]) for i in range(len(have)) for j in range(i + 4, len(have) + 1)}
    return F, W


def criteria(NB, n, structural=None):
    have = sorted(NB); lmax = have[-1]
    F, W = family(have, NB, lmax)
    rF = {k: fit(g, NB, n) for k, g in F.items()}
    rW = {k: fit(g, NB, n) for k, g in W.items()}
    pick = lambda rows, key, sign=1: max(rows.items(), key=lambda kv: sign * kv[1][key])
    out = {}
    for name, (lab, r) in (("maxR2_F", pick(rF, "R2")), ("maxadjR2_F", pick(rF, "adjR2")),
                           ("minSE_F", pick(rF, "se", -1)), ("maxR2_W", pick(rW, "R2") if rW else (None, None))):
        if r is not None: out[name] = dict(label=lab, **r)
    out["median_F"] = dict(dB=float(np.median([r["dB"] for r in rF.values()])), n_cand=len(rF))
    if rW:
        v = [r["dB"] for r in rW.values()]
        out["median_W"] = dict(dB=float(np.median(v)), lo=float(min(v)), hi=float(max(v)), n_cand=len(rW))
    out["all"] = fit(tuple(have), NB, n)
    if structural:
        g = tuple(l for l in structural if l in NB)
        out["structural"] = fit(g, NB, n)
    return out


CACHE = f"{R}/a12_NB_cache.json"
try:
    _cache = json.load(open(CACHE))
except Exception:
    _cache = {}


def scan_graph(G, ls, key=None):
    if key and key in _cache:
        c = _cache[key]; return {int(k): v for k, v in c["NB"].items()}, c["n"]
    nodes, idx, A = csr_from_graph(G); n = A.shape[0]
    NB = {}
    for l in ls:
        if n > 20000:
            from snap_networks import cover_stream
            NB[l] = len(cover_stream(A, l))
        else:
            NB[l] = len(cover_one(A, l))
    if key:
        _cache[key] = dict(n=n, NB={str(k): v for k, v in NB.items()}); json.dump(_cache, open(CACHE, "w"))
    return NB, n


def giant(G):
    c = max(nx.connected_components(G), key=len)
    return nx.convert_node_labels_to_integers(G.subgraph(c).copy(), ordering="sorted")


nets = []
t0 = time.time()
# --- calibration networks already covered at every l_B (A6, A6b) ---
for x in json.load(open(f"{R}/a6_grid.json")):
    if "NB" in x:
        NB = {int(k): v for k, v in x["NB"].items()}
        nets.append(dict(name=x["name"] + " (open)", truth=x["truth"], n=x["n"], NB=NB, structural=None))
for x in json.load(open(f"{R}/a6b_periodic.json")):
    n = 40000 if "200" in x["name"] else 34 ** 3
    nets.append(dict(name=x["name"] + " greedy", truth=x["truth"], n=n, NB={int(k): v for k, v in x["NB_greedy"].items()}, structural=None))
# --- FSFN t=5, canonical labelling of the paper ---
for which in ("GB", "GA"):
    G = yf.generate_fsfn(which, 5)
    G = nx.convert_node_labels_to_integers(G) if not all(isinstance(u, int) for u in G.nodes()) else G
    NB, n = scan_graph(G, list(range(2, 41)), key=f"FSFN {which} t=5")
    nets.append(dict(name=f"FSFN {which} t=5", truth=DF, n=n, NB=NB, structural=(2, 4, 10, 28)))
    print(f"FSFN {which}: N={n}, NB(2,4,10,28)={[NB[l] for l in (2,4,10,28)]} [{time.time()-t0:.0f}s]", flush=True)
# --- real networks ---
w = json.load(open(_os.path.join(ROOT, "results", "www_scan.json")))
NBw = {int(k): v["NB"] for k, v in w["per_lB"].items()}
try:
    w7 = json.load(open(_os.path.join(ROOT, "results", "www_l7.json")))
    NBw.update({int(l): v for l, v in zip(w7["main"]["l_B"], w7["main"]["NB"])})
except Exception as e:
    print("www l7:", e)
nets.append(dict(name="WWW nd.edu", truth=None, n=w["N"], NB=NBw, structural=(2, 4, 6)))
# Internet AS and DBLP backbones. Their N_B(l_B) are cached in results/a12_NB_cache.json,
# so the data files are read only if a key is missing from the cache (delete the cache to
# recompute). Data: data/as20000102.txt.gz (data/get_data.py) and data/dblp_weighted.tsv.gz
# (built locally, see data/README.md).
if "AS" in _cache:
    NB, n = scan_graph(None, None, key="AS")
else:
    G = nx.Graph()
    with gzip.open(_os.path.join(ROOT, "data", "as20000102.txt.gz"), "rt") as f:
        for line in f:
            if line.startswith("#"): continue
            a, b = line.split()[:2]
            if a != b: G.add_edge(int(a), int(b))
    G = giant(G); NB, n = scan_graph(G, list(range(2, 10)), key="AS")
nets.append(dict(name="Internet AS", truth=None, n=n, NB=NB, structural=None))
print(f"AS: N={n} [{time.time()-t0:.0f}s]", flush=True)
Gw = None
for wmin in (60, 25):
    key = f"DBLP{wmin}"
    if key in _cache:
        NB, n = scan_graph(None, None, key=key)
    else:
        if Gw is None:
            Gw = []
            with gzip.open(_os.path.join(ROOT, "data", "dblp_weighted.tsv.gz"), "rt") as f:
                for line in f:
                    if line.startswith("#"): continue
                    a, b, c = line.rstrip("\n").split("\t")
                    c = int(c)
                    if c >= 25: Gw.append((a, b, c))
        G = nx.Graph(); G.add_edges_from((a, b) for a, b, c in Gw if c >= wmin and a != b)
        G = giant(G)
        from snap_networks import _approx_diameter
        D = nx.diameter(G) if G.number_of_nodes() < 5000 else _approx_diameter(G)
        NB, n = scan_graph(G, list(range(2, min(D // 2 + 2, 22 if wmin >= 60 else 13))), key=key)
    nets.append(dict(name=f"DBLP w>={wmin}", truth=None, n=n, NB=NB, structural=None))
    print(f"DBLP w>={wmin}: N={n} [{time.time()-t0:.0f}s]", flush=True)

out = []
for net in nets:
    c = criteria(net["NB"], net["n"], net["structural"])
    out.append(dict(name=net["name"], truth=net["truth"], n=net["n"], NB={str(k): v for k, v in net["NB"].items()}, crit=c))
    tr = net["truth"]
    s = " | ".join(f"{k} {v['dB']:.3f}" + (f" ({100*(v['dB']-tr)/tr:+.0f}%)" if tr else "") for k, v in c.items())
    print(f"{net['name']}: {s}", flush=True)
json.dump(out, open(f"{R}/a12_grid_criteria.json", "w"), indent=1)
print(f"done [{time.time()-t0:.0f}s]")

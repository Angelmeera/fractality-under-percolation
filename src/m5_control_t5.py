"""M5 control: does the alpha/beta asymmetry survive a change of mass convention?
Three conventions on identical bond configurations, FSFN G^B t=5, 30 realisations.
  pub  : published -- box = LARGEST connected component of the construction group,
         k = deg(hub) if hub in that component else max deg in it
  mass : total surviving mass -- m counts EVERY surviving member of the group,
         L from the largest component, same k rule
  hub  : largest component but hub required present; k = deg(hub), else box dropped
"""
import sys, os, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, networkx as nx
from yakubo_fujiki import build_parented, hub_seed, analytic_pc_nu, DF_THEORY, LAM, KAPPA, MGEN

T, WHICH = 5, "GB"
edges, N, parent, gen = build_parented(WHICH, T)
G = nx.Graph(); G.add_edges_from(edges)
seeds_by_tau = {tau: hub_seed(parent, gen, N, T, tau) for tau in range(1, T)}
E = list(G.edges()); pc = analytic_pc_nu(WHICH)[0]
print(f"{WHICH} t={T} N={N} M={len(E)} pc={pc:.4f}", flush=True)

def _fit(M_, L_, K_):
    if len(M_) < 12: return np.nan, np.nan
    M_, L_, K_ = (np.array(x, float) for x in (M_, L_, K_))
    ok = (M_ > 1) & (L_ > 0) & (K_ > 0)
    if ok.sum() < 12 or len(set(L_[ok])) < 2 or len(set(K_[ok])) < 2: return np.nan, np.nan
    A = np.column_stack([np.ones(ok.sum()), np.log(L_[ok]), np.log(K_[ok])])
    sol, *_ = np.linalg.lstsq(A, np.log(M_[ok]), rcond=None)
    return float(sol[1]), float(sol[2])

def measure3(H, degp):
    rec = {k: ([], [], []) for k in ("pub", "mass", "hub")}
    for tau, seed in seeds_by_tau.items():
        order = np.argsort(seed, kind="stable"); s_sorted = seed[order]
        for grp in np.split(order, np.flatnonzero(np.diff(s_sorted)) + 1):
            hub = int(seed[grp[0]])
            members = [int(x) for x in grp if H.has_node(x)]
            if len(members) < 3: continue
            sub = H.subgraph(members)
            if sub.number_of_edges() == 0: continue
            comp = max(nx.connected_components(sub), key=len)
            if len(comp) < 3: continue
            cs = sub.subgraph(comp)
            a = next(iter(comp))
            d1 = nx.single_source_shortest_path_length(cs, a)
            d2 = nx.single_source_shortest_path_length(cs, max(d1, key=d1.get))
            L = max(max(d2.values()), 1)
            in_comp = hub in comp
            k = degp[hub] if in_comp else max(degp[x] for x in comp)
            if k < 1: continue
            rec["pub"][0].append(len(comp));   rec["pub"][1].append(L);  rec["pub"][2].append(k)
            rec["mass"][0].append(len(members)); rec["mass"][1].append(L); rec["mass"][2].append(k)
            if in_comp:
                rec["hub"][0].append(len(comp)); rec["hub"][1].append(L); rec["hub"][2].append(degp[hub])
    return {k: _fit(*v) for k, v in rec.items()}

NREAL = int(os.environ.get("NREAL", 30))
rng = np.random.default_rng(4)
ps = np.round(np.arange(0.40, 1.0001, 0.05), 3)
out = []; t0 = time.time()
for p in ps:
    acc = {k: ([], []) for k in ("pub", "mass", "hub")}
    for _ in range(NREAL if p < 1.0 else 1):
        if p >= 1.0:
            H = G; degp = dict(G.degree())
        else:
            keep = rng.random(len(E)) < p
            H = nx.Graph(); H.add_nodes_from(G.nodes())
            H.add_edges_from([e for e, kk in zip(E, keep) if kk])
            H = H.subgraph(max(nx.connected_components(H), key=len)).copy()
            degp = dict(H.degree())
        r = measure3(H, degp)
        for k, (a, b) in r.items():
            if np.isfinite(a): acc[k][0].append(a); acc[k][1].append(b)
    row = dict(p=float(p))
    for k, (av, bv) in acc.items():
        row[f"{k}_alpha"]    = float(np.mean(av)) if av else np.nan
        row[f"{k}_alpha_sd"] = float(np.std(av))  if av else np.nan
        row[f"{k}_beta"]     = float(np.mean(bv)) if bv else np.nan
        row[f"{k}_beta_sd"]  = float(np.std(bv))  if bv else np.nan
    out.append(row)
    print("  p=%.2f  pub %.3f/%.3f | mass %.3f/%.3f | hub %.3f/%.3f   [%.0fs]" % (
        p, row["pub_alpha"], row["pub_beta"], row["mass_alpha"], row["mass_beta"],
        row["hub_alpha"], row["hub_beta"], time.time() - t0), flush=True)
json.dump({"which": WHICH, "t": T, "N": int(N), "pc": pc, "nreal": NREAL, "rows": out,
           "alpha_theory": float(np.log(MGEN / KAPPA) / np.log(LAM)), "beta_theory": 1.0},
          open(os.environ.get("M5_OUT", os.path.join(HERE, "..", "results", "m5_control_t5.json")), "w"), indent=2)
print("saved", flush=True)

"""Re-measure alpha(p), beta(p) on the FSFN with HUB-CENTRED boxes and the hub's
CURRENT degree -- the estimator validated in yakubo_fujiki.hub_centred_exponents().
The previous version of this figure used edge-centred boxes and returned beta~0.2
at p=1 instead of the exact 1.0."""
import sys, os, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.environ.get("RESULTS_DIR", os.path.join(HERE, "..", "results"))
FIGS = os.environ.get("FIGS_DIR", os.path.join(HERE, "..", "figures"))
import numpy as np, networkx as nx
from yakubo_fujiki import build_parented, hub_seed, analytic_pc_nu, DF_THEORY, LAM, KAPPA, MGEN

T, WHICH = 5, "GB"
edges, N, parent, gen = build_parented(WHICH, T)
G = nx.Graph(); G.add_edges_from(edges)
seeds_by_tau = {tau: hub_seed(parent, gen, N, T, tau) for tau in range(1, T)}
E = list(G.edges())
pc = analytic_pc_nu(WHICH)[0]
print(f"{WHICH} t={T} N={N} M={len(E)}  analytic pc={pc:.4f}", flush=True)

def measure(H, degp):
    """alpha,beta from hub-centred boxes on the percolated graph H."""
    M_, L_, K_ = [], [], []
    for tau, seed in seeds_by_tau.items():
        # group surviving nodes by their construction hub
        order = np.argsort(seed, kind="stable")
        s_sorted = seed[order]
        bounds = np.flatnonzero(np.diff(s_sorted)) + 1
        for grp in np.split(order, bounds):
            hub = int(seed[grp[0]])
            members = [int(x) for x in grp if H.has_node(x)]
            if len(members) < 3:
                continue
            sub = H.subgraph(members)
            if sub.number_of_edges() == 0:
                continue
            comp = max(nx.connected_components(sub), key=len)
            if len(comp) < 3:
                continue
            cs = sub.subgraph(comp)
            # diameter by double sweep (exact on these tree-like pieces)
            a = next(iter(comp))
            d1 = nx.single_source_shortest_path_length(cs, a)
            far = max(d1, key=d1.get)
            d2 = nx.single_source_shortest_path_length(cs, far)
            L = max(max(d2.values()), 1)
            k = degp[hub] if hub in comp else max(degp[x] for x in comp)
            if k < 1:
                continue
            M_.append(len(comp)); L_.append(L); K_.append(k)
    if len(M_) < 12:
        return np.nan, np.nan
    M_, L_, K_ = (np.array(x, float) for x in (M_, L_, K_))
    ok = (M_ > 1) & (L_ > 0) & (K_ > 0)
    if ok.sum() < 12 or len(set(L_[ok])) < 2 or len(set(K_[ok])) < 2:
        return np.nan, np.nan
    A = np.column_stack([np.ones(ok.sum()), np.log(L_[ok]), np.log(K_[ok])])
    sol, *_ = np.linalg.lstsq(A, np.log(M_[ok]), rcond=None)
    return float(sol[1]), float(sol[2])

# Realisations per p. The intact network (p = 1) is deterministic, so one pass
# is exact there; below p = 1 the spread is over bond configurations.
NREAL = int(os.environ.get('NREAL', 30))
rng = np.random.default_rng(4)
ps = np.round(np.arange(0.40, 1.0001, 0.05), 3)
out = []
t0 = time.time()
for p in ps:
    av, bv = [], []
    for _ in range(NREAL if p < 1.0 else 1):
        if p >= 1.0:
            H = G; degp = dict(G.degree())
        else:
            keep = rng.random(len(E)) < p
            H = nx.Graph(); H.add_nodes_from(G.nodes())
            H.add_edges_from([e for e, k in zip(E, keep) if k])
            gc = max(nx.connected_components(H), key=len)
            H = H.subgraph(gc).copy()
            degp = dict(H.degree())
        a, b = measure(H, degp)
        if np.isfinite(a):
            av.append(a); bv.append(b)
    out.append(dict(p=float(p),
                    alpha=float(np.mean(av)) if av else np.nan,
                    alpha_sd=float(np.std(av)) if av else np.nan,
                    beta=float(np.mean(bv)) if bv else np.nan,
                    beta_sd=float(np.std(bv)) if bv else np.nan))
    print(f"  p={p:.2f}: alpha={out[-1]['alpha']:.3f} beta={out[-1]['beta']:.3f}  [{time.time()-t0:.0f}s]", flush=True)
json.dump({"which":WHICH,"t":T,"N":int(N),"pc":pc,"rows":out,
           "alpha_theory":float(np.log(MGEN/KAPPA)/np.log(LAM)),"beta_theory":1.0},
          open(os.path.join(RES, "results_fsfn_alphabeta.json"), "w"), indent=2)
print("saved", os.path.join(RES, "results_fsfn_alphabeta.json"), flush=True)

# ---------------------------------------------------------------- the figure
# Drawn here rather than in a separate ad-hoc script, so that the command in
# MASTER_README ("python redo_fig2.py") really does reproduce fig3.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 140, "font.size": 10, "axes.grid": True,
                     "grid.alpha": .3, "axes.axisbelow": True,
                     "legend.frameon": False})
pp = np.array([r["p"] for r in out], float)
al = np.array([r["alpha"] for r in out], float)
be = np.array([r["beta"] for r in out], float)
als = np.array([r["alpha_sd"] for r in out], float)
bes = np.array([r["beta_sd"] for r in out], float)
a_th = float(np.log(MGEN / KAPPA) / np.log(LAM))
dk = float(np.log(KAPPA) / np.log(LAM))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].errorbar(pp, al, yerr=als, marker="o", ms=6, lw=1.6, color="#0072B2",
               mec="white", mew=.7, capsize=2, label=r"$\alpha(p)$")
ax[0].errorbar(pp, be, yerr=bes, marker="s", ms=6, lw=1.6, ls="--",
               color="#E69F00", mec="white", mew=.7, capsize=2,
               label=r"$\beta(p)$")
ax[0].axhline(a_th, color="#0072B2", lw=.9, ls=":")
ax[0].axhline(1.0, color="#E69F00", lw=.9, ls=":")
ax[0].axvline(pc, color="k", lw=.9, ls="-.")
ax[0].text(0.02, a_th, r"exact $\alpha=\ln 4/\ln 3$", fontsize=7.5,
           va="bottom", ha="left", transform=ax[0].get_yaxis_transform())
ax[0].text(0.02, 1.0, r"exact $\beta=1$", fontsize=7.5, va="bottom",
           ha="left", transform=ax[0].get_yaxis_transform())
ax[0].text(pc, 0.02, r" $p_c$", fontsize=8, transform=ax[0].get_xaxis_transform())
ax[0].set(xlabel="$p$", ylabel="exponent",
          title=r"(a) hub-centred boxes, current hub degree")
ax[0].legend(fontsize=9)
ax[1].plot(pp, al + be * dk, marker="D", ms=6, lw=1.6, color="#009E73",
           mec="white", mew=.7, label=r"$\alpha+\beta d_k$")
ax[1].axhline(DF_THEORY, color="k", lw=1, ls=":",
              label=r"exact $d_B=\ln 8/\ln 3$")
ax[1].axvline(pc, color="k", lw=.9, ls="-.")
ax[1].set(xlabel="$p$", ylabel=r"$\alpha+\beta d_k$",
          title="(b) the relation is not recovered below $p=1$")
ax[1].legend(fontsize=9)
fig.tight_layout()
os.makedirs(FIGS, exist_ok=True)
fp = os.path.join(FIGS, "fig3_alpha_beta_corrected.png")
fig.savefig(fp, bbox_inches="tight")
print("wrote", fp, flush=True)

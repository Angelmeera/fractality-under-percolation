"""
Paper 1 -- Fractality Under Network Dynamics, on the Yakubo-Fujiki FSFN.

SCOPE: this script produces fig1 (susceptibility + d_B(p)) and fig4 (clustering
comparison GA vs GB). It no longer measures alpha(p), beta(p): that analysis used
edge-centred boxes, which return beta ~ 0.2 on the intact network where the exact
value is 1. The corrected measurement -- hub-centred boxes with the hub's current
degree -- lives in redo_fig2.py and produces fig3_alpha_beta_corrected.png.
This is the model that BOTH is mass-law self-similar (scaling relation closes at p=1)
AND has a genuine interior p_c with ANALYTICAL p_c, nu, beta to validate against.

Compares GA (clustered) vs GB (unclustered) -- the plan's decoupling question plus
Yakubo-Fujiki's clustering-vs-robustness angle.
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, networkx as nx
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from numpy.linalg import lstsq
from yakubo_fujiki import (generate_fsfn, analytic_pc_nu, hub_centred_exponents,
                           DF_THEORY, GAMMA_THEORY)
from fast_cover import csr_from_graph, cover_one
from fractal_dynamics import (all_pairs_dist, measure_dB_from_D, greedy_box_cover_D,
                              bond_percolation, giant_component, susceptibility)

# Portable output paths: the figures must land where LaTeX compiles them from,
# so that the reproduce command in MASTER_README really does refresh the paper.
FIG = os.environ.get("FIGS_DIR", os.path.join(HERE, "..", "figures"))
RES = os.environ.get("RESULTS_DIR", os.path.join(HERE, "..", "results"))
os.makedirs(FIG, exist_ok=True); os.makedirs(RES, exist_ok=True)
plt.rcParams.update({"figure.dpi":140,"font.size":10,"axes.grid":True,"grid.alpha":.3,"axes.axisbelow":True})
BLUE,RED,GREEN,GREY,PURP="#2563eb","#dc2626","#059669","#6b7280","#7c3aed"
LB=[2,4,10,28]
res={}; t0=time.time()

def lcc(G): return max((G.subgraph(c).copy() for c in nx.connected_components(G)),key=len) if G.number_of_edges() else G
def gamma_ccdf(G):
    ks=np.sort(np.array([d for _,d in G.degree() if d>=2],float))
    if len(ks)<8: return np.nan
    u=np.unique(ks); cc=np.array([(ks>=k).mean() for k in u]); m=cc>0
    return 1-np.polyfit(np.log(u[m]),np.log(cc[m]),1)[0]

def dB_boxcount(H):
    """d_B of H on the commensurate grid, with the canonical greedy cover of
    Sec. 2.3: seeds in decreasing degree, ties broken on str(node)
    (fast_cover.csr_from_graph + cover_one), the same order as every other
    cover in the paper. (Before 25 Sep 2026 this used np.argsort over the
    graph's iteration order, which moved G^A's intact d_B from 1.61 to 1.76.)"""
    N=H.number_of_nodes()
    if N<12: return np.nan
    nodes,D=all_pairs_dist(H)
    diam=int(D[D<10**9].max()); grid=[l for l in LB if l<=diam+1]
    if len(grid)<3: grid=LB[:3]
    _,_,A=csr_from_graph(H)
    NB=np.array([len(cover_one(A,l)) for l in grid],float)
    return float(-np.polyfit(np.log(grid),np.log(NB/N),1)[0])

for which in ("GB","GA"):
    print(f"\n########## {which} ##########")
    pc_an,nu_an=analytic_pc_nu(which)
    kin=hub_centred_exponents(which,5,verbose=False)
    g0=gamma_ccdf(generate_fsfn(which,5)); dk0=kin['dB']/(g0-1)
    close0=abs(kin['dB']-(kin['alpha']+kin['beta']*dk0))/kin['dB']*100
    print(f" BASELINE: dB={kin['dB']:.3f} alpha={kin['alpha']:.3f} beta={kin['beta']:.3f} "
          f"closure={kin['closure_pct']:.2f}%  (analytic pc={pc_an:.4f} nu={nu_an:.4f})")

    # ---- Task 1: susceptibility -> pc_sim, and dB(p) ----
    Gt=generate_fsfn(which,4); rng=np.random.default_rng(0)
    ps=np.linspace(0.2,1.0,33)
    chi=susceptibility(Gt,ps,n_trials=30,rng=rng)
    pc_sim=float(ps[int(np.argmax(chi))])
    dBp=np.full(len(ps),np.nan); dBsd=np.full(len(ps),np.nan)
    for i,p in enumerate(ps):
        dv=[]
        for _ in range(6):
            gc=giant_component(bond_percolation(Gt,p,rng))
            if gc.number_of_nodes()>12:
                d=dB_boxcount(gc)
                if np.isfinite(d): dv.append(d)
        if dv: dBp[i]=np.mean(dv); dBsd[i]=np.std(dv)
    print(f" Task1: pc_sim={pc_sim:.3f} vs analytic {pc_an:.4f}  dB(1)={dBp[-1]:.2f}")
    res[which]={"pc_analytic":pc_an,"nu_analytic":nu_an,"pc_sim":pc_sim,
                "baseline":{"dB":kin['dB'],"alpha":kin['alpha'],"beta":kin['beta'],"gamma":g0,"dk":dk0,"closure_pct":close0},
                "ps":ps.tolist(),"chi":chi.tolist(),"dBp":dBp.tolist(),"dBsd":dBsd.tolist()}

# ---- Figures ----
# Drawn at print size by make_fig1_fig4.py (which can also redraw them from
# results/results_yf.json alone): fig1_pc_dB (paper Fig. 1) and
# fig4_clustering (paper Fig. 3), PNG + EPS.
json.dump(res, open(os.path.join(RES, "results_yf.json"), "w"), indent=2)
import make_fig1_fig4
make_fig1_fig4.draw(res)
print(f"\nDONE {time.time()-t0:.0f}s")

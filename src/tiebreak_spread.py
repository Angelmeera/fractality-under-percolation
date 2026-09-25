"""Sensitivity of the intact FSFN box dimension to the greedy seeding tie-break.

    python src/tiebreak_spread.py --t 4 --orders 30 --out results/tiebreak_spread.json

For G^A and G^B at generation t, covers the intact network on l_B = 2, 4, 10, 28
with the greedy diameter rule, seeding by decreasing degree and breaking degree
ties (i) on str(node) (the canonical order used throughout the paper), (ii) on the
numeric label, (iii) by NumPy's default (unstable) argsort, which the pre-25-Sep
run_yf.py used, and (iv) over `--orders` random orders within degree classes.
Backs the numbers quoted in Sec. 4.5: G^A 1.70 +- 0.07 (N_B(28) = 10-16) against
G^B 1.848 +- 0.003 over random orders.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from yakubo_fujiki import generate_fsfn
from fractal_dynamics import all_pairs_dist

LB = [2, 4, 10, 28]


def greedy(D, order, l_B):
    thr = l_B - 1
    ok = []
    for v in order:
        for b in ok:
            if b[v]:
                b &= (D[v] <= thr)
                break
        else:
            ok.append(D[v] <= thr)
    return len(ok)


def fit(NB, N):
    return float(-np.polyfit(np.log(LB), np.log(np.asarray(NB, float) / N), 1)[0])


ap = argparse.ArgumentParser()
ap.add_argument("--t", type=int, default=4)
ap.add_argument("--orders", type=int, default=30)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--out", required=True)
a = ap.parse_args()
res = {"argv": sys.argv[1:], "t": a.t, "l_B": LB, "orders": a.orders, "seed": a.seed, "generators": {}}
for w in ("GB", "GA"):
    G = generate_fsfn(w, a.t)
    N = G.number_of_nodes()
    nodes, D = all_pairs_dist(G)
    deg = np.array([G.degree(u) for u in nodes])
    pos = {u: i for i, u in enumerate(nodes)}
    fixed = {
        "string": [pos[u] for u in sorted(nodes, key=lambda n: (-G.degree(n), str(n)))],
        "numeric": [pos[u] for u in sorted(nodes, key=lambda n: (-G.degree(n), n))],
        "numpy_argsort": list(np.argsort(-deg.astype(float))),
    }
    out = {"N": N}
    for k, o in fixed.items():
        NB = [greedy(D, o, l) for l in LB]
        out[k] = {"NB": NB, "dB": fit(NB, N)}
    rng = np.random.default_rng(a.seed)
    rows, ds = [], []
    for _ in range(a.orders):
        o = np.lexsort((rng.random(N), -deg))
        NB = [greedy(D, o, l) for l in LB]
        rows.append(NB)
        ds.append(fit(NB, N))
    rows = np.array(rows)
    ds = np.array(ds)
    out["random"] = {"dB_mean": float(ds.mean()), "dB_sd": float(ds.std(ddof=1)),
                     "dB_min": float(ds.min()), "dB_max": float(ds.max()),
                     "NB_min": rows.min(0).tolist(), "NB_max": rows.max(0).tolist()}
    res["generators"][w] = out
    print(w, {k: (v["NB"], round(v["dB"], 4)) for k, v in out.items() if k in fixed},
          "random d_B %.4f +- %.4f [%.3f, %.3f], N_B min %s max %s" % (
              ds.mean(), ds.std(ddof=1), ds.min(), ds.max(),
              rows.min(0).tolist(), rows.max(0).tolist()), flush=True)
with open(a.out, "w") as fh:
    json.dump(res, fh, indent=1)
print("wrote", a.out)

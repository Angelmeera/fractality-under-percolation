"""MEMB-only covers of the FSFN, for the MEMB rows of the covering-comparison table.

    python src/memb_fsfn.py --t 4 5 --r-b 1 4 13 --key commensurate --out results/covering_memb.json
    python src/memb_fsfn.py --t 5 --r-b 1 2 3 4 5 6 9 --key integer_ladder_t5 --out results/covering_memb.json

Each call stores its records under --key in --out, keeping what is already there.

At r_B = (lambda^tau - 1)/2 = 1, 4, 13 MEMB returns exactly N_{t-1}, N_{t-2}, N_{t-3}
(the coarse-grained networks); on an arbitrary integer ladder it returns a staircase.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from yakubo_fujiki import generate_fsfn, fsfn_size
from covering import cover_memb_lean, _fit
from fast_cover import csr_from_graph

ap = argparse.ArgumentParser()
ap.add_argument("--t", type=int, nargs="+", required=True)
ap.add_argument("--r-b", type=int, nargs="+", required=True)
ap.add_argument("--generator", default="GB")
ap.add_argument("--out", required=True)
ap.add_argument("--key", required=True)
a = ap.parse_args()
res = []
for t in a.t:
    G = generate_fsfn(a.generator, t)
    nodes, idx, A = csr_from_graph(G)
    N = len(nodes)
    NB = []
    for r in a.r_b:
        boxes, centres = cover_memb_lean(A, r)
        NB.append(len(boxes))
        print(f"{a.generator} t={t} r_B={r}: N_B={len(boxes)}", flush=True)
    lbe = [2 * r + 1 for r in a.r_b]
    d, r2 = _fit(lbe, NB, N)
    print(f"  d_B={d:.4f} R2={r2:.5f}", flush=True)
    res.append({"name": f"FSFN-{a.generator}-t{t}", "N": N,
                "memb": {"r_B": a.r_b, "l_B_equiv": lbe, "NB": NB, "dB": d, "R2": r2},
                "exact_dB": float(np.log(8) / np.log(3))})
doc = {}
if os.path.exists(a.out):
    with open(a.out) as fh:
        doc = json.load(fh)
doc.setdefault("argv", {})[a.key] = sys.argv[1:]
doc["note"] = ("MEMB rows of the covering-comparison table. At r_B = (3^tau - 1)/2 "
               "MEMB returns N_{t-1}, N_{t-2}, N_{t-3} exactly.")
doc[a.key] = res
with open(a.out, "w") as fh:
    json.dump(doc, fh, indent=1)
print("wrote", a.out)

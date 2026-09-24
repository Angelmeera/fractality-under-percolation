"""Compare the four box-covering algorithms on networks where the answer is
known (FSFN) and on the two empirical networks.

    python run_covering.py --fsfn 4 5 --dblp 60 --out ../results/results_covering.json
    python run_covering.py --www ../data/web-NotreDame.txt.gz --memb-only

The FSFN rows are the calibration: d_B = ln 8 / ln 3 = 1.8928 exactly, so an
algorithm that misses it there cannot be trusted on a real network.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from covering import compare, cover_memb, _fit, _balls_all
from fast_cover import csr_from_graph

DF = np.log(8) / np.log(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fsfn", type=int, nargs="*", default=[])
    ap.add_argument("--generator", default="GB")
    ap.add_argument("--dblp", type=int, nargs="*", default=[],
                    help="weak-tie thresholds")
    ap.add_argument("--dblp-path", default="dblp_weighted.tsv.gz")
    ap.add_argument("--www", default=None)
    ap.add_argument("--l-b", type=int, nargs="+", default=None)
    ap.add_argument("--r-b", type=int, nargs="+", default=None)
    ap.add_argument("--memb-only", action="store_true")
    ap.add_argument("--cbb-reps", type=int, default=3)
    ap.add_argument("--out", default="results_covering.json")
    args = ap.parse_args()

    res = []

    def emit(rec):
        res.append(rec)
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1, default=float)

    for t in args.fsfn:
        from yakubo_fujiki import generate_fsfn
        G = generate_fsfn(args.generator, t)
        lb = args.l_b or (2, 4, 10, 28)
        rb = args.r_b or (1, 3, 9)
        print(f"\n=== FSFN {args.generator} t={t}: N={G.number_of_nodes()} "
              f"(exact d_B={DF:.4f}) ===", flush=True)
        rec = compare(G, lb, r_B_values=rb, cbb_reps=args.cbb_reps,
                      name=f"FSFN-{args.generator}-t{t}")
        rec["exact_dB"] = DF
        emit(rec)

    for thr in args.dblp:
        from real_network import load_weighted_edgelist, backbone
        from snap_networks import giant
        Gw = load_weighted_edgelist(args.dblp_path)
        B = giant(backbone(Gw, thr))
        lb = args.l_b or (2, 3, 5, 9, 17)
        rb = args.r_b or (1, 2, 4, 8)
        print(f"\n=== DBLP w>={thr}: N={B.number_of_nodes()} ===", flush=True)
        emit(compare(B, lb, r_B_values=rb, cbb_reps=args.cbb_reps,
                     name=f"DBLP(w>={thr})"))

    if args.www:
        from snap_networks import load_snap_edgelist, giant
        G = giant(load_snap_edgelist(args.www))
        nodes, idx, A = csr_from_graph(G)
        N = len(nodes)
        rb = args.r_b or (1, 2, 3)
        print(f"\n=== WWW: N={N} -- MEMB at r_B={rb} ===", flush=True)
        NB, lbe = [], []
        for r in rb:
            balls = _balls_all(A, r)
            boxes, centres = cover_memb(A, r, balls=balls, verbose=True)
            del balls
            NB.append(len(boxes)); lbe.append(2 * r + 1)
            print(f"  MEMB r_B={r}: N_B={len(boxes)}  (l_B equiv {2*r+1})",
                  flush=True)
        d, r2 = _fit(lbe, NB, N)
        print(f"  MEMB d_B={d:.3f} (R2={r2:.4f})", flush=True)
        emit({"name": "WWW", "N": N, "M": G.number_of_edges(),
              "memb": {"r_B": list(rb), "l_B_equiv": lbe, "NB": NB,
                       "dB": d, "R2": r2}})

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

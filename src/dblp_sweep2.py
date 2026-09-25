"""Regenerate the DBLP weak-tie sweep (paper Table 6) with the canonical
seeding order and the streaming cover.

Two reasons this replaces the earlier `run_dblp.py --sweep` numbers:

 1. The greedy seeding order used to break degree ties on the graph's iteration
    order, which for string node labels depends on PYTHONHASHSEED and therefore
    changed between processes: on the w>=25 backbone N_B(l_B=2) came out 24,935
    in one run and 24,962 in another. fast_cover.csr_from_graph now breaks ties
    on str(node), so covers are reproducible; every box count in the paper had to
    be regenerated under that order.
 2. It also reports d_k from Song's renormalisation, which the old sweep did not
    have, and the power-law-vs-exponential comparison for each backbone.

The box-size grid is held fixed at the paper's l_B in {2,3,4,6,9} for every row,
so the rows stay comparable to each other and to the published table.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from real_network import load_weighted_edgelist, backbone
from snap_networks import giant, analyse

GRID = tuple(int(x) for x in
             os.environ.get("SWEEP_GRID", "2 3 4 6 9").split())
THRESHOLDS = [int(x) for x in
              os.environ.get("SWEEP_THRESHOLDS", "25 30 35 40 45 50 60").split()]
EDGELIST = os.environ.get("SWEEP_EDGELIST", "dblp_weighted.tsv.gz")
OUT = os.environ.get("SWEEP_OUT", "results_dblp_sweep_canonical.json")
JOBS = int(os.environ.get("SWEEP_JOBS", "2"))


def main():
    print(f"loading {EDGELIST} ...", flush=True)
    Gw = load_weighted_edgelist(EDGELIST)
    print(f"  full weighted graph: N={Gw.number_of_nodes()} "
          f"M={Gw.number_of_edges()}", flush=True)
    rows = []
    for thr in THRESHOLDS:
        B = giant(backbone(Gw, thr))
        if B.number_of_nodes() < 60:
            print(f"  w>={thr}: backbone too small ({B.number_of_nodes()})",
                  flush=True)
            continue
        print(f"\n-- w>={thr}: N={B.number_of_nodes()} M={B.number_of_edges()}",
              flush=True)
        r = analyse(B, f"DBLP(w>={thr})", GRID, n_jobs=JOBS)
        r["threshold"] = thr
        rows.append(r)
        with open(OUT, "w") as fh:
            json.dump({"grid": list(GRID), "rows": rows}, fh, indent=1,
                      default=float)
    print("\n" + "=" * 78)
    print(f"{'w>=':>5} {'N':>7} {'M':>7} {'<k>':>5} {'d_B':>7} {'R2':>7} "
          f"{'gamma':>6} {'delta':>6} {'d_k(ren)':>9} {'d_k(gam)':>9}")
    for r in rows:
        print(f"{r['threshold']:>5} {r['N']:>7} {r['M_edges']:>7} "
              f"{r['mean_degree']:>5.2f} {r['dB']:>7.3f} {r['dB_R2']:>7.4f} "
              f"{r['gamma']:>6.2f} {r['delta']:>6.2f} "
              f"{r['dk_renorm']:>9.3f} {r['dk_from_gamma']:>9.3f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

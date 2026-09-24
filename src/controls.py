"""Non-fractal controls.

Song, Havlin & Makse (Nature 433, 392 (2005); Nature Physics 2, 275 (2006))
distinguish fractal from non-fractal networks by the SHAPE of the box-counting
curve: fractal networks give a straight line for N_B(l_B)/N on log-log axes,
non-fractal ones on log-linear axes. They report the Internet as non-fractal,
and models grown by pure preferential attachment likewise -- fractality comes
from hub-hub *repulsion*, which preferential attachment does not produce.

These controls need no download, so they are run regardless of dataset access:

  BA m=2   scale-free (gamma ~ 3) but grown by preferential attachment
  BA m=1   the same, as a tree -- extreme hub-hub attraction
  ER       neither scale-free nor fractal, small-world by construction

If our pipeline returns a convincing power law and a stable d_B for any of
these, then the pipeline manufactures power laws and every d_B in the paper is
suspect. This is the cheapest available check that the method can say "no".
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import networkx as nx

from snap_networks import analyse, grid_scan, giant, _approx_diameter

OUT = os.environ.get("CONTROLS_OUT", "results_nonfractal_controls.json")

CASES = [
    ("BA m=2 N=6474", lambda: nx.barabasi_albert_graph(6474, 2, seed=11)),
    ("BA m=1 N=6474 (tree)", lambda: nx.barabasi_albert_graph(6474, 1, seed=12)),
    ("ER <k>=4 N=6474", lambda: nx.gnm_random_graph(6474, 12948, seed=13)),
]


def main():
    out = {}
    for label, make in CASES:
        G = giant(make())
        d = _approx_diameter(G)
        print(f"\n===== {label}: N={G.number_of_nodes()} "
              f"M={G.number_of_edges()} diam~{d}", flush=True)
        rows, best, cover = grid_scan(G, lambdas=(1.6, 2.0, 2.5, 3.0), n_jobs=2,
                                      l_max=max(6, d // 2 + 1))
        grid = tuple(best["grid"]) if best else (2, 3, 4, 6, 9)
        r = analyse(G, label, grid, n_jobs=2, cover=cover)
        out[label] = {"grid_scan": rows, "grid": list(grid), "main": r}
        with open(OUT, "w") as fh:
            json.dump(out, fh, indent=1, default=float)
    print("\nDONE")


if __name__ == "__main__":
    main()

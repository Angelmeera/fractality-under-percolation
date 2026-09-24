"""
Run the second and third empirical networks.

    python run_snap.py --www  data/web-NotreDame.txt.gz     # WWW  (Fronczak's own)
    python run_snap.py --as   data/as20000102.txt.gz        # Internet AS control
    python run_snap.py --dblp dblp_weighted.tsv.gz --min-joint 60
    python run_snap.py --fsfn 4                             # calibration on a
                                                            # network with exact
                                                            # exponents

Useful flags
    --scan            choose the l_B grid by maximising the R^2 of N_B(l_B)
                      instead of using a hand-picked grid (recommended; see
                      snap_networks.grid_scan for why the grid matters)
    --l-b 2 4 10 28   use this grid explicitly
    --l-max 24        cap the grid (cost grows steeply with the largest l_B)
    --kcore 2 3 4 5   also analyse k-core reductions, so d_B can be checked for
                      stability against network size the way the weak-tie sweep
                      does for DBLP
    --jobs 2          processes (the l_B values are independent)
    --out results_snap.json

Everything it prints is also written to the JSON, including the full grid scan,
so nothing is quoted without the reader being able to see how much it moved.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import networkx as nx

from snap_networks import (load_snap_edgelist, giant, kcore_series, analyse,
                           grid_scan, geometric_grid, _approx_diameter)


def _load(args):
    if args.www:
        _need(args.www, "https://snap.stanford.edu/data/web-NotreDame.txt.gz")
        print(f"WWW (nd.edu web graph) from {args.www}")
        return "WWW", giant(load_snap_edgelist(args.www))
    if args.as_graph:
        _need(args.as_graph, "https://snap.stanford.edu/data/as20000102.txt.gz")
        print(f"Internet AS-level graph from {args.as_graph}")
        return "Internet-AS", giant(load_snap_edgelist(args.as_graph))
    if args.dblp:
        from real_network import load_weighted_edgelist, backbone
        print(f"DBLP weighted coauthorship from {args.dblp}")
        Gw = load_weighted_edgelist(args.dblp)
        B = giant(backbone(Gw, args.min_joint))
        return f"DBLP(w>={args.min_joint})", B
    if args.fsfn is not None:
        from yakubo_fujiki import generate_fsfn
        G = generate_fsfn(args.generator, args.fsfn)
        print(f"FSFN {args.generator} t={args.fsfn}: N={G.number_of_nodes()} "
              f"-- CALIBRATION: exact d_B=1.8928, d_k=0.6309, alpha=1.2619, beta=1")
        return f"FSFN-{args.generator}-t{args.fsfn}", G
    raise SystemExit("choose one of --www / --as / --dblp / --fsfn")


def _need(path, url):
    if not os.path.exists(path):
        raise SystemExit(
            f"\nMissing dataset: {path}\n"
            f"Download it first (Windows PowerShell; curl.exe ships with "
            f"Windows 10/11):\n\n"
            f"    curl.exe -L -o {os.path.basename(path)} {url}\n")


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_argument_group("network (pick one)")
    src.add_argument("--www", metavar="PATH",
                     help="SNAP web-NotreDame edge list (.txt/.txt.gz)")
    src.add_argument("--as", dest="as_graph", metavar="PATH",
                     help="SNAP autonomous-systems edge list -- the control")
    src.add_argument("--dblp", metavar="PATH",
                     help="cached weighted DBLP coauthorship graph")
    src.add_argument("--fsfn", type=int, metavar="T",
                     help="calibrate on the FSFN at generation T")
    ap.add_argument("--generator", default="GB", choices=("GA", "GB"))
    ap.add_argument("--min-joint", type=int, default=60)
    ap.add_argument("--l-b", type=int, nargs="+", default=None,
                    help="explicit l_B grid")
    ap.add_argument("--l-max", type=int, default=None,
                    help="cap on the largest l_B")
    ap.add_argument("--scan", action="store_true",
                    help="pick the grid by maximising R2 of N_B(l_B)")
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=(1.6, 2.0, 2.5, 3.0, 3.5, 4.0),
                    help="candidate scale factors for --scan")
    ap.add_argument("--min-points", type=int, default=4,
                    help="minimum grid points for a candidate scale factor")
    ap.add_argument("--kcore", type=int, nargs="*", default=None,
                    help="also analyse these k-cores")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--out", default="results_snap.json")
    args = ap.parse_args()

    name, G = _load(args)
    print(f"  giant component: N={G.number_of_nodes()} M={G.number_of_edges()} "
          f"<k>={2 * G.number_of_edges() / G.number_of_nodes():.2f}", flush=True)
    diam = _approx_diameter(G)
    print(f"  diameter (double-sweep lower bound) = {diam}", flush=True)

    out = {"name": name, "N": G.number_of_nodes(), "M": G.number_of_edges(),
           "diameter_lb": diam, "argv": sys.argv[1:]}

    grid = tuple(args.l_b) if args.l_b else None
    cover = None
    if args.scan:
        print("\n-- l_B grid scan (fit quality vs scale factor) --", flush=True)
        rows, best, cover = grid_scan(G, lambdas=args.lambdas,
                                      n_jobs=args.jobs, chunk=args.chunk,
                                      l_max=args.l_max,
                                      min_points=args.min_points)
        out["grid_scan"] = rows
        if best:
            grid = tuple(best["grid"])
            out["grid_chosen"] = list(grid)
            out["grid_chosen_lambda"] = best["lam"]
    if grid is None:
        grid = geometric_grid(2.0, diam, l_max=args.l_max)
        print(f"  (no grid given; defaulting to {grid})")

    print(f"\n-- full analysis on l_B = {grid} --", flush=True)
    out["main"] = analyse(G, name, grid, n_jobs=args.jobs, chunk=args.chunk,
                          cover=cover)

    if args.kcore:
        print("\n-- k-core reductions (is d_B stable against size?) --",
              flush=True)
        out["kcore"] = []
        for k, H in kcore_series(G, args.kcore):
            g = grid
            r = analyse(H, f"{name} {k}-core", g, n_jobs=args.jobs,
                        chunk=args.chunk)
            r["k"] = k
            out["kcore"].append(r)

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

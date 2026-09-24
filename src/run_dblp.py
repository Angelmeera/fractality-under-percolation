#!/usr/bin/env python3
"""
Run the Paper-1 analysis on the REAL DBLP coauthorship network.

Reproduces Fronczak et al. (2024) Methods: build the coauthorship network,
weight each edge by the number of joint papers, keep only edges of weight
>= 25 (weak-tie removal), take the largest component -- that backbone is
naturally fractal (~2.5k nodes / ~3.2k edges in their run) -- then measure
d_B, gamma, delta, the microscopic exponents, and follow them through bond
percolation.

USAGE (run on a machine with internet; this sandbox has no dataset access)

  # option A -- official DBLP XML dump (~1 GB gz, no account needed)
  wget https://dblp.org/xml/dblp.xml.gz
  python run_dblp.py --xml dblp.xml.gz

  # option B -- the exact dataset Fronczak used: AMiner DBLP-Citation-network V12
  #   from https://www.aminer.org/citation   (dblp.v12.json)
  python run_dblp.py --v12 dblp.v12.json

  # option C -- any weighted edge list, "author_a author_b n_joint_papers"
  python run_dblp.py --edgelist my_coauthors.txt

  # quick smoke test of the whole pipeline on a surrogate (no download)
  python run_dblp.py --surrogate

Options: --min-joint 25 (threshold), --percolate (also run p-sweep),
         --out results_dblp.json
"""
import argparse, json, os, sys
import numpy as np
import networkx as nx
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from real_network import (dblp_from_xml, dblp_from_v12_json, load_weighted_edgelist,
                          backbone, fronczak_protocol, percolate_real,
                          threshold_sweep, save_weighted)
from fractal_dynamics import bond_percolation, giant_component, susceptibility


def surrogate_coauthor(n_communities=90, size=14, p_in=0.45, seed=1):
    """A stand-in for the thresholded DBLP backbone: many small dense research
    groups sparsely bridged in a hierarchy. NOT a scientific result -- it exists
    only so the pipeline can be smoke-tested without the real download."""
    rng = np.random.default_rng(seed)
    G = nx.Graph(); nid = 0; hubs = []
    for _ in range(n_communities):
        members = list(range(nid, nid + size)); nid += size
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if rng.random() < p_in:
                    G.add_edge(members[i], members[j])
        hubs.append(members[0])
    # sparse tree-like bridging between groups -> keeps it fractal-ish, not small-world
    for i in range(1, len(hubs)):
        G.add_edge(hubs[i], hubs[rng.integers(0, i)])
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return nx.convert_node_labels_to_integers(G)


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--xml"); src.add_argument("--v12")
    src.add_argument("--edgelist"); src.add_argument("--surrogate", action="store_true")
    ap.add_argument("--min-joint", type=int, default=25)
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--single-pass", action="store_true",
                    help="disable the author prefilter (needs much more RAM)")
    ap.add_argument("--percolate", action="store_true")
    ap.add_argument("--save-weighted", metavar="PATH",
                    help="save the parsed weighted graph (e.g. dblp_weighted.tsv.gz) "
                         "so you never repeat the XML parse")
    ap.add_argument("--jobs", type=int, default=None,
                    help="parallel worker processes for box covering "
                         "(default: min(#l_B values, CPU count))")
    ap.add_argument("--l-b", type=int, nargs="+", default=[2, 3, 4, 6, 9, 13],
                    dest="l_b",
                    help="box sizes l_B; DROP the largest ones to go much faster")
    ap.add_argument("--sweep-max-nodes", type=int, default=60000,
                    help="skip sweep thresholds whose backbone exceeds this size")
    ap.add_argument("--thresholds", type=int, nargs="+",
                    default=[25, 40, 60, 100, 150, 250], help="thresholds for --sweep")
    ap.add_argument("--sweep", action="store_true",
                    help="report backbone size and exponents vs the weak-tie threshold")
    ap.add_argument("--out", default="results_dblp.json")
    a = ap.parse_args()

    # -- fail helpfully if the dataset is not there (the most common mistake)
    for flag, path in (("--xml", a.xml), ("--v12", a.v12), ("--edgelist", a.edgelist)):
        if path and not os.path.exists(path):
            print(f"ERROR: {flag} file not found: {path}\n")
            print("You are in the right place -- the dataset just is not downloaded yet.")
            print("Windows (PowerShell), official DBLP dump, ~1 GB compressed:")
            print("   curl.exe -L -o dblp.xml.gz https://dblp.org/xml/dblp.xml.gz")
            print("   # or:  Invoke-WebRequest https://dblp.org/xml/dblp.xml.gz -OutFile dblp.xml.gz")
            print("Then re-run this command. To try the pipeline first without any")
            print("download:   python run_dblp.py --surrogate")
            print("For a fast trial on the real file:  --max-records 200000")
            sys.exit(1)

    if a.surrogate:
        print("Building SURROGATE coauthorship backbone (not real data)...")
        G = surrogate_coauthor(); name = "surrogate-coauthor"
    else:
        if a.xml:
            print(f"Parsing DBLP XML {a.xml} ... (slow: minutes)")
            Gw = dblp_from_xml(a.xml, max_records=a.max_records,
                               min_joint=a.min_joint, two_pass=not a.single_pass)
        elif a.v12:
            print(f"Parsing DBLP V12 {a.v12} ...")
            Gw = dblp_from_v12_json(a.v12, max_records=a.max_records,
                                    min_joint=a.min_joint, two_pass=not a.single_pass)
        else:
            Gw = load_weighted_edgelist(a.edgelist)
        print("  (loaded from edge list -- no parsing needed)" if a.edgelist else "", end="")
        print(f"  full weighted coauthorship graph: N={Gw.number_of_nodes()} "
              f"M={Gw.number_of_edges()}")
        if a.save_weighted:
            save_weighted(Gw, a.save_weighted)
            print(f"  saved weighted graph -> {a.save_weighted}  "
                  f"(re-run analysis with --edgelist {a.save_weighted}, no re-parse)")
        if a.sweep:
            print("\n  backbone size and exponents vs weak-tie threshold:")
            res_sweep = threshold_sweep(Gw, thresholds=tuple(a.thresholds),
                                        max_nodes=a.sweep_max_nodes)
            json.dump(res_sweep, open("results_dblp_sweep.json", "w"),
                      indent=2, default=float)
            print("  wrote results_dblp_sweep.json")
        G = backbone(Gw, a.min_joint); name = f"DBLP-backbone(w>={a.min_joint})"
        print(f"  backbone after weak-tie removal: N={G.number_of_nodes()} "
              f"M={G.number_of_edges()}")
        print(f"  NOTE Fronczak et al. got ~2.5k/~3.2k at this threshold on the 2020 "
              f"V12 snapshot; DBLP has roughly doubled since, so a fixed threshold "
              f"no longer reproduces their size. Use --sweep to calibrate.")
    if G.number_of_nodes() < 50:
        print("Backbone too small -- lower --min-joint."); return

    res = {"intact": fronczak_protocol(G, l_B_values=tuple(a.l_b), name=name,
                                       n_jobs=a.jobs)}

    if a.percolate:
        print("\nPercolation sweep (susceptibility -> p_c, then exponents vs p)...")
        ps = np.round(np.linspace(0.30, 1.0, 15), 3)
        chi = susceptibility(G, ps, n_trials=20, rng=np.random.default_rng(0))
        pc = float(ps[int(np.argmax(chi))])
        print(f"  p_c (finite-cluster chi peak) = {pc:.3f}")
        rows = percolate_real(G, ps, l_B_values=(2, 3, 4, 6, 9, 13), n_trials=3)
        res["percolation"] = {"pc": pc, "ps": ps.tolist(),
                              "chi": chi.tolist(), "rows": rows}

    json.dump(res, open(a.out, "w"), indent=2, default=float)
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()

"""Delta-AIC over EVERY >=3-point subgrid of the l_B values covered for each network.

The point of this script is negative. A single Delta-AIC, computed on one box-size
grid, looks like a verdict on whether N_B(l_B) is a power law. It is not: on the
networks in this study -- fractal and control alike -- *both signs* are reachable by
choosing a different subset of the same covered box sizes. The Erdos-Renyi graph,
which the selected grid rejects at +7.7, reaches -10.9; the FSFN at t = 6, which is
fractal by construction, reaches +35.7.

So Delta-AIC has to be quoted together with its range, and the grid-selection rule has
to be stated, or the number means very little. That is what this script produces.

Usage
-----
    python subgrid_daic.py                 # every JSON in ../results
    python subgrid_daic.py a.json b.json   # named files

Each input is a run_snap.py / controls.py output. The l_B values considered are the
union of the analysed grid and every grid in the file's `grid_scan` block, so a file
written with `--scan` covers many more sizes than one written with `--l-b`.
"""
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from snap_networks import powerlaw_vs_exponential

RESULTS = os.environ.get("RESULTS_DIR", os.path.join(HERE, "..", "results"))


def nb_map(d):
    """Collect every (l_B, N_B) pair the file knows about, plus N."""
    out = {}
    m = d.get("main", d)
    if m.get("l_B"):
        for l, n in zip(m["l_B"], m["NB"]):
            out[int(l)] = float(n)
    for g in d.get("grid_scan", []) or []:
        for l, n in zip(g["grid"], g["NB"]):
            out[int(l)] = float(n)
    return out, m.get("N")


def ranges(d, label):
    nb, N = nb_map(d)
    ls = sorted(nb)
    if len(ls) < 3 or not N:
        return None
    vals = []
    for r in range(3, len(ls) + 1):
        for sub in itertools.combinations(ls, r):
            x = np.array(sub, float)
            y = np.array([nb[l] for l in sub], float) / N
            a = powerlaw_vs_exponential(x, y).get("dAIC_pl_minus_exp")
            if a is not None and np.isfinite(a):
                vals.append((a, sub))
    if not vals:
        return None
    vals.sort()
    print("%-26s l_B covered=%-34s n_subgrids=%4d  dAIC %+.1f .. %+.1f   "
          "best=%s worst=%s" % (label, str(tuple(ls)), len(vals),
                                vals[0][0], vals[-1][0], vals[0][1], vals[-1][1]))
    return dict(label=label, l_B=list(ls), n_subgrids=len(vals),
                dAIC_min=vals[0][0], dAIC_max=vals[-1][0],
                grid_min=list(vals[0][1]), grid_max=list(vals[-1][1]))


def main(argv):
    paths = argv[1:] or sorted(
        os.path.join(RESULTS, f) for f in os.listdir(RESULTS) if f.endswith(".json"))
    out = []
    for p in paths:
        try:
            d = json.load(open(p))
        except Exception:
            continue
        name = os.path.basename(p)
        if isinstance(d, dict) and ("main" in d or "grid_scan" in d):
            r = ranges(d, name)
            if r:
                out.append(r)
        elif isinstance(d, dict):
            # controls.py writes {label: {grid_scan, grid, main}}
            for k, v in d.items():
                if isinstance(v, dict) and "main" in v:
                    r = ranges({"main": v["main"],
                                "grid_scan": v.get("grid_scan")}, k)
                    if r:
                        out.append(r)
    dest = os.environ.get("SUBGRID_OUT")
    if dest:
        json.dump(out, open(dest, "w"), indent=1)
        print("wrote", dest)


if __name__ == "__main__":
    main(sys.argv)

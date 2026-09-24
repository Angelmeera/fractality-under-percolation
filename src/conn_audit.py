"""Disconnected-box audit: how many greedy boxes are internally disconnected, and what
the direct mass-law fit looks like once they are removed.

A greedy box is a set of nodes pairwise within l_B - 1 of one another IN THE WHOLE
GRAPH. Nothing in the covering constraint makes the induced subgraph connected, and on
a hierarchical network most boxes above l_B = 2 are not. Two consequences:

  * "the diameter of the box" is not defined for such a box. What the pipeline computes
    (`snap_networks._box_diameter`) is the largest distance between a REACHABLE pair,
    which is a different quantity;
  * a disconnected box draws its mass from parts of the network that its diameter does
    not span -- exactly the kind of point that breaks a two-variable mass law.

On the FSFN, where alpha = ln4/ln3 = 1.2619 and beta = 1 exactly, removing them moves
beta from about -0.53 to about -0.12 and lifts R^2 from ~0.69 to ~0.95, stably across
three system sizes. So beta ~ -0.5 is an artefact of the disconnected boxes rather than
a property of greedy covering. The residual failure is real and sits in alpha ~ 1.75.

The script also reports the fit under the NOMINAL box size l_B in place of the measured
diameter. With all boxes that swap changes the sign of beta (-0.53 -> +0.13); with
disconnected boxes removed the two conventions agree to about 0.02. One cause, not two.

    python src/conn_audit.py            # writes results/conn_audit.json
    CONN_OUT=/tmp/x.json python src/conn_audit.py

Runtime is a few minutes, dominated by t = 6 (N = 149,798).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import scipy.sparse as sp

import snap_networks as S
from fast_cover import csr_from_graph
from yakubo_fujiki import generate_fsfn

GENERATIONS = (4, 5, 6)
GRID = (2, 4, 10, 28)
MIN_MASS = 3


def fit(M, L, K):
    """The same joint OLS as snap_networks.joint_mass_law, with the design's condition
    number, so the four variants below are strictly comparable."""
    M, L, K = (np.asarray(x, float) for x in (M, L, K))
    keep = (M >= MIN_MASS) & (L >= 1) & (K >= 1)
    M, L, K = M[keep], L[keep], K[keep]
    if M.size < 20:
        return dict(n=int(M.size))
    X = np.column_stack([np.ones(M.size), np.log(L), np.log(K)])
    y = np.log(M)
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ c
    r2 = 1 - float(r @ r) / float(((y - y.mean()) ** 2).sum())
    s = np.linalg.svd(X, compute_uv=False)
    return dict(n=int(M.size), alpha=float(c[1]), beta=float(c[2]), R2=r2,
                cond=float(s[0] / s[-1]))


VARIANTS = ("all_meas", "conn_meas", "all_nom", "conn_nom")
LABELS = {"all_meas": "all boxes, measured L",
          "conn_meas": "connected only, measured L",
          "all_nom": "all boxes, nominal l_B",
          "conn_nom": "connected only, nominal l_B"}


def audit(t):
    G = generate_fsfn("GB", t)
    nodes, idx, A = csr_from_graph(G)
    deg = np.asarray(A.sum(1)).ravel()
    rows = {v: ([], [], []) for v in VARIANTS}
    per = {}
    for lb in GRID:
        boxes = S.cover_stream(A, lb)
        n_disc = 0
        n_fitted = n_disc_fitted = 0
        for b in boxes:
            b = np.asarray(sorted(b))
            ncomp, _ = sp.csgraph.connected_components(A[np.ix_(b, b)],
                                                       directed=False)
            connected = (ncomp == 1)
            n_disc += (not connected)
            m = b.size
            k = int(deg[b].max())
            if m >= MIN_MASS:
                n_fitted += 1
                n_disc_fitted += (not connected)
            # measured diameter over reachable pairs, as the pipeline does
            L_meas = S._box_diameter(A, b, pos=None) if m > 1 else 1
            for tag, L_val, include in (("all_meas", L_meas, True),
                                        ("conn_meas", L_meas, connected),
                                        ("all_nom", lb, True),
                                        ("conn_nom", lb, connected)):
                if include:
                    rows[tag][0].append(m)
                    rows[tag][1].append(max(L_val, 1))
                    rows[tag][2].append(k)
        per[lb] = dict(n_boxes=len(boxes), n_disconnected=int(n_disc),
                       frac=100.0 * n_disc / len(boxes),
                       n_fitted=n_fitted,
                       frac_fitted=(100.0 * n_disc_fitted / n_fitted)
                       if n_fitted else float("nan"))
    return dict(N=G.number_of_nodes(), grid=list(GRID), per_lB=per,
                **{v: fit(*rows[v]) for v in VARIANTS})


def main():
    out = {}
    for t in GENERATIONS:
        r = out[t] = audit(t)
        print("t=%d  N=%d" % (t, r["N"]), flush=True)
        for lb, v in r["per_lB"].items():
            print("   l_B=%2d  n=%6d  disconnected=%5.1f%%   fitted n=%6d  "
                  "disconnected=%5.1f%%"
                  % (lb, v["n_boxes"], v["frac"], v["n_fitted"], v["frac_fitted"]),
                  flush=True)
        for v in VARIANTS:
            f = r[v]
            print("   %-28s n=%6d  alpha=%7.4f  beta=%+7.4f  R2=%.4f  cond=%5.2f"
                  % (LABELS[v], f["n"], f.get("alpha", float("nan")),
                     f.get("beta", float("nan")), f.get("R2", float("nan")),
                     f.get("cond", float("nan"))), flush=True)
    dest = os.environ.get("CONN_OUT",
                          os.path.join(HERE, "..", "results", "conn_audit.json"))
    json.dump(out, open(dest, "w"), indent=1)
    print("wrote", dest)


if __name__ == "__main__":
    main()

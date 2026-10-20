"""Shared helpers for a8_iv.py, a11_closure_iv.py, ivFE.py and ivL.py.

tsls  : two-stage least squares, y on X with instruments Z (both include the constant).
boxes : construction-aware hub-centred boxes of a percolated FSFN under the 'pub'
        (largest connected fragment) and 'mass' (all surviving members) conventions,
        L = diameter + 1, boxes whose largest fragment has fewer than 3 nodes dropped;
        returns, per convention, the box mass M, L, damaged hub degree K, intact hub
        degree K0 and level.

Identical to the two functions defined in a8_iv.py (the file a11, ivFE and ivL imported
was not kept with the original run, so it is restored here from a8_iv.py; the
reproduction check in revision/README.md confirms it against a11_GB_t5.json).
"""
import sys
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, lib
from scipy.sparse.csgraph import connected_components


def tsls(y, X, Z):
    """2SLS: y on X with instruments Z (both include the constant)."""
    P = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    return np.linalg.lstsq(P, y, rcond=None)[0]

def boxes(E, N, seeds, t, deg0):
    gc = np.unique(E.ravel()); deg = np.bincount(E.ravel(), minlength=N); A = lib.adj(E, N)
    out = {c: dict(M=[], L=[], K=[], K0=[], lev=[]) for c in ("pub", "mass")}
    for tau in range(1, t):
        seed = seeds[tau]; same = seed[E[:, 0]] == seed[E[:, 1]]
        _, lab = connected_components(lib.adj(E[same], N), directed=False)
        fids, inv, fsize = np.unique(lab[gc], return_inverse=True, return_counts=True)
        fseed = np.empty(len(fids), int); fseed[inv] = seed[gc]
        maxdeg = np.zeros(len(fids)); np.maximum.at(maxdeg, inv, deg[gc])
        kfrag = maxdeg.copy(); ishub = seed[gc] == gc; kfrag[inv[ishub]] = deg[gc[ishub]]
        order = np.argsort(inv, kind="stable"); members = np.split(gc[order], np.flatnonzero(np.diff(inv[order])) + 1)
        gs, ginv = np.unique(fseed, return_inverse=True); gmass = np.bincount(ginv, weights=fsize)
        big = np.full(len(gs), -1)
        for fi in np.argsort(fsize, kind="stable"):
            big[ginv[fi]] = fi
        for gi, fi in enumerate(big):
            if fsize[fi] < 3:
                continue
            L = lib.diam_of(A, members[fi]) + 1
            for c, m in (("pub", fsize[fi]), ("mass", gmass[gi])):
                out[c]["M"].append(m); out[c]["L"].append(L); out[c]["K"].append(kfrag[fi])
                out[c]["K0"].append(deg0[gs[gi]]); out[c]["lev"].append(tau)
    return out

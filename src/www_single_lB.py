"""
The direct joint mass-law fit restricted to the boxes of a SINGLE box size.

Table 4 of the paper has a row "WWW, l_B = 6 alone": the joint OLS of
ln m on (ln L, ln k_hub) over the greedy boxes of the l_B = 6 cover only, rather
than over the pooled l_B = 2, 4, 6 boxes of `www_comm.json`. With one nominal box
size the only spread in L is the scatter of the MEASURED box diameter within that
level, so this row shows what the fit does when the box-size lever arm is removed
(Sec. 4.4, admissibility check: it returns alpha < 0).

    python src/www_single_lB.py --www data/web-NotreDame.txt.gz --l-b 2 4 6 \
        --jobs 2 --check results/www_comm.json --out results/www_l6.json

What it computes, per l_B in --l-b
    single_lB[l].direct            joint fit over that level's boxes, mass >= 3
    single_lB[l].direct_connected  the same, internally connected boxes only
    single_lB[l].N_B, n_disconnected, ...
and, when more than one l_B is covered,
    pooled.direct / pooled.direct_connected   the fit over all levels together,
                                              identical to snap_networks.analyse()

The pooled fit is there as a reproduction check: covering l_B = 2, 4, 6 and
passing --check results/www_comm.json asserts that N_B at every shared level and
the pooled (alpha, beta, R2, cond, n) agree with the committed file, so the
single-level numbers are known to come from the same covers.

Nothing here is new machinery: the cover is snap_networks.cover_all (canonical
seeding order of fast_cover.csr_from_graph), the fit is
snap_networks.joint_mass_law with its default mass cut m >= 3, and L is the
measured box diameter, exactly as in every other row of Table 4.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from snap_networks import (load_snap_edgelist, giant, cover_all,
                           joint_mass_law)


def _records_to_arrays(recs):
    M = np.array([r[0] for r in recs], float)
    L = np.array([r[1] for r in recs], float)
    K = np.array([r[2] for r in recs], float)
    C = np.array([bool(r[4]) if len(r) > 4 else True for r in recs], bool)
    return M, L, K, C


def _fit(M, L, K, min_mass, mask=None):
    """joint_mass_law, but a level whose design is singular is recorded, not fatal.

    At l_B = 2 every box is a single node or an edge, so L = 1 for all of them and
    ln L is collinear with the intercept; there is no single-level fit to report.
    """
    try:
        return joint_mass_law(M, L, K, min_mass=min_mass, mask=mask)
    except np.linalg.LinAlgError as exc:
        keep = (M >= min_mass) & (L >= 1) & (K >= 1)
        if mask is not None:
            keep &= mask
        return dict(n=int(keep.sum()), alpha=float("nan"), beta=float("nan"),
                    error=f"singular design ({exc}); distinct L = "
                          f"{int(np.unique(L[keep]).size)}")


def _save_cache(path, per_l):
    arrs = {}
    for l, (NB, recs) in per_l.items():
        arrs[f"NB_{l}"] = np.array(NB)
        arrs[f"rec_{l}"] = np.array([list(r[:5]) for r in recs], float)
    np.savez_compressed(path, **arrs)


def _load_cache(path, l_B):
    z = np.load(path)
    per_l = {}
    for l in l_B:
        if f"NB_{l}" in z:
            rec = z[f"rec_{l}"]
            per_l[l] = (int(z[f"NB_{l}"]),
                        [(int(r[0]), int(r[1]), float(r[2]), float(r[3]), int(r[4]))
                         for r in rec])
    return per_l


def _level_summary(l, NB, recs, min_mass):
    M, L, K, C = _records_to_arrays(recs)
    fitted = M >= min_mass
    return dict(
        l_B=int(l), N_B=int(NB),
        n_disconnected=int((~C).sum()),
        frac_disconnected=float((~C).mean()) if C.size else float("nan"),
        n_fitted=int(fitted.sum()),
        frac_fitted_disconnected=(float((~C[fitted]).mean())
                                  if fitted.any() else float("nan")),
        L_measured_range=[float(L.min()), float(L.max())] if L.size else [],
        L_measured_distinct=int(np.unique(L[fitted]).size) if fitted.any() else 0,
        direct=_fit(M, L, K, min_mass),
        direct_connected=_fit(M, L, K, min_mass, mask=C))


def _check(out, path, tol=1e-9):
    """Compare against a committed run_snap.py output on the same network."""
    ref = json.load(open(path))["main"]
    problems = []
    ref_NB = dict(zip([int(x) for x in ref["l_B"]], [int(x) for x in ref["NB"]]))
    for l, s in out["single_lB"].items():
        if int(l) in ref_NB and ref_NB[int(l)] != s["N_B"]:
            problems.append(f"N_B(l_B={l}) = {s['N_B']} vs {ref_NB[int(l)]} in {path}")
    pooled = out.get("pooled")
    same_levels = sorted(int(l) for l in out["single_lB"]) == sorted(ref_NB)
    if pooled and same_levels:
        for key in ("direct", "direct_connected"):
            if key not in ref:
                continue
            for f in ("n", "alpha", "beta", "R2", "cond"):
                a, b = pooled[key].get(f), ref[key].get(f)
                if a is None or b is None:
                    continue
                if abs(float(a) - float(b)) > tol * max(1.0, abs(float(b))):
                    problems.append(f"pooled {key}.{f} = {a} vs {b} in {path}")
    return dict(reference=os.path.basename(path),
                levels_compared=sorted(set(int(l) for l in out["single_lB"]) & set(ref_NB)),
                pooled_compared=bool(pooled and same_levels),
                passed=not problems, problems=problems)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--www", required=True, metavar="PATH",
                    help="SNAP web-NotreDame edge list (.txt or .txt.gz)")
    ap.add_argument("--l-b", type=int, nargs="+", default=[6],
                    help="box sizes to cover; each gets its own single-level fit")
    ap.add_argument("--min-mass", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--check", metavar="JSON", default=None,
                    help="committed run_snap.py output to reproduce (e.g. results/www_comm.json)")
    ap.add_argument("--cache", metavar="NPZ", default=None,
                    help="save the per-box records here after covering, and reuse "
                         "them on a rerun (the l_B = 6 cover takes ~40 min)")
    ap.add_argument("--out", default="results/www_l6.json")
    args = ap.parse_args()

    t0 = time.time()
    G = giant(load_snap_edgelist(args.www))
    print(f"WWW giant component: N={G.number_of_nodes()} M={G.number_of_edges()}",
          flush=True)
    per_l = {}
    if args.cache and os.path.exists(args.cache):
        per_l = _load_cache(args.cache, args.l_b)
        print(f"  box records for l_B={sorted(per_l)} read from {args.cache}", flush=True)
    todo = [l for l in args.l_b if l not in per_l]
    if todo:
        new, _N, _deg = cover_all(G, todo, n_jobs=min(args.jobs, len(todo)),
                                  chunk=args.chunk)
        per_l.update(new)
        if args.cache:
            _save_cache(args.cache, per_l)
            print(f"  box records saved to {args.cache}", flush=True)
    N = G.number_of_nodes()

    import hashlib
    h = hashlib.sha256()
    with open(args.www, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    out = dict(name="WWW", N=int(N), M=int(G.number_of_edges()),
               argv=sys.argv[1:],
               input=dict(path=args.www, sha256=h.hexdigest(),
                          bytes=os.path.getsize(args.www)),
               note=("Direct joint fit of ln m on (ln L, ln k_hub) with L the measured "
                     "box diameter and mass cut m >= %d, restricted to the boxes of one "
                     "l_B at a time (single_lB) and, if several l_B are covered, over all "
                     "of them together (pooled, = snap_networks.analyse 'direct')."
                     % args.min_mass),
               single_lB={})
    for l in sorted(per_l):
        NB, recs = per_l[l]
        out["single_lB"][str(int(l))] = _level_summary(l, NB, recs, args.min_mass)

    if len(per_l) > 1:
        ls = sorted(per_l)
        arrs = [_records_to_arrays(per_l[l][1]) for l in ls]
        M, L, K, C = (np.concatenate([a[i] for a in arrs]) for i in range(4))
        out["pooled"] = dict(
            l_B=[int(l) for l in ls],
            direct=joint_mass_law(M, L, K, min_mass=args.min_mass),
            direct_connected=joint_mass_law(M, L, K, min_mass=args.min_mass, mask=C))

    if args.check:
        out["check"] = _check(out, args.check)

    out["runtime_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)

    for l, s in out["single_lB"].items():
        d = s["direct"]
        print(f"  l_B={l}: N_B={s['N_B']}  alpha={d.get('alpha', float('nan')):.3f} "
              f"beta={d.get('beta', float('nan')):.3f} R2={d.get('R2', float('nan')):.3f} "
              f"cond={d.get('cond', float('nan')):.1f} boxes={d.get('n')}")
    if "pooled" in out:
        d = out["pooled"]["direct"]
        print(f"  pooled {out['pooled']['l_B']}: alpha={d['alpha']:.4f} "
              f"beta={d['beta']:.4f} R2={d['R2']:.4f} cond={d['cond']:.2f} boxes={d['n']}")
    if "check" in out:
        c = out["check"]
        print(f"  check against {c['reference']}: "
              f"{'PASSED' if c['passed'] else 'FAILED'} {c['problems']}")
    print(f"wrote {args.out} ({out['runtime_s']} s)")


if __name__ == "__main__":
    main()

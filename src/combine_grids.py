"""Merge several per-l_B runs of the same network and evaluate every candidate
l_B grid offline.

Covering is the expensive step and it is done once per l_B, so the grid scan does
not need to repeat it: given N_B(l_B), the per-l_B box-mass exponents and the
renormalisation ratio s(l_B) = <k_B>/<k_hub>, every candidate grid can be scored
afterwards. On the WWW that matters -- the l_B = 7 cover alone takes the better
part of an hour -- so the large runs are done piecewise
(`run_snap.py --l-b 2 3 4 5`, then `--l-b 6 7`) and combined here.

Usage:
    python combine_grids.py results_www_quick.json results_www_l67.json \\
        --name WWW --out ../results/results_www_scan.json \\
        --lambdas 1.3 1.4 1.5 1.7 2.0
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from snap_networks import powerlaw_vs_exponential, geometric_grid


def load_parts(paths):
    """Collect {l_B: (N_B, delta_l, s_l)} plus the shared scalars."""
    per = {}
    meta = {}
    for p in paths:
        with open(p) as fh:
            d = json.load(fh)
        m = d.get("main", d)
        for key in ("N", "M_edges", "mean_degree", "gamma", "name"):
            if key in m and key not in meta:
                meta[key] = m[key]
        if "diameter_lb" in d:
            meta.setdefault("diameter_lb", d["diameter_lb"])
        lbs = [int(round(x)) for x in m["l_B"]]
        NB = m["NB"]
        # delta_per_lB only contains the l_B where the tail had enough boxes, so
        # it can be shorter than l_B; s_of_l is aligned with the full grid.
        s = m.get("s_of_l", [None] * len(lbs))
        dl = m.get("delta_per_lB", [])
        for i, l in enumerate(lbs):
            per[l] = {"NB": float(NB[i]),
                      "s": (float(s[i]) if i < len(s) and s[i] is not None
                            and np.isfinite(s[i]) else None)}
        # attach deltas positionally to the l_B that produced them
        for i, val in enumerate(dl):
            if i < len(lbs):
                per[lbs[i]]["delta"] = float(val)
    return per, meta


def score(per, meta, grid):
    ls = [l for l in grid if l in per]
    if len(ls) < 3:
        return None
    lb = np.array(ls, float)
    NB = np.array([per[l]["NB"] for l in ls], float)
    N = meta["N"]
    keep = NB > 1
    fB = powerlaw_vs_exponential(lb[keep], NB[keep] / N)
    s = np.array([per[l]["s"] if per[l]["s"] else np.nan for l in ls], float)
    ok = keep & np.isfinite(s) & (s > 0)
    fk = powerlaw_vs_exponential(lb[ok], s[ok]) if ok.sum() >= 3 else {}
    deltas = [per[l]["delta"] for l in ls if "delta" in per[l]]
    delta = float(np.median(deltas)) if deltas else float("nan")
    gamma = meta.get("gamma", float("nan"))
    dB = fB.get("exponent", float("nan"))
    dk_gamma = dB / (gamma - 1) if gamma and gamma > 1 else float("nan")
    alpha = (((delta - 2) / (delta - 1)) * dB
             if np.isfinite(delta) and delta > 1 else float("nan"))
    beta = ((gamma - 1) / (delta - 1)
            if np.isfinite(delta) and delta > 1 else float("nan"))
    return dict(grid=list(ls), n_points=int(keep.sum()),
                NB=NB[keep].tolist(), s=[float(x) for x in s],
                dB=dB, dB_R2=fB.get("R2_powerlaw", float("nan")),
                dB_verdict=fB.get("verdict"),
                dAIC=fB.get("dAIC_pl_minus_exp", float("nan")),
                R2_exponential=fB.get("R2_exponential", float("nan")),
                dk_renorm=fk.get("exponent", float("nan")),
                dk_R2=fk.get("R2_powerlaw", float("nan")),
                dk_verdict=fk.get("verdict"),
                delta=delta, delta_per_lB=deltas, gamma=gamma,
                dk_from_gamma=dk_gamma,
                alpha_macroscopic=alpha, beta_macroscopic=beta,
                fractality=fB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="+")
    ap.add_argument("--name", default=None)
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=(1.3, 1.4, 1.5, 1.7, 2.0, 2.5))
    ap.add_argument("--l-max", type=int, default=None)
    ap.add_argument("--min-points", type=int, default=4)
    ap.add_argument("--out", default="results_combined.json")
    args = ap.parse_args()

    per, meta = load_parts(args.parts)
    if args.name:
        meta["name"] = args.name
    diam = meta.get("diameter_lb", max(per) * 4)
    have = sorted(per)
    print(f"{meta.get('name')}  N={meta['N']}  covered l_B = {have}")
    print(f"  N_B = {[int(per[l]['NB']) for l in have]}")
    print(f"  s   = {[None if per[l]['s'] is None else round(per[l]['s'], 4) for l in have]}")

    rows = []
    seen = set()
    cands = {}
    for lam in args.lambdas:
        g = tuple(l for l in geometric_grid(lam, diam, l_max=args.l_max)
                  if l in per)
        if len(g) < args.min_points or g in seen:
            continue
        seen.add(g)
        cands[float(lam)] = g
    # Constant-STEP grids. This is the general form of the commensurability rule:
    # a greedy box with diameter < l_B is a ball of radius (l_B-1)/2, so
    # consecutive integer l_B alternate between integer and half-integer radii and
    # interleave two series with different prefactors. Sampling l_B with a constant
    # step keeps the radius step constant and removes the alternation. On the WWW
    # this is decisive: consecutive l_B give R^2 = 0.957, step 2 gives 0.999998.
    for step in (2, 3):
        for start in range(step):
            g = tuple(l for l in have if (l - have[0]) % step == start % step)
            if len(g) >= 3:
                cands[f"step {step} from l_B={g[0]}"] = g
    cands["all (consecutive)"] = tuple(have)
    if len(have) > 3:
        cands["all except l_B=2"] = tuple(l for l in have if l != 2)

    print(f"\n{'label':>18} {'grid':>24} {'d_B':>7} {'R2':>7} {'dAIC':>7} "
          f"{'delta':>6} {'alpha':>6} {'beta':>6} {'dk_ren':>7}")
    for lam, g in cands.items():
        r = score(per, meta, g)
        if not r:
            continue
        r["lam"] = lam if isinstance(lam, float) else None
        r["label"] = str(lam)
        rows.append(r)
        print(f"{str(lam):>18} {str(g):>24} {r['dB']:>7.3f} {r['dB_R2']:>7.4f} "
              f"{r['dAIC']:>7.1f} {r['delta']:>6.2f} "
              f"{r['alpha_macroscopic']:>6.3f} {r['beta_macroscopic']:>6.3f} "
              f"{r['dk_renorm']:>7.3f}")

    best = max((r for r in rows if np.isfinite(r["dB_R2"])),
               key=lambda r: r["dB_R2"], default=None)
    if best:
        print(f"\n  -> best R2: {best['label']}, grid={tuple(best['grid'])}, "
              f"d_B={best['dB']:.3f}")
    out = {"name": meta.get("name"), "N": meta["N"],
           "M": meta.get("M_edges"), "mean_degree": meta.get("mean_degree"),
           "diameter_lb": diam, "per_lB": {str(k): v for k, v in per.items()},
           "grid_scan": rows, "parts": args.parts}
    if best:
        out["grid_chosen"] = best["grid"]
        out["main"] = {"name": meta.get("name"), "N": meta["N"],
                       "M_edges": meta.get("M_edges"),
                       "mean_degree": meta.get("mean_degree"),
                       "l_B": [float(x) for x in best["grid"]],
                       "NB": best["NB"], "dB": best["dB"],
                       "dB_R2": best["dB_R2"], "fractality": best["fractality"],
                       "gamma": best["gamma"], "delta": best["delta"],
                       "delta_per_lB": best["delta_per_lB"],
                       "dk_renorm": best["dk_renorm"],
                       "dk_from_gamma": best["dk_from_gamma"],
                       "alpha_macroscopic": best["alpha_macroscopic"],
                       "beta_macroscopic": best["beta_macroscopic"],
                       "s_of_l": best["s"]}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

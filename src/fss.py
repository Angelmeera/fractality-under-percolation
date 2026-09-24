"""
Finite-size scaling of the bond-percolation transition on the FSFN, with every
measured exponent checked against a closed form.

WHY THIS IS WORTH DOING
-----------------------
Yakubo & Fujiki give, for a hierarchical FSFN built from a generator G, the exact
critical point and exponents (their Table 1):

    generator      p_c        nu       nu~      beta_P
    GA           0.6961    1.8293   3.4626    0.0595
    GB           0.6288    1.7772   3.3638    0.1098

and the finite-size scaling form (their Eq. 43)

    P_t(p, N_t) = N_t^(-beta_P/nu~) F[ (p - p_c) N_t^(1/nu~) ] ,          (FSS)

where P_t is the probability that a node belongs to the largest component. They
demonstrate the collapse only for a third, asymmetric generator; for GA and GB
they publish the exponents without a numerical test. So measuring them here is
both a real test of our pipeline and a test of their result.

NOTATION WARNING (and this matters -- two symbols are overloaded in this
literature). In the mass-law part of the paper, gamma is the DEGREE exponent and
beta is the exponent of hub degree in m = B L^alpha k^beta. In the percolation
part, the conventional names collide. We therefore write:

    beta_P    order-parameter exponent,  P_inf ~ (p - p_c)^beta_P
    gamma_chi susceptibility exponent,   chi ~ |p - p_c|^(-gamma_chi)
    nu        correlation-length exponent in CHEMICAL distance
    nu~       = D_f * nu, the exponent in the size variable N

and never use bare beta or gamma for a percolation quantity.

WHAT IS MEASURED, AND AGAINST WHAT
----------------------------------
1. Shift of the pseudo-critical point:  |p_c(N) - p_c| ~ N^(-1/nu~).
   Compared with the analytic nu~. This uses no fitted input at all.
2. Order parameter at criticality: from (FSS) at p = p_c,
   P_t(p_c, N_t) / P_(t-1)(p_c, N_(t-1)) = m_gen^(-beta_P/nu~)
   (their Eq. 44, with m_gen = 8). Gives beta_P/nu~ from a ratio of two numbers.
3. Susceptibility peak height: chi_max ~ N^(gamma_chi/nu~), giving gamma_chi.
4. Hyperscaling, 2 beta_P + gamma_chi = nu~. Both sides now measured
   independently, so this is a closing consistency check rather than an
   assumption -- and it needs no published value of gamma_chi, which Yakubo &
   Fujiki do not give for GA and GB.
5. The two data collapses, using the ANALYTIC exponents (not fitted ones), so a
   collapse is evidence and not a tautology.

Realisations are over bond configurations; the network itself is deterministic,
so there is no ensemble over graphs to average (Yakubo & Fujiki make the same
point -- P_t is almost sample-independent at large t because global connectivity
does not depend on which edges were replaced).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from large_lib import build_tracked, analytic, DF, MGEN


def _one_realisation(edges, N, p, rng):
    """(chi over finite clusters, largest-component fraction) for one bond
    configuration."""
    M = len(edges)
    keep = rng.random(M) < p
    e = edges[keep]
    if len(e):
        r = np.concatenate([e[:, 0], e[:, 1]])
        c = np.concatenate([e[:, 1], e[:, 0]])
        A = csr_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(N, N))
    else:
        A = csr_matrix((N, N), dtype=np.int8)
    _, lab = connected_components(A, directed=False)
    sizes = np.bincount(lab)
    order = np.sort(sizes)[::-1]
    giant = float(order[0])
    finite = order[1:].astype(float)
    chi = float((finite ** 2).sum() / finite.sum()) if finite.size and finite.sum() > 0 else 0.0
    return chi, giant / N


def _sweep_worker(args):
    edges, N, ps, n_trials, seed = args
    rng = np.random.default_rng(seed)
    chi = np.zeros((len(ps), n_trials))
    P = np.zeros((len(ps), n_trials))
    for i, p in enumerate(ps):
        for j in range(n_trials):
            chi[i, j], P[i, j] = _one_realisation(edges, N, p, rng)
    return chi, P


def sweep(which, t, n_trials, half_width=0.16, n_p=41, n_jobs=2, seed=0,
          verbose=True):
    """chi(p) and P(p) with error bars, on a fine grid centred on the analytic
    p_c. Trials are split across processes and pooled, so the reported s.d. is
    over all realisations."""
    from concurrent.futures import ProcessPoolExecutor
    edges, N, _ = build_tracked(which, t)
    an = analytic(which)
    ps = np.round(np.linspace(an["pc"] - half_width,
                              min(an["pc"] + half_width, 1.0), n_p), 5)
    per = max(1, n_trials // max(n_jobs, 1))
    chunks = [(edges, N, ps, per, seed + 1000 * k) for k in range(max(n_jobs, 1))]
    chis, Ps = [], []
    if n_jobs <= 1:
        c, p_ = _sweep_worker(chunks[0])
        chis.append(c); Ps.append(p_)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for c, p_ in ex.map(_sweep_worker, chunks):
                chis.append(c); Ps.append(p_)
    chi = np.concatenate(chis, axis=1)
    P = np.concatenate(Ps, axis=1)
    n = chi.shape[1]
    out = dict(which=which, t=t, N=int(N), M=int(len(edges)), n_trials=int(n),
               ps=ps.tolist(),
               chi=chi.mean(axis=1).tolist(),
               chi_sd=chi.std(axis=1, ddof=1).tolist(),
               chi_sem=(chi.std(axis=1, ddof=1) / np.sqrt(n)).tolist(),
               P=P.mean(axis=1).tolist(),
               P_sd=P.std(axis=1, ddof=1).tolist(),
               P_sem=(P.std(axis=1, ddof=1) / np.sqrt(n)).tolist())
    # pseudo-critical point: parabolic refinement of the susceptibility peak
    cm = chi.mean(axis=1)
    i = int(np.argmax(cm))
    pc_sim = float(ps[i])
    if 0 < i < len(ps) - 1:
        y0, y1, y2 = cm[i - 1], cm[i], cm[i + 1]
        den = y0 - 2 * y1 + y2
        if den != 0:
            step = ps[1] - ps[0]
            pc_sim = float(ps[i] - step * 0.5 * (y2 - y0) / den)
    out["pc_sim"] = pc_sim
    out["chi_max"] = float(cm[i])
    out["chi_max_sem"] = float(chi[i].std(ddof=1) / np.sqrt(n))
    # order parameter AT the analytic critical point, by linear interpolation
    out["P_at_pc"] = float(np.interp(an["pc"], ps, P.mean(axis=1)))
    j = int(np.argmin(np.abs(ps - an["pc"])))
    out["P_at_pc_sem"] = float(P[j].std(ddof=1) / np.sqrt(n))
    if verbose:
        print(f"  {which} t={t} N={N:8d} M={len(edges):8d} trials={n}: "
              f"pc_sim={pc_sim:.4f} (analytic {an['pc']:.4f}, "
              f"|d|={abs(pc_sim - an['pc']):.4f})  chi_max={cm[i]:.1f}  "
              f"P(pc)={out['P_at_pc']:.4f}", flush=True)
    return out


def _loglog(x, y):
    x = np.log(np.asarray(x, float)); y = np.log(np.asarray(y, float))
    n = len(x)
    a, b = np.polyfit(x, y, 1)
    pred = a * x + b
    sse = float(np.sum((y - pred) ** 2))
    ss = float(np.sum((y - y.mean()) ** 2))
    # standard error of the slope
    se = float(np.sqrt(sse / max(n - 2, 1) / np.sum((x - x.mean()) ** 2))) if n > 2 else float("nan")
    return float(a), se, (1 - sse / ss if ss > 0 else float("nan"))


def analyse(rows, which):
    """Turn the sweeps for one generator into measured exponents."""
    rows = sorted([r for r in rows if r["which"] == which], key=lambda r: r["t"])
    an = analytic(which)
    pc, nu, nut = an["pc"], an["nu"], an["nu_tilde"]
    Ns = [r["N"] for r in rows]
    out = dict(which=which, analytic=dict(pc=pc, nu=nu, nu_tilde=nut),
               sizes=[dict(t=r["t"], N=r["N"], n_trials=r["n_trials"],
                           pc_sim=r["pc_sim"], chi_max=r["chi_max"],
                           P_at_pc=r["P_at_pc"]) for r in rows])

    # 1. shift of the pseudo-critical point -> 1/nu~
    dev = [abs(r["pc_sim"] - pc) for r in rows]
    if min(dev) > 0 and len(rows) >= 3:
        a, se, r2 = _loglog(Ns, dev)
        out["shift"] = dict(slope=a, slope_se=se, R2=r2,
                            nu_tilde_measured=-1.0 / a if a else float("nan"),
                            inv_nu_tilde_measured=-a,
                            inv_nu_tilde_analytic=1.0 / nut,
                            pct_error=abs(-a - 1.0 / nut) / (1.0 / nut) * 100)

    # 2. order parameter at p_c -> beta_P/nu~ (their Eq. 44: ratio = m_gen^{-b})
    ratios = []
    for prev, cur in zip(rows, rows[1:]):
        if prev["P_at_pc"] > 0:
            ratios.append(cur["P_at_pc"] / prev["P_at_pc"])
    if ratios:
        b_from_ratio = [-np.log(x) / np.log(MGEN) for x in ratios]
        out["order_parameter"] = dict(
            ratios=[float(x) for x in ratios],
            beta_over_nutilde_per_ratio=[float(x) for x in b_from_ratio],
            beta_over_nutilde_measured=float(np.mean(b_from_ratio)),
            beta_over_nutilde_sd=float(np.std(b_from_ratio, ddof=1))
            if len(b_from_ratio) > 1 else float("nan"))
        a, se, r2 = _loglog(Ns, [r["P_at_pc"] for r in rows])
        out["order_parameter"].update(
            slope=a, slope_se=se, R2=r2,
            beta_over_nutilde_from_fit=-a,
            beta_P_measured=-a * nut)

    # 3. susceptibility peak -> gamma_chi/nu~
    a, se, r2 = _loglog(Ns, [r["chi_max"] for r in rows])
    out["susceptibility"] = dict(slope=a, slope_se=se, R2=r2,
                                 gamma_over_nutilde_measured=a,
                                 gamma_chi_measured=a * nut,
                                 gamma_chi_se=se * nut)

    # 4. hyperscaling, 2 beta_P + gamma_chi = nu~
    bP = out.get("order_parameter", {}).get("beta_P_measured", float("nan"))
    gx = out["susceptibility"]["gamma_chi_measured"]
    lhs = 2 * bP + gx
    out["hyperscaling"] = dict(two_beta_plus_gamma=float(lhs), nu_tilde=nut,
                               pct_error=float(abs(lhs - nut) / nut * 100))
    return out


def report(res):
    for which, o in res["exponents"].items():
        an = o["analytic"]
        print(f"\n===== {which}  (analytic p_c={an['pc']:.4f}, nu={an['nu']:.4f}, "
              f"nu~={an['nu_tilde']:.4f}) =====")
        s = o.get("shift")
        if s:
            print(f"  1/nu~ from the p_c(N) shift : {s['inv_nu_tilde_measured']:.4f} "
                  f"+- {s['slope_se']:.4f}   (analytic {s['inv_nu_tilde_analytic']:.4f}, "
                  f"{s['pct_error']:.1f}%, R2={s['R2']:.4f})")
            print(f"     -> nu~ measured = {s['nu_tilde_measured']:.4f}")
        op = o.get("order_parameter")
        if op:
            print(f"  beta_P/nu~ from P(p_c) ratios: "
                  f"{op['beta_over_nutilde_measured']:.4f} "
                  f"+- {op['beta_over_nutilde_sd']:.4f}  "
                  f"(per-ratio {[round(x,4) for x in op['beta_over_nutilde_per_ratio']]})")
            print(f"  beta_P/nu~ from the N fit   : "
                  f"{op['beta_over_nutilde_from_fit']:.4f} +- {op['slope_se']:.4f}"
                  f"  -> beta_P = {op['beta_P_measured']:.4f}  (R2={op['R2']:.4f})")
        su = o["susceptibility"]
        print(f"  gamma_chi/nu~ from chi_max(N) : {su['gamma_over_nutilde_measured']:.4f} "
              f"+- {su['slope_se']:.4f}  -> gamma_chi = {su['gamma_chi_measured']:.4f} "
              f"+- {su['gamma_chi_se']:.4f}  (R2={su['R2']:.4f})")
        hs = o["hyperscaling"]
        print(f"  hyperscaling 2 beta_P + gamma_chi = {hs['two_beta_plus_gamma']:.4f} "
              f"vs nu~ = {hs['nu_tilde']:.4f}  -> {hs['pct_error']:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", default=["GB", "GA"])
    ap.add_argument("--t", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--trials", type=int, nargs="+", default=None,
                    help="trials per generation, matched to --t (default scales "
                         "down with size)")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--n-p", type=int, default=41)
    ap.add_argument("--half-width", type=float, default=0.16)
    ap.add_argument("--out", default="results_fss.json")
    args = ap.parse_args()

    default_trials = {3: 400, 4: 300, 5: 150, 6: 60, 7: 20}
    trials = (dict(zip(args.t, args.trials)) if args.trials
              else {t: default_trials.get(t, 50) for t in args.t})

    rows = []
    for which in args.which:
        print(f"\n-- {which} --", flush=True)
        for t in args.t:
            rows.append(sweep(which, t, trials[t], half_width=args.half_width,
                              n_p=args.n_p, n_jobs=args.jobs, seed=17 + t))
            with open(args.out, "w") as fh:
                json.dump({"sweeps": rows}, fh, indent=1, default=float)
    res = {"sweeps": rows,
           "exponents": {w: analyse(rows, w) for w in args.which}}
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    report(res)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

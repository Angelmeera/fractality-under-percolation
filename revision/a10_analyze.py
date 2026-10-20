import sys, json, numpy as np
for path in sys.argv[1:]:
    d = json.load(open(path)); R = d["recs"]; t = d["t"]
    ps = sorted(set(r["p"] for r in R), reverse=True)
    print(f"\n===== {d['which']} t={t}  ({len(R)} configs)")
    def lev(G, tau, k):
        v = np.array([g["levels"][str(tau)].get(k, np.nan) for g in G], float); v = v[np.isfinite(v)]
        return (v.mean(), v.std()) if len(v) else (np.nan, np.nan)
    intact = [r for r in R if r["p"] == 1.0]
    for p in ps:
        G = [r for r in R if r["p"] == p]
        print(f"\n p={p:.2f} (n={len(G)})  tau: n | pub OLS -> IV | mass OLS -> IV | thin ln(k/k0) | ln(m/m0) pub | stretch ln(L/L0) | unit dim d(tau->tau+1) pub [intact]")
        for tau in range(1, t):
            n = G[0]["levels"][str(tau)]["n"]
            s = (f"  {tau}: {n:6d} | {lev(G,tau,'pub_ols')[0]:.3f} -> {lev(G,tau,'pub_iv')[0]:.3f}±{lev(G,tau,'pub_iv')[1]:.3f} | "
                 f"{lev(G,tau,'mass_ols')[0]:.3f} -> {lev(G,tau,'mass_iv')[0]:.3f} | {lev(G,tau,'thin')[0]:+.3f} | {lev(G,tau,'massfrac_pub')[0]:+.3f} | {lev(G,tau,'stretch')[0]:+.3f}")
            if tau < t - 1:
                dm = lev(G, tau + 1, "mean_lnm")[0] - lev(G, tau, "mean_lnm")[0]; dL = lev(G, tau + 1, "mean_lnL")[0] - lev(G, tau, "mean_lnL")[0]
                dmi = lev(intact, tau + 1, "mean_lnm")[0] - lev(intact, tau, "mean_lnm")[0]; dLi = lev(intact, tau + 1, "mean_lnL")[0] - lev(intact, tau, "mean_lnL")[0]
                s += f" | {dm/dL:.3f} [{dmi/dLi:.3f}]"
            print(s)
        print(f"   ln p = {np.log(p):+.3f}")

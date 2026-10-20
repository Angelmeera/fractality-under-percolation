import sys, json, numpy as np
for path in sys.argv[1:]:
    d=json.load(open(path)); R=d["recs"]; t=d["t"]
    print(f"\n== {d['which']} t={t}: per-level beta  OLS / IV(k0) / MM(p only)   and joint fits (all, ge2)")
    for c in ("pub","mass"):
        print(f" [{c}]")
        for p in sorted(set(r["p"] for r in R),reverse=True):
            G=[r for r in R if r["p"]==p]
            parts=[]
            for tau in range(1,t):
                v=[g[c]["per"].get(str(tau)) for g in G]; v=[x for x in v if x]
                if not v: continue
                f=lambda k: np.nanmean([x[k] for x in v])
                parts.append(f"t{tau}:{f('ols'):.3f}/{f('iv'):.3f}/{f('mm'):.3f}")
            J=lambda m,k: np.mean([g[c]["joint"][m][k] for g in G])
            print(f"  p={p:.2f} "+" ".join(parts)+f" || all: OLS a,b={J('all','ols_alpha'):.3f},{J('all','ols_beta'):.3f} IV a,b={J('all','iv_alpha'):.3f},{J('all','iv_beta'):.3f}"
                  f" | ge2: OLS b={J('ge2','ols_beta'):.3f} IV a,b={J('ge2','iv_alpha'):.3f},{J('ge2','iv_beta'):.3f}")

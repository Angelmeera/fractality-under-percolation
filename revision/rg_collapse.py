import json, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SM={"GA":{3:2,4:14,5:34,6:25,7:8,8:1},"GB":{3:4,4:20,5:40,6:26,7:8,8:1}}
def pi(p,w): return sum(s*p**m*(1-p)**(8-m) for m,s in SM[w].items())
def iters(p,w,n=6):
    out=[p]
    for _ in range(n): out.append(pi(out[-1],w))
    return out
for w in ("GB","GA"):
    print(w, "RG flow p_k (k=0..4):")
    for p in (0.95,0.9,0.85,0.8,0.75,0.72,0.7,0.65):
        print("   p=%.2f"%p, [round(x,4) for x in iters(p,w,4)])
# collapse: per-level beta (pub, L=diam+1) vs renormalized occupation
for w,t in (("GB",6),("GB",5),("GA",6),("GA",5)):
    try: d=json.load(open(os.path.join(HERE, "results", f"a24_{w}_t{t}.json")))
    except Exception as e: print(w,t,"missing"); continue
    R=d["recs"]; ps=sorted(set(r["p"] for r in R),reverse=True)
    print(f"\n{w} t={t}: per-level beta_tau (pub | mass | retile) and p_(tau-1)")
    for p in ps:
        G=[r for r in R if r["p"]==p]; fl=iters(p,w,6)
        row=[]
        for tau in range(1,t):
            vals=[np.mean([g[f"{c}_plus1"]["levels"].get(str(tau),np.nan) for g in G]) for c in ("pub","mass","retile")]
            row.append(f"tau{tau}: {vals[0]:.3f}/{vals[1]:.3f}/{vals[2]:.3f} (p_{tau-1}={fl[tau-1]:.3f})")
        print(f"  p={p:.2f} "+" | ".join(row))

import json, sys, numpy as np
def summ(files):
    recs=[]
    for f in files: recs += json.load(open(f))["recs"]
    ps=sorted(set(r["p"] for r in recs), reverse=True)
    out={}
    for p in ps:
        G=[r for r in recs if r["p"]==p]; o={"n":len(G)}
        for c in ("pub","mass","retile"):
            g=lambda f: np.array([f(r[c]) for r in G])
            d={}
            for k in ("ols_beta","iv_beta","ols_alpha","iv_alpha"):
                v=g(lambda x: x["joint"]["ge2"][k]); d["ge2_"+k]=(v.mean(), v.std())
                v=g(lambda x: x["joint"]["all"][k]); d["all_"+k]=(v.mean(), v.std())
            for k in ("ols","iv"):
                v=g(lambda x: x["fe"][k]); d["fe_"+k]=(v.mean(), v.std(), v.std()/np.sqrt(len(v)))
            for tau in ("1","2","3","4"):
                for k in ("ols","iv","within_k0","first","reduced"):
                    v=np.array([r[c]["per"][tau][k] for r in G if tau in r[c]["per"]])
                    if len(v): d[f"t{tau}_{k}"]=(np.nanmean(v), np.nanstd(v))
            v=g(lambda x: x["hub_out"]["all"]); d["hubout_all"]=(v.mean(), v.max())
            v=g(lambda x: x["hub_out"]["ge2"]); d["hubout_ge2"]=(v.mean(), v.max())
            o[c]=d
        out[p]=o
    return out
if __name__=="__main__":
    out=summ(sys.argv[1:])
    for p,o in out.items():
        print(f"p={p} n={o['n']}")
        for c in ("pub","mass","retile"):
            d=o[c]
            print(f"  {c:6s} ge2 OLS b={d['ge2_ols_beta'][0]:.3f} IV b={d['ge2_iv_beta'][0]:.3f}±{d['ge2_iv_beta'][1]:.3f} IV a={d['ge2_iv_alpha'][0]:.3f} | all OLS b={d['all_ols_beta'][0]:.3f} IV b={d['all_iv_beta'][0]:.3f} | FE OLS {d['fe_ols'][0]:.3f} FE IV {d['fe_iv'][0]:.3f}±{d['fe_iv'][2]:.3f}(se) | within-k0 t2 {d.get('t2_within_k0',(np.nan,))[0]:.2f} t3 {d.get('t3_within_k0',(np.nan,))[0]:.2f} | IV t2 {d.get('t2_iv',(np.nan,))[0]:.3f} t3 {d.get('t3_iv',(np.nan,))[0]:.3f} | first t2 {d.get('t2_first',(np.nan,))[0]:.3f} | hubout all {100*d['hubout_all'][0]:.2f}% ge2 {100*d['hubout_ge2'][0]:.2f}% (max {100*d['hubout_ge2'][1]:.1f}%)")

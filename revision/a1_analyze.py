"""Analyse A1 output: closure of d_B = alpha + beta d_k on a single MEMB covering."""
import sys, json
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import lib, os
ELL_LEVEL = os.environ.get("ELL_LEVEL") == "1"

def analyse(path, lconv="plus1", mode="all", kname="kc", cut=3):
    d = json.load(open(path)); radii = d["radii"]
    rows = {}
    rng = np.random.default_rng(0)
    for rec in d["recs"]:
        ell = [3 ** (i + 1) for i in range(len(radii))] if ELL_LEVEL else [2 * r + 1 for r in radii]
        NB = [rec["levels"][str(r)]["NB"] for r in radii]
        s = [rec["levels"][str(r)]["s"] for r in radii]
        dB = -lib.slope(ell, np.array(NB) / rec["Ngc"]); dk = -lib.slope(ell, s)
        M, L, K, lev = [], [], [], []
        for i, r in enumerate(radii, start=1):
            lv = rec["levels"][str(r)]
            Mi = np.array(lv["M"]); Di = np.array(lv["D"])
            Li = Di + 1 if lconv == "plus1" else (np.full(len(Mi), 2 * r + 1) if lconv == "nominal" else np.maximum(Di, 1))
            M += list(Mi); L += list(Li); K += list(lv[kname]); lev += [i] * len(Mi)
        f = lib.joint_fit(M, L, K, lev, mode=mode, cut=cut)
        pl = lib.per_level_beta(M, K, lev, cut=cut)
        sm, sk = lib.level_slopes(M, L, K, lev, cut=cut)
        nul = []
        for _ in range(5):
            Kp = np.array(K, float); rng.shuffle(Kp)
            nul.append(lib.joint_fit(M, L, Kp, lev, mode=mode, cut=cut))
        an = np.mean([x["alpha"] for x in nul]); bn = np.mean([x["beta"] for x in nul])
        rows.setdefault(rec["p"], []).append(dict(dB=dB, dk=dk, a=f["alpha"], b=f["beta"], r2=f["r2"],
            sm=sm, sk=sk, bl=pl, an=an, bn=bn, Ngc=rec["Ngc"], NB=NB))
    a1, b1 = rows[1.0][0]["a"], rows[1.0][0]["b"]
    table = []
    for p in sorted(rows, reverse=True):
        R = rows[p]
        def ms(key):
            v = np.array([x[key] for x in R], float); return float(v.mean()), float(v.std()), float(v.std() / np.sqrt(len(v)))
        clo = np.array([100 * (x["dB"] - (x["a"] + x["b"] * x["dk"])) / x["dB"] for x in R])
        fro = np.array([100 * (x["dB"] - (a1 + b1 * x["dk"])) / x["dB"] for x in R])
        shu = np.array([100 * (x["dB"] - (x["an"] + x["bn"] * x["dk"])) / x["dB"] for x in R])
        # beta that would close exactly given alpha: b* = (dB - alpha)/dk ; and band for |closure|<=3%
        levs = sorted(set(k for x in R for k in x["bl"]))
        bl = {l: float(np.mean([x["bl"][l] for x in R if l in x["bl"]])) for l in levs}
        table.append(dict(p=p, n=len(R), Ngc=ms("Ngc")[0], dB=ms("dB"), dk=ms("dk"), alpha=ms("a"), beta=ms("b"),
                          r2=ms("r2")[0], sm=ms("sm")[0], sk=ms("sk")[0],
                          closure=(float(clo.mean()), float(clo.std()), float(clo.std() / np.sqrt(len(clo)))),
                          frozen=(float(fro.mean()), float(fro.std())), shuffled=(float(shu.mean()), float(shu.std())),
                          beta_levels=bl, NB=[float(np.mean([x["NB"][i] for x in R])) for i in range(len(R[0]["NB"]))]))
    return table

def show(table, title):
    print("\n" + title)
    print(" p    n   dB           dk           alpha        beta         R2     s_m    s_k   | closure%      frozen%   shuffled% | per-level beta | NB")
    for r in table:
        print(f"{r['p']:.2f} {r['n']:3d} {r['dB'][0]:.3f}±{r['dB'][1]:.3f} {r['dk'][0]:.3f}±{r['dk'][1]:.3f} "
              f"{r['alpha'][0]:.3f}±{r['alpha'][1]:.3f} {r['beta'][0]:.3f}±{r['beta'][1]:.3f} {r['r2']:.4f} {r['sm']:.3f} {r['sk']:.3f} | "
              f"{r['closure'][0]:+6.2f}±{r['closure'][2]:.2f}  {r['frozen'][0]:+6.1f}  {r['shuffled'][0]:+6.1f} | "
              + " ".join(f"{k}:{v:.3f}" for k, v in r["beta_levels"].items()) + " | " + ",".join(f"{x:.0f}" for x in r["NB"]))

if __name__ == "__main__":
    path = sys.argv[1]
    out = {}
    for lconv in ("plus1", "nominal", "diam"):
        for mode in ("all", "ge2", "weighted"):
            tb = analyse(path, lconv=lconv, mode=mode)
            out[f"{lconv}_{mode}"] = tb
            if (lconv, mode) in (("plus1", "all"), ("plus1", "ge2"), ("nominal", "all"), ("diam", "all"), ("plus1", "weighted")):
                show(tb, f"== {path.split('/')[-1]}  L={lconv}  fit={mode}")
    json.dump(out, open(path.replace(".json", "_summary.json"), "w"), indent=1)

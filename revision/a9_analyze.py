"""A9 analysis: single MEMB covering over all radii. d_B and d_k from N_B(r), s(r)
(l = 2r+1) over windows; alpha, beta by OLS and by IV (centre's intact degree);
closure for each, plus shuffled-degree null."""
import sys, json, numpy as np

def tsls(y, X, Z):
    P = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    return np.linalg.lstsq(P, y, rcond=None)[0]

path = sys.argv[1]; d = json.load(open(path)); R = d["recs"]; rmax = d["rmax"]
wins = {"r1-16": range(1, rmax + 1), "r1-8": range(1, 9), "r8-16": range(8, rmax + 1)}
rng = np.random.default_rng(0)
rows = {}
for rec in R:
    out = {}
    for wn, w in wins.items():
        ell = np.array([2 * r + 1 for r in w], float)
        NB = np.array([rec["levels"][str(r)]["NB"] for r in w], float); s = np.array([rec["levels"][str(r)]["s"] for r in w])
        out[f"dB_{wn}"] = -np.polyfit(np.log(ell), np.log(NB / rec["Ngc"]), 1)[0]
        out[f"dk_{wn}"] = -np.polyfit(np.log(ell), np.log(s), 1)[0]
    M, L, K, K0 = [], [], [], []
    for r in range(1, rmax + 1):
        lv = rec["levels"][str(r)]
        M += lv["M"]; L += [x + 1 for x in lv["D"]]; K += lv["kc"]; K0 += lv["k0"]
    y, lL, lK, lK0 = (np.log(np.array(x, float)) for x in (M, L, K, K0))
    X = np.column_stack([np.ones(len(y)), lL, lK]); Z = np.column_stack([np.ones(len(y)), lL, lK0])
    o = np.linalg.lstsq(X, y, rcond=None)[0]; v = tsls(y, X, Z)
    Kp = lK.copy(); rng.shuffle(Kp); nu = np.linalg.lstsq(np.column_stack([np.ones(len(y)), lL, Kp]), y, rcond=None)[0]
    out.update(ols_a=o[1], ols_b=o[2], iv_a=v[1], iv_b=v[2], null_a=nu[1], null_b=nu[2])
    rows.setdefault(rec["p"], []).append(out)
print(f"== {d['which']} t={d['t']} MEMB r=1..{rmax}")
print(" p   dB(1-16) dB(1-8) dB(8-16) | dk(1-16) | OLS a,b | IV a,b | closure% OLS / IV / null (dB,dk over r=1-16)")
for p in sorted(rows, reverse=True):
    G = rows[p]; m = lambda k: np.mean([g[k] for g in G])
    cl = lambda a, b: np.mean([100 * (g["dB_r1-16"] - (g[a] + g[b] * g["dk_r1-16"])) / g["dB_r1-16"] for g in G])
    print(f"{p:.2f} {m('dB_r1-16'):.3f} {m('dB_r1-8'):.3f} {m('dB_r8-16'):.3f} | {m('dk_r1-16'):.3f} | {m('ols_a'):.3f},{m('ols_b'):.3f} | {m('iv_a'):.3f},{m('iv_b'):.3f} | "
          f"{cl('ols_a','ols_b'):+.1f} / {cl('iv_a','iv_b'):+.1f} / {cl('null_a','null_b'):+.1f}")

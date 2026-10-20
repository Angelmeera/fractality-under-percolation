"""Analyse A2-A5 output (construction boxes + greedy d_B, d_k on same configurations)."""
import sys, json
import numpy as np

path = sys.argv[1]
d = json.load(open(path)); R = d["recs"]
ps = sorted(set(r["p"] for r in R), reverse=True)
def grp(p): return [r for r in R if r["p"] == p]
def ms(v):
    v = np.array([x for x in v if x is not None and np.isfinite(x)], float)
    return (float(v.mean()), float(v.std()), float(v.std() / np.sqrt(max(len(v), 1)))) if len(v) else (np.nan, np.nan, np.nan)
out = {"which": d["which"], "t": d["t"], "rows": []}

print(f"\n===== {d['which']} t={d['t']}  ({path.split('/')[-1]})")
print("\n-- A2/A3: alpha, beta by convention and fit (L = diam+1); per-level beta; kept mass fraction")
for conv in ("pub", "mass", "retile"):
    print(f" [{conv}]  p | all: alpha beta R2 | ge2: alpha beta | weighted: alpha beta | per-level beta | mass kept")
    for p in ps:
        G = grp(p); k = f"{conv}_plus1"
        a = ms([g[k]["all"]["alpha"] for g in G]); b = ms([g[k]["all"]["beta"] for g in G]); r2 = ms([g[k]["all"]["r2"] for g in G])
        a2 = ms([g[k]["ge2"]["alpha"] for g in G]); b2 = ms([g[k]["ge2"]["beta"] for g in G])
        aw = ms([g[k]["weighted"]["alpha"] for g in G]); bw = ms([g[k]["weighted"]["beta"] for g in G])
        levs = sorted(set(l for g in G for l in g[k]["levels"]))
        bl = {l: ms([g[k]["levels"].get(l) for g in G])[0] for l in levs}
        mf = ms([g[k]["massfrac"] for g in G])[0]
        print(f"   {p:.2f} | {a[0]:.3f} {b[0]:.3f}±{b[1]:.3f} {r2[0]:.4f} | {a2[0]:.3f} {b2[0]:.3f}±{b2[1]:.3f} | {aw[0]:.3f} {bw[0]:.3f} | "
              + " ".join(f"{l}:{v:.3f}" for l, v in bl.items()) + f" | {mf:.3f}")
        out["rows"].append(dict(p=p, conv=conv, alpha=a, beta=b, r2=r2, alpha_ge2=a2, beta_ge2=b2, alpha_w=aw, beta_w=bw, beta_levels=bl, massfrac=mf))

print("\n-- A5: L convention (pub, all-level fit): beta / alpha for L = diam, diam+1, nominal")
for p in ps:
    G = grp(p)
    s = " | ".join(f"{lc}: a={ms([g[f'pub_{lc}']['all']['alpha'] for g in G])[0]:.3f} b={ms([g[f'pub_{lc}']['all']['beta'] for g in G])[0]:.3f}"
                   for lc in ("diam", "plus1", "nominal"))
    s2 = " | ".join(f"{lc} ge2 b={ms([g[f'pub_{lc}']['ge2']['beta'] for g in G])[0]:.3f}" for lc in ("diam", "plus1", "nominal"))
    print(f"   {p:.2f} {s}   || {s2}")

print("\n-- A5: mass cut (pub, L = diam+1): beta for m>=2,3,4,6 (all levels | levels>=2)")
for p in ps:
    G = grp(p)
    s = " ".join(f"{c}:{ms([g['pub_plus1']['cuts'][str(c)]['beta'] for g in G])[0]:.3f}" for c in (2, 3, 4, 6))
    s2 = " ".join(f"{c}:{ms([g['pub_plus1']['cuts_ge2'][str(c)]['beta'] for g in G])[0]:.3f}" for c in (2, 3, 4, 6))
    print(f"   {p:.2f} all {s} | ge2 {s2}")

if "greedy" in R[0]:
    print("\n-- A4: mixed test, greedy d_B, d_k vs construction alpha, beta (all-level fit, L = diam+1)")
    print("   p   conv   dB      dk     | closure%(SE)  frozen  shuffled |  beta   b*     beta band for |closure|<=3%")
    intact = {c: grp(1.0)[0][f"{c}_plus1"]["all"] for c in ("pub", "mass", "retile")}
    for p in ps:
        G = grp(p)
        for conv in ("pub", "mass", "retile"):
            k = f"{conv}_plus1"
            clo, fro, shu, bstar, half = [], [], [], [], []
            for g in G:
                dB, dk = g["greedy"]["dB"], g["greedy"]["dk"]
                f = g[k]["all"]; nl = g[k]["null"]
                clo.append(100 * (dB - (f["alpha"] + f["beta"] * dk)) / dB)
                fro.append(100 * (dB - (intact[conv]["alpha"] + intact[conv]["beta"] * dk)) / dB)
                shu.append(100 * (dB - (nl["alpha"] + nl["beta"] * dk)) / dB)
                den = g[k]["sk"] - dk
                bstar.append((g[k]["sm"] - dB) / den if abs(den) > 1e-9 else np.nan)
                half.append(0.03 * dB / abs(den) if abs(den) > 1e-9 else np.nan)
            dBm = ms([g["greedy"]["dB"] for g in G]); dkm = ms([g["greedy"]["dk"] for g in G])
            c = ms(clo); bs = ms(bstar); hw = ms(half)[0]; bb = ms([g[k]["all"]["beta"] for g in G])[0]
            print(f"   {p:.2f} {conv:6s} {dBm[0]:.3f} {dkm[0]:.3f} | {c[0]:+6.2f}({c[2]:.2f}) {ms(fro)[0]:+6.1f} {ms(shu)[0]:+6.1f} | "
                  f"{bb:.3f} {bs[0]:.3f}  [{bs[0]-hw:.2f}, {bs[0]+hw:.2f}]")
            out["rows"].append(dict(p=p, conv=conv, test="mixed", dB=dBm, dk=dkm, closure=c, frozen=ms(fro), shuffled=ms(shu),
                                    beta=bb, bstar=bs, band=(bs[0] - hw, bs[0] + hw)))
json.dump(out, open(path.replace(".json", "_summary.json"), "w"), indent=1)

import json, sys, numpy as np
f=sys.argv[1]; d=json.load(open(f)); recs=d['recs']; t=d['t']
ps=sorted(set(r['p'] for r in recs), reverse=True)
print("== mass cut:", f)
for p in ps:
    G=[r for r in recs if r['p']==p]
    for c in ('pub','mass','retile'):
        row=[]
        for cut in ('2','3','4','6'):
            g=lambda k1,k2: np.mean([r[c][cut][k1][k2] for r in G])
            row.append(f"cut{cut}: allOLS {g('all','ols_beta'):.3f} allIV {g('all','iv_beta'):.3f} ge2IV {g('ge2','iv_beta'):.3f} FEIV {g('fe','iv'):.3f}")
        rem=np.mean([r[c]['2']['n_removed'] for r in G]), np.mean([r[c]['3']['n_removed'] for r in G])
        print(f"p={p} {c:6s} removed(cut2,cut3)={rem[0]:.1f},{rem[1]:.1f} | "+" | ".join(row))
print("== construction frame (cut 3, levels >= 2):")
for p in ps:
    G=[r for r in recs if r['p']==p]
    for c in ('pub','mass','retile'):
        out=[]
        for r in G:
            lv=r[c]['3']['levels']; T=[k for k in sorted(lv,key=int) if int(k)>=2]
            L=np.array([lv[k]['lnL'] for k in T]); K=np.array([lv[k]['lnk'] for k in T]); M=np.array([lv[k]['lnm'] for k in T]); n=np.array([lv[k]['n'] for k in T])
            dB=-np.polyfit(L,np.log(n/r['Ngc']),1)[0]; sk=np.polyfit(L,K,1)[0]; sm=np.polyfit(L,M,1)[0]
            j=r[c]['3']['ge2']
            out.append((dB,sk,sm,j['ols_alpha']+j['ols_beta']*sk,j['iv_alpha']+j['iv_beta']*sk))
        o=np.mean(out,0)
        print(f"p={p} {c:6s} dB_c={o[0]:.3f} dk_c={o[1]:.3f} sigma_m={o[2]:.3f} | OLS a+b dk={o[3]:.3f} ({100*(o[0]-o[3])/o[0]:+.1f}%) IV a+b dk={o[4]:.3f} ({100*(o[0]-o[4])/o[0]:+.1f}%)")

"""LARGE-GENERATION run: sharpen every headline number.
 A) intact exponents (d_B, alpha, beta, d_k, gamma, closure) at t=4,5,6 via EXACT kinship covering
 B) p_c(N) convergence to the analytical threshold, t=3..6
 C) alpha(p), beta(p) through the transition at t=5 using fixed kinship boxes
Writes results_large.json incrementally so partial results always survive.
"""
import json, time, sys
import numpy as np
from collections import defaultdict
from large_lib import (build_tracked, kinship_colors, degrees_of, analytic,
                       susceptibility_fast, double_sweep_diameter,
                       DF, GAMMA_TH, ALPHA_TH, DK_TH, sparse_adj)
from scipy.sparse.csgraph import connected_components

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _HERE)
OUT = _os.environ.get("RESULTS_OUT",
                      _os.path.join(_HERE, "..", "results", "results_large.json"))
_os.makedirs(_os.path.dirname(_os.path.abspath(OUT)), exist_ok=True)
R = {}
def save():
    json.dump(R, open(OUT,"w"), indent=2, default=float)
t0=time.time()
def el(): return f"[{time.time()-t0:6.0f}s]"

def gamma_ccdf(deg):
    ks=np.sort(deg[deg>=2].astype(float))
    if len(ks)<20: return np.nan
    u=np.unique(ks); cc=np.array([(ks>=k).mean() for k in u])
    m=cc>0
    return float(1-np.polyfit(np.log(u[m]),np.log(cc[m]),1)[0])

# ================= A) INTACT EXPONENTS, EXACT KINSHIP COVERING =================
print("="*70); print("A) INTACT EXPONENTS (exact kinship covering)"); print("="*70)
R["A_intact"]={}
for which in ("GB","GA"):
    R["A_intact"][which]={}
    for t in (4,5,6):
        edges,N,snaps=build_tracked(which,t)
        deg=degrees_of(edges,N)
        # --- d_k EXACT: node degrees multiply by kappa each generation (old nodes persist)
        dk_pts=[]
        for tau in range(1,t):
            old=snaps[t-tau]; n_old=len(old)
            sel=np.where(old>=1)[0]
            ratio=deg[sel].astype(float)/old[sel].astype(float)
            dk_pts.append((3.0**tau, float(np.exp(np.mean(np.log(ratio))))))
        x=np.log([p[0] for p in dk_pts]); y=np.log([p[1] for p in dk_pts])
        dk=float(np.polyfit(x,y,1)[0])      # k grows as l_B^{+dk}; d_k = slope
        # --- box statistics per level tau (N_B and l_B are EXACT)
        NBs, lBs = [], []
        M_all,L_all,K_all=[],[],[]
        for tau in range(1,t):
            e2,N2,col=kinship_colors(which,t,tau)
            d2=degrees_of(e2,N2)
            NBs.append(8.0**(t-tau)); lBs.append(3.0**tau)
            # mass and hub degree per box (vectorised)
            valid=col>=0
            mass=np.bincount(col[valid])
            hub=np.zeros(len(mass))
            np.maximum.at(hub,col[valid],d2[valid])
            keep=mass>1
            M_all.append(mass[keep].astype(float))
            L_all.append(np.full(keep.sum(),3.0**tau))
            K_all.append(hub[keep])
        dB=float(-np.polyfit(np.log(lBs),np.log(np.array(NBs)/N),1)[0])
        Mv=np.concatenate(M_all); Lv=np.concatenate(L_all); Kv=np.concatenate(K_all)
        ok=(Mv>1)&(Kv>0)
        A=np.column_stack([np.ones(ok.sum()),np.log(Lv[ok]),np.log(Kv[ok])])
        sol,*_=np.linalg.lstsq(A,np.log(Mv[ok]),rcond=None)
        alpha,beta=float(sol[1]),float(sol[2])
        g=gamma_ccdf(deg)
        rhs=alpha+beta*dk
        res=abs(dB-rhs)/dB*100
        R["A_intact"][which][t]=dict(N=int(N),M=int(len(edges)),dB=dB,alpha=alpha,beta=beta,
                                     dk=dk,gamma=g,rhs=rhs,closure_pct=res,nboxes=int(ok.sum()))
        print(f"{el()} {which} t={t} N={N}: dB={dB:.4f} alpha={alpha:.4f} beta={beta:.4f} "
              f"dk={dk:.4f} gamma={g:.2f} | a+b*dk={rhs:.4f} closure={res:.2f}%  (boxes={ok.sum()})")
        save()
print(f"  theory: dB={DF:.4f} alpha={ALPHA_TH:.4f} beta=1.0000 dk={DK_TH:.4f} gamma={GAMMA_TH:.2f}")

# ================= B) p_c(N) CONVERGENCE =================
print("="*70); print("B) p_c(N) -> analytical p_c"); print("="*70)
R["B_pc"]={}
for which in ("GB","GA"):
    an=analytic(which); pc_an=an["pc"]
    R["B_pc"][which]={"analytic":an,"sizes":{}}
    for t,ntr in ((3,400),(4,200),(5,80),(6,25)):
        edges,N,_=build_tracked(which,t)
        ps=np.round(np.arange(pc_an-0.20, min(pc_an+0.20,1.0)+1e-9, 0.01),4)
        chi=susceptibility_fast(edges,N,ps,n_trials=ntr,seed=11)
        i=int(np.argmax(chi)); pc_sim=float(ps[i])
        # parabolic refinement of the peak
        if 0<i<len(ps)-1:
            y0,y1,y2=chi[i-1],chi[i],chi[i+1]
            den=(y0-2*y1+y2)
            if den!=0: pc_sim=float(ps[i]-0.01*0.5*(y2-y0)/den)
        R["B_pc"][which]["sizes"][t]=dict(N=int(N),pc_sim=pc_sim,
              err=abs(pc_sim-pc_an), ps=ps.tolist(), chi=chi.tolist(), n_trials=ntr)
        print(f"{el()} {which} t={t} N={N:6d}: pc_sim={pc_sim:.4f}  analytic={pc_an:.4f}  |diff|={abs(pc_sim-pc_an):.4f}")
        save()

# ================= C) alpha(p), beta(p) AT LARGE SIZE =================
print("="*70); print("C) alpha(p), beta(p) at t=5 (fixed kinship boxes)"); print("="*70)
R["C_dyn"]={}
T=5
for which in ("GB",):
    pc_an=analytic(which)["pc"]
    edges,N,_=build_tracked(which,T)
    boxsets=[]
    for tau in (1,2,3,4):
        e2,N2,col=kinship_colors(which,T,tau)
        d={}
        for node,c in enumerate(col):
            if c>=0: d.setdefault(c,[]).append(node)
        boxsets.append((tau,list(d.values())))
    adj_full=defaultdict(list)
    for a,b in edges: adj_full[a].append(b); adj_full[b].append(a)
    ps=np.round(np.arange(0.30,1.0001,0.05),3)
    rng=np.random.default_rng(5)
    al=[];be=[];als=[];bes=[]
    for p in ps:
        av,bv=[],[]
        for _ in range(4):
            keep=rng.random(len(edges))<p
            e=edges[keep]
            A=sparse_adj(e,N); nc,lab=connected_components(A,directed=False)
            sizes=np.bincount(lab); gi=int(np.argmax(sizes))
            ingc=(lab==gi)
            adj={}
            for a,b in e:
                if ingc[a] and ingc[b]:
                    adj.setdefault(a,[]).append(b); adj.setdefault(b,[]).append(a)
            degp=np.bincount(e.ravel(),minlength=N)
            M_,L_,K_=[],[],[]
            for tau,boxes in boxsets:
                for bx in boxes:
                    surv=[n for n in bx if ingc[n]]
                    if len(surv)<3: continue
                    sub={n:[m for m in adj.get(n,()) if ingc[m] and m in set(surv)] for n in surv} if len(surv)<4000 else None
                    if sub is None: continue
                    # largest connected piece inside the box
                    seen=set(); best=[]
                    for s in surv:
                        if s in seen: continue
                        comp=[s]; seen.add(s); q=[s]
                        while q:
                            nq=[]
                            for x in q:
                                for y in sub.get(x,()):
                                    if y not in seen: seen.add(y); comp.append(y); nq.append(y)
                            q=nq
                        if len(comp)>len(best): best=comp
                    if len(best)<3: continue
                    L=double_sweep_diameter({n:sub.get(n,[]) for n in best}, best)
                    if L<1: continue
                    M_.append(len(best)); L_.append(L); K_.append(max(degp[n] for n in best))
            if len(M_)<10: continue
            Mv=np.array(M_,float);Lv=np.array(L_,float);Kv=np.array(K_,float)
            ok=(Mv>1)&(Kv>0)&(Lv>0)
            if ok.sum()<10 or len(set(Lv[ok]))<2 or len(set(Kv[ok]))<2: continue
            A2=np.column_stack([np.ones(ok.sum()),np.log(Lv[ok]),np.log(Kv[ok])])
            s2,*_=np.linalg.lstsq(A2,np.log(Mv[ok]),rcond=None)
            av.append(float(s2[1])); bv.append(float(s2[2]))
        al.append(np.mean(av) if av else np.nan); als.append(np.std(av) if av else np.nan)
        be.append(np.mean(bv) if bv else np.nan); bes.append(np.std(bv) if bv else np.nan)
        print(f"{el()} {which} p={p:.2f}: alpha={al[-1]:.3f} beta={be[-1]:.3f}")
        R["C_dyn"][which]=dict(N=int(N),T=T,pc=pc_an,ps=ps.tolist(),
                               alpha=al,alpha_sd=als,beta=be,beta_sd=bes)
        save()
print(f"DONE {time.time()-t0:.0f}s")
save()

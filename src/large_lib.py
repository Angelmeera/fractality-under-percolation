"""Memory-efficient FSFN tools for LARGE generations (t=5,6; N up to ~150k).
Avoids dense all-pairs distance matrices entirely.

Key exact facts for the Yakubo-Fujiki FSFN used here (n_gen=6, m_gen=8, kappa=2, lambda=3):
  * kinship box at level tau  == the tau-generation expansion of one edge of G_{t-tau}
      -> number of boxes  N_B(tau) = m_gen^(t-tau) = 8^(t-tau)      (EXACT, no covering needed)
      -> box diameter     l_B(tau) = lambda^tau    = 3^tau          (EXACT on G^B; on G^A the
                                                                 diameter is 29, 89 at tau = 3, 4)
  * old nodes persist; a node's degree multiplies by kappa=2 each generation
      -> d_k measurable EXACTLY as log(k_t/k_{t-tau}) / log(3^tau)
"""
import numpy as np, networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.optimize import brentq

GEN = {"GA": [(0,2),(0,3),(1,4),(1,5),(2,3),(2,4),(3,5),(4,5)],
       "GB": [(0,2),(0,3),(1,4),(1,5),(2,4),(2,5),(3,4),(3,5)]}
SM  = {"GA": {3:2,4:14,5:34,6:25,7:8,8:1}, "GB": {3:4,4:20,5:40,6:26,7:8,8:1}}
MGEN, KAPPA, LAM = 8, 2, 3
DF = np.log(8)/np.log(3); GAMMA_TH = 1+np.log(8)/np.log(2)
ALPHA_TH = np.log(4)/np.log(3); DK_TH = np.log(2)/np.log(3)

def analytic(which):
    sm=SM[which]; pi=lambda p: sum(s*p**m*(1-p)**(MGEN-m) for m,s in sm.items())
    pc=brentq(lambda p: pi(p)-p, .3,.99)
    d=(pi(pc+1e-7)-pi(pc-1e-7))/2e-7
    nu=np.log(LAM)/np.log(d)
    return dict(pc=pc, nu=nu, nu_tilde=DF*nu)

def build_tracked(which, t):
    """Build G_t. Returns (edges ndarray (M,2), N, deg_snapshots list of arrays).
    deg_snapshots[s] = degree array of G_s (indexed by node id, len = #nodes at gen s)."""
    ge=GEN[which]; rem=[2,3,4,5]
    edges=np.array([[0,1]],dtype=np.int64); ctr=2
    snaps=[np.bincount(edges.ravel(),minlength=ctr)]
    for _ in range(t):
        M=len(edges)
        fresh=np.arange(ctr, ctr+4*M, dtype=np.int64).reshape(M,4)
        ctr+=4*M
        u=edges[:,0]; v=edges[:,1]
        node=np.empty((M,6),dtype=np.int64)
        node[:,0]=u; node[:,1]=v; node[:,2:]=fresh
        new=np.empty((M*len(ge),2),dtype=np.int64)
        for i,(a,b) in enumerate(ge):
            new[i*M:(i+1)*M,0]=node[:,a]; new[i*M:(i+1)*M,1]=node[:,b]
        edges=new
        snaps.append(np.bincount(edges.ravel(),minlength=ctr))
    return edges, ctr, snaps

def kinship_colors(which, t, tau):
    """Return (edges of G_t, N, color array) where color[node] = index of its level-tau box."""
    ge=GEN[which]; rem=[2,3,4,5]
    edges=np.array([[0,1]],dtype=np.int64); ctr=2
    for _ in range(t-tau):                      # coarse network G_{t-tau}
        M=len(edges)
        fresh=np.arange(ctr,ctr+4*M,dtype=np.int64).reshape(M,4); ctr+=4*M
        node=np.empty((M,6),dtype=np.int64); node[:,0]=edges[:,0]; node[:,1]=edges[:,1]; node[:,2:]=fresh
        new=np.empty((M*8,2),dtype=np.int64)
        for i,(a,b) in enumerate(ge):
            new[i*M:(i+1)*M,0]=node[:,a]; new[i*M:(i+1)*M,1]=node[:,b]
        edges=new
    col=np.full(ctr, -1, dtype=np.int64)         # colors seeded by coarse edges
    ecol=np.arange(len(edges),dtype=np.int64)
    for j,(a,b) in enumerate(edges):
        if col[a]<0: col[a]=ecol[j]
        if col[b]<0: col[b]=ecol[j]
    for _ in range(tau):                         # expand tau gens, inherit colour
        M=len(edges)
        fresh=np.arange(ctr,ctr+4*M,dtype=np.int64).reshape(M,4)
        newctr=ctr+4*M
        col=np.concatenate([col, np.repeat(ecol,4)])
        ctr=newctr
        node=np.empty((M,6),dtype=np.int64); node[:,0]=edges[:,0]; node[:,1]=edges[:,1]; node[:,2:]=fresh
        new=np.empty((M*8,2),dtype=np.int64); nec=np.empty(M*8,dtype=np.int64)
        for i,(a,b) in enumerate(ge):
            new[i*M:(i+1)*M,0]=node[:,a]; new[i*M:(i+1)*M,1]=node[:,b]
            nec[i*M:(i+1)*M]=ecol
        edges=new; ecol=nec
    return edges, ctr, col

def degrees_of(edges, N):
    return np.bincount(edges.ravel(), minlength=N)

def sparse_adj(edges, N):
    r=np.concatenate([edges[:,0],edges[:,1]]); c=np.concatenate([edges[:,1],edges[:,0]])
    return csr_matrix((np.ones(len(r),dtype=np.int8),(r,c)),shape=(N,N))

def susceptibility_fast(edges, N, ps, n_trials, seed=0):
    """chi(p) over FINITE clusters (giant excluded), via sparse connected_components."""
    rng=np.random.default_rng(seed); M=len(edges); chi=np.zeros(len(ps))
    for i,p in enumerate(ps):
        acc=[]
        for _ in range(n_trials):
            keep=rng.random(M)<p
            e=edges[keep]
            A=sparse_adj(e,N) if len(e) else csr_matrix((N,N),dtype=np.int8)
            ncomp,lab=connected_components(A,directed=False)
            sizes=np.bincount(lab)
            sizes=np.sort(sizes)[::-1][1:]              # drop giant
            acc.append(float((sizes.astype(float)**2).sum()/sizes.sum()) if len(sizes) and sizes.sum()>0 else 0.0)
        chi[i]=np.mean(acc)
    return chi

def double_sweep_diameter(adj_list, nodes):
    """Approximate (lower-bound, usually exact on these graphs) diameter via 2 BFS."""
    if len(nodes)<2: return 0
    def bfs(src):
        dist={src:0}; q=[src]
        while q:
            nq=[]
            for x in q:
                for y in adj_list.get(x,()):
                    if y not in dist: dist[y]=dist[x]+1; nq.append(y)
            q=nq
        far=max(dist,key=dist.get)
        return far,dist[far]
    a,_=bfs(nodes[0]); b,d=bfs(a)
    return d

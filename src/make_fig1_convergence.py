"""fig5_convergence -- paper Fig. 1: convergence with system size on G^B.

  left   closure residual |d_B - (alpha + beta d_k)| / d_B of the hub-centred
         mass-law fit, and the within-level estimate of beta, at t = 4, 5, 6
         (N = 2342, 18726, 149798). Read from results/table3_nulls.json, the
         "correct" entry of each generation, i.e. the first row of Table 3.
  right  simulated percolation threshold (peak of the finite-cluster
         susceptibility) against N for both generators, t = 3..6, with the
         analytical thresholds. Read from results/results_large.json
         ("B_pc"), written by run_large.py.

Neither input is recomputed here; the script only draws. Drawn at print size
(paper_style.py): writes figures/fig5_convergence.png and .eps.

Usage (from the repository root):
    python src/make_fig1_convergence.py
Environment overrides: RESULTS_DIR, FIGS_DIR, FIGS_EPS_DIR.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import paper_style as ps

ps.use()


def main():
    with open(os.path.join(ps.RES, "table3_nulls.json")) as fh:
        t3 = json.load(fh)
    with open(os.path.join(ps.RES, "results_large.json")) as fh:
        large = json.load(fh)
    pc = large["B_pc"]
    ts = ["4", "5", "6"]
    N = np.array([large["A_intact"]["GB"][t]["N"] for t in ts], float)
    clo = np.array([t3[t]["correct"]["closure_pct"] for t in ts])
    bw = np.array([t3[t]["correct"]["beta_within"] for t in ts])

    fig, ax = plt.subplots(1, 2, figsize=(ps.WIDTH, 2.15))
    C1, C2 = ps.OI["blue"], ps.OI["green"]

    # ---- left: closure residual and within-level beta
    a = ax[0]
    a.plot(N, clo, marker="o", ls="-", color=C1, ms=4,
           label=r"closure $|d_B-(\alpha+\beta d_k)|/d_B$")
    for i, (x, y) in enumerate(zip(N, clo)):
        last = i == len(N) - 1
        a.annotate(f"{y:.2f}%", (x, y), textcoords="offset points",
                   xytext=(4 if i == 0 else (-4 if last else 0), -9),
                   ha="left" if i == 0 else ("right" if last else "center"),
                   fontsize=6, color=C1)
    a.set(xscale="log", xlabel="$N$", ylim=(0, 0.75),
          title=r"closure residual and within-level $\beta$")
    a.set_ylabel("closure residual (%)", color=C1)
    a.tick_params(axis="y", colors=C1)
    b = a.twinx()
    b.plot(N, bw, marker="s", ls=":", color=C2, ms=4,
           label=r"$\beta$ (within-level)")
    b.axhline(1.0, color=C2, ls=(0, (1, 2)), lw=.8)
    b.text(N[0] * 1.08, 1.002, r"exact $\beta=1$", color=C2, fontsize=6.5)
    b.set_ylim(0.90, 1.015)
    b.set_ylabel(r"$\beta$ (within-level)", color=C2)
    b.tick_params(axis="y", colors=C2)
    b.grid(False)
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = b.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, loc="lower left")
    a.set_xlim(N[0] / 1.3, N[-1] * 1.3)

    # ---- right: simulated p_c against N
    a = ax[1]
    for which in ("GB", "GA"):
        st = ps.GEN[which]
        d = pc[which]
        ks = sorted(d["sizes"], key=int)
        n = np.array([d["sizes"][k]["N"] for k in ks], float)
        p = np.array([d["sizes"][k]["pc_sim"] for k in ks], float)
        a.plot(n, p, marker=st["marker"], ls=st["ls"], color=st["color"],
               ms=4, label=st["label"] + " simulated")
        pa = d["analytic"]["pc"]
        a.axhline(pa, color=st["color"], ls=(0, (1, 2)), lw=.8)
        a.text(n[0], pa + 0.003, st["label"] + f" analytic $p_c={pa:.4f}$",
               color=st["color"], fontsize=6.5)
    a.set(xscale="log", xlabel="$N$", ylabel="$p_c$", ylim=(0.555, 0.715),
          title=r"simulated $p_c$ against system size")
    a.legend(loc="lower right")

    fig.tight_layout(pad=.3, w_pad=1.0)
    ps.save(fig, "fig5_convergence")


if __name__ == "__main__":
    main()

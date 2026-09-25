"""fig1_pc_dB and fig4_clustering -- paper Figs 1 and 3.

  fig1_pc_dB       G^B at t = 4: left, the finite-cluster susceptibility with
                   the analytical p_c and the sweep's highest point; right,
                   d_B(p) of the giant component with the exact D_f.
  fig4_clustering  G^A (clustered) against G^B (triangle-free): left, the
                   susceptibility with both analytical thresholds; right, d_B(p).

Drawn from results/results_yf.json, which run_yf.py writes; nothing is
recomputed here, so the figures can be redrawn without networkx. run_yf.py
calls draw() after it has measured the data. Print size, via paper_style.py.

Usage (from the repository root):
    python src/make_fig1_fig4.py
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
DF_EXACT = float(np.log(8) / np.log(3))   # ln m_gen / ln lambda = 1.8928


def draw(res):
    # ---------------- fig1: G^B susceptibility and d_B(p)
    d = res["GB"]
    p = np.array(d["ps"])
    B = ps.GEN["GB"]
    fig, ax = plt.subplots(1, 2, figsize=(ps.WIDTH, 2.1))
    a = ax[0]
    a.plot(p, d["chi"], marker="o", ls="-", color=B["color"], ms=2.8)
    a.axvline(d["pc_analytic"], ls="--", lw=.9, color=ps.OI["red"],
              label=r"analytic $p_c$")
    a.axvline(d["pc_sim"], ls=":", lw=1.1, color=ps.OI["black"],
              label="highest point")
    a.set(xlabel="$p$", ylabel=r"$\chi$", title=r"$G^{B}$: finite-cluster susceptibility")
    a.legend(loc="upper right")
    a = ax[1]
    a.errorbar(p, d["dBp"], yerr=d["dBsd"], marker="o", ls="-", color=B["color"],
               ms=2.8, ecolor=ps.OI["grey"], elinewidth=.6)
    a.axvline(d["pc_analytic"], ls="--", lw=.9, color=ps.OI["red"],
              label=f"analytic $p_c={d['pc_analytic']:.4f}$")
    a.axhline(DF_EXACT, ls=":", lw=1.1, color=ps.OI["black"],
              label=f"exact $D_f={DF_EXACT:.3f}$")
    a.set(xlabel="$p$", ylabel="$d_B$", title=r"$G^{B}$: box dimension against $p$")
    a.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none", framealpha=1)
    fig.tight_layout(pad=.3, w_pad=1.0)
    ps.save(fig, "fig1_pc_dB")

    # ---------------- fig4: G^A against G^B
    fig, ax = plt.subplots(1, 2, figsize=(ps.WIDTH, 2.1))
    for which, name in (("GB", "triangle-free"), ("GA", "clustered")):
        st = ps.GEN[which]
        dd = res[which]
        pp = np.array(dd["ps"])
        ax[0].plot(pp, dd["chi"], marker=st["marker"], ls=st["ls"], color=st["color"],
                   ms=2.6, label=st["label"] + f" ({name})")
        ax[0].axvline(dd["pc_analytic"], ls=(0, (1, 2)), lw=.9, color=st["color"])
        ax[1].plot(pp, dd["dBp"], marker=st["marker"], ls=st["ls"], color=st["color"],
                   ms=2.6, label=st["label"] + f" ({name})")
    ax[0].set(xlabel="$p$", ylabel=r"$\chi$",
              title="finite-cluster susceptibility; dotted: analytic $p_c$")
    ax[1].set(xlabel="$p$", ylabel="$d_B$", title=r"box dimension against $p$")
    ax[1].legend(loc="upper left")
    fig.tight_layout(pad=.3, w_pad=1.0)
    ps.save(fig, "fig4_clustering")


def main():
    with open(os.path.join(ps.RES, "results_yf.json")) as fh:
        draw(json.load(fh))


if __name__ == "__main__":
    main()

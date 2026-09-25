"""fig9_fss -- paper Fig. 4: finite-size scaling of the percolation transition.

  (a) susceptibility over finite clusters, G^B at t = 3..6, with error bars
  (b) shift of the pseudo-critical point against N, with the ANALYTIC slope
  (c) collapse of the order parameter using Yakubo-Fujiki's analytic p_c, nu~
      and beta_P -- no fitted parameter enters, so a collapse is evidence
  (d) collapse of the susceptibility, which needs gamma_chi and therefore uses
      our measured value (they do not publish one for these generators)

Generation is carried by colour AND marker AND line style, generator by filled
symbol (G^B) against cross (G^A), so the panels survive greyscale printing;
the palette is Okabe-Ito. Data from results/results_fss.json; print size via
paper_style.py.

Usage (from the repository root):
    python src/make_fss_fig.py
Environment overrides: RESULTS_DIR, FIGS_DIR, FIGS_EPS_DIR.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import paper_style as ps

ps.use()
OI = ps.OI
# fixed order by generation, never cycled
GEN_STYLE = [(3, OI["sky"], "v", ":"), (4, OI["green"], "s", "-."),
             (5, OI["orange"], "^", "--"), (6, OI["blue"], "o", "-")]
# Yakubo-Fujiki Table 1
ANALYTIC = {"GB": dict(pc=0.6288, nu=1.7772, nut=3.3638, beta=0.1098),
            "GA": dict(pc=0.6961, nu=1.8293, nut=3.4626, beta=0.0595)}
LAB = {"GB": r"$G^{B}$", "GA": r"$G^{A}$"}


def collapse_legend(ax, loc):
    h = [Line2D([], [], color=col, marker=mk, ls="none", ms=3, label=f"$t={t}$")
         for t, col, mk, ls in GEN_STYLE]
    h += [Line2D([], [], color="k", marker="o", ls="none", ms=3,
                 label=r"$G^{B}$ (filled)"),
          Line2D([], [], color="k", marker="x", ls="none", ms=3.5,
                 label=r"$G^{A}$ ($\times$)")]
    ax.legend(handles=h, loc=loc, ncol=2, columnspacing=.5,
              handletextpad=.1, fontsize=6)


def sweep(sweeps, which, t):
    return next((s for s in sweeps if s["which"] == which and s["t"] == t), None)


def main():
    with open(os.path.join(ps.RES, "results_fss.json")) as fh:
        d = json.load(fh)
    sweeps = d["sweeps"]
    exps = d["exponents"]
    fig, ax = plt.subplots(2, 2, figsize=(ps.WIDTH, 3.9))
    a, b, c, e = ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]

    # ---------------- (a) susceptibility, generator G^B
    an = ANALYTIC["GB"]
    for t, col, mk, ls in GEN_STYLE:
        r = sweep(sweeps, "GB", t)
        if not r:
            continue
        a.errorbar(r["ps"], r["chi"], yerr=r["chi_sem"], marker=mk, ls=ls,
                   color=col, ms=2.4, lw=.9, elinewidth=.5, capsize=1,
                   label=f"$t={t}$, $N=" + f"{r['N']:,}".replace(",", r"\,") + "$")
    a.axvline(an["pc"], color="k", lw=.7, ls="-.")
    a.text(an["pc"], 0.97, r" analytic $p_c$", fontsize=6, va="top",
           transform=a.get_xaxis_transform())
    a.set(xlabel="$p$", ylabel=r"$\chi$ (finite clusters)", yscale="log",
          title=r"(a) $G^{B}$: susceptibility")
    a.set_ylim(bottom=0.25)
    a.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="none",
             framealpha=1, borderpad=.2)

    # ---------------- (b) shift of the pseudo-critical point
    for which in ("GB", "GA"):
        st = ps.GEN[which]
        rows = sorted([s for s in sweeps if s["which"] == which], key=lambda s: s["t"])
        if not rows:
            continue
        pc = ANALYTIC[which]["pc"]
        N = np.array([r["N"] for r in rows], float)
        dev = np.array([abs(r["pc_sim"] - pc) for r in rows], float)
        ex = exps[which]["shift"]
        b.plot(N, dev, marker=st["marker"], ls="none", color=st["color"], ms=4,
               label=LAB[which] + fr" measured, $1/\tilde\nu={ex['inv_nu_tilde_measured']:.3f}"
                     fr"\pm{ex['slope_se']:.3f}$")
        xx = np.logspace(np.log10(N.min()), np.log10(N.max()), 30)
        slope = -1.0 / ANALYTIC[which]["nut"]     # ANALYTIC slope, anchored at largest N
        b.plot(xx, dev[-1] * (xx / N[-1]) ** slope, color=st["color"], ls=st["ls"],
               lw=1, label=LAB[which] + fr" analytic slope $-1/\tilde\nu={slope:.3f}$")
    b.set(xscale="log", yscale="log", xlabel="$N$", ylabel=r"$|p_c(N)-p_c|$",
          title=r"(b) shift exponent $1/\tilde\nu$")
    b.set_ylim(0.0028, 0.095)
    b.set_yticks([0.005, 0.01, 0.02, 0.05])
    b.set_yticklabels(["0.005", "0.01", "0.02", "0.05"])
    b.yaxis.set_minor_formatter(plt.NullFormatter())
    b.legend(loc="lower left", fontsize=6)

    # ---------------- (c), (d) collapses
    g_meas = {w: exps[w]["susceptibility"]["gamma_over_nutilde_measured"] for w in ("GB", "GA")}
    for which in ("GB", "GA"):
        an = ANALYTIC[which]
        for t, col, mk, ls in GEN_STYLE:
            r = sweep(sweeps, which, t)
            if not r:
                continue
            N = r["N"]
            x = (np.array(r["ps"]) - an["pc"]) * N ** (1.0 / an["nut"])
            kw = dict(ls="none", color=col, ms=2.4 if which == "GB" else 2.8,
                      marker=mk if which == "GB" else "x",
                      mec="white" if which == "GB" else col, mew=.3 if which == "GB" else .6)
            c.plot(x, np.array(r["P"]) * N ** (an["beta"] / an["nut"]), **kw)
            e.plot(x, np.array(r["chi"]) * N ** (-g_meas[which]), **kw)
    c.set(xlabel=r"$(p-p_c)\,N^{1/\tilde\nu}$", ylabel=r"$P_t\,N^{\beta_P/\tilde\nu}$",
          title="(c) order-parameter collapse")
    e.set(xlabel=r"$(p-p_c)\,N^{1/\tilde\nu}$", ylabel=r"$\chi\,N^{-\gamma_\chi/\tilde\nu}$",
          yscale="log", title="(d) susceptibility collapse")
    e.set_ylim(bottom=4e-6)
    collapse_legend(c, "lower right")
    collapse_legend(e, "lower left")

    fig.tight_layout(pad=.3, w_pad=1.0, h_pad=.8)
    ps.save(fig, "fig9_fss")


if __name__ == "__main__":
    main()

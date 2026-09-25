"""fig3_alpha_beta_corrected -- paper Fig. 2: alpha(p), beta(p) under three
box-mass conventions, plus the mass-law reconstruction alpha + beta d_k.
Data from results/m5_control_t5.json (m5_control_t5.py); nothing recomputed.
Drawn at print size via paper_style.py.

Usage (from the repository root):
    python src/make_fig3_conventions.py
Environment overrides: M5_JSON, RESULTS_DIR, FIGS_DIR, FIGS_EPS_DIR.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import paper_style as ps

ps.use()
# generator G^B: lambda = 3, kappa = 2, m_gen = 8 (as in yakubo_fujiki.py; set
# here so that drawing does not need networkx)
LAM, KAPPA, MGEN = 3, 2, 8


def main():
    d = json.load(open(os.environ.get("M5_JSON", os.path.join(ps.RES, "m5_control_t5.json"))))
    rows = d["rows"]; pc = d["pc"]
    p = np.array([r["p"] for r in rows], float)
    a_th = float(np.log(MGEN / KAPPA) / np.log(LAM))
    dk = float(np.log(KAPPA) / np.log(LAM))
    dB = float(np.log(MGEN) / np.log(LAM))

    fig, ax = plt.subplots(1, 3, figsize=(ps.WIDTH, 2.05))
    styles = [("pub", "largest component", ps.OI["blue"], "o", "-"),
              ("mass", "total surviving mass", ps.OI["red"], "s", "--"),
              ("hub", "hub present only", ps.OI["green"], "^", ":")]
    for key, lab, c, mk, ls in styles:
        for k, q in ((0, "alpha"), (1, "beta")):
            ax[k].errorbar(p, [r[f"{key}_{q}"] for r in rows],
                           yerr=[r[f"{key}_{q}_sd"] for r in rows],
                           marker=mk, ms=2.6, ls=ls, color=c, elinewidth=.6,
                           label=lab)
    for axis, th, name in ((ax[0], a_th, r"exact $\alpha=\ln4/\ln3$"),
                           (ax[1], 1.0, r"exact $\beta=1$")):
        axis.axhline(th, color="k", lw=.7, ls=":")
        axis.axvline(pc, color="k", lw=.7, ls="-.")
        axis.text(0.97, th, name, fontsize=6, va="bottom", ha="right",
                  bbox=dict(fc="white", ec="none", pad=0.3),
                  transform=axis.get_yaxis_transform())
    for key, lab, c, mk, ls in styles[:2]:
        rec = (np.array([r[key + "_alpha"] for r in rows])
               + np.array([r[key + "_beta"] for r in rows]) * dk)
        ax[2].plot(p, rec, marker=mk, ms=2.6, ls=ls, color=c)
    ax[2].axhline(dB, color="k", lw=.7, ls=":")
    ax[2].axvline(pc, color="k", lw=.7, ls="-.")
    ax[2].text(0.97, dB, r"exact $d_{B}=\ln8/\ln3$", fontsize=6, va="bottom",
               ha="right", bbox=dict(fc="white", ec="none", pad=0.3), transform=ax[2].get_yaxis_transform())
    ax[0].set(ylabel=r"$\alpha(p)$", title="(a) spreading exponent")
    ax[1].set(ylabel=r"$\beta(p)$", title="(b) mass–degree exponent")
    ax[2].set(ylabel=r"$\alpha+\beta d_{k}$", title="(c) reconstruction")
    ax[0].set_ylim(top=a_th + 0.07)
    ax[1].set_ylim(top=1.06)
    ax[2].set_ylim(top=dB + 0.09)
    for a in ax:
        a.set_xlabel("$p$")
        a.set_xticks([0.4, 0.6, 0.8, 1.0])
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=.3, w_pad=.6, rect=(0, 0, 1, 0.9))
    ps.save(fig, "fig3_alpha_beta_corrected")


if __name__ == "__main__":
    main()

"""Revised figures for the referee reply.
fig2: mass-law exponents under damage, OLS on the damaged degree vs IV with the
      pre-damage degree (G^B t=5, 20 configurations; per level at t=7).
fig3: large-scale restoration at t=7 (hub-degree thinning, surviving mass,
      chemical stretch) for G^A and G^B.
Output: figures/fig2_mass_law_iv and figures/fig3_restoration (PNG + EPS, via src/paper_style.py).
Data: revision/results/a13_GB_t5.json (three conventions; pub/mass identical to a8_GB_t5.json), a10_GB_t7.json, a10_GA_t7.json."""
import json, os, sys
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, _os.path.join(HERE, "..", "src"))
import numpy as np
import matplotlib.pyplot as plt
import paper_style as ps

R = os.environ.get("RES", os.path.join(HERE, "results"))
ps.use()
PC = {"GB": 0.6288, "GA": 0.6961}
B, Rd = ps.OI["blue"], ps.OI["red"]

def fig2():
    d = json.load(open(f"{R}/a13_GB_t5.json")); recs = d["recs"]
    ps_ = sorted(set(r["p"] for r in recs))
    G = ps.OI["green"]
    fig, ax = plt.subplots(1, 3, figsize=(ps.WIDTH, 2.15))
    CONV = (("pub", B, "o", "largest component"), ("mass", Rd, "s", "total surviving mass"), ("retile", G, "^", "re-tiled"))
    for conv, c, mk, lab in CONV:
        # (a) usual joint OLS fit over levels >= 2 (dashed, open) vs within-level IV with level fixed effects (solid, filled)
        for key, ls, fill, elab in ((lambda g: g[conv]["joint"]["ge2"]["ols_beta"], "--", "white", "OLS on $k$"),
                                    (lambda g: g[conv]["fe"]["iv"], "-", c, "IV with $k_{0}$")):
            y = [np.mean([key(g) for g in recs if g["p"] == p]) for p in ps_]
            e = [np.std([key(g) for g in recs if g["p"] == p]) for p in ps_]
            ax[0].errorbar(ps_, y, yerr=e, color=c, ls=ls, marker=mk, ms=2.8, mfc=fill, mec=c, elinewidth=.6, label=f"{lab}, {elab}")
        # (c) alpha, joint fit over levels >= 2, OLS vs IV
        for est, ls, fill in (("ols", "--", "white"), ("iv", "-", c)):
            y = [np.mean([g[conv]["joint"]["ge2"][f"{est}_alpha"] for g in recs if g["p"] == p]) for p in ps_]
            e = [np.std([g[conv]["joint"]["ge2"][f"{est}_alpha"] for g in recs if g["p"] == p]) for p in ps_]
            ax[2].errorbar(ps_, y, yerr=e, color=c, ls=ls, marker=mk, ms=2.8, mfc=fill, mec=c, elinewidth=.6)
    # (b) per level at t=7, p = 0.8, largest-component convention; generators in black / grey
    for which, c, mk in (("GB", "k", "o"), ("GA", ps.OI["grey"], "D")):
        dd = json.load(open(f"{R}/a10_{which}_t7.json"))["recs"]
        Gp = [g for g in dd if g["p"] == 0.8]
        taus = np.arange(1, 7)
        for est, ls, fill in (("ols", "--", "white"), ("iv", "-", c)):
            y = [np.mean([g["levels"][str(t)][f"pub_{est}"] for g in Gp]) for t in taus]
            e = [np.std([g["levels"][str(t)][f"pub_{est}"] for g in Gp]) for t in taus]
            ax[1].errorbar(taus + (0.1 if which == "GA" else -0.1), y, yerr=e, color=c, ls=ls, marker=mk, ms=2.8, mfc=fill, mec=c, elinewidth=.6,
                           label=f"{'$G^{B}$' if which == 'GB' else '$G^{A}$'}, {'OLS' if est == 'ols' else 'IV'}")
    for k in (0, 2):
        ax[k].axvline(PC["GB"], color="k", lw=.7, ls="-.")
        ax[k].set_xlabel("$p$"); ax[k].set_xticks([0.65, 0.75, 0.85, 0.95])
    ax[0].axhline(0.989, color="k", lw=.7, ls=":")
    ax[2].axhline(1.267, color="k", lw=.7, ls=":")
    ax[1].axhline(1.0, color="k", lw=.7, ls=":")
    ax[0].set(ylabel=r"$\beta$ (levels $\tau\geq2$)", title=r"(a) $\beta(p)$, $G^{B}$, $t=5$")
    ax[1].set(ylabel=r"$\beta_{\tau}$", xlabel=r"level $\tau$", title=r"(b) per level, $p=0.8$, $t=7$")
    ax[2].set(ylabel=r"$\alpha$ (levels $\tau\geq2$)", title=r"(c) $\alpha(p)$, $G^{B}$, $t=5$")
    ax[1].set_xticks(range(1, 7)); ax[1].legend(loc="lower right", fontsize=5.6)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), fontsize=5.6, columnspacing=.8)
    fig.tight_layout(pad=.3, w_pad=.6, rect=(0, 0, 1, 0.80))
    ps.save(fig, "fig2_mass_law_iv")

def fig3():
    fig, ax = plt.subplots(1, 3, figsize=(ps.WIDTH, 2.0))
    cols = {0.9: ps.OI["blue"], 0.8: ps.OI["green"], 0.75: ps.OI["orange"], 0.72: ps.OI["red"], 0.65: ps.OI["red"]}
    for which, ls, mk, plist in (("GB", "-", "o", (0.9, 0.8, 0.65)), ("GA", "--", "D", (0.9, 0.8, 0.72))):
        dd = json.load(open(f"{R}/a10_{which}_t7.json"))["recs"]
        for p in plist:
            G = [g for g in dd if g["p"] == p]; taus = np.arange(1, 7)
            lab = f"{'$G^{B}$' if which == 'GB' else '$G^{A}$'}, $p={p}$"
            for k, key, off in ((0, "thin", np.log(p)), (1, "massfrac_pub", 0.0), (2, "stretch", 0.0)):
                y = [np.mean([g["levels"][str(t)][key] for g in G]) - off for t in taus]
                ax[k].plot(taus, y, ls=ls, marker=mk, ms=2.6, color=cols[p], mfc="white" if which == "GA" else cols[p], label=lab)
    for a in ax:
        a.axhline(0, color="k", lw=.7, ls=":"); a.set_xlabel(r"level $\tau$"); a.set_xticks(range(1, 7))
    ax[0].set(ylabel=r"$\langle\ln(k/k_{0})\rangle-\ln p$", title="(a) hub degree")
    ax[0].set_ylim(-0.1, 0.1)
    ax[1].set(ylabel=r"$\langle\ln(m/m_{0})\rangle$", title="(b) surviving mass")
    ax[2].set(ylabel=r"$\langle\ln(L/\lambda^{\tau})\rangle$", title="(c) chemical stretch")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), fontsize=6.2)
    fig.tight_layout(pad=.3, w_pad=.6, rect=(0, 0, 1, 0.80))
    ps.save(fig, "fig3_restoration")

if __name__ == "__main__":
    fig2(); fig3()

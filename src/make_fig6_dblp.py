"""fig6_dblp -- paper Fig. 6: bond percolation of the DBLP w>=60 backbone.

  (a) finite-cluster susceptibility chi(p), mean and standard error; the band
      marks every p whose chi lies within two standard errors of the maximum
  (b) d_B, gamma and delta of the largest component against p
  (c) the macroscopic alpha = (delta-2)/(delta-1) d_B and beta = (gamma-1)/(delta-1)

Everything is read from results/dblp60_percolation.json, written by
src/dblp_percolation.py; nothing is recomputed here. Error bars are the
standard deviation over realizations. delta (and so alpha, beta) exists only
for realizations whose largest component has at least one l_B level with 25
boxes at mu >= 1; rows where fewer than half the realizations qualify are drawn
with open symbols, and rows with fewer than two are not drawn. The p = 1 point
is the intact backbone, i.e. the w>=60 row of Table 6.

The old version of this figure had a d_B-versus-size panel (a), which duplicated
Fig. 5(c), and was drawn from results_dblp_final.json; neither is used now.

Drawn at print size via paper_style.py; no transparency (EPS).

Usage (from the repository root):
    python src/make_fig6_dblp.py
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
BLUE, GREY, RED = ps.OI["blue"], ps.OI["grey"], ps.OI["red"]


def series(rows, key, need_delta=False):
    p, m, s, full = [], [], [], []
    for r in rows:
        n = r.get("delta_n", r["n_used"]) if need_delta else r["n_used"]
        if n < (2 if r["p"] < 1 else 1) or not np.isfinite(r[key]):
            continue
        p.append(r["p"]); m.append(r[key]); s.append(r[key + "_sd"])
        full.append(n >= r["n_used"] / 2)
    return (np.array(p), np.array(m), np.array(s), np.array(full, bool))


def draw(a, rows, key, col, mk, label, need_delta=False, ls="-"):
    p, m, s, full = series(rows, key, need_delta)
    a.plot(p, m, ls=ls, color=col, lw=.9)
    a.errorbar(p[full], m[full], yerr=s[full], fmt=mk, color=col, ms=3,
               elinewidth=.6, label=label)
    if (~full).any():
        a.errorbar(p[~full], m[~full], yerr=s[~full], fmt=mk, color=col,
                   mfc="white", ms=3, elinewidth=.6)


def main():
    with open(os.path.join(ps.RES, "dblp60_percolation.json")) as fh:
        d = json.load(fh)
    rows = d["rows"]
    pc = np.array(d["ps_chi"]); chi = np.array(d["chi"]); se = np.array(d["chi_se"])
    keep = pc < 1.0          # chi vanishes identically at p = 1 (no finite clusters)
    pc, chi, se = pc[keep], chi[keep], se[keep]

    fig, ax = plt.subplots(1, 3, figsize=(ps.WIDTH, 2.1))

    # ---- (a) susceptibility
    a = ax[0]
    i = int(np.argmax(chi))
    near = pc[chi >= chi[i] - 2 * se[i]]
    a.axvspan(near.min(), near.max(), ymax=0.6, color=ps.tint(RED, .86), lw=0,
              label="$\\chi$ within two standard\nerrors of its maximum\n" + f"($p={near.min():.2f}$–${near.max():.2f}$)")
    a.fill_between(pc, chi - se, chi + se, color=ps.tint(BLUE, .6), lw=0)
    a.plot(pc, chi, "-", color=BLUE, lw=.9, label="mean $\\pm$ one\nstandard error")
    a.set(xlabel="$p$", ylabel=r"$\chi$ (finite clusters)",
          title="(a) susceptibility")
    a.legend(loc="upper left")
    a.set_ylim(0, chi.max() * 1.9)

    # ---- (b) d_B, gamma, delta
    a = ax[1]
    draw(a, rows, "dB", BLUE, "o", r"$d_B$")
    draw(a, rows, "gamma", GREY, "s", r"$\gamma$", ls="--")
    draw(a, rows, "delta", RED, "^", r"$\delta$", need_delta=True, ls=":")
    a.set(xlabel="$p$", ylabel="exponent", ylim=(1.4, 4.0),
          title="(b) largest component")
    a.legend(loc="center", bbox_to_anchor=(0.5, 0.42), ncol=3, columnspacing=.8, handletextpad=.2)

    # ---- (c) alpha, beta
    a = ax[2]
    draw(a, rows, "alpha_macroscopic", BLUE, "o",
         r"$\alpha=\frac{\delta-2}{\delta-1}d_B$", need_delta=True)
    draw(a, rows, "beta_macroscopic", RED, "s",
         r"$\beta=\frac{\gamma-1}{\delta-1}$", need_delta=True, ls="--")
    a.set(xlabel="$p$", ylabel="exponent", ylim=(0.6, 1.4),
          title=r"(c) derived $\alpha$ and $\beta$")
    a.legend(loc="upper left", ncol=1)
    for a in ax:
        a.set_xlim(0.28, 1.02)
        a.set_xticks([0.4, 0.6, 0.8, 1.0])

    fig.tight_layout(pad=.3, w_pad=.6)
    ps.save(fig, "fig6_dblp")
    print("chi band: p = %.2f-%.2f" % (near.min(), near.max()))


if __name__ == "__main__":
    main()

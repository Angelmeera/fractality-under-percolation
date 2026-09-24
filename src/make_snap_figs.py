"""Figures for the empirical-networks section.

  fig7_fractality   does the box-counting curve actually follow a power law?
                        (a) log-log, (b) log-linear, (c) the sign of d_k
  fig8_grid_and_size  how the l_B grid was chosen, and whether d_B survives
                        changing the network's size
                        (a) d_B vs R^2 for every candidate grid, (b) the WWW
                        parity effect, (c) d_B vs N across the DBLP sweep

Inputs (all in results/): the *_comm.json records behind Table 9, the
*_scan.json candidate-grid scans, controls.json and dblp_sweep_canonical.json.
The earlier version read results_nonfractal_controls.json and the old
results_*_scan.json files, which predate the canonical cover, so Fig. 8 and
Fig. 6(a) disagreed with Table 9 in the second decimal.

Reads whatever result JSONs exist and silently skips the rest, so it can be run
before every dataset has been fetched.

Style notes: series identity is carried by colour AND marker AND line style, so
the panels survive greyscale printing and colour-vision deficiency; the palette
is Okabe-Ito, which is CVD-safe by construction. No panel uses two y-scales.
Drawn at print size via paper_style.py (PNG + vector EPS, no transparency);
the series keys below are the names stored in the result files, and DISPLAY
holds the typeset labels printed in the figures.

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
RES = os.environ.get("SNAP_RESULTS", ps.RES)
OI = ps.OI

# (key in the result files, colour, marker, linestyle, is_expected_fractal)
STYLE = [
    ("WWW", OI["blue"], "o", "-", True),
    ("DBLP(w>=25)", OI["green"], "s", "-", True),
    ("DBLP(w>=60)", OI["sky"], "P", "-", True),
    ("FSFN-GB-t6", OI["black"], "D", "-", True),
    ("Internet-AS", OI["red"], "v", "--", False),
    ("BA m=2 N=6474", OI["orange"], "^", "--", False),
    ("ER <k>=4 N=6474", OI["purple"], "X", ":", False),
    # Reported even though it PASSES the power-law test: a random scale-free tree
    # does have self-similar structure, so the discriminator tests fractality,
    # not "was this grown by preferential attachment". Hiding it would be
    # cherry-picking the controls.
    ("BA m=1 N=6474 (tree)", "#B8A800", "*", "-.", False),
]
# typeset labels, as the paper writes them
DISPLAY = {
    "WWW": "WWW",
    "DBLP(w>=25)": r"DBLP $w\geq25$",
    "DBLP(w>=60)": r"DBLP $w\geq60$",
    "FSFN-GB-t6": r"FSFN $G^{B}$, $t=6$",
    "Internet-AS": "Internet AS",
    "BA m=2 N=6474": r"BA, $m=2$",
    "ER <k>=4 N=6474": r"ER, $\langle k\rangle=4$",
    "BA m=1 N=6474 (tree)": r"BA tree, $m=1$",
}


def network_legend(fig, keys, y=0.0):
    """One legend for all panels, below the axes (the panels are too narrow)."""
    h = [Line2D([], [], color=c, marker=mk, ls=ls, ms=3.5, lw=.9, label=DISPLAY[k])
         for k, c, mk, ls, _ in STYLE if k in keys]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, y),
               columnspacing=1.2, handletextpad=.4)


def _load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        print(f"  (missing {name} -- skipped)")
        return None
    with open(p) as fh:
        return json.load(fh)


# Where each series comes from. "main" is the record behind Table 9 (the
# selected grid, canonical str(node)-tie-broken cover); "scan" holds the
# candidate grids for Fig. 6(a) and, for the WWW, the per-l_B counts for
# Fig. 6(b). Every file is in results/ and is written by the command given in
# docs/REPRODUCING.md.
SOURCES = [
    # (main file, scan file or None)
    ("www_comm.json", "www_scan.json"),
    ("as_comm.json", "as_scan.json"),
    ("dblp25_comm.json", "dblp25_scan.json"),
    ("dblp60_comm.json", "dblp60_scan.json"),
    ("fsfn6_comm.json", "fsfn6_scan.json"),
]


def collect():
    """Gather {label: record} where record has l_B, NB, N, dB, fractality, ...,
    plus grid_scan (candidate grids) and per_lB (WWW only)."""
    out = {}
    for main_fn, scan_fn in SOURCES:
        d = _load(main_fn)
        if not d:
            continue
        m = d["main"]
        rec = {**m, "grid_scan": None, "per_lB": None}
        s = _load(scan_fn) if scan_fn else None
        if s:
            rec["grid_scan"] = s.get("grid_scan")
            rec["per_lB"] = s.get("per_lB")
        out[m["name"]] = rec
    d = _load("controls.json")
    if d:
        for label, rec in d.items():
            m = rec["main"]
            out[m["name"]] = {**m, "grid_scan": rec.get("grid_scan")}
    return out


def _pick(data, label):
    """Exact match first, then a prefix match, so 'DBLP(w>=25)' never picks up
    'DBLP(w>=60)'."""
    if label in data:
        return data[label]
    hits = [v for k, v in data.items() if k == label]
    return hits[0] if hits else None


def _barh(a, rows, title, xlabel, fmt="{:.2f}", zero_line=True, yticks=True):
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    cols = [r[2] for r in rows]
    y = np.arange(len(labels))
    a.barh(y, vals, color=cols, height=.62, edgecolor="k", linewidth=.3)
    if zero_line:
        a.axvline(0, color="k", lw=.7)
    a.set_yticks(y)
    a.set_yticklabels(labels if yticks else [], fontsize=6.5)
    a.set(xlabel=xlabel, title=title)
    a.grid(axis="y", visible=False)
    span = max(abs(v) for v in vals) if vals else 1
    for yi, v in zip(y, vals):
        a.text(v + (.03 * span if v >= 0 else -.03 * span), yi,
               fmt.format(v).replace("-", "\u2212"),
               va="center", ha="left" if v >= 0 else "right", fontsize=6)
    a.margins(x=.42)


def fig_fractality(data):
    """(a) the curves; (b) which model the data prefer, and by how much;
    (c) the sign of the renormalization exponent."""
    fig = plt.figure(figsize=(ps.WIDTH, 2.45))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, .9])
    ax = [fig.add_subplot(gs[0])]
    ax.append(fig.add_subplot(gs[1]))
    ax.append(fig.add_subplot(gs[2]))
    aic_rows, dk_rows, keys = [], [], []
    for label, col, mk, ls, frac in STYLE:
        rec = _pick(data, label)
        if not rec:
            continue
        keys.append(label)
        lb = np.array(rec["l_B"], float)
        NB = np.array(rec["NB"], float) / rec["N"]
        f = rec.get("fractality", {})
        ax[0].plot(lb, NB, marker=mk, ls="none", color=col, ms=3.2)
        c = np.polyfit(np.log(lb), np.log(NB), 1)
        xx = np.linspace(lb.min(), lb.max(), 60)
        ax[0].plot(xx, np.exp(np.polyval(c, np.log(xx))), color=col,
                   lw=1.0 if frac else .8, ls=ls)
        if np.isfinite(f.get("dAIC_pl_minus_exp", float("nan"))):
            aic_rows.append((DISPLAY[label], f["dAIC_pl_minus_exp"], col))
        if np.isfinite(rec.get("dk_renorm", float("nan"))):
            dk_rows.append((DISPLAY[label], rec["dk_renorm"], col))
    ax[0].set(xscale="log", yscale="log", xlabel=r"$\ell_B$",
              ylabel=r"$N_B(\ell_B)/N$", title="(a) box counting")
    ax[0].set_xticks([2, 3, 5, 10, 20])
    ax[0].set_xticklabels(["2", "3", "5", "10", "20"])
    ax[0].minorticks_off()
    _barh(ax[1], aic_rows, r"(b) $\Delta$AIC",
          r"$\Delta$AIC (negative: power law)", fmt="{:+.1f}")
    # Only the SIGN is diagnostic, and only where s(l_B) is itself a power law
    # (Table 9, Delta AIC_s); the caption says so.
    _barh(ax[2], dk_rows, r"(c) sign of $d_k$", r"renormalization $d_k$", yticks=False)
    ax[1].set_xlim(-47, 16)
    ax[2].set_xlim(-4.3, 4.5)
    network_legend(fig, keys)
    fig.tight_layout(pad=.3, w_pad=.5, rect=(0, 0.13, 1, 1))
    ps.save(fig, "fig7_fractality")


def fig_grid_and_size(data):
    """(a) every candidate grid as a (d_B, R^2) point, so the reader sees the whole
    sensitivity rather than the winner alone; (b) the WWW parity effect that makes
    the grid choice matter; (c) d_B against backbone size for DBLP."""
    fig, ax = plt.subplots(1, 3, figsize=(ps.WIDTH, 2.55))
    keys = []
    below = dict(loc="upper center", bbox_to_anchor=(0.5, -0.27), handlelength=1.8)

    # ---- (a) the full grid scan as a cloud, selected grid ringed
    for label, col, mk, ls, frac in STYLE:
        rec = _pick(data, label)
        if not rec or not rec.get("grid_scan"):
            continue
        rows = [r for r in rec["grid_scan"]
                if np.isfinite(r.get("dB_R2", np.nan)) and np.isfinite(r.get("dB", np.nan))]
        if not rows:
            continue
        keys.append(label)
        ax[0].plot([r["dB"] for r in rows], [r["dB_R2"] for r in rows],
                   marker=mk, ls="none", color=col, ms=3.2)
        # Ring the grid used in the tables (the "main" record). On every
        # network but DBLP w>=60 that is also the scan's R^2 maximum; there the
        # lambda = 2.5 grid is higher by 2e-4 (see results/README.md).
        used = [int(round(x)) for x in rec["l_B"]]
        hit = [r for r in rows if [int(x) for x in r["grid"]] == used]
        best = hit[0] if hit else {"dB": rec["dB"], "dB_R2": rec["dB_R2"]}
        ax[0].plot([best["dB"]], [best["dB_R2"]], marker="o", ls="none",
                   mfc="none", mec=col, ms=7.5, mew=1.0)
    ax[0].set(xlabel=r"$d_B$ from that grid",
              ylabel=r"$R^2$ of the $N_B(\ell_B)$ power law",
              title="(a) every candidate grid")
    ax[0].text(0.97, 0.2, "ringed: grid used", transform=ax[0].transAxes,
               ha="right", va="bottom", fontsize=6)

    # ---- (b) why the grid matters: the parity effect, on the WWW
    rec = _pick(data, "WWW")
    if rec and rec.get("per_lB"):
        per = {int(k): v for k, v in rec["per_lB"].items()}
        ls_all = sorted(per)
        N = rec["N"]
        y = np.array([per[l]["NB"] for l in ls_all], float) / N
        x = np.array(ls_all, float)
        ax[1].plot(x, y, marker="o", ls="none", color=OI["blue"], ms=3.8,
                   label=r"WWW, every $\ell_B$")
        for grid, col, lsty, tag in (
                (tuple(l for l in ls_all if l % 2 == 0), OI["blue"], "-",
                 r"even $\ell_B$"),
                (tuple(l for l in ls_all if l % 2 == 1), OI["orange"], "--",
                 r"odd $\ell_B$"),
                (tuple(ls_all), OI["red"], ":", r"all $\ell_B$")):
            if len(grid) < 2:
                continue
            xx = np.array(grid, float)
            yy = np.array([per[l]["NB"] for l in grid], float) / N
            c = np.polyfit(np.log(xx), np.log(yy), 1)
            pr = np.polyval(c, np.log(xx))
            ss = np.sum((np.log(yy) - np.log(yy).mean()) ** 2)
            r2 = 1 - np.sum((np.log(yy) - pr) ** 2) / ss if ss > 0 else np.nan
            fine = np.linspace(min(xx), max(xx), 40)
            ax[1].plot(fine, np.exp(np.polyval(c, np.log(fine))), color=col,
                       ls=lsty, lw=1.0,
                       label=f"{tag}: ${-c[0]:.2f}$, $R^2={r2:.4f}$"
                       if len(grid) > 2 else f"{tag}: ${-c[0]:.2f}$")
            print(f"  WWW {tag}: dB={-c[0]:.3f} R2={r2:.6f}")
        ax[1].set(xscale="log", yscale="log", xlabel=r"$\ell_B$",
                  ylabel=r"$N_B(\ell_B)/N$",
                  title=r"(b) WWW: parity of $\ell_B$")
        ax[1].set_xticks(ls_all)
        ax[1].set_xticklabels([str(l) for l in ls_all])
        ax[1].minorticks_off()
        ax[1].legend(**below)

    # ---- (c) d_B against backbone size, DBLP weak-tie sweep
    sweep = _load("dblp_sweep_canonical.json")
    if sweep:
        rows = sweep if isinstance(sweep, list) else sweep.get("rows", [])
        Ns = np.array([r["N"] for r in rows if "N" in r], float)
        dBs = np.array([r["dB"] for r in rows if "dB" in r], float)
        if Ns.size:
            o = np.argsort(Ns)
            Ns, dBs = Ns[o], dBs[o]
            m, sd = dBs.mean(), dBs.std()
            ax[2].axhspan(m - sd, m + sd, color=ps.tint(OI["green"], .85), lw=0,
                          label=fr"mean $\pm$ std. dev., ${m:.2f}\pm{sd:.2f}$")
            ax[2].axhline(2.0, color="k", lw=.8, ls=":",
                          label=r"Fronczak $et\ al.$")
            ax[2].plot(Ns, dBs, marker="s", ls="-", color=OI["green"], ms=3.2,
                       label=r"DBLP, varying $w$")
            ax[2].set_ylim(1.76, max(2.04, dBs.max() + .05))
            print(f"  DBLP sweep: mean {m:.3f} sd {sd:.3f}, N {Ns.min():.0f}-{Ns.max():.0f}")
    ax[2].set(xscale="log", xlabel="$N$ (backbone)", ylabel="$d_B$",
              title=r"(c) DBLP: $d_B$ against $N$")
    # band = mean +/- one standard deviation, dotted = Fronczak et al.'s 2.0;
    # both are stated in the caption, so no legend is drawn in this narrow panel
    ax[2].text(0.97, 0.03, r"$d_B=%.2f\pm%.2f$" % (m, sd) if sweep else "",
               transform=ax[2].transAxes, ha="right", va="bottom", fontsize=6.5)
    h = [Line2D([], [], color=c, marker=mk, ls="none", ms=3.2, label=DISPLAY[k])
         for k, c, mk, ls, _ in STYLE if k in keys]
    ax[0].legend(handles=h, ncol=2, columnspacing=.6, handletextpad=.2, **below)
    fig.subplots_adjust(left=.1, right=.985, top=.925, bottom=.36, wspace=.52)
    ps.save(fig, "fig8_grid_and_size")


if __name__ == "__main__":
    d = collect()
    print("series found:", ", ".join(sorted(d)) or "(none)")
    fig_fractality(d)
    fig_grid_and_size(d)

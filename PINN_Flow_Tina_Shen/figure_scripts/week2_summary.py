"""Two figures for the 21 September 2026 talk: what else we tried.

Companion to ``week_summary.py``, which covers the empty-layer fix. This one
covers the parameterizations that were tested and rejected, and the physical
validity checks. Numbers are named by the run that produced them; 8000
evaluation showers each.

    python figure_scripts/week2_summary.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

GEANT4 = "#2c3e50"
REJECTED = "#e67e22"
KEPT = "#27ae60"
NEUTRAL = "#7fb3d5"

# ---------------------------------------------------------------- figure 1
# chi2 as a multiple of the Geant4 floor (0.9532), every parameterization tried.
FLOOR_CHI2 = 0.9532
TRIED = [
    ("absolute energies\n(the baseline)", [45.52, 47.66], REJECTED,
     "what we started with"),
    ("E_tot relative\nto E_inc", [46.44], KEPT, "fixes the energy response"),
    ("the empty-layer fix", [9.42, 10.53], KEPT, "what worked"),
]
# Three more were tried and dropped, and are left off the figure to keep it to
# what the talk covers: every energy relative to E_inc (56.02), widths as
# sqrt(x) (23.85) and layer energies as sqrt(E/E_inc) (29.55). DEVLOG 23-24.


def figure_tried(path):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for i, (label, values, color, _) in enumerate(TRIED):
        mean = np.mean(values) / FLOOR_CHI2
        ax.bar(i, mean, color=color, width=0.62)
        if len(values) > 1:
            lo, hi = min(values) / FLOOR_CHI2, max(values) / FLOOR_CHI2
            ax.plot([i, i], [lo, hi], color=GEANT4, lw=2)
            for y in (lo, hi):
                ax.plot([i - 0.1, i + 0.1], [y, y], color=GEANT4, lw=2)
        ax.text(i, max(values) / FLOOR_CHI2 + 1.5, f"{mean:.0f}x", ha="center",
                fontsize=9)
    ax.axhline(1.0, color=GEANT4, ls=":", lw=2)
    ax.text(0.6, 3.5, "Geant4 vs Geant4 = 1x", fontsize=9, color=GEANT4)
    ax.set_xticks(range(len(TRIED)))
    ax.set_xticklabels([t[0] for t in TRIED], fontsize=8)
    ax.set_ylabel("chi2, as a multiple of the Geant4 floor")
    # Green means "in the recipe we use", not "scored best": E_tot-relative is
    # kept for the energy response at no chi2 gain, and sqrt widths scored well
    # but became redundant once empty layers were handled directly.
    ax.set_title("Three ways of writing down the same showers\n"
                 "(lower is better; green is what we kept, orange we dropped)")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, 66)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
# Percent of 8000 generated showers breaking each physical rule
# (python -m pinnde_eval.validate_physical, 2026-09-22). Two configurations,
# both scored with the current checker: the baseline (clip_abs) and the
# recommended setting (fix_s0), which includes the two-way emptiness rule.
# Real Geant4 is 0.00% on every row. The last rule needs --derive-total, which
# takes it to 0% and is discussed in the text rather than shown here.
CHECKS = [
    ("width or radius\nbelow zero", 72.56, 4.79),
    ("energy below\nan empty layer", 49.88, 0.03),
    ("empty layer with\na live centre", 49.88, 0.03),
    ("energy in\nzero voxels", 62.96, 5.90),
    ("sparsity outside\n[0, 1]", 58.25, 6.96),
    ("E_tot is not the\nsum of the layers", 100.0, 100.0),
]


def figure_physical(path):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(CHECKS))
    w = 0.36
    before = [c[1] for c in CHECKS]
    after = [c[2] for c in CHECKS]
    ax.bar(x - w / 2, before, w, color=REJECTED, label="before")
    ax.bar(x + w / 2, after, w, color=KEPT, label="with the fixes")
    for xi, v in zip(x + w / 2, after):
        if v < 1.0:
            ax.text(xi, 2.0, f"{v:.2f}%", ha="center", fontsize=8, color=KEPT)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in CHECKS], fontsize=8.5)
    ax.set_ylabel("generated showers breaking the rule (%)")
    ax.set_title("Physically impossible showers\n"
                 "(real Geant4 breaks none of these: 0% on every bar)")
    ax.legend(frameon=False, fontsize=9, loc="upper center")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, 118)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    figure_tried(os.path.join(HERE, "week2_parameterisations.png"))
    figure_physical(os.path.join(HERE, "week2_physical.png"))
    print("wrote week2_parameterisations.png, week2_physical.png")

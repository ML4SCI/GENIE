"""Three figures for the week of 2026-09-15 to 2026-09-19.

Every number below was measured on the GPU machine (RTX 5090) with
8000 evaluation showers from ds2_2, and is named by the run that produced it so
it can be traced back to the DEVLOG. The numbers are written out here rather
than read from the ``.npz`` files because those files are not in git.

    python figure_scripts/week_summary.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

GEANT4 = "#2c3e50"      # real data, everywhere
BEFORE = "#e67e22"      # the model before this week's fix
AFTER = "#27ae60"       # the model after it

# ---------------------------------------------------------------- figure 1
# Fraction of showers in which each layer holds no energy at all.
# real / baseline: runs/clip_abs.npz, atom: runs/thresh.npz (2026-09-17).
# "No energy" means log10 E_layer <= -8. For the baseline that also counts
# values *below* -8, which are impossible -- the generous reading.
EMPTY_REAL = [0.00013, 0.0005, 0.00025, 0.00038, 0.00075, 0.00112, 0.00125,
              0.001, 0.0025, 0.00325, 0.00675, 0.00887, 0.01087, 0.01738,
              0.02513, 0.03375, 0.04537, 0.05475, 0.06875, 0.08925, 0.11025,
              0.12462, 0.13725, 0.15937, 0.17763, 0.19612, 0.21413, 0.23162,
              0.25075, 0.26175, 0.2795, 0.29725, 0.314, 0.32588, 0.33975,
              0.35387, 0.36688, 0.385, 0.39537, 0.41375, 0.42925, 0.43788,
              0.45463, 0.4775, 0.50562]
EMPTY_BASELINE = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00013, 0.00025, 0.0005,
                  0.0015, 0.0015, 0.00337, 0.007, 0.00637, 0.01337, 0.016,
                  0.02187, 0.02587, 0.034, 0.04075, 0.04375, 0.05563, 0.0435,
                  0.06237, 0.06775, 0.0575, 0.09513, 0.09137, 0.08762,
                  0.09237, 0.12825, 0.12462, 0.12675, 0.13275, 0.14475,
                  0.15375, 0.13537, 0.16225, 0.174, 0.19787, 0.17675,
                  0.20112, 0.21525, 0.23975, 0.21563]
EMPTY_ATOM = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00025, 0.00038, 0.0005, 0.00162,
              0.00287, 0.00462, 0.00988, 0.01438, 0.0195, 0.02813, 0.03862,
              0.04537, 0.06475, 0.076, 0.0895, 0.11125, 0.11075, 0.14688,
              0.156, 0.15812, 0.178, 0.20888, 0.22662, 0.233, 0.255, 0.26375,
              0.29275, 0.296, 0.3135, 0.33, 0.3575, 0.36875, 0.38438, 0.40437,
              0.40663, 0.41587, 0.43438, 0.46487, 0.48425]


def figure_empty_layers(path):
    layers = np.arange(45)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(layers, 100 * np.array(EMPTY_REAL), color=GEANT4, lw=2.5,
            label="Geant4 (the truth)")
    ax.plot(layers, 100 * np.array(EMPTY_BASELINE), color=BEFORE, lw=2,
            ls="--", label="flow matching, before")
    ax.plot(layers, 100 * np.array(EMPTY_ATOM), color=AFTER, lw=2,
            label="flow matching, with --atom-snap")
    ax.set_xlabel("calorimeter layer (0 = front, 44 = back)")
    ax.set_ylabel("showers where this layer is empty (%)")
    ax.set_title("How often a layer holds no energy at all")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 44)
    ax.set_ylim(0, 55)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
# chi2 as a multiple of the Geant4-vs-Geant4 floor (0.9532 at d=362).
# Each bar is one 100k-step run at 1024x8 with --clip-grad 1.0.
CONFIGS = [
    ("before\n(2 seeds)", [45.52, 47.66], BEFORE),
    ("the fix\n(2 seeds)", [9.42, 10.53], AFTER),
    ("the fix, wider\nempty band", [9.38], AFTER),
]
# Left off this figure, which is about the fix that worked: sqrt widths (23.85)
# and layer energies as a sqrt fraction (29.55), both rejected, and
# --derive-total (9.87 / 14.54), which fixes a physical rule without improving
# chi2 and belongs with the physical checks instead. All in DEVLOG 24.
FLOOR_CHI2 = 0.9532


def figure_metrics(path):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, (label, values, color) in enumerate(CONFIGS):
        mean = np.mean(values) / FLOOR_CHI2
        ax.bar(i, mean, color=color, width=0.62)
        if len(values) > 1:                       # show the seed spread
            lo, hi = min(values) / FLOOR_CHI2, max(values) / FLOOR_CHI2
            ax.plot([i, i], [lo, hi], color=GEANT4, lw=2)
            ax.plot([i - 0.1, i + 0.1], [lo, lo], color=GEANT4, lw=2)
            ax.plot([i - 0.1, i + 0.1], [hi, hi], color=GEANT4, lw=2)
        top = max(values) / FLOOR_CHI2           # clear of the spread bar
        ax.text(i, top + 1.6, f"{mean:.0f}x", ha="center", fontsize=9)
    ax.axhline(1.0, color=GEANT4, ls=":", lw=2)
    ax.text(0.55, 3.0, "Geant4 vs Geant4 = 1x", ha="left", fontsize=9,
            color=GEANT4)
    ax.set_xticks(range(len(CONFIGS)))
    ax.set_xticklabels([c[0] for c in CONFIGS], fontsize=8.5)
    ax.set_ylabel("chi2, as a multiple of the Geant4 floor")
    ax.set_title("Agreement with Geant4 across the 362 official features\n"
                 "(lower is better; 1x is as good as real data)")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, 56)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
# Percentage of the 8000 generated showers that break each physical rule
# (python -m pinnde_eval.validate_physical, 2026-09-17). Real Geant4 showers
# score 0.00% on every row, which is why there is no bar for them.
CHECKS = [
    ("width or radius\nbelow zero", 72.56, 4.79, 7.16),
    ("energy below\nan empty layer", 49.88, 0.03, 0.07),
    ("empty layer with\na live centre", 49.88, 0.03, 0.07),
    ("sparsity outside\n[0, 1]", 58.25, 44.38, 42.19),
    ("E_tot is not the\nsum of the layers", 100.0, 100.0, 0.0),
]


def figure_physical(path):
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    x = np.arange(len(CHECKS))
    w = 0.27
    before = [c[1] for c in CHECKS]
    atom = [c[2] for c in CHECKS]
    both = [c[3] for c in CHECKS]
    ax.bar(x - w, before, w, color=BEFORE, label="before this week")
    ax.bar(x, atom, w, color="#7fb3d5", label="--atom-snap")
    ax.bar(x + w, both, w, color=AFTER, label="--atom-snap --derive-total")
    # Bars this small are invisible, so print what they actually are -- 0.07%
    # is not zero, and saying so matters more than a tidy label.
    for xi, v in zip(x + w, both):
        if v < 0.5:
            ax.text(xi, 2.0, f"{v:.2f}%", ha="center", fontsize=8, color=AFTER)
    for xi, v in zip(x, atom):
        if v < 0.5:
            ax.text(xi, 2.0, f"{v:.2f}%", ha="center", fontsize=8,
                    color="#4a86a8")
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in CHECKS], fontsize=8.5)
    ax.set_ylabel("generated showers breaking the rule (%)")
    ax.set_title("Physically impossible showers\n"
                 "(real Geant4 showers break none of these rules: 0% on every bar)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, 108)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    figure_empty_layers(os.path.join(HERE, "week_empty_layers.png"))
    figure_metrics(os.path.join(HERE, "week_metrics.png"))
    figure_physical(os.path.join(HERE, "week_physical.png"))
    print("wrote week_empty_layers.png, week_metrics.png, week_physical.png")

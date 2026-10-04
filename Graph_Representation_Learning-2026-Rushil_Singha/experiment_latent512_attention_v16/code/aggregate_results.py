#!/usr/bin/env python
"""
aggregate_results.py — combine all 6 runs into the paper figures + table
=======================================================================

Reads every results/<tag>/metrics.json and produces, in SAVE_ROOT/_aggregate/:

  summary_table.md        full numbers, markdown, paste straight into the draft
  summary_table.csv       same, for pandas
  fig1_receptive_field.png    long-range error vs receptive field, one line
                              per arm (local / global).  THE MONEY FIGURE.
  fig2_attention_gap.png      gap = local - global, per encoder.
                              "how much of attention does ChebNet replace?"
  fig3_longrange_vs_control.png  per-observable bars; the control (ptD) must
                              stay flat or the result is just "bigger model".
  fig4_per_jet_type.png       top jets expected to show the largest effect.

Per-run figures (real vs reconstruction) live in results/<tag>/ -- see
recon_pairs.png, observables.png, multiplicity.png, dist_comparison.png.

Run:   python aggregate_results.py
"""
import os
import sys
import csv
import glob
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ae_core import (SAVE_ROOT, PAIRWISE_KEYS, MOMENT_KEYS,   # noqa: E402
                     CONTROL_KEYS, LONGRANGE_KEYS)

OUT_DIR = os.path.join(SAVE_ROOT, "_aggregate")
os.makedirs(OUT_DIR, exist_ok=True)

# Derived from the runs actually found on disk, ordered by receptive field.
# Do NOT hardcode encoder names here: if they do not match the tags the run
# scripts produced, every figure silently comes out EMPTY with no error.
ENCODER_ORDER = []
ENCODER_LABEL = {}


def encoder_key(m):
    return "sage" if m["conv"] == "sage" else f"chebK{m['K']}"


def build_encoder_meta(rows):
    """Populate ENCODER_ORDER / ENCODER_LABEL from the loaded runs, so any
    choice of K in the run scripts works with no edit here."""
    global ENCODER_ORDER, ENCODER_LABEL
    hops, labels = {}, {}
    for r in rows:
        k = encoder_key(r)
        hops[k] = r.get("receptive_field_hops", 0)
        labels[k] = ("1-hop\n(GraphSAGE-style)" if r["conv"] == "sage"
                     else f"ChebNet\nK={r['K']}")
    ENCODER_ORDER = sorted(hops, key=lambda k: (hops[k], k))
    ENCODER_LABEL = labels
    print("encoders found (by receptive field): "
          + ", ".join(f"{k}={hops[k]}hops" for k in ENCODER_ORDER))


def load_all():
    rows = []
    for p in sorted(glob.glob(os.path.join(SAVE_ROOT, "*", "metrics.json"))):
        with open(p) as f:
            rows.append(json.load(f))
    return rows


def write_table(rows):
    cols = ["tag", "config", "conv", "K", "layers", "knn",
            "receptive_field_hops", "params_total", "latent_dim",
            "pairwise_relerr_mean", "moment_relerr_mean"] + \
           [f"paired_relerr_{k}" for k in PAIRWISE_KEYS + MOMENT_KEYS + CONTROL_KEYS] + \
           ["emd_mean", "mult_mae", "mult_mae_mean_baseline", "mult_corr",
            "w1_eta", "w1_phi", "w1_pt", "final_train_loss"]

    with open(os.path.join(OUT_DIR, "summary_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    def fmt(v):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)

    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")

    md = os.path.join(OUT_DIR, "summary_table.md")
    with open(md, "w") as f:
        f.write("# Ablation summary\n\n")
        f.write("Lower is better for every error column.\n\n")
        f.write("\n".join(lines) + "\n")
    print(f"wrote {md}")


def fig_receptive_field(rows):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for cfg, color, mk in [("local", "tab:blue", "o"),
                           ("global", "tab:red", "s")]:
        sub = {encoder_key(r): r for r in rows if r["config"] == cfg}
        xs, ys, lbl = [], [], []
        for e in ENCODER_ORDER:
            if e in sub and sub[e].get("pairwise_relerr_mean") is not None:
                xs.append(sub[e]["receptive_field_hops"])
                ys.append(sub[e]["pairwise_relerr_mean"])
                lbl.append(e)
        if xs:
            ax.plot(xs, ys, marker=mk, color=color, lw=2, ms=9,
                    label=f"{cfg}  ({'stats pool, no attn' if cfg=='local' else 'Set2Set + attn'})")
            for x, y, l in zip(xs, ys, lbl):
                ax.annotate(l, (x, y), textcoords="offset points",
                            xytext=(6, 6), fontsize=8, color=color)
    ax.set_xlabel("encoder receptive field (hops)")
    ax.set_ylabel("mean paired relative error\n(pairwise observables only)")
    ax.set_title("Long-range structure preserved vs encoder receptive field")
    ax.grid(alpha=.3); ax.legend()
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "fig1_receptive_field.png")
    plt.savefig(p, dpi=140); plt.close()
    print(f"wrote {p}")


def fig_attention_gap(rows):
    loc = {encoder_key(r): r for r in rows if r["config"] == "local"}
    glo = {encoder_key(r): r for r in rows if r["config"] == "global"}
    enc = [e for e in ENCODER_ORDER if e in loc and e in glo]
    if not enc:
        print("skip fig2: need both arms")
        return

    x = np.arange(len(enc)); w = 0.35
    l = [loc[e]["pairwise_relerr_mean"] for e in enc]
    g = [glo[e]["pairwise_relerr_mean"] for e in enc]
    gap = [a - b for a, b in zip(l, g)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    ax1.bar(x - w / 2, l, w, label="local (no attention)", color="tab:blue")
    ax1.bar(x + w / 2, g, w, label="global (Set2Set + attn)", color="tab:red")
    ax1.set_xticks(x); ax1.set_xticklabels([ENCODER_LABEL[e] for e in enc])
    ax1.set_ylabel("pairwise paired relative error")
    ax1.set_title("Reconstruction error, both arms")
    ax1.grid(axis="y", alpha=.3); ax1.legend()

    cols = ["tab:green" if v >= 0 else "tab:orange" for v in gap]
    ax2.bar(x, gap, 0.5, color=cols)
    ax2.axhline(0, color="k", lw=.8)
    ax2.set_xticks(x); ax2.set_xticklabels([ENCODER_LABEL[e] for e in enc])
    ax2.set_ylabel("gap = local − global")
    ax2.set_title("Attention gap\n(shrinking with K ⇒ ChebNet substitutes "
                  "for attention)")
    ax2.grid(axis="y", alpha=.3)
    for xi, v in zip(x, gap):
        ax2.annotate(f"{v:+.4f}", (xi, v), ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=9)

    plt.tight_layout()
    p = os.path.join(OUT_DIR, "fig2_attention_gap.png")
    plt.savefig(p, dpi=140); plt.close()
    print(f"wrote {p}")


def fig_longrange_vs_control(rows):
    keys = PAIRWISE_KEYS + MOMENT_KEYS + CONTROL_KEYS
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 5),
                             sharex=True)
    if len(keys) == 1:
        axes = [axes]
    for ax, k in zip(axes, keys):
        for cfg, color, mk in [("local", "tab:blue", "o"),
                               ("global", "tab:red", "s")]:
            sub = {encoder_key(r): r for r in rows if r["config"] == cfg}
            xs, ys = [], []
            for i, e in enumerate(ENCODER_ORDER):
                if e in sub and sub[e].get(f"paired_relerr_{k}") is not None:
                    xs.append(i); ys.append(sub[e][f"paired_relerr_{k}"])
            if xs:
                ax.plot(xs, ys, marker=mk, color=color, lw=2, ms=8, label=cfg)
        ax.set_xticks(range(len(ENCODER_ORDER)))
        ax.set_xticklabels([ENCODER_LABEL[e] for e in ENCODER_ORDER],
                           fontsize=8)
        role = ("PAIRWISE" if k in PAIRWISE_KEYS else
                "MOMENT (0 hops)" if k in MOMENT_KEYS else "CONTROL")
        ax.set_title(f"{k}\n[{role}]", fontsize=10)
        ax.grid(alpha=.3)
    axes[0].set_ylabel("paired relative error")
    axes[0].legend(fontsize=8)
    fig.suptitle("Per-observable: the control (ptD) should stay FLAT.\n"
                 "If everything improves equally, it is capacity, not "
                 "receptive field.", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    p = os.path.join(OUT_DIR, "fig3_longrange_vs_control.png")
    plt.savefig(p, dpi=140); plt.close()
    print(f"wrote {p}")


def fig_per_jet_type(rows):
    types = ["gluon", "quark", "top"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, cfg in zip(axes, ["local", "global"]):
        sub = {encoder_key(r): r for r in rows if r["config"] == cfg}
        enc = [e for e in ENCODER_ORDER if e in sub]
        if not enc:
            ax.set_title(f"{cfg} — no runs"); continue
        x = np.arange(len(enc)); w = 0.26
        for j, t in enumerate(types):
            ys = []
            for e in enc:
                pt = sub[e].get("per_jet_type", {}).get(t, {})
                vals = [pt.get(f"paired_relerr_{k}") for k in PAIRWISE_KEYS]
                vals = [v for v in vals if v is not None]
                ys.append(np.mean(vals) if vals else np.nan)
            ax.bar(x + (j - 1) * w, ys, w, label=t)
        ax.set_xticks(x)
        ax.set_xticklabels([ENCODER_LABEL[e] for e in enc], fontsize=8)
        ax.set_title(f"{cfg} arm"); ax.grid(axis="y", alpha=.3)
    axes[0].set_ylabel("pairwise paired relative error")
    axes[0].legend()
    fig.suptitle("Per jet type — top jets (3-prong W decay) carry the most "
                 "long-range correlation", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(OUT_DIR, "fig4_per_jet_type.png")
    plt.savefig(p, dpi=140); plt.close()
    print(f"wrote {p}")


def print_data_report(rows):
    print("\n" + "=" * 78)
    print("DATA / TRAINING REPORT")
    print("=" * 78)
    r0 = rows[0]
    print(f"  eval split           : {r0.get('eval_split')}")
    print(f"  train jets           : {r0.get('n_train'):,}  "
          f"{r0.get('n_train_per_type')}")
    print(f"  test jets            : {r0.get('n_test'):,}  "
          f"{r0.get('n_test_per_type')}")
    print(f"  particles per jet    : {r0.get('total_particles')}")
    print(f"  kNN k                : {r0.get('knn')}   conv layers: "
          f"{r0.get('layers')}")
    print(f"  graph median diameter: "
          f"{(r0.get('graph_stats') or {}).get('median_diameter')}")
    print()
    hdr = (f"  {'tag':44s} {'epochs':>7s} {'hops':>5s} {'params':>11s} "
           f"{'train loss':>11s}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        print(f"  {r['tag']:44s} {r.get('epochs', 0):7d} "
              f"{r.get('receptive_field_hops', 0):5d} "
              f"{r.get('params_total', 0):11,d} "
              f"{r.get('final_train_loss') or float('nan'):11.5f}")
    print("=" * 78)


def main():
    rows = load_all()
    if not rows:
        print(f"No metrics.json found under {SAVE_ROOT}. Run the two arms first.")
        return
    print(f"loaded {len(rows)} runs: {[r['tag'] for r in rows]}")
    build_encoder_meta(rows)
    print_data_report(rows)
    write_table(rows)
    fig_receptive_field(rows)
    fig_attention_gap(rows)
    fig_longrange_vs_control(rows)
    fig_per_jet_type(rows)
    print(f"\nAll aggregate outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()

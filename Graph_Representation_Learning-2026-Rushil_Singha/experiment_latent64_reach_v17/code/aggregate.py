#!/usr/bin/env python
"""
aggregate.py — the paper's tables and figures, from suite.json files.
====================================================================

    python aggregate.py \\
        --pooling     $SCRATCH/gsoc_rushil/v16_ablation/results \\
        --compression $SCRATCH/gsoc_rushil/v16_ablation/results_scout \\
        --reach       $SCRATCH/gsoc_rushil/v17_final/results \\
        --out         $SCRATCH/gsoc_rushil/v17_final/paper

Any of the three can be left out.  Every number comes from a run's
suite.json (written by v17 training, or by reeval.py for older runs).

REACH (headline)   latent 64, 3 seeds: GraphSAGE (2 hops), ChebNet K=2 (2),
                   K=4 (6) and K=6 (10 hops).  Does seeing further help,
                   once the operator is held fixed?          -> --reach
ATTENTION @ 64     same folder: GraphSAGE + StatsPool vs + Set2Set, seeded.
ATTENTION @ 512    round 1, one seed: StatsPool vs Set2Set.  -> --pooling
COMPRESSION        GraphSAGE at latent 512 / 128 / 64 scouts. -> --compression

Outputs in --out:  results.md, and fig_pooling / fig_compression /
fig_reach / fig_reach_by_size / fig_reach_curve / fig_attention64 .png
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import suite as S  # noqa: E402

HEADLINE = S.PRIMARY + ["primary_mean"] + S.CONTROL + ["emd"] + \
           S.SUPPORT + ["support_mean", "mult_mae"]
COL = {"primary_mean": "**primary (mean)**", "support_mean": "support (mean)",
       "mult_mae": "count MAE", **S.LABELS}


def _group_means(d):
    """Recompute the group means from the CURRENT suite groups, so a suite.json
    scored under an older grouping (e.g. with D2 in PRIMARY) is read
    consistently."""
    for name, keys in [("primary_mean", S.PRIMARY),
                       ("support_mean", S.SUPPORT)]:
        v = [d.get(k) for k in keys]
        if all(x is not None for x in v):
            d[name] = float(np.mean(v))


def load(folder):
    rows = []
    for p in sorted(glob.glob(os.path.join(folder, "*", "suite.json"))):
        r = json.load(open(p))
        _group_means(r)
        for sub in list(r.get("split_by_size", {}).values()) + \
                list(r.get("per_jet_type", {}).values()):
            _group_means(sub)
        rows.append(r)
    return rows


def model_name(r):
    base = "GraphSAGE" if r["conv"] == "sage" else f"ChebNet K={r['K']}"
    return base + (" + Set2Set" if r.get("pool") == "set2set" else "")


def fmt(v, d=4):
    return "—" if v is None else f"{v:.{d}f}"


def pct(new, ref):
    if new is None or ref in (None, 0):
        return "—"
    return f"{100 * (new - ref) / ref:+.1f}%"


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — POOLING
# ══════════════════════════════════════════════════════════════════════════
def exp_pooling(rows, out, md):
    rows = [r for r in rows if r["conv"] == "sage" and r["latent_dim"] == 512]
    st = next((r for r in rows if r["pool"] == "stats"), None)
    s2 = next((r for r in rows if r["pool"] == "set2set"), None)
    if not (st and s2):
        md.append("## Attention at latent 512\n\n_Missing a StatsPool or "
                  "Set2Set GraphSAGE run at latent 512._\n")
        return
    same = (st.get("epochs") == s2.get("epochs")
            and st.get("n_train") == s2.get("n_train")
            and st.get("use_attn") == s2.get("use_attn"))
    md += ["## Attention at latent 512 (round 1, one seed)",
           "",
           "GraphSAGE encoder, latent 512, identical decoder and training. "
           "Only the pooling differs. Paired relative error, lower is better.",
           "",
           f"Parameters: StatsPool {st.get('params_total'):,} · "
           f"Set2Set {s2.get('params_total'):,} "
           f"({pct(s2.get('params_total'), st.get('params_total'))}). "
           f"Training budget identical: {'yes' if same else '**NO — check**'}.",
           "",
           "| metric | StatsPool | Set2Set | Set2Set vs StatsPool |",
           "|---|---|---|---|"]
    for k in HEADLINE:
        md.append(f"| {COL.get(k, k)} | {fmt(st.get(k))} | {fmt(s2.get(k))} "
                  f"| {pct(s2.get(k), st.get(k))} |")
    md += ["",
           f"Exact particle count: StatsPool {100*st['mult_exact_frac']:.1f}% "
           f"of jets, Set2Set {100*s2['mult_exact_frac']:.1f}%.",
           "",
           "_One seed per arm, directed graph. The seeded comparison is "
           "'Attention at latent 64' below._", ""]

    keys = S.PRIMARY + S.CONTROL + ["emd"] + S.SUPPORT
    x = np.arange(len(keys)); w = 0.38
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.bar(x - w/2, [st.get(k) or 0 for k in keys], w, label="StatsPool",
           color="#1B7C77")
    ax.bar(x + w/2, [s2.get(k) or 0 for k in keys], w, label="Set2Set",
           color="#8F6410")
    ax.set_xticks(x); ax.set_xticklabels([S.LABELS.get(k, k) for k in keys],
                                         rotation=20, ha="right")
    ax.set_ylabel("paired relative error  (lower = better)")
    ax.set_title("Attention at latent 512 (GraphSAGE, one seed)")
    ax.grid(axis="y", alpha=.3); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out, "fig_pooling.png"), dpi=140)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — COMPRESSION
# ══════════════════════════════════════════════════════════════════════════
def exp_compression(rows, out, md):
    rows = sorted([r for r in rows if r["conv"] == "sage"
                   and r["pool"] == "stats"],
                  key=lambda r: -r["latent_dim"])
    if len(rows) < 2:
        md.append("## Compression\n\n_Need GraphSAGE runs at "
                  "two or more latent sizes._\n")
        return
    md += ["## Compression: when does the bottleneck bind?",
           "",
           "GraphSAGE + StatsPool. A typical jet is ~56 particles x 3 = ~169 "
           "numbers. Paired relative error, lower is better.",
           "",
           "| latent | numbers per jet ÷ latent | " +
           " | ".join(COL.get(k, k) for k in HEADLINE) + " |",
           "|" + "---|" * (len(HEADLINE) + 2)]
    for r in rows:
        md.append(f"| {r['latent_dim']} | {169 / r['latent_dim']:.1f}x | " +
                  " | ".join(fmt(r.get(k)) for k in HEADLINE) + " |")
    md += ["", f"_Trained for {rows[0].get('epochs')} epochs on "
           f"{rows[0].get('n_train')} jets — compare these rows with each "
           f"other, not with the latent-64 experiments._", ""]

    lat = [r["latent_dim"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for k, c, ls in [("primary_mean", "#1B7C77", "-"),
                     ("support_mean", "#6A4FC8", "--"),
                     ("ptD", "#8F6410", ":")]:
        ax.plot(lat, [r.get(k) for r in rows], ls, marker="o", color=c,
                label=COL.get(k, k).strip("*"))
    ax.set_xscale("log", base=2); ax.set_xticks(lat)
    ax.set_xticklabels([str(l) for l in lat]); ax.invert_xaxis()
    ax.set_xlabel("latent size  (smaller = more compression)")
    ax.set_ylabel("paired relative error")
    ax.set_title("Compression (GraphSAGE)")
    ax.grid(alpha=.3); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_compression.png"), dpi=140); plt.close()


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — REACH  (the seeded one)
# ══════════════════════════════════════════════════════════════════════════
ORDER = ["GraphSAGE", "ChebNet K=2", "ChebNet K=4", "ChebNet K=6"]
COLOURS = {"GraphSAGE": "#8a8797", "ChebNet K=2": "#6A4FC8",
           "ChebNet K=4": "#3B82B0", "ChebNet K=6": "#1B7C77",
           "GraphSAGE + Set2Set": "#8F6410"}


def group(rows):
    g = {}
    for r in rows:
        g.setdefault(model_name(r), []).append(r)
    return g


def ms(vals):
    v = np.array([x for x in vals if x is not None], dtype=float)
    if len(v) == 0:
        return None, None, 0
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v)


def contrast(ga, gb, key):
    """Mean difference a - b, its Welch t-test p-value, and n per group."""
    a = np.array([r[key] for r in ga if r.get(key) is not None], float)
    b = np.array([r[key] for r in gb if r.get(key) is not None], float)
    if len(a) < 2 or len(b) < 2:
        return None, None
    return float(a.mean() - b.mean()), float(stats.ttest_ind(
        a, b, equal_var=False).pvalue)


def exp_reach(rows, out, md):
    rows = [r for r in rows if r["latent_dim"] == 64 and r["pool"] == "stats"]
    g = group(rows)
    have = [m for m in ORDER if m in g]
    if len(have) < 2:
        md.append("## Reach\n\n_Not enough latent-64 runs "
                  "yet._\n")
        return

    md += ["## Reach: does seeing further help?",
           "",
           "Latent 64, StatsPool, identical decoder and training. "
           "Mean ± standard deviation over seeds. Lower is better.",
           "",
           "| model | reach | seeds | params | " +
           " | ".join(COL.get(k, k) for k in HEADLINE) + " | train loss |",
           "|" + "---|" * (len(HEADLINE) + 5)]
    for m in have:
        rs = g[m]
        cells = []
        for k in HEADLINE + ["final_train_loss"]:
            mu, sd, n = ms([r.get(k) for r in rs])
            cells.append("—" if mu is None else f"{mu:.4f} ± {sd:.4f}")
        md.append(f"| {m} | {rs[0].get('receptive_field_hops')} hops | "
                  f"{len(rs)} | {rs[0].get('params_total'):,} | " +
                  " | ".join(cells) + " |")

    md += ["", "### The comparisons that matter", "",
           "Difference in paired relative error, **positive = the second "
           "model is better**. p from a Welch t-test across seeds — with 3 "
           "seeds per model, treat p < 0.05 as suggestive, not conclusive.",
           "",
           "| comparison | isolates | primary Δ (p) | support Δ (p) | "
           "ptD control Δ (p) | reading |",
           "|---|---|---|---|---|---|"]
    pairs = [("GraphSAGE", "ChebNet K=2", "the operator (reach held at 2 hops)"),
             ("ChebNet K=2", "ChebNet K=4", "reach 2 → 6 hops"),
             ("ChebNet K=4", "ChebNet K=6", "reach 6 → 10 hops"),
             ("ChebNet K=2", "ChebNet K=6", "**reach 2 → 10 hops** (operator held fixed)"),
             ("GraphSAGE", "ChebNet K=6", "operator + reach together")]
    for a, b, what in pairs:
        if a not in g or b not in g:
            continue
        cells, verdict = [], []
        res = {}
        for k in ["primary_mean", "support_mean", "ptD"]:
            d, p = contrast(g[a], g[b], k)
            res[k] = (d, p)
            cells.append("—" if d is None else f"{d:+.4f} ({p:.3f})")
        dp, pp = res["primary_mean"]
        ds, _ = res["support_mean"]
        dc, pc = res["ptD"]
        if dp is None:
            reading = "—"
        elif pp is not None and pp < 0.05 and abs(dp) > 2 * abs(ds or 0) \
                and (pc is None or pc >= 0.05):
            reading = ("second better on PRIMARY only — a reach-type effect"
                       if dp > 0 else "first better on PRIMARY only")
        elif pp is not None and pp < 0.05:
            reading = "separates, but support/control move too — not clean"
        else:
            reading = "no separation beyond seed noise"
        md.append(f"| {a} → {b} | {what} | " + " | ".join(cells) +
                  f" | {reading} |")
    md += ["",
           "**How to read this.** A reach effect should show up in the "
           "*primary* column and **not** in the support column (those "
           "observables need no reach) or the control column (no geometry at "
           "all). If all three move together, the models differ in how well "
           "they trained, not in what they capture.", ""]

    # size split
    md += ["### Split by jet size", "",
           "A compression-driven effect should concentrate in jets carrying "
           "more numbers than the latent holds "
           f"(> {64/3:.0f} particles at latent 64).", "",
           "| model | primary, small jets | primary, large jets |",
           "|---|---|---|"]
    for m in have:
        sm = ms([r["split_by_size"]["small"].get("primary_mean")
                 for r in g[m] if "split_by_size" in r])
        lg = ms([r["split_by_size"]["large"].get("primary_mean")
                 for r in g[m] if "split_by_size" in r])
        md.append(f"| {m} | {fmt(sm[0])} ± {fmt(sm[1])} | "
                  f"{fmt(lg[0])} ± {fmt(lg[1])} |")
    md += ["", "### Per jet type (primary mean)", "",
           "Top jets carry three-prong structure — where reach should matter "
           "most, if anywhere. Each column is normalised by that jet type's "
           "own spread, so **compare models down a column, never across "
           "columns**.", "",
           "| model | gluon | quark | top |", "|---|---|---|---|"]
    for m in have:
        cells = []
        for t in ["gluon", "quark", "top"]:
            mu, sd, _ = ms([r.get("per_jet_type", {}).get(t, {})
                            .get("primary_mean") for r in g[m]])
            cells.append("—" if mu is None else f"{mu:.4f} ± {sd:.4f}")
        md.append(f"| {m} | " + " | ".join(cells) + " |")
    md.append("")

    # figure: primary / support / control with error bars
    groups = [("primary_mean", "primary\n(needs pairs)"),
              ("support_mean", "support\n(zero reach suffices)"),
              ("ptD", "ptD\n(control)")]
    colours = COLOURS
    x = np.arange(len(groups)); w = 0.8 / len(have)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(have):
        mu = [ms([r.get(k) for r in g[m]])[0] or 0 for k, _ in groups]
        sd = [ms([r.get(k) for r in g[m]])[1] or 0 for k, _ in groups]
        ax.bar(x + (i - (len(have) - 1) / 2) * w, mu, w, yerr=sd, capsize=4,
               label=f"{m}  ({g[m][0].get('receptive_field_hops')} hops)",
               color=colours.get(m))
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in groups])
    ax.set_ylabel("paired relative error  (lower = better)")
    ax.set_title("Reach at latent 64 (mean ± sd over seeds)")
    ax.grid(axis="y", alpha=.3); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out, "fig_reach.png"), dpi=140)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, m in enumerate(have):
        vals = [ms([r["split_by_size"][s].get("primary_mean") for r in g[m]
                    if "split_by_size" in r]) for s in ["small", "large"]]
        ax.errorbar([0 + i * 0.06, 1 + i * 0.06], [v[0] for v in vals],
                    yerr=[v[1] for v in vals], marker="o", capsize=4,
                    color=colours.get(m), label=m)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"small jets (≤{64/3:.0f} particles)",
                        f"large jets (>{64/3:.0f} particles)"])
    ax.set_ylabel("primary paired relative error")
    ax.set_title("Reach — does the gap live in the big jets?")
    ax.grid(alpha=.3); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_reach_by_size.png"), dpi=140); plt.close()
    fig_reach_curve(g, out)


def fig_reach_curve(g, out):
    """Error against receptive field for the ChebNet family: the dose-response
    view of reach.  GraphSAGE (2 hops) is drawn as a separate marker."""
    cheb = [(m, g[m]) for m in ORDER if m.startswith("ChebNet") and m in g]
    if len(cheb) < 2:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for k, c, ls, lab in [("primary_mean", "#1B7C77", "-",
                           "pairwise (needs pairs)"),
                          ("support_mean", "#6A4FC8", "--",
                           "per-particle (zero reach suffices)")]:
        hops = [rs[0].get("receptive_field_hops") for _, rs in cheb]
        mu = [ms([r.get(k) for r in rs])[0] for _, rs in cheb]
        sd = [ms([r.get(k) for r in rs])[1] for _, rs in cheb]
        ax.errorbar(hops, mu, yerr=sd, ls=ls, marker="o", capsize=4, color=c,
                    label=f"ChebNet — {lab}")
        if "GraphSAGE" in g:
            smu, ssd, _ = ms([r.get(k) for r in g["GraphSAGE"]])
            ax.errorbar([2.25], [smu], yerr=[ssd], marker="s", capsize=4,
                        color=c, alpha=.55, ls="none",
                        label=f"GraphSAGE — {lab}")
    ax.set_xticks(hops)
    ax.set_xticklabels([f"{h} hops\n({m})" for h, (m, _) in zip(hops, cheb)])
    ax.set_ylabel("paired relative error  (lower = better)")
    ax.set_title("Reach at latent 64 — error vs receptive field")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_reach_curve.png"), dpi=140)
    plt.close()


def exp_attention64(rows, out, md):
    """StatsPool vs Set2Set on a GraphSAGE encoder at latent 64, with seeds --
    the same setup as the reach experiment, so the two are comparable."""
    rows = [r for r in rows if r["latent_dim"] == 64 and r["conv"] == "sage"]
    st = [r for r in rows if r["pool"] == "stats"]
    s2 = [r for r in rows if r["pool"] == "set2set"]
    if not (st and s2):
        md.append("## Attention at latent 64\n\n_Needs GraphSAGE runs with "
                  "both StatsPool and Set2Set at latent 64._\n")
        return
    keys = HEADLINE + ["mult_exact_frac", "final_train_loss"]
    md += ["## Attention at latent 64 (seeded)", "",
           "GraphSAGE encoder, latent 64, undirected graph, 200 epochs — the "
           "reach experiment's setup. Only the pooling differs. Mean ± sd "
           "over seeds.", "",
           f"Parameters: StatsPool {st[0].get('params_total'):,} · Set2Set "
           f"{s2[0].get('params_total'):,} "
           f"({pct(s2[0].get('params_total'), st[0].get('params_total'))}). "
           "Set2Set's LSTM does not shrink with the latent, so at latent 64 "
           "the attention arm has MORE capacity.", "",
           "| metric | StatsPool | Set2Set | Δ = Set2Set − StatsPool (p) |",
           "|---|---|---|---|"]
    lab = dict(COL, mult_exact_frac="exact count (fraction)",
               final_train_loss="train loss")
    for k in keys:
        a = ms([r.get(k) for r in st]); b = ms([r.get(k) for r in s2])
        d, p = contrast(st, s2, k)
        cell = "—" if d is None else f"{-d:+.4f} ({p:.3f})"
        md.append(f"| {lab.get(k, k)} | "
                  f"{'—' if a[0] is None else f'{a[0]:.4f} ± {a[1]:.4f}'} | "
                  f"{'—' if b[0] is None else f'{b[0]:.4f} ± {b[1]:.4f}'} | "
                  f"{cell} |")
    md += ["", "_For errors, Δ > 0 means Set2Set is worse. For exact count, "
           "Δ > 0 means Set2Set counts better._", ""]

    groups = [("primary_mean", "pairwise"), ("support_mean", "per-particle"),
              ("ptD", "ptD (control)")]
    x = np.arange(len(groups)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, (rs, name) in enumerate([(st, "StatsPool"), (s2, "Set2Set")]):
        mu = [ms([r.get(k) for r in rs])[0] or 0 for k, _ in groups]
        sd = [ms([r.get(k) for r in rs])[1] or 0 for k, _ in groups]
        ax.bar(x + (i - .5) * w, mu, w, yerr=sd, capsize=4, label=name,
               color=["#1B7C77", "#8F6410"][i])
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in groups])
    ax.set_ylabel("paired relative error  (lower = better)")
    ax.set_title("Attention at latent 64 (GraphSAGE, mean ± sd over seeds)")
    ax.grid(axis="y", alpha=.3); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_attention64.png"), dpi=140)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooling")
    ap.add_argument("--compression")
    ap.add_argument("--reach")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    md = ["# Results — what a compressed jet keeps", "",
          "All metrics: paired relative error on the JetNet test split "
          "(|recon − real| averaged over jets, divided by the real spread). "
          "**Primary** observables need pairs of particles; **support** "
          "observables factorise into per-particle sums and need no reach; "
          "**ptD** is the control and uses no geometry.", ""]
    if a.pooling:
        exp_pooling(load(a.pooling), a.out, md)
    if a.compression:
        exp_compression(load(a.compression), a.out, md)
    if a.reach:
        rows = load(a.reach)
        exp_reach(rows, a.out, md)
        exp_attention64(rows, a.out, md)

    path = os.path.join(a.out, "results.md")
    with open(path, "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwritten -> {a.out}")


if __name__ == "__main__":
    main()

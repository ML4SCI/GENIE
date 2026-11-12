"""
build_comparison_plots.py
==========================
Compares our FactorizedCaloAE against the 18 officially published Dataset-2
submissions to the Fast Calorimeter Simulation Challenge 2022.

Reference numbers for the other submissions come straight from the official,
organizer-published results table:
    CaloChallenge/CaloChallenge/final_results/CaloChallenge_results_ds2.hdf
(the same file behind the figures in the challenge write-up -- not hand-copied
from the paper). Our own numbers are parsed live from whatever is currently in
evaluation_results_ds2/, so this script can be re-run at any point while
long-running evaluate.py jobs (cls-high, cls-resnet, prdc) are still filling
in results -- metrics that haven't finished yet show up as missing/"pending"
rather than fabricated.

Usage:
    python build_comparison_plots.py
        --eval-dir   evaluation_results_ds2
        --official   CaloChallenge/final_results/CaloChallenge_results_ds2.hdf
        --outdir     ../docs/figures   (where the committed figures live)
        --our-name   "Our AE (FactorizedCaloAE)"
"""

import os
import re
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUR_METRIC_COLS = [
    'KPD * 1e3', 'KPD * 1e3 std', 'FPD * 1e3', 'FPD * 1e3 std',
    'precision', 'recall', 'density', 'coverage',
    'binary cls low', 'binary cls high', 'binary cls cnn',
]


def parse_fpd_kpd(path):
    if not os.path.exists(path):
        return {}
    text = open(path).read()
    out = {}
    m = re.search(r'FPD \(x10\^3\):\s*([\d.eE+-]+)\s*±\s*([\d.eE+-]+)', text)
    if m:
        out['FPD * 1e3'], out['FPD * 1e3 std'] = float(m.group(1)), float(m.group(2))
    m = re.search(r'KPD \(x10\^3\):\s*([\d.eE+-]+)\s*±\s*([\d.eE+-]+)', text)
    if m:
        out['KPD * 1e3'], out['KPD * 1e3 std'] = float(m.group(1)), float(m.group(2))
    return out


def parse_prdc(path):
    if not os.path.exists(path):
        return {}
    text = open(path).read()
    out = {}
    for key in ['precision', 'recall', 'density', 'coverage']:
        m = re.search(rf'{key}:\s*([\d.eE+-]+)', text)
        if m:
            out[key] = float(m.group(1))
    return out


def parse_classifier(path):
    """Returns (auc, jsd) or None if the run hasn't finished."""
    if not os.path.exists(path):
        return None
    text = open(path).read()
    m = re.search(
        r'Final result of classifier test \(AUC / JSD\):\s*\n?\s*'
        r'([\d.eE+-]+)\s*/\s*([\d.eE+-]+)', text)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def collect_our_results(eval_dir, dataset='2'):
    row = {}
    row.update(parse_fpd_kpd(os.path.join(eval_dir, f'fpd_kpd_{dataset}.txt')))
    row.update(parse_prdc(os.path.join(eval_dir, f'prdc_{dataset}.txt')))

    cls_low = parse_classifier(os.path.join(eval_dir, f'classifier_all_{dataset}.txt'))
    if cls_low is None:
        cls_low = parse_classifier(os.path.join(eval_dir, f'classifier_cls-low_{dataset}.txt'))
    cls_high = parse_classifier(os.path.join(eval_dir, f'classifier_cls-high_{dataset}.txt'))
    cls_cnn = parse_classifier(os.path.join(eval_dir, f'classifier_cls-resnet_{dataset}.txt'))

    if cls_low is not None:
        row['binary cls low'] = cls_low[0]
    if cls_high is not None:
        row['binary cls high'] = cls_high[0]
    if cls_cnn is not None:
        row['binary cls cnn'] = cls_cnn[0]

    missing = [c for c in OUR_METRIC_COLS if c not in row]
    return row, missing


def load_official(path):
    df = pd.read_hdf(path, key='ds2')
    return df.reset_index(drop=True)


OUR_PALETTE = ['crimson', 'darkorange', 'mediumorchid', 'teal', 'saddlebrown']


def _color_map(our_names):
    """submission name -> color. GEANT4 is black, each of our own models gets
    its own color from OUR_PALETTE (cycled), everything else is steelblue."""
    palette = {name: OUR_PALETTE[i % len(OUR_PALETTE)] for i, name in enumerate(our_names)}

    def color_for(name):
        if name in palette:
            return palette[name]
        if name == 'GEANT4':
            return 'black'
        return 'steelblue'
    return color_for


def make_plots(df, our_names, outdir):
    os.makedirs(outdir, exist_ok=True)
    color_for = _color_map(our_names)

    # ---------------- FPD / KPD ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, col, std_col, title in [
        (axes[0], 'FPD * 1e3', 'FPD * 1e3 std', 'Frechet Physics Distance (x1e3, lower=better)'),
        (axes[1], 'KPD * 1e3', 'KPD * 1e3 std', 'Kernel Physics Distance (x1e3, lower=better)'),
    ]:
        sub = df.dropna(subset=[col]).sort_values(col)
        colors = [color_for(n) for n in sub['submission name']]
        ax.barh(sub['submission name'], sub[col],
                xerr=sub[std_col] if std_col in sub else None, color=colors)
        ax.set_xlabel(title)
        ax.set_xscale('log')
        ax.grid(alpha=0.3, axis='x')
    fig.suptitle('FPD / KPD vs. official Dataset-2 submissions', fontsize=13)
    fig.savefig(os.path.join(outdir, 'comparison_FPD_KPD.png'), dpi=140, bbox_inches='tight')
    plt.close(fig)

    # ---------------- Classifier AUC ----------------
    auc_cols = ['binary cls low', 'binary cls high', 'binary cls cnn']
    present = [c for c in auc_cols if df[c].notna().any()]
    if present:
        fig, axes = plt.subplots(1, len(present), figsize=(5.5 * len(present), 5.5),
                                 constrained_layout=True)
        axes = np.atleast_1d(axes)
        for ax, col in zip(axes, present):
            sub = df.dropna(subset=[col]).sort_values(col)
            colors = [color_for(n) for n in sub['submission name']]
            ax.barh(sub['submission name'], sub[col], color=colors)
            ax.axvline(0.5, color='gray', ls='--', lw=1, label='chance (0.5)')
            ax.set_xlim(0.45, 1.02)
            ax.set_xlabel(col + ' AUC (closer to 0.5 = better)')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3, axis='x')
        fig.suptitle('Classifier AUC vs. official Dataset-2 submissions', fontsize=13)
        fig.savefig(os.path.join(outdir, 'comparison_AUC.png'), dpi=140, bbox_inches='tight')
        plt.close(fig)

    # ---------------- PRDC ----------------
    prdc_cols = ['precision', 'recall', 'density', 'coverage']
    if df[prdc_cols].notna().any().any():
        fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
        for ax, col in zip(axes.ravel(), prdc_cols):
            sub = df.dropna(subset=[col]).sort_values(col)
            colors = [color_for(n) for n in sub['submission name']]
            ax.barh(sub['submission name'], sub[col], color=colors)
            ax.set_xlabel(col)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.3, axis='x')
        fig.suptitle('Precision / Recall / Density / Coverage vs. official submissions', fontsize=13)
        fig.savefig(os.path.join(outdir, 'comparison_PRDC.png'), dpi=140, bbox_inches='tight')
        plt.close(fig)

    # ---------------- Pareto: FPD vs binary cls low ----------------
    if df['binary cls low'].notna().any():
        sub = df.dropna(subset=['FPD * 1e3', 'binary cls low'])
        fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
        for _, r in sub.iterrows():
            color = color_for(r['submission name'])
            ax.scatter(r['FPD * 1e3'], r['binary cls low'], color=color, s=60, zorder=3)
            ax.annotate(r['submission name'], (r['FPD * 1e3'], r['binary cls low']),
                        fontsize=7, xytext=(4, 3), textcoords='offset points')
        ax.set_xscale('log')
        ax.set_xlabel('FPD x1e3 (lower = better)')
        ax.set_ylabel('binary cls low AUC (closer to 0.5 = better)')
        ax.set_title('Fidelity trade-off: FPD vs. classifier AUC')
        ax.grid(alpha=0.3)
        fig.savefig(os.path.join(outdir, 'comparison_pareto_AUC_FPD.png'), dpi=140, bbox_inches='tight')
        plt.close(fig)

    # ---------------- Summary table ----------------
    cols = ['submission name'] + OUR_METRIC_COLS
    table_df = df[cols].copy()
    for c in OUR_METRIC_COLS:
        if 'std' in c:
            continue
        table_df[c] = table_df[c].map(lambda v: f'{v:.4g}' if pd.notna(v) else '—')
    table_df = table_df.drop(columns=[c for c in table_df.columns if c.endswith('std')])

    fig, ax = plt.subplots(figsize=(18, 0.32 * len(table_df) + 1.0))
    ax.axis('off')
    tbl = ax.table(cellText=table_df.values, colLabels=table_df.columns,
                   cellLoc='center', loc='upper center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(col=list(range(len(table_df.columns))))
    tbl.scale(1, 1.6)
    our_names_set = set(our_names)
    for j, name in enumerate(table_df['submission name']):
        if name in our_names_set:
            for k in range(len(table_df.columns)):
                tbl[j + 1, k].set_facecolor('#ffdede')
    fig.suptitle('Dataset 2 comparison summary (— = not yet computed)', fontsize=13)
    fig.savefig(os.path.join(outdir, 'comparison_summary_table.png'), dpi=140, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--eval-dir', default='evaluation_results_ds2',
                    help="Used together with --our-name for a single model "
                         "(back-compat shorthand for one --model entry).")
    p.add_argument('--official', default='CaloChallenge/final_results/CaloChallenge_results_ds2.hdf')
    p.add_argument('--outdir', default='../docs/figures')
    p.add_argument('--our-name', default='Our AE (FactorizedCaloAE)')
    p.add_argument('--model', action='append', default=[],
                    metavar='NAME:EVAL_DIR',
                    help="Add another one of our own models to the comparison, "
                         "e.g. --model 'Sweep d96 (460ep):Hyper_paramater_tuning/"
                         "latent_sweep_runs/lat_c8_l12_d96/eval_results'. "
                         "Repeatable.")
    args = p.parse_args()

    # --model entries plus the --eval-dir/--our-name shorthand, in order.
    model_specs = [(args.our_name, args.eval_dir)]
    for spec in args.model:
        name, _, eval_dir = spec.partition(':')
        if not eval_dir:
            raise SystemExit(f"--model expects NAME:EVAL_DIR, got: {spec!r}")
        model_specs.append((name, eval_dir))

    df = load_official(args.official)
    df = df[['submission name'] + OUR_METRIC_COLS]

    our_names = []
    for name, eval_dir in model_specs:
        row, missing = collect_our_results(eval_dir)
        print(f'{name} ({eval_dir}) results collected:', row)
        if missing:
            print(f'  still pending for {name}:', missing)
        full_row = {'submission name': name}
        full_row.update(row)
        df = pd.concat([df, pd.DataFrame([full_row])], ignore_index=True)
        our_names.append(name)

    make_plots(df, our_names, args.outdir)
    print(f'Wrote comparison plots to {args.outdir}/')


if __name__ == '__main__':
    main()

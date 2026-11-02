#!/usr/bin/env python3
"""
make_tables.py -- regenerate the results tables from the committed text files.

Every number in docs/ should be traceable to a file under results/. This script
is what makes that true: it parses the official evaluate.py outputs and emits
summary.csv plus a markdown table, so a number in the README cannot silently
drift from the evaluation that produced it.

That has already gone wrong twice in this project: an f_scale value measured on
a 20k subset instead of the full 100k, and an autoencoder row attributed to the
wrong checkpoint. Both would have been caught by generating the table instead of
typing it.

    python results/make_tables.py            # print the table
    python results/make_tables.py --csv      # also write results/summary.csv
"""

import argparse
import csv
import os
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))

# display order: best first, then by model family
ORDER = [
    "two_stage_100k",
    "conv_unet_runA",
    "conv_unet_runB",
    "film_mlp_100k",
    "autoencoder_prev4_ep85",
    "autoencoder_sweep_d96_ep460",
    "latent_diffusion",
]

CHI2_CLASSES = [
    ("Etot/Einc", lambda k: k.startswith("Etot")),
    ("E layer", lambda k: k.startswith("E layer")),
    ("EC eta", lambda k: k.startswith("EC Eta")),
    ("EC phi", lambda k: k.startswith("EC Phi")),
    ("Width eta", lambda k: k.startswith("Width Eta")),
    ("Width phi", lambda k: k.startswith("Width Phi")),
    ("Voxel dist", lambda k: k.startswith("Voxel")),
]


def parse_classifier(path):
    """Return every (input_file, auc, jsd) in one classifier_*.txt.

    These files are APPENDED to across runs, so one file can hold several
    results -- including failed attempts that wrote a header and no result.
    Pair each result with the input_file header that precedes it.
    """
    out, pending = [], None
    lines = open(path).read().splitlines()
    for i, line in enumerate(lines):
        m = re.search(r"input_file='([^']*)'", line)
        if m:
            pending = os.path.basename(m.group(1))
        if line.startswith("Final result of classifier test") and i + 1 < len(lines):
            nums = re.findall(r"[0-9]*\.[0-9]+", lines[i + 1])
            if len(nums) >= 2:
                out.append((pending, float(nums[0]), float(nums[1])))
    return out


def parse_chi2(path):
    vals, lines = {}, open(path).read().splitlines()
    for i, line in enumerate(lines):
        s = line.rstrip()
        if s.endswith(":") and i + 1 < len(lines):
            try:
                vals[s[:-1].strip()] = float(lines[i + 1])
            except ValueError:
                pass
    return vals


def parse_fpd_kpd(path):
    txt = open(path).read()
    f = re.search(r"FPD \(x10\^3\):\s*([0-9.]+)\s*±\s*([0-9.]+)", txt)
    k = re.search(r"KPD \(x10\^3\):\s*([0-9.]+)\s*±\s*([0-9.]+)", txt)
    return ((float(f.group(1)), float(f.group(2))) if f else None,
            (float(k.group(1)), float(k.group(2))) if k else None)


def collect(run):
    d = os.path.join(HERE, run)
    row = {"run": run}
    for mode, key in [("cls-low", "cls_low"), ("cls-high", "cls_high"),
                      ("cls-resnet", "cls_resnet"), ("all", "cls_all")]:
        p = os.path.join(d, f"classifier_{mode}_2.txt")
        if os.path.exists(p):
            res = parse_classifier(p)
            if res:
                # several results in one file = repeats; report mean and spread
                aucs = [a for _, a, _ in res]
                row[key] = statistics.mean(aucs)
                if len(aucs) > 1:
                    row[key + "_n"] = len(aucs)
                    row[key + "_spread"] = max(aucs) - min(aucs)
    p = os.path.join(d, "histogram_chi2_2.txt")
    if os.path.exists(p):
        v = parse_chi2(p)
        if v:
            means = []
            for name, pred in CHI2_CLASSES:
                xs = [x for k, x in v.items() if pred(k)]
                if xs:
                    row["chi2_" + name] = statistics.mean(xs)
                    means.append(statistics.mean(xs))
            row["chi2_mean"] = statistics.mean(means) if means else None
            row["chi2_n"] = len(v)
    p = os.path.join(d, "fpd_kpd_2.txt")
    if os.path.exists(p):
        fpd, kpd = parse_fpd_kpd(p)
        if fpd:
            row["fpd"], row["fpd_err"] = fpd
        if kpd:
            row["kpd"], row["kpd_err"] = kpd
    return row


def fmt(x, nd=4):
    return "-" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="store_true", help="also write summary.csv")
    args = ap.parse_args()

    runs = [r for r in ORDER if os.path.isdir(os.path.join(HERE, r))]
    runs += sorted(d for d in os.listdir(HERE)
                   if os.path.isdir(os.path.join(HERE, d)) and d not in ORDER)
    rows = [collect(r) for r in runs]

    print("\n## Classifier AUC and histogram chi-squared\n")
    print("| run | cls-low | cls-high | cls-resnet | cls-all | mean chi2 | FPD x1e3 | KPD x1e3 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['run']} | {fmt(r.get('cls_low'))} | {fmt(r.get('cls_high'))} "
              f"| {fmt(r.get('cls_resnet'))} | {fmt(r.get('cls_all'))} "
              f"| {fmt(r.get('chi2_mean'))} | {fmt(r.get('fpd'), 1)} "
              f"| {fmt(r.get('kpd'), 2)} |")

    print("\n## Histogram chi-squared by observable class\n")
    hdr = [n for n, _ in CHI2_CLASSES]
    print("| run | " + " | ".join(hdr) + " |")
    print("|---" * (len(hdr) + 1) + "|")
    for r in rows:
        if r.get("chi2_mean") is None:
            continue
        print(f"| {r['run']} | " + " | ".join(fmt(r.get('chi2_' + n)) for n in hdr) + " |")

    reps = [(r["run"], k[:-7], r[k], r[k[:-7] + "_n"])
            for r in rows for k in r if k.endswith("_spread")]
    if reps:
        print("\n## Repeat measurements (the only run-to-run scatter available)\n")
        print("| run | mode | n | AUC spread |")
        print("|---|---|---:|---:|")
        for run, mode, spread, n in reps:
            print(f"| {run} | {mode} | {n} | {spread:.4f} |")
        print("\nTreat AUC differences below ~0.015 as noise.")

    if args.csv:
        keys = sorted({k for r in rows for k in r})
        out = os.path.join(HERE, "summary.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["run"] + [k for k in keys if k != "run"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

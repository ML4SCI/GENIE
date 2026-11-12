#!/usr/bin/env python3
"""
run_full_evaluation.py
=======================
Full CaloChallenge Dataset-2 evaluation + comparison pipeline for a
workstation (no SLURM). Chains generate_caloeval_hdf5.py, the official
evaluate.py (three classifier modes + FPD/KPD/PRDC), and
build_comparison_plots.py into one script, with a sanity gate in between so a
bad reconstruction file can't silently propagate into a multi-hour
evaluation run.

Does, in order:
  1. Downloads the official reference file (dataset_2_2.hdf5) from Zenodo, if
     not already present, and checks its byte size against the known-good
     value so a truncated download can't silently propagate downstream.
  2. Regenerates ae_recon_ds2_full.hdf5 via generate_caloeval_hdf5.py --
     ALWAYS regenerates, never reuses a stale file (see the "units bug" note
     in CaloChallenge/README.md's "Known gotchas" -- an old export sitting on
     disk from a previous script version produced FPD/KPD/AUC numbers
     ~1000x worse than reality for days before anyone noticed) -- and gates
     on a physical E_dep/E_inc sanity check before proceeding to the
     expensive part.
  3. Runs the official evaluate.py for every classifier/metric mode needed:
       mode=all       (avg, avg-E, hist, cls-low classifier, fpd, kpd, prdc)
       mode=cls-high
       mode=cls-resnet
  4. Builds the comparison plots against the 18 official Dataset-2
     submissions (CaloChallenge/final_results/CaloChallenge_results_ds2.hdf).

Usage:
    python run_full_evaluation.py
    python run_full_evaluation.py --repo-dir /path/to/CaloChallenge
    python run_full_evaluation.py --skip-generate --skip-evaluate   # just replot

Requires (in whatever Python environment you invoke this with): numpy, h5py,
torch, scikit-learn, scipy, matplotlib, pandas, jetnet, prdc. This script
does not manage conda/venv environments for you -- activate the right one
first (jetnet/prdc will be auto-installed via pip if missing).

This is meant for a machine that can actually sustain a multi-hour GPU job.
If you're about to run it on a laptop, check `systemctl status thermald`
first -- this project already lost two runs to a laptop with no working
thermal management (thermald: "Unsupported cpu model or platform") crashing
mid-evaluation.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ZENODO_REF_URL = "https://zenodo.org/api/records/6366271/files/dataset_2_2.hdf5/content"
ZENODO_REF_SIZE = 1366708391  # bytes -- matches the file's own HDF5 superblock


def run(cmd, **kwargs):
    print(f"\n>> {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def check_cuda(python_exe):
    out = subprocess.run(
        [python_exe, "-c", "import torch; print(torch.cuda.is_available())"],
        capture_output=True, text=True,
    )
    available = out.stdout.strip() == "True"
    if not available:
        print("WARNING: CUDA not available in this environment -- classifier "
              "training (mode=all/cls-high/cls-resnet) will run on CPU and "
              "may take much longer than expected.")
    return available


def ensure_pkg(python_exe, module, pip_name=None):
    check = subprocess.run([python_exe, "-c", f"import {module}"], capture_output=True)
    if check.returncode != 0:
        print(f">> {module} not found, installing {pip_name or module} ...")
        run([python_exe, "-m", "pip", "install", "-q", pip_name or module])


def ensure_reference_file(ref_file: Path):
    if not ref_file.exists():
        print(f">> Downloading {ref_file.name} from Zenodo "
              f"(DOI 10.5281/zenodo.6366270) ...")
        run(["curl", "-L", "-o", str(ref_file), ZENODO_REF_URL])
    size = ref_file.stat().st_size
    if size != ZENODO_REF_SIZE:
        sys.exit(
            f"ERROR: {ref_file} is {size} bytes, expected {ZENODO_REF_SIZE} -- "
            f"looks truncated/corrupt. Delete it and re-run."
        )
    print(f"Reference file OK ({size} bytes).")


def regenerate_reconstruction(python_exe, repo_dir, weights, data_pt, gen_file):
    print(f">> Regenerating {gen_file} ...")
    run([
        python_exe, str(repo_dir / "models" / "1_latent_diffusion"
                        / "generate_caloeval_hdf5.py"),
        "--weights", str(weights),
        "--data", str(data_pt),
        "--out", str(gen_file),
        "--split", "all",
    ])


def sanity_check_units(gen_file: Path, ref_file: Path):
    import h5py
    with h5py.File(gen_file) as f:
        gen = f["showers"][:5000].sum(axis=1) / f["incident_energies"][:5000].squeeze()
    with h5py.File(ref_file) as f:
        ref = f["showers"][:5000].sum(axis=1) / f["incident_energies"][:5000].squeeze()
    g, r = float(gen.mean()), float(ref.mean())
    ratio = g / r
    print(f"Sanity check: generated E_dep/E_inc={g:.4f}, "
          f"reference={r:.4f}, ratio={ratio:.3f}")
    if not (0.5 < ratio < 2.0):
        sys.exit(
            f"ERROR: generated/reference E_dep/E_inc ratio is {ratio:.3f}, "
            f"expected close to 1.0 -- this looks like the same 1000x units "
            f"bug that hit this project before. Aborting rather than running "
            f"a multi-hour evaluation against bad data."
        )
    print("Sanity check passed.")


def run_evaluate(python_exe, code_dir, gen_file, ref_file, out_dir, modes):
    """Run each evaluate.py mode. A failure in one mode (e.g. mode=all dying
    partway through its fpd/kpd sub-step on a broken jetnet/numpy install)
    does NOT abort the rest -- whatever that mode already wrote to out_dir
    before failing (classifier results, histograms, etc. all get flushed to
    disk as they complete, not just at the end) stays usable, and the
    remaining modes still get a chance to run so build_comparison_plots.py
    has as much real data as possible instead of nothing."""
    failed = []
    for mode in modes:
        print(f"\n>> evaluate.py --mode {mode} ...")
        try:
            run([
                python_exe, "evaluate.py",
                "--input_file", str(gen_file),
                "--reference_file", str(ref_file),
                "--dataset", "2",
                "--mode", mode,
                "--save_mem",
                "--output_dir", str(out_dir),
            ], cwd=code_dir)
        except subprocess.CalledProcessError as e:
            failed.append(mode)
            print(f"\n  WARNING: evaluate.py --mode {mode} failed (exit "
                  f"{e.returncode}) -- continuing with the remaining modes. "
                  f"Whatever this mode already wrote to {out_dir} before "
                  f"failing is still there and still valid; only the parts "
                  f"after the failure point are missing. Check the output "
                  f"above for the actual error and re-run just this mode "
                  f"once fixed -- no need to redo the modes that succeeded.")
    if failed:
        print(f"\nModes that failed: {failed} -- comparison plots will show "
              f"'pending' for whatever those modes were supposed to produce.")


def build_comparison_plots(python_exe, repo_dir, out_dir, official_hdf,
                            comparison_outdir, our_name):
    run([
        python_exe, str(repo_dir / "scripts" / "build_comparison_plots.py"),
        "--eval-dir", str(out_dir),
        "--official", str(official_hdf),
        "--outdir", str(comparison_outdir),
        "--our-name", our_name,
    ])


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-dir", default=str(Path(__file__).resolve().parents[2]),
                   help="Repository root; every other default hangs off it "
                        "(default: inferred from this script's location)")
    p.add_argument("--code-dir", default=None,
                   help="Directory containing evaluate.py "
                        "(default: <repo-dir>/CaloChallenge/code)")
    p.add_argument("--weights", default=None,
                   help="Checkpoint path (default: <repo-dir>/best_calo_ae_cond_improved.pt)")
    p.add_argument("--data", default=None,
                   help="Dataset2.pt path (default: <repo-dir>/data/Dataset2.pt)")
    p.add_argument("--ref-file", default=None,
                   help="Reference hdf5 path (default: <repo-dir>/dataset_2_2.hdf5)")
    p.add_argument("--gen-file", default=None,
                   help="Reconstruction hdf5 path (default: <repo-dir>/ae_recon_ds2_full.hdf5)")
    p.add_argument("--out-dir", default=None,
                   help="evaluate.py output dir (default: <repo-dir>/evaluation_results_ds2)")
    p.add_argument("--comparison-outdir", default=None,
                   help="Comparison plots dir (default: <repo-dir>/docs/figures, "
                        "which is the tracked location the README embeds)")
    p.add_argument("--official-hdf", default=None,
                   help="Official results table "
                        "(default: <repo-dir>/CaloChallenge/final_results/"
                        "CaloChallenge_results_ds2.hdf)")
    p.add_argument("--python", default=sys.executable,
                   help="Python interpreter to use for the sub-scripts "
                        "(default: whatever runs this script)")
    p.add_argument("--our-name", default="Our AE (FactorizedCaloAE)")
    p.add_argument("--modes", default="all,cls-high,cls-resnet",
                   help="Comma-separated evaluate.py modes to run")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip the reference-file download/check step")
    p.add_argument("--skip-generate", action="store_true",
                   help="Skip regenerating the reconstruction file (uses the "
                        "existing --gen-file as-is). NOT recommended -- this "
                        "is exactly the mistake that caused the original "
                        "1000x units bug: an evaluation running against a "
                        "stale file nobody regenerated.")
    p.add_argument("--skip-evaluate", action="store_true",
                   help="Skip evaluate.py entirely, only rebuild comparison "
                        "plots from whatever is already in --out-dir")
    args = p.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    code_dir = (Path(args.code_dir).resolve() if args.code_dir
                else repo_dir / "CaloChallenge" / "code")
    weights = (Path(args.weights).resolve() if args.weights
               else repo_dir / "best_calo_ae_cond_improved.pt")
    data_pt = (Path(args.data).resolve() if args.data
               else repo_dir / "data" / "Dataset2.pt")
    ref_file = (Path(args.ref_file).resolve() if args.ref_file
                else repo_dir / "dataset_2_2.hdf5")
    gen_file = (Path(args.gen_file).resolve() if args.gen_file
                else repo_dir / "ae_recon_ds2_full.hdf5")
    out_dir = (Path(args.out_dir).resolve() if args.out_dir
               else repo_dir / "evaluation_results_ds2")
    comparison_outdir = (Path(args.comparison_outdir).resolve() if args.comparison_outdir
                         else repo_dir / "docs" / "figures")
    official_hdf = (Path(args.official_hdf).resolve() if args.official_hdf
                    else repo_dir / "CaloChallenge" / "final_results" /
                         "CaloChallenge_results_ds2.hdf")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    print(f"repo_dir          = {repo_dir}")
    print(f"code_dir          = {code_dir}")
    print(f"weights           = {weights}")
    print(f"data_pt           = {data_pt}")
    print(f"ref_file          = {ref_file}")
    print(f"gen_file          = {gen_file}")
    print(f"out_dir           = {out_dir}")
    print(f"comparison_outdir = {comparison_outdir}")
    print(f"official_hdf      = {official_hdf}")
    print(f"modes             = {modes}")

    if not code_dir.exists():
        sys.exit(f"ERROR: code_dir {code_dir} does not exist -- pass --code-dir.")
    if not official_hdf.exists():
        sys.exit(f"ERROR: official_hdf {official_hdf} does not exist -- pass --official-hdf.")

    out_dir.mkdir(parents=True, exist_ok=True)
    comparison_outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    if not args.skip_evaluate:
        check_cuda(args.python)
        ensure_pkg(args.python, "jetnet")
        ensure_pkg(args.python, "prdc")

    if not args.skip_download:
        ensure_reference_file(ref_file)
    elif not ref_file.exists():
        sys.exit(f"--skip-download given but {ref_file} does not exist.")

    if not args.skip_generate:
        if not weights.exists():
            sys.exit(f"ERROR: checkpoint {weights} does not exist -- pass --weights.")
        if not data_pt.exists():
            sys.exit(f"ERROR: {data_pt} does not exist -- pass --data.")
        regenerate_reconstruction(args.python, repo_dir, weights, data_pt, gen_file)
    else:
        print(f"Skipping regeneration (--skip-generate) -- using existing "
              f"{gen_file} as-is. Make sure you trust it.")
        if not gen_file.exists():
            sys.exit(f"ERROR: --skip-generate given but {gen_file} does not exist.")

    sanity_check_units(gen_file, ref_file)

    if not args.skip_evaluate:
        run_evaluate(args.python, code_dir, gen_file, ref_file, out_dir, modes)
    else:
        print("Skipping evaluate.py (--skip-evaluate) -- only rebuilding comparison plots.")

    build_comparison_plots(args.python, repo_dir, out_dir, official_hdf,
                            comparison_outdir, args.our_name)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed / 60:.1f} minutes.")
    print(f"  Evaluation results : {out_dir}/")
    print(f"  Comparison plots   : {comparison_outdir}/")


if __name__ == "__main__":
    main()

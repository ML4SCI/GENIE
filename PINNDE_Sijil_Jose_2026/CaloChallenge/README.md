# CaloChallenge — the official evaluation code

This directory holds **only a clone of the upstream challenge repository**. It
is not part of this project's source, and it is gitignored (it carries its own
`.git`, so committing it here would nest one repository inside another).

```bash
git clone https://github.com/CaloChallenge/homepage.git CaloChallenge/CaloChallenge
```

That gives you:

- `CaloChallenge/code/evaluate.py` — the official evaluation suite, plus
  `HighLevelFeatures.py`, `resnet.py`, `XMLHandler.py`.
- `CaloChallenge/final_results/CaloChallenge_results_ds2.hdf` — the official
  numeric results for all 17 published Dataset-2 submissions plus a Geant4
  self-comparison row (`pd.read_hdf(..., key='ds2')`). This is the source
  `scripts/build_comparison_plots.py` reads; the comparison numbers in
  `docs/` are never hand-copied from the paper.

One local change is needed on a machine with no LaTeX install: set
`usetex=False` in `code/evaluate.py`.

Everything that used to live loose in this directory has moved — see
`scripts/`, `sweeps/`, `notebooks/` and
`models/1_latent_diffusion/generate_caloeval_hdf5.py`. The old README is kept
at `docs/archive/calo_ae_README_2026-08.md`.

# Results

The committed output of every official `evaluate.py` run, one directory per
run, with the configuration that produced it alongside.

**Why these files are in version control and the rest is not.** Across the six
evaluation directories in the working tree, the text results total **164 KB**
and everything else — classifier checkpoints and 280 plots per run — totals
**1.7 GB**. The results are 0.01% of the bytes. So the text is committed and
the artifacts are gitignored and regenerable.

## Layout

```
results/
├── make_tables.py          regenerates every table below from the files here
├── summary.csv             generated: one row per run
└── <run_name>/
    ├── config.yaml         what produced it: model, hyperparameters, caveats
    ├── classifier_*.txt    AUC / JSD per classifier mode
    ├── histogram_chi2_2.txt   227 chi-squared values
    └── fpd_kpd_2.txt       where it was run
```

## Regenerating the tables

```bash
python results/make_tables.py          # print
python results/make_tables.py --csv    # also write summary.csv
```

**Always generate, never transcribe.** Two numbers in this project's docs have
already drifted from their source: an `f_scale` measured on a 20k subset
instead of the full 100k (1.9714 against the correct 2.9756), and the
autoencoder row attributed to a v4 checkpoint when the evaluated file came
from a pre-v4 one. Both would have been impossible if the table were generated.

## The runs

| directory | model | what it is |
|---|---|---|
| `two_stage_100k` | two-stage PINN | **the headline model.** cls-low 0.578 |
| `conv_unet_runA` | direct PINN | single-stage 3D conv U-Net |
| `conv_unet_runB` | direct PINN | **same file as run A, classifier retrained** |
| `film_mlp_100k` | direct PINN | flat MLP with FiLM — superseded |
| `autoencoder_prev4_ep85` | AE reconstruction | the ceiling for the latent path |
| `autoencoder_sweep_d96_ep460` | AE reconstruction | same latent size, 5× the epochs |
| `latent_diffusion` | latent diffusion | reached its own ceiling |

Two things to know when reading these:

**`conv_unet_runA` and `runB` are the same generated file evaluated twice**,
with the classifier retrained from scratch. That accident is the only
error bar in the project: cls-low 0.6700 against 0.6827. **Treat AUC
differences below ~0.015 as noise.** `make_tables.py` reports repeat spreads
in a separate table for this reason.

**Some `classifier_*.txt` files hold more than one run**, because
`evaluate.py` appends. `two_stage_100k/classifier_cls-low_2.txt` contains a
failed 20k attempt (a header with no result — `evaluate.py` hard-codes a
60k/20k/20k split for dataset 2, so classifier modes need exactly 100k
showers) followed by two successful 100k runs. The parser pairs each result
with the `input_file` header above it and ignores headers with no result.

## Missing

- **FPD/KPD for the two-stage model.** The largest reporting gap: it is the
  metric the challenge summary paper leads with, and the only values we have
  are the autoencoder's, which are worse than the worst published submission.
- **10-run classifier averaging.** The published tables average 10 trainings
  and quote ±0.004; our single runs carry ±0.015.
- **Any evaluation of the v4 autoencoder.** It exists only on the workstation.

See `docs/EXPERIMENTS.md` for how to close these.

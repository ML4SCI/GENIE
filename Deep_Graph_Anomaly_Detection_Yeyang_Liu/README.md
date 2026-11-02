# Deep Graph Anomaly Detection with Contrastive Learning for New Physics Searches

Author: Yeyang Liu.

Graph-based anomaly detection for jets.

[Project repository](https://github.com/yyLeaves/gsoc26-genie-graph-ad) ·
[v1.0.0](https://github.com/yyLeaves/gsoc26-genie-graph-ad/tree/v1.0.0) ·
[Results](docs/results.md) · [Data pipeline](docs/data_pipeline.md)

## Get started

### 1. Set up the environment

Linux with Git, Bash 4+, curl and tar. Experiments use Python 3.12 and an NVIDIA
RTX 2080 Ti (11 GB). Allow 80 GiB for the full data pipeline, plus about 7 GiB
per signal for packed AE inputs.

```bash
git clone https://github.com/yyLeaves/GENIE.git
cd GENIE/Deep_Graph_Anomaly_Detection_Yeyang_Liu
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv sync --locked
source .venv/bin/activate
```

Or alternatively, with Conda:

```bash
conda create -n genie python=3.12 pip
conda activate genie
python -m pip install -r requirements.txt
```

Select the training device:

```bash
export DEVICE=cuda:0  # cpu for CPU runs
```

### 2. Prepare data

Start with LHCO: download, split, preprocess and build graphs.

```bash
bash scripts/prepare_data.sh LHCO
```

Omit `LHCO` for all datasets, or select datasets and an injection count:

```bash
NSIG=1000 bash scripts/prepare_data.sh LHCO BB1 XtoWRto3W
```

Defaults: CWoLa, `Nsig=1000`. Other counts: `0`, `50`, `100`, `500`.

#### 2.1 Download data

```bash
bash scripts/download_data.sh
```

Or alternatively, pass a directory and dataset names:

```bash
bash scripts/download_data.sh data/raw BB1 XtoWRto3W
```

#### 2.2 Create data splits

```bash
python -m scripts.make_splits --signal LHCO --n-signal 1000
```

Use `--mode IAD` for IAD splits. Mass bands and event counts are in the
[data pipeline](docs/data_pipeline.md#2-cwola-settings).

#### 2.3 Prepare jets

```bash
python -m scripts.preprocess
```

Create BB1 full-range and SR evaluation splits using the preprocessed `mJJ`
(also included in `prepare_data.sh`):

```bash
python -m scripts.make_bb1_splits
```

#### 2.4 Build graphs

```bash
python -m scripts.build_graph
```

Default: 30-subjet Unique-6 graphs. `preprocess` and `build_graph` accept
`--datasets LHCO BB1` to select sources.

For the 60-subjet ATII settings:

```bash
python -m scripts.build_graph --max-subjets 60 --output-dir data/graphs60
```

### 3. Train

Defaults: seed **42**, 50 epochs, test evaluation every 5 epochs and at the final
epoch. Models train on adjacent SB events (3.1–3.3 and 3.7–3.9 TeV), using
`reference` in the code. Validation also uses SB; test evaluation uses SR.
Use `SAMPLE=SR bash scripts/train.sh` to train on SR instead.

Set `SPLIT_FILE`, `RUN_DIR` and `DEVICE` as needed. Extra Python options can be
passed after `scripts/train.sh`. Optimizers and losses follow each model's recipe.

#### 3.1 Baseline and EB3

```bash
bash scripts/train.sh
REG_WEIGHT=1 RUN_DIR=runs/edgeconv_eb3_sb bash scripts/train.sh
```

#### 3.2 EdgeConv–GLADC

```bash
MODEL=gladc RUN_DIR=runs/gladc_sb bash scripts/train.sh
MODEL=gladc RUN_DIR=runs/gladc_fraction_sb bash scripts/train.sh --gladc-variant fraction
```

`--gladc-variant`: `original`, `neighbor`, `standardized`, `fraction`.
`--objective`: `reconstruction`, `cycle`, `contrast`, `full` (default).
[Methods and results](docs/edgeconv-gladc.md).

#### 3.3 NetGe

```bash
MODEL=netge RUN_DIR=runs/netge_sb bash scripts/train.sh --netge-variant max_bilinear
MODEL=netge RUN_DIR=runs/netge_gladc_sb bash scripts/train.sh --netge-variant max_bilinear --objective full
```

`--netge-variant`: `original`, `jet`, `fraction`, `max_bilinear`.
[NetGe variants](docs/netge.md) · [NetGe–GLADC](docs/netge-gladc.md).

#### 3.4 ATII

```bash
VARIANT=original bash scripts/atii.sh
VARIANT=energy SUBJETS=30 bash scripts/atii.sh
VARIANT=energy SUBJETS=60 bash scripts/atii.sh
VARIANT=context SUBJETS=60 bash scripts/atii.sh
```

Each command trains and evaluates one model. `SIGNAL`, `NSIG` and `DEVICE` select
the signal, injection count and GPU. `SAMPLE=SR` selects SR training.
[Methods and recipes](docs/sdm.md#2-atii).

#### 3.5 ATI and NAT

```bash
VARIANT=energy bash scripts/ati.sh
VARIANT=energy bash scripts/nat.sh
```

Each command trains on SB and evaluates on SR. Set `SIGNAL`, `NSIG`,
`DEVICE` or `SAMPLE=SR` as needed.
[ATI variants and results](docs/sdm.md#11-run) · [NAT variants and results](docs/sdm.md#31-run).

#### 3.6 Run each signal separately

Prepare all datasets, then train:

```bash
bash scripts/prepare_data.sh
bash scripts/baseline.sh
bash scripts/regularization.sh
bash scripts/calibration.sh
```

These cover LHCO and all four KS signals. Calibration uses the completed baseline
and EB3 checkpoints. To select signals:

```bash
NSIG=500 DEVICE=cuda:1 bash scripts/baseline.sh LHCO XtoWRto3W
```

The KS tree classifier has its own entry point:

```bash
FEATURES=subjettiness bash scripts/kitchensink.sh
```

For 50 classifiers per run and the median over ten runs:

```bash
FEATURES=subjettiness ENSEMBLE=1 RUNS=10 bash scripts/kitchensink.sh
```

Each run rebuilds the signal injection and training pool. Per-run scores and
`summary.json` go under `runs/kitchensink_cwola/ensemble/`.

It prepares native-table features in `data/features/kitchensink/` and splits in
`data/splits/kitchensink/`, then trains. EFP tables are downloaded/extracted when
selected. `FEATURE_DIR` and `SPLIT_DIR` select the output locations.

#### 3.7 Resume

```bash
RESUME=runs/edgeconv_sb/last.pt bash scripts/train.sh
```

Continues from the saved model, optimizer, schedule and RNG state.

### 4. Inference

```bash
RUN_DIR=runs/edgeconv_sb bash scripts/inference.sh
```

Uses the checkpoint's model, graph directory and score. `SPLIT_FILE` selects the
evaluation split; `--score` selects a different scoring rule. `total` follows the
checkpoint's selected score. Larger scores mean more anomalous.

Outputs event identifiers, labels, `mjj` and scores in NPZ, plus AUC/maxSIC in JSON.
Training writes `settings.json`, `losses.csv`, `metrics.csv` and `last.pt`.
[Score definitions and reported results](docs/results.md).

For a saved calibration:

```bash
RUN_DIR=runs/edgeconv_eb3_sb CALIBRATION=path/to/calibration.pt bash scripts/inference.sh
```

To calibrate two different architectures, set `BASELINE_CHECKPOINT` and
`REGULARIZED_CHECKPOINT` when running `scripts/calibration.sh` for one signal.
Each branch uses its checkpoint's graph directory.

Plot saved BB1 scores:

```bash
python -m scripts.plot_bump --scores path/to/bb1_scores.npz --output path/to/bb1_bump
```

### 5. AE features + tree readout

Train AEs on adjacent SB events, then freeze them. Use their physical predictions,
reconstruction errors and graph masses to train 50 gradient-boosted tree models
to separate SR from SB. The event score is their mean predicted probability.

Uses `Nsig=1000`, 30-subjet graphs and default seed `42`.
After step 2, choose a method. Each command prepares the feature tables and packed
graphs, trains the AE, fits the tree readout and evaluates:

```bash
# NetGe with physics prediction heads: 80 readout features
bash scripts/physics_ae.sh LHCO

# NetGe + pair-attention: 154 features, with and without InfoNCE
bash scripts/dual_ae.sh LHCO
```

Pass several signals after preparing their data, for example
`bash scripts/dual_ae.sh LHCO XtoWRto3W`. `SEED` selects the AE seed.

Select a NetGe AE variant (default: `ratios`):

```bash
VARIANT=graph_mass bash scripts/physics_ae.sh LHCO
```

| Variant | Tree inputs | Count |
|---|---|---:|
| `predicted` | Predicted mass, τ and EFP | 72 |
| `graph_mass` | Replace predicted mass with graph mass | 72 |
| `errors` | Graph mass + predicted τ/EFP + AE errors | 74 |
| `ratios` | Add τ-ratio predictions | 80 |

The first three reuse one AE without the τ-ratio head; `ratios` trains a separate AE.

Repeat NetGe AE training and tree readout on four LHCO KS splits:

```bash
python -m scripts.ks_runs --runs 4 --device cuda:0
```

`--runs` counts repetitions starting at run 0; default: 1.
[Methods and feature sources](docs/ae_readout.md).

## Structure

```text
src/       Models, data loading, losses, training and scoring
scripts/   Data preparation, train/inference and plotting entry points
docs/      Data pipeline, methods and results
assets/    Documentation figures
data/      Raw data, splits and graphs (not tracked)
runs/      Checkpoints and results (not tracked)
```

## References

### Data sources

- LHCO: [R&D Dataset for LHC Olympics 2020 Anomaly Detection Challenge](https://zenodo.org/records/6466204).
- `extra_qcd`: [Additional QCD Background Events for LHCO2020 R&D (signal region only)](https://zenodo.org/records/8370758).
- BB1: [Official Datasets for LHC Olympics 2020 Anomaly Detection Challenge](https://zenodo.org/records/4536624).
- Kitchen Sink signals: [Additional Signal Models for the LHCO2020 R&D](https://zenodo.org/records/18983506).
- LHCO and extra-QCD subjettiness tables: [Back to the Roots: Tree-Based Algorithms for Weakly Supervised Anomaly Detection](https://github.com/uhh-pd-ml/treebased_anomaly_detection#dataset).

### Repositories

- [Graph Autoencoder for Model Agnostic Anomaly Detection](https://github.com/DimAthanasakos/Graph-AutoEncoder-for-Model-Agnostic-Anomaly-Detection): graph construction and EdgeConv autoencoder baseline.
- [Kitchen Sink](https://github.com/lang-lukas/KitchenSink): jet preprocessing and CWoLa data splits.
- [GLADC](https://github.com/XuexiongLuoMQ/GLADC): NetGe and contrastive graph anomaly detection.

### Papers

- [Graph theory inspired anomaly detection at the LHC](https://arxiv.org/abs/2506.19920)
- [Kitchen Sink Anomaly Detection](https://arxiv.org/abs/2604.20965)
- [Back To The Roots: Tree-Based Algorithms for Weakly Supervised Anomaly Detection](https://arxiv.org/abs/2309.13111)
- [Deep graph level anomaly detection with contrastive learning](https://www.nature.com/articles/s41598-022-22086-3)
- [Self-Discriminative Modeling for Anomalous Graph Detection](https://proceedings.mlr.press/v267/cai25k.html)

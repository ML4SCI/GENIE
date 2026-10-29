# Data pipeline

We use the Kitchen Sink (KS) event splits and jet definition, then build our
subjet graphs.

Prepare all datasets from the project root:

```bash
bash scripts/prepare_data.sh
```

```text
data/raw/
  ├─ jet feature tables → data/splits/     (event selection)
  └─ particles → data/processed/ → data/graphs/

split indices + graph files → two jet graphs per event
```

## 1. Raw data

Counts before mass cuts and split selection:

| Dataset | Background | Signal |
| --- | ---: | ---: |
| LHCO | 1,000,000 | 100,000 |
| extra-QCD | 612,858 | 0 |
| BB1 | 999,166 | 834 |
| XtoWRto3W | 0 | 32,801 |
| XtoYYprime | 0 | 44,673 |
| ZpToTpTp | 0 | 32,322 |
| YtoHHto4T | 0 | 48,079 |

Files under `data/raw/`:

- `lhco/`: LHCO particles (`events_anomalydetection.h5`), extra-QCD particles
  (`extra_qcd_cartesian.h5`), and their feature tables.
- `bb1/`: `events_LHCO2020_BlackBox1.h5` and its `.masterkey` truth labels.
- `kitchensink/<signal>/`: `events.h5` and `<signal>_subjettinesses.h5`.

The split step uses these tables' two-jet momenta and masses to calculate `mJJ`:

| Source | Feature table |
| --- | --- |
| LHCO | `lhco/lhco_subjettiness.h5` |
| extra-QCD | `lhco/extra_qcd_features.h5` (same row order as its particle file) |
| KS signal | `kitchensink/<signal>/<signal>_subjettinesses.h5` |

These tables select events; graph features come from particles.

Data sources:

- [LHCO R&D](https://zenodo.org/records/6466204)
- [Extra-QCD](https://zenodo.org/records/8370758)
- [BB1](https://zenodo.org/records/4536624)
- [Kitchen Sink signals](https://zenodo.org/records/18983506)
- [LHCO and extra-QCD subjettiness tables](https://github.com/uhh-pd-ml/treebased_anomaly_detection/tree/main/dataset)

## 2. CWoLa settings

CWoLa trains a classifier to distinguish events in the signal region (SR) from
events in the neighbouring sidebands (SB). `mJJ` defines the regions; it is not
a classifier input.

[`make_splits.py`](../scripts/make_splits.py) calls
[`src/splits.py`](../src/splits.py). The default is CWoLa, `Nsig=1000`.

1. Take LHCO background and one signal dataset: LHCO or one of the four KS signals.
2. Shuffle and inject `Nsig` signal events **before mass cuts**.
   Supported values: `0`, `50`, `100`, `500`, `1000`.
3. Select up to 120,000 events in the signal region (SR), plus events in the
   two sidebands (SB).
4. Divide SR + SB equally into training and validation, preserving their proportions.
5. Use held-out extra-QCD background and unused signal events for testing, within SR.

| Sample | `mJJ` range [TeV] | Role |
| --- | --- | --- |
| SR | `3.3 < mJJ < 3.7` | Training/validation sample that may contain signal |
| SB | `3.1 < mJJ <= 3.3` or `3.7 <= mJJ < 3.9` | Training/validation background reference |
| Test | `3.3 < mJJ < 3.7` | Held-out background and signal |

The classifier trains on region labels (`weak_label`: SR=1, SB=0), not on
signal/background labels. Evaluation uses `truth`: background=0, signal=1.

Our autoencoders train and validate on SB by default (`SAMPLE=reference`);
`scripts/train.sh` also accepts `SAMPLE=SR`. Test evaluation uses SR.
For [AE features + tree readout](ae_readout.md), freeze the SB-trained AE,
then train the trees on SR/SB region labels and evaluate in SR.

Default CWoLa splits (`Nsig=1000`), in events:

| Signal dataset | Training: SR + SB | Validation: SR + SB | Test: background + signal (SR) |
| --- | ---: | ---: | ---: |
| LHCO | 60,000 + 65,118 | 60,000 + 65,119 | 340,000 + 20,000 |
| XtoWRto3W | 60,000 + 65,121 | 60,000 + 65,121 | 340,000 + 20,000 |
| XtoYYprime | 60,000 + 65,153 | 60,000 + 65,153 | 340,000 + 20,000 |
| ZpToTpTp | 60,000 + 65,129 | 60,000 + 65,130 | 340,000 + 20,000 |
| YtoHHto4T | 60,000 + 65,138 | 60,000 + 65,138 | 340,000 + 20,000 |

Each signal has its own split. Changing `Nsig` also changes the split shuffle.
IAD remains available via `MODE=IAD`: it replaces SB with a separate extra-QCD
reference sample in SR. BB1 is used for separate evaluation.

## 3. Prepare jets

[`preprocess.py`](../scripts/preprocess.py) calls
[`src/preprocessing.py`](../src/preprocessing.py):

1. Remove zero-padded particles (`pT=0`).
2. Treat particles as massless and convert `(pT, eta, phi)` to four-momenta.
3. Cluster with FastJet anti-kT, `R=1`.
4. Keep the two highest-pT jets.
5. Compute `mJJ` from their summed four-momenta, using all their constituents.

Save jet and constituent four-vectors in `data/processed/`, preserving event rows
and truth labels. Momenta are in GeV; `mJJ` is in TeV.
This particle-derived `mJJ` is saved with the jets; KS splits and the event loader
use the feature-table `mJJ` from step 2.

For BB1, [`make_bb1_splits.py`](../scripts/make_bb1_splits.py) uses the preprocessed
`mJJ` to save full-range and SR (`3.3 < mJJ < 3.7` TeV) evaluation indices.

## 4. Build graphs

[`build_graph.py`](../scripts/build_graph.py) calls
[`src/graph.py`](../src/graph.py). The default representation is:

1. Recluster each jet into at most **30 exclusive-kT subjets**, `R=1`.
2. Sort subjets by decreasing pT; use `ln(pT)` as the node feature (`x`: N × 1),
   with pT in GeV.
3. Centre `(rapidity, phi)` on their pT-weighted means (circular mean for phi).
4. Build a **Unique-6** graph, storing both edge directions.
5. Use `ln(theta)`, `ln(kT)`, and `ln(z)` as edge features (`edge_attr`: E × 3);
   compute `kT` with jet-normalized pT (`pT_i / sum(pT)` within each jet).

Here, `theta` is the distance between subjet coordinates, `kT` combines this
distance with the softer subjet's momentum fraction, and `z` measures momentum
sharing. Coordinates and pT are also saved for geometry-based losses and scoring.

The [ATII resolution study](sdm.md#29-run-the-retained-versions) uses up to 60
subjets, stored separately in `data/graphs60/`.

## 5. KS tree-classifier features

[`prepare_kitchensink.py`](../scripts/prepare_kitchensink.py) builds the baseline,
subjettiness and EFP feature sets from the published tables, ordering jets by mass.
Combined concatenates subjettiness and EFP features. `mJJ` selects regions, not
classifier inputs.

Features are shared across `Nsig` settings in `data/features/kitchensink/`;
native-table split indices go in `data/splits/kitchensink/`. This path uses
`extra_qcd_subjettiness.h5` row order and the seed-42 classifier's internal
50/50 validation split. [`kitchensink.sh`](../scripts/kitchensink.sh) runs
preparation and training together.

`ENSEMBLE=1 RUNS=10` trains 50 classifiers per run with new signal injection and
pool shuffles each run. Average member probabilities, then report the median
AUROC and maxSIC across runs.

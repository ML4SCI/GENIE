# Source modules

| Module | Purpose |
| --- | --- |
| [splits.py](splits.py) | CWoLa and IAD samples, signal injection and train/validation/test splits. |
| [preprocessing.py](preprocessing.py) | Particle four-vectors, jet clustering and dijet mass. |
| [graph.py](graph.py) | Subjet reclustering, Unique-6 connections and node/edge features. |
| [dataset.py](dataset.py) | Load, pack and batch paired jet graphs. |
| [models/](models/README.md) | Neural network architectures. |
| [physics.py](physics.py) | Graph masses, EFPs, KS subjettiness targets and feature standardization. |
| [softdrop.py](softdrop.py) | Soft Drop observables used as prediction targets. |
| [kitchensink.py](kitchensink.py) | Published feature tables and KS tree-classifier splits. |
| [losses.py](losses.py) | Reconstruction, EB3, cycle, contrastive and ATII objectives. |
| [training.py](training.py) | Training epochs, validation and input feature scaling. |
| [readout.py](readout.py) | Train and evaluate the 50-model CWoLa tree ensemble. |
| [scoring.py](scoring.py) | Model anomaly scores and two-jet event scores. |
| [calibration.py](calibration.py) | Percentile score calibration and structure-based model gating. |
| [metrics.py](metrics.py) | Event-level AUROC and maxSIC. |
| [checkpoint.py](checkpoint.py) | Save and load training checkpoints. |

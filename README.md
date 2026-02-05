# GENIE – ML4SCI Neutrino Interaction Toolkit

GENIE (by ML4SCI) is a hybrid toolkit that combines traditional neutrino Monte Carlo modeling with modern ML/quantum-assisted modules. This README walks you through installation, configuration, and execution so you can go from cloning the repo to generating results in minutes.

---

## Table of Contents
1. [Prerequisites](#prerequisites)  
2. [Installation](#installation)  
3. [Directory Layout](#directory-layout)  
4. [Usage](#usage)  
5. [Monitoring & Outputs](#monitoring--outputs)  
6. [Troubleshooting](#troubleshooting)  
7. [Contributing](#contributing)  
8. [License](#license)

---

## Prerequisites

- Python 3.9 or newer  
- Git, GCC/Clang, CMake ≥ 3.21  
- (Optional) ROOT for advanced analysis  
- Conda or virtualenv for dependency management

> Quick setup tip: `conda env create -f environment.yml` guarantees dependency parity with our reference environment.

---

## Installation

### Option A – Full Build (C++ + Python)

```bash
git clone https://github.com/ML4SCI/GENIE.git
cd GENIE
cmake -B build
cmake --build build
Option B – Python-Only Workflow
bashDownloadCopy codegit clone https://github.com/ML4SCI/GENIE.git
cd GENIE
pip install -r requirements.txt

Directory Layout
PathDescriptionsrc/Core physics kernels and ML/quantum modulesconfigs/YAML configs for baseline and custom studiesscripts/CLI utilities to launch experimentsnotebooks/Tutorials and exploratory analysisdocs/Extended documentation and design notesresults/Default location for run artifacts

Usage
1. Minimal CLI Run
bashDownloadCopy codepython scripts/run_genie.py \
  --config configs/baseline.yaml \
  --out results/baseline_run

* --config: selects physics + ML settings
* --out: destination folder for logs, checkpoints, plots
* Add flags like --use-ml or --subset N if listed in scripts/run_genie.py --help

2. Notebook Workflow

1. jupyter notebook notebooks/
2. Open 01_quickstart.ipynb
3. Run cells sequentially to load data, simulate events, and visualize outputs

3. Custom Experiment
bashDownloadCopy codecp configs/baseline.yaml configs/my_study.yaml
# edit the YAML (cross-sections, learning rate, circuit depth, etc.)
python scripts/run_genie.py --config configs/my_study.yaml --out results/my_study

Monitoring & Outputs

* Text logs: logs/<run-id>.txt
* TensorBoard (if enabled):
bashDownloadCopy codetensorboard --logdir runs/

* Plots, metrics, and checkpoints: results/<run-id>/


Troubleshooting
SymptomQuick FixCMake build failsCheck CMake version and compiler support for C++17Missing ROOT warningsInstall ROOT or run with -DENABLE_ROOT=OFF during CMakeExtremely slow runsUse --subset N for smaller test batchesPackage import errorsEnsure virtual environment is activated and dependencies installed

Contributing

1. Pick an open issue (or create a feature request) and announce your intent.
2. Follow coding style and testing guidance in CONTRIBUTING.md.
3. Submit a PR that includes:

Linked issue number
Summary of config/log changes
Before/after metrics if behavior changed




License
Distributed under the terms described in LICENSE. Please ensure downstream use respects both GENIE and ML4SCI licensing requirements.

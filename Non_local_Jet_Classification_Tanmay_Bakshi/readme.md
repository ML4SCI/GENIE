# Non-local Jet Classification with Topological Features

**Author**: Tanmay Bakshi  
**GSoC 2025 Project**: Advanced jet classification using persistent homology and topological data analysis

## Overview

This project implements sophisticated neural network architectures for classifying particle jets, with a focus on capturing non-local geometric features through topological data analysis. The approach combines traditional jet features with persistent homology to improve classification performance on quark vs gluon discrimination tasks.

## Dataset

The project uses the **Quark Gluon Tagging Reference Dataset** by Kasieczka et al., featuring:
- 1.2M training events, 400k validation, 400k test events
- 14 TeV hadronic tops (signal) vs QCD dijets (background)
- Anti-kT 0.8 jets in pT range [550,650] GeV
- Leading 200 jet constituents stored per jet
- Constituents sorted by pT (highest first)

## Project Structure

```
Non_local_Jet_Classification_Tanmay_Bakshi/
├── main.py                    # Main entry point
├── datasets.py                # Data loading utilities
├── coordinates_extract.py     # Feature extraction
├── data_arrange.py           # Data preprocessing
├── preprocess_dask.py        # Parallel preprocessing
├── persistent_net-2.ipynb   # Interactive demo notebook
├── console/                  # Console utilities
├── helper/                   # Helper functions
├── nn/                       # Neural network models
├── persistence/              # Topological analysis
├── scnn/                     # Simplicial CNN implementation
└── Weaver/                   # Weaver framework integration
```

## Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 1.8+
- awkward-array
- scikit-learn
- h5py
- pandas
- numpy

### Installation
```bash
# Navigate to project directory
cd Non_local_Jet_Classification_Tanmay_Bakshi

# Install dependencies (create requirements.txt if needed)
pip install torch awkward scikit-learn h5py pandas numpy matplotlib

# For topological analysis
pip install gudhi  # for persistent homology
```

### Running the Code

**Option 1: Python Script**
```bash
python main.py
```

**Option 2: Interactive Notebook (Recommended)**
```bash
jupyter notebook persistent_net-2.ipynb
```

**Option 3: Data Preprocessing**
```bash
# For large datasets, use parallel preprocessing
python preprocess_dask.py
```

## Key Features

- **Topological Feature Extraction**: Uses persistent homology to capture jet topology
- **Multi-scale Analysis**: Analyzes jets at different geometric scales
- **Advanced Architectures**: Implements Simplicial CNNs and graph-based methods
- **Weaver Integration**: Compatible with the Weaver framework for particle physics ML

## Expected Outputs

- Classification accuracy metrics
- ROC curves and performance plots
- Topological feature visualizations
- Model checkpoints in respective subdirectories

## Troubleshooting

**Common Issues:**
1. **Memory errors**: Reduce batch size or use `preprocess_dask.py` for large datasets
2. **Missing dependencies**: Install `gudhi` for topological analysis features
3. **CUDA errors**: Ensure PyTorch CUDA version matches your system

## Citation

If you use this code, please cite:
```
Kasieczka, G., Plehn, T., Thompson, J., & Russell, M. 
"Quark Gluon Tagging Reference Dataset"
```
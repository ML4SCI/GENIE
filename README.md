# GENIE - Generative Networks for Interpretable Event Generation

[![ML4SCI](https://img.shields.io/badge/ML4SCI-GSoC-blue)](https://ml4sci.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

**GENIE** is a collection of machine learning projects developed as part of Google Summer of Code (GSoC) with [Machine Learning for Science (ML4SCI)](https://ml4sci.org/). This repository contains cutting-edge implementations of generative models, physics-informed neural networks, and graph-based learning techniques applied to high-energy particle physics and scientific computing.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Projects](#projects)
  - [Graph Representation Learning](#1-graph-representation-learning)
  - [Non-local Jet Classification](#2-non-local-jet-classification)
  - [Physics-Informed Neural Networks for Diffusion Equation](#3-physics-informed-neural-networks-for-diffusion-equation)
- [Getting Started](#getting-started)
- [Usage Instructions](#usage-instructions)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🌟 Overview

GENIE explores the intersection of machine learning and particle physics, focusing on:
- **Event Generation**: Creating realistic particle physics events using generative models
- **Anomaly Detection**: Identifying rare or unusual patterns in high-energy physics data
- **Fast Simulation**: Developing efficient alternatives to traditional Monte Carlo simulations
- **Graph-based Learning**: Leveraging graph neural networks for particle jet analysis

Each subproject in this repository represents a complete GSoC contribution with its own methodology, implementation, and results.

---

## 📁 Repository Structure

```
GENIE/
├── Graph_Representation_Learning_Rushil_Singha/
│   ├── code.py                    # Main implementation
│   ├── requirements.txt           # Python dependencies
│   └── README.md                  # Project-specific documentation
│
├── Non_local_Jet_Classification_Tanmay_Bakshi/
│   ├── main.py                    # Entry point
│   ├── datasets.py                # Data loading utilities
│   ├── persistent_net-2.ipynb     # Jupyter notebook demo
│   └── readme.md                  # Project-specific documentation
│
├── Physics_Informed_Neural_Network_Diffusion_Equation_Sijil_Jose/
│   ├── flow_de/                   # Core implementation
│   ├── Jupyter Notebooks/         # Interactive examples
│   ├── Figures/                   # Result visualizations
│   └── README.md                  # Project-specific documentation
│
└── README.md                      # This file
```

---

## 🚀 Projects

### 1. Graph Representation Learning
**Author**: Rushil Singha  
**Focus**: Graph-based diffusion models for jet generation

A PyTorch/PyTorch-Geometric implementation that constructs k-nearest neighbor graphs from particle jets, learns Chebyshev GCN embeddings, and trains diffusion models in latent space to generate realistic jets from the JetNet dataset.

**Key Features**:
- kNN graph construction from particle clouds
- Chebyshev Graph Convolutional Networks (ChebNet)
- Latent diffusion with denoising MLP
- Evaluation using KL divergence & Wasserstein distance

---

### 2. Non-local Jet Classification
**Author**: Tanmay Bakshi  
**Focus**: Advanced jet classification using topological and non-local features

This project implements sophisticated neural network architectures for classifying particle jets, incorporating persistent homology and topological data analysis to capture non-local geometric features.

**Key Features**:
- Persistent homology-based feature extraction
- Advanced neural network architectures
- Integration with Weaver framework
- Topological data analysis for jet classification

---

### 3. Physics-Informed Neural Networks for Diffusion Equation
**Author**: Sijil Jose  
**Focus**: PINNDE - Fast sampling via reverse-time diffusion

Develops a proof-of-concept for building fast and reliable samplers by solving reverse-time diffusion equations using Physics-Informed Neural Networks (PINNs). Successfully demonstrated on 1D, 2D, and 3D probability distributions.

**Key Features**:
- Accurate q-function approximation for reverse-time diffusion
- Multiple PINN architectures tested
- Validated on Gaussian Mixture Models in 1D, 2D, and 3D
- Fast Calorimeter Challenge 2022 benchmarking (in progress)

---

## 🛠️ Getting Started

### Prerequisites

- **Python**: 3.8 or higher (3.9+ recommended)
- **pip**: Latest version (22.0+)
- **Git**: For cloning the repository
- **CUDA**: Optional but recommended for GPU acceleration (CUDA 11.8+ for PyTorch compatibility)
- **Memory**: At least 8GB RAM (16GB+ recommended for larger datasets)
- **Storage**: At least 5GB free space for datasets and model checkpoints

### System Requirements by Project

| Project | Python | GPU Memory | Estimated Runtime |
|---------|--------|------------|-------------------|
| Graph Representation Learning | 3.8+ | 4GB+ (optional) | 2-4 hours |
| Non-local Jet Classification | 3.8+ | 6GB+ (recommended) | 1-3 hours |
| Physics-Informed Neural Networks | 3.8+ | 2GB+ (optional) | 30min-2 hours |

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ML4SCI/GENIE.git
   cd GENIE
   ```

2. **Choose a project** and navigate to its directory:
   ```bash
   cd Graph_Representation_Learning_Rushil_Singha
   # OR
   cd Non_local_Jet_Classification_Tanmay_Bakshi
   # OR
   cd Physics_Informed_Neural_Network_Diffusion_Equation_Sijil_Jose
   ```

3. **Install project-specific dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   > **Note**: Each project has its own `requirements.txt` file. Make sure you're in the correct project directory.

---

## 📖 Usage Instructions

### Graph Representation Learning

```bash
cd Graph_Representation_Learning_Rushil_Singha
pip install -r requirements.txt
python code.py
```

**What it does**:
- Downloads and preprocesses JetNet dataset
- Constructs kNN graphs from particle jets
- Trains Chebyshev GCN encoder
- Runs diffusion model training
- Generates synthetic jets
- Evaluates with KL divergence and Wasserstein distance
- Saves visualizations to `results/`

**Expected Output**: Training logs, evaluation metrics, and visualization plots in the `results/` directory.

**Estimated Runtime**: 2-4 hours on CPU, 30-60 minutes with GPU

**Key Output Files**:
- `results/training_logs.txt` - Training progress and metrics
- `results/generated_jets.png` - Visualization of generated vs real jets
- `results/evaluation_metrics.json` - KL divergence and Wasserstein distances

---

### Non-local Jet Classification

```bash
cd Non_local_Jet_Classification_Tanmay_Bakshi
pip install -r requirements.txt  # If available
python main.py
```

**Alternative - Jupyter Notebook**:
```bash
jupyter notebook persistent_net-2.ipynb
```

**What it does**:
- Loads and preprocesses jet datasets
- Extracts topological features using persistent homology
- Trains classification models
- Evaluates model performance

**Expected Output**: Model checkpoints, classification metrics, and performance visualizations.

---

### Physics-Informed Neural Networks for Diffusion Equation

#### Option 1: Python Scripts

```bash
cd Physics_Informed_Neural_Network_Diffusion_Equation_Sijil_Jose
pip install -r requirements.txt  # If available

# For 1D Gaussian Mixture Model
python flow_de/train_1d_GMM.py

# For 2D Gaussian Mixture Model
python flow_de/train_2d_GMM.py

# For 3D Gaussian Mixture Model
python flow_de/train_3d_GMM.py
```

> **Note**: Uncomment the last line in each training script to run the optimizer.

#### Option 2: Jupyter Notebooks (Recommended for beginners)

```bash
cd "Jupyter Notebooks"
jupyter notebook FlowDE_PINN-1D_GMM.ipynb
# OR
jupyter notebook FlowDE_PINN-2D_GMM.ipynb
# OR
jupyter notebook FlowDE_PINN-3D_GMM.ipynb
```

**What it does**:
- Trains PINN to solve reverse-time diffusion ODE
- Generates samples from target distributions
- Compares PINN solutions with numerical solvers
- Visualizes trajectories and distributions

**Expected Output**: 
- Trained model checkpoints
- Comparison plots between target and generated distributions
- Trajectory visualizations in `Figures/` directory

---

## 🔧 Troubleshooting

### Common Issues Across Projects

#### Installation Problems
**Issue**: PyTorch installation fails or CUDA version mismatch
```bash
# Solution: Install specific PyTorch version
pip install torch==2.0.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

**Issue**: `ModuleNotFoundError` for project-specific packages
```bash
# Solution: Ensure you're in the correct project directory
cd Graph_Representation_Learning_Rushil_Singha  # or other project
pip install -r requirements.txt
```

#### Runtime Issues
**Issue**: CUDA out of memory errors
- Reduce batch size in training scripts
- Use CPU-only mode: `export CUDA_VISIBLE_DEVICES=""`
- Close other GPU-intensive applications

**Issue**: Dataset download failures
- Check internet connection
- For JetNet: datasets auto-download to `jetnet_data/` directory
- Manual download links available in individual project READMEs

#### Performance Issues
**Issue**: Very slow training on CPU
- Expected behavior for deep learning models
- Consider using Google Colab, Kaggle, or cloud GPU services
- Reduce dataset size for testing (modify `num_particles` parameters)

### Project-Specific Help

| Issue Type | Graph Representation | Jet Classification | Physics-Informed NN |
|------------|---------------------|-------------------|-------------------|
| Memory errors | Reduce `BATCH_SIZE` | Use `preprocess_dask.py` | Reduce collocation points |
| Slow convergence | Increase epochs to 200+ | Check data preprocessing | Adjust learning rate |
| Poor results | Try different K values | Verify dataset format | Increase PINN depth |

### Getting Help

1. **Check individual project READMEs** for specific troubleshooting
2. **Open an issue** on GitHub with:
   - Your operating system and Python version
   - Complete error message
   - Steps to reproduce the problem
3. **Join ML4SCI discussions** for community support

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository** on GitHub
2. **Create a new branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** and commit:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```
4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Open a Pull Request** on the main repository

### Contribution Guidelines

- Follow the existing code style and structure
- Add documentation for new features
- Include tests where applicable
- Update the README if you add new functionality
- Reference any related issues in your PR description

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Google Summer of Code (GSoC)** for funding and support
- **ML4SCI Organization** for mentorship and guidance
- **Mentors**: Prof. Harrison Prosper, Prof. Pushpalatha Bhat, Prof. Sergei Gleyzer, and others
- **Contributors**: Rushil Singha, Tanmay Bakshi, Sijil Jose

### Related Links

- [ML4SCI Website](https://ml4sci.org/)
- [GSoC 2025 Projects](https://ml4sci.org/activities/gsoc2025.html)
- [JetNet Dataset](https://huggingface.co/datasets/jetnet)
- [Fast Calorimeter Challenge](https://calochallenge.github.io/homepage/)

---

## 📞 Contact & Support

For questions, issues, or discussions:
- **Open an issue** on this repository
- **Visit** [ML4SCI](https://ml4sci.org/) for more information
- **Check** individual project READMEs for project-specific documentation

---

**Made with ❤️ by the ML4SCI community**

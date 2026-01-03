# JetNet Graph Diffusion Model

A PyTorch/PyTorch-Geometric implementation of a **graph-based diffusion model** for generating realistic jets from the [JetNet dataset](https://huggingface.co/datasets/jetnet).  

This model builds **k-nearest neighbor (kNN) jet graphs**, learns **Chebyshev GCN (ChebNet) embeddings**, trains a **diffusion model in latent space**, and decodes generated samples back into particle-level jets.

---

## 🚀 Features
- kNN graph construction from jet particle clouds  
- Graph encoder using **Chebyshev GCN** (`SimpleChebNet`)  
- Latent **diffusion process** with denoising MLP  
- Jet particle **decoder** network  
- Evaluation with **KL divergence** & **Wasserstein distance**  
- Visualization utilities for jet properties  

---

## ⚙️ Installation

Clone the ML4Sci GENIE repository and navigate to this project directory :

```bash
git clone https://github.com/ML4SCI/GENIE.git
cd GENIE/Graph_Representation_Learning_Rushil_Singha
```
Install dependencies:
```bash
pip install -r requirements.txt
```
## 🏃 Quick Start

After installing dependencies, run:

```bash
python code.py
```
# This script will:

- Encodes jets into latent space

- Runs diffusion training

- Decodes jets back into particle space

- Logs evaluation metrics

- Saves visualizations to results/





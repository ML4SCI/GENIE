# JetNet Graph Diffusion Model

A PyTorch/PyTorch-Geometric implementation of a **graph-based diffusion model** for generating realistic jets from the [JetNet dataset](https://huggingface.co/datasets/jetnet).  

This project builds **k-nearest neighbor (kNN) jet graphs**, learns **Chebyshev GCN (ChebNet) embeddings**, trains a **diffusion model in latent space**, and decodes back into particle-level jets.

---

## 🚀 Features

- kNN graph construction from jet particle clouds  
- Graph encoder using **Chebyshev GCN** (`SimpleChebNet`)  
- Latent **diffusion process** with denoising MLP  
- Jet particle **decoder** network  
- Evaluation using **KL divergence** & **Wasserstein distance**  
- Visualization utilities for jet properties  

---

## ⚙️ Installation

Clone the repository and install dependencies:

 
```sh
git clone https://github.com/your-username/jetnet-graph-diffusion.git
cd jetnet-graph-diffusion
pip install -r requirements.txt
```

requirements.txt
```sh
numpy==1.24.3
torch==2.0.0
torch-geometric
torch-scatter
torch-sparse
torch-cluster
networkx
scikit-learn
jetnet
```

## 📜 Script Overview

The main script performs the following steps:

- Encodes jets into latent space
- Runs diffusion training
- Decodes jets back into particle space
- Logs evaluation metrics
- Saves visualizations to results/ directory

## To run the main workflow:
```sh
python code.py
```
##  Usage

This repository contains a research prototype for graph-based jet generation using latent diffusion models.

##  Note:
Dataset paths, hyperparameters, and output locations may need to be adjusted depending on the local environment and JetNet configuration.

 

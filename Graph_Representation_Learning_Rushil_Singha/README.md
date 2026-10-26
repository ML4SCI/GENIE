# JetNet Graph Diffusion Model

## Overview
This project implements a graph-based diffusion model using PyTorch and PyTorch Geometric to generate realistic particle jets from the JetNet dataset.

The model constructs k-Nearest Neighbor (kNN) graphs from particle data, encodes them using a Graph Neural Network (Chebyshev GCN), applies a diffusion process in latent space, and reconstructs the jets using a decoder network.

---

## Features
- kNN graph construction from particle data  
- Graph encoding using Chebyshev Graph Convolution (ChebNet)  
- Diffusion model in latent space  
- Decoder for reconstructing particle-level jets  
- Evaluation using KL Divergence and Wasserstein Distance  
- Visualization of results  

---

## Project Structure
Graph_Representation_Learning_Rushil_Singha/
├── code.py
├── requirements.txt
├── results/
└── README.md

---

## Installation

### Step 1: Clone the repository
git clone https://github.com/ML4SCI/GENIE.git

cd GENIE/Graph_Representation_Learning_Rushil_Singha

### Step 2: Install dependencies
pip install -r requirements.txt


---

## How to Run

python code.py


---

## Workflow

The script performs the following steps:

1. Loads the JetNet dataset  
2. Converts particle data into kNN graphs  
3. Encodes graphs into latent representations  
4. Applies diffusion process  
5. Trains a denoising model  
6. Decodes latent vectors into particle-level jets  
7. Evaluates results using statistical metrics  
8. Saves output visualizations  

---

## Output

After running the code, the following outputs are generated:

- Generated jet samples  
- Evaluation metrics (KL Divergence, Wasserstein Distance)  
- Visualization plots  

All outputs are saved in the results/ directory.

---

## Requirements
- Python 3.x  
- PyTorch  
- PyTorch Geometric  
- NumPy  
- NetworkX  
- Scikit-learn  
- JetNet  

Install all dependencies using:

pip install -r requirements.txt


---

## Notes
- GPU is recommended for faster training  
- Make sure PyTorch Geometric dependencies are installed correctly  

---

## Contributing
Contributions are welcome. You can improve documentation, optimize code, or add new features.

---

## Acknowledgement
This project is part of the ML4SCI GENIE repository and focuses on applying machine learning techniques in particle physics.
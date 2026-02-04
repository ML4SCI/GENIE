# GENIE

GENIE is a collection of small research and learning projects related to **Machine Learning for Science**, mainly focused on **graph neural networks, jet physics, diffusion models, and physics‑informed neural networks**.

---

## Repository Structure

```
GENIE/
├── Graph_Representation_Learning_Rushil_Singha/
│   ├── code.py
│   ├── README.md
│   └── requirements.txt
│
├── Non_local_Jet_Classification_Tanmay_Bakshi/
│   ├── main.py
│   ├── datasets.py
│   ├── preprocess_dask.py
│   ├── persistent_net-2.ipynb
│   └── readme.md
│
├── Physics_Informed_Neural_Network_Diffusion_Equation_Sijil_Jose/
│   ├── Figures/
│   ├── Jupyter Notebooks/
│   └── Tests/
│
└── README.md
```

---

## Projects

### Graph Representation Learning (JetNet)

* Builds kNN graphs from jet particles
* Uses Chebyshev GCN for graph embeddings
* Trains a latent diffusion model
* Evaluates using KL divergence and Wasserstein distance

Run:

```bash
python code.py
```

---

### Non‑local Jet Classification

* Jet classification using deep learning
* Includes preprocessing and dataset utilities
* Contains experimental notebooks

---

### Physics‑Informed Neural Networks

* Experiments with PINNs for diffusion equations
* Includes notebooks, tests, and figures

---

## Notes

* Each folder is an independent project
* Paths and datasets may need adjustment depending on environment
* Intended for learning and research use

---

## Contributions

Pull requests and documentation improvements are welcome.

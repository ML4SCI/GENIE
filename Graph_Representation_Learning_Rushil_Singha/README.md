# JetNet Graph Diffusion Model

**Author**: Rushil Singha  
**GSoC 2025 Project**: Graph-based diffusion models for realistic jet generation

A PyTorch/PyTorch-Geometric implementation of a **graph-based diffusion model** for generating realistic jets from the [JetNet dataset](https://huggingface.co/datasets/jetnet).  

This project builds **k-nearest neighbor (kNN) jet graphs**, learns **Chebyshev GCN (ChebNet) embeddings**, trains a **diffusion model in latent space**, and decodes back into particle-level jets.

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

### Prerequisites
- Python 3.8+ (tested on 3.9)
- CUDA 11.8+ (for GPU acceleration)
- At least 8GB RAM (16GB recommended)

### Setup

```bash
# Clone and navigate to project
git clone https://github.com/ML4SCI/GENIE.git
cd GENIE/Graph_Representation_Learning_Rushil_Singha

# Install dependencies
pip install -r requirements.txt
```

**Note**: If you encounter PyTorch Geometric installation issues, install manually:
```bash
pip install torch==2.0.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-geometric torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

---

## 🏃‍♂️ Usage

### Basic Run
```bash
python code.py
```

### What the script does:
1. **Downloads JetNet dataset** (~2GB) to `jetnet_data/` directory
2. **Preprocesses jets** - extracts particle features (eta, phi, pt) and masks
3. **Builds kNN graphs** - constructs k=8 nearest neighbor graphs for each jet
4. **Trains ChebNet encoder** - learns 64-dimensional latent representations
5. **Runs diffusion training** - trains denoising model in latent space
6. **Generates synthetic jets** - samples new jets from trained model
7. **Evaluates results** - computes KL divergence and Wasserstein distances
8. **Saves outputs** to `results/` directory

### Expected Runtime
- **CPU**: 3-4 hours
- **GPU (RTX 3080+)**: 45-90 minutes
- **Memory usage**: 6-12GB RAM

### Output Files
```
results/
├── training_logs.txt          # Training progress and losses
├── generated_jets.png         # Comparison plots
├── evaluation_metrics.json    # KL divergence, Wasserstein distances
├── model_checkpoints/         # Saved model weights
└── jet_visualizations/        # Individual jet plots
```

---

## 🔧 Configuration

Key parameters in `code.py`:
```python
# Graph construction
K_NEIGHBORS = 8              # kNN graph connectivity
LATENT_DIM = 64             # Embedding dimension

# Training
BATCH_SIZE = 32             # Adjust based on GPU memory
LEARNING_RATE = 1e-4        # Adam optimizer learning rate
NUM_EPOCHS = 100            # Training epochs
```

---

## 📊 Expected Results

**Good results show:**
- KL divergence < 0.1 for jet mass and pT distributions
- Wasserstein distance < 0.05 for particle multiplicity
- Generated jets visually similar to real jets in eta-phi space

**If results are poor:**
- Increase training epochs (try 200+)
- Adjust learning rate (try 5e-5 or 2e-4)
- Check GPU memory usage (reduce batch size if needed)

---

## 🐛 Troubleshooting

**Common Issues:**

1. **CUDA out of memory**
   ```python
   # Reduce batch size in code.py
   BATCH_SIZE = 16  # or 8
   ```

2. **JetNet download fails**
   ```bash
   # Manual download alternative
   wget https://zenodo.org/record/6975118/files/jetnet.tar.gz
   tar -xzf jetnet.tar.gz
   ```

3. **PyTorch Geometric errors**
   ```bash
   # Reinstall with specific CUDA version
   pip uninstall torch-geometric torch-scatter torch-sparse
   pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
   ```

4. **Slow training on CPU**
   - Expected behavior - consider using Google Colab or cloud GPU
   - Reduce dataset size by modifying `num_particles=50` in `load_jetnet_data()`

---

## 📈 Performance Tips

- **GPU acceleration**: Ensure CUDA is properly installed
- **Memory optimization**: Use gradient checkpointing for large models
- **Faster convergence**: Try learning rate scheduling
- **Better results**: Experiment with different graph construction methods (radius graphs, etc.)

---

## 🤝 Contributing

Found a bug or want to improve the model? 
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📚 References

- [JetNet Dataset](https://huggingface.co/datasets/jetnet)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [Chebyshev Graph Convolutions](https://arxiv.org/abs/1606.09375)





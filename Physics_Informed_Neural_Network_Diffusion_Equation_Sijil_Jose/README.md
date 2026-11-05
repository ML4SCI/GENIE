# PINNDE: Physics Informed Neural Networks for Diffusion Equation

**Author**: Sijil Jose  
**GSoC 2025 Project**: Fast sampling via reverse-time diffusion using PINNs

![ML4Sci@GSoC2024](https://miro.medium.com/v2/resize:fit:1100/format:webp/0*8KAp7eW2atsaRwdS.jpeg)

## 🎯 Project Overview

This project develops a proof-of-concept for building fast and reliable samplers by solving reverse-time diffusion equations using Physics-Informed Neural Networks (PINNs). PINNDE combines the high accuracy of diffusion models with the flexibility of physics-informed neural networks to sample from complicated and intractable distributions in multiple dimensions.

### ✅ What Was Accomplished

- ✅ Implemented accurate q-function approximation for reverse-time diffusion ODE
- ✅ Developed multiple PINN architectures for solving diffusion equations  
- ✅ Validated on 1D, 2D, and 3D Gaussian Mixture Models
- ✅ Tested different optimization strategies for PINN training
- 🚧 Integration with Fast Calorimeter Challenge 2022 (in progress)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 1.8+
- NumPy, Matplotlib, SciPy
- Jupyter (for notebooks)

### Installation
```bash
cd Physics_Informed_Neural_Network_Diffusion_Equation_Sijil_Jose

# Install dependencies (create requirements.txt if needed)
pip install torch numpy matplotlib scipy jupyter corner
```

### Running the Code

#### Option 1: Python Scripts (Advanced Users)
```bash
# 1D Gaussian Mixture Model
python flow_de/train_1d_GMM.py

# 2D Gaussian Mixture Model  
python flow_de/train_2d_GMM.py

# 3D Gaussian Mixture Model
python flow_de/train_3d_GMM.py
```
**Note**: Uncomment the last line in each script to run the optimizer.

#### Option 2: Jupyter Notebooks (Recommended for Beginners)
```bash
cd "Jupyter Notebooks"

# Start with 1D case
jupyter notebook FlowDE_PINN-1D_GMM.ipynb

# Then try 2D and 3D
jupyter notebook FlowDE_PINN-2D_GMM.ipynb
jupyter notebook FlowDE_PINN-3D_GMM.ipynb
```

#### Option 3: Numerical Solver Demo
```bash
cd FlowDE
jupyter notebook FlowDE.ipynb  # 1D numerical solution demo
```

---

## 📁 Project Structure

```
Physics_Informed_Neural_Network_Diffusion_Equation_Sijil_Jose/
├── flow_de/                          # Core implementation
│   ├── flow_de.py                    # qVectorField and FlowDE classes
│   ├── gendata.py                    # Data generation utilities
│   ├── networks_1d.py                # 1D PINN architectures
│   ├── networks_2d.py                # 2D PINN architectures  
│   ├── networks_3d.py                # 3D PINN architectures
│   └── train_*d_GMM.py              # Training scripts
├── Jupyter Notebooks/                # Interactive examples
│   ├── FlowDE_PINN-1D_GMM.ipynb     # 1D demo with explanations
│   ├── FlowDE_PINN-2D_GMM.ipynb     # 2D demo with visualizations
│   └── FlowDE_PINN-3D_GMM.ipynb     # 3D demo with corner plots
├── Figures/                          # Result visualizations
├── Tests/                            # Unit tests (to be expanded)
└── slides_docs/                      # Project documentation
```

---

## 🎯 Expected Results

### Training Process
- **Runtime**: 30 minutes - 2 hours depending on dimension and complexity
- **Convergence**: Loss should decrease steadily over epochs
- **Memory**: 2-4GB RAM typically sufficient

### Output Files
- **Model checkpoints**: `*.pth` files with trained parameters
- **Visualizations**: Comparison plots in `Figures/` directory
- **Trajectories**: ODE solution paths (PINN vs numerical solver)

### Success Indicators
✅ **Good Results:**
- Generated samples match target distribution visually
- Low residual loss for physics constraints
- Smooth trajectory plots without oscillations

⚠️ **Poor Results May Indicate:**
- Insufficient training epochs (try 5000+)
- Learning rate too high/low (try 1e-4 to 1e-3)
- Network architecture needs adjustment

---

## 🔬 Key Results Achieved

### Distribution Matching
![Trained Distributions](Figures/trained_distributions.png)

*Comparison of target distributions (black) vs PINNDE samples (blue) for 1D, 2D, and 3D cases*

### Trajectory Validation  
![Normal Trajectories](Figures/normal_trajectories.png)
![Uniform Trajectories](Figures/uniform_trajectories.png)

*PINN solutions (black) vs numerical Runge-Kutta solver (green) showing excellent agreement*

---

## 🛠️ Troubleshooting

**Training doesn't converge:**
- Increase number of collocation points
- Adjust learning rate (try 5e-4)
- Check physics loss weighting

**Memory issues:**
- Reduce batch size in training scripts
- Use CPU instead of GPU for smaller problems

**Poor sample quality:**
- Increase training epochs
- Verify target distribution implementation
- Check boundary conditions

---

## 📚 Documentation & Resources

### Project Links
- [Official Repository](https://github.com/ML4SCI/GENIE/tree/main/Physics_Informed_Neural_Network_Diffusion_Equation_Sijil_Jose)
- [Author's Fork](https://github.com/sijil-jose/GENIE/blob/PINNDE/Physics_Informed_Neural_Network_Diffusion_Equation_Sijil_Jose/README.md)
- [Mid-term Blog](https://medium.com/@sijiljose.999/gsoc-2025-with-ml4sci-part-i-physics-informed-neural-network-for-diffusion-equation-pinnde-491d46a5b84d)

### Academic References
- [Original ML4SCI Proposal](https://ml4sci.org/gsoc/2025/proposal_GENIE5.html)
- [GSoC Abstract](https://summerofcode.withgoogle.com/programs/2025/projects/uGmyAV1q)
- [Fast Calorimeter Challenge](https://calochallenge.github.io/homepage/)

---

## 🔮 Future Work

- Complete Fast Calorimeter Challenge integration
- Explore advanced PINN architectures (DeepONet, etc.)
- Add comprehensive unit test coverage
- Benchmark against other sampling methods

---

## 🙏 Acknowledgments

**Mentors**: Prof. Harrison Prosper, Prof. Pushpalatha Bhat, Prof. Sergei Gleyzer  
**Organization**: [ML4SCI](https://ml4sci.org/) - Machine Learning for Science  
**Program**: Google Summer of Code 2025

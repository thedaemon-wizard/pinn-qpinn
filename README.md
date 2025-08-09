# Quantum Physics-Informed Neural Networks (QPINNs) for 3D Heat Equation

A comprehensive benchmark implementation of Quantum Physics-Informed Neural Networks (QPINNs) for solving the 3D heat conduction equation forward problem, featuring state-of-the-art quantum circuit generation using GQE-GPT integration, multi-objective optimization, and comparison with enhanced classical PINNs.

## Overview

This repository provides highly optimized implementations of both QPINNs and classical PINNs with cutting-edge techniques:

### Quantum PINN Features:
- **GQE-GPT Integration**: Generative Quantum Eigensolver enhanced with GPT-based circuit generation
- **Multi-Objective Optimization**: NSGA-II algorithm with dynamic circuit updates
- **Unsupervised Quantum Energy Estimation**: Novel approach for noise-aware energy estimation
- **Hardware-Efficient Design**: Optimized for NISQ (Noisy Intermediate-Scale Quantum) devices
- **Parallel Processing**: Efficient batch evaluation for large-scale problems

### Classical PINN Features:
- **Fourier Neural Operator (FNO) Integration**: 3D spectral convolution layers for enhanced expressivity
- **Temporal Attention Mechanism**: Improved time dynamics modeling
- **Multi-Scale Fourier Features**: Better temporal and spatial resolution
- **Hard Boundary Constraints**: Smooth distance function enforcement
- **Memory-Efficient FNO Mode**: Optimized for GPU memory constraints

## Key Features

### 1. Advanced Quantum Circuit Generation
- **GQE-GPT Generator**: Combines rule-based and GPT-based circuit generation
- **Dynamic Circuit Updates**: Automatically adapts circuit architecture during optimization
- **Multi-objective Bayesian Optimization**: Optimizes 9 circuit quality metrics simultaneously:
  - Hardware Efficiency
  - Noise Resilience
  - Expressivity
  - Error Mitigation Compatibility
  - Trainability
  - Entanglement Capability
  - Circuit Depth Efficiency
  - Parameter Efficiency
  - Energy Estimation Quality

### 2. Enhanced Classical PINN Architecture
- **Fourier Neural Operator (FNO)**: 3D spectral convolution for capturing multi-scale features
- **Temporal Attention**: Self-attention mechanism for temporal dynamics
- **Multi-Scale Fourier Feature Mapping**: Separate coarse/fine spatial and slow/fast temporal features
- **Hard Constraints**: Smooth boundary enforcement using tanh-based distance functions
- **Memory-Efficient Implementation**: Chunked processing for large-scale problems

### 3. Multi-Objective Optimization
Both QPINN and classical PINN use NSGA-II (Non-dominated Sorting Genetic Algorithm II) with:
- **Multiple Objectives**:
  - Classical PINN: Initial condition, Peak value, Boundary condition, PDE residual
  - QPINN: Initial condition, Peak value, Boundary condition, PDE residual, Trace loss
- **REX Crossover**: Real-coded crossover operator
- **Dynamic Parameter Bounds**: Adaptive scaling based on network/circuit complexity
- **Pareto Front Tracking**: Complete history of non-dominated solutions

### 4. Problem-Agnostic Design
The implementation uses scientifically grounded, problem-agnostic transformations:
- Trainable embedding functions for spatial and temporal features
- Learnable measurement combination weights (QPINN)
- Adaptive activation functions

### 5. Noise Mitigation (QPINN)
- Zero-noise extrapolation for error mitigation
- Noise-aware circuit evaluation
- Support for different noise models (light, realistic, heavy)

## Algorithm Details

### Classical PINN with FNO

The enhanced PINN implementation combines several state-of-the-art techniques:

1. **Fourier Neural Operator Layer**
   ```
   SpectralConv3d: Performs convolution in Fourier space
   - Input → FFT → Multiply with learnable weights → IFFT → Output
   - Captures global features efficiently
   - Handles multiple frequency modes
   ```

2. **Temporal Attention Mechanism**
   ```
   Based on Vaswani et al. (2017) self-attention:
   - Query, Key, Value projections from hidden states
   - Scaled dot-product attention
   - Residual connections
   ```

3. **Multi-Scale Feature Engineering**
   - Spatial features: Coarse (σ=5.0) and Fine (σ=20.0) scales
   - Temporal features: Slow (2π/T) and Fast (10π/T) frequencies
   - Physics-aware features: Diffusion scale √(t/T + ε)

4. **Hard Boundary Constraints**
   ```
   u(x,y,z,t) = D(x,y,z) × N(x,y,z,t)
   where D is smooth distance function: ε·tanh(d_min/ε)
   ```

### Core Algorithm: GQE-QPINNs

The implementation follows a hierarchical optimization approach:

1. **Circuit Generation Phase**
   - Initialize with KetGPT dataset or rule-based ansatz
   - Use GPT model to generate circuit candidates
   - Evaluate using multi-objective Bayesian optimization

2. **Parameter Optimization Phase**
   - NSGA-II optimization with objectives:
     - Initial condition loss
     - Peak value loss
     - Boundary condition loss
     - PDE residual loss
     - Trace loss (quantum state normalization)

3. **Dynamic Update Phase**
   - Monitor performance improvement
   - Trigger circuit architecture updates when improvement stagnates
   - Use context-aware generation for new circuits

### Mathematical Formulation

The 3D heat equation being solved:
```
∂u/∂t = α∇²u
```

With:
- Initial condition: Gaussian distribution centered at domain center
- Boundary condition: u = 0 at all boundaries
- Domain: [0,L]³ × [0,T]

### NSGA-II Multi-Objective Optimization

Both PINN and QPINN implementations use NSGA-II with:
- **Population-based search**: Maintains diversity of solutions
- **Non-dominated sorting**: Identifies Pareto-optimal solutions
- **Crowding distance**: Preserves solution diversity
- **REX crossover**: Effective for real-valued parameters
- **Unified configuration**: Ensures fair comparison between methods

## Key References

### Classical PINN with FNO
- Li et al. "Fourier Neural Operator for Parametric PDEs" (2023)
- Wang et al. "When and why PINNs fail to train" (2022)
- Krishnapriyan et al. "Characterizing possible failure modes in PINNs" (2021)
- Vaswani et al. "Attention is All You Need" (2017)
- Lu et al. "NSGA-PINN: A Multi-Objective Optimization Method for Physics-Informed Neural Network Training" (2023)
- Ma et al. "A comprehensive survey on NSGA-II for multi-objective optimization and applications" (2023)

### Quantum Physics-Informed Neural Networks
- Trahan et al. "Quantum Physics-Informed Neural Networks" Entropy 26(8):649 (2024)
- Panichi et al. "Quantum physics informed neural networks for multi-variable PDEs" arXiv:2503.12244 (2025)
- "Trainable embedding quantum physics informed neural networks" Scientific Reports (2025)

### Generative Quantum Eigensolver (GQE)
- Nakaji et al. "The generative quantum eigensolver (GQE)" arXiv:2401.09253 (2024)
- "Generative quantum combinatorial optimization by conditional-GQE" arXiv:2501.16986 (2025)
- "QAOA-GPT: Efficient Generation of Adaptive and Regular QAOA Circuits" arXiv:2504.16350 (2025)

### Quantum Circuit Learning
- Mitarai et al. "Quantum circuit learning" Phys. Rev. A 98, 032309 (2018)
- Abbas et al. "The power of quantum neural networks" Nat Comput Sci 1, 403-409 (2021)
- Schuld et al. "Evaluating analytic gradients on quantum hardware" Phys. Rev. A 99, 032331 (2019)

### Error Mitigation
- Temme et al. "Error mitigation for short-depth quantum circuits" Phys. Rev. Lett. 119, 180509 (2017)
- Endo et al. "Practical Quantum Error Mitigation for Near-Future Applications" Phys. Rev. X 11, 031057 (2021)
- Li & Benjamin "Efficient Variational Quantum Simulator Incorporating Active Error Minimization" Phys. Rev. X 7, 021050 (2017)

### Optimization and Trainability
- McClean et al. "Barren plateaus in quantum neural network training landscapes" Nature Communications 9, 4812 (2018)
- Cerezo et al. "Cost function dependent barren plateaus in shallow parametrized quantum circuits" Nature Communications 12, 1791 (2021)
- Larocca et al. "Diagnosing Barren Plateaus with Tools from Quantum Optimal Control" Quantum 6, 824 (2022)

### Hardware Efficiency
- Kandala et al. "Hardware-efficient variational quantum eigensolver" Nature 549, 242-246 (2017)
- "Trainability enhancement of parameterized quantum circuits" Phys. Rev. Applied 22, 054005 (2024)

### Datasets
- Apak et al. "KetGPT – Dataset Augmentation of Quantum Circuits using Transformers" arXiv:2402.13352 (2024)

### Additional Resources
- NVIDIA Technical Blog "Advancing Quantum Algorithm Design with GPTs" (2024)
- "Guaranteed efficient energy estimation using ShadowGrouping" Nature Communications 15, 799 (2025)
- Meyer et al. "Fisher Information in Noisy Intermediate-Scale Quantum Applications" Quantum 5, 539 (2021)

## Requirements

- Python 3.8+
- PennyLane 0.28+
- PyTorch 2.0+
- NumPy
- SciPy
- Matplotlib
- Transformers (Hugging Face)
- BoTorch (for Bayesian optimization)
- Scikit-learn
- NSGA2 optimizer (C++ extension)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage - Quantum PINN:
```python
# Initialize QPINN
qpinn = GQEQuantumPINN(
    n_qubits=6,
    backend='default.mixed',
    shots=None,  # Use statevector for simulation
    noise_model=None,
    use_parallel=True,
    use_gpt_circuit_generation=True
)

# Train with NSGA-II
params, loss_history, training_time = qpinn.train_with_nsga2(n_samples=1500)

# Evaluate model
predictions = qpinn.evaluate()
```

### Basic Usage - Classical PINN with FNO:
```python
# Initialize Enhanced PINN
pinn = PINN(
    layers=[5, 128, 256, 256, 128, 1],
    use_hard_constraints=True,
    boundary_epsilon=0.1,
    fourier_features=True,
    num_fourier_features=64,
    use_fno=True,
    fno_modes=(8, 8, 8),
    use_temporal_attention=True,
    fno_memory_efficient=True
)

# Train with NSGA-II
state_dict, losses, training_time = pinn.train_with_nsga2(n_samples=10000)

# Evaluate model
predictions = evaluate_pinn_nsga2(pinn)
```

## Configuration

Key parameters in the implementation:
- `n_qubits`: Number of qubits (default: 6) [QPINN]
- `layers`: Neural network architecture [PINN]
- `use_fno`: Enable Fourier Neural Operator [PINN]
- `use_temporal_attention`: Enable temporal attention mechanism [PINN]
- `fno_memory_efficient`: Use memory-efficient FNO implementation [PINN]
- `pinn_epochs`: PINN training epochs (for standard training)
- `qnn_epochs`: QPINN training epochs (for standard training)
- `N_PARALLEL_DEVICES`: Number of parallel devices
- `alpha`: Thermal diffusivity
- `L`: Cube side length
- `T`: Final time

## Output

The code generates comprehensive results including:

### For QPINN:
- Quantum circuit diagrams
- Circuit quality metrics
- GQE optimization history
- GPT generation statistics
- Gate evolution heatmaps
- Energy estimation analysis

### For Classical PINN:
- Network architecture visualization
- FNO feature analysis
- Attention weight visualization
- Multi-scale feature maps

### For Both:
- Optimization history plots
- Pareto front visualizations
- Performance metrics
- Detailed reports in JSON/CSV/LaTeX formats
- Comparative analysis between methods

## Performance Comparison

The implementation provides fair comparison between classical PINN and QPINN using:
- Identical NSGA-II optimization framework
- Unified progress intervals
- Normalized objective functions
- Comprehensive metrics:
  - Mean Squared Error (MSE)
  - Relative L2 error
  - Boundary condition satisfaction
  - Initial condition accuracy
  - Conservation properties
  - Computational efficiency

## Citation

If you use this code in your research, please cite the relevant papers listed in the references section.

## License

This implementation is provided for research purposes. Please check individual paper licenses for specific algorithm implementations.

## Acknowledgments

This implementation builds upon numerous research contributions in quantum machine learning, QPINNs, classical PINNs with advanced architectures, and multi-objective optimization. Special thanks to all researchers whose work is referenced in this implementation.
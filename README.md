# GQE-GPT-QPINN: Generative Quantum Eigensolver with GPT for Quantum Physics-Informed Neural Networks

## Overview

This repository implements a novel approach to solving the 3D heat equation using both classical Physics-Informed Neural Networks (PINNs) and Quantum Physics-Informed Neural Networks (QPINNs) enhanced with Generative Quantum Eigensolver (GQE) and GPT-based circuit generation. The quantum implementation features an innovative optimization strategy using Real-Coded Genetic Algorithm (RCGA).

## Key Features

### 🧠 Classical PINN
- Deep neural network implementation for solving 3D heat equation
- Automatic differentiation for PDE residual computation
- Boundary condition enforcement
- PyTorch-based implementation with GPU acceleration

### 🌌 GQE-GPT-QPINN (Quantum Implementation)
- **GPT-based Quantum Circuit Generation**: Uses a transformer model to generate optimal quantum circuits
- **Generative Quantum Eigensolver (GQE)**: Novel approach that optimizes circuit structure rather than just parameters
- **Hardware-aware optimization**: Designed for real quantum devices with noise resilience
- **Parallel quantum device simulation**: Efficient batch processing across multiple quantum devices

### 🧬 RCGA Optimization
- **Real-Coded Genetic Algorithm** implemented in C++ for high performance
- **Latin Hypercube Sampling (LHS)**: Ensures well-distributed initial population
- **REX Crossover**: Real-valued crossover operator with expansion factor
- **JGG Selection**: Just Generation Gap selection strategy
- Python bindings via pybind11

## Requirements
- Python ≥ 3.12

- C++17 compiler (GCC ≥ 10 / Clang ≥ 12 / MSVC ≥ 19.3)

- CMake≥ 3.25

- CUDA≥12.6(If using GPU)

- PyPl package

```
numpy>=2.2.6
matplotlib>=3.10.3
torch>=2.7.0
pennylane>=0.41.1
transformers>=4.52.0
scipy>=1.15.0
pybind11>=2.13.6
setuptools>=80.7.0
```

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/thedaemon-wizard/pinn-qpinn.git
cd pinn-qpinn.git
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Build RCGA optimizer
```bash
pip install -U  . rcga_optimizer
```

## Usage

### Basic Usage
```python
python pinns_d3.py
```

### Configuration Options

The main script supports various configuration parameters:

```python
# Problem parameters
alpha = 0.01  # Thermal diffusivity
L = 1.0       # Cube side length
T = 1.0       # Final time

# Discretization parameters
nx, ny, nz = 20, 20, 20  # Spatial divisions
nt = 20                  # Time divisions

# Training parameters
pinn_epochs = 2000       # PINN epochs
qnn_epochs = 2000        # QPINN epochs
```

### Running with RCGA Optimization
```python
# Enable RCGA optimization (default if available)
qsolver = GQEQuantumPINN(
    n_qubits=6,
    backend='default.mixed',
    shots=1000,
    noise_model='realistic',
    use_parallel=True,
    use_gpt_circuit_generation=True,
    use_rcga=True  # Enable RCGA optimization
)
```

## Implementation Details

### Heat Equation
The code solves the 3D heat equation:
```
∂u/∂t = α∇²u
```
with Dirichlet boundary conditions (u = 0 on all boundaries) and Gaussian initial condition.

### GQE-GPT Architecture
1. **GPT Model**: Generates quantum circuit sequences as tokens
2. **Circuit Templates**: Hardware-efficient ansätze with noise resilience
3. **Adaptive Optimization**: Switches between RCGA, SPSA, and Adam based on hardware constraints

### RCGA Implementation
- **Population Size**: 50 individuals (configurable)
- **REX Parameters**: ξ = 1.2 (expansion factor)
- **JGG Parameters**: 3 parents, 10 offspring per generation
- **Termination**: Maximum generations or convergence criteria

## Results

The implementation produces:
- Comparative visualizations of PINN vs GQE-GPT-QPINN solutions
- Error analysis over time
- Boundary condition satisfaction metrics
- Training loss curves
- Performance benchmarks

Output files(at results directory):
- `heat_equation_comparison_gqe_gpt.png`: Solution comparison
- `heat_equation_profile_comparison_gqe_gpt.png`: 1D temperature profiles
- `heat_equation_error_analysis_gqe_gpt.png`: Error metrics
- `heat_equation_boundary_analysis_gqe_gpt.png`: Boundary condition analysis

## Performance

Typical performance characteristics:
- **PINN**: ~100-200 seconds training time(benchmark using CUDA with NVIDIA RTX A2000 12GB) , MSE ~1e-5
- **GQE-GPT-QPINN**: ~1-2 hours training time (with quantum simulation and benchmark i5-13600K 4core CPU)
- **RCGA Convergence**: 500 generations typical for good solutions

## Citation

If you use this code in your research, please cite:

```bibtex
@article{nakaji2024gqe,
  title={The generative quantum eigensolver (GQE) and its application for ground state search},
  author={Nakaji, Kouhei and others},
  journal={arXiv preprint arXiv:2401.09253},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on the GQE algorithm proposed by Nakaji et al.
- GPT architecture inspired by nanoGPT implementation
- RCGA implementation follows standard real-coded genetic algorithm principles

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].
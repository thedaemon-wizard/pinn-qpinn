# GQE-GPT-QPINN: Generative Quantum Eigensolver with GPT for Quantum Physics-Informed Neural Networks

## Overview

This repository implements a novel approach to solving the 3D heat equation using both classical Physics-Informed Neural Networks (PINNs) and Quantum Physics-Informed Neural Networks (QPINNs) enhanced with Generative Quantum Eigensolver (GQE) and GPT-based circuit generation. The quantum implementation features innovative optimization strategies including Real-Coded Genetic Algorithm (RCGA), NSGA-II multi-objective optimization, and unsupervised quantum energy estimation with comprehensive circuit visualization capabilities.

## Key Features

### 🧠 Classical PINN
- Deep neural network implementation for solving 3D heat equation
- Automatic differentiation for PDE residual computation
- Boundary condition enforcement with improved accuracy
- PyTorch-based implementation with GPU acceleration

### 🌌 GQE-GPT-QPINN (Quantum Implementation)
- **GPT-based Quantum Circuit Generation**: Uses a transformer model to generate optimal quantum circuits
- **Generative Quantum Eigensolver (GQE)**: Novel approach that optimizes circuit structure rather than just parameters
- **Unsupervised Quantum Energy Estimation**: New feature for data-driven energy estimation without labeled data
- **Hardware-aware optimization**: Designed for real quantum devices with enhanced noise resilience
- **Parallel quantum device simulation**: Efficient batch processing across multiple quantum devices
- **Circuit Novelty Tracking**: Promotes diversity in circuit generation
- **Comprehensive Circuit Visualization**: Detailed quantum circuit diagrams and performance metrics

### 🧬 Advanced Optimization Methods
- **NSGA-II Multi-Objective Optimization**: Simultaneously optimizes multiple objectives (initial conditions, boundary conditions, data fitting, PDE residuals, peak values)
- **Real-Coded Genetic Algorithm (RCGA)**: C++ implementation for high performance
- **Latin Hypercube Sampling (LHS)**: Ensures well-distributed initial population
- **REX Crossover**: Real-valued crossover operator with expansion factor
- **JGG Selection**: Just Generation Gap selection strategy
- **Zero-Noise Extrapolation**: Error mitigation technique for noisy quantum devices

### 📊 Enhanced Visualization and Analysis
- **Quantum Circuit Diagrams**: Visual representation of GQE-generated circuits
- **Performance Metrics**: Radar charts for circuit characteristics
- **Evolution Tracking**: RCGA and NSGA-II optimization progress visualization
- **Pareto Front Visualization**: 3D and 2D views of multi-objective optimization results
- **Novelty Evolution Plots**: Track circuit diversity over time
- **Hypervolume Evolution**: Monitor multi-objective optimization quality
- **Detailed Circuit Information**: JSON and text format circuit specifications

## Requirements
- Python ≥ 3.12
- C++17 compiler (GCC ≥ 10 / Clang ≥ 12 / MSVC ≥ 19.3)
- CMake ≥ 3.25
- CUDA ≥ 12.6 (If using GPU)
- PyPI packages:

```
numpy>=2.2.6
matplotlib>=3.10.3
torch>=2.7.0
pennylane>=0.41.1
transformers>=4.52.0
scipy>=1.15.0
pybind11>=2.13.6
setuptools>=80.7.0
scikit-learn>=1.7.0  # For unsupervised learning features
```

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/thedaemon-wizard/pinn-qpinn.git
cd pinn-qpinn
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Build optimization modules
```bash
pip install -U . rcga_optimizer
pip install -U . nsga2_optimizer  # For multi-objective optimization
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

# Parallel processing parameters
N_PARALLEL_DEVICES = min(4, cpu_count() // 2)
USE_PARALLEL_TRAINING = True
```

### Running with Different Optimization Methods

#### RCGA Optimization (Default for hardware mode)
```python
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

#### NSGA-II Multi-Objective Optimization
```python
# Train with NSGA-II if available
if NSGA2_AVAILABLE:
    circuit_params, loss_history, training_time = qsolver.train_with_nsga2(n_samples=1500)
else:
    circuit_params, loss_history, training_time = qsolver.train(n_samples=1500)
```

### Accessing Enhanced Features

#### Circuit Visualization and Analysis
```python
# After training, visualize the quantum circuit
circuit_image_path = qsolver.visualize_quantum_circuit('results/')

# Save detailed circuit information
json_path, summary_path = qsolver.save_circuit_information('results/')

# Visualize circuit performance metrics
metrics_path = qsolver.visualize_circuit_metrics('results/')

# Visualize GQE generation process
report_path = qsolver.visualize_gqe_generation_process('results/')

# Create GQE optimization animation
qsolver.save_gqe_animation('results/')
```

#### Novelty Analysis
```python
# Get novelty statistics
novelty_stats = qsolver.gqe_generator.get_novelty_statistics()
print(f"Mean novelty: {novelty_stats['mean_novelty']:.3f}")
print(f"Recent trend: {novelty_stats['recent_trend']:.3f}")

# Visualize novelty evolution
qsolver.gqe_generator.visualize_novelty_evolution('results/')
```

## Implementation Details

### Heat Equation
The code solves the 3D heat equation:
```
∂u/∂t = α∇²u
```
with Dirichlet boundary conditions (u = 0 on all boundaries) and Gaussian initial condition. The boundary condition handling has been improved for better accuracy near domain edges.

### GQE-GPT Architecture
1. **GPT Model**: Generates quantum circuit sequences as tokens
2. **Circuit Templates**: Hardware-efficient ansätze with noise resilience
3. **Adaptive Optimization**: Switches between NSGA-II, RCGA, SPSA, and Adam based on hardware constraints
4. **Circuit Evaluation**: Comprehensive metrics for noise resilience, hardware efficiency, and expressivity
5. **Novelty Tracking**: Ensures diverse circuit exploration

### Unsupervised Quantum Energy Estimation
- **Quantum Feature Extraction**: Uses multiple measurement bases
- **Zero-Noise Extrapolation**: Mitigates hardware noise effects
- **Clustering-based Energy Estimation**: Groups similar quantum states
- **Adaptive Learning**: Updates estimates based on measurement results

### NSGA-II Implementation
- **Objectives**: 5 simultaneous objectives (initial condition, peak value, boundary condition, data fitting, PDE residual)
- **Population Size**: 100 individuals (configurable)
- **REX Crossover**: ξ = 1.2 (expansion factor)
- **Crowding Distance**: Equidistant selection for diversity
- **Batch Evaluation**: Parallel evaluation support

### RCGA Implementation
- **Population Size**: 50 individuals (configurable)
- **REX Parameters**: ξ = 1.2 (expansion factor)
- **JGG Parameters**: 3 parents, 10 offspring per generation
- **Termination**: Maximum generations or convergence criteria
- **Progress Reporting**: Detailed statistics every 50 generations

### Noise Models
Three noise levels supported:
- **Light**: Minimal noise for near-term devices
- **Realistic**: Typical NISQ device noise levels
- **Heavy**: Stress testing with high noise

## Results

The implementation produces:
- Comparative visualizations of PINN vs GQE-GPT-QPINN solutions
- Multi-objective optimization results with Pareto fronts
- Circuit novelty and diversity analysis
- Error analysis over time with improved boundary handling
- Training loss curves for multiple objectives
- Performance benchmarks
- Quantum circuit diagrams and specifications

### Output Files (in results directory):

#### Solution Comparison
- `heat_equation_comparison_gqe_gpt.png`: Solution comparison
- `heat_equation_profile_comparison_gqe_gpt.png`: 1D temperature profiles
- `heat_equation_error_analysis_gqe_gpt.png`: Error metrics
- `heat_equation_boundary_analysis_gqe_gpt.png`: Boundary condition analysis

#### Quantum Circuit Information
- `gqe_quantum_circuit.png`: Visual quantum circuit diagram
- `gqe_circuit_text.txt`: PennyLane text representation
- `gqe_circuit_info.json`: Detailed circuit specification in JSON
- `gqe_circuit_summary.txt`: Human-readable circuit summary
- `gqe_circuit_latex.tex`: LaTeX circuit description (for publications)
- `gqe_circuit_metrics.png`: Circuit performance metrics visualization

#### Optimization Results
- `gqe_rcga_evolution.png`: RCGA optimization progress (if RCGA used)
- `nsga2_optimization_results.json`: NSGA-II detailed results
- `nsga2_pareto_fronts.csv`: Pareto front evolution data
- `nsga2_pareto_front_3d.png`: 3D visualization of final Pareto front
- `nsga2_objectives_evolution.png`: Evolution of all objectives
- `nsga2_hypervolume_evolution.png`: Hypervolume metric evolution
- `nsga2_diversity_evolution.png`: Population diversity metrics

#### GPT and Novelty Analysis
- `gpt_generation_history.json`: GPT model generation history
- `novelty_evolution.png`: Circuit novelty score evolution
- `gqe_gpt_statistics.png`: GPT generation statistics
- `gqe_gate_evolution_heatmap.png`: Gate type evolution heatmap
- `gqe_optimization_animation.gif`: Animated optimization process

## Performance

Typical performance characteristics:
- **PINN**: ~100-200 seconds training time (benchmark using CUDA with NVIDIA RTX A2000 12GB), MSE ~1e-5
- **GQE-GPT-QPINN with RCGA**: ~1-2 hours training time (with quantum simulation on i5-13600K 4core CPU)
- **GQE-GPT-QPINN with NSGA-II**: ~2-3 hours for comprehensive multi-objective optimization
- **RCGA Convergence**: 500 generations typical for good solutions
- **NSGA-II Convergence**: 100-200 generations with stable Pareto front
- **Circuit Generation**: GPT model generates circuits with 20-50 gates typically
- **Novelty Score**: Maintains >0.3 average novelty throughout optimization

## Advanced Features

### Multi-Objective Trade-offs
The NSGA-II implementation allows exploring trade-offs between:
- Initial condition accuracy
- Peak value preservation
- Boundary condition satisfaction
- Data fitting quality
- PDE residual minimization

### Error Mitigation
- Zero-noise extrapolation for hardware noise mitigation
- Richardson extrapolation with configurable noise scaling factors
- Readout error correction
- Dynamic decoupling compatibility

### Circuit Diversity
- Novelty scoring based on structural similarity
- Diversity bonuses in fitness evaluation
- Exploration rate adaptation
- Elite preservation with mutation

## Citation

If you use this code in your research, please cite:

```bibtex
@article{nakaji2024gqe,
  title={The generative quantum eigensolver (GQE) and its application for ground state search},
  author={Nakaji, Kouhei and others},
  journal={arXiv preprint arXiv:2401.09253},
  year={2024}
}

@article{mitarai2018quantum,
  title={Quantum circuit learning},
  author={Mitarai, Kosuke and others},
  journal={Physical Review A},
  volume={98},
  number={3},
  pages={032309},
  year={2018}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on the GQE algorithm proposed by Nakaji et al.
- Quantum circuit learning concepts from Mitarai et al.
- GPT architecture inspired by nanoGPT implementation
- NSGA-II implementation follows Deb et al.'s algorithm
- Zero-noise extrapolation based on Li & Benjamin's work
- Circuit visualization inspired by quantum circuit diagram standards

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Areas of particular interest:
- Additional error mitigation techniques
- Support for more quantum backends (IonQ, Rigetti, etc.)
- Enhanced multi-objective optimization strategies
- Improved unsupervised learning methods
- Performance improvements for large-scale problems

## Troubleshooting

### Common Issues

1. **RCGA/NSGA2 optimizer not available**: Ensure C++ compiler is properly installed and run the build commands again
2. **Circuit visualization errors**: Check matplotlib and required fonts are installed
3. **Memory issues with large circuits**: Reduce batch size or use sequential evaluation
4. **Novelty tracking memory growth**: Limit history size in configuration
5. **Multi-objective convergence issues**: Adjust population size or crowding distance type

### Performance Tips

1. **For hardware simulation**: Use `shots=1000` and `noise_model='realistic'`
2. **For faster training**: Enable parallel processing with `use_parallel=True`
3. **For better solutions**: Use NSGA-II for exploring trade-offs
4. **For circuit diversity**: Monitor novelty scores and adjust exploration rate

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.
# GQE-GPT-QPINN: Generative Quantum Eigensolver with GPT for Quantum Physics-Informed Neural Networks

## Overview

This repository implements a novel approach to solving the 3D heat equation using both classical Physics-Informed Neural Networks (PINNs) and Quantum Physics-Informed Neural Networks (QPINNs) enhanced with Generative Quantum Eigensolver (GQE) and GPT-based circuit generation. The quantum implementation features an innovative optimization strategy using Real-Coded Genetic Algorithm (RCGA), NSGA-II multi-objective optimization, AI-enhanced energy prediction, and comprehensive circuit visualization capabilities.

## Key Features

### 🧠 Classical PINN
- Deep neural network implementation for solving 3D heat equation
- Automatic differentiation for PDE residual computation
- Boundary condition enforcement with time-dependent behavior
- PyTorch-based implementation with GPU acceleration

### 🌌 GQE-GPT-QPINN (Quantum Implementation)
- **GPT-based Quantum Circuit Generation**: Uses a transformer model to generate optimal quantum circuits
- **AI-Enhanced Energy Prediction**: Three-mode energy estimation system (ensemble, transformer, feature-based)
- **Generative Quantum Eigensolver (GQE)**: Novel approach that optimizes circuit structure rather than just parameters
- **Hardware-aware optimization**: Designed for real quantum devices with noise resilience
- **Parallel quantum device simulation**: Efficient batch processing across multiple quantum devices
- **Comprehensive Circuit Visualization**: Detailed quantum circuit diagrams and performance metrics

### 🔬 Multi-Objective Optimization
- **NSGA-II**: Non-dominated Sorting Genetic Algorithm II for multi-objective optimization
- **Five Objective Functions**: Initial condition, peak value, boundary condition, data fitting, and PDE residual
- **Pareto Front Analysis**: Comprehensive trade-off analysis between objectives
- **Hypervolume Evolution**: Performance metrics tracking over generations

### 🧬 RCGA Optimization
- **Real-Coded Genetic Algorithm** implemented in C++ for high performance
- **Latin Hypercube Sampling (LHS)**: Ensures well-distributed initial population
- **REX Crossover**: Real-valued crossover operator with expansion factor
- **JGG Selection**: Just Generation Gap selection strategy
- Python bindings via pybind11

### 🤖 AI-Enhanced Energy Estimation
- **Ensemble Predictor**: Combines feature-based and transformer-based models
- **Circuit Transformer**: Attention-based architecture for circuit sequence analysis
- **Feature Extractor**: Comprehensive circuit characteristic analysis
- **Online Learning**: Adaptive model improvement during optimization
- **Uncertainty Quantification**: Confidence scores for predictions

### 📊 Advanced Visualization and Analysis
- **Quantum Circuit Diagrams**: Visual representation of GQE-generated circuits with gate annotations
- **Performance Metrics**: Radar charts for circuit characteristics
- **Evolution Tracking**: RCGA/NSGA-II optimization progress visualization
- **Pareto Front Visualization**: 2D/3D plots of multi-objective solutions
- **Animation Generation**: GIF animations of optimization progress
- **Detailed Circuit Information**: JSON, LaTeX, and text format circuit specifications
- **Novelty Analysis**: Circuit diversity and innovation tracking

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
scikit-learn>=1.5.0
pybind11>=2.13.6
setuptools>=80.7.0
pandas>=2.2.0
seaborn>=0.13.0
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

### 3. Build optimizers
```bash
pip install -U . rcga_optimizer
pip install -U . nsga2_optimizer
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

### Running with Advanced Optimization

#### NSGA-II Multi-Objective Optimization
```python
# Enable NSGA-II multi-objective optimization
qsolver = GQEQuantumPINN(
    n_qubits=6,
    backend='default.mixed',
    shots=1000,
    noise_model='realistic',
    use_parallel=True,
    use_gpt_circuit_generation=True
)

# Train with NSGA-II
circuit_params, loss_history, training_time = qsolver.train_with_nsga2(n_samples=1500)
```

#### RCGA Optimization
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

#### AI-Enhanced Energy Prediction
```python
# Configure AI energy prediction mode
gqe_generator = GQEQuantumCircuitGeneratorWithGPT(
    n_qubits=6,
    use_ai_energy_prediction=True,
    energy_prediction_mode='ensemble'  # 'ensemble', 'transformer', or 'feature'
)
```

### Accessing Advanced Visualization
```python
# After training, visualize the quantum circuit
circuit_image_path = qsolver.visualize_quantum_circuit('results/')

# Save detailed circuit information
json_path, summary_path = qsolver.save_circuit_information('results/')

# Visualize circuit performance metrics
metrics_path = qsolver.visualize_circuit_metrics('results/')

# Generate GQE optimization process visualization
qsolver.visualize_gqe_generation_process('results/')

# Create optimization animation
qsolver.save_gqe_animation('results/')

# Visualize AI energy prediction performance (if used)
qsolver.visualize_ai_energy_performance('results/')
```

## Implementation Details

### Heat Equation
The code solves the 3D heat equation:
```
∂u/∂t = α∇²u
```
with time-dependent Dirichlet boundary conditions and Gaussian initial condition with improved physical modeling.

### GQE-GPT Architecture
1. **GPT Model**: Generates quantum circuit sequences as tokens with vocabulary of ~1000 gate combinations
2. **Circuit Templates**: Hardware-efficient ansätze with comprehensive noise resilience analysis
3. **Adaptive Optimization**: Automatically switches between NSGA-II, RCGA, SPSA, and Adam based on hardware constraints
4. **Circuit Evaluation**: Multi-dimensional metrics including noise resilience, hardware efficiency, expressivity, and parameter efficiency
5. **Novelty Tracking**: Advanced diversity and innovation analysis for generated circuits

### AI-Enhanced Energy Estimation

#### Ensemble Mode
- Combines feature-based and transformer-based predictors
- Uncertainty quantification with confidence scores
- Online learning with adaptive ensemble weights
- Fallback mechanisms for robust prediction

#### Transformer Mode
- Attention-based sequence modeling for quantum circuits
- Circuit tokenization with gate-level representation
- Multi-head attention for circuit pattern recognition
- Positional encoding for gate ordering

#### Feature Mode
- Comprehensive circuit feature extraction (20+ features)
- Random Forest fallback for robustness
- Circuit complexity and efficiency metrics
- Hardware-aware feature engineering

### NSGA-II Multi-Objective Optimization
- **Objectives**: 5 simultaneous objectives (initial condition, peak value, boundary condition, data fitting, PDE residual)
- **Population Size**: 100 individuals (configurable)
- **REX Crossover**: Multi-parent real-valued crossover
- **Crowding Distance**: Diversity preservation mechanism
- **Pareto Front**: Non-dominated solution sets
- **Hypervolume**: Performance indicator tracking

### RCGA Implementation
- **Population Size**: 50 individuals (configurable for hardware constraints)
- **REX Parameters**: ξ = 1.2 (expansion factor)
- **JGG Parameters**: 3 parents, 10 offspring per generation
- **LHS Initialization**: Latin Hypercube Sampling for better space coverage
- **Termination**: Maximum generations or convergence criteria

### Advanced Circuit Visualization Features
- **Interactive Circuit Diagrams** with parameter annotations
- **Performance Radar Charts** showing 5+ metrics simultaneously
- **Evolution Animations** showing optimization progress over time
- **Pareto Front Analysis** with 2D/3D projections
- **Novelty Tracking** with diversity evolution plots
- **LaTeX Export** for publication-quality circuit descriptions
- **JSON Export** for programmatic analysis

## Results

The implementation produces comprehensive analysis including:

### Solution Analysis
- Comparative visualizations of PINN vs GQE-GPT-QPINN solutions
- Error analysis over space and time
- Boundary condition satisfaction metrics with time-dependent behavior
- Training loss curves for multiple objectives
- Performance benchmarks across different optimization methods

### Circuit Analysis
- Quantum circuit diagrams with detailed annotations
- Circuit performance metrics and trade-off analysis
- Pareto front evolution for multi-objective cases
- Circuit novelty and diversity statistics
- AI prediction accuracy assessment

### Optimization Analysis
- RCGA/NSGA-II convergence analysis
- Population diversity evolution
- Hypervolume indicator progression
- Multi-objective trade-off visualization

### Output Files (in results directory):

#### Solution Comparison
- `heat_equation_comparison_gqe_gpt.png`: Solution comparison with enhanced visualization
- `heat_equation_profile_comparison_gqe_gpt.png`: 1D temperature profiles at multiple time points
- `heat_equation_error_analysis_gqe_gpt.png`: Comprehensive error metrics including relative errors
- `heat_equation_boundary_analysis_gqe_gpt.png`: Time-dependent boundary condition analysis

#### Quantum Circuit Information
- `gqe_quantum_circuit.png`: Visual quantum circuit diagram with gate annotations
- `gqe_circuit_text.txt`: PennyLane text representation
- `gqe_circuit_info.json`: Detailed circuit specification in JSON format
- `gqe_circuit_summary.txt`: Human-readable circuit summary with performance metrics
- `gqe_circuit_latex.tex`: LaTeX circuit description (publication-ready)
- `gqe_circuit_metrics.png`: Circuit performance radar charts and distributions

#### GQE Optimization Process
- `gqe_optimization_history.png`: Multi-round optimization progress
- `gqe_round_X_circuit.png`: Individual round circuit diagrams
- `gqe_rounds_summary.png`: Comprehensive round comparison
- `gqe_optimization_animation.gif`: Animated optimization progress
- `gqe_optimization_report.txt`: Detailed optimization report

#### AI Energy Prediction (if enabled)
- `ensemble_energy_performance.png`: Ensemble prediction accuracy
- `ai_energy_accuracy.png`: AI prediction vs actual energy comparison
- `circuit_energy_model.pth`: Trained energy prediction models

#### Multi-Objective Optimization (NSGA-II)
- `nsga2_pareto_front_3d.png`: 3D Pareto front visualization
- `nsga2_objectives_evolution.png`: Individual objective evolution
- `nsga2_hypervolume_evolution.png`: Hypervolume indicator progression
- `nsga2_pareto_pairs.png`: Pairwise objective trade-offs
- `nsga2_diversity_evolution.png`: Population diversity metrics
- `nsga2_optimization_results.json`: Comprehensive NSGA-II results
- `nsga2_pareto_fronts.csv`: Pareto front evolution data
- `nsga2_optimization_summary.txt`: Human-readable NSGA-II summary

#### RCGA Evolution (if used)
- `gqe_rcga_evolution.png`: RCGA fitness evolution with statistics
- `rcga_optimization_results.json`: RCGA optimization history

#### Novelty and Diversity Analysis
- `novelty_evolution.png`: Circuit novelty progression
- `gqe_gpt_statistics.png`: GPT generation vs fallback statistics

## Performance

Typical performance characteristics:

### Training Time
- **PINN**: ~100-200 seconds (NVIDIA RTX A2000 12GB)
- **GQE-GPT-QPINN with RCGA**: ~1-2 hours (Intel i5-13600K)
- **GQE-GPT-QPINN with NSGA-II**: ~2-4 hours (depending on population size)

### Accuracy
- **PINN**: MSE ~1e-5 to 1e-6
- **GQE-GPT-QPINN**: MSE ~1e-4 to 1e-5 (with quantum noise considerations)

### Optimization Convergence
- **RCGA**: 500-1000 generations typical for good solutions
- **NSGA-II**: 100-200 generations for Pareto front convergence
- **AI Energy Prediction**: 90%+ accuracy after 100+ training samples

### Circuit Characteristics
- **GPT-Generated Circuits**: 20-50 gates typically
- **Hardware Efficiency**: 0.8-0.95 typical scores
- **Noise Resilience**: 0.7-0.9 typical scores
- **Circuit Depth**: 8-20 layers typical

### Memory and Computational Requirements
- **RAM**: 8-16 GB recommended for full features
- **CPU**: Multi-core recommended for parallel processing
- **GPU**: Optional but recommended for large-scale training

## Advanced Configuration

### Optimization Method Selection
```python
# Automatic method selection based on hardware
if NSGA2_AVAILABLE and not is_hardware:
    method = "NSGA-II"  # Multi-objective for simulation
elif use_rcga and RCGA_AVAILABLE:
    method = "RCGA"     # Hardware-aware optimization
else:
    method = "SPSA/Adam"  # Fallback methods
```

### AI Energy Prediction Configuration
```python
# Configure prediction mode based on requirements
config = {
    'ensemble': {        # Best accuracy, higher computational cost
        'feature_weight': 0.6,
        'transformer_weight': 0.4,
        'uncertainty_threshold': 0.5
    },
    'transformer': {     # Good for circuit sequence analysis
        'd_model': 256,
        'nhead': 8,
        'num_layers': 4
    },
    'feature': {         # Fast, good for real-time optimization
        'n_features': 20,
        'fallback_model': 'RandomForest'
    }
}
```

### Parallel Processing Configuration
```python
# Adjust based on available resources
parallel_config = {
    'use_parallel': True,
    'n_parallel_devices': min(4, cpu_count() // 2),
    'batch_size_per_device': 50,
    'timeout': 90  # seconds
}
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{nakaji2024gqe,
  title={The generative quantum eigensolver (GQE) and its application for ground state search},
  author={Nakaji, Kouhei and others},
  journal={arXiv preprint arXiv:2401.09253},
  year={2024}
}

@article{raissi2019physics,
  title={Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations},
  author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
  journal={Journal of Computational Physics},
  volume={378},
  pages={686--707},
  year={2019}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on the GQE algorithm proposed by Nakaji et al.
- GPT architecture inspired by nanoGPT implementation
- PINN methodology based on Raissi et al.
- RCGA implementation follows standard real-coded genetic algorithm principles
- NSGA-II implementation based on Deb et al.
- Circuit visualization inspired by quantum circuit diagram standards

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Areas of particular interest:
- Additional circuit optimization strategies
- Enhanced AI energy prediction models
- Support for more quantum backends
- Performance improvements
- New visualization features
- Multi-objective optimization enhancements

## Troubleshooting

### Common Issues

1. **RCGA/NSGA-II optimizer not available**: 
   - Ensure C++ compiler is properly installed
   - Run the build command again with verbose output: `pip install -v . rcga_optimizer nsga2_optimizer`
   
2. **Circuit visualization errors**: 
   - Check matplotlib and required fonts are installed
   - Verify write permissions in results directory
   
3. **Memory issues with large circuits**: 
   - Reduce batch size or use sequential evaluation
   - Adjust `N_PARALLEL_DEVICES` parameter
   
4. **AI energy prediction errors**:
   - Ensure sufficient training data (50+ samples)
   - Check model file permissions and disk space
   
5. **NSGA-II convergence issues**:
   - Increase population size or generation count
   - Adjust crossover and mutation parameters

### Performance Optimization Tips

1. **For Hardware Devices**: Use RCGA with smaller population sizes
2. **For Simulation**: Use NSGA-II for comprehensive analysis
3. **For Quick Testing**: Disable AI energy prediction and use simple optimization
4. **For Production**: Enable all features with appropriate resource allocation

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

**Note**: This implementation represents a research prototype. For production use, consider additional validation and testing appropriate to your specific application requirements.
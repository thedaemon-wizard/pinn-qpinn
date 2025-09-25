# Quantum Physics-Informed Neural Networks (QPINNs) for 3D Heat Equation

A comprehensive benchmark implementation of Quantum Physics-Informed Neural Networks (QPINNs) for solving the 3D heat conduction equation forward problem, featuring state-of-the-art quantum circuit generation using GQE-GPT integration, multi-objective optimization, and comparison with enhanced classical PINNs using PINNsFormer architecture.

## Documentation(In progress)
Project Documentation: [HTML Pages](https://thedaemon-wizard.github.io/pinn-qpinn/build/html)

NSGA2 Code Documents: [Doxygen Page](https://thedaemon-wizard.github.io/pinn-qpinn/doxyxml/html)

## Overview

This repository provides highly optimized implementations of both QPINNs and classical PINNs with cutting-edge techniques:

### Quantum PINN Features:
- **GQE-GPT Integration**: Generative Quantum Eigensolver enhanced with GPT-based circuit generation
- **Multi-Objective Optimization**: NSGA-II algorithm with dynamic circuit updates
- **Unsupervised Quantum Energy Estimation**: Novel approach for noise-aware energy estimation
- **Hardware-Efficient Design**: Optimized for NISQ (Noisy Intermediate-Scale Quantum) devices
- **Parallel Processing**: Efficient batch evaluation for large-scale problems

### Classical PINN Features (PINNsFormer):
- **Transformer Architecture (PINNsFormer)**: State-of-the-art attention-based neural network for PDEs
- **Wavelet Activation Function**: ω₁*sin(x) + ω₂*cos(x) with learnable parameters for enhanced expressivity
- **Pseudo-Sequence Generation**: Converts point-wise inputs to sequences for Transformer processing
- **Spatio-Temporal Mixing**: Dedicated attention mechanisms for spatial and temporal dynamics
- **Multi-Scale Fourier Features**: Coarse/fine spatial and slow/fast temporal frequencies
- **Hard Boundary Constraints**: Smooth distance function enforcement using tanh-based transitions
- **Memory-Efficient Implementation**: Optimized configurations for GPU memory constraints

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

### 2. PINNsFormer Architecture (Classical PINN)
The implementation features a cutting-edge Transformer-based architecture specifically designed for PDEs:

#### Core Components:
- **Wavelet Activation Function**: Based on Real Fourier Transform with learnable weights (ω₁, ω₂)
- **Pseudo-Sequence Generator**: Creates temporal sequences from spatial points with learnable time offsets
- **Spatio-Temporal Mixer**: Dual attention mechanism for spatial and temporal correlations
- **Transformer Encoder**: Multi-layer self-attention blocks with residual connections
- **Transformer Decoder**: Optional encoder-decoder attention for complex architectures
- **Output Projection**: Maps sequence representations to final predictions

#### Technical Specifications:
```python
# Memory-Efficient Configuration
transformer_config = {
    'seq_length': 8,       # Sequence length for pseudo-temporal points
    'd_model': 64,         # Model dimension
    'n_heads': 4,          # Number of attention heads
    'n_layers': 2,         # Number of Transformer layers
    'd_ff': 256,          # Feedforward dimension
    'dropout': 0.1        # Dropout rate
}

# Full Configuration (for high-memory systems)
transformer_config = {
    'seq_length': 16,
    'd_model': 128,
    'n_heads': 8,
    'n_layers': 4,
    'd_ff': 512,
    'dropout': 0.1
}
```

### 3. Multi-Objective Optimization (NSGA-II)
Both QPINN and classical PINN use NSGA-II (Non-dominated Sorting Genetic Algorithm II) with:
- **Multiple Objectives**:
  - Classical PINN: Initial condition, Boundary condition, PDE residual
  - QPINN: Initial condition, Peak value, Boundary condition, PDE residual, Trace loss
- **REX Crossover**: Real-coded crossover operator for continuous parameters
- **Dynamic Parameter Bounds**: Adaptive scaling based on network/circuit complexity
- **Pareto Front Tracking**: Complete history of non-dominated solutions
- **Unified Configuration**: Ensures fair comparison between methods

### 4. Feature Engineering

#### Multi-Scale Fourier Features:
- **Spatial Features**: 
  - Coarse-scale (σ=5.0): Captures global structure
  - Fine-scale (σ=20.0): Captures local variations
- **Temporal Features**: 
  - Slow frequencies (2π/T): Long-term dynamics
  - Fast frequencies (10π/T): Rapid oscillations
- **Physics-aware scaling**: √(t/T + ε) for diffusion processes

#### Hard Boundary Constraints:
```python
u(x,y,z,t) = D(x,y,z) × N(x,y,z,t)
where D is smooth distance function: ε·tanh(d_min/ε)
```

### 5. Problem-Agnostic Design
The implementation uses scientifically grounded, problem-agnostic transformations:
- Trainable embedding functions for spatial and temporal features
- Learnable measurement combination weights (QPINN)
- Adaptive activation functions (Wavelet for PINN, tanh for QPINN)

### 6. Noise Mitigation (QPINN)
- Zero-noise extrapolation for error mitigation
- Noise-aware circuit evaluation
- Support for different noise models (light, realistic, heavy)

## Algorithm Details

### Classical PINN with PINNsFormer

The enhanced PINN implementation leverages the Transformer architecture adapted for physics-informed learning:

1. **Pseudo-Sequence Generation**
   ```
   Point (x,y,z,t) → Sequence of length L with temporal offsets
   Each element: embedded features + time shift encoding
   ```

2. **Spatio-Temporal Mixing**
   ```
   Dual attention mechanism:
   - Spatial attention: Self-attention on spatial features
   - Temporal attention: Cross-attention between time steps
   - Feature mixing: Non-linear combination layer
   ```

3. **Transformer Processing**
   ```
   Encoder: Multiple self-attention blocks with Wavelet activation
   Decoder (optional): Encoder-decoder attention for refinement
   Output: First sequence element projected to solution
   ```

4. **Physics-Informed Loss**
   ```
   L = λ_ic·L_ic + λ_bc·L_bc + λ_pde·L_pde
   where:
   - L_ic: Initial condition loss
   - L_bc: Boundary condition loss  
   - L_pde: PDE residual loss
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

### Classical PINN with PINNsFormer
- **PINNsFormer Architecture**: Transformer-based PDE solvers with attention mechanisms
- Wang et al. "When and why PINNs fail to train" (2022)
- Krishnapriyan et al. "Characterizing possible failure modes in PINNs" (2021)
- Vaswani et al. "Attention is All You Need" (2017) - Transformer architecture
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

### Additional Resources
- NVIDIA Technical Blog "Advancing Quantum Algorithm Design with GPTs" (2024)
- "Guaranteed efficient energy estimation using ShadowGrouping" Nature Communications 15, 799 (2025)
- Meyer et al. "Fisher Information in Noisy Intermediate-Scale Quantum Applications" Quantum 5, 539 (2021)

## Requirements

- Python 3.12+
- PennyLane 0.41+
- PyTorch 2.7+
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

### Basic Usage - Classical PINN with PINNsFormer:
```python
# Initialize Enhanced PINN with PINNsFormer
pinn = PINN(
    layers=[5, 128, 256, 256, 128, 1],  # Kept for compatibility
    use_hard_constraints=True,
    boundary_epsilon=0.1,
    fourier_features=True,
    num_fourier_features=64,
    use_transformer=True,  # Enable PINNsFormer
    transformer_config=None,  # Use default config
    transformer_memory_efficient=True  # Recommended for GPU
)

# Train with NSGA-II
state_dict, losses, training_time = pinn.train_with_nsga2(
    n_samples=100000,
    nsga2_config=NSGA2_COMMON_CONFIG
)

# Evaluate model
predictions = evaluate_pinn_nsga2(pinn)
```

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

## Configuration

Key parameters in the implementation:

### Classical PINN (PINNsFormer):
- `use_transformer`: Enable PINNsFormer architecture (default: True)
- `transformer_config`: Configuration dict for Transformer
  - `seq_length`: Length of pseudo-sequence (8 or 16)
  - `d_model`: Model dimension (64 or 128)
  - `n_heads`: Number of attention heads (4 or 8)
  - `n_layers`: Number of Transformer layers (2 or 4)
- `transformer_memory_efficient`: Use memory-efficient config (default: True)
- `use_hard_constraints`: Enable boundary constraints
- `fourier_features`: Enable multi-scale Fourier features
- `num_fourier_features`: Number of Fourier features (default: 64)

### Quantum PINN:
- `n_qubits`: Number of qubits (default: 6)
- `use_gpt_circuit_generation`: Enable GPT-based circuit generation

### Common:
- `alpha`: Thermal diffusivity
- `L`: Cube side length
- `T`: Final time
- `N_PARALLEL_DEVICES`: Number of parallel devices

## Output

The code generates comprehensive results including:

### For Classical PINN (PINNsFormer):
- Network architecture visualization
- Attention weight heatmaps
- Pseudo-sequence analysis
- Multi-scale feature maps
- Wavelet activation patterns
- Transformer layer outputs

### For QPINN:
- Quantum circuit diagrams
- Circuit quality metrics
- GQE optimization history
- GPT generation statistics
- Gate evolution heatmaps
- Energy estimation analysis

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

This implementation builds upon numerous research contributions in quantum machine learning, QPINNs, classical PINNs with Transformer architectures, and multi-objective optimization. Special thanks to all researchers whose work is referenced in this implementation.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Areas of particular interest:
- Additional Transformer architectures for PINNs
- Enhanced attention mechanisms for PDEs
- Additional quantum circuit optimization strategies
- Performance improvements for large-scale problems
- New visualization features
- Multi-objective optimization enhancements

## Troubleshooting

### Common Issues

1. **NSGA-II optimizer not available**: 
   - Ensure C++ compiler is properly installed
   - Run the build command again with verbose output: `pip install -v . module/nsga2_optimizer`
   
2. **Transformer memory issues**: 
   - Enable `transformer_memory_efficient=True`
   - Reduce `seq_length` in transformer_config
   - Decrease batch size
   
3. **Attention weight visualization errors**: 
   - Check matplotlib and required fonts are installed
   - Verify write permissions in results directory
   
4. **Circuit visualization errors**: 
   - Check matplotlib backend settings
   - Verify PennyLane installation
   
5. **NSGA-II convergence issues**:
   - Increase population size or generation count
   - Adjust crossover and mutation parameters
   - Check objective function scaling

### Performance Optimization Tips

1. **For GPU Training**: 
   - Use `transformer_memory_efficient=True`
   - Enable mixed precision with `torch.cuda.amp`
   
2. **For Hardware Devices**: 
   - Use smaller population sizes in NSGA-II
   - Enable parallel evaluation
   
3. **For Quick Testing**: 
   - Disable Transformer layers temporarily
   - Use smaller network architectures
   
4. **For Production**: 
   - Enable all features with appropriate resource allocation
   - Use checkpoint saving/loading

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

**Note**: This implementation represents a research prototype combining state-of-the-art techniques in both classical and quantum physics-informed neural networks. For production use, consider additional validation and testing appropriate to your specific application requirements.
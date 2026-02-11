# Quantum Physics-Informed Neural Networks (QPINNs) for 3D Heat Equation

A comprehensive benchmark implementation of Quantum Physics-Informed Neural Networks (QPINNs) for solving the 3D heat conduction equation forward problem, featuring state-of-the-art quantum circuit generation using GQE-GPT integration, adaptive loss weighting via ReLoBRaLo, and comparison with enhanced classical PINNs using PINNsFormer architecture.

## Documentation (In progress)
Project Documentation: [HTML Pages](https://thedaemon-wizard.github.io/pinn-qpinn/build/html)

## Overview

This repository provides highly optimized implementations of both QPINNs and classical PINNs with cutting-edge techniques:

### Quantum PINN Features:
- **GQE-GPT Integration**: Generative Quantum Eigensolver enhanced with GPT-based circuit generation
- **SPSA Optimizer**: Simultaneous Perturbation Stochastic Approximation for gradient-free quantum parameter optimization
- **ReLoBRaLo Adaptive Loss Weighting**: Dynamic multi-objective loss balancing using softmax-based relative balancing with exponential moving average
- **Unsupervised Quantum Energy Estimation**: Novel approach for noise-aware energy estimation
- **Hardware-Efficient Design**: Optimized for NISQ (Noisy Intermediate-Scale Quantum) devices
- **Parallel Processing**: Efficient batch evaluation for large-scale problems

### Classical PINN Features (SPINN + PINNsFormer):
- **SPINN Separable Architecture** (NeurIPS 2023 Spotlight): Per-axis body networks reducing O(N^d) → O(Nd) complexity
- **PINNsFormer Transformer** (ICLR 2024): Encoder-decoder attention with Wavelet activation for temporal dependencies
- **RAdam Optimizer**: Rectified Adam with decoupled weight decay (RAdamW behavior) for stable, adaptive training
- **L-BFGS Refinement**: Second-order quasi-Newton method for breaking through first-order plateaus
- **ReLoBRaLo Adaptive Loss Weighting**: Dynamic multi-objective loss balancing replacing fixed/equal weights
- **Curriculum Learning**: Three-phase training (IC-only → PDE ramp-up → full ReLoBRaLo) for improved convergence
- **Causal Temporal Weighting**: w(t) = exp(-epsilon * t/T) for PDE residuals respects parabolic PDE causality
- **Non-Negativity Constraint**: Soft penalty on negative temperature predictions (physical consistency)
- **Wavelet Activation Function**: w1*sin(x) + w2*cos(x) with learnable parameters (PINNsFormer Eq. 4)
- **Pseudo-Sequence Generation**: Converts point-wise inputs to sequences for Transformer processing
- **Multi-Scale Fourier Features**: Coarse/fine spatial and slow/fast temporal frequencies
- **Hard Boundary Constraints with IC Lifting**: Product distance function + free-space Green's function ansatz for initial condition
- **Comprehensive Benchmark Output**: MAE, MSE, RMSE, RelL2, MaxAE, energy conservation, physics metrics
- **Validation Monitoring**: Periodic MSE/RelL2 evaluation on held-out grid during training
- **Performance Monitoring**: GPU memory, utilization, CPU stats tracked during training
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

### 2. SPINN + PINNsFormer Architecture (Classical PINN)
The implementation integrates two state-of-the-art approaches for an efficient and accurate PINN architecture:

#### SPINN: Separable Physics-Informed Neural Networks (NeurIPS 2023 Spotlight)
- **Per-axis body networks**: Each spatial/temporal axis (x, y, z, t) is processed by a small independent MLP (R^1 → R^r)
- **Hadamard product aggregation**: Axis features are combined via element-wise product, producing a low-rank tensor approximation
- **Computational efficiency**: Reduces complexity from O(N^d) to O(Nd) for d-dimensional problems
- Reference: Cho et al. (2023) "Separable Physics-Informed Neural Networks"

#### PINNsFormer: Transformer-based PINN (ICLR 2024)
- **Wavelet Activation Function**: WaveAct(x) = w1*sin(x) + w2*cos(x) with learnable weights
- **Pseudo-Sequence Generator**: Creates temporal sequences from spatial points with physics-aware decay
- **Encoder-Decoder Architecture**: Self-attention encoder + cross-attention decoder with Wavelet residuals
- **Output Projection**: Learned temporal weights aggregate sequence positions to scalar output
- Reference: Zhao, Ding & Prakash (2024) "PINNsFormer"

#### S-PFormer: Simplified Decoder-Only Variant (2025)
- **Decoder-only Transformer**: Replaces separate encoder when SPINN + Fourier features provide sufficiently rich embeddings
- **Pre-norm design**: LayerNorm before attention for stable training
- **Scaled residual connections**: Factor 0.3 balances gradient flow without explosion
- Available as `SPFormerDecoder` / `SPFormerDecoderBlock` in `pinnsformer.py`

#### Architecture Pipeline:
```
Input (x,y,z,t)
  → SPINN body networks (4 per-axis MLPs, rank=64)
  → Hadamard product aggregation
  → Multi-scale Fourier features (spatial + temporal)
  → Feature concatenation
  → PINNsFormer pseudo-sequence generation
  → PINNsFormer encoder (self-attention + WaveAct)
  → PINNsFormer decoder (cross-attention + WaveAct)
  → Output projection (temporal weighting → scalar)
  → Hard constraints: g(x,y,z,t) + D(x,y,z) * output  (IC lifting + correction)
```

#### Technical Specifications:
```python
# Memory-Efficient Configuration (default)
transformer_config = {
    'seq_length': 8,       # Sequence length for pseudo-temporal points
    'd_model': 64,         # Model dimension / SPINN rank
    'n_heads': 4,          # Number of attention heads
    'n_layers': 2,         # Number of Transformer layers
    'd_ff': 256,           # Feedforward dimension
    'dropout': 0.1         # Dropout rate
}
spinn_config = {
    'rank': 64,            # Feature rank (matches d_model)
    'hidden_dim': 64,      # Hidden dimension per body network
    'n_hidden_layers': 2   # Hidden layers per body network
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

### 3. Optimization

#### Classical PINN: Two-Phase Hybrid Optimization (RAdam + L-BFGS)
The classical PINN uses a two-phase hybrid optimization approach:
- **Phase 1: RAdam Warm-up** (3000 epochs) - `torch.optim.RAdam` with `decoupled_weight_decay=True`:
  - Rectified Adam with automatic variance rectification for stable early training
  - Cosine annealing warm restarts scheduler
  - ReLoBRaLo adaptive loss weighting
- **Phase 2: L-BFGS Refinement** (200 iterations) - `torch.optim.LBFGS`:
  - Second-order quasi-Newton method with strong Wolfe line search
  - Breaks through first-order optimizer plateaus
  - Uses frozen ReLoBRaLo weights from Phase 1

#### Quantum PINN: SPSA Optimizer
The quantum PINN uses PennyLane's `qml.SPSAOptimizer`:
- **Gradient-Free**: Estimates gradients using only two function evaluations per step, regardless of parameter count
- **Noise Resilient**: Designed for noisy objective functions common in quantum computing
- **Scalable**: Computational cost per step is independent of the number of parameters

#### Adaptive Loss Weighting: ReLoBRaLo
Both PINN and QPINN use ReLoBRaLo (Relative Loss Balancing with Random Lookback) for adaptive multi-objective loss weighting:
- **Dynamic Weight Adjustment**: Loss weights are updated each epoch/step using softmax-based relative balancing
- **Exponential Moving Average**: Smooths weight updates with a configurable temperature parameter
- **Random Lookback**: Introduces stochasticity by randomly choosing between comparing to the previous step or an initial reference
- **Loss Components**:
  - Classical PINN: Initial condition, Peak value, Boundary condition, PDE residual (causal-weighted), Non-negativity
  - QPINN: Initial condition, Boundary condition, Interior (PDE), Trace loss

### 4. Feature Engineering

#### Multi-Scale Fourier Features:
- **Spatial Features**:
  - Coarse-scale (sigma=2.0): Captures global structure
  - Fine-scale (sigma=10.0): Captures local variations and narrow Gaussian peak
- **Temporal Features**:
  - Slow frequencies (2pi/T): Long-term dynamics
  - Fast frequencies (10pi/T): Rapid oscillations
- **Physics-aware scaling**: sqrt(t/T + epsilon) for diffusion processes

#### Hard Boundary Constraints with IC Lifting:
```python
u(x,y,z,t) = g(x,y,z,t) + D(x,y,z) * N(x,y,z,t)
where:
  g(x,y,z,t) = [σ₀²/(σ₀²+2αt)]^(3/2) * exp(-r²/(2(σ₀²+2αt)))
  # Free-space Green's function: exact Gaussian diffusion
  # At t=0: g = IC(x,y,z) exactly; at t>0: captures spreading + decay
  D(x,y,z) = 4*x/L*(1-x/L) * 4*y/L*(1-y/L) * 4*z/L*(1-z/L)
  # Product distance: exactly 0 on all 6 faces, max=1.0 at center
  N(x,y,z,t) = network correction (learned small residual)
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

## Modular File Structure

The codebase is organized into 8 focused modules (refactored from a single monolithic file):

| File | Description |
|------|-------------|
| `benchmark_config.json` | Externalized training/physics parameters |
| `config.py` | Global configuration, `BackendConfig`, config loading |
| `physics.py` | Physics equations, analytical solutions, metrics |
| `device_manager.py` | Quantum device management (CPU/GPU/QPU) |
| `pinnsformer.py` | SPINN body networks + PINNsFormer Transformer components |
| `gpt_circuit.py` | GPT-based quantum circuit generation |
| `pinn_model.py` | SPINN+PINNsFormer PINN with RAdam + L-BFGS + ReLoBRaLo |
| `qpinn_model.py` | Quantum PINN with SPSA + ReLoBRaLo |
| `main.py` | CLI entry point, benchmark orchestration, CSV output |

## Algorithm Details

### Classical PINN with SPINN + PINNsFormer

The PINN integrates SPINN separable architecture with PINNsFormer Transformer:

1. **SPINN Feature Extraction**
   ```
   Per-axis body networks: x → f_x(x), y → f_y(y), z → f_z(z), t → f_t(t)
   Each: R^1 → R^rank via small MLP with WaveletActivation
   Aggregation: f_x ⊙ f_y ⊙ f_z ⊙ f_t  (Hadamard product)
   → Learned projection to feature space
   ```

2. **Multi-Scale Fourier Feature Encoding**
   ```
   Spatial: coarse (σ=2.0) + fine (σ=10.0) random Fourier features
   Temporal: slow (2π/T) + fast (10π/T) frequencies
   Concatenated with SPINN aggregated features
   ```

3. **PINNsFormer Transformer Processing**
   ```
   Pseudo-sequence generation: point features → L-length sequence
     (with physics-aware temporal decay and position embeddings)
   Encoder: Self-attention + FFN + WaveletActivation residuals
   Decoder: Cross-attention from encoder output + FFN
   Output: Learned temporal weight aggregation → scalar
   ```

4. **IC Lifting with Free-Space Green's Function**
   ```
   u(x,y,z,t) = g(x,y,z,t) + D(x,y,z) * N(x,y,z,t)
   g(x,y,z,t) = [σ₀²/(σ₀²+2αt)]^(3/2) * exp(-r²/(2(σ₀²+2αt)))
   The network N only learns the small correction term, dramatically
   improving convergence (36x MSE reduction vs. zero-lifting baseline).
   ```

5. **Physics-Informed Loss with ReLoBRaLo**
   ```
   L = w_ic * L_ic + w_peak * L_peak + w_pde * L_pde + w_nonneg * L_nonneg
   where weights are dynamically adjusted via ReLoBRaLo:
   - L_ic: Initial condition loss (MSE)
   - L_peak: Gaussian peak accuracy loss
   - L_pde: PDE residual loss (causal-weighted)
   - L_nonneg: Non-negativity penalty (on PDE + IC points)
   ```

6. **Two-Phase Hybrid Optimization with Curriculum**
   ```
   Phase 1: RAdam (3000 epochs) with curriculum learning:
     Phase 1a [0-30%]: IC-only training
     Phase 1b [30-70%]: PDE ramp-up
     Phase 1c [70-100%]: Full ReLoBRaLo
   Phase 2: L-BFGS refinement (200 iterations)
     Frozen ReLoBRaLo weights, strong Wolfe line search
   ```

### Core Algorithm: GQE-QPINNs

The implementation follows a hierarchical optimization approach:

1. **Circuit Generation Phase**
   - Initialize with KetGPT dataset or rule-based ansatz
   - Use GPT model to generate circuit candidates
   - Evaluate using multi-objective Bayesian optimization

2. **Parameter Optimization Phase**
   - SPSA optimization with ReLoBRaLo-weighted objectives:
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
du/dt = alpha * nabla^2(u)
```

With:
- Initial condition: Gaussian distribution centered at domain center
- Boundary condition: u = 0 at all boundaries
- Domain: [0,L]^3 x [0,T]

## Key References

### Optimization
- **RAdam**: Liu et al. "On the Variance of the Adaptive Learning Rate and Beyond" (ICLR 2020)
- **L-BFGS**: Nocedal & Wright "Numerical Optimization" (2006); standard second-phase optimizer for PINNs
- **SPSA**: Spall (1998) "Implementation of the Simultaneous Perturbation Algorithm for Stochastic Optimization"
- **ReLoBRaLo**: Bischof & Kraus (2025) "Multi-Objective Loss Balancing for Physics-Informed Deep Learning", Computer Methods in Applied Mechanics and Engineering
- **Hard Constraints**: Sukumar & Srivastava (2022) "Exact imposition of boundary conditions with distance functions in PINNs"

### Classical PINN with SPINN + PINNsFormer
- **SPINN**: Cho et al. "Separable Physics-Informed Neural Networks" (NeurIPS 2023 Spotlight)
- **PINNsFormer**: Zhao, Ding & Prakash "PINNsFormer: A Transformer-Based Framework for Physics-Informed Neural Networks" (ICLR 2024)
- **S-PFormer**: "Spectral PINNsformer: A Simplified Decoder-Only Architecture" (2025) -- decoder-only variant with Fourier embeddings
- Wang et al. "When and why PINNs fail to train" (2022)
- Krishnapriyan et al. "Characterizing possible failure modes in PINNs" (2021)
- Vaswani et al. "Attention is All You Need" (2017) - Transformer architecture

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
- PyTorch >= 2.10
- PennyLane >= 0.44
- pennylane-lightning >= 0.41
- NumPy >= 2.0
- Matplotlib >= 3.10
- Transformers (Hugging Face) >= 4.50
- BoTorch >= 0.14 (for Bayesian optimization)
- Scikit-learn >= 1.7
- pandas >= 2.3
- psutil >= 7.0

## Development Environment

### Tested Configuration

| Component | Specification |
|-----------|--------------|
| OS | AlmaLinux 9.7 |
| CPU | Intel Core i5-13600K |
| Memory | 128 GB DDR5-5200 |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation 96 GB |
| Storage | 1 TB SSD (system) + 4 TB SSD (data) |
| Motherboard | MSI MAG Z790 TOMAHAWK MAX WIFI |
| Python | 3.12 (virtual environment) |
| PyTorch | 2.10.0+cu128 |
| PennyLane | 0.44.0 |

## Installation

```bash
# Create and activate virtual environment (Python 3.12)
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: KetGPT dataset support for QPINN circuit initialization
pip install aiohttp h5py fsspec
```

## Usage

### Running the Benchmark

```bash
# Activate virtual environment
source .venv/bin/activate

# Run benchmark (auto-detects GPU/CPU)
python main.py --backend auto

# Run with custom configuration
python main.py --backend auto --config benchmark_config.json

# Force specific backend
python main.py --backend cpu     # CPU only
python main.py --backend cuda    # NVIDIA GPU (CUDA)
python main.py --backend gpu     # Alias for cuda
python main.py --backend qpu     # Quantum processing unit (requires AWS Braket)
```

### Expected Runtime

With the tested hardware configuration (RTX PRO 6000 Blackwell):
- PINN training (3000 RAdam epochs + 200 L-BFGS iterations): ~55 minutes
- QPINN training (200 SPSA iterations): ~3 hours
- Total benchmark: ~4 hours

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

# Train with RAdam optimizer and ReLoBRaLo adaptive loss weighting
model.train_radam(n_samples=100000, epochs=3000)
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

# Train with SPSA optimizer and ReLoBRaLo adaptive loss weighting
qsolver.train_spsa(n_samples=1500, max_iterations=200)
```

## Configuration

All training parameters are externalized to `benchmark_config.json`. Use `--config` to specify a custom config file:

```bash
python main.py --backend auto --config benchmark_config.json
```

### Key Config Sections:

| Section | Description |
|---------|-------------|
| `physics` | alpha, L, T, sigma_0 |
| `grid` | nx, ny, nz, nt spatial/temporal resolution |
| `pinn.training` | epochs, lr, weight_decay, n_samples |
| `pinn.lbfgs` | L-BFGS refinement settings |
| `pinn.architecture` | layers, transformer, fourier features |
| `pinn.accuracy` | nonneg_weight, causal_epsilon, curriculum phases |
| `pinn.relobralo` | alpha, rho, tau for ReLoBRaLo |
| `qpinn.training` | max_iterations, spsa_c, spsa_a, n_samples |
| `qpinn.circuit` | n_qubits, shots, backend, noise_model |
| `validation` | interval, grid_size for periodic validation |
| `monitoring` | GPU/CPU performance tracking |

### Classical PINN (SPINN + PINNsFormer):
- `use_transformer`: Enable PINNsFormer architecture (default: True)
- `transformer_memory_efficient`: Use memory-efficient config (default: True)
- `pinn.architecture.spinn`: SPINN body network configuration
  - `rank`: Feature rank / output dimension (default: 64, matches d_model)
  - `hidden_dim`: Hidden layer width per body network (default: 64)
  - `n_hidden_layers`: Number of hidden layers per body network (default: 2)
- `pinn.architecture.transformer`: PINNsFormer Transformer configuration
  - `seq_length`: Length of pseudo-sequence (8 or 16)
  - `d_model`: Model dimension (64 or 128)
  - `n_heads`: Number of attention heads (4 or 8)
  - `n_layers`: Number of Transformer layers (2 or 4)
  - `d_ff`: Feedforward dimension (256 or 512)
  - `dropout`: Dropout rate (default: 0.1)
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

All results are saved to the `results/` directory (20 files per benchmark run):

### Plots (PNG)
| File | Description |
|------|-------------|
| `comparison_heat_equation.png` | Temperature field comparison (analytical vs PINN vs QPINN) |
| `loss_comparison.png` | Training loss curves for both methods |
| `profile_comparison.png` | Temperature profiles along domain center |
| `pinn_relobralo_evolution.png` | PINN ReLoBRaLo adaptive weight evolution |
| `qpinn_relobralo_evolution.png` | QPINN ReLoBRaLo adaptive weight evolution |
| `error_distribution.png` | Spatial error distribution for both methods |
| `pinn_learning_rate.png` | PINN learning rate schedule |

### Data (CSV)
| File | Description |
|------|-------------|
| `pinn_training_losses.csv` | Per-epoch PINN total loss |
| `pinn_loss_components.csv` | Per-epoch PINN loss components (IC, peak, BC, PDE, nonneg) and ReLoBRaLo weights |
| `qpinn_training_losses.csv` | Per-step QPINN total loss |
| `qpinn_relobralo_weights.csv` | Per-step QPINN ReLoBRaLo weights |
| `metrics_over_time.csv` | Per-timestep MSE, MAE, RMSE, RelL2, MaxAE, energy, boundary errors |
| `pinn_validation_metrics.csv` | PINN validation MSE/RelL2 per time slice during training |
| `qpinn_validation_metrics.csv` | QPINN validation metrics during training |
| `performance_metrics.csv` | GPU memory, utilization, CPU stats during PINN training |

### Reports and Checkpoints
| File | Description |
|------|-------------|
| `comparative_analysis.json` | Full comparative analysis with metadata and metrics |
| `benchmark_summary.txt` | Human-readable summary report |
| `gqe_circuit_info.json` | Quantum circuit structure and optimized parameters |
| `gqe_circuit_summary.txt` | Circuit performance metrics summary |
| `gqe_circuit_text.txt` | PennyLane-format circuit diagrams |
| `benchmark.log` | Detailed training log with timestamps |
| `pinn_radam_checkpoint.pth` | PINN model checkpoint |
| `qpinn_spsa_checkpoint.pth` | QPINN model checkpoint |

## Performance Comparison

The implementation provides fair comparison between SPINN+PINNsFormer PINN and QPINN using:
- Consistent adaptive loss weighting (ReLoBRaLo) for both methods
- Identical problem configuration and evaluation metrics
- Comprehensive metrics: MSE, MAE, RMSE, RelL2, MaxAE, energy conservation, boundary satisfaction
- Per-timestep metrics output (CSV) and formatted research summary (TXT/JSON)

### Latest Benchmark Results (RTX PRO 6000 Blackwell, PyTorch 2.10 + PennyLane 0.44)

**Architecture**: SPINN + PINNsFormer with IC lifting (separable per-axis body networks + Transformer encoder-decoder + free-space Green's function ansatz)

| Metric | PINN (SPINN+PINNsFormer) | QPINN (SPSA) | Better |
|--------|-------------------------|--------------|--------|
| MSE | 2.125e-06 | 1.352e-06 | QPINN |
| MAE | 1.022e-03 | 1.679e-04 | QPINN |
| RMSE | 1.458e-03 | 1.163e-03 | QPINN |
| Relative L2 | 1.381e-01 | 1.101e-01 | QPINN |
| Max AE | 3.975e-03 | 1.235e-01 | PINN |
| Peak Error | 3.811e-03 | 1.862e-03 | QPINN |
| Neg. Violations | 0 | 0 | Tie |
| Training Time | 3,745 s | 10,605 s | PINN |
| Parameters | 454,527 | 9 (1 circuit + 8 classical) | QPINN |
| Training Steps | 3,000 RAdam + 200 L-BFGS | 200 SPSA | - |
| Adaptive Weighting | ReLoBRaLo | ReLoBRaLo | - |

Both methods use ReLoBRaLo adaptive loss weighting and hard boundary constraints for fair comparison. The PINN achieves comparable MSE to QPINN (within 2x) and superior Max AE (124x better), while QPINN achieves better overall accuracy with 50,000x fewer parameters. Comprehensive metrics (MSE, MAE, RMSE, RelL2, MaxAE, energy conservation, boundary satisfaction) are output after each benchmark run.

## Citation

If you use this code in your research, please cite the relevant papers listed in the references section.

## License

This implementation is provided for research purposes. Please check individual paper licenses for specific algorithm implementations.

## Acknowledgments

This implementation builds upon numerous research contributions in quantum machine learning, QPINNs, classical PINNs with Transformer architectures, and adaptive loss weighting. Special thanks to all researchers whose work is referenced in this implementation.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Areas of particular interest:
- Additional Transformer architectures for PINNs
- Enhanced attention mechanisms for PDEs
- Additional quantum circuit optimization strategies
- Performance improvements for large-scale problems
- New visualization features
- Alternative adaptive loss weighting strategies

## Troubleshooting

### Common Issues

1. **Transformer memory issues**:
   - Enable `transformer_memory_efficient=True`
   - Reduce `seq_length` in transformer_config
   - Decrease batch size

2. **Attention weight visualization errors**:
   - Check matplotlib and required fonts are installed
   - Verify write permissions in results directory

3. **Circuit visualization errors**:
   - Check matplotlib backend settings
   - Verify PennyLane installation

4. **Backend detection issues**:
   - Use `--backend cpu` to force CPU if CUDA detection fails
   - Ensure CUDA toolkit is installed for GPU backends
   - Verify PennyLane device availability for QPU backend

### Performance Optimization Tips

1. **For GPU Training**:
   - Use `transformer_memory_efficient=True` (default)
   - Full float32 precision is recommended for PINNs (mixed precision can cause convergence issues)

2. **For Hardware Quantum Devices**:
   - Enable parallel evaluation
   - Adjust SPSA perturbation parameters for noisy hardware

3. **For Quick Testing**:
   - Reduce `pinn_epochs` in `config.py`
   - Set `qnn_epochs = 50` for faster QPINN evaluation

4. **For Resuming from Checkpoints**:
   - Checkpoints are automatically saved to `results/`
   - Re-running `main.py` will load existing checkpoints if found
   - Delete checkpoint files to force re-training

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

---

**Note**: This implementation represents a research prototype combining state-of-the-art techniques in both classical and quantum physics-informed neural networks. For production use, consider additional validation and testing appropriate to your specific application requirements.

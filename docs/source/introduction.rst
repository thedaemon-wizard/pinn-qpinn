.. _introduction:

Introduction
============

Overview
--------

This project presents a comprehensive benchmark comparison between classical Physics-Informed Neural Networks (PINNs) and Quantum Physics-Informed Neural Networks (QPINNs) for solving the 3D heat conduction equation. The implementation represents the state-of-the-art in both classical and quantum approaches to solving partial differential equations (PDEs).

Key Innovations
---------------

Classical PINNs Enhancement: SPINN + PINNsFormer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **SPINN: Separable Physics-Informed Neural Networks (NeurIPS 2023 Spotlight)**

   - Per-axis body networks (x, y, z, t) reducing O(N^d) to O(Nd) complexity
   - Hadamard product aggregation for low-rank tensor approximation
   - Reference: Cho et al. (2023)

2. **PINNsFormer Transformer Architecture (ICLR 2024)**

   - Wavelet activation function: w1*sin(x) + w2*cos(x) with learnable weights
   - Pseudo-sequence generation with physics-aware temporal decay
   - Encoder-decoder attention with Wavelet residual connections
   - Reference: Zhao, Ding & Prakash (2024)

3. **Multi-Scale Fourier Feature Mapping**

   - Coarse (sigma=2.0) and fine (sigma=10.0) spatial frequencies
   - Slow and fast temporal frequencies for diffusion dynamics
   - Concatenated with SPINN aggregated features

4. **RAdam + L-BFGS Hybrid Optimization with ReLoBRaLo**

   - Phase 1: RAdam with decoupled weight decay and curriculum learning
   - Phase 2: L-BFGS second-order refinement with frozen ReLoBRaLo weights
   - ReLoBRaLo (Bischof & Kraus, 2025) for dynamic balancing of IC, peak, PDE, and non-negativity losses

5. **Accuracy Improvements**

   - Curriculum learning: 3-phase training (IC-only warm-up, PDE ramp-up, full ReLoBRaLo)
   - Causal temporal weighting: :math:`w(t) = \exp(-\varepsilon\, t/T)` on PDE residuals
   - Non-negativity soft constraint: :math:`\text{mean}(\text{ReLU}(-u)^2)` penalty
   - Hard boundary constraints via product distance function

6. **Comprehensive Benchmark Output**

   - MSE, MAE, RMSE, Relative L2, Max AE, Relative MAE
   - Thermal energy conservation tracking per timestep
   - Boundary satisfaction metrics, peak accuracy, non-negativity statistics

5. **Structured Logging and JSON Configuration**

   - All ``print()`` replaced by hierarchical ``logging.getLogger('benchmark.MODULE')``
   - Output to ``results/benchmark.log`` (DEBUG) and console (INFO)
   - ``benchmark_config.json`` externalises all hyperparameters
   - ``--config`` CLI argument for reproducible experiments

6. **Validation and Performance Monitoring**

   - Periodic validation on held-out grids during training (PINN every 200 epochs, QPINN every 50 steps)
   - GPU memory/utilisation and CPU/RAM metrics via ``psutil`` and ``nvidia-smi``
   - New CSV outputs: ``pinn_validation_metrics.csv``, ``qpinn_validation_metrics.csv``, ``performance_metrics.csv``

Quantum PINNs with GQE-GPT
^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **ketGPT Integration**
   
   - Pre-trained on 50,000+ quantum circuits
   - Transformer-based circuit generation
   - Context-aware circuit design

2. **Generative Quantum Eigensolver (GQE)**
   
   - GPT-based circuit architecture search
   - Dynamic circuit update during optimization
   - Preference-based Pareto selection

3. **Nine-Objective Bayesian Optimization**
   
   - Hardware efficiency
   - Noise resilience
   - Expressivity
   - Mitigation compatibility
   - Trainability
   - Entanglement capability
   - Circuit depth efficiency
   - Parameter efficiency
   - **Unsupervised energy estimation quality**

4. **Trainable Embeddings (TE-QPINN)**
   
   - Learnable spatial features (8 polynomial basis)
   - Learnable temporal frequencies (mixed basis)
   - Enhanced expressivity over fixed encodings

Scientific Contributions
------------------------

1. **Fair Comparison Framework**

   Both PINNs and QPINNs use identical:
   - ReLoBRaLo adaptive loss weighting
   - Training data generation (Latin Hypercube Sampling)
   - Evaluation metrics
   - Problem configuration

2. **Novel QPINN Features**
   
   - First implementation combining GQE with GPT-based generation
   - Unsupervised energy estimation as an optimization objective
   - Bayesian multi-objective circuit optimization
   - Hardware-aware noise modeling

3. **Comprehensive Benchmarking**
   
   - Multiple performance metrics (MSE, Relative L2, boundary satisfaction)
   - Computational resource analysis
   - Statistical significance testing
   - Pareto front visualization

Problem Statement
-----------------

The 3D heat conduction equation:

.. math::

   \frac{\partial u}{\partial t} = \alpha \nabla^2 u = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + \frac{\partial^2 u}{\partial z^2} \right)

With:
- Domain: :math:`[0, 1]^3 \times [0, 0.1]`
- Initial condition: Gaussian centered at (0.5, 0.5, 0.5)
- Boundary conditions: Homogeneous Dirichlet
- Thermal diffusivity: :math:`\alpha = 0.01`

Implementation Highlights
-------------------------

**Classical PINN**:

- Network architecture: [5, 128, 256, 256, 128, 1] (237,399 parameters)
- Multi-scale Fourier features: 64 features (coarse :math:`\sigma = 2`, fine :math:`\sigma = 10`)
- PINNsFormer Transformer with Wavelet activation and spatio-temporal mixing
- Hard boundary constraints via parabolic product distance function
- Curriculum learning: IC-only warm-up, PDE ramp-up, full ReLoBRaLo, L-BFGS refinement
- Causal temporal weighting on PDE residuals
- Non-negativity soft constraint for physical plausibility

**Quantum PINN**:

- Quantum circuit: 6 qubits, GQE-GPT generated with ketGPT pre-training
- Shots: 2,048 per evaluation (realistic hardware mode)
- Noise model: Realistic NISQ device characteristics (depolarizing + amplitude damping)
- Backend: PennyLane with ``lightning.qubit`` simulator
- Trainable embeddings (TE-QPINN): 8 spatial + 11 temporal features

**Optimization**:

- PINN: PyTorch RAdam (lr=1e-3, weight_decay=1e-2) + L-BFGS
- QPINN: PennyLane SPSAOptimizer (c=0.2, 200 iterations)
- Both: ReLoBRaLo adaptive loss weighting (alpha=0.999, rho=0.999, tau=1.0)

Results Preview
---------------

The benchmark demonstrates that QPINNs can achieve significantly better accuracy
than classical PINNs with orders-of-magnitude fewer parameters. Key findings
include:

- **Accuracy**: QPINN achieves lower MSE than the classical PINN
- **Parameter efficiency**: QPINN uses ~9 parameters vs. PINN SPINN+PINNsFormer
- **Circuit Quality**: GQE-GPT generated circuits with ketGPT pre-training achieve high hardware efficiency and noise resilience scores
- **Adaptive loss balancing**: ReLoBRaLo is effective for both classical and quantum training

See the :doc:`experimental_results` chapter for detailed benchmark numbers.

Document Organization
---------------------

1. **Theoretical Background**: Mathematical foundations and algorithms
2. **PINNs Implementation**: Classical approach with enhancements
3. **QPINNs Implementation**: Quantum approach with GQE-GPT
4. **Optimizers and Adaptive Loss Weighting**: RAdam, SPSA, and ReLoBRaLo
5. **Experimental Results**: Comprehensive benchmark results
6. **API Reference**: Complete code documentation
7. **References**: Scientific papers and resources

Getting Started
---------------

To reproduce the experiments:

.. code-block:: bash

   # Install dependencies
   pip install -r requirements.txt

   # Run benchmark (auto-detect backend)
   python main.py --backend auto

   # Or specify a backend explicitly
   python main.py --backend cuda   # GPU
   python main.py --backend cpu    # CPU only

   # Use an external JSON config to override all hyperparameters
   python main.py --backend auto --config benchmark_config.json

The ``--config`` argument loads physics constants, grid sizes, training
epochs, optimizer settings, curriculum learning phases, validation intervals,
and parallelism options from an external JSON file.  See
:func:`config.load_benchmark_config` in the :doc:`api_reference`.

For detailed setup instructions, see the README file in the project repository.
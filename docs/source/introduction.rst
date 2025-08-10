.. _introduction:

Introduction
============

Overview
--------

This project presents a comprehensive benchmark comparison between classical Physics-Informed Neural Networks (PINNs) and Quantum Physics-Informed Neural Networks (QPINNs) for solving the 3D heat conduction equation. The implementation represents the state-of-the-art in both classical and quantum approaches to solving partial differential equations (PDEs).

Key Innovations
---------------

Classical PINNs Enhancement
^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Fourier Neural Operator (FNO) Integration**
   
   - Spectral convolution layers for global feature learning
   - Memory-efficient implementation for large-scale problems
   - Multi-scale Fourier feature mapping

2. **Temporal Attention Mechanism**
   
   - Enhanced time-dependent dynamics modeling
   - Attention-based feature aggregation
   - Improved long-term prediction accuracy

3. **NSGA-II Multi-objective Optimization**
   
   - Four objectives: Initial condition, boundary condition, PDE residual, peak value
   - REX crossover with V-shaped distribution
   - High-performance C++ implementation with Python bindings

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
   - NSGA-II multi-objective optimization
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

- Network architecture: [5, 64, 128, 128, 64, 1]
- Fourier features: 64 multi-scale features
- FNO modes: (8, 8, 8) for 3D spectral convolution
- Hard boundary constraints with smooth transitions

**Quantum PINN**:

- Quantum circuit: 6 qubits, depth ~18
- Shots: 1000 per evaluation
- Noise model: Realistic NISQ device characteristics
- Backend: PennyLane with mixed state simulator

**Optimization**:

- NSGA-II populations: 50 (PINN), 30 (QPINN)
- Generations: 100 (PINN), 200 (QPINN)
- REX crossover: 3 parents → 10/5 children
- Progress interval: 10 generations

Results Preview
---------------

The benchmark demonstrates:

- **Accuracy**: Classical PINN achieves ~45% lower MSE
- **Efficiency**: PINN is ~7.4× faster in training time
- **Scalability**: PINN uses 33,281 parameters vs 144 for QPINN
- **Circuit Quality**: QPINN achieves high scores in hardware efficiency (0.723) and noise resilience (0.812)

These results provide valuable insights into the current state of quantum machine learning for PDE solving and highlight areas for future improvement.

Document Organization
---------------------

1. **Theoretical Background**: Mathematical foundations and algorithms
2. **PINNs Implementation**: Classical approach with enhancements
3. **QPINNs Implementation**: Quantum approach with GQE-GPT
4. **NSGA-II Optimizer**: C++ implementation details
5. **Experimental Results**: Comprehensive benchmark results
6. **API Reference**: Complete code documentation
7. **References**: Scientific papers and resources

Getting Started
---------------

To reproduce the experiments:

.. code-block:: bash

   # Install dependencies
   pip install -r requirements.txt
   
   # Build C++ optimizer
   pip install -U module/nsga2_optimizer
   
   # Run benchmark
   python pinns_d3.py

For detailed setup instructions, see the README file in the project repository.
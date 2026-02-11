PINNs-QPINNs Heat Conduction Benchmark Documentation
====================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   theoretical_background
   pinns_implementation
   qpinns_implementation
   experimental_results
   api_reference
   references

Introduction
============

This documentation describes a comprehensive benchmark comparison between Physics-Informed Neural Networks (PINNs) and Quantum Physics-Informed Neural Networks (QPINNs) for solving the 3D heat conduction equation. The implementation includes state-of-the-art techniques such as:

* **Classical PINNs**: SPINN (Separable PINN, NeurIPS 2023) + PINNsFormer (ICLR 2024) Transformer architecture with multi-scale Fourier features, trained with RAdam + L-BFGS
* **Quantum PINNs**: Featuring ketGPT, GQE-GPT circuit generation, and trainable embeddings (TE-QPINN), trained with PennyLane SPSAOptimizer
* **Adaptive loss weighting**: ReLoBRaLo (Relative Loss Balancing with Random Lookback) for dynamic multi-loss balancing (Bischof & Kraus, 2025)
* **Modular architecture**: 8-file codebase with CLI-based backend selection (auto/cpu/cuda/gpu/qpu)

Project Overview
----------------

The project aims to provide a fair comparison framework between classical and quantum machine learning approaches for solving partial differential equations (PDEs), specifically focusing on the 3D heat conduction equation:

.. math::

   \frac{\partial u}{\partial t} = \alpha \nabla^2 u

where :math:`u(x,y,z,t)` is the temperature field and :math:`\alpha` is the thermal diffusivity.

Key Features
------------

1. **Advanced PINNs Implementation (SPINN + PINNsFormer)**

   * SPINN separable per-axis body networks (NeurIPS 2023 Spotlight) for O(Nd) complexity
   * PINNsFormer Transformer encoder-decoder (ICLR 2024) with Wavelet activation
   * Multi-scale Fourier feature mapping (coarse + fine spatial, slow + fast temporal)
   * Hard boundary constraints with product distance function
   * Curriculum learning (3-phase: IC → PDE ramp → full ReLoBRaLo)

2. **Quantum PINNs with GQE-GPT**
   
   * Generative Quantum Eigensolver (GQE) with GPT-based circuit generation
   * KetGPT dataset integration for pre-training
   * Unsupervised energy estimation
   * Bayesian multi-objective optimization for circuit selection

3. **Optimizers and Adaptive Loss Weighting**

   * PINNs: PyTorch RAdam with decoupled weight decay + L-BFGS refinement
   * QPINNs: PennyLane SPSAOptimizer
   * ReLoBRaLo adaptive loss weighting for dynamic multi-loss balancing
   * Curriculum learning: IC-only warm-up, PDE ramp-up, full ReLoBRaLo
   * CLI-based backend selection: auto/cpu/cuda/gpu/qpu

4. **Comprehensive Benchmarking**
   
   * Multiple loss functions: initial condition, boundary condition, PDE residual, peak value, and trace
   * Hardware-aware quantum circuit optimization
   * Noise resilience evaluation
   * Pareto front analysis

System Requirements
-------------------

* Python 3.12+
* PyTorch 2.10+
* PennyLane 0.44+
* CUDA-capable GPU (recommended)

Installation
------------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/yourusername/pinns-qpinns-benchmark.git
   cd pinns-qpinns-benchmark

   # Install Python dependencies
   pip install -r requirements.txt

Quick Start
-----------

.. code-block:: bash

   # Run the complete benchmark (auto-detect backend)
   python main.py --backend auto

This will execute both PINN and QPINN training with RAdam/SPSA optimizers and ReLoBRaLo adaptive loss weighting, and generate comparative visualizations.

Documentation Structure
-----------------------

* **Theoretical Background**: Mathematical foundations and algorithm descriptions
* **PINNs Implementation**: SPINN + PINNsFormer architecture with RAdam + L-BFGS + ReLoBRaLo
* **QPINNs Implementation**: Quantum circuit design and optimization strategies
* **Experimental Results**: Comprehensive benchmark results (MSE, MAE, RMSE, RelL2, energy, etc.)
* **API Reference**: Complete API documentation for all modules
* **References**: Scientific papers and resources

Contact
-------

For questions or contributions, please contact the research team at [email@example.com]

License
-------

This project is licensed under the MIT License - see the LICENSE file for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
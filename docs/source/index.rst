PINNs-QPINNs Heat Conduction Benchmark Documentation
====================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   theoretical_background
   pinns_implementation
   qpinns_implementation
   nsga2_optimizer
   experimental_results
   api_reference
   references

Introduction
============

This documentation describes a comprehensive benchmark comparison between Physics-Informed Neural Networks (PINNs) and Quantum Physics-Informed Neural Networks (QPINNs) for solving the 3D heat conduction equation. The implementation includes state-of-the-art techniques such as:

* **Classical PINNs**: Enhanced with Fourier Neural Operators (FNO) and temporal attention mechanisms
* **Quantum PINNs**: Featuring ketGPT, GQE-GPT circuit generation, and trainable embeddings (TE-QPINN)
* **Multi-objective optimization**: NSGA-II algorithm for both classical and quantum approaches
* **High-performance C++ implementation**: NSGA-II optimizer with Python bindings

Project Overview
----------------

The project aims to provide a fair comparison framework between classical and quantum machine learning approaches for solving partial differential equations (PDEs), specifically focusing on the 3D heat conduction equation:

.. math::

   \frac{\partial u}{\partial t} = \alpha \nabla^2 u

where :math:`u(x,y,z,t)` is the temperature field and :math:`\alpha` is the thermal diffusivity.

Key Features
------------

1. **Advanced PINNs Implementation**
   
   * Fourier Neural Operator (FNO) integration for improved spectral properties
   * Multi-scale Fourier feature mapping
   * Temporal attention mechanism for better time-dependent learning
   * Hard boundary constraints with smooth transitions

2. **Quantum PINNs with GQE-GPT**
   
   * Generative Quantum Eigensolver (GQE) with GPT-based circuit generation
   * KetGPT dataset integration for pre-training
   * Unsupervised energy estimation
   * Bayesian multi-objective optimization for circuit selection

3. **NSGA-II Multi-objective Optimization**
   
   * Unified optimization framework for fair comparison
   * REX crossover operator with V-shaped distribution
   * Latin Hypercube Sampling (LHS) for initialization
   * Equidistant selection crowding distance

4. **Comprehensive Benchmarking**
   
   * Multiple loss functions: initial condition, boundary condition, PDE residual, peak value, and trace
   * Hardware-aware quantum circuit optimization
   * Noise resilience evaluation
   * Pareto front analysis

System Requirements
-------------------

* Python 3.12+
* PyTorch 2.0+
* PennyLane 0.39+
* C++17 compiler (for NSGA-II optimizer)
* CUDA-capable GPU (recommended)

Installation
------------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/yourusername/pinns-qpinns-benchmark.git
   cd pinns-qpinns-benchmark

   # Install Python dependencies
   pip install -r requirements.txt

   # Build the C++ NSGA-II optimizer
   python setup.py build_ext --inplace

Quick Start
-----------

.. code-block:: python

   from pinns_d3 import main

   # Run the complete benchmark
   main()

This will execute both PINN and QPINN training with NSGA-II optimization and generate comparative visualizations.

Documentation Structure
-----------------------

* **Theoretical Background**: Mathematical foundations and algorithm descriptions
* **PINNs Implementation**: Detailed explanation of the classical approach with FNO
* **QPINNs Implementation**: Quantum circuit design and optimization strategies
* **NSGA-II Optimizer**: C++ implementation details and Python bindings
* **Experimental Results**: Benchmark results and analysis
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
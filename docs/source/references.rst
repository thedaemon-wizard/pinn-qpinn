References
==========

This chapter provides a comprehensive list of scientific references used in the implementation of PINNs and QPINNs for solving the heat conduction equation.

Physics-Informed Neural Networks (PINNs)
----------------------------------------

Foundational Works
^^^^^^^^^^^^^^^^^^

* **Raissi et al. (2019)** "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations" *Journal of Computational Physics*, 378, 686-707.

* **Karniadakis et al. (2021)** "Physics-informed machine learning" *Nature Reviews Physics*, 3(6), 422-440.

Separable PINNs (SPINN)
^^^^^^^^^^^^^^^^^^^^^^^^

* **Cho et al. (2023)** "Separable Physics-Informed Neural Networks" *NeurIPS 2023 Spotlight*. arXiv:2306.15969

  - Per-axis body networks reducing O(N^d) to O(Nd) complexity
  - Hadamard product aggregation for low-rank tensor approximation
  - Efficient evaluation on high-dimensional PDEs

PINNsFormer
^^^^^^^^^^^^

* **Zhao, Ding & Prakash (2024)** "PINNsFormer: A Transformer-Based Framework for Physics-Informed Neural Networks" *ICLR 2024*. arXiv:2307.11833

  - Wavelet activation function (learnable sin/cos combination)
  - Pseudo-sequence generation for Transformer processing
  - Encoder-decoder architecture with Wavelet residual connections

Fourier Neural Operators
^^^^^^^^^^^^^^^^^^^^^^^^

* **Li et al. (2021)** "Fourier Neural Operator for Parametric Partial Differential Equations" *ICLR 2021*. arXiv:2010.08895

  - Introduces the FNO architecture for learning mappings between function spaces
  - Demonstrates superior performance on various PDE benchmarks

* **Li et al. (2023)** "Fourier Neural Operator: Learning Resolution-Invariant Operators on Manifolds" *Journal of Machine Learning Research*

Training Challenges and Solutions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Wang et al. (2022)** "When and why PINNs fail to train: A neural tangent kernel perspective" *SIAM Journal on Scientific Computing*, 44(5), A3016-A3040.
  
  - Analyzes failure modes of PINNs from NTK perspective
  - Proposes adaptive weight strategies

* **Krishnapriyan et al. (2021)** "Characterizing possible failure modes in physics-informed neural networks" *NeurIPS 2021*.

  - Systematic study of PINN failure modes
  - Guidelines for robust PINN training

Adaptive Loss Balancing
^^^^^^^^^^^^^^^^^^^^^^^

* **Bischof & Kraus (2025)** "Multi-Objective Loss Balancing for Physics-Informed Deep Learning" *Computer Methods in Applied Mechanics and Engineering*, 431, 117521.

  - ReLoBRaLo (Relative Loss Balancing with Random Lookback)
  - Softmax-based adaptive weighting with exponential moving average
  - Applicable to both classical and quantum PINN training

* **Wang et al. (2022)** "Respecting Causality for Training Physics-Informed Neural Networks" *Computer Methods in Applied Mechanics and Engineering*, 397, 115135.

  - Causal loss weighting for temporal PDEs
  - Temporal causality constraints

Hard Boundary Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^

* **Sukumar & Srivastava (2022)** "Exact imposition of boundary conditions with distance functions in physics-informed deep neural networks" *Computer Methods in Applied Mechanics and Engineering*, 389, 114333.

  - Distance function approach for exact Dirichlet BC enforcement
  - Product distance functions for rectangular domains
  - Eliminates boundary loss term from the objective

Optimization
^^^^^^^^^^^^

* **Liu & Nocedal (1989)** "On the limited memory BFGS method for large scale optimization" *Mathematical Programming*, 45, 503-528.

  - L-BFGS second-order optimization for PINN refinement
  - Strong Wolfe line search
  - Used as refinement phase after first-order warm-up


Quantum Physics-Informed Neural Networks (QPINNs)
-------------------------------------------------

Foundational QPINNs
^^^^^^^^^^^^^^^^^^^

* **Trahan et al. (2024)** "Quantum Physics-Informed Neural Networks: Applications to Heat Equations" *Entropy*, 26(8):649.
  
  - Comprehensive study of QPINNs for heat equation
  - Recommends tanh activation for quantum circuits
  - Analyzes noise effects on NISQ devices

* **Panichi et al. (2025)** "Quantum physics informed neural networks for multi-variable partial differential equations" *arXiv:2503.12244*
  
  - Extension to multi-variable PDEs
  - Feature decomposition techniques
  - Comparative analysis with classical methods

Trainable Embeddings
^^^^^^^^^^^^^^^^^^^^

* **TE-QPINN (2025)** "Trainable embedding quantum physics informed neural networks for solving partial differential equations" *Scientific Reports* (in press)
  
  - Introduces learnable embedding functions
  - Polynomial and Fourier basis functions
  - Improved expressivity over fixed encodings

Quantum Circuit Generation
^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Apak et al. (2024)** "KetGPT – Dataset Augmentation of Quantum Circuits using Transformers" *arXiv:2402.13352*
  
  - Large-scale dataset of quantum circuits
  - Transformer-based generation model
  - Pre-training strategies for circuit design

* **Nakaji & Yamamoto (2021)** "Quantum circuit design by Generative Quantum Eigensolver" *arXiv:2106.10985*
  
  - GQE framework for circuit optimization
  - Variational approach to circuit design

Hardware Considerations
^^^^^^^^^^^^^^^^^^^^^^^

* **Kandala et al. (2017)** "Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets" *Nature*, 549(7671), 242-246.
  
  - Hardware-efficient ansätze design
  - Gate time and error rate analysis
  - Connectivity constraints

* **Temme et al. (2017)** "Error mitigation for short-depth quantum circuits" *Physical Review Letters*, 119(18), 180509.
  
  - Zero-noise extrapolation
  - Probabilistic error cancellation
  - Practical implementation strategies



Additional References
---------------------

Numerical Methods
^^^^^^^^^^^^^^^^^

* **LeVeque (2007)** "Finite Difference Methods for Ordinary and Partial Differential Equations" *SIAM*
  
  - Reference solutions for heat equation
  - Stability analysis

Machine Learning Theory
^^^^^^^^^^^^^^^^^^^^^^^

* **Vaswani et al. (2017)** "Attention is all you need" *NeurIPS 2017*
  
  - Transformer architecture
  - Self-attention mechanism

* **Goodfellow et al. (2016)** "Deep Learning" *MIT Press*
  
  - Neural network fundamentals
  - Training techniques

Quantum Computing
^^^^^^^^^^^^^^^^^

* **Nielsen & Chuang (2010)** "Quantum Computation and Quantum Information" *Cambridge University Press*
  
  - Quantum computing fundamentals
  - Quantum circuits and gates

* **Preskill (2018)** "Quantum Computing in the NISQ era and beyond" *Quantum*, 2, 79.
  
  - NISQ device characteristics
  - Near-term applications

* **Cerezo et al. (2021)** "Cost function dependent barren plateaus in shallow parametrized quantum circuits" *Nature Communications*, 12, 1791.
  
  - Barren plateau analysis
  - Energy estimation quality metrics

* **Larocca et al. (2022)** "Diagnosing Barren Plateaus with Tools from Quantum Optimal Control" *Quantum*, 6, 824.
  
  - Quantum control theory
  - Trainability diagnostics

Circuit Optimization
^^^^^^^^^^^^^^^^^^^^

* **Li et al. (2020)** "Quantum optimization with a novel Gibbs objective function and ansatz architecture search" *Physical Review Research*, 2, 013020.
  
  - Energy landscape analysis
  - Circuit architecture search

* **Skolik et al. (2023)** "Equivariant quantum circuits for learning on weighted graphs" *npj Quantum Information*, 9, 47.
  
  - Parameter distribution analysis
  - Gradient flow optimization

Software and Tools
^^^^^^^^^^^^^^^^^^

* **Bergholm et al. (2022)** "PennyLane: Automatic differentiation of hybrid quantum-classical computations" *arXiv:1811.04968*
  
  - Quantum machine learning framework
  - Differentiable quantum computing

* **Paszke et al. (2019)** "PyTorch: An imperative style, high-performance deep learning library" *NeurIPS 2019*
  
  - Deep learning framework
  - Automatic differentiation

Implementation Resources
------------------------

GitHub Repositories
^^^^^^^^^^^^^^^^^^^

* **PINNs Official Repository**: https://github.com/maziarraissi/PINNs
* **FNO Repository**: https://github.com/neuraloperator/neuraloperator
* **PennyLane**: https://github.com/PennyLaneAI/pennylane
* **ketGPT**: Available through PennyLane data module

Datasets
^^^^^^^^

* **KetGPT Dataset**: Available through PennyLane data loader
  
  .. code-block:: python
  
     import pennylane as qml
     [ketgpt_dataset] = qml.data.load("ketgpt")

* **PDE Benchmarks**: Various benchmark datasets for PDE solving

Citation Format
---------------

When citing this work, please use:

.. code-block:: bibtex

   @article{pinns_qpinns_benchmark_2025,
     title={Comprehensive Benchmark of Physics-Informed Neural Networks 
            and Quantum Physics-Informed Neural Networks for 3D Heat Conduction},
     author={Research Team},
     journal={arXiv preprint},
     year={2025}
   }
API Reference
=============

This chapter provides detailed API documentation for all modules in the PINNs-QPINNs benchmark implementation.

Python API
----------

The codebase is organized into 8 modular Python files plus a JSON configuration file:

* ``config.py`` -- Centralised configuration, hyperparameters, and JSON config loading (``load_benchmark_config``, ``get_nested``, ``apply_config``)
* ``physics.py`` -- Physics equations, analytical solutions, and training data generation
* ``device_manager.py`` -- Backend detection and device allocation (auto/cpu/cuda/gpu/qpu)
* ``pinnsformer.py`` -- SPINN body networks + PINNsFormer Transformer architecture (Wavelet activation, separable per-axis MLPs, encoder-decoder)
* ``gpt_circuit.py`` -- GQE-GPT quantum circuit generation and ketGPT integration
* ``pinn_model.py`` -- SPINN+PINNsFormer PINN model, training loop with RAdam + L-BFGS + ReLoBRaLo, validation and performance monitoring
* ``qpinn_model.py`` -- Quantum PINN model, training loop with SPSA + ReLoBRaLo, validation monitoring
* ``main.py`` -- CLI entry point (``--backend``, ``--config``), orchestration, CSV/JSON/PNG output
* ``benchmark_config.json`` -- External JSON configuration file with all hyperparameters

config Module
^^^^^^^^^^^^^

.. module:: config

Centralised configuration for the benchmark.  This module defines global constants
(``alpha``, ``L``, ``T``, ``sigma_0``, grid sizes), configuration dataclasses, and
functions for loading external JSON configuration files.

.. autoclass:: BackendConfig
   :members:
   :undoc-members:

.. autoclass:: TrainingConfig
   :members:
   :undoc-members:

.. autofunction:: load_benchmark_config

   Load benchmark configuration from a JSON file and return a dict.

   :param path: Path to the JSON config file (e.g. ``benchmark_config.json``).
   :returns: Configuration dictionary with nested sections for ``physics``,
             ``grid``, ``pinn``, ``qpinn``, ``validation``, ``parallel``, etc.

.. autofunction:: get_nested

   Safely access nested dict values using dot-separated keys.

   :param config: Configuration dictionary.
   :param dotted_key: Dot-separated key path (e.g. ``'pinn.training.lr'``).
   :param default: Default value if the key path does not exist.
   :returns: The value at the key path, or *default*.

   Example::

       lr = get_nested(config, 'pinn.training.lr', 1e-3)

.. autofunction:: apply_config

   Apply a loaded JSON config dict to module-level global variables.  Overrides
   ``alpha``, ``L``, ``T``, ``sigma_0``, grid sizes (``nx``, ``ny``, ``nz``,
   ``nt``), epoch counts, parallelism settings, and ``OMP_NUM_THREADS``.

physics Module
^^^^^^^^^^^^^^

.. module:: physics

Physics equations and data generation.

.. autofunction:: analytical_solution
.. autofunction:: initial_condition
.. autofunction:: boundary_condition
.. autofunction:: compute_analytical_solution
.. autofunction:: calculate_metrics
.. autofunction:: calculate_full_metrics

   Compute comprehensive metrics: MSE, RMSE, MAE, MaxAE, RelL2, RelMAE,
   peak accuracy, energy conservation per timestep, boundary satisfaction,
   and non-negativity statistics.

   :param u_pred: Predicted solution array (flattened).
   :param u_true: Analytical/reference solution array (flattened).
   :returns: Dictionary with all computed metrics.

.. autofunction:: to_python_float

.. autoclass:: TrainingPoint
   :members:
   :undoc-members:

device_manager Module
^^^^^^^^^^^^^^^^^^^^^

.. module:: device_manager

Backend detection and quantum device management.

.. autoclass:: QuantumDeviceManager
   :members:
   :undoc-members:

.. autoclass:: QPUConfig
   :members:
   :undoc-members:

pinnsformer Module
^^^^^^^^^^^^^^^^^^

.. module:: pinnsformer

SPINN body networks + PINNsFormer Transformer architecture components.

.. autoclass:: WaveletActivation
   :members:
   :show-inheritance:

   Learnable wavelet activation: :math:`f(x) = w_1 \sin(x) + w_2 \cos(x)`.

.. autoclass:: SPINNBodyNetwork
   :members:
   :show-inheritance:

   Single-axis body network for SPINN: small MLP mapping R^1 -> R^r with Wavelet activation.

.. autoclass:: SPINNAggregator
   :members:
   :show-inheritance:

   Aggregates per-axis SPINN body network outputs via Hadamard product + learned projection.

.. autoclass:: PseudoSequenceGenerator
   :members:
   :show-inheritance:

   Converts point-wise (x,y,z,t) inputs into temporal sequences for Transformer processing.

.. autoclass:: SpatioTemporalMixer
   :members:
   :show-inheritance:

   Dual attention mechanism for spatial and temporal feature mixing (backward compatibility).

.. autoclass:: PINNsFormerEncoder
   :members:
   :show-inheritance:

   Transformer encoder stack with WaveletActivation in the feedforward network.

.. autoclass:: PINNsFormerDecoder
   :members:
   :show-inheritance:

   Optional Transformer decoder with encoder-decoder cross-attention.

.. autoclass:: SPFormerDecoderBlock
   :members:
   :show-inheritance:

   S-PFormer (2025) decoder block: self-attention + FFN with pre-norm and scaled residuals.

.. autoclass:: SPFormerDecoder
   :members:
   :show-inheritance:

   Multi-layer S-PFormer decoder-only Transformer with final LayerNorm.

.. autoclass:: OutputProjection
   :members:
   :show-inheritance:

   Projects sequence representations to scalar PDE solution values.

.. autoclass:: TransformerBlock
   :members:
   :show-inheritance:

   Alias for PINNsFormerEncoderLayer (backward compatibility).

gpt_circuit Module
^^^^^^^^^^^^^^^^^^

.. module:: gpt_circuit

GQE-GPT quantum circuit generation.

.. autoclass:: QuantumCircuitGPT
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: QuantumCircuitTemplate
   :members:
   :undoc-members:
   :show-inheritance:

pinn_model Module
^^^^^^^^^^^^^^^^^

.. module:: pinn_model

Classical PINN implementation with SPINN + PINNsFormer architecture.

.. autoclass:: PINN
   :members:
   :undoc-members:
   :special-members: __init__
   :show-inheritance:

Key methods added for accuracy improvements and monitoring:

.. method:: PINN._compute_validation_metrics(grid_size=10)

   Compute validation metrics on a held-out uniform grid.  Creates an
   :math:`N \times N \times N \times N` evaluation grid (default :math:`N = 10`,
   configurable via ``validation.grid_size`` in the JSON config), evaluates the
   model in ``eval()`` mode, and compares against the analytical Fourier sine
   series solution.

   :param grid_size: Number of points per spatial/temporal dimension.
   :returns: Dict with ``mse``, ``rel_l2``, ``min_pred``, ``max_pred``,
             ``neg_count``, and ``per_timeslice`` sub-dict containing per-time-slice
             MSE and relative L2 error.

   Called every ``validation.interval_pinn`` epochs (default 200) during
   ``train_radam()``.

.. staticmethod:: PINN._collect_performance_metrics()

   Collect GPU and CPU performance metrics using PyTorch CUDA APIs and ``psutil``.

   :returns: Dict with keys ``gpu_memory_allocated_mb``, ``gpu_memory_reserved_mb``,
             ``gpu_max_memory_mb``, ``gpu_util_pct``, ``gpu_temp_c``, ``cpu_pct``,
             ``ram_pct``, ``ram_used_gb``.  GPU utilisation is obtained via
             ``nvidia-smi``; if unavailable, returns ``-1.0``.

qpinn_model Module
^^^^^^^^^^^^^^^^^^

.. module:: qpinn_model

Quantum PINN implementation with GQE-GPT circuit generation.

.. autoclass:: GQEQuantumPINN
   :members:
   :undoc-members:
   :special-members: __init__
   :show-inheritance:

.. method:: GQEQuantumPINN._compute_validation_metrics(grid_size=5)

   Compute validation metrics on a held-out uniform grid for the QPINN.
   Uses a smaller default grid (:math:`5 \times 5 \times 5 \times 5`) than
   the PINN because quantum circuit evaluation is significantly slower.

   :param grid_size: Number of points per spatial/temporal dimension.
   :returns: Dict with ``mse``, ``rel_l2``, and ``per_timeslice`` sub-dict.

   Called every ``validation.interval_qpinn`` steps (default 50) during
   ``train_spsa()``.

main Module
^^^^^^^^^^^

.. module:: main

CLI entry point and benchmark orchestration.  Accepts ``--backend`` for device
selection and ``--config`` for loading an external JSON configuration file.

.. autofunction:: main
.. autofunction:: setup_logging
.. autofunction:: visualize_results_comparison
.. autofunction:: save_comparative_results
.. autofunction:: save_summary_report
.. autofunction:: save_pinn_training_csv
.. autofunction:: save_qpinn_training_csv
.. autofunction:: save_metrics_over_time_csv

.. autofunction:: save_validation_csv

   Write PINN and/or QPINN validation metrics collected during training to CSV.
   Produces ``pinn_validation_metrics.csv`` and ``qpinn_validation_metrics.csv``
   with columns for epoch/step, elapsed time, global MSE, relative L2 error, and
   per-time-slice metrics.

   :param results_dir: Output directory.
   :param pinn_history: PINN training history dict (from ``train_radam``).
   :param qpinn_history: QPINN training history dict (from ``train_spsa``).
   :returns: List of file paths written.

.. autofunction:: save_performance_csv

   Write GPU/CPU performance metrics collected during PINN training to CSV.
   Produces ``performance_metrics.csv`` with columns for epoch, elapsed time,
   throughput, GPU memory, GPU utilisation, CPU percent, and RAM usage.

   :param results_dir: Output directory.
   :param pinn_history: PINN training history dict containing ``performance_history``.
   :returns: File path written, or ``None`` if no data available.

.. autofunction:: save_checkpoint
.. autofunction:: load_checkpoint
.. autofunction:: check_existing_checkpoints

Constants (defined in ``config.py``)
-------------------------------------

Physical Constants
^^^^^^^^^^^^^^^^^^

.. data:: config.L
   :type: float
   :value: 1.0

   Domain length in each spatial dimension

.. data:: config.T_FINAL
   :type: float
   :value: 1.0

   Total simulation time

.. data:: config.alpha
   :type: float
   :value: 0.01

   Thermal diffusivity coefficient

.. data:: config.sigma_0
   :type: float
   :value: 0.05

   Standard deviation for Gaussian initial condition

Numerical Constants
^^^^^^^^^^^^^^^^^^^

.. data:: config.nx
   :type: int
   :value: 20

   Number of grid points in x direction

.. data:: config.ny
   :type: int
   :value: 20

   Number of grid points in y direction

.. data:: config.nz
   :type: int
   :value: 20

   Number of grid points in z direction

.. data:: config.nt
   :type: int
   :value: 20

   Number of time steps

Example Usage
-------------

Basic PINN Training
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pinn_model import PINN, train_pinn

   # Create and train PINN with RAdam + ReLoBRaLo
   model = PINN(
       use_hard_constraints=True,
       fourier_features=True,
       use_transformer=True,
       transformer_memory_efficient=True
   )

   # Train with RAdam optimizer
   model, losses = train_pinn(model, epochs=100)

QPINN with GQE-GPT
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from qpinn_model import GQEQuantumPINN

   # Initialize QPINN
   qpinn = GQEQuantumPINN(
       n_qubits=6,
       backend='default.mixed',
       shots=1000,
       noise_model='realistic',
       use_gpt_circuit_generation=True
   )

   # Train with SPSA + ReLoBRaLo
   params, losses, training_time = qpinn.train_spsa(
       n_samples=1500, max_iterations=200
   )

   # Evaluate
   u_qpinn = qpinn.evaluate()

CLI Usage
^^^^^^^^^

.. code-block:: bash

   # Auto-detect backend
   python main.py --backend auto

   # Force CPU
   python main.py --backend cpu

   # Use CUDA GPU
   python main.py --backend cuda

   # Load all hyperparameters from external JSON config
   python main.py --backend auto --config benchmark_config.json

The ``--config`` flag loads an external JSON file (see ``benchmark_config.json``
for the full schema).  The config overrides physics constants, grid sizes,
training epochs, optimizer hyperparameters, curriculum learning phases,
validation intervals, and parallel-processing settings via the
:func:`config.load_benchmark_config` / :func:`config.apply_config` pipeline.

Thread Safety
-------------

* **PINN**: Thread-safe for inference, not for training
* **QPINN**: Not thread-safe due to quantum simulator limitations

Backend Support
---------------

* **PINN**: Full GPU acceleration with CUDA via ``device_manager.py``
* **QPINN**: GPU simulation with lightning.gpu backend; QPU-ready
* **Backend options**: auto, cpu, cuda, gpu, qpu
* **Memory requirements**: Minimum 8GB VRAM recommended for GPU backends

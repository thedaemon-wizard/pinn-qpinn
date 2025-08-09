API Reference
=============

This chapter provides detailed API documentation for all modules in the PINNs-QPINNs benchmark implementation.

Python API
----------

pinns_d3 Module
^^^^^^^^^^^^^^^

Main module containing PINN and QPINN implementations.

.. module:: pinns_d3

Classes
~~~~~~~

.. autoclass:: PINN
   :members:
   :undoc-members:
   :special-members: __init__
   :show-inheritance:

   .. automethod:: forward(x, y, z, t)
   .. automethod:: compute_pde_residual(x, y, z, t)
   .. automethod:: train_with_nsga2(n_samples=10000, nsga2_config=None)

.. autoclass:: GQEQuantumPINN
   :members:
   :undoc-members:
   :special-members: __init__
   :show-inheritance:

   .. automethod:: forward(inputs)
   .. automethod:: train_with_nsga2(n_samples=10000, nsga2_config=None)
   .. automethod:: evaluate()

.. autoclass:: QuantumCircuitTemplate
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: QuantumCircuitGPT
   :members:
   :undoc-members:
   :show-inheritance:

Functions
~~~~~~~~~

.. autofunction:: train_pinn_nsga2
.. autofunction:: train_qpinn_gqe_gpt
.. autofunction:: compute_analytical_solution
.. autofunction:: calculate_metrics
.. autofunction:: visualize_results_nsga2_comparison

Neural Network Components
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SpectralConv3d
   :members:
   :show-inheritance:

   3D Spectral Convolution layer for Fourier Neural Operator.

   :param in_channels: Number of input channels
   :param out_channels: Number of output channels
   :param modes1: Number of Fourier modes in x direction
   :param modes2: Number of Fourier modes in y direction
   :param modes3: Number of Fourier modes in z direction

.. autoclass:: TemporalAttention
   :members:
   :show-inheritance:

   Temporal attention mechanism for enhanced time-dependent learning.

   :param hidden_dim: Hidden dimension size for attention computation

Quantum Components
~~~~~~~~~~~~~~~~~~

.. autoclass:: MultiObjectiveBayesianCircuitOptimizer
   :members:
   :undoc-members:
   :show-inheritance:

   Multi-objective Bayesian optimizer for quantum circuit search with 9 objectives.

   .. automethod:: evaluate_circuit_multi_objective(template, training_data=None)
   
   Evaluates 9 objectives:
   
   1. Hardware efficiency (inverse)
   2. Noise resilience (inverse)
   3. Expressivity (inverse)
   4. Mitigation compatibility (inverse)
   5. Trainability (inverse)
   6. Entanglement capability (inverse)
   7. Circuit depth (normalized)
   8. Parameter efficiency (inverse)
   9. Energy estimation quality (inverse)
   
   :param template: Quantum circuit template
   :param training_data: Optional training data for energy estimation
   :return: torch.Tensor of shape (9,) with objective values

Data Structures
~~~~~~~~~~~~~~~

.. autoclass:: TrainingPoint
   :members:
   :show-inheritance:

   Data structure for training points.

   :param x: X coordinate
   :param y: Y coordinate  
   :param z: Z coordinate
   :param t: Time value
   :param u_true: True solution value (optional)

Configuration
~~~~~~~~~~~~~

.. data:: NSGA2_COMMON_CONFIG

   Common configuration for NSGA-II optimization.

   .. code-block:: python

      NSGA2_COMMON_CONFIG = {
          'population_size_pinn': 50,
          'population_size_qpinn': 30,
          'max_generations_pinn': 100,
          'max_generations_qpinn': 200,
          'n_parents': 3,
          'n_children_pinn': 10,
          'n_children_qpinn': 5,
          'random_seed': 42,
          'progress_interval': 10
      }

C++ API (NSGA-II Optimizer)
---------------------------

.. doxygennamespace:: nsga2
   :project: nsga2_optimizer
   :members:
   :undoc-members:

Core Classes
~~~~~~~~~~~~

.. doxygenclass:: nsga2::NSGA2Optimizer
   :project: nsga2_optimizer
   :members:
   :protected-members:
   :private-members:

.. doxygenclass:: nsga2::Individual
   :project: nsga2_optimizer
   :members:

.. doxygenclass:: nsga2::Population
   :project: nsga2_optimizer
   :members:

Configuration
~~~~~~~~~~~~~

.. doxygenstruct:: nsga2::NSGA2Config
   :project: nsga2_optimizer
   :members:

Crossover Operators
~~~~~~~~~~~~~~~~~~~

.. doxygenclass:: nsga2::REXCrossover
   :project: nsga2_optimizer
   :members:

.. doxygenclass:: nsga2::PolynomialMutation
   :project: nsga2_optimizer
   :members:

Utility Classes
~~~~~~~~~~~~~~~

.. doxygenclass:: nsga2::LatinHypercubeSampler
   :project: nsga2_optimizer
   :members:

.. doxygenclass:: nsga2::CrowdingDistanceCalculator
   :project: nsga2_optimizer
   :members:

Python Bindings
---------------

The C++ NSGA-II optimizer is exposed to Python through pybind11 bindings.

.. py:module:: nsga2_optimizer

.. py:class:: NSGA2Config

   Configuration class for NSGA-II optimizer.

   .. py:attribute:: population_size
      :type: int
      :value: 100

      Size of the population

   .. py:attribute:: max_generations
      :type: int
      :value: 100

      Maximum number of generations

   .. py:attribute:: n_objectives
      :type: int
      :value: 2

      Number of objectives to optimize

   .. py:attribute:: lower_bounds
      :type: List[float]

      Lower bounds for each parameter

   .. py:attribute:: upper_bounds
      :type: List[float]

      Upper bounds for each parameter

.. py:class:: NSGA2Optimizer(config: NSGA2Config)

   Main NSGA-II optimizer class.

   .. py:method:: optimize(objectives: List[Callable], callback: Optional[Callable] = None, batch_evaluator: Optional[Callable] = None) -> Tuple[List[List[float]], List[List[float]]]

      Run NSGA-II optimization.

      :param objectives: List of objective functions
      :param callback: Optional callback function called each generation
      :param batch_evaluator: Optional batch evaluation function
      :return: Tuple of (parameters, objectives) for Pareto front

   .. py:method:: get_pareto_front() -> List[Dict]

      Get the current Pareto front.

      :return: List of dictionaries containing individual information

Utility Functions
-----------------

Mathematical Functions
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pinns_d3.initial_condition
.. autofunction:: pinns_d3.boundary_condition
.. autofunction:: pinns_d3.analytical_solution

Visualization Functions
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pinns_d3.plot_solution_3d_slices
.. autofunction:: pinns_d3.plot_error_distribution
.. autofunction:: pinns_d3.plot_pareto_fronts

Helper Functions
^^^^^^^^^^^^^^^^

.. autofunction:: pinns_d3.set_random_seeds
.. autofunction:: pinns_d3.get_device
.. autofunction:: pinns_d3.save_checkpoint
.. autofunction:: pinns_d3.load_checkpoint

Constants
---------

Physical Constants
^^^^^^^^^^^^^^^^^^

.. data:: L
   :type: float
   :value: 1.0

   Domain length in each spatial dimension

.. data:: T
   :type: float
   :value: 0.1

   Total simulation time

.. data:: alpha
   :type: float
   :value: 0.01

   Thermal diffusivity coefficient

.. data:: u0_max
   :type: float
   :value: 10.0

   Maximum initial temperature

.. data:: sigma
   :type: float
   :value: 0.1

   Standard deviation for Gaussian initial condition

Numerical Constants
^^^^^^^^^^^^^^^^^^^

.. data:: nx
   :type: int
   :value: 20

   Number of grid points in x direction

.. data:: ny
   :type: int
   :value: 20

   Number of grid points in y direction

.. data:: nz
   :type: int
   :value: 20

   Number of grid points in z direction

.. data:: nt
   :type: int
   :value: 20

   Number of time steps

Error Handling
--------------

Custom Exceptions
^^^^^^^^^^^^^^^^^

.. autoexception:: pinns_d3.ConvergenceError
   :show-inheritance:

.. autoexception:: pinns_d3.CircuitGenerationError
   :show-inheritance:

.. autoexception:: pinns_d3.OptimizationError
   :show-inheritance:

Example Usage
-------------

Basic PINN Training
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pinns_d3 import train_pinn_nsga2, compute_analytical_solution

   # Train PINN with NSGA-II
   model, losses, training_time = train_pinn_nsga2(
       use_hard_constraints=True,
       use_fno=True,
       use_temporal_attention=True
   )

   # Evaluate on test grid
   u_pred = model.evaluate()
   u_true = compute_analytical_solution()

   # Calculate metrics
   mse, rel_l2 = calculate_metrics(u_pred, u_true)

QPINN with GQE-GPT
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pinns_d3 import GQEQuantumPINN

   # Initialize QPINN
   qpinn = GQEQuantumPINN(
       n_qubits=6,
       backend='default.mixed',
       shots=1000,
       noise_model='realistic',
       use_gpt_circuit_generation=True
   )

   # Train with NSGA-II
   state_dict, losses, time = qpinn.train_with_nsga2(
       n_samples=20000,
       nsga2_config=NSGA2_COMMON_CONFIG
   )

   # Evaluate
   u_qpinn = qpinn.evaluate()

Thread Safety
-------------

* **PINN**: Thread-safe for inference, not for training
* **QPINN**: Not thread-safe due to quantum simulator limitations
* **NSGA-II**: Thread-safe with OpenMP parallelization

GPU Support
-----------

* **PINN**: Full GPU acceleration with CUDA
* **QPINN**: GPU simulation with lightning.gpu backend
* **Memory requirements**: Minimum 8GB VRAM recommended
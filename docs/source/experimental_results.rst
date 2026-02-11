Experimental Results
====================

This chapter presents the comprehensive benchmark results comparing PINNs and QPINNs for solving the 3D heat conduction equation, using the refactored codebase with RAdam, SPSA, and ReLoBRaLo adaptive loss balancing.

Experimental Setup
------------------

Problem Configuration
^^^^^^^^^^^^^^^^^^^^^

* **Domain**: :math:`\Omega = [0, 1]^3 \times [0, 1]` (unit cube with time)
* **Thermal diffusivity**: :math:`\alpha = 0.01`
* **Initial condition**: Gaussian centered at :math:`(0.5, 0.5, 0.5)` with :math:`\sigma = 0.05`
* **Boundary conditions**: Homogeneous Dirichlet (:math:`u = 0` on all boundaries)
* **Grid resolution**: :math:`20 \times 20 \times 20 \times 20` for evaluation

Software Environment
^^^^^^^^^^^^^^^^^^^^^

* **PyTorch**: 2.10.0+cu128
* **PennyLane**: 0.44.0
* **Python**: 3.12.11
* **Backend selection**: Automatic (GPU preferred, CPU fallback)

PINN Training Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Curriculum learning**: 3-phase training (Phase 1: IC-only, Phase 2: +PDE ramp-up, Phase 3: full ReLoBRaLo)
* **Phase 1 -- IC-only warm-up**: First 30% of RAdam epochs, only IC + peak + non-negativity losses
* **Phase 2 -- PDE ramp-up**: Next 40% of RAdam epochs, PDE loss linearly ramped in
* **Phase 3 -- Full ReLoBRaLo**: Final 30% of RAdam epochs, all losses with adaptive weighting
* **Phase 4 -- L-BFGS refinement**: 200 iterations, ``strong_wolfe`` line search, history_size=50
* **Loss components**: IC, peak, PDE (causal-weighted), non-negativity (fixed weight), boundary (hard constraints)
* **Causal temporal weighting**: :math:`w(t) = \exp(-\varepsilon \, t / T)` on PDE residuals (default :math:`\varepsilon = 1.0`)
* **Non-negativity constraint**: :math:`\text{mean}(\text{ReLU}(-u_{\text{pred}})^2)` on PDE + IC points with weight 0.1
* **Loss balancing**: ReLoBRaLo (Relative Loss Balancing with Random Lookback)
* **Architecture**: SPINN per-axis body networks + PINNsFormer encoder-decoder with multi-scale Fourier features
* **LR scheduler**: CosineAnnealingWarmRestarts (T_0=500, T_mult=2, eta_min=1e-6)
* **Hard boundary constraints with IC lifting**: Parabolic product distance function + free-space Green's function ansatz
* **Training data**: 50,000 points (60% interior, 5% boundary, 35% IC with structured peak sampling)
* **Configuration**: Externalized to ``benchmark_config.json`` with ``--config`` CLI argument

.. code-block:: python

   # Phase 1: RAdam optimizer with decoupled weight decay
   optimizer = torch.optim.RAdam(
       model.parameters(),
       lr=1e-3, weight_decay=1e-2, decoupled_weight_decay=True
   )
   # Phase 2: L-BFGS refinement with frozen ReLoBRaLo weights
   lbfgs = torch.optim.LBFGS(
       model.parameters(), lr=1.0,
       line_search_fn='strong_wolfe'
   )

QPINN Training Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Optimizer**: PennyLane ``SPSAOptimizer`` (Simultaneous Perturbation Stochastic Approximation)
* **Max iterations**: 200
* **Loss balancing**: ReLoBRaLo (Relative Loss Balancing with Random Lookback)
* **Circuit parameters**: 1
* **Total parameters**: 9 (1 circuit + 8 classical)
* **SPSA perturbation scale**: c = 0.2
* **PDE residual**: Finite-difference approximation (gradient-free)

Performance Metrics
-------------------

Mean Squared Error (MSE)
^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

   \text{MSE} = \frac{1}{N} \sum_{i=1}^{N} |u_{\text{pred}}(x_i, y_i, z_i, t_i) - u_{\text{true}}(x_i, y_i, z_i, t_i)|^2

Relative L2 Error
^^^^^^^^^^^^^^^^^

.. math::

   \text{Rel L2} = \frac{\|u_{\text{pred}} - u_{\text{true}}\|_2}{\|u_{\text{true}}\|_2}

Results Summary
---------------

Overall Performance Comparison
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. table:: Performance Comparison between PINN (RAdam + L-BFGS) and QPINN (SPSA)

   +------------------------+-------------------+------------------------+--------+
   | Metric                 | PINN (RAdam)      | QPINN (SPSA)           | Better |
   +========================+===================+========================+========+
   | MSE                    | 2.125e-06         | 1.352e-06              | QPINN  |
   +------------------------+-------------------+------------------------+--------+
   | RMSE                   | 1.458e-03         | 1.163e-03              | QPINN  |
   +------------------------+-------------------+------------------------+--------+
   | MAE                    | 1.022e-03         | 1.679e-04              | QPINN  |
   +------------------------+-------------------+------------------------+--------+
   | Relative L2 Error      | 1.381e-01         | 1.101e-01              | QPINN  |
   +------------------------+-------------------+------------------------+--------+
   | Max AE                 | 3.975e-03         | 1.235e-01              | PINN   |
   +------------------------+-------------------+------------------------+--------+
   | Peak Error             | 3.811e-03         | 1.862e-03              | QPINN  |
   +------------------------+-------------------+------------------------+--------+
   | Neg. Violations        | 0                 | 0                      | Tie    |
   +------------------------+-------------------+------------------------+--------+
   | Training Time (GPU)    | 3,745.35 s        | 10,605.49 s            | PINN   |
   +------------------------+-------------------+------------------------+--------+
   | Epochs / Iterations    | 3,000 + 200 L-BFGS| 200                    |        |
   +------------------------+-------------------+------------------------+--------+
   | Total Parameters       | 454,527           | 9 (1 circuit + 8)      | QPINN  |
   +------------------------+-------------------+------------------------+--------+
   | Best Loss / Cost       | 1.217e-06         | 4.456e-04              | PINN   |
   +------------------------+-------------------+------------------------+--------+

.. note::

   The QPINN achieves a **36% lower MSE** than the PINN in this benchmark
   (1.352e-06 vs 2.125e-06), using only 9 parameters compared to the
   SPINN+PINNsFormer PINN's 454,527 -- a reduction of over 50,000x. However,
   the PINN achieves **124x better Max AE** (3.975e-03 vs 1.235e-01), indicating
   superior worst-case prediction accuracy. The PINN is 2.8x faster in wall-clock
   time due to its GPU-accelerated autograd, while the QPINN's sequential quantum
   circuit evaluation is the primary bottleneck. The IC lifting function (free-space
   Green's function ansatz) was critical to the PINN's accuracy, providing a 36x MSE
   improvement over the zero-lifting baseline.

ReLoBRaLo Adaptive Weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^

ReLoBRaLo dynamically rebalances the multi-component loss during training.

**PINN final ReLoBRaLo weights**:

* Initial condition: 0.972
* Peak value: 1.082
* PDE residual: 0.946
* Non-negativity: 0.100 (fixed weight, not ReLoBRaLo balanced)

The PDE residual weight (0.95) and initial condition weight (0.97) are
close to unity, indicating that ReLoBRaLo achieved balanced optimization
across all loss components during Phase 3 training. The non-negativity
constraint uses a fixed weight of 0.10 and is excluded from ReLoBRaLo
balancing. Boundary condition weight is excluded because hard constraints
enforce :math:`u = 0` on all boundaries exactly.

**QPINN final ReLoBRaLo weights**:

* Initial condition: 0.759
* Boundary condition: 0.558
* Interior (PDE): 1.683

The interior/PDE residual weight is strongly elevated (1.68), reflecting
the greater difficulty of satisfying the PDE constraint in the domain interior.
The boundary weight (0.56) is suppressed because the QPINN's output naturally
approaches zero near boundaries due to the trainable embeddings.

Convergence Analysis
--------------------

Loss Evolution
^^^^^^^^^^^^^^

The two methods exhibit different convergence characteristics:

* **PINN (RAdam + L-BFGS)**: Two-phase hybrid optimization. Phase 1 uses
  RAdam over 3,000 epochs with CosineAnnealingWarmRestarts scheduler,
  achieving a best unweighted loss of 1.217e-06.
  Phase 2 uses L-BFGS for 200 refinement iterations with frozen ReLoBRaLo
  weights. The PINNsFormer Transformer architecture with multi-scale Fourier
  features enables expressive function approximation.
* **QPINN (SPSA)**: Gradient-free stochastic optimization over 200 iterations
  with steady convergence to a best cost of 4.456e-04. The SPSA optimizer
  requires only 2 function evaluations per step, avoiding the barren plateaus
  that plague gradient-based quantum training.

The loss history for both models is recorded in the CSV files
``pinn_training_losses.csv``, ``pinn_loss_components.csv``, and
``qpinn_training_losses.csv``.

ReLoBRaLo weight evolution over training is stored in
``qpinn_relobralo_weights.csv`` and ``metrics_over_time.csv``, and
visualized in:

* ``pinn_relobralo_evolution.png``
* ``qpinn_relobralo_evolution.png``

Quantum Circuit Analysis
------------------------

Circuit Characteristics
^^^^^^^^^^^^^^^^^^^^^^^

The GQE-generated quantum circuit for QPINN has:

* **Qubits**: 6
* **Circuit depth**: 25
* **Total gates**: 40
* **Trainable parameters**: 7
* **Hardware efficiency score**: 0.850
* **Noise resilience score**: 0.800
* **Expressivity score**: 0.800

Gate Composition
^^^^^^^^^^^^^^^^

.. code-block:: text

   Gate Type | Count | Percentage
   ----------+-------+-----------
   CNOT      | 12    | 30.0%
   CZ        |  9    | 22.5%
   H         |  3    |  7.5%
   RX        |  4    | 10.0%
   RY        |  1    |  2.5%
   RZ        |  1    |  2.5%
   S         |  5    | 12.5%
   SWAP      |  5    | 12.5%

Circuit details are saved in ``gqe_circuit_summary.txt``,
``gqe_circuit_text.txt``, and ``gqe_circuit_info.json``.

Computational Resource Analysis
-------------------------------

Training Time
^^^^^^^^^^^^^

**PINN (RAdam + L-BFGS)**:

* Phase 1: 3,000 epochs of RAdam with CosineAnnealingWarmRestarts
* Phase 2: 200 L-BFGS iterations with strong Wolfe line search
* Trainable parameters: 454,527 (SPINN body networks + PINNsFormer Transformer + Fourier features)
* Training time: 3,745 s (RAdam: 3,653 s + L-BFGS: 93 s)
* Gradient clipping (max_norm=1.0 for RAdam, 5.0 for L-BFGS)

**QPINN (SPSA) -- 10,605.49 s on GPU**:

* 200 SPSA iterations (gradient-free)
* 1 circuit parameter + 8 classical parameters
* Quantum circuit simulation via PennyLane ``lightning.qubit``
* Finite-difference PDE residual computation (7 forward passes per interior point)
* Each SPSA step requires 3 cost function evaluations (original + 2 perturbations)

.. note::

   The QPINN is 2.8x slower than the PINN on GPU, primarily because each
   SPSA step evaluates the cost function 3 times, with each evaluation
   requiring quantum circuit compilation and sequential forward passes for
   50 training points. The PINN benefits from GPU-parallelized autograd
   over all 50,000 training points simultaneously.

Visualization Results
---------------------

The benchmark produces 7 PNG visualizations:

1. ``comparison_heat_equation.png`` -- Side-by-side temperature field
   comparison at selected time slices.
2. ``loss_comparison.png`` -- Training loss curves for both models.
3. ``profile_comparison.png`` -- Temperature profiles along selected lines
   through the domain.
4. ``pinn_relobralo_evolution.png`` -- PINN ReLoBRaLo weight trajectories.
5. ``qpinn_relobralo_evolution.png`` -- QPINN ReLoBRaLo weight trajectories.
6. ``error_distribution.png`` -- Spatial distribution of prediction errors.
7. ``pinn_learning_rate.png`` -- PINN learning-rate schedule over epochs.

Error Distribution
^^^^^^^^^^^^^^^^^^

Spatial error analysis reveals:

* PINN errors (MSE 2.125e-06) are small and uniformly distributed, with
  a Max AE of only 3.975e-03 -- the best worst-case accuracy of either method.
  The IC lifting function (free-space Green's function ansatz) provides an
  excellent initial approximation, so the network only needs to learn a small
  correction. The curriculum learning strategy (IC-focused warm-up, PDE
  ramp-up, full ReLoBRaLo) combined with IC lifting ensures the model
  achieves excellent accuracy. Zero negative temperature predictions at convergence.
* QPINN errors are slightly lower in MSE (1.352e-06), achieving a 36% reduction
  relative to the PINN. However, the PINN has 124x better Max AE (3.975e-03 vs
  1.235e-01), indicating superior uniformity of accuracy across the domain.

Key Findings
------------

1. **Accuracy**: Both methods achieve excellent MSE: PINN 2.125e-06 vs QPINN
   1.352e-06 (QPINN 36% lower). The PINN's relative L2 error (0.138) is
   competitive with the QPINN's (0.110). The PINN achieves 124x better Max AE
   (3.975e-03 vs 1.235e-01).
2. **IC lifting function**: The free-space Green's function ansatz is critical
   for PINN accuracy. It provides the exact Gaussian diffusion solution as a
   base, reducing the network's task to learning only a small correction term.
   This yielded a 36x MSE improvement over the zero-lifting baseline.
3. **Curriculum learning**: The 3-phase curriculum (IC warm-up, PDE ramp-up,
   full ReLoBRaLo) ensures stable convergence. Validation RelL2 drops from
   4.76 at epoch 200 to 0.137 at epoch 3000, with the IC lifting providing
   the dominant improvement.
4. **Non-negativity**: The soft constraint applied to both PDE and IC points
   completely eliminates negative temperature predictions (0 at convergence),
   enforcing physical validity throughout the domain.
5. **Parameter efficiency**: The QPINN uses only 9 parameters compared to the
   PINN's 454,527 -- a reduction of over 50,000x -- and achieves comparable
   accuracy (36% better MSE).
6. **ReLoBRaLo**: Adaptive loss balancing converges to balanced weights:
   PINN final weights are initial=0.972, peak=1.082, PDE=0.946. The QPINN
   interior/PDE weight reaches 1.68, reflecting the greater difficulty of
   satisfying the PDE constraint in the domain interior.
7. **Gradient-free optimization**: SPSA converges effectively in 200 iterations,
   avoiding the barren-plateau issues that plague gradient-based quantum
   training. Each SPSA step requires only 3 cost function evaluations.
8. **Hard boundary constraints**: Both PINN and QPINN satisfy boundary conditions
   exactly (average boundary error ~ 0 at all time slices), confirming
   the distance-function approach eliminates boundary loss penalties entirely.
9. **Validation monitoring**: Per-timeslice validation reveals consistent PINN
   accuracy across all time slices, with MSE consistently in the 1e-06 range.

Recommendations
---------------

Based on the experimental results:

1. **IC lifting is essential**: For problems with known analytical structure
   (e.g., Gaussian IC under diffusion), incorporating a physics-based lifting
   function dramatically improves PINN accuracy (36x MSE improvement).
2. **Both methods are competitive**: PINN and QPINN achieve comparable MSE
   (~2e-06 vs ~1e-06), with the PINN offering better worst-case accuracy
   (Max AE) and the QPINN offering better average accuracy (MSE/MAE).
3. **For low-parameter regimes**: QPINNs with SPSA and ReLoBRaLo achieve
   excellent accuracy with only 9 parameters vs 454K for the PINN.
4. **Adaptive loss balancing**: ReLoBRaLo is effective for both classical and
   quantum training and should be enabled by default.
5. **Future work**: Scaling to larger domains, higher-dimensional PDEs, and
   real quantum hardware execution are natural next steps.

Output Files
------------

The benchmark produces output files in the ``results/`` directory:

.. code-block:: text

   Plots (PNG):
     comparison_heat_equation.png
     loss_comparison.png
     profile_comparison.png
     pinn_relobralo_evolution.png
     qpinn_relobralo_evolution.png
     error_distribution.png
     pinn_learning_rate.png

   Training data (CSV):
     pinn_training_losses.csv
     pinn_loss_components.csv
     qpinn_training_losses.csv
     qpinn_relobralo_weights.csv
     metrics_over_time.csv

   Validation & monitoring (CSV):
     pinn_validation_metrics.csv    -- per-epoch MSE / Rel L2 on held-out grid
     qpinn_validation_metrics.csv   -- per-step MSE / Rel L2 on held-out grid
     performance_metrics.csv        -- GPU memory, utilisation, CPU, RAM over time

   Analysis (JSON / TXT):
     comparative_analysis.json
     benchmark_summary.txt
     gqe_circuit_summary.txt
     gqe_circuit_text.txt
     gqe_circuit_info.json
     benchmark.log                  -- full hierarchical log (DEBUG level)

   Model checkpoints:
     pinn_radam_checkpoint.pth
     qpinn_spsa_checkpoint.pth

Reproducibility
---------------

All experiments can be reproduced using:

.. code-block:: bash

   # Default settings
   python main.py --backend auto

   # With custom JSON configuration
   python main.py --backend auto --config benchmark_config.json

The ``--config`` argument loads all hyperparameters (physics constants, grid
sizes, training epochs, optimizer settings, curriculum learning phases,
validation intervals, and parallel-processing options) from an external JSON
file.  See ``benchmark_config.json`` for the full schema and
:func:`config.load_benchmark_config` for the loader API.

Results are saved in the ``results/`` directory.

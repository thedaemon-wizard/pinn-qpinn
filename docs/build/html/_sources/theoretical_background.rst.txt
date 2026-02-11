Theoretical Background
======================

This chapter provides the mathematical and theoretical foundations for Physics-Informed Neural Networks (PINNs) and Quantum Physics-Informed Neural Networks (QPINNs).

Heat Conduction Equation
------------------------

The 3D heat conduction equation (diffusion equation) is given by:

.. math::

   \frac{\partial u}{\partial t} = \alpha \nabla^2 u = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + \frac{\partial^2 u}{\partial z^2} \right)

where:

* :math:`u(x,y,z,t)` is the temperature field
* :math:`\alpha` is the thermal diffusivity coefficient
* :math:`\nabla^2` is the Laplacian operator

Boundary and Initial Conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the benchmark problem, we consider:

* **Domain**: :math:`\Omega = [0, L]^3 \times [0, T]`
* **Initial condition**: :math:`u(x,y,z,0) = u_0(x,y,z)`
* **Boundary conditions**: Homogeneous Dirichlet :math:`u|_{\partial\Omega} = 0`

Analytical Solution
^^^^^^^^^^^^^^^^^^^

For a Gaussian initial condition :math:`u_0(x,y,z) = \exp\!\bigl(-\frac{(x-x_0)^2+(y-y_0)^2+(z-z_0)^2}{2\sigma^2}\bigr)`
on a bounded domain :math:`[0, L]^3` with homogeneous Dirichlet boundaries, the
analytical solution is a **Fourier sine series**:

.. math::

   u(x,y,z,t) = \sum_{l,m,n=1}^{N_{\max}} C_{lmn} \, \sin\!\left(\frac{l\pi x}{L}\right) \sin\!\left(\frac{m\pi y}{L}\right) \sin\!\left(\frac{n\pi z}{L}\right) \, e^{-\alpha\pi^2(l^2+m^2+n^2)t/L^2}

where the Fourier coefficients are computed from the initial condition:

.. math::

   C_{lmn} = \left(\frac{2}{L}\right)^3 \int_0^L \!\!\int_0^L \!\!\int_0^L u_0(x,y,z) \,\sin\!\left(\frac{l\pi x}{L}\right)\sin\!\left(\frac{m\pi y}{L}\right)\sin\!\left(\frac{n\pi z}{L}\right) \,dx\,dy\,dz

For the Gaussian initial condition centered at :math:`(L/2, L/2, L/2)` with
:math:`\sigma = 0.05`, the integrals are computed via the error function. The
series is truncated at :math:`N_{\max} = 30` modes per axis, which provides
convergence to machine precision for this problem.

.. note::

   The unbounded-domain Green's function solution
   :math:`u \propto (\sigma^2 + 4\alpha t)^{-3/2} \exp(-r^2/(\sigma^2 + 4\alpha t))`
   is **not** used, because it does not satisfy homogeneous Dirichlet boundary
   conditions on the finite domain :math:`[0, L]^3`.

Physics-Informed Neural Networks (PINNs)
----------------------------------------

PINNs incorporate the governing PDE into the loss function, enabling the neural network to learn solutions that satisfy both the data and the physics.

Loss Function Components
^^^^^^^^^^^^^^^^^^^^^^^^

The total loss function for PINNs consists of multiple components:

.. math::

   \mathcal{L} = \lambda_1 \mathcal{L}_{\text{IC}} + \lambda_2 \mathcal{L}_{\text{BC}} + \lambda_3 \mathcal{L}_{\text{PDE}} + \lambda_4 \mathcal{L}_{\text{peak}}

where:

1. **Initial Condition Loss**:

   .. math::

      \mathcal{L}_{\text{IC}} = \frac{1}{N_{\text{IC}}} \sum_{i=1}^{N_{\text{IC}}} |u(x_i, y_i, z_i, 0) - u_0(x_i, y_i, z_i)|^2

2. **Boundary Condition Loss**:

   .. math::

      \mathcal{L}_{\text{BC}} = \frac{1}{N_{\text{BC}}} \sum_{i=1}^{N_{\text{BC}}} |u(x_i, y_i, z_i, t_i)|^2 \quad \text{for } (x_i,y_i,z_i) \in \partial\Omega

3. **PDE Residual Loss**:

   .. math::

      \mathcal{L}_{\text{PDE}} = \frac{1}{N_{\text{PDE}}} \sum_{i=1}^{N_{\text{PDE}}} \left|\frac{\partial u}{\partial t} - \alpha \nabla^2 u\right|^2

4. **Peak Value Loss** (for better physical behavior):

   .. math::

      \mathcal{L}_{\text{peak}} = \frac{1}{N_{\text{peak}}} \sum_{i=1}^{N_{\text{peak}}} |u(L/2, L/2, L/2, t_i) - u_{\text{analytical}}(L/2, L/2, L/2, t_i)|^2

Fourier Neural Operator (FNO)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The FNO enhances PINNs by learning in the frequency domain. The spectral convolution layer is defined as:

.. math::

   (K * u)(x) = \mathcal{F}^{-1}[\mathcal{F}[K] \cdot \mathcal{F}[u]](x)

where :math:`\mathcal{F}` denotes the Fourier transform. This allows the network to capture global features efficiently.

**References**:

* Li et al. (2021) "Fourier Neural Operator for Parametric Partial Differential Equations" *ICLR 2021*
* Wang et al. (2022) "When and why PINNs fail to train: A neural tangent kernel perspective" *SIAM Journal on Scientific Computing*

Temporal Attention Mechanism
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The temporal attention mechanism enhances the network's ability to capture time-dependent features:

.. math::

   \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V

where :math:`Q`, :math:`K`, and :math:`V` are query, key, and value matrices derived from the temporal features.

Quantum Physics-Informed Neural Networks (QPINNs)
-------------------------------------------------

QPINNs leverage quantum computing principles to solve PDEs, potentially offering advantages in expressivity and computational efficiency.

Quantum Circuit Architecture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The quantum circuit consists of:

1. **Input Encoding**: Angle encoding of spatial and temporal coordinates

   .. math::

      |x,y,z,t\rangle = R_y(\theta_x)|0\rangle \otimes R_y(\theta_y)|0\rangle \otimes R_y(\theta_z)|0\rangle \otimes R_y(\theta_t)|0\rangle

   where :math:`\theta_i = \pi x_i / 2L` for normalized inputs.

2. **Variational Layers**: Parameterized quantum gates

   .. math::

      U(\theta) = \prod_{l=1}^{L} \left( \prod_{i} R_y(\theta_{l,i}) \prod_{(i,j)} \text{CNOT}_{i,j} \right)

3. **Measurement**: Expectation values of Pauli operators

   .. math::

      u(x,y,z,t) = f\left(\langle Z_1 \rangle, \langle Z_2 \rangle, ..., \langle X_1 X_2 \rangle, ...\right)

GQE-GPT Circuit Generation
^^^^^^^^^^^^^^^^^^^^^^^^^^

The Generative Quantum Eigensolver (GQE) uses a GPT model to generate quantum circuits:

1. **Circuit Tokenization**: Convert quantum gates to tokens
2. **GPT Training**: Learn circuit patterns from the ketGPT dataset
3. **Circuit Generation**: Sample new circuits based on problem characteristics

**Algorithm**: GQE-GPT Circuit Generation

.. code-block:: text

   1. Initialize GPT model with ketGPT pre-training
   2. For each generation:
      a. Generate N candidate circuits using GPT
      b. Evaluate circuits on training data
      c. Select Pareto-optimal circuits
      d. Update GPT with best circuits
   3. Return best circuit for QPINN

**References**:

* Apak et al. (2024) "KetGPT – Dataset Augmentation of Quantum Circuits using Transformers" *arXiv:2402.13352*
* Nakaji & Yamamoto (2021) "Quantum circuit design by Generative Quantum Eigensolver" *arXiv:2106.10985*

Trainable Embedding (TE-QPINN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TE-QPINN introduces learnable embedding functions:

.. math::

   \Phi(x,y,z,t) = \sum_{i} w_i^{(s)} \phi_i^{(s)}(x,y,z) + \sum_{j} w_j^{(t)} \phi_j^{(t)}(t)

where:

* :math:`\phi_i^{(s)}` are spatial basis functions (polynomials)
* :math:`\phi_j^{(t)}` are temporal basis functions (Fourier + exponential)
* :math:`w_i^{(s)}, w_j^{(t)}` are trainable weights

**References**:

* "Trainable embedding quantum physics informed neural networks" *Scientific Reports* (2025)
* Panichi et al. (2025) "Quantum physics informed neural networks for multi-variable PDEs" *arXiv:2503.12244*

SPINN: Separable Physics-Informed Neural Networks
---------------------------------------------------

SPINN (Cho et al., NeurIPS 2023 Spotlight) decomposes multi-dimensional PDE solving
into independent per-axis networks, dramatically reducing computational complexity.

Separable Body Networks
^^^^^^^^^^^^^^^^^^^^^^^^

For a :math:`d`-dimensional problem, SPINN uses :math:`d` independent body networks:

.. math::

   f_k: \mathbb{R}^1 \to \mathbb{R}^r, \quad k = 1, \ldots, d

where :math:`r` is the feature rank. The aggregated feature is computed via Hadamard
(element-wise) product:

.. math::

   \mathbf{h}(x,y,z,t) = f_x(x) \odot f_y(y) \odot f_z(z) \odot f_t(t)

This reduces complexity from :math:`O(N^d)` to :math:`O(Nd)` for :math:`N` grid points
per axis and :math:`d=4` dimensions (3 spatial + 1 temporal). Each body network is a
small MLP with Wavelet activation.

**References**:

* Cho et al. (2023) "Separable Physics-Informed Neural Networks" *NeurIPS 2023 Spotlight*

PINNsFormer Architecture
------------------------------------

PINNsFormer (Zhao, Ding & Prakash, ICLR 2024) extends Physics-Informed Neural Networks
with a Transformer-based architecture that captures temporal dependencies. In our
implementation, PINNsFormer processes SPINN-aggregated features combined with
multi-scale Fourier features through an encoder-decoder pipeline.

Wavelet Activation Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PINNsFormer uses a learnable wavelet activation function (Eq. 4 in the original paper):

.. math::

   \sigma_{\text{wavelet}}(x) = w_1 \sin(x) + w_2 \cos(x)

where :math:`w_1` and :math:`w_2` are learnable parameters initialized to 1.0. This
activation naturally encodes oscillatory behavior relevant to PDE solutions and provides
smooth derivatives needed for PDE residual computation via autograd.

Pseudo-Sequence Generator
^^^^^^^^^^^^^^^^^^^^^^^^^^

The pseudo-sequence generator converts the concatenated SPINN + Fourier features into
temporal sequences suitable for the Transformer architecture. Given a feature vector
at a collocation point, it produces a sequence of :math:`L` positions with physics-aware
temporal decay:

.. math::

   \mathbf{s}_i = M_i(\mathbf{h}) \cdot e^{-\lambda i \Delta t} + \mathbf{p}_i

where :math:`M_i` is a learned temporal modulator, :math:`\lambda = 0.296` is the
dominant decay rate for the 3D heat equation, and :math:`\mathbf{p}_i` is a sinusoidal
position embedding.

**References**:

* Zhao, Ding & Prakash (2024) "PINNsFormer: A Transformer-Based Framework for Physics-Informed Neural Networks" *ICLR 2024*

Transformer Encoder/Decoder Blocks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The architecture uses Transformer encoder and decoder blocks with Wavelet residual connections:

* **Encoder**: Self-attention where Q=K=V from the same source, followed by feed-forward
  network with WaveletActivation. Uses residual connections with Wavelet activation instead
  of standard addition, following the PINNsFormer paper.
* **Decoder**: Cross-attention from encoder output, where queries come from the decoder and
  keys/values from the encoder. Also uses Wavelet residual connections.

Output Projection
^^^^^^^^^^^^^^^^^^

The decoder output is aggregated across sequence positions using learned temporal weights
(initialized with exponential decay), then projected to scalar output:

.. math::

   \hat{u} = W_{\text{out}} \left( \sum_{i=0}^{L-1} \alpha_i \mathbf{h}_i \right) + b_{\text{out}}

where :math:`\alpha_i = \text{softmax}(w_i)` are learned temporal weights.

Optimization Strategy
------------------------------------

PINN Optimizer
^^^^^^^^^^^^^^

The classical PINN uses RAdam (Rectified Adam) with decoupled weight decay regularization. RAdam provides an adaptive learning rate with variance rectification that stabilizes early training:

.. math::

   \theta_{t+1} = \theta_t - \eta_t \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \quad \text{with variance rectification term } r_t

The learning rate schedule uses **CosineAnnealingWarmRestarts**, which periodically resets the learning rate following a cosine curve to escape local minima.

QPINN Optimizer
^^^^^^^^^^^^^^^^

The quantum PINN uses the PennyLane **SPSAOptimizer** (Simultaneous Perturbation Stochastic Approximation), which estimates gradients using only two circuit evaluations per step regardless of the number of parameters:

.. math::

   \hat{g}_k(\theta) = \frac{L(\theta + c_k \Delta_k) - L(\theta - c_k \Delta_k)}{2 c_k \Delta_k}

where :math:`\Delta_k` is a random perturbation vector with Bernoulli-distributed components.

ReLoBRaLo Adaptive Loss Weighting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Both PINN (RAdam) and QPINN (SPSA) use **ReLoBRaLo** (Relative Loss Balancing with Random Lookback) to adaptively rebalance multi-component losses using softmax-based relative balancing.

**Hyperparameters**:

* :math:`\alpha = 0.999` -- exponential moving average (EMA) decay
* :math:`\rho = 0.999` -- random lookback rate (probability of using the previous-step reference)
* :math:`\tau = 1.0` -- softmax temperature controlling weight sharpness

**Two reference points** are maintained:

1. **Previous-step losses**: the loss values from the immediately preceding training step
2. **Initial (epoch 0) losses**: the loss values recorded at the start of training

At each step, softmax-normalized weights are computed relative to each reference point. The step-based weights :math:`\hat{\lambda}_k` compare the current losses to the previous step:

.. math::

   \hat{\lambda}_k = \frac{\exp\left(\mathcal{L}_k^{(t)} / (\tau \cdot \mathcal{L}_k^{(t-1)})\right)}{\sum_j \exp\left(\mathcal{L}_j^{(t)} / (\tau \cdot \mathcal{L}_j^{(t-1)})\right)}

The initial-reference weights :math:`\hat{\lambda}_{0,k}` compare the current losses to the initial losses:

.. math::

   \hat{\lambda}_{0,k} = \frac{\exp\left(\mathcal{L}_k^{(t)} / (\tau \cdot \mathcal{L}_k^{(0)})\right)}{\sum_j \exp\left(\mathcal{L}_j^{(t)} / (\tau \cdot \mathcal{L}_j^{(0)})\right)}

These are combined via an exponential moving average with a random lookback governed by :math:`\rho`:

.. math::

   \lambda_k = \rho \cdot \alpha \cdot \lambda_k + (1 - \rho) \cdot \alpha \cdot \hat{\lambda}_{0,k} + (1 - \alpha) \cdot \hat{\lambda}_k

The random lookback (controlled by :math:`\rho`) stochastically mixes the history with the initial-reference weights, preventing the weighting from overfitting to a single reference trajectory.


**References**:

* Liu et al. (2020) "On the Variance of the Adaptive Learning Rate and Beyond" *ICLR 2020* (RAdam)
* Spall (1998) "Implementation of the Simultaneous Perturbation Algorithm for Stochastic Optimization" *IEEE Transactions on Aerospace and Electronic Systems* (SPSA)
* Bischof & Kraus (2025) "Multi-Objective Loss Balancing for Physics-Informed Deep Learning" *Computer Methods in Applied Mechanics and Engineering*, 431, 117521 (ReLoBRaLo)
* Zhao et al. (2023) "PINNsFormer: A Transformer-Based Framework For Physics-Informed Neural Networks" *arXiv:2307.02049*

Bayesian Multi-Objective Optimization for Energy Estimation
------------------------------------------------------------

The QPINN circuit optimization uses nine objectives evaluated through Bayesian multi-objective optimization:

**Objective Scores**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  
1. **Hardware Efficiency**: 
   
   .. math::
      \mathcal{O}_1 = 1 - \left( 0.3 \cdot e^{-T_{\text{total}}/5000} + 0.3 \cdot (1-p_{\text{error}})^2 + 0.2 \cdot \eta_{\text{conn}} + 0.2 \cdot \eta_{\text{parallel}} \right)

   where :math:`T_{\text{total}}` is total gate time, :math:`p_{\text{error}}` is total error probability, :math:`\eta_{\text{conn}}` is connectivity efficiency, and :math:`\eta_{\text{parallel}}` is parallelization capability.

2. **Noise Resilience**:
   
   .. math::
      \mathcal{O}_2 = 1 - \left( 0.4 \cdot e^{-d/20} + 0.3 \cdot (1-p_{\text{dep}})^{10} + 0.2 \cdot \eta_{\text{symmetry}} + 0.1 \cdot \eta_{\text{diversity}} \right)

   where :math:`d` is circuit depth, :math:`p_{\text{dep}}` is depolarizing error rate.

3. **Expressivity**:
   
   .. math::
      \mathcal{O}_3 = 1 - \min\left(1, \frac{R_{\text{eff}}}{R_{\text{max}}}\right)

   where :math:`R_{\text{eff}}` is the effective rank of the quantum circuit unitary.

4. **Mitigation Compatibility**:
   
   .. math::
      \mathcal{O}_4 = 1 - \left( 0.3 \cdot \eta_{\text{ZNE}} + 0.3 \cdot \eta_{\text{PEC}} + 0.2 \cdot \eta_{\text{symmetry}} + 0.2 \cdot \eta_{\text{CD}} \right)

   Compatibility with zero-noise extrapolation (ZNE), probabilistic error cancellation (PEC), symmetry verification, and Clifford data regression (CD).

5. **Trainability**:
   
   .. math::
      \mathcal{O}_5 = 1 - \eta_{\text{QFI}}

   where :math:`\eta_{\text{QFI}}` is the normalized quantum Fisher information score.

6. **Entanglement Capability**:
   
   .. math::
      \mathcal{O}_6 = 1 - \min\left(1, \frac{S_{\text{ent}}}{S_{\text{max}}}\right)

   where :math:`S_{\text{ent}}` is the entanglement entropy capability.

7. **Circuit Depth Efficiency**:
   
   .. math::
      \mathcal{O}_7 = \frac{d_{\text{actual}}}{100}

   Normalized circuit depth.

8. **Parameter Efficiency**:
   
   .. math::
      \mathcal{O}_8 = 1 - \eta_{\text{param}}

   where :math:`\eta_{\text{param}}` evaluates parameter distribution and gradient flow.

9. **Energy Estimation Quality** (Unsupervised):
   
   .. math::
      \mathcal{O}_9 = 1 - \left( 0.25 \cdot \eta_{\text{smooth}} + 0.20 \cdot \eta_{\text{conv}} + 0.20 \cdot \eta_{\text{stable}} + 0.20 \cdot \eta_{\text{info}} + 0.15 \cdot \eta_{\text{QFI}} \right)

   where :math:`\eta_{\text{smooth}}` is energy landscape smoothness, :math:`\eta_{\text{conv}}` is convergence score, :math:`\eta_{\text{stable}}` is noise stability, :math:`\eta_{\text{info}}` is information quality, and :math:`\eta_{\text{QFI}}` is quantum Fisher information score.

**References**:

* Cerezo et al. (2021) "Cost function dependent barren plateaus in shallow parametrized quantum circuits" *Nature Communications*
* Li et al. (2020) "Quantum optimization with a novel Gibbs objective function and ansatz architecture search" *Physical Review Research*

Noise Models and Hardware Efficiency
------------------------------------

Quantum Noise Modeling
^^^^^^^^^^^^^^^^^^^^^^

The implementation includes realistic noise models based on current NISQ devices:

.. math::

   \mathcal{E}(\rho) = (1-p)\rho + p\sum_{i} \sigma_i \rho \sigma_i

where :math:`p` is the error probability and :math:`\sigma_i` are Pauli matrices.

Hardware Efficiency Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Gate Time**: :math:`T_{\text{total}} = \sum_i t_i n_i` where :math:`t_i` is the time for gate type :math:`i`
2. **Error Rate**: :math:`\epsilon_{\text{total}} = 1 - \prod_i (1 - \epsilon_i)^{n_i}`
3. **Connectivity Overhead**: SWAP gates required for non-adjacent qubits

**References**:

* Kandala et al. (2017) "Hardware-efficient variational quantum eigensolver for small molecules" *Nature*
* Temme et al. (2017) "Error mitigation for short-depth quantum circuits" *Physical Review Letters*
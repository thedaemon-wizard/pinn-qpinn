PINNs Implementation
====================

This chapter details the implementation of Physics-Informed Neural Networks (PINNs)
integrating SPINN (Separable Physics-Informed Neural Networks, NeurIPS 2023 Spotlight)
with PINNsFormer (ICLR 2024) Transformer architecture, multi-scale Fourier feature
mappings, and RAdam + L-BFGS hybrid optimization with ReLoBRaLo adaptive loss weighting.

Architecture Overview
---------------------

The SPINN + PINNsFormer PINN architecture pipeline:

1. **SPINN body networks**: Per-axis MLPs (x, y, z, t) → rank-r features
2. **Hadamard aggregation**: Element-wise product of axis features + learned projection
3. **Multi-scale Fourier features**: Coarse/fine spatial + slow/fast temporal encoding
4. **Feature concatenation**: SPINN features + Fourier features
5. **PINNsFormer pseudo-sequence**: Point-wise → sequence with physics-aware decay
6. **PINNsFormer encoder**: Self-attention + FFN + WaveletActivation residuals
7. **PINNsFormer decoder**: Cross-attention from encoder output
8. **Output projection**: Learned temporal weights → scalar prediction
9. **Hard boundary constraints with IC lifting**: g(x,y,z,t) + D(x,y,z) * network_output

.. code-block:: python

   class PINN(nn.Module):
       def __init__(self,
                    layers=[5, 128, 256, 256, 128, 1],
                    use_hard_constraints=True,
                    boundary_epsilon=0.1,
                    fourier_features=True,
                    num_fourier_features=64,
                    use_transformer=True,
                    transformer_config=None,
                    transformer_memory_efficient=True):

SPINN: Separable Per-Axis Body Networks
----------------------------------------

SPINN (Cho et al., 2023) decomposes the multi-dimensional input into independent
per-axis networks, each processing a single coordinate dimension:

.. math::

   f_x: \mathbb{R}^1 \to \mathbb{R}^r, \quad
   f_y: \mathbb{R}^1 \to \mathbb{R}^r, \quad
   f_z: \mathbb{R}^1 \to \mathbb{R}^r, \quad
   f_t: \mathbb{R}^1 \to \mathbb{R}^r

The aggregated feature is computed via Hadamard product:

.. math::

   \mathbf{h} = f_x(x) \odot f_y(y) \odot f_z(z) \odot f_t(t)

followed by a learned projection. This reduces computational complexity from
:math:`O(N^d)` to :math:`O(Nd)` where :math:`d=4` (3 spatial + 1 temporal).

Each body network is a small MLP with Wavelet activation:

.. code-block:: python

   class SPINNBodyNetwork(nn.Module):
       def __init__(self, rank=64, hidden_dim=64, n_hidden_layers=2):
           # R^1 → hidden_dim → ... → rank
           # Uses WaveletActivation between layers

Multi-Scale Fourier Feature Mapping
-----------------------------------

The implementation uses multi-scale Fourier features to capture both fine and coarse spatial patterns:

Spatial Features
^^^^^^^^^^^^^^^^

.. math::

   \phi_{\text{spatial}}(\mathbf{x}) = \begin{bmatrix}
   \sin(2\pi \mathbf{B}_{\text{coarse}} \mathbf{x}) \\
   \cos(2\pi \mathbf{B}_{\text{coarse}} \mathbf{x}) \\
   \sin(2\pi \mathbf{B}_{\text{fine}} \mathbf{x}) \\
   \cos(2\pi \mathbf{B}_{\text{fine}} \mathbf{x})
   \end{bmatrix}

where:

* :math:`\mathbf{B}_{\text{coarse}} \sim \mathcal{N}(0, 2I)` for coarse features
* :math:`\mathbf{B}_{\text{fine}} \sim \mathcal{N}(0, 10I)` for fine features

The fine-scale variance (10.0) is chosen to resolve the narrow Gaussian initial condition
(:math:`\sigma = 0.05`, spatial frequency :math:`\sim 20`), while the coarse-scale variance
(2.0) captures the slower thermal diffusion dynamics.

Temporal Features
^^^^^^^^^^^^^^^^^

Based on the characteristic diffusion time :math:`t_c = L^2/(4\alpha)`:

.. math::

   \phi_{\text{temporal}}(t) = \begin{bmatrix}
   \sin(2\pi t/T) \\
   \cos(2\pi t/T) \\
   \sin(10\pi t/T) \\
   \cos(10\pi t/T)
   \end{bmatrix}

Implementation:

.. code-block:: python

   def fourier_feature_mapping(self, coords, B):
       """Multi-scale Fourier feature mapping"""
       x_proj = 2 * np.pi * coords @ B
       return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

PINNsFormer Transformer Architecture
-------------------------------------

The PINNsFormer architecture (Zhao, Ding & Prakash, ICLR 2024) processes pseudo-sequences
of SPINN-aggregated and Fourier-encoded features through a Transformer encoder-decoder
pipeline with Wavelet activation. Combined with SPINN's separable feature extraction,
it captures long-range temporal dependencies and complex spatio-temporal interactions
through attention mechanisms while maintaining computational efficiency.

Wavelet Activation
^^^^^^^^^^^^^^^^^^

The Wavelet activation function replaces standard nonlinearities throughout the
PINNsFormer architecture. It is defined as a learnable linear combination of
sinusoidal basis functions:

.. math::

   \text{Wavelet}(x) = \omega_1 \sin(x) + \omega_2 \cos(x)

where :math:`\omega_1` and :math:`\omega_2` are learnable scalar parameters initialized
to 1.0. This activation is used in the transformer feed-forward networks, the
spatio-temporal mixer, and the output projection.

.. code-block:: python

   class WaveletActivation(nn.Module):
       def __init__(self, learnable=True):
           super().__init__()
           self.omega1 = nn.Parameter(torch.ones(1))
           self.omega2 = nn.Parameter(torch.ones(1))

       def forward(self, x):
           return self.omega1 * torch.sin(x) + self.omega2 * torch.cos(x)

Pseudo-Sequence Generator
^^^^^^^^^^^^^^^^^^^^^^^^^^

The Pseudo-Sequence Generator converts point-wise inputs :math:`(x, y, z, t)` into a
temporal sequence of length ``seq_length`` suitable for transformer processing.
Each sequence position applies a learnable temporal modulation to the projected input,
combined with a physics-aware exponential decay factor and sinusoidal position embeddings:

.. code-block:: python

   class PseudoSequenceGenerator(nn.Module):
       def __init__(self, input_dim=4, seq_length=16, d_model=128, delta_t=0.1):
           super().__init__()
           self.input_projection = nn.Linear(input_dim, d_model)
           self.temporal_modulators = nn.ModuleList([
               nn.Linear(d_model, d_model) for _ in range(seq_length)
           ])
           self.position_embeddings = nn.Parameter(
               torch.zeros(1, seq_length, d_model)
           )

       def forward(self, features):
           projected = self.input_projection(features)
           sequences = []
           for i in range(self.seq_length):
               modulated = self.temporal_modulators[i](projected)
               # Physics-based decay: dominant mode rate alpha*pi^2*3/L^2 ~ 0.296
               decay_factor = torch.exp(-0.296 * i * self.delta_t)
               sequences.append(modulated * decay_factor)
           seq = torch.stack(sequences, dim=1)
           return seq + self.position_embeddings

The temporal modulators are initialized close to the identity matrix with gradual
variation across positions, so the initial pseudo-sequence represents a smooth
temporal evolution of the input features.

Spatio-Temporal Mixer
^^^^^^^^^^^^^^^^^^^^^

The Spatio-Temporal Mixer applies dual attention -- first spatial, then temporal --
followed by a mixing layer. This two-stage attention mechanism allows the network
to separately model spatial correlations and temporal dynamics before combining them:

.. code-block:: python

   class SpatioTemporalMixer(nn.Module):
       def __init__(self, d_model=128, n_heads=8, dropout=0.1):
           super().__init__()
           self.spatial_attention = nn.MultiheadAttention(
               embed_dim=d_model, num_heads=n_heads,
               dropout=dropout, batch_first=True
           )
           self.temporal_attention = nn.MultiheadAttention(
               embed_dim=d_model, num_heads=n_heads,
               dropout=dropout, batch_first=True
           )
           self.mixing_layer = nn.Sequential(
               nn.Linear(d_model, d_model),
               nn.LayerNorm(d_model),
               WaveletActivation(),
               nn.Dropout(dropout),
               nn.Linear(d_model, d_model)
           )

       def forward(self, seq):
           spatial_out, _ = self.spatial_attention(seq, seq, seq)
           seq = self.norm1(seq + 0.3 * spatial_out)

           temporal_out, _ = self.temporal_attention(seq, seq, seq)
           seq = self.norm2(seq + 0.3 * temporal_out)

           mixed = self.mixing_layer(seq)
           return self.norm3(seq + 0.3 * mixed)

Residual connections are scaled by a factor of 0.3 to balance gradient flow.
A smaller factor (e.g. 0.1) was found to over-suppress the transformer signal,
while 0.3 allows sufficient expressive power without gradient explosion.

PINNsFormer Encoder
^^^^^^^^^^^^^^^^^^^

The encoder consists of multiple ``TransformerBlock`` layers, each containing
multi-head self-attention and a feed-forward network with Wavelet activation.
All residual connections use the same 0.3 scaling as the mixer:

.. code-block:: python

   class PINNsFormerEncoder(nn.Module):
       def __init__(self, n_layers=4, d_model=128, n_heads=8,
                    d_ff=512, dropout=0.1):
           super().__init__()
           self.layers = nn.ModuleList([
               TransformerBlock(d_model, n_heads, d_ff, dropout)
               for _ in range(n_layers)
           ])
           self.norm = nn.LayerNorm(d_model)

       def forward(self, x, mask=None):
           for layer in self.layers:
               x = layer(x, mask)
           return self.norm(x)

PINNsFormer Decoder (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An optional decoder module performs encoder-decoder cross-attention. When enabled,
it uses the encoder output as both target and memory, providing an additional
refinement stage. The decoder uses half as many layers as the encoder:

.. code-block:: python

   class PINNsFormerDecoder(nn.Module):
       def __init__(self, n_layers=2, d_model=128, n_heads=8,
                    d_ff=512, dropout=0.1):
           super().__init__()
           self.layers = nn.ModuleList([
               PINNsFormerDecoderLayer(d_model, n_heads, d_ff, dropout)
               for _ in range(n_layers)
           ])
           self.norm = nn.LayerNorm(d_model)

       def forward(self, tgt, memory):
           for layer in self.layers:
               tgt = layer(tgt, memory)
           return self.norm(tgt)

S-PFormer: Simplified Decoder-Only Variant
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

S-PFormer (2025) provides a simplified decoder-only Transformer that replaces the
separate encoder-decoder architecture when SPINN + Fourier features provide
sufficiently rich embeddings. It uses pre-norm design with scaled residual connections:

.. code-block:: python

   class SPFormerDecoderBlock(nn.Module):
       def __init__(self, d_model=64, n_heads=4, d_ff=256,
                    dropout=0.1, residual_scale=0.3):
           super().__init__()
           self.norm1 = nn.LayerNorm(d_model)
           self.self_attention = nn.MultiheadAttention(...)
           self.norm2 = nn.LayerNorm(d_model)
           self.ffn = FeedForward(d_model, d_ff, dropout)

       def forward(self, x, mask=None):
           normed = self.norm1(x)
           attn_out, _ = self.self_attention(normed, normed, normed)
           x = x + 0.3 * self.dropout(attn_out)  # scaled residual
           normed = self.norm2(x)
           ffn_out = self.ffn(normed)
           x = x + 0.3 * self.dropout(ffn_out)
           return x

The ``SPFormerDecoder`` stacks multiple ``SPFormerDecoderBlock`` layers with
a final LayerNorm. This is available as an alternative to the full
encoder-decoder pipeline in ``pinnsformer.py``.

Output Projection
^^^^^^^^^^^^^^^^^

The ``OutputProjection`` module aggregates the transformer sequence output into a
scalar prediction. It applies learnable temporal weights (initialized with
exponential decay emphasizing early time steps) via a softmax, computes a weighted
sum over the sequence dimension, and projects to the output dimension:

.. code-block:: python

   class OutputProjection(nn.Module):
       def __init__(self, seq_length=16, d_model=128, output_dim=1):
           super().__init__()
           self.temporal_weights = nn.Parameter(torch.ones(seq_length) / seq_length)
           self.output_layer = nn.Sequential(
               nn.Linear(d_model, d_model // 2),
               nn.LayerNorm(d_model // 2),
               WaveletActivation(),
               nn.Dropout(0.1),
               nn.Linear(d_model // 2, output_dim)
           )

       def forward(self, x):
           weights = F.softmax(self.temporal_weights, dim=0).view(1, -1, 1)
           x_aggregated = (x * weights).sum(dim=1)
           return self.output_layer(x_aggregated)

Hard Boundary Constraints with IC Lifting
------------------------------------------

Forward propagation with PINNsFormer and hard boundary constraints uses an IC lifting
ansatz based on the free-space Green's function for Gaussian diffusion:

.. math::

   u_{\text{output}} = g(x,y,z,t) + D(x,y,z) \cdot N_{\text{network}}(x,y,z,t)

where :math:`g(x,y,z,t)` is the IC lifting function, :math:`D(x,y,z)` is the distance function,
and :math:`N_{\text{network}}` is the network's correction term.

**IC Lifting Function (Free-Space Green's Function)**:

.. math::

   g(x,y,z,t) = \left(\frac{\sigma_0^2}{\sigma_0^2 + 2\alpha t}\right)^{3/2} \exp\!\left(-\frac{r^2}{2(\sigma_0^2 + 2\alpha t)}\right)

where :math:`r^2 = (x - L/2)^2 + (y - L/2)^2 + (z - L/2)^2`. This is the exact solution for a
Gaussian diffusing in free space and serves as an excellent ansatz when :math:`\sigma_0 \ll L`
(boundary effects are negligible while the Gaussian is significantly non-zero).

Key properties:

- At :math:`t = 0`: :math:`g = \exp(-r^2/(2\sigma_0^2))` = initial condition exactly
- At :math:`t > 0`: correctly captures amplitude decay and spatial spreading
- Near boundaries: :math:`g \approx 0` (Gaussian negligible for :math:`\sigma_0 = 0.05, L = 1`)

The network correction :math:`D(x,y,z) \cdot N` is zero on boundaries and learns only the
small residual difference between the free-space solution and the actual bounded-domain solution.
This dramatically improves convergence (36x MSE reduction vs. zero-lifting baseline).

The enhanced forward pass includes:

1. **Input normalization**: :math:`x_{\text{norm}} = x/L`, :math:`t_{\text{norm}} = t/T`
2. **Multi-scale Fourier features**: Spatial and temporal encodings
3. **Pseudo-sequence generation**: Convert combined features into a temporal sequence
4. **Spatio-temporal mixing**: Dual spatial and temporal attention with mixing layer
5. **Transformer encoding**: Multi-layer self-attention encoder (with optional decoder)
6. **Output projection**: Temporal-weighted aggregation to scalar output
7. **IC lifting + distance modulation**: g(x,y,z,t) + D(x,y,z) * network_output

Distance Function
^^^^^^^^^^^^^^^^^

The distance function uses a parabolic per-axis product form that satisfies
:math:`D = 0` on all boundaries and :math:`D = 1` at the domain center:

.. math::

   D(x,y,z) = \frac{4x}{L}\!\left(1 - \frac{x}{L}\right) \cdot \frac{4y}{L}\!\left(1 - \frac{y}{L}\right) \cdot \frac{4z}{L}\!\left(1 - \frac{z}{L}\right)

Each axis factor :math:`4a(1-a)` is a parabola with range :math:`[0, 1]` and maximum 1
at the midpoint. The product form ensures the network output is multiplied by zero at
every boundary face, automatically satisfying homogeneous Dirichlet conditions without
any boundary loss penalty.

Implementation:

.. code-block:: python

   def compute_distance_function(self, x, y, z):
       xn = x / L
       yn = y / L
       zn = z / L
       dx = 4.0 * xn * (1.0 - xn)
       dy = 4.0 * yn * (1.0 - yn)
       dz = 4.0 * zn * (1.0 - zn)
       return dx * dy * dz

Loss Function Components
------------------------

The PINN uses multiple loss components for comprehensive physics enforcement:

Initial Condition Loss
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def compute_initial_loss(self):
       u_pred_initial = self.forward(
           self.training_data['initial']['x'],
           self.training_data['initial']['y'],
           self.training_data['initial']['z'],
           self.training_data['initial']['t']
       )
       return mse_loss(u_pred_initial, self.training_data['initial']['u'])

Peak Value Loss
^^^^^^^^^^^^^^^

To ensure physical behavior at the center of the domain:

.. math::

   \mathcal{L}_{\text{peak}} = \frac{1}{N_{\text{peak}}} \sum_{i=1}^{N_{\text{peak}}} |u(L/2, L/2, L/2, t_i) - u_{\text{analytical}}(L/2, L/2, L/2, t_i)|^2

PDE Residual Loss
^^^^^^^^^^^^^^^^^

The PDE residual is computed using automatic differentiation. Inputs are
detached and re-attached with ``requires_grad=True`` to create a fresh
computation graph for second-order gradients. No ``torch.clamp`` is applied
to intermediate derivatives, as clamping produces zero gradients at the
clamp boundary and corrupts the Laplacian computation:

.. code-block:: python

   def compute_pde_residual(self, x, y, z, t):
       # Fresh computation graph for second-order gradients
       x = x.clone().detach().requires_grad_(True)
       y = y.clone().detach().requires_grad_(True)
       z = z.clone().detach().requires_grad_(True)
       t = t.clone().detach().requires_grad_(True)

       u = self.forward(x, y, z, t)

       # First derivatives (no clamping)
       u_t = torch.autograd.grad(u.sum(), t,
                                create_graph=True, retain_graph=True)[0]
       u_x = torch.autograd.grad(u.sum(), x,
                                create_graph=True, retain_graph=True)[0]
       u_y = torch.autograd.grad(u.sum(), y,
                                create_graph=True, retain_graph=True)[0]
       u_z = torch.autograd.grad(u.sum(), z,
                                create_graph=True, retain_graph=True)[0]

       # Second derivatives (no clamping)
       u_xx = torch.autograd.grad(u_x.sum(), x,
                                 create_graph=True, retain_graph=True)[0]
       u_yy = torch.autograd.grad(u_y.sum(), y,
                                 create_graph=True, retain_graph=True)[0]
       u_zz = torch.autograd.grad(u_z.sum(), z,
                                 create_graph=True, retain_graph=True)[0]

       # PDE residual: u_t - α(u_xx + u_yy + u_zz)
       residual = u_t - alpha * (u_xx + u_yy + u_zz)
       return residual

.. note::

   Applying ``torch.clamp`` to intermediate derivatives (as done in some PINN
   implementations for numerical stability) is incorrect because PyTorch's clamp
   produces zero gradients when the input hits the boundary. This corrupts the
   second-order gradient computation needed for the Laplacian. Mixed-precision
   (``torch.amp.autocast``) should also be avoided in PDE residual computation,
   as float16 causes catastrophic cancellation in the finite-difference-like
   operations within autograd.

The residual loss is then:

.. math::

   \mathcal{L}_{\text{PDE}} = \frac{1}{N_{\text{PDE}}} \sum_{i=1}^{N_{\text{PDE}}} \left|\frac{\partial u}{\partial t} - \alpha \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + \frac{\partial^2 u}{\partial z^2}\right)\right|^2

Causal Temporal Weighting
^^^^^^^^^^^^^^^^^^^^^^^^^

PDE residuals are weighted by a causal temporal factor that prioritises early time
steps, respecting the parabolic PDE's forward-in-time causality structure.
Points near :math:`t = 0` receive higher weight so the network learns the
initial-condition--to--dynamics transition before attempting to resolve
late-time behaviour:

.. math::

   w(t) = \exp\!\bigl(-\varepsilon \, t / T\bigr)

where :math:`\varepsilon` is a configurable parameter (default 1.0, set via
``pinn.accuracy.causal_epsilon`` in the JSON config).  The weighted PDE loss
becomes:

.. math::

   \mathcal{L}_{\text{PDE}}^{\text{causal}} = \frac{1}{N_{\text{PDE}}} \sum_{i=1}^{N_{\text{PDE}}} w(t_i) \left|\frac{\partial u}{\partial t}\bigg|_{i} - \alpha \, \nabla^2 u\bigg|_{i}\right|^2

Implementation:

.. code-block:: python

   causal_epsilon = getattr(self, '_causal_epsilon', 1.0)
   causal_weights = torch.exp(-causal_epsilon * t_chunk.squeeze() / T)
   pde_loss_chunk = torch.mean(causal_weights * pde_residual ** 2)

Non-Negativity Soft Constraint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Temperature must be non-negative for the heat equation.  A soft penalty
term penalises any negative predictions from the network:

.. math::

   \mathcal{L}_{\text{nonneg}} = \frac{1}{N} \sum_{i=1}^{N} \bigl[\operatorname{ReLU}(-u_{\text{pred},i})\bigr]^2

This loss is evaluated on both PDE interior points and initial condition points
to ensure non-negativity across the full spatio-temporal domain. It uses a fixed
(non-ReLoBRaLo-balanced) weight ``nonneg_weight`` (default 0.1, configurable via
``pinn.accuracy.nonneg_weight`` in the JSON config).

Implementation:

.. code-block:: python

   # Evaluate non-negativity on PDE + IC points
   nonneg_parts = [torch.relu(-self.forward(x_pde, y_pde, z_pde, t_pde)) ** 2]
   nonneg_parts.append(torch.relu(-u_pred_initial) ** 2)
   nonneg_loss = torch.mean(torch.cat(nonneg_parts, dim=0))

RAdam Training with ReLoBRaLo Adaptive Loss Weighting
------------------------------------------------------

The PINN is trained using the PyTorch RAdam optimizer with decoupled weight decay,
combined with the ReLoBRaLo (Relative Loss Balancing with Random Lookback) adaptive
loss weighting scheme (Bischof & Kraus, 2025).

Curriculum Learning with Three-Phase Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Training uses curriculum learning to introduce loss components gradually,
followed by L-BFGS refinement.  The curriculum is controlled by three
configurable phase ratios (``pinn.accuracy.curriculum``):

1. **Phase 1 -- IC-only warm-up** (default 0--30% of RAdam epochs):
   Only the initial-condition, peak, and non-negativity losses are active.
   This ensures the network first learns the narrow Gaussian initial condition
   without interference from PDE residuals.
2. **Phase 2 -- PDE ramp-up** (default 30--70% of RAdam epochs):
   The PDE residual loss is introduced with a linear ramp multiplier that
   increases from 0 to 1 over this phase.  This avoids the loss-landscape
   disruption that occurs when a large PDE loss is suddenly activated.
3. **Phase 3 -- Full ReLoBRaLo balancing** (default 70--100% of RAdam epochs):
   All loss components are active with full ReLoBRaLo adaptive weighting.
4. **L-BFGS refinement** (200 iterations): Second-order optimization
   for fine-tuning with frozen ReLoBRaLo weights from Phase 3.

.. code-block:: python

   # Curriculum configuration (from JSON config or defaults)
   phase1_end = int(epochs * 0.30)   # IC-only
   phase2_end = int(epochs * 0.70)   # +PDE ramp-up
   # phase3: epochs 0.70*epochs .. epochs  (full ReLoBRaLo)
   # phase4: L-BFGS refinement

This multi-phase approach extends the standard two-phase PINN training
(Krishnapriyan et al., 2021; Raissi et al., 2019) with a curriculum-learning
front-end that significantly improves initial-condition accuracy.

RAdam Optimizer
^^^^^^^^^^^^^^^

RAdam (Rectified Adam) provides variance-rectified adaptive learning rates, offering
more stable training than standard Adam, especially in the early stages of optimization:

.. code-block:: python

   optimizer = torch.optim.RAdam(
       model.parameters(),
       lr=1e-3,
       weight_decay=1e-2,     # Decoupled weight decay
       decoupled_weight_decay=True
   )

ReLoBRaLo Adaptive Loss Weighting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ReLoBRaLo dynamically balances the loss weights during training. At each epoch :math:`t`, the
weight for loss component :math:`i` is computed as:

.. math::

   \alpha_i^{(t)} = \rho \cdot \tilde{\alpha}_i^{(t)} + (1 - \rho) \cdot \hat{\alpha}_i^{(t)}

where :math:`\rho \in [0,1]` is an exponential decay (balancing historic vs. instantaneous
information), and:

**Rate-of-change balancing** (exponential moving contribution):

.. math::

   \tilde{\alpha}_i^{(t)} = \frac{\exp\!\bigl(\mathcal{L}_i^{(t)} / (\tau \cdot \mathcal{L}_i^{(t-1)})\bigr)}
   {\sum_j \exp\!\bigl(\mathcal{L}_j^{(t)} / (\tau \cdot \mathcal{L}_j^{(t-1)})\bigr)}

**Random-lookback balancing** (compares against a randomly chosen earlier epoch
:math:`t_0`):

.. math::

   \hat{\alpha}_i^{(t)} = \frac{\exp\!\bigl(\mathcal{L}_i^{(t)} / (\tau \cdot \mathcal{L}_i^{(t_0)})\bigr)}
   {\sum_j \exp\!\bigl(\mathcal{L}_j^{(t)} / (\tau \cdot \mathcal{L}_j^{(t_0)})\bigr)}

Here :math:`\tau` is a temperature parameter that controls the sharpness of the
softmax rebalancing.

Implementation:

.. code-block:: python

   def relobralo_weights(losses, prev_losses, init_losses, tau=0.1, rho=0.999):
       """Compute ReLoBRaLo adaptive loss weights.

       Args:
           losses: Current loss values (dict or list).
           prev_losses: Loss values from the previous epoch.
           init_losses: Loss values from a randomly selected earlier epoch.
           tau: Temperature parameter.
           rho: Exponential decay for blending.
       """
       # Rate-of-change term
       ratios_prev = [l / (tau * p + 1e-8) for l, p in zip(losses, prev_losses)]
       exp_prev = [math.exp(r) for r in ratios_prev]
       sum_prev = sum(exp_prev)
       alpha_tilde = [e / sum_prev for e in exp_prev]

       # Random-lookback term
       ratios_init = [l / (tau * i + 1e-8) for l, i in zip(losses, init_losses)]
       exp_init = [math.exp(r) for r in ratios_init]
       sum_init = sum(exp_init)
       alpha_hat = [e / sum_init for e in exp_init]

       # Blend
       weights = [rho * a + (1 - rho) * b
                  for a, b in zip(alpha_tilde, alpha_hat)]
       return weights

The combined training loss becomes:

.. math::

   \mathcal{L}_{\text{total}}^{(t)} = \sum_{i} \alpha_i^{(t)} \, \mathcal{L}_i^{(t)} + w_{\text{nonneg}} \, \mathcal{L}_{\text{nonneg}}^{(t)}

where the individual loss components :math:`\mathcal{L}_i` are the initial condition,
peak value, PDE residual (with causal weighting), and optionally boundary losses.
The non-negativity loss uses a fixed weight :math:`w_{\text{nonneg}}` and does not
participate in ReLoBRaLo balancing.

.. note::

   With hard boundary constraints enabled, the boundary loss is excluded from
   ReLoBRaLo weighting entirely, leaving four ReLoBRaLo-balanced components
   (IC, peak, PDE) plus the fixed-weight non-negativity term.

Training Loop
^^^^^^^^^^^^^

Phase 1 -- RAdam with CosineAnnealingWarmRestarts:

.. code-block:: python

   def train_radam(self, n_samples=50000, epochs=3000, lr=1e-3,
                   weight_decay=1e-2, lbfgs_epochs=200):
       optimizer = torch.optim.RAdam(
           self.parameters(), lr=lr,
           weight_decay=weight_decay, decoupled_weight_decay=True
       )
       scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
           optimizer, T_0=500, T_mult=2, eta_min=1e-6
       )

       for epoch in range(epochs):
           optimizer.zero_grad()
           losses = self._compute_individual_losses()  # IC, Peak, BC, PDE

           # ReLoBRaLo adaptive weighting (excludes boundary due to hard constraints)
           weights = relobralo_update(losses, prev_losses, init_losses)

           total_loss = sum(w * l for w, l in zip(weights, losses))
           total_loss.backward()
           torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
           optimizer.step()
           scheduler.step(epoch + 1)

Phase 2 -- L-BFGS refinement with frozen ReLoBRaLo weights:

.. code-block:: python

       # L-BFGS refinement (200 iterations)
       lbfgs_optimizer = torch.optim.LBFGS(
           self.parameters(), lr=1.0, max_iter=20, max_eval=25,
           history_size=50, tolerance_grad=1e-9, tolerance_change=1e-11,
           line_search_fn='strong_wolfe'
       )

       for step in range(lbfgs_epochs):
           def closure():
               lbfgs_optimizer.zero_grad()
               # Compute IC, peak, PDE losses inline (no detach)
               total = (frozen_w_ic * ic_loss +
                        frozen_w_peak * peak_loss +
                        frozen_w_pde * pde_loss)
               total.backward()
               torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
               return total
           lbfgs_optimizer.step(closure)

.. note::

   The L-BFGS closure computes PDE residual gradients inline rather than calling
   ``compute_pde_residual()``, because that method calls ``clone().detach()`` on
   inputs, which would break the computation graph required by L-BFGS's
   multiple closure evaluations per step.

Validation and Performance Monitoring
--------------------------------------

During training the PINN periodically evaluates on a held-out uniform grid
and collects hardware performance counters.

Validation Metrics
^^^^^^^^^^^^^^^^^^

Every ``validation.interval_pinn`` epochs (default 200), the model is evaluated
on a uniform :math:`N \times N \times N \times N` grid (default :math:`N = 10`)
against the analytical Fourier sine series solution.  Metrics recorded per
time slice include MSE and relative L2 error, along with the count of negative
predictions and the prediction range:

.. code-block:: python

   def _compute_validation_metrics(self, grid_size=10):
       """Evaluate on held-out grid and return MSE, rel_l2, per-timeslice metrics"""
       ...

Results are written to ``pinn_validation_metrics.csv`` via the
:func:`save_validation_csv` function in ``main.py``.

Performance Metrics
^^^^^^^^^^^^^^^^^^^

At the same interval, GPU and CPU statistics are collected via
:func:`_collect_performance_metrics`:

* GPU: allocated memory (MB), reserved memory, max memory, utilisation
  (percent), temperature (via ``nvidia-smi``)
* CPU: utilisation (percent), RAM usage (percent and GB) via ``psutil``

These are written to ``performance_metrics.csv`` via :func:`save_performance_csv`.

Optimization Techniques
-----------------------

Weight Initialization
^^^^^^^^^^^^^^^^^^^^^

Each sub-component (SPINN body networks, PINNsFormer encoder/decoder, OutputProjection,
PseudoSequenceGenerator) performs its own careful Xavier initialization with gain=0.5.
The top-level ``_initialize_weights()`` method only initializes non-transformer (MLP
fallback) layers to avoid overriding sub-component initialization:

.. code-block:: python

   def _initialize_weights(self):
       """Initialize only MLP fallback layers (non-transformer path).

       SPINN body networks, PINNsFormer encoder/decoder, OutputProjection,
       and PseudoSequenceGenerator each perform their own careful initialization
       (Xavier uniform with gain=0.5). This method must NOT override them.
       """
       if not self.use_transformer and hasattr(self, 'layers'):
           for m in self.layers:
               if isinstance(m, nn.Linear):
                   nn.init.xavier_uniform_(m.weight)
                   if m.bias is not None:
                       nn.init.constant_(m.bias, 0.0)

Gradient Checkpointing
^^^^^^^^^^^^^^^^^^^^^^

For memory-efficient training with PINNsFormer:

.. code-block:: python

   if use_checkpoint and self.use_transformer:
       h = checkpoint(self.transformer_forward, combined_input)

Memory-Efficient PINNsFormer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When ``transformer_memory_efficient=True``, the PINNsFormer uses a reduced
configuration to lower GPU memory consumption while preserving the transformer
architecture. The memory-efficient configuration uses fewer heads, shorter
sequences, and smaller model dimensions:

.. code-block:: python

   # Memory-efficient configuration (default)
   transformer_config = {
       'seq_length': 8,       # vs 16 in full mode
       'd_model': 64,         # vs 128 in full mode
       'n_heads': 4,          # vs 8 in full mode
       'n_layers': 2,         # vs 4 in full mode
       'd_ff': 256,           # vs 512 in full mode
       'dropout': 0.1
   }

Learning Rate Scheduling
^^^^^^^^^^^^^^^^^^^^^^^^

Cosine annealing with warm restarts provides periodic LR resets that help escape
local minima:

.. code-block:: python

   scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
       optimizer, T_0=500, T_mult=2, eta_min=1e-6
   )

The first cycle has period 500 epochs, with each subsequent cycle doubling in
length (:math:`T_{\text{mult}} = 2`). This schedule produces LR spikes at
epochs 500, 1500, 3500, ... that periodically re-explore the loss landscape.

Training Data Generation
^^^^^^^^^^^^^^^^^^^^^^^^

Training data uses a balanced split with enhanced initial-condition sampling
near the Gaussian peak:

* **Interior (PDE)**: 60% of total samples -- uniform random in :math:`\Omega \times (0, T]`
* **Boundary**: 5% of total samples -- uniform on :math:`\partial\Omega \times [0, T]`
* **Initial condition**: 35% of total samples, structured as:

  - 40% structured grid near peak (within :math:`4\sigma` of center)
  - 30% Gaussian-sampled near peak (std = :math:`\sigma_0`)
  - 30% uniform across the full domain at :math:`t = 0`

The high IC ratio (35%) and structured near-peak sampling ensure the network
learns the narrow Gaussian initial condition (:math:`\sigma = 0.05`) accurately.

.. note::

   Boundary loss is excluded from ReLoBRaLo weighting because hard boundary
   constraints (the product distance function) already enforce :math:`u = 0`
   on all boundaries exactly.

Best Practices
--------------

1. **Data Generation**: Use structured + Gaussian + uniform sampling for IC points near the peak
2. **Batch Processing**: Process large batches in chunks to avoid memory issues
3. **Normalization**: Normalize inputs to :math:`[0, 1]` for consistent gradient flow
4. **Regularization**: Decoupled weight decay (1e-2) via RAdam
5. **Curriculum Learning**: Introduce PDE loss gradually to avoid destabilising the initial-condition fit
6. **Causal Weighting**: Weight PDE residuals by :math:`\exp(-\varepsilon t/T)` to respect causality
7. **Non-Negativity**: Add soft constraint to prevent unphysical negative temperatures
8. **Consistent N_max**: Ensure the same ``N_max`` is used in both the per-point ``analytical_solution()`` and the vectorised ``compute_analytical_solution()`` functions (fixed from 15 to 25)

Common Issues and Solutions
---------------------------

Memory Management
^^^^^^^^^^^^^^^^^

For large 3D problems, use memory-efficient PINNsFormer:

.. code-block:: python

   # Use reduced transformer configuration for GPU memory constraints
   use_transformer=True
   transformer_memory_efficient=True

Gradient Explosion
^^^^^^^^^^^^^^^^^^

Use gradient clipping:

.. code-block:: python

   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

Slow Convergence
^^^^^^^^^^^^^^^^

* Increase the number of Fourier features
* Use ReLoBRaLo adaptive weight balancing for loss components
* Enable PINNsFormer transformer architecture for better temporal modeling
* Enable curriculum learning to introduce PDE loss gradually
* Increase causal epsilon to strengthen early-time focus

Analytical Solution N_max Mismatch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An earlier version of ``physics.py`` used ``N_max = 15`` in
``compute_analytical_solution()`` while the per-point ``analytical_solution()``
used ``N_max = 25``.  For a narrow Gaussian with :math:`\sigma_0 = 0.05`, 15
Fourier modes are insufficient to resolve the initial condition, causing the
evaluation grid to under-represent the true solution and inflating the reported
MSE.  Both functions now use ``N_max = 25`` for consistency.

Structured Logging
^^^^^^^^^^^^^^^^^^

All ``print()`` statements have been replaced by a hierarchical logger
(``logging.getLogger('benchmark.MODULE')``).  Log output is sent to both
the console (INFO level) and ``results/benchmark.log`` (DEBUG level with
timestamps).  This makes it easy to filter output by module and severity.

References
----------

* Cho et al. (2023) "Separable Physics-Informed Neural Networks" NeurIPS 2023 Spotlight
* Zhao et al. (2023) "PINNsFormer: A Transformer-Based Framework For Physics-Informed Neural Networks"
* Li et al. (2021) "Fourier Neural Operator for Parametric Partial Differential Equations"
* Wang et al. (2022) "When and why PINNs fail to train"
* Krishnapriyan et al. (2021) "Characterizing possible failure modes in PINNs"
* Liu et al. (2020) "On the Variance of the Adaptive Learning Rate and Beyond" (RAdam)
* Bischof & Kraus (2025) "ReLoBRaLo: Relative Loss Balancing with Random Lookback for Multi-Task Learning"
* Liu & Nocedal (1989) "On the limited memory BFGS method for large scale optimization" (L-BFGS)
* Sukumar & Srivastava (2022) "Exact imposition of boundary conditions with distance functions in PINNs"

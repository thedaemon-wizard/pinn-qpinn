PINNs Implementation
====================

This chapter details the implementation of Physics-Informed Neural Networks (PINNs) with advanced features including Fourier Neural Operators (FNO) and temporal attention mechanisms.

Architecture Overview
---------------------

The enhanced PINN architecture consists of several key components:

.. code-block:: python

   class PINN(nn.Module):
       def __init__(self, 
                    layers=[5, 128, 256, 256, 128, 1],
                    use_hard_constraints=True,
                    boundary_epsilon=0.1,
                    fourier_features=True,
                    num_fourier_features=64,
                    use_fno=True,
                    fno_modes=(8, 8, 8),
                    use_temporal_attention=True,
                    fno_memory_efficient=True):

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

* :math:`\mathbf{B}_{\text{coarse}} \sim \mathcal{N}(0, 5I)` for coarse features
* :math:`\mathbf{B}_{\text{fine}} \sim \mathcal{N}(0, 20I)` for fine features

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

Fourier Neural Operator (FNO) Integration
-----------------------------------------

The FNO layer performs spectral convolution in 3D:

.. code-block:: python

   class SpectralConv3d(nn.Module):
       def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
           super().__init__()
           self.scale = (1 / (in_channels * out_channels))
           self.weights1 = nn.Parameter(
               self.scale * torch.rand(in_channels, out_channels, 
                                     modes1, modes2, modes3, dtype=torch.cfloat)
           )

The spectral convolution operation:

.. math::

   \mathcal{S}[u](k_1, k_2, k_3) = W(k_1, k_2, k_3) \cdot \mathcal{F}[u](k_1, k_2, k_3)

where :math:`W` are learnable complex weights in Fourier space.

Memory-Efficient FNO
^^^^^^^^^^^^^^^^^^^^

For GPU memory constraints, a memory-efficient version uses learned projections instead of full 3D FFTs:

.. code-block:: python

   self.fno_projection = nn.Sequential(
       nn.Linear(12, 64),
       nn.GELU(),
       nn.Linear(64, 32),
       nn.GELU(),
       nn.Linear(32, 1)
   )

Temporal Attention Mechanism
----------------------------

The temporal attention enhances the network's ability to model time-dependent dynamics:

.. code-block:: python

   class TemporalAttention(nn.Module):
       def forward(self, x, temporal_encoding):
           Q = self.query(x)
           K = self.key(temporal_encoding)
           V = self.value(temporal_encoding)
           
           scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
           weights = F.softmax(scores, dim=-1)
           attended = torch.matmul(weights, V)
           
           return attended + x  # Residual connection

Hard Boundary Constraints
-------------------------

Forward propagation with FNO and temporal attention:

.. math::

   u_{\text{output}} = D(x,y,z) \cdot u_{\text{network}}(x,y,z,t)

where :math:`D(x,y,z)` is the distance function and :math:`u_{\text{network}}` is the network output.

The enhanced forward pass includes:

1. **Input normalization**: :math:`x_{\text{norm}} = 2x/L - 1`
2. **Multi-scale Fourier features**: Spatial and temporal encodings
3. **FNO features**: Spectral convolution or memory-efficient projection
4. **Temporal attention**: Applied after first hidden layer
5. **Distance modulation**: For hard boundary constraints

Implementation:

.. code-block:: python

   def compute_distance_function(self, x, y, z):
       distances = torch.stack([
           x / L, (L - x) / L,
           y / L, (L - y) / L,
           z / L, (L - z) / L
       ], dim=-1)
       
       d_min = torch.min(distances, dim=-1)[0]
       distance = self.boundary_epsilon * torch.tanh(d_min / self.boundary_epsilon)
       return distance

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

The PDE residual is computed using automatic differentiation:

.. code-block:: python

   def compute_pde_residual(self, x, y, z, t):
       x.requires_grad_(True)
       y.requires_grad_(True)
       z.requires_grad_(True)
       t.requires_grad_(True)
       
       u = self.forward(x, y, z, t)
       
       # Compute gradients
       u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                create_graph=True, retain_graph=True)[0]
       u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                create_graph=True, retain_graph=True)[0]
       u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u),
                                create_graph=True, retain_graph=True)[0]
       u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u),
                                create_graph=True, retain_graph=True)[0]
       
       # Second derivatives
       u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                 create_graph=True, retain_graph=True)[0]
       u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y),
                                 create_graph=True, retain_graph=True)[0]
       u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(u_z),
                                 create_graph=True, retain_graph=True)[0]
       
       # PDE residual: u_t - α(u_xx + u_yy + u_zz)
       residual = u_t - alpha * (u_xx + u_yy + u_zz)
       return residual

The residual loss is then:

.. math::

   \mathcal{L}_{\text{PDE}} = \frac{1}{N_{\text{PDE}}} \sum_{i=1}^{N_{\text{PDE}}} \left|\frac{\partial u}{\partial t} - \alpha \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} + \frac{\partial^2 u}{\partial z^2}\right)\right|^2

NSGA-II Multi-Objective Training
---------------------------------

The PINN is trained using NSGA-II with multiple objectives:

.. code-block:: python

   def train_with_nsga2(self, n_samples=10000, nsga2_config=None):
       # Configure NSGA-II
       config = nsga2_optimizer.NSGA2Config()
       config.population_size = nsga2_config['population_size_pinn']
       config.max_generations = nsga2_config['max_generations_pinn']
       config.n_objectives = 4  # IC, BC, PDE, Peak
       
       # Define objective function
       def objective_function(params_array):
           # Set parameters
           self._set_parameters_from_array(params_array[:total_nn_params])
           
           # Compute losses
           losses = self._compute_individual_losses()
           
           # Return objectives (to minimize)
           return [
               losses['initial'].item(),
               losses['boundary'].item(),
               losses['pde'].item(),
               losses['peak'].item()
           ]

Optimization Techniques
-----------------------

Weight Initialization
^^^^^^^^^^^^^^^^^^^^^

Xavier initialization adapted for temporal dynamics:

.. code-block:: python

   def _initialize_weights(self):
       for m in self.layers:
           if isinstance(m, nn.Linear):
               fan_in = m.weight.size(1)
               fan_out = m.weight.size(0)
               std = np.sqrt(2.0 / (fan_in + fan_out))
               nn.init.normal_(m.weight, mean=0.0, std=std)
               nn.init.constant_(m.bias, 0.01)

Gradient Checkpointing
^^^^^^^^^^^^^^^^^^^^^^

For memory-efficient training with FNO:

.. code-block:: python

   if use_checkpoint and self.use_fno:
       h = checkpoint(self.fno_forward, grid_points)

Learning Rate Scheduling
^^^^^^^^^^^^^^^^^^^^^^^^

Adaptive learning rate based on loss plateaus:

.. code-block:: python

   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
       optimizer, mode='min', factor=0.5, patience=100
   )

Best Practices
--------------

1. **Data Generation**: Use Latin Hypercube Sampling for better coverage of the domain
2. **Batch Processing**: Process large batches in chunks to avoid memory issues
3. **Normalization**: Normalize inputs to [-1, 1] for better gradient flow
4. **Regularization**: Add L2 regularization to prevent overfitting

Common Issues and Solutions
---------------------------

Memory Management
^^^^^^^^^^^^^^^^^

For large 3D problems, use memory-efficient FNO:

.. code-block:: python

   # Instead of full 3D FFT
   use_fno=True
   fno_memory_efficient=True

Gradient Explosion
^^^^^^^^^^^^^^^^^^

Use gradient clipping:

.. code-block:: python

   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

Slow Convergence
^^^^^^^^^^^^^^^^

* Increase the number of Fourier features
* Use adaptive weight balancing for loss components
* Enable temporal attention mechanism

References
----------

* Li et al. (2021) "Fourier Neural Operator for Parametric Partial Differential Equations"
* Wang et al. (2022) "When and why PINNs fail to train"
* Krishnapriyan et al. (2021) "Characterizing possible failure modes in PINNs"
* Lu et al. (2023) "NSGA-PINN: A Multi-Objective Optimization Method"
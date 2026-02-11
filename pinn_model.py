"""Classical PINN Model with SPINN + PINNsFormer Architecture

Integrates Separable Physics-Informed Neural Networks (SPINN, NeurIPS 2023)
with PINNsFormer (ICLR 2024) Transformer architecture for the 3D heat equation.

Architecture: SPINN per-axis body networks → PINNsFormer encoder-decoder → Output
Optimization: RAdam + L-BFGS hybrid with ReLoBRaLo adaptive loss weighting

References:
- Cho et al. (2023): Separable Physics-Informed Neural Networks (NeurIPS 2023 Spotlight)
- Zhao, Ding & Prakash (2024): PINNsFormer (ICLR 2024)
- Bischof & Kraus (2025): ReLoBRaLo Multi-Objective Loss Balancing
- Liu et al. (2020): RAdam (ICLR 2020)
"""
from config import *
from config import get_nested
from physics import TrainingPoint, initial_condition, boundary_condition, to_python_float, analytical_solution
from pinnsformer import (
    WaveletActivation, SPINNBodyNetwork, SPINNAggregator,
    PseudoSequenceGenerator, SpatioTemporalMixer,
    PINNsFormerEncoder, PINNsFormerDecoder, OutputProjection
)

import logging
_logger = logging.getLogger('benchmark.pinn_model')


class PINN(nn.Module):
    """SPINN + PINNsFormer PINN for 3D Heat Equation

    Architecture pipeline:
      1. SPINN: Per-axis body networks (x, y, z, t) → rank-r features each
      2. Fourier features: Multi-scale spatial + temporal frequency encoding
      3. Feature fusion: Concatenate SPINN aggregated + Fourier features
      4. PINNsFormer: Pseudo-sequence → Encoder → Decoder → Output projection
      5. Hard boundary constraints: D(x,y,z) * network_output

    This architecture combines SPINN's separable efficiency (reducing computation
    from O(N^d) to O(Nd)) with PINNsFormer's temporal dependency capture via
    Transformer attention, while maintaining the proven RAdam+L-BFGS+ReLoBRaLo
    training pipeline.

    References:
        Cho et al. (2023): Separable Physics-Informed Neural Networks
        Zhao et al. (2024): PINNsFormer Transformer-based PINN framework
        Wang et al. (2022): When and why PINNs fail to train
    """

    def __init__(self,
                 layers=[5, 128, 256, 256, 128, 1],
                 use_hard_constraints=True,
                 boundary_epsilon=0.1,
                 fourier_features=True,
                 num_fourier_features=64,
                 use_transformer=True,
                 transformer_config=None,
                 transformer_memory_efficient=True):

        super(PINN, self).__init__()

        self.use_fourier_features = fourier_features
        self.num_fourier_features = num_fourier_features
        self.use_transformer = use_transformer
        self.transformer_memory_efficient = transformer_memory_efficient
        self.use_hard_constraints = use_hard_constraints

        # Transformer configuration
        if transformer_config is None:
            if transformer_memory_efficient:
                transformer_config = {
                    'seq_length': 8, 'd_model': 64, 'n_heads': 4,
                    'n_layers': 2, 'd_ff': 256, 'dropout': 0.1
                }
            else:
                transformer_config = {
                    'seq_length': 16, 'd_model': 128, 'n_heads': 8,
                    'n_layers': 4, 'd_ff': 512, 'dropout': 0.1
                }
        self.transformer_config = transformer_config

        # =============================================
        # SPINN: Separable per-axis body networks
        # =============================================
        spinn_rank = transformer_config.get('d_model', 64)
        spinn_hidden = transformer_config.get('spinn_hidden', 64)
        spinn_layers = transformer_config.get('spinn_layers', 2)

        self.spinn_x = SPINNBodyNetwork(spinn_rank, spinn_hidden, spinn_layers)
        self.spinn_y = SPINNBodyNetwork(spinn_rank, spinn_hidden, spinn_layers)
        self.spinn_z = SPINNBodyNetwork(spinn_rank, spinn_hidden, spinn_layers)
        self.spinn_t = SPINNBodyNetwork(spinn_rank, spinn_hidden, spinn_layers)
        self.spinn_aggregator = SPINNAggregator(spinn_rank, spinn_rank)

        # =============================================
        # Multi-scale Fourier features
        # =============================================
        if self.use_fourier_features:
            self.B_spatial_coarse = nn.Parameter(
                torch.randn(3, num_fourier_features // 4) * 2.0, requires_grad=True)
            self.B_spatial_fine = nn.Parameter(
                torch.randn(3, num_fourier_features // 4) * 10.0, requires_grad=True)
            t_characteristic = L**2 / (4 * alpha)
            self.B_temporal_slow = nn.Parameter(
                torch.randn(1, num_fourier_features // 4) * (2 * np.pi / T), requires_grad=True)
            self.B_temporal_fast = nn.Parameter(
                torch.randn(1, num_fourier_features // 4) * (10 * np.pi / T), requires_grad=True)
            self.spatial_scale_coarse = nn.Parameter(torch.tensor(1.0))
            self.spatial_scale_fine = nn.Parameter(torch.tensor(1.0))
            self.temporal_scale = nn.Parameter(torch.tensor(2 * np.pi / T))

        # =============================================
        # Input dimension calculation
        # =============================================
        # SPINN aggregated features
        spinn_out_dim = spinn_rank

        # Fourier feature dimensions
        if self.use_fourier_features:
            spatial_fourier_dim = num_fourier_features  # sin+cos for coarse+fine
            temporal_fourier_dim = num_fourier_features
            self.input_dim = spinn_out_dim + spatial_fourier_dim + temporal_fourier_dim
        else:
            self.input_dim = spinn_out_dim

        # =============================================
        # PINNsFormer Transformer pipeline
        # =============================================
        if self.use_transformer:
            config = self.transformer_config

            self.pseudo_seq_generator = PseudoSequenceGenerator(
                input_dim=self.input_dim,
                seq_length=config['seq_length'],
                d_model=config['d_model'],
                delta_t=T / config['seq_length']
            )

            self.transformer_encoder = PINNsFormerEncoder(
                n_layers=config['n_layers'],
                d_model=config['d_model'],
                n_heads=config['n_heads'],
                d_ff=config['d_ff'],
                dropout=config['dropout']
            )

            self.use_decoder = True
            if self.use_decoder:
                self.transformer_decoder = PINNsFormerDecoder(
                    n_layers=max(1, config['n_layers'] // 2),
                    d_model=config['d_model'],
                    n_heads=config['n_heads'],
                    d_ff=config['d_ff'],
                    dropout=config['dropout']
                )

            self.output_projection = OutputProjection(
                seq_length=config['seq_length'],
                d_model=config['d_model'],
                output_dim=1,
                use_hard_constraints=False,
                domain_size=L
            )

            # Backward compat: keep spatio_temporal_mixer as passthrough
            self.spatio_temporal_mixer = None
        else:
            # MLP fallback
            self.layers = nn.ModuleList()
            layer_sizes = [self.input_dim] + layers[1:]
            for i in range(len(layer_sizes) - 1):
                in_dim = self.input_dim if i == 0 else layers[i]
                self.layers.append(nn.Linear(in_dim, layers[i + 1]))
                if i < len(layers) - 2:
                    self.layers.append(nn.Tanh())

        # Hard constraints
        if use_hard_constraints:
            self.boundary_epsilon = nn.Parameter(torch.tensor(boundary_epsilon))

        self._initialize_weights()

        # Training history tracking
        self.loss_history = []
        self.mean_fitness_history = []
        self.objective_history = {
            'initial': [], 'peak': [], 'boundary': [], 'pde': [], 'combined': []
        }
        self.pareto_front_history = []
        self.best_objectives = {
            'initial': float('inf'), 'peak': float('inf'),
            'boundary': float('inf'), 'pde': float('inf')
        }
        self.training_data = None

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

    def compute_distance_function(self, x, y, z):
        """Smooth distance function to boundaries for hard constraint.

        D(x,y,z) = 4*x/L*(1-x/L) * 4*y/L*(1-y/L) * 4*z/L*(1-z/L)

        Exactly 0 on all 6 faces, max=1.0 at center (L/2, L/2, L/2).

        References:
            Sukumar & Srivastava (2022): Exact imposition of boundary conditions
            Lu et al. (2021): PINNs with hard constraints for inverse design
        """
        if x.dim() == 1: x = x.unsqueeze(1)
        if y.dim() == 1: y = y.unsqueeze(1)
        if z.dim() == 1: z = z.unsqueeze(1)
        xn, yn, zn = x / L, y / L, z / L
        return (4.0 * xn * (1.0 - xn)) * (4.0 * yn * (1.0 - yn)) * (4.0 * zn * (1.0 - zn))

    def forward(self, x, y, z, t):
        """Forward pass: SPINN → Fourier features → PINNsFormer → Hard constraints.

        Args:
            x, y, z, t: [batch_size, 1] coordinate tensors

        Returns:
            [batch_size, 1] predicted temperature field
        """
        x_orig, y_orig, z_orig = x, y, z

        # Normalize inputs
        x_norm = x / L
        y_norm = y / L
        z_norm = z / L
        t_norm = t / T

        # =============================================
        # 1. SPINN: Separable per-axis feature extraction
        # =============================================
        feat_x = self.spinn_x(x_norm)
        feat_y = self.spinn_y(y_norm)
        feat_z = self.spinn_z(z_norm)
        feat_t = self.spinn_t(t_norm)

        spinn_features = self.spinn_aggregator([feat_x, feat_y, feat_z, feat_t])

        # =============================================
        # 2. Multi-scale Fourier features
        # =============================================
        inputs = [spinn_features]

        if self.use_fourier_features:
            spatial_coords = torch.cat([x_norm, y_norm, z_norm], dim=1)

            sc_coarse = torch.clamp(self.spatial_scale_coarse, 0.1, 10.0)
            sc_fine = torch.clamp(self.spatial_scale_fine, 0.1, 20.0)

            sp_coarse = torch.clamp(spatial_coords @ self.B_spatial_coarse * sc_coarse, -10, 10)
            sp_fine = torch.clamp(spatial_coords @ self.B_spatial_fine * sc_fine, -10, 10)

            spatial_fourier = torch.cat([
                torch.sin(sp_coarse), torch.cos(sp_coarse),
                torch.sin(sp_fine), torch.cos(sp_fine)
            ], dim=1)

            ts = torch.clamp(self.temporal_scale, 0.1, 10.0)
            tp_slow = torch.clamp(t_norm @ self.B_temporal_slow * ts, -10, 10)
            tp_fast = torch.clamp(t_norm @ self.B_temporal_fast * ts * 2, -10, 10)

            temporal_fourier = torch.cat([
                torch.sin(tp_slow), torch.cos(tp_slow),
                torch.sin(tp_fast), torch.cos(tp_fast)
            ], dim=1)

            inputs.append(spatial_fourier)
            inputs.append(temporal_fourier)

        combined_input = torch.cat(inputs, dim=1)

        # =============================================
        # 3. PINNsFormer Transformer pipeline
        # =============================================
        if self.use_transformer:
            pseudo_seq = self.pseudo_seq_generator(combined_input)
            encoded = self.transformer_encoder(pseudo_seq)

            if self.use_decoder and hasattr(self, 'transformer_decoder'):
                decoded = self.transformer_decoder(encoded, encoded)
                network_output = self.output_projection(decoded)
            else:
                network_output = self.output_projection(encoded)
        else:
            h = combined_input
            for i, layer in enumerate(self.layers[:-1]):
                h = layer(h)
                h = torch.tanh(h)
            network_output = self.layers[-1](h)

        # =============================================
        # 4. Hard boundary constraints with IC lifting
        # =============================================
        if self.use_hard_constraints:
            distance = self.compute_distance_function(x_orig, y_orig, z_orig)

            # IC lifting function using free-space Green's function for Gaussian IC:
            # g(x,y,z,t) = [sigma_0^2 / (sigma_0^2 + 2*alpha*t)]^(3/2)
            #              * exp(-r^2 / (2*(sigma_0^2 + 2*alpha*t)))
            # This is the exact solution for a Gaussian diffusing in free space,
            # and is an excellent approximation when sigma_0 << L (boundary effects
            # are negligible while the Gaussian is significantly non-zero).
            # At t=0: g = exp(-r^2/(2*sigma_0^2)) = IC exactly.
            # At t>0: correctly captures the spreading and amplitude decay.
            sigma2_t = sigma_0 ** 2 + 2.0 * alpha * t  # Effective variance at time t
            r2 = (x_orig - L / 2.0) ** 2 + (y_orig - L / 2.0) ** 2 + (z_orig - L / 2.0) ** 2
            amplitude_decay = (sigma_0 ** 2 / sigma2_t) ** 1.5
            g_vec = amplitude_decay * torch.exp(-r2 / (2.0 * sigma2_t))

            constrained_output = g_vec + distance * network_output
        else:
            constrained_output = network_output

        return constrained_output

    def compute_pde_residual(self, x, y, z, t):
        """Calculate heat equation residual: u_t - alpha * (u_xx + u_yy + u_zz)."""
        x = x.clone().detach().requires_grad_(True)
        y = y.clone().detach().requires_grad_(True)
        z = z.clone().detach().requires_grad_(True)
        t = t.clone().detach().requires_grad_(True)

        u = self.forward(x, y, z, t)

        u_t = grad(u.sum(), t, create_graph=True, retain_graph=True, allow_unused=True)[0]
        u_x = grad(u.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]
        u_y = grad(u.sum(), y, create_graph=True, retain_graph=True, allow_unused=True)[0]
        u_z = grad(u.sum(), z, create_graph=True, retain_graph=True, allow_unused=True)[0]

        if u_t is None: u_t = torch.zeros_like(t)
        if u_x is None: u_x = torch.zeros_like(x)
        if u_y is None: u_y = torch.zeros_like(y)
        if u_z is None: u_z = torch.zeros_like(z)

        u_xx = grad(u_x.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]
        u_yy = grad(u_y.sum(), y, create_graph=True, retain_graph=True, allow_unused=True)[0]
        u_zz = grad(u_z.sum(), z, create_graph=True, retain_graph=True, allow_unused=True)[0]

        if u_xx is None: u_xx = torch.zeros_like(x)
        if u_yy is None: u_yy = torch.zeros_like(y)
        if u_zz is None: u_zz = torch.zeros_like(z)

        laplacian = u_xx + u_yy + u_zz
        return u_t - alpha * laplacian

    def _generate_training_data(self, n_samples):
        """Generate training data with improved temporal sampling.

        Sampling strategy:
          - 35% initial condition (dense near Gaussian peak)
          - 60% interior PDE residual
          - 5% boundary (hard constraints handle most BC enforcement)
        """
        n_initial = int(n_samples * 0.35)
        n_boundary = int(n_samples * 0.05)
        n_interior = n_samples - n_boundary - n_initial

        _logger.info(f"Generating balanced training data: total={n_samples}")
        _logger.info(f"  interior={n_interior}, boundary={n_boundary}, initial={n_initial}")

        all_training_points = []
        time = torch.linspace(0, T, nt + 1, dtype=torch.float32)
        num_times = time.size(0)

        full_repeats = n_interior // num_times
        remainder = n_interior % num_times
        t_samples = time.repeat(full_repeats)
        if remainder > 0:
            extra_idxs = torch.randperm(num_times)[:remainder]
            t_samples = torch.cat([t_samples, time[extra_idxs]], dim=0)
        t_samples = t_samples[torch.randperm(t_samples.size(0))]

        for t_val in t_samples:
            x_val = float(torch.empty(1).uniform_(0.01, L - 0.01))
            y_val = float(torch.empty(1).uniform_(0.01, L - 0.01))
            z_val = float(torch.empty(1).uniform_(0.01, L - 0.01))
            all_training_points.append(TrainingPoint(x_val, y_val, z_val, t_val.item(), None, type='interior'))

        # Initial condition: structured grid near peak + Gaussian + uniform
        n_ic_grid = min(int(0.4 * n_initial), 2000)
        n_ic_near_peak = int(0.3 * n_initial)
        n_ic_uniform = n_initial - n_ic_grid - n_ic_near_peak

        half_width = min(4 * sigma_0, 0.45 * L)
        n_grid_1d = max(3, int(round(n_ic_grid ** (1 / 3))))
        grid_1d = np.linspace(L / 2 - half_width, L / 2 + half_width, n_grid_1d)
        ic_count = 0
        for gx in grid_1d:
            for gy in grid_1d:
                for gz in grid_1d:
                    if ic_count >= n_ic_grid:
                        break
                    u_true = initial_condition(gx, gy, gz)
                    all_training_points.append(TrainingPoint(gx, gy, gz, 0.0, u_true, type='initial'))
                    ic_count += 1

        for _ in range(n_ic_near_peak):
            x_val = float(torch.normal(L / 2, sigma_0, (1,)).clamp(0.01 * L, 0.99 * L))
            y_val = float(torch.normal(L / 2, sigma_0, (1,)).clamp(0.01 * L, 0.99 * L))
            z_val = float(torch.normal(L / 2, sigma_0, (1,)).clamp(0.01 * L, 0.99 * L))
            u_true = initial_condition(x_val, y_val, z_val)
            all_training_points.append(TrainingPoint(x_val, y_val, z_val, 0.0, u_true, type='initial'))

        for _ in range(n_ic_uniform):
            x_val = float(torch.FloatTensor(1).uniform_(0.01 * L, 0.99 * L))
            y_val = float(torch.FloatTensor(1).uniform_(0.01 * L, 0.99 * L))
            z_val = float(torch.FloatTensor(1).uniform_(0.01 * L, 0.99 * L))
            u_true = initial_condition(x_val, y_val, z_val)
            all_training_points.append(TrainingPoint(x_val, y_val, z_val, 0.0, u_true, type='initial'))

        # Boundary points with temporal stratification
        n_boundary_per_face = n_boundary // 6
        for face in range(6):
            for t_idx in range(n_boundary_per_face):
                t_val = t_idx * T / n_boundary_per_face
                if face == 0:
                    x_val, y_val, z_val = 0.0, float(torch.FloatTensor(1).uniform_(0, L)), float(torch.FloatTensor(1).uniform_(0, L))
                elif face == 1:
                    x_val, y_val, z_val = L, float(torch.FloatTensor(1).uniform_(0, L)), float(torch.FloatTensor(1).uniform_(0, L))
                elif face == 2:
                    x_val, y_val, z_val = float(torch.FloatTensor(1).uniform_(0, L)), 0.0, float(torch.FloatTensor(1).uniform_(0, L))
                elif face == 3:
                    x_val, y_val, z_val = float(torch.FloatTensor(1).uniform_(0, L)), L, float(torch.FloatTensor(1).uniform_(0, L))
                elif face == 4:
                    x_val, y_val, z_val = float(torch.FloatTensor(1).uniform_(0, L)), float(torch.FloatTensor(1).uniform_(0, L)), 0.0
                else:
                    x_val, y_val, z_val = float(torch.FloatTensor(1).uniform_(0, L)), float(torch.FloatTensor(1).uniform_(0, L)), L

                all_training_points.append(TrainingPoint(
                    x=x_val, y=y_val, z=z_val, t=t_val,
                    u_true=boundary_condition(x_val, y_val, z_val, t_val), type='boundary'
                ))

        # Convert to tensors
        interior_points = [p for p in all_training_points if p.type == 'interior']
        initial_points = [p for p in all_training_points if p.type == 'initial']
        boundary_points = [p for p in all_training_points if p.type == 'boundary']

        data = {
            'interior': {
                'x': torch.tensor([[p.x] for p in interior_points], dtype=torch.float32).to(device),
                'y': torch.tensor([[p.y] for p in interior_points], dtype=torch.float32).to(device),
                'z': torch.tensor([[p.z] for p in interior_points], dtype=torch.float32).to(device),
                't': torch.tensor([[p.t] for p in interior_points], dtype=torch.float32).to(device)
            },
            'initial': {
                'x': torch.tensor([[p.x] for p in initial_points], dtype=torch.float32).to(device),
                'y': torch.tensor([[p.y] for p in initial_points], dtype=torch.float32).to(device),
                'z': torch.tensor([[p.z] for p in initial_points], dtype=torch.float32).to(device),
                't': torch.tensor([[p.t] for p in initial_points], dtype=torch.float32).to(device),
                'u': torch.tensor([[p.u_true] for p in initial_points], dtype=torch.float32).to(device)
            },
            'boundary': {
                'x': torch.tensor([[p.x] for p in boundary_points], dtype=torch.float32).to(device),
                'y': torch.tensor([[p.y] for p in boundary_points], dtype=torch.float32).to(device),
                'z': torch.tensor([[p.z] for p in boundary_points], dtype=torch.float32).to(device),
                't': torch.tensor([[p.t] for p in boundary_points], dtype=torch.float32).to(device),
                'u': torch.tensor([[p.u_true] for p in boundary_points], dtype=torch.float32).to(device)
            }
        }
        return data

    def _compute_individual_losses(self, record_history=True):
        """Compute individual loss components with GPU-optimized processing."""
        mse_loss = nn.MSELoss()

        # 1. Initial condition loss
        sizes_init = [self.training_data['initial'][k].shape[0] for k in ['x', 'y', 'z', 't', 'u']]
        n0 = int(min(sizes_init))

        if device.type == 'cuda':
            optimal_batch = min(2048, n0) if self.use_transformer else min(8192, n0)
        else:
            optimal_batch = min(10000, n0)

        idx_initial = torch.randperm(n0, device=device)[:optimal_batch]

        initial_batch = torch.stack([
            self.training_data['initial']['x'][:n0][idx_initial],
            self.training_data['initial']['y'][:n0][idx_initial],
            self.training_data['initial']['z'][:n0][idx_initial],
            self.training_data['initial']['t'][:n0][idx_initial]
        ], dim=0)

        x_init, y_init, z_init, t_init = initial_batch[0], initial_batch[1], initial_batch[2], initial_batch[3]
        u_init = self.training_data['initial']['u'][:n0][idx_initial]

        u_pred_initial = self.forward(x_init, y_init, z_init, t_init)
        initial_loss = mse_loss(u_pred_initial, u_init)

        # 2. Peak loss
        n_peak_samples = 9 if device.type == 'cuda' else 5
        peak_offsets = torch.tensor([
            [0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [-0.01, 0.0, 0.0],
            [0.0, 0.01, 0.0], [0.0, -0.01, 0.0],
            [0.0, 0.0, 0.01], [0.0, 0.0, -0.01],
            [0.01, 0.01, 0.01], [-0.01, -0.01, -0.01]
        ], device=device)[:n_peak_samples]

        x_peak = torch.clamp(L / 2 + peak_offsets[:, 0:1], 0, L)
        y_peak = torch.clamp(L / 2 + peak_offsets[:, 1:2], 0, L)
        z_peak = torch.clamp(L / 2 + peak_offsets[:, 2:3], 0, L)
        t_peak = torch.zeros(n_peak_samples, 1, device=device)

        u_peak_true = torch.tensor([
            [initial_condition(x_peak[i].item(), y_peak[i].item(), z_peak[i].item())]
            for i in range(n_peak_samples)
        ], dtype=torch.float32, device=device)

        u_pred_peak = self.forward(x_peak, y_peak, z_peak, t_peak)
        peak_loss = mse_loss(u_pred_peak, u_peak_true)

        # 3. Boundary condition loss
        t_boundary_all = self.training_data['boundary']['t'].squeeze(1)
        n_boundary_all = t_boundary_all.shape[0]

        if device.type == 'cuda':
            n_boundary = min(1024 if self.use_transformer else 2048, n_boundary_all)
        else:
            n_boundary = min(1000, n_boundary_all)

        unique_times = torch.unique(t_boundary_all)
        n_time_bins = min(100, len(unique_times))
        samples_per_bin = n_boundary // n_time_bins

        boundary_indices = []
        for i in range(n_time_bins):
            if i < len(unique_times):
                mask = (t_boundary_all == unique_times[i])
                bin_indices = torch.nonzero(mask, as_tuple=True)[0]
                if len(bin_indices) > 0:
                    selected = bin_indices[torch.randperm(len(bin_indices))[:samples_per_bin]]
                    boundary_indices.extend(selected.tolist())

        remaining_samples = n_boundary - len(boundary_indices)
        if remaining_samples > 0:
            available = list(set(range(n_boundary_all)) - set(boundary_indices))
            if available:
                extra = torch.randperm(len(available))[:remaining_samples]
                boundary_indices.extend([available[i] for i in extra])

        boundary_indices = torch.tensor(boundary_indices[:n_boundary], device=device)

        boundary_batch = torch.stack([
            self.training_data['boundary']['x'][boundary_indices],
            self.training_data['boundary']['y'][boundary_indices],
            self.training_data['boundary']['z'][boundary_indices],
            self.training_data['boundary']['t'][boundary_indices]
        ], dim=0)

        x_boundary, y_boundary = boundary_batch[0], boundary_batch[1]
        z_boundary, t_boundary = boundary_batch[2], boundary_batch[3]
        u_boundary = self.training_data['boundary']['u'][boundary_indices]

        u_pred_boundary = self.forward(x_boundary, y_boundary, z_boundary, t_boundary)
        boundary_loss = mse_loss(u_pred_boundary, u_boundary)

        # 4. PDE residual loss with causal temporal weighting
        t_pde_all = self.training_data['interior']['t'].squeeze(1)
        n_pde_all = t_pde_all.shape[0]

        if device.type == 'cuda':
            n_pde = min(2048 if self.use_transformer else 4096, n_pde_all)
            chunk_size = 256 if self.use_transformer else 512
        else:
            n_pde = min(1000, n_pde_all)
            chunk_size = 200

        pde_indices = self._stratified_sampling(t_pde_all, n_pde, n_bins=nt)

        pde_batch = torch.stack([
            self.training_data['interior']['x'][pde_indices],
            self.training_data['interior']['y'][pde_indices],
            self.training_data['interior']['z'][pde_indices],
            self.training_data['interior']['t'][pde_indices]
        ], dim=0)

        x_pde, y_pde, z_pde, t_pde = pde_batch[0], pde_batch[1], pde_batch[2], pde_batch[3]

        causal_epsilon = getattr(self, '_causal_epsilon', 1.0)
        pde_losses = []

        if device.type == 'cuda':
            streams = [torch.cuda.Stream() for _ in range(2)]
            for i, chunk_start in enumerate(range(0, len(x_pde), chunk_size)):
                chunk_end = min(chunk_start + chunk_size, len(x_pde))
                stream = streams[i % 2]
                with torch.cuda.stream(stream):
                    pde_residual = self.compute_pde_residual(
                        x_pde[chunk_start:chunk_end], y_pde[chunk_start:chunk_end],
                        z_pde[chunk_start:chunk_end], t_pde[chunk_start:chunk_end])
                    causal_weights = torch.exp(-causal_epsilon * t_pde[chunk_start:chunk_end].squeeze() / T)
                    pde_losses.append(torch.mean(causal_weights * pde_residual ** 2))
            for s in streams:
                s.synchronize()
        else:
            for chunk_start in range(0, len(x_pde), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(x_pde))
                pde_residual = self.compute_pde_residual(
                    x_pde[chunk_start:chunk_end], y_pde[chunk_start:chunk_end],
                    z_pde[chunk_start:chunk_end], t_pde[chunk_start:chunk_end])
                causal_weights = torch.exp(-causal_epsilon * t_pde[chunk_start:chunk_end].squeeze() / T)
                pde_losses.append(torch.mean(causal_weights * pde_residual ** 2))

        pde_loss = torch.mean(torch.stack(pde_losses))

        # 5. Non-negativity constraint (on PDE + IC points)
        nonneg_parts = [torch.relu(-self.forward(x_pde, y_pde, z_pde, t_pde)) ** 2]
        nonneg_parts.append(torch.relu(-u_pred_initial) ** 2)
        nonneg_loss = torch.mean(torch.cat(nonneg_parts, dim=0))

        return {
            'initial': initial_loss, 'peak': peak_loss,
            'boundary': boundary_loss, 'pde': pde_loss, 'nonneg': nonneg_loss,
        }

    def _stratified_sampling(self, t_values, n_samples, n_bins=100):
        """Stratified sampling across time bins for temporal coverage."""
        t_min, t_max = t_values.min(), t_values.max()
        bin_edges = torch.linspace(t_min, t_max, n_bins + 1, device=t_values.device)
        samples_per_bin = n_samples // n_bins

        selected_indices = []
        for i in range(n_bins):
            mask = (t_values >= bin_edges[i]) & (t_values < bin_edges[i + 1])
            if i == n_bins - 1:
                mask = (t_values >= bin_edges[i]) & (t_values <= bin_edges[i + 1])
            bin_indices = torch.nonzero(mask, as_tuple=True)[0]
            if len(bin_indices) > 0:
                n_select = min(samples_per_bin, len(bin_indices))
                selected = bin_indices[torch.randperm(len(bin_indices))[:n_select]]
                selected_indices.extend(selected.tolist())

        remaining = n_samples - len(selected_indices)
        if remaining > 0:
            available = list(set(range(len(t_values))) - set(selected_indices))
            if available:
                extra = torch.randperm(len(available))[:remaining]
                selected_indices.extend([available[i] for i in extra])

        return torch.tensor(selected_indices[:n_samples], device=t_values.device)

    def _compute_validation_metrics(self, grid_size=10):
        """Compute validation metrics on held-out grid against analytical solution."""
        from physics import analytical_solution

        x_v = np.linspace(0, L, grid_size)
        y_v = np.linspace(0, L, grid_size)
        z_v = np.linspace(0, L, grid_size)
        t_v = np.linspace(0, T, grid_size)

        results = {'per_timeslice': {}}
        all_pred, all_true = [], []

        self.eval()
        with torch.no_grad():
            for ti, t_val in enumerate(t_v):
                preds, trues = [], []
                for xi in x_v:
                    for yi in y_v:
                        for zi in z_v:
                            u_true = analytical_solution(xi, yi, zi, t_val)
                            x_t = torch.tensor([[xi]], dtype=torch.float32, device=device)
                            y_t = torch.tensor([[yi]], dtype=torch.float32, device=device)
                            z_t = torch.tensor([[zi]], dtype=torch.float32, device=device)
                            t_t = torch.tensor([[t_val]], dtype=torch.float32, device=device)
                            u_pred = self.forward(x_t, y_t, z_t, t_t).item()
                            preds.append(u_pred)
                            trues.append(u_true)

                preds_arr = np.array(preds)
                trues_arr = np.array(trues)
                slice_mse = np.mean((preds_arr - trues_arr) ** 2)
                denom = np.sqrt(np.sum(trues_arr ** 2) + 1e-10)
                slice_rel_l2 = np.sqrt(np.sum((preds_arr - trues_arr) ** 2)) / denom

                results['per_timeslice'][f't={t_val:.3f}'] = {
                    'mse': float(slice_mse), 'rel_l2': float(slice_rel_l2),
                }
                all_pred.extend(preds)
                all_true.extend(trues)

        self.train()
        all_pred = np.array(all_pred)
        all_true = np.array(all_true)
        results['mse'] = float(np.mean((all_pred - all_true) ** 2))
        results['rel_l2'] = float(np.sqrt(np.sum((all_pred - all_true) ** 2)) /
                                  np.sqrt(np.sum(all_true ** 2) + 1e-10))
        results['min_pred'] = float(np.min(all_pred))
        results['max_pred'] = float(np.max(all_pred))
        results['neg_count'] = int(np.sum(all_pred < 0))
        return results

    @staticmethod
    def _collect_performance_metrics():
        """Collect GPU and CPU performance metrics."""
        metrics = {}
        if torch.cuda.is_available():
            metrics['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / 1e6
            metrics['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / 1e6
            metrics['gpu_max_memory_mb'] = torch.cuda.max_memory_allocated() / 1e6
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu',
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    parts = result.stdout.strip().split(',')
                    metrics['gpu_util_pct'] = float(parts[0].strip())
                    metrics['gpu_temp_c'] = float(parts[1].strip())
            except Exception:
                metrics['gpu_util_pct'] = -1.0
        else:
            metrics['gpu_memory_allocated_mb'] = 0.0
            metrics['gpu_util_pct'] = -1.0
        try:
            import psutil
            metrics['cpu_pct'] = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            metrics['ram_pct'] = mem.percent
            metrics['ram_used_gb'] = mem.used / 1e9
        except ImportError:
            metrics['cpu_pct'] = -1.0
            metrics['ram_pct'] = -1.0
        return metrics

    def train_radam(self, n_samples=100000, epochs=None, lr=1e-3, weight_decay=1e-2,
                    config=None):
        """Train PINN using RAdam + L-BFGS hybrid with ReLoBRaLo adaptive loss weighting.

        Phase 1: RAdam warm-up with curriculum learning
        Phase 2: L-BFGS second-order refinement

        References:
            Liu et al. (2020): RAdam (ICLR 2020)
            Bischof & Kraus (2025): ReLoBRaLo (CMAME)
        """
        if config:
            epochs = epochs if epochs is not None else get_nested(config, 'pinn.training.epochs', pinn_epochs)
            lr = get_nested(config, 'pinn.training.lr', lr)
            weight_decay = get_nested(config, 'pinn.training.weight_decay', weight_decay)
            n_samples = get_nested(config, 'pinn.training.n_samples', n_samples)
        elif epochs is None:
            epochs = pinn_epochs
        if epochs is None:
            epochs = pinn_epochs

        _logger.info(f"Training SPINN+PINNsFormer PINN with RAdam + ReLoBRaLo...")
        _logger.info(f"  Architecture: SPINN body networks → PINNsFormer encoder-decoder")
        _logger.info(f"  Epochs: {epochs}, LR: {lr}, Weight Decay: {weight_decay}")

        n_params = sum(p.numel() for p in self.parameters())
        _logger.info(f"  Total parameters: {n_params}")

        start_time = time.time()
        self.training_data = self._generate_training_data(n_samples)

        betas = tuple(get_nested(config, 'pinn.training.betas', [0.9, 0.999])) if config else (0.9, 0.999)
        eps = get_nested(config, 'pinn.training.eps', 1e-8) if config else 1e-8

        optimizer = torch.optim.RAdam(
            self.parameters(), lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, decoupled_weight_decay=True)

        sched_cfg = get_nested(config, 'pinn.scheduler', {}) if config else {}
        T_0 = max(50, epochs // sched_cfg.get('T_0_divisor', 6))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=sched_cfg.get('T_mult', 2),
            eta_min=sched_cfg.get('eta_min', 1e-6))

        losses = []
        all_loss_names = ['initial', 'peak', 'boundary', 'pde', 'nonneg']
        loss_component_history = {k: [] for k in all_loss_names}
        weight_history = {k: [] for k in all_loss_names}
        lr_history = []
        best_loss = float('inf')
        best_state = None

        val_cfg = get_nested(config, 'validation', {}) if config else {}
        val_enabled = val_cfg.get('enabled', True)
        val_interval = val_cfg.get('interval_pinn', 200)
        val_grid_size = val_cfg.get('grid_size', 10)
        validation_history = []
        performance_history = []

        acc_cfg = get_nested(config, 'pinn.accuracy', {}) if config else {}
        nonneg_weight = acc_cfg.get('nonneg_weight', 0.01)
        self._causal_epsilon = acc_cfg.get('causal_epsilon', 1.0)

        curriculum_cfg = acc_cfg.get('curriculum', {})
        curriculum_enabled = curriculum_cfg.get('enabled', True)
        phase1_ratio = curriculum_cfg.get('phase1_ratio', 0.30)
        phase2_ratio = curriculum_cfg.get('phase2_ratio', 0.40)
        phase1_end = int(epochs * phase1_ratio)
        phase2_end = int(epochs * (phase1_ratio + phase2_ratio))

        if curriculum_enabled:
            _logger.info(f"  Curriculum: Phase1 IC [0-{phase1_end}], "
                         f"Phase2 +PDE [{phase1_end}-{phase2_end}], "
                         f"Phase3 ReLoBRaLo [{phase2_end}-{epochs}]")

        # ReLoBRaLo
        if self.use_hard_constraints:
            loss_keys = ['initial', 'peak', 'pde', 'nonneg']
        else:
            loss_keys = ['initial', 'peak', 'boundary', 'pde', 'nonneg']

        relo_cfg = get_nested(config, 'pinn.relobralo', {}) if config else {}
        relobralo_alpha = relo_cfg.get('alpha', 0.999)
        relobralo_rho = relo_cfg.get('rho', 0.999)
        relobralo_tau = relo_cfg.get('tau', 1.0)

        lambdas = {k: 1.0 for k in loss_keys}
        lambdas['nonneg'] = nonneg_weight
        initial_losses = None
        prev_losses = None
        relobralo_keys = [k for k in loss_keys if k != 'nonneg']

        for epoch in range(epochs):
            optimizer.zero_grad()
            loss_components = self._compute_individual_losses()

            clean_losses = {k: torch.nan_to_num(loss_components[k], nan=1e6) for k in loss_keys}

            if curriculum_enabled:
                if epoch < phase1_end:
                    active_keys = [k for k in loss_keys if k in ['initial', 'peak', 'nonneg']]
                    if not self.use_hard_constraints and 'boundary' in loss_keys:
                        active_keys.append('boundary')
                elif epoch < phase2_end:
                    active_keys = list(loss_keys)
                    pde_ramp = (epoch - phase1_end) / max(phase2_end - phase1_end, 1)
                    lambdas['pde'] = pde_ramp  # Linear ramp from 0 to 1 (not compounding)
                else:
                    active_keys = list(loss_keys)
            else:
                active_keys = list(loss_keys)

            active_relo_keys = [k for k in active_keys if k != 'nonneg']

            if epoch == 0:
                initial_losses = {k: max(clean_losses[k].item(), 1e-12) for k in loss_keys}
                prev_losses = {k: max(clean_losses[k].item(), 1e-12) for k in loss_keys}
                total_loss = sum(clean_losses[k] for k in active_keys)
            else:
                current_loss_vals = {k: max(clean_losses[k].item(), 1e-12) for k in loss_keys}

                if len(active_relo_keys) > 1 and (not curriculum_enabled or epoch >= phase2_end):
                    n_active = len(active_relo_keys)
                    logits_prev = [current_loss_vals[k] / (prev_losses[k] * relobralo_tau + 1e-12) for k in active_relo_keys]
                    max_lp = max(logits_prev)
                    exp_prev = [math.exp(l - max_lp) for l in logits_prev]
                    sum_ep = sum(exp_prev)
                    lambdas_hat = {k: (exp_prev[i] / sum_ep) * n_active for i, k in enumerate(active_relo_keys)}

                    logits_init = [current_loss_vals[k] / (initial_losses[k] * relobralo_tau + 1e-12) for k in active_relo_keys]
                    max_li = max(logits_init)
                    exp_init = [math.exp(l - max_li) for l in logits_init]
                    sum_ei = sum(exp_init)
                    lambdas0_hat = {k: (exp_init[i] / sum_ei) * n_active for i, k in enumerate(active_relo_keys)}

                    for k in active_relo_keys:
                        lambdas[k] = (relobralo_rho * relobralo_alpha * lambdas[k]
                                      + (1 - relobralo_rho) * relobralo_alpha * lambdas0_hat[k]
                                      + (1 - relobralo_alpha) * lambdas_hat[k])

                prev_losses = current_loss_vals
                total_loss = sum(lambdas.get(k, 1.0) * clean_losses[k] for k in active_keys)

            total_loss.backward()
            grad_cfg = get_nested(config, 'pinn.gradient_clip', {}) if config else {}
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=grad_cfg.get('radam_max_norm', 1.0))
            optimizer.step()
            scheduler.step()

            loss_val = total_loss.item()
            losses.append(loss_val)

            for k in all_loss_names:
                loss_component_history[k].append(loss_components[k].item())
                weight_history[k].append(lambdas.get(k, 0.0))
            lr_history.append(optimizer.param_groups[0]['lr'])

            unweighted_loss = sum(clean_losses[k].item() for k in loss_keys)
            if unweighted_loss < best_loss:
                best_loss = unweighted_loss
                best_state = copy.deepcopy(self.state_dict())

            if (epoch + 1) % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                phase_str = ""
                if curriculum_enabled:
                    if epoch < phase1_end:
                        phase_str = " [Phase1:IC]"
                    elif epoch < phase2_end:
                        phase_str = f" [Phase2:+PDE ramp={(epoch - phase1_end) / max(phase2_end - phase1_end, 1):.2f}]"
                    else:
                        phase_str = " [Phase3:ReLoBRaLo]"
                _logger.info(f"  Epoch {epoch + 1}/{epochs}{phase_str}, Loss: {loss_val:.6e}, "
                             f"Unweighted: {unweighted_loss:.6e}, LR: {current_lr:.2e}")
                parts = [f"{k}: {loss_components[k].item():.6e} (w={lambdas.get(k, 0.0):.3f})" for k in all_loss_names]
                _logger.info(f"    {', '.join(parts)}")

            if val_enabled and (epoch + 1) % val_interval == 0:
                val_metrics = self._compute_validation_metrics(grid_size=val_grid_size)
                perf_metrics = self._collect_performance_metrics()
                elapsed = time.time() - start_time
                validation_history.append({
                    'epoch': epoch + 1, 'elapsed_sec': elapsed,
                    'mse': val_metrics['mse'], 'rel_l2': val_metrics['rel_l2'],
                    'min_pred': val_metrics['min_pred'], 'neg_count': val_metrics['neg_count'],
                    'per_timeslice': val_metrics['per_timeslice'],
                })
                performance_history.append({
                    'epoch': epoch + 1, 'elapsed_sec': elapsed,
                    'throughput_epochs_per_sec': (epoch + 1) / max(elapsed, 1e-6),
                    **perf_metrics,
                })
                _logger.info(f"    Validation: MSE={val_metrics['mse']:.6e}, "
                             f"RelL2={val_metrics['rel_l2']:.4f}, NegPreds={val_metrics['neg_count']}")

        if best_state is not None:
            self.load_state_dict(best_state)

        radam_time = time.time() - start_time
        _logger.info(f"  RAdam phase completed in {radam_time:.2f} seconds")
        _logger.info(f"  Best unweighted loss: {best_loss:.6e}")

        # ================================================================
        # Phase 2: L-BFGS refinement
        # ================================================================
        lbfgs_cfg = get_nested(config, 'pinn.lbfgs', {}) if config else {}
        lbfgs_enabled = lbfgs_cfg.get('enabled', True)
        lbfgs_epochs = lbfgs_cfg.get('max_iter', 200)
        if not lbfgs_enabled:
            lbfgs_epochs = 0

        _logger.info(f"Starting L-BFGS refinement ({lbfgs_epochs} iterations)...")

        lbfgs_optimizer = torch.optim.LBFGS(
            self.parameters(), lr=lbfgs_cfg.get('lr', 1.0), max_iter=20,
            max_eval=lbfgs_cfg.get('max_eval', 25),
            history_size=lbfgs_cfg.get('history_size', 50),
            tolerance_grad=lbfgs_cfg.get('tolerance_grad', 1e-9),
            tolerance_change=lbfgs_cfg.get('tolerance_change', 1e-11),
            line_search_fn=lbfgs_cfg.get('line_search_fn', 'strong_wolfe'))

        frozen_lambdas = {k: lambdas.get(k, 1.0) for k in loss_keys}
        mse_loss = nn.MSELoss()

        n_ic = min(4096, self.training_data['initial']['x'].shape[0])
        ic_idx = torch.randperm(self.training_data['initial']['x'].shape[0], device=device)[:n_ic]
        lbfgs_x_ic = self.training_data['initial']['x'][ic_idx]
        lbfgs_y_ic = self.training_data['initial']['y'][ic_idx]
        lbfgs_z_ic = self.training_data['initial']['z'][ic_idx]
        lbfgs_t_ic = self.training_data['initial']['t'][ic_idx]
        lbfgs_u_ic = self.training_data['initial']['u'][ic_idx]

        n_peak_samples = 9
        peak_offsets = torch.tensor([
            [0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [-0.01, 0.0, 0.0],
            [0.0, 0.01, 0.0], [0.0, -0.01, 0.0],
            [0.0, 0.0, 0.01], [0.0, 0.0, -0.01],
            [0.01, 0.01, 0.01], [-0.01, -0.01, -0.01]
        ], device=device)
        lbfgs_x_peak = torch.clamp(L / 2 + peak_offsets[:, 0:1], 0, L)
        lbfgs_y_peak = torch.clamp(L / 2 + peak_offsets[:, 1:2], 0, L)
        lbfgs_z_peak = torch.clamp(L / 2 + peak_offsets[:, 2:3], 0, L)
        lbfgs_t_peak = torch.zeros(n_peak_samples, 1, device=device)
        lbfgs_u_peak = torch.tensor([
            [initial_condition(lbfgs_x_peak[i].item(), lbfgs_y_peak[i].item(), lbfgs_z_peak[i].item())]
            for i in range(n_peak_samples)
        ], dtype=torch.float32, device=device)

        n_pde = min(512, self.training_data['interior']['x'].shape[0])
        pde_idx = torch.randperm(self.training_data['interior']['x'].shape[0], device=device)[:n_pde]

        for lbfgs_epoch in range(lbfgs_epochs):
            def closure():
                lbfgs_optimizer.zero_grad()
                u_pred_ic = self.forward(lbfgs_x_ic, lbfgs_y_ic, lbfgs_z_ic, lbfgs_t_ic)
                ic_loss = mse_loss(u_pred_ic, lbfgs_u_ic)
                u_pred_peak = self.forward(lbfgs_x_peak, lbfgs_y_peak, lbfgs_z_peak, lbfgs_t_peak)
                peak_loss = mse_loss(u_pred_peak, lbfgs_u_peak)

                x_p = self.training_data['interior']['x'][pde_idx].clone().detach().requires_grad_(True)
                y_p = self.training_data['interior']['y'][pde_idx].clone().detach().requires_grad_(True)
                z_p = self.training_data['interior']['z'][pde_idx].clone().detach().requires_grad_(True)
                t_p = self.training_data['interior']['t'][pde_idx].clone().detach().requires_grad_(True)

                u_p = self.forward(x_p, y_p, z_p, t_p)
                u_t = grad(u_p.sum(), t_p, create_graph=True, retain_graph=True)[0]
                u_x = grad(u_p.sum(), x_p, create_graph=True, retain_graph=True)[0]
                u_y = grad(u_p.sum(), y_p, create_graph=True, retain_graph=True)[0]
                u_z = grad(u_p.sum(), z_p, create_graph=True, retain_graph=True)[0]
                if u_t is None: u_t = torch.zeros_like(t_p)
                if u_x is None: u_x = torch.zeros_like(x_p)
                if u_y is None: u_y = torch.zeros_like(y_p)
                if u_z is None: u_z = torch.zeros_like(z_p)
                u_xx = grad(u_x.sum(), x_p, create_graph=True, retain_graph=True)[0]
                u_yy = grad(u_y.sum(), y_p, create_graph=True, retain_graph=True)[0]
                u_zz = grad(u_z.sum(), z_p, create_graph=True, retain_graph=True)[0]
                if u_xx is None: u_xx = torch.zeros_like(x_p)
                if u_yy is None: u_yy = torch.zeros_like(y_p)
                if u_zz is None: u_zz = torch.zeros_like(z_p)

                pde_residual = u_t - alpha * (u_xx + u_yy + u_zz)
                pde_loss = torch.mean(pde_residual ** 2)
                nonneg_loss = torch.mean(torch.relu(-u_p) ** 2)

                total = (frozen_lambdas.get('initial', 1.0) * ic_loss +
                         frozen_lambdas.get('peak', 1.0) * peak_loss +
                         frozen_lambdas.get('pde', 1.0) * pde_loss +
                         frozen_lambdas.get('nonneg', nonneg_weight) * nonneg_loss)
                total.backward()
                lbfgs_max_norm = grad_cfg.get('lbfgs_max_norm', 5.0)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=lbfgs_max_norm)
                return total

            try:
                loss_val = lbfgs_optimizer.step(closure)
            except RuntimeError as e:
                _logger.error(f"  L-BFGS error at step {lbfgs_epoch + 1}: {e}")
                break

            if loss_val is not None:
                lv = loss_val.item() if hasattr(loss_val, 'item') else float(loss_val)
                losses.append(lv)
            else:
                losses.append(losses[-1] if losses else 0.0)
                lv = losses[-1]

            with torch.no_grad():
                u_pred_ic = self.forward(lbfgs_x_ic, lbfgs_y_ic, lbfgs_z_ic, lbfgs_t_ic)
                ic_lv = mse_loss(u_pred_ic, lbfgs_u_ic).item()
                u_pred_pk = self.forward(lbfgs_x_peak, lbfgs_y_peak, lbfgs_z_peak, lbfgs_t_peak)
                pk_lv = mse_loss(u_pred_pk, lbfgs_u_peak).item()
                loss_component_history['initial'].append(ic_lv)
                loss_component_history['peak'].append(pk_lv)
                loss_component_history['boundary'].append(0.0)
                loss_component_history['pde'].append(max(lv - ic_lv - pk_lv, 0.0))
                loss_component_history['nonneg'].append(0.0)
                for k in all_loss_names:
                    weight_history[k].append(frozen_lambdas.get(k, 0.0))
                lr_history.append(1.0)
                unweighted = ic_lv + pk_lv + max(lv - ic_lv - pk_lv, 0.0)
                if unweighted < best_loss:
                    best_loss = unweighted
                    best_state = copy.deepcopy(self.state_dict())

            if (lbfgs_epoch + 1) % 20 == 0:
                _logger.info(f"  L-BFGS step {lbfgs_epoch + 1}/{lbfgs_epochs}, "
                             f"Loss: {lv:.6e}, Unweighted: {unweighted:.6e}")

        if best_state is not None:
            self.load_state_dict(best_state)

        training_time = time.time() - start_time
        _logger.info(f"  Total training completed in {training_time:.2f} seconds "
                     f"(RAdam: {radam_time:.2f}s + L-BFGS: {training_time - radam_time:.2f}s)")

        training_history = {
            'total_losses': losses,
            'loss_components': loss_component_history,
            'relobralo_weights': weight_history,
            'lr_history': lr_history,
            'best_unweighted_loss': best_loss,
            'final_weights': {k: frozen_lambdas[k] for k in loss_keys},
            'epochs': epochs + lbfgs_epochs,
            'radam_epochs': epochs,
            'lbfgs_epochs': lbfgs_epochs,
            'n_params': n_params,
            'validation_history': validation_history,
            'performance_history': performance_history,
        }
        return self.state_dict(), losses, training_time, training_history

    def train_adam(self, n_samples=10000):
        """Standard Adam training method (fallback)."""
        _logger.info("Training with Adam optimizer (fallback)...")
        n_params = sum(p.numel() for p in self.parameters())
        _logger.info(f"Total Parameters: {n_params}")
        start_time = time.time()
        if not hasattr(self, '_causal_epsilon'):
            self._causal_epsilon = 1.0

        self.training_data = self._generate_training_data(n_samples)
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        T_0 = max(50, pinn_epochs // 6)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=2, eta_min=1e-6)

        losses = []
        mse_loss = nn.MSELoss()

        for epoch in range(pinn_epochs):
            optimizer.zero_grad()
            loss_components = self._compute_individual_losses()
            total_loss = (
                torch.nan_to_num(loss_components['initial'], nan=1e30) +
                torch.nan_to_num(loss_components['peak'], nan=1e30) +
                torch.nan_to_num(loss_components['boundary'], nan=1e30) +
                torch.nan_to_num(loss_components['pde'], nan=1e30) +
                0.01 * torch.nan_to_num(loss_components['nonneg'], nan=0.0))
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            losses.append(total_loss.item())
            if (epoch + 1) % 100 == 0:
                _logger.info(f"Epoch {epoch + 1}/{pinn_epochs}, Loss: {total_loss.item():.6e}")

        training_time = time.time() - start_time
        return self.state_dict(), losses, training_time

    def _print_test_predictions(self):
        """Print test predictions for validation."""
        test_cases = [
            (L / 2, L / 2, L / 2, 0.0, "center, t=0"),
            (L / 2, L / 2, L / 2, 0.01, "center, t=0.01"),
            (L / 2, L / 2, L / 2, 0.1, "center, t=0.1"),
            (L / 2, L / 2, L / 2, 0.5, "center, t=0.5"),
            (L / 2, L / 2, L / 2, 1.0, "center, t=1.0"),
            (0.0, L / 2, L / 2, 0.1, "boundary(x=0)"),
            (L, L / 2, L / 2, 0.5, "boundary(x=L)"),
            (0.7 * L, 0.7 * L, 0.7 * L, 0.3, "off-center, t=0.3")
        ]
        _logger.info(f"Current predictions (SPINN+PINNsFormer: {'Enabled' if self.use_transformer else 'Off'})")
        _logger.info("-" * 90)
        _logger.info(f"{'Location':^25} | {'True':^10} | {'Predicted':^10} | {'Error':^10} | {'Relative':^10}")
        _logger.info("-" * 90)
        with torch.no_grad():
            for x_val, y_val, z_val, t_val, desc in test_cases:
                x_t = torch.tensor([[x_val]], dtype=torch.float32, device=device)
                y_t = torch.tensor([[y_val]], dtype=torch.float32, device=device)
                z_t = torch.tensor([[z_val]], dtype=torch.float32, device=device)
                t_t = torch.tensor([[t_val]], dtype=torch.float32, device=device)
                u_pred = self.forward(x_t, y_t, z_t, t_t).item()
                u_true = analytical_solution(x_val, y_val, z_val, t_val)
                error = abs(u_pred - u_true)
                rel_error = 100 * error / (abs(u_true) + sys.float_info.epsilon)
                _logger.info(f"{desc:^25} | {u_true:^10.6f} | {u_pred:^10.6f} | {error:^10.6f} | {rel_error:^10.2f}%")


# ================================================
# Evaluation function
# ================================================
def evaluate_pinn(model: PINN) -> np.ndarray:
    """Evaluate PINN model on the full spatial-temporal grid."""
    _logger.info("Evaluating SPINN+PINNsFormer PINN model...")
    model.eval()

    x = np.linspace(0, L, nx)
    y = np.linspace(0, L, ny)
    z = np.linspace(0, L, nz)
    t = np.linspace(0, T, nt)

    X, Y, Z, T_mesh = np.meshgrid(x, y, z, t, indexing='ij')

    X_flat = X.flatten().reshape(-1, 1)
    Y_flat = Y.flatten().reshape(-1, 1)
    Z_flat = Z.flatten().reshape(-1, 1)
    T_flat = T_mesh.flatten().reshape(-1, 1)

    X_tensor = torch.FloatTensor(X_flat).to(device)
    Y_tensor = torch.FloatTensor(Y_flat).to(device)
    Z_tensor = torch.FloatTensor(Z_flat).to(device)
    T_tensor = torch.FloatTensor(T_flat).to(device)

    batch_size = 2000
    u_pred_list = []

    with torch.no_grad():
        for i in range(0, len(X_flat), batch_size):
            end_idx = min(i + batch_size, len(X_flat))
            u_pred_batch = model(X_tensor[i:end_idx], Y_tensor[i:end_idx],
                                 Z_tensor[i:end_idx], T_tensor[i:end_idx]).cpu().numpy()
            u_pred_list.append(u_pred_batch)

    u_pred = np.vstack(u_pred_list).flatten()

    _logger.info(f"  SPINN+PINNsFormer evaluation completed.")
    _logger.info(f"  Prediction range: [{np.min(u_pred):.6f}, {np.max(u_pred):.6f}]")
    _logger.info(f"  Mean: {np.mean(u_pred):.6f}, Std: {np.std(u_pred):.6f}")

    return u_pred


# ================================================
# Training wrapper function
# ================================================
def train_pinn(
    use_hard_constraints=True,
    boundary_epsilon=0.1,
    use_fourier_features=True,
    use_transformer=True,
    transformer_memory_efficient=True,
    layers=[5, 128, 256, 256, 128, 1],
    config=None) -> Tuple[PINN, List[float], float]:
    """Train PINN with SPINN + PINNsFormer architecture."""

    if config:
        arch_cfg = get_nested(config, 'pinn.architecture', {})
        layers = arch_cfg.get('layers', layers)
        use_hard_constraints = arch_cfg.get('use_hard_constraints', use_hard_constraints)
        boundary_epsilon = arch_cfg.get('boundary_epsilon', boundary_epsilon)
        use_fourier_features = arch_cfg.get('use_fourier_features', use_fourier_features)
        use_transformer = arch_cfg.get('use_transformer', use_transformer)
        transformer_memory_efficient = arch_cfg.get('transformer_memory_efficient', transformer_memory_efficient)

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Build transformer_config with SPINN settings from JSON config
    transformer_config = None
    if config:
        t_cfg = get_nested(config, 'pinn.architecture.transformer', {})
        spinn_cfg = get_nested(config, 'pinn.architecture.spinn', {})
        if t_cfg:
            transformer_config = {
                'seq_length': t_cfg.get('seq_length', 8),
                'd_model': t_cfg.get('d_model', 64),
                'n_heads': t_cfg.get('n_heads', 4),
                'n_layers': t_cfg.get('n_layers', 2),
                'd_ff': t_cfg.get('d_ff', 256),
                'dropout': t_cfg.get('dropout', 0.1),
                'spinn_hidden': spinn_cfg.get('hidden_dim', 64),
                'spinn_layers': spinn_cfg.get('n_hidden_layers', 2),
            }

    model = PINN(
        layers=layers,
        use_hard_constraints=use_hard_constraints,
        boundary_epsilon=boundary_epsilon,
        fourier_features=use_fourier_features,
        use_transformer=use_transformer,
        transformer_config=transformer_config,
        transformer_memory_efficient=transformer_memory_efficient
    ).to(device)

    _logger.info(f"SPINN+PINNsFormer PINN Configuration:")
    _logger.info(f"  Architecture: SPINN body networks → PINNsFormer encoder-decoder")
    _logger.info(f"  Hard constraints: {'Enabled' if use_hard_constraints else 'Disabled'}")
    _logger.info(f"  Transformer: {'Memory Efficient' if transformer_memory_efficient else 'Full'}")
    _logger.info(f"  Fourier Features: {use_fourier_features}")
    _logger.info(f"  Device: {device}")

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    n_training_samples = get_nested(config, 'pinn.training.n_samples', None) if config else None
    if n_training_samples is None:
        if use_transformer and not transformer_memory_efficient:
            n_training_samples = 30000
        elif use_transformer:
            n_training_samples = 50000
        else:
            n_training_samples = 100000

    state_dict, losses, training_time, training_history = model.train_radam(
        n_samples=n_training_samples, config=config)

    _logger.info(f"SPINN+PINNsFormer training completed in {training_time:.2f} seconds")

    model.training_config = {
        'use_hard_constraints': use_hard_constraints,
        'boundary_epsilon': boundary_epsilon,
        'use_fourier_features': use_fourier_features,
        'use_transformer': use_transformer,
        'transformer_memory_efficient': transformer_memory_efficient,
        'layers': layers
    }
    model.training_history = training_history

    return model, losses, training_time

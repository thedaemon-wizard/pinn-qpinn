"""GPT-based Quantum Circuit Generation Pipeline

Contains:
- QuantumCircuitTemplate dataclass
- QuantumCircuitGPT (GPT model for circuit generation)
- QuantumCircuitDataset
- UnsupervisedQuantumEnergyEstimator
- MultiObjectiveBayesianCircuitOptimizer
- GQEQuantumCircuitGeneratorWithGPT
- OptimizedQuantumDevice (PennyLane device wrapper)
- Parallel processing utilities
"""
from config import *
from physics import TrainingPoint, to_python_float
from physics import initial_condition, boundary_condition, analytical_solution
from device_manager import QuantumDeviceManager, QPUConfig
from pinnsformer import WaveletActivation

import logging
_logger = logging.getLogger('benchmark.gpt_circuit')

#================================================
# Data class definitions
#================================================
@dataclass
class QuantumCircuitTemplate:
    """GQE optimized quantum circuit template"""
    n_qubits: int
    n_layers: int
    gate_sequence: List[Dict[str, Any]]
    parameter_map: Dict[str, int]
    entangling_pattern: str
    noise_resilience_score: float
    hardware_efficiency: float
    expressivity_score: float
    estimated_energy: float
    depth_score: float
    diversity_score: float
    mitigation_score: float
    param_efficiency: float
    metadata: Dict[str, Any] = field(default_factory=dict)

if hasattr(torch.serialization, 'add_safe_globals'):
    # Register custom classes as safe globals
    torch.serialization.add_safe_globals([QuantumCircuitTemplate])
    torch.serialization.add_safe_globals([np._core.multiarray.scalar])
    torch.serialization.add_safe_globals([np.dtype])
    torch.serialization.add_safe_globals([np.dtypes.Float32DType])
    torch.serialization.add_safe_globals([np.dtypes.Float64DType])
    torch.serialization.add_safe_globals([np.dtypes.StrDType])

#================================================
# GPT-based quantum circuit generator
#================================================
class QuantumCircuitGPT(nn.Module):
    """GPT model for quantum circuit generation"""

    def __init__(self, vocab_size, n_embd=256, n_head=8, n_layer=6,
                 block_size=128, dropout=0.1):
        super().__init__()

        # GPT-2 configuration
        self.config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            n_ctx=block_size,
            n_positions=block_size,
            dropout=dropout,
            use_cache=False
        )

        # GPT-2 model
        self.transformer = GPT2Model(self.config)

        # Language modeling head
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Energy prediction head (predicts expected energy of circuit)
        self.energy_head = nn.Sequential(
            nn.Linear(n_embd, n_embd // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(n_embd // 2, 1)
        )

        # Initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, energies=None):
        # Transformer processing
        transformer_outputs = self.transformer(idx)
        hidden_states = transformer_outputs.last_hidden_state

        # Language modeling output
        logits = self.lm_head(hidden_states)

        # Energy prediction (from hidden state of last token)
        energy_pred = self.energy_head(hidden_states[:, -1, :])

        loss = None
        if targets is not None:
            # Cross entropy loss (next token prediction)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            loss_ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

            # Energy prediction loss
            if energies is not None:
                loss_energy = F.mse_loss(energy_pred.squeeze(), energies)
                loss = loss_ce + 0.1 * loss_energy  # Weighted sum
            else:
                loss = loss_ce

        return logits, loss, energy_pred

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=0.9):
        """Generate quantum circuit sequences"""
        self.eval()

        for _ in range(max_new_tokens):
            # Predict with current sequence
            idx_cond = idx if idx.size(1) <= self.config.n_ctx else idx[:, -self.config.n_ctx:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Find positions where cumulative probability exceeds top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float('Inf')

            # Sampling
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

class QuantumCircuitDataset(Dataset):
    """Quantum circuit dataset (for GPT training)"""

    def __init__(self, sequences, energies, block_size=128):
        self.sequences = sequences
        self.energies = energies
        self.block_size = block_size

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        energy = self.energies[idx]

        # Padding
        if len(seq) > self.block_size:
            seq = seq[:self.block_size]
        else:
            seq = seq + [0] * (self.block_size - len(seq))  # Pad with 0

        return torch.tensor(seq, dtype=torch.long), torch.tensor(energy, dtype=torch.float32)

#================================================
# UnsupervisedQuantumEnergyEstimator
#================================================

class UnsupervisedQuantumEnergyEstimator:
    """Unsupervised quantum energy estimator (noise-aware version)

    References:
    - Mitarai et al. "Quantum circuit learning" Phys. Rev. A 98, 032309 (2018)
    - Abbas et al. "The power of quantum neural networks" Nat Comput Sci 1, 403-409 (2021)
    - Schuld et al. "Evaluating analytic gradients on quantum hardware" Phys. Rev. A 99, 032331 (2019)
    - Endo et al. "Practical Quantum Error Mitigation for Near-Future Applications" Phys. Rev. X 11, 031057 (2021)
    """

    def __init__(self, n_qubits: int, n_layers: int = 4,
                 use_noise: bool = False, noise_model: str = 'realistic',
                 shots: Optional[int] = None):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.use_noise = use_noise
        self.noise_model = noise_model
        self.shots = shots if shots is not None else (2048 if use_noise else None)

        self.measurement_history = []
        self.circuit_features = []

        # Parameters for quantum kernel estimation
        self.kernel_bandwidth = 1.0
        self.n_measurement_bases = 2**n_qubits  # Limit number of measurement bases

        # For unsupervised clustering
        self.n_energy_clusters = 10
        self.energy_estimator = None

        # Fix feature dimension
        self.feature_dim = None
        self.pca = None
        self.kmeans = None
        self.scaler = None

        # Noise parameters
        self.noise_params = self._initialize_noise_params()

        # Error mitigation parameters
        self.error_mitigation_enabled = use_noise
        self.zero_noise_extrapolation_factors = [1.0, 1.5, 2.0] if use_noise else [1.0]

    def _initialize_noise_params(self) -> Dict[str, float]:
        """Initialize noise parameters"""
        if self.noise_model == 'light':
            return {
                'depolarizing_1q': 0.001,
                'depolarizing_2q': 0.01,
                'amplitude_damping': 0.0005,
                'phase_damping': 0.0005,
                'readout_error': 0.01
            }
        elif self.noise_model == 'realistic':
            return {
                'depolarizing_1q': 0.002,
                'depolarizing_2q': 0.02,
                'amplitude_damping': 0.001,
                'phase_damping': 0.001,
                'readout_error': 0.02
            }
        elif self.noise_model == 'heavy':
            return {
                'depolarizing_1q': 0.005,
                'depolarizing_2q': 0.05,
                'amplitude_damping': 0.002,
                'phase_damping': 0.002,
                'readout_error': 0.05
            }
        else:
            return {
                'depolarizing_1q': 0.0,
                'depolarizing_2q': 0.0,
                'amplitude_damping': 0.0,
                'phase_damping': 0.0,
                'readout_error': 0.0
            }

    def _create_device(self, shots: Optional[int] = None):
        """Create appropriate quantum device"""
        if self.use_noise:
            # Device with noise
            return qml.device('default.mixed', wires=self.n_qubits,
                             shots=shots if shots is not None else self.shots)
        else:
            # Noiseless device
            if shots is not None:
                return qml.device('default.qubit', wires=self.n_qubits, shots=shots)
            else:
                return qml.device('default.qubit', wires=self.n_qubits)

    def _apply_noise_to_circuit(self, wire: int, gate_type: str = '1q'):
        """Apply noise to circuit"""
        if not self.use_noise:
            return

        # Gate noise
        if gate_type == '1q':
            if np.random.rand() < self.noise_params['depolarizing_1q']:
                qml.DepolarizingChannel(self.noise_params['depolarizing_1q'], wires=wire)
        else:  # 2q gate
            if np.random.rand() < self.noise_params['depolarizing_2q']:
                qml.DepolarizingChannel(self.noise_params['depolarizing_2q'], wires=wire)

        # Amplitude damping
        if np.random.rand() < self.noise_params['amplitude_damping']:
            qml.AmplitudeDamping(self.noise_params['amplitude_damping'], wires=wire)

        # Phase damping
        if np.random.rand() < self.noise_params['phase_damping']:
            qml.PhaseDamping(self.noise_params['phase_damping'], wires=wire)

    def _apply_circuit_template_with_noise(self, template, params: np.ndarray,
                                          noise_scale: float = 1.0):
        """Apply circuit template with noise"""
        param_idx = 0

        for gate_info in template.gate_sequence:
            gate_type = gate_info['gate']
            qubits = gate_info['qubits']

            # Validate qubit indices
            if any(q >= self.n_qubits for q in qubits):
                continue

            # Pre-gate noise (with scaling)
            if self.use_noise and noise_scale > 0:
                for q in qubits:
                    if np.random.rand() < noise_scale * 0.1:
                        self._apply_noise_to_circuit(q, '1q' if len(qubits) == 1 else '2q')

            # Parameterized gates
            if gate_type == 'RY' and gate_info.get('trainable', False):
                if param_idx < len(params):
                    qml.RY(params[param_idx], wires=qubits[0])
                    param_idx += 1
            elif gate_type == 'RZ' and gate_info.get('trainable', False):
                if param_idx < len(params):
                    qml.RZ(params[param_idx], wires=qubits[0])
                    param_idx += 1
            elif gate_type == 'RX' and gate_info.get('trainable', False):
                if param_idx < len(params):
                    qml.RX(params[param_idx], wires=qubits[0])
                    param_idx += 1
            # Fixed gates
            elif gate_type == 'H':
                qml.Hadamard(wires=qubits[0])
            elif gate_type == 'CNOT' and len(qubits) >= 2:
                if qubits[0] != qubits[1]:
                    qml.CNOT(wires=qubits[:2])
            elif gate_type == 'CZ' and len(qubits) >= 2:
                if qubits[0] != qubits[1]:
                    qml.CZ(wires=qubits[:2])

            # Post-gate noise (with scaling)
            if self.use_noise and noise_scale > 0:
                for q in qubits:
                    self._apply_noise_to_circuit(q, '1q' if len(qubits) == 1 else '2q')

    def _extract_quantum_features(self, template, input_data: np.ndarray) -> np.ndarray:
        """Extract quantum features (noise-aware version)"""
        dev = self._create_device(shots=self.shots if self.use_noise else None)
        measurement_bases = self._generate_measurement_bases()
        features = []

        # Prepare input data
        prepared_data = self._prepare_input_data(input_data)

        # Calculate expectation values with different measurement bases
        for basis in measurement_bases:
            if self.use_noise and self.error_mitigation_enabled:
                # Use zero noise extrapolation
                extrapolated_features = self._zero_noise_extrapolation(
                    template, prepared_data, basis
                )
                features.extend(extrapolated_features)
            else:
                # Normal measurement
                @qml.qnode(dev)
                def feature_circuit():
                    # Data encoding
                    if self.use_noise:
                        # Consider noise in initial state preparation
                        qml.AmplitudeEmbedding(
                            prepared_data,
                            wires=range(self.n_qubits),
                            normalize=True,
                            pad_with=0.0
                        )
                        # Noise after state preparation
                        for i in range(self.n_qubits):
                            if np.random.rand() < 0.05:
                                self._apply_noise_to_circuit(i, '1q')
                    else:
                        qml.AmplitudeEmbedding(
                            prepared_data,
                            wires=range(self.n_qubits),
                            normalize=True,
                            pad_with=0.0
                        )

                    # Apply variational circuit
                    param_values = np.random.uniform(-np.pi, np.pi,
                                                   size=len(template.parameter_map))
                    self._apply_circuit_template_with_noise(template, param_values)

                    # Measurement
                    expectations = []
                    for obs in basis:
                        expectations.append(qml.expval(obs))
                    return expectations

                try:
                    result = feature_circuit()
                    features.extend(result)
                except Exception as e:
                    _logger.warning(f"Feature extraction error: {e}")
                    features.extend([0.0] * len(basis))

        features_array = np.array(features)

        # Record feature dimension on first run
        if self.feature_dim is None:
            self.feature_dim = len(features_array)
            _logger.info(f"Feature dimension set: {self.feature_dim}")

        return features_array

    def _zero_noise_extrapolation(self, template, prepared_data: np.ndarray,
                                  basis: List[qml.operation.Operator]) -> List[float]:
        """Zero noise extrapolation measurement

        References:
        - Li & Benjamin "Efficient Variational Quantum Simulator Incorporating Active Error Minimization"
          Phys. Rev. X 7, 021050 (2017)
        """
        results_at_different_noise = []

        for noise_scale in self.zero_noise_extrapolation_factors:
            dev = self._create_device(shots=self.shots)

            @qml.qnode(dev)
            def scaled_noise_circuit():
                # Data encoding
                qml.AmplitudeEmbedding(
                    prepared_data,
                    wires=range(self.n_qubits),
                    normalize=True,
                    pad_with=0.0
                )

                # State preparation with scaled noise
                if noise_scale > 1.0:
                    for i in range(self.n_qubits):
                        if np.random.rand() < 0.05 * (noise_scale - 1.0):
                            self._apply_noise_to_circuit(i, '1q')

                # Apply variational circuit (with scaled noise)
                param_values = np.random.uniform(-np.pi, np.pi,
                                               size=len(template.parameter_map))
                self._apply_circuit_template_with_noise(template, param_values, noise_scale)

                # Measurement
                expectations = []
                for obs in basis:
                    expectations.append(qml.expval(obs))
                return expectations

            try:
                result = scaled_noise_circuit()
                results_at_different_noise.append(result)
            except Exception as e:
                _logger.warning(f"Zero noise extrapolation error: {e}")
                results_at_different_noise.append([0.0] * len(basis))

        # Richardson extrapolation
        if len(results_at_different_noise) >= 2:
            # Linear extrapolation
            extrapolated = []
            for i in range(len(basis)):
                values = [r[i] for r in results_at_different_noise]
                # Least squares extrapolation
                coeffs = np.polyfit(self.zero_noise_extrapolation_factors[:len(values)],
                                   values, deg=1)
                extrapolated_value = np.polyval(coeffs, 0.0)  # Value at noise=0
                extrapolated.append(extrapolated_value)
            return extrapolated
        else:
            return results_at_different_noise[0]

    def _analyze_measurements(self, template, input_data: np.ndarray) -> Dict[str, float]:
        """Statistical analysis of measurement results (noise-aware version)"""
        dev = self._create_device(shots=self.shots if self.shots else 1000)

        # Prepare input data
        prepared_data = self._prepare_input_data(input_data)

        @qml.qnode(dev)
        def measurement_circuit():
            # Data encoding
            qml.AmplitudeEmbedding(
                prepared_data,
                wires=range(self.n_qubits),
                normalize=True,
                pad_with=0.0
            )

            # State preparation with noise
            if self.use_noise:
                for i in range(self.n_qubits):
                    if np.random.rand() < 0.05:
                        self._apply_noise_to_circuit(i, '1q')

            # Circuit execution
            param_values = np.random.uniform(-np.pi, np.pi,
                                           size=len(template.parameter_map))
            self._apply_circuit_template_with_noise(template, param_values)

            # Add readout error
            if self.use_noise and self.noise_params['readout_error'] > 0:
                for i in range(self.n_qubits):
                    if np.random.rand() < self.noise_params['readout_error']:
                        qml.BitFlip(self.noise_params['readout_error'], wires=i)

            # Measurement in computational basis
            return qml.counts(wires=range(self.n_qubits))

        try:
            # Multiple measurements to improve statistics
            all_counts = {}
            n_repetitions = 3 if self.use_noise else 1

            for _ in range(n_repetitions):
                counts = measurement_circuit()
                for state, count in counts.items():
                    all_counts[state] = all_counts.get(state, 0) + count

            # Calculate statistics
            total_shots = sum(all_counts.values())
            probabilities = {state: count/total_shots for state, count in all_counts.items()}

            # Energy statistics
            mean_bitstring_value = np.mean([
                int(state, 2) * prob
                for state, prob in probabilities.items()
            ])

            # Calculate variance (ensure minimum value)
            bitstring_values = [
                int(state, 2)
                for state, count in all_counts.items()
                for _ in range(count)
            ]

            if len(bitstring_values) > 1:
                variance = np.var(bitstring_values)
            else:
                variance = 1e-6  # Set minimum variance

            # Handle zero variance case
            variance = max(variance, 1e-6)

            # Readout error correction
            if self.use_noise and self.noise_params['readout_error'] > 0:
                # Simple readout error mitigation
                correction_factor = 1.0 / (1.0 - 2 * self.noise_params['readout_error'])
                mean_bitstring_value *= correction_factor
                variance *= correction_factor**2

            return {
                'mean': mean_bitstring_value,
                'variance': variance,  # Guaranteed minimum value
                'entropy': self._compute_shannon_entropy(probabilities),
                'purity': self._estimate_purity(probabilities)
            }

        except Exception as e:
            _logger.error(f"Measurement analysis error: {e}")
            # Default values on error (variance set to positive value)
            return {
                'mean': 2**(self.n_qubits-1),
                'variance': 1.0,  # Positive value instead of 0
                'entropy': 0.5,
                'purity': 0.5
            }

    def _estimate_purity(self, probabilities: Dict[str, float]) -> float:
        """Estimate state purity"""
        # Estimate purity from measurement results
        purity = sum(prob**2 for prob in probabilities.values())
        return purity

    def _apply_quantum_error_mitigation(self, raw_energy: float,
                                      measurement_stats: Dict[str, float]) -> float:
        """Quantum error mitigation (enhanced version)

        References:
        - Temme et al. "Error mitigation for short-depth quantum circuits"
          Phys. Rev. Lett. 119, 180509 (2017)
        - Endo et al. "Hybrid quantum-classical algorithms and quantum error mitigation"
          J. Phys. Soc. Jpn. 90, 032001 (2021)
        """
        if not self.use_noise:
            # Simple correction only for noiseless case
            return raw_energy

        # 1. Statistical error-based correction
        noise_factor = 1.0 + 0.1 * measurement_stats['variance'] / (measurement_stats['mean'] + 1e-10)

        # 2. Purity-based correction
        purity = measurement_stats.get('purity', 1.0)
        purity_correction = 1.0 / (purity + 0.1)  # Larger correction for lower purity

        # 3. Entropy-based correction
        entropy = measurement_stats.get('entropy', 0.0)
        entropy_correction = 1.0 + 0.05 * entropy

        # Overall correction
        mitigated_energy = raw_energy / (noise_factor * purity_correction * entropy_correction)

        # 4. Apply physical constraints
        min_energy = -2.0 * (self.n_qubits - 1)
        mitigated_energy = max(mitigated_energy, min_energy)

        # 5. Limit variation
        if hasattr(self, '_last_mitigated_energy'):
            # Smooth if difference from previous value is too large
            max_change = 0.5 * abs(self._last_mitigated_energy)
            if abs(mitigated_energy - self._last_mitigated_energy) > max_change:
                mitigated_energy = self._last_mitigated_energy + np.sign(
                    mitigated_energy - self._last_mitigated_energy) * max_change

        self._last_mitigated_energy = mitigated_energy

        return mitigated_energy

    def estimate_energy_unsupervised(self, template, input_data: np.ndarray) -> float:
        """Energy estimation by unsupervised learning (noise-aware version)"""
        try:
            # Prepare input data
            prepared_input = self._prepare_input_data(input_data)

            # 1. Quantum feature map calculation
            quantum_features = self._extract_quantum_features(template, prepared_input)

            # 2. Variational energy estimation
            energy = self._variational_energy_estimation(template, quantum_features)

            # 3. Statistical analysis of measurement results
            measurement_stats = self._analyze_measurements(template, prepared_input)

            # 4. Energy correction (stronger correction if noise present)
            corrected_energy = self._apply_quantum_error_mitigation(energy, measurement_stats)

            # 5. Additional statistical correction (if noise present)
            if self.use_noise:
                # Bayesian correction
                if hasattr(self, 'energy_history') and len(self.energy_history) > 10:
                    # Estimate prior distribution from past energy values
                    prior_mean = np.mean(self.energy_history[-10:])
                    prior_std = np.std(self.energy_history[-10:])

                    # Set minimum standard deviation (prevent division by zero)
                    min_std = 1e-10
                    prior_std = max(prior_std, min_std)

                    # Bayesian update
                    likelihood_std = measurement_stats['variance']**0.5
                    likelihood_std = max(likelihood_std, min_std)

                    # Safe calculation
                    posterior_variance = 1.0 / (1.0/(prior_std**2) + 1.0/(likelihood_std**2))
                    posterior_mean = posterior_variance * (
                        prior_mean/(prior_std**2) + corrected_energy/(likelihood_std**2)
                    )

                    # Apply correction (prevent sudden changes)
                    corrected_energy = 0.7 * corrected_energy + 0.3 * posterior_mean

                # Update energy history
                if not hasattr(self, 'energy_history'):
                    self.energy_history = []
                self.energy_history.append(corrected_energy)
                if len(self.energy_history) > 100:
                    self.energy_history = self.energy_history[-100:]

            return corrected_energy

        except Exception as e:
            _logger.error(f"Unsupervised energy estimation error: {e}")
            import traceback
            traceback.print_exc()
            return -1.0 * self.n_qubits

    def _prepare_input_data(self, input_data: np.ndarray) -> np.ndarray:
        """Prepare input data to appropriate dimensions (PennyLane optimized version)"""
        required_dim = 2**self.n_qubits

        # Validate input data
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data)

        # Adjust data shape
        input_data = input_data.flatten()

        if len(input_data) >= required_dim:
            # Extract principal components if data is large
            # Select most important components (energetically)
            if len(input_data) > required_dim * 2:
                # Use FFT to select important components in frequency domain
                fft_data = np.fft.fft(input_data)
                # Prioritize low frequency components
                indices = np.argsort(np.abs(fft_data))[:required_dim]
                prepared_data = input_data[indices]
            else:
                prepared_data = input_data[:required_dim]
        else:
            # Physically meaningful padding if data is small
            prepared_data = np.zeros(required_dim)
            prepared_data[:len(input_data)] = input_data

            # Smoothly interpolate remaining elements
            if len(input_data) > 1:
                # Linear interpolation
                for i in range(len(input_data), required_dim):
                    # Assume periodic boundary conditions
                    idx1 = (i - len(input_data)) % len(input_data)
                    idx2 = (idx1 + 1) % len(input_data)
                    weight = (i - len(input_data)) / len(input_data)
                    prepared_data[i] = (1 - weight) * input_data[idx1] + weight * input_data[idx2]

        # Normalization (following PennyLane requirements)
        norm = np.linalg.norm(prepared_data)
        if norm > 1e-10:
            prepared_data = prepared_data / norm
        else:
            # Physically meaningful initial state for zero vector case
            # Ground state with small perturbation
            prepared_data = np.zeros(required_dim)
            prepared_data[0] = 1.0  # Ground state
            # Add small excitation
            for i in range(1, min(4, required_dim)):
                prepared_data[i] = 0.1 * np.exp(-i/2)
            # Renormalize
            prepared_data = prepared_data / np.linalg.norm(prepared_data)

        return prepared_data

    def _generate_measurement_bases(self) -> List[List[qml.operation.Operator]]:
        """Generate random measurement bases (based on Haar measure)"""
        measurement_bases = []

        # Pauli measurement bases
        pauli_ops = [qml.PauliX, qml.PauliY, qml.PauliZ]

        # Random single-qubit measurements
        for _ in range(self.n_measurement_bases):
            basis = []
            for i in range(self.n_qubits):
                # Randomly select Pauli basis
                op = np.random.choice(pauli_ops)
                basis.append(op(i))
            measurement_bases.append(basis)

        # Two-qubit correlations for entanglement measurement
        if self.n_qubits >= 2:
            for i in range(min(self.n_qubits - 1, 5)):
                # ZZ correlation
                measurement_bases.append([qml.PauliZ(i) @ qml.PauliZ(i+1)])
                # XX correlation
                measurement_bases.append([qml.PauliX(i) @ qml.PauliX(i+1)])

        return measurement_bases

    def _variational_energy_estimation(self, template, quantum_features: np.ndarray) -> float:
        """Energy estimation by variational method"""
        from sklearn.preprocessing import StandardScaler

        # Add features to history
        self.circuit_features.append(quantum_features)

        # Feature standardization
        if self.scaler is None:
            self.scaler = StandardScaler()

        # Execute clustering when enough data is collected
        if len(self.circuit_features) >= self.n_energy_clusters:
            # Convert all features to array
            all_features = np.array(self.circuit_features)
            scaled_features = self.scaler.fit_transform(all_features)

            # Dimensionality reduction (PCA)
            target_dim = min(5, scaled_features.shape[0] - 1, scaled_features.shape[1])

            if self.pca is None:
                self.pca = PCA(n_components=target_dim)
                reduced_features = self.pca.fit_transform(scaled_features)
            else:
                try:
                    reduced_features = self.pca.transform(scaled_features)
                except Exception as e:
                    # Refit PCA
                    self.pca = PCA(n_components=target_dim)
                    reduced_features = self.pca.fit_transform(scaled_features)

            # Reduced version of current features
            current_reduced = reduced_features[-1]

            # k-means clustering
            if self.kmeans is None or len(self.measurement_history) % 50 == 0:
                # Retrain on first time or every X times
                self.kmeans = KMeans(n_clusters=min(self.n_energy_clusters, len(reduced_features)),
                                   random_state=42)
                self.kmeans.fit(reduced_features)

            try:
                # Predict nearest cluster
                cluster_idx = self.kmeans.predict(current_reduced.reshape(1, -1))[0]
                cluster_center = self.kmeans.cluster_centers_[cluster_idx]

                # Energy is defined as negative of feature vector norm
                energy = -np.linalg.norm(cluster_center)

                # Normalize energy range
                energy = energy * self.n_qubits / 2.0

            except Exception as e:
                _logger.warning(f"Clustering error: {e}")
                # Fallback
                energy = -self.n_qubits * 0.5

        else:
            # Initial estimation when data is insufficient
            # Quantum entropy based
            entropy = self._compute_quantum_entropy(quantum_features)
            energy = -self.n_qubits * (1 - entropy)

        return energy

    def _compute_quantum_entropy(self, features: np.ndarray) -> float:
        """Calculate quantum entropy"""
        # Build probability distribution from feature vector
        probs = np.abs(features)**2
        probs = probs / (np.sum(probs) + 1e-10)

        # Approximation of von Neumann entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        normalized_entropy = entropy / np.log(len(probs))

        return normalized_entropy

    def _compute_shannon_entropy(self, probabilities: Dict[str, float]) -> float:
        """Calculate Shannon entropy"""
        entropy = 0.0
        for prob in probabilities.values():
            if prob > 0:
                entropy -= prob * np.log2(prob)
        return entropy

    def update_learning(self, template, measurement_results: np.ndarray):
        """Update learning with measurement results (corrected version)"""
        # Adjust measurement result size
        if len(measurement_results) < 2**self.n_qubits:
            # Padding
            padded_results = np.zeros(2**self.n_qubits)
            padded_results[:len(measurement_results)] = measurement_results
            measurement_results = padded_results

        self.measurement_history.append(measurement_results)

        # Feature updates are automatically done in estimate_energy_unsupervised

        # Limit history size
        max_history = 10000
        if len(self.measurement_history) > max_history:
            self.measurement_history = self.measurement_history[-max_history:]
            self.circuit_features = self.circuit_features[-max_history:]


#================================================
# BayesianCircuitOptimizer
#================================================
class MultiObjectiveBayesianCircuitOptimizer:
    """Multi-objective Bayesian optimization for quantum circuit search (scientifically grounded)"""

    def __init__(self, n_qubits, device='cuda', n_objectives=9, use_parallel=False,
                 energy_estimator=None):  # Add energy estimator
        self.n_qubits = n_qubits
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.n_objectives = n_objectives  # Increased from 8 to 9
        self.observations_X = []
        self.observations_Y = []
        self.models = None
        self.ref_point = None
        self.use_parallel = use_parallel
        self.energy_estimator = energy_estimator  # Unsupervised energy estimator

        # Statistics for Min-Max scaling
        self.X_min = None
        self.X_max = None
        self.Y_min = None
        self.Y_max = None

        # Energy estimation history
        self.energy_estimation_history = []
        self.energy_prediction_accuracy = []

        # Ideal and worst values for each objective function (for normalization)
        self.objective_bounds = {
            'hardware_efficiency': (0.0, 1.0),
            'noise_resilience': (0.0, 1.0),
            'expressivity': (0.0, 1.0),
            'mitigation_compatibility': (0.0, 1.0),
            'trainability': (0.0, 1.0),
            'entanglement_capability': (0.0, 1.0),
            'circuit_depth_efficiency': (0.0, 1.0),
            'parameter_efficiency': (0.0, 1.0),
            'energy_estimation_quality': (0.0, 1.0)  # Newly added
        }

    def evaluate_circuit_multi_objective(self, template, training_data=None):
        """Multi-objective circuit evaluation (including energy estimation quality)"""
        objectives = []

        # Existing 8 objective functions

        # 1-8. Existing objective functions (listed without omission)
        hw_efficiency = self._compute_hardware_efficiency_scientific(template)
        objectives.append(max(0.0, min(1.0, hw_efficiency)))

        noise_resilience = self._compute_noise_resilience_scientific(template)
        objectives.append(max(0.0, min(1.0, noise_resilience)))

        expressivity = self._compute_expressivity_scientific(template)
        objectives.append(max(0.0, min(1.0, expressivity)))

        mitigation = self._compute_mitigation_compatibility_scientific(template)
        objectives.append(max(0.0, min(1.0, mitigation)))

        trainability = self._compute_trainability_scientific(template)
        objectives.append(max(0.0, min(1.0, trainability)))

        entanglement = self._compute_entanglement_capability_scientific(template)
        objectives.append(max(0.0, min(1.0, entanglement)))

        depth_efficiency = self._compute_depth_efficiency_scientific(template)
        objectives.append(max(0.0, min(1.0, depth_efficiency)))

        param_efficiency = self._compute_parameter_efficiency_scientific(template)
        objectives.append(max(0.0, min(1.0, param_efficiency)))

        # 9. Energy estimation quality (newly added)
        energy_quality = self._compute_energy_estimation_quality_scientific(template, training_data)
        objectives.append(max(0.0, min(1.0, energy_quality)))

        return torch.tensor(objectives, dtype=torch.float64).to(self.device)

    def _compute_hardware_efficiency_scientific(self, template):
        """Scientific calculation of hardware efficiency

        Reference: Kandala et al. "Hardware-efficient variational quantum eigensolver" Nature 549, 242-246 (2017)
        """
        # Gate times based on IBMQ device measurements (ns)
        gate_times = {
            'RX': 35.56, 'RY': 35.56, 'RZ': 0.0,  # Virtual Z
            'H': 35.56, 'S': 35.56, 'T': 35.56,
            'CNOT': 300.8, 'CZ': 300.8, 'SWAP': 902.4  # 3 CNOTs
        }

        # Gate error rates (IBM Quantum measurements)
        gate_errors = {
            'RX': 2.16e-4, 'RY': 2.16e-4, 'RZ': 0.0,
            'H': 2.16e-4, 'S': 2.16e-4, 'T': 2.16e-4,
            'CNOT': 9.11e-3, 'CZ': 9.11e-3, 'SWAP': 2.73e-2
        }

        total_time = 0.0
        total_error_prob = 0.0
        connectivity_overhead = 1.0

        # Connectivity map (linear topology)
        connectivity = self._get_hardware_connectivity(template.n_qubits)

        for gate_info in template.gate_sequence:
            gate_type = gate_info['gate']
            qubits = gate_info['qubits']

            # Gate time
            total_time += gate_times.get(gate_type, 50.0)

            # Accumulate error probability
            gate_error = gate_errors.get(gate_type, 5e-3)
            total_error_prob = 1 - (1 - total_error_prob) * (1 - gate_error)

            # Check connectivity for two-qubit gates
            if len(qubits) == 2:
                q1, q2 = qubits[0], qubits[1]
                if q1 < template.n_qubits and q2 < template.n_qubits:
                    if not self._are_connected(q1, q2, connectivity):
                        # Calculate required SWAP count
                        swap_count = self._compute_swap_count(q1, q2, connectivity)
                        connectivity_overhead *= (1 + swap_count * 0.1)
                        total_time += swap_count * gate_times['SWAP']
                        swap_error = gate_errors['SWAP']
                        for _ in range(swap_count):
                            total_error_prob = 1 - (1 - total_error_prob) * (1 - swap_error)

        # Parallelization capability
        parallelization = self._evaluate_parallelization_scientific(template)

        # Overall score (time efficiency x error rate x connectivity x parallelization)
        time_efficiency = np.exp(-total_time / 5000.0)  # 5us reference
        error_efficiency = (1 - total_error_prob) ** 2
        connectivity_efficiency = 1.0 / connectivity_overhead

        score = (0.3 * time_efficiency +
                0.3 * error_efficiency +
                0.2 * connectivity_efficiency +
                0.2 * parallelization)

        return max(0.0, min(1.0, score))

    def _compute_noise_resilience_scientific(self, template):
        """Scientific calculation of noise resilience

        Reference:
        - Temme et al. "Error mitigation for short-depth quantum circuits" PRL 119, 180509 (2017)
        - Endo et al. "Practical Quantum Error Mitigation" PRX Quantum 2, 040337 (2021)
        """
        # Decoherence times (based on measurements)
        T1 = 100e3  # 100 us
        T2 = 150e3  # 150 us

        # Average Gate Infidelity (AGI) calculation
        total_agi = 0.0
        total_time = 0.0

        gate_times = {
            'RX': 35.56, 'RY': 35.56, 'RZ': 0.0,
            'H': 35.56, 'CNOT': 300.8, 'CZ': 300.8
        }

        gate_infidelities = {
            'RX': 1.08e-4, 'RY': 1.08e-4, 'RZ': 0.0,
            'H': 1.08e-4, 'CNOT': 4.56e-3, 'CZ': 4.56e-3
        }

        for gate_info in template.gate_sequence:
            gate_type = gate_info['gate']
            gate_time = gate_times.get(gate_type, 50.0)
            gate_infidelity = gate_infidelities.get(gate_type, 1e-3)

            total_time += gate_time
            total_agi += gate_infidelity

        # Effective circuit fidelity
        circuit_fidelity = np.exp(-total_agi)

        # Decoherence effects
        coherence_factor = np.exp(-total_time / T2) * np.sqrt(np.exp(-total_time / T1))

        # Quasi-probability decomposition overhead (error mitigation)
        qpd_overhead = 1.0
        if self._supports_quasi_probability_decomposition(template):
            # Based on Clifford gates ratio
            clifford_ratio = self._compute_clifford_ratio(template)
            qpd_overhead = 1.0 + 2.0 * (1.0 - clifford_ratio)

        # Overall noise resilience score
        noise_resilience = circuit_fidelity * coherence_factor / qpd_overhead

        return max(0.0, min(1.0, noise_resilience))

    def _compute_expressivity_scientific(self, template):
        """Scientific calculation of expressivity

        Reference: Sim et al. "Expressibility and entangling capability of parameterized quantum circuits"
        Advanced Quantum Technologies 2, 1900070 (2019)
        """
        n_qubits = template.n_qubits

        # 1. Expressibility (2-design approximation)
        # Estimate distance from random unitary
        n_params = len(template.parameter_map)
        param_density = n_params / (template.n_qubits * len(template.gate_sequence))

        # 2. Entangling capability
        entangling_gates = ['CNOT', 'CZ', 'SWAP']
        n_entangling = sum(1 for g in template.gate_sequence if g['gate'] in entangling_gates)

        # Meyer-Wallach entanglement measure estimation
        if n_entangling == 0:
            meyer_wallach = 0.0
        else:
            # Simplified estimation (actual requires state sampling)
            connectivity_graph = self._build_connectivity_graph(template)
            meyer_wallach = self._estimate_meyer_wallach(connectivity_graph, n_qubits)

        # 3. Circuit depth vs width balance
        depth = self._calculate_circuit_depth_scientific(template)
        width = n_qubits
        depth_width_balance = 1.0 - abs(depth / width - 2.0) / 10.0  # Ideal ratio 2:1

        # Overall expressivity score
        expressivity = (0.4 * param_density +
                       0.4 * meyer_wallach +
                       0.2 * max(0, depth_width_balance))

        return max(0.0, min(1.0, expressivity))

    def _compute_mitigation_compatibility_scientific(self, template):
        """Scientific calculation of error mitigation compatibility

        Reference: Endo et al. "Hybrid quantum-classical algorithms and quantum error mitigation"
        J. Phys. Soc. Jpn. 90, 032001 (2021)
        """
        # 1. Zero-noise extrapolation (ZNE) compatibility
        # Ratio of pulse-stretchable gates
        stretchable_gates = ['RX', 'RY', 'RZ', 'CNOT']
        n_stretchable = sum(1 for g in template.gate_sequence
                           if g['gate'] in stretchable_gates)
        zne_compatibility = n_stretchable / len(template.gate_sequence) if template.gate_sequence else 0

        # 2. Probabilistic error cancellation (PEC) compatibility
        # Pauli twirling capable structure
        pauli_twirling_score = self._compute_pauli_twirling_compatibility(template)

        # 3. Clifford data regression (CDR) compatibility
        clifford_ratio = self._compute_clifford_ratio(template)

        # 4. Virtual distillation compatibility
        # Circuit structure with symmetry
        symmetry_score = self._compute_circuit_symmetry(template)

        # Overall score
        mitigation_score = (0.3 * zne_compatibility +
                          0.3 * pauli_twirling_score +
                          0.2 * clifford_ratio +
                          0.2 * symmetry_score)

        return max(0.0, min(1.0, mitigation_score))

    def _compute_trainability_scientific(self, template):
        """Scientific calculation of trainability (Barren plateau avoidance)

        Reference: McClean et al. "Barren plateaus in quantum neural network training landscapes"
        Nature Communications 9, 4812 (2018)
        """
        n_qubits = template.n_qubits
        n_params = len(template.parameter_map)

        # 1. Parameter concentration
        # Theoretically Var[dC/dtheta] ~ 1/2^n for global cost functions
        if n_params == 0:
            param_concentration = 0.0
        else:
            # Assume local cost function
            locality = self._estimate_cost_function_locality(template)
            param_concentration = 1.0 / (2 ** (n_qubits * (1 - locality)))

        # 2. Initialization strategy score
        # Hardware efficient ansatz requires good initialization
        init_strategy_score = self._evaluate_initialization_strategy(template)

        # 3. Gradient flow
        # Ease of gradient flow through circuit layer structure
        gradient_flow = self._evaluate_gradient_flow(template)

        # 4. Entanglement growth
        # Gradual entanglement growth improves trainability
        entanglement_growth = self._evaluate_entanglement_growth(template)

        # Overall trainability score
        trainability = (0.3 * np.log10(param_concentration + 1e-10) / 10 + 1.0 +  # Normalized
                       0.3 * init_strategy_score +
                       0.2 * gradient_flow +
                       0.2 * entanglement_growth)

        return max(0.0, min(1.0, trainability))

    def _compute_entanglement_capability_scientific(self, template):
        """Scientific calculation of entangling capability (Meyer-Wallach measure)

        Reference: Meyer & Wallach "Global entanglement in multiparticle systems"
        J. Math. Phys. 43, 4273 (2002)
        """
        n_qubits = template.n_qubits

        # Build entangling gate graph structure
        import networkx as nx
        entanglement_graph = nx.Graph()
        entanglement_graph.add_nodes_from(range(n_qubits))

        entangling_gates = ['CNOT', 'CZ', 'SWAP', 'iSWAP', 'CRX', 'CRY', 'CRZ']

        for gate_info in template.gate_sequence:
            if gate_info['gate'] in entangling_gates and len(gate_info['qubits']) >= 2:
                q1, q2 = gate_info['qubits'][0], gate_info['qubits'][1]
                if q1 < n_qubits and q2 < n_qubits:
                    entanglement_graph.add_edge(q1, q2)

        # Graph theoretical metrics
        if entanglement_graph.number_of_edges() == 0:
            return 0.0

        # 1. Algebraic connectivity (Fiedler value)
        laplacian = nx.laplacian_matrix(entanglement_graph).todense()
        eigenvalues = np.linalg.eigvalsh(laplacian)
        algebraic_connectivity = eigenvalues[1] if len(eigenvalues) > 1 else 0

        # 2. Average clustering coefficient
        clustering = nx.average_clustering(entanglement_graph)

        # 3. Graph diameter (normalized)
        if nx.is_connected(entanglement_graph):
            diameter = nx.diameter(entanglement_graph)
            normalized_diameter = 1.0 - diameter / (n_qubits - 1)
        else:
            normalized_diameter = 0.0

        # 4. Edge density
        max_edges = n_qubits * (n_qubits - 1) // 2
        edge_density = entanglement_graph.number_of_edges() / max_edges

        # Meyer-Wallach measure approximation
        meyer_wallach_approx = (0.3 * algebraic_connectivity / n_qubits +
                               0.2 * clustering +
                               0.3 * normalized_diameter +
                               0.2 * edge_density)

        return max(0.0, min(1.0, meyer_wallach_approx))

    def _compute_depth_efficiency_scientific(self, template):
        """Scientific calculation of circuit depth efficiency"""
        depth = self._calculate_circuit_depth_scientific(template)
        n_qubits = template.n_qubits

        # Theoretical minimum depth (problem-dependent, but general heuristic)
        min_theoretical_depth = np.ceil(np.log2(n_qubits))

        # Practical depth limit (NISQ era)
        nisq_depth_limit = 100

        # Depth efficiency
        if depth <= min_theoretical_depth:
            efficiency = 1.0
        elif depth <= nisq_depth_limit:
            efficiency = 1.0 - (depth - min_theoretical_depth) / (nisq_depth_limit - min_theoretical_depth)
        else:
            efficiency = 0.1 * nisq_depth_limit / depth

        return max(0.0, min(1.0, efficiency))

    def _compute_parameter_efficiency_scientific(self, template):
        """Scientific calculation of parameter efficiency"""
        n_params = len(template.parameter_map)
        n_qubits = template.n_qubits

        # Theoretically required parameters (full quantum state representation)
        full_params = 2 ** (2 * n_qubits) - 1

        # Practical parameter count (polynomial scaling)
        practical_params = n_qubits ** 2

        if n_params == 0:
            return 0.0
        elif n_params <= practical_params:
            efficiency = n_params / practical_params
        else:
            # Over-parameterization penalty
            efficiency = practical_params / n_params * 0.8

        return max(0.0, min(1.0, efficiency))

    def _compute_energy_estimation_quality_scientific(self, template, training_data=None):
        """Scientific calculation of energy estimation quality

        References:
        - Cerezo et al. "Cost function dependent barren plateaus in shallow parametrized quantum circuits"
          Nature Communications 12, 1791 (2021)
        - Larocca et al. "Diagnosing Barren Plateaus with Tools from Quantum Optimal Control"
          Quantum 6, 824 (2022)
        """

        if self.energy_estimator is None:
            return 0.5  # Default value

        try:
            # 1. Evaluate energy landscape smoothness
            landscape_smoothness = self._evaluate_energy_landscape_smoothness(template)

            # 2. Energy estimation convergence
            convergence_score = self._evaluate_energy_convergence(template, training_data)

            # 3. Energy estimation stability under noise
            noise_stability = self._evaluate_energy_noise_stability(template)

            # 4. Information theoretical quality of energy estimation
            information_quality = self._evaluate_energy_information_quality(template)

            # 5. Estimation accuracy based on quantum Fisher information
            fisher_score = self._compute_quantum_fisher_information_score(template)

            # Overall score (weighted average)
            energy_quality = (
                0.25 * landscape_smoothness +
                0.20 * convergence_score +
                0.20 * noise_stability +
                0.20 * information_quality +
                0.15 * fisher_score
            )

            # Record in history
            self.energy_estimation_history.append({
                'template': template,
                'quality_score': energy_quality,
                'components': {
                    'landscape_smoothness': landscape_smoothness,
                    'convergence': convergence_score,
                    'noise_stability': noise_stability,
                    'information_quality': information_quality,
                    'fisher_score': fisher_score
                }
            })

            return energy_quality

        except Exception as e:
            _logger.error(f"Energy estimation quality calculation error: {e}")
            return 0.5

    def _evaluate_energy_landscape_smoothness(self, template):
        """Evaluate energy landscape smoothness

        Reference: Li et al. "Quantum optimization with a novel Gibbs objective function
        and ansatz architecture search" Phys. Rev. Research 2, 023074 (2020)
        """
        n_samples = min(20, len(template.parameter_map))
        if n_samples < 2:
            return 0.5

        try:
            # Sample energy at random parameter points
            energies = []
            params_list = []

            for _ in range(n_samples):
                params = np.random.uniform(-np.pi, np.pi, size=len(template.parameter_map))
                params_list.append(params)

                if self.energy_estimator:
                    # Estimate with dummy input data
                    input_data = np.random.randn(2**template.n_qubits)
                    input_data = input_data / (np.linalg.norm(input_data) + 1e-10)
                    energy = self.energy_estimator.estimate_energy_unsupervised(template, input_data)
                    energies.append(energy)

            if len(energies) < 2:
                return 0.5

            # Calculate variance of energy gradient
            gradients = []
            for i in range(1, len(energies)):
                param_diff = np.linalg.norm(params_list[i] - params_list[i-1])
                if param_diff > 1e-6:
                    gradient = abs(energies[i] - energies[i-1]) / param_diff
                    gradients.append(gradient)

            if gradients:
                # Smaller gradient variance means smoother
                gradient_var = np.var(gradients)
                smoothness = 1.0 / (1.0 + gradient_var)
                return min(1.0, smoothness)

            return 0.5

        except Exception:
            return 0.5

    def _evaluate_energy_convergence(self, template, training_data=None):
        """Evaluate energy estimation convergence

        Reference: Kuebler et al. "The inductive bias of quantum kernels"
        NeurIPS 2021
        """
        if not hasattr(self.energy_estimator, 'measurement_history') or \
           len(self.energy_estimator.measurement_history) < 10:
            return 0.5

        try:
            # Evaluate convergence of recent estimates
            recent_measurements = self.energy_estimator.measurement_history[-20:]

            # Calculate moving averages
            window_size = 5
            moving_averages = []

            for i in range(len(recent_measurements) - window_size + 1):
                window = recent_measurements[i:i+window_size]
                avg = np.mean([np.mean(m) for m in window])
                moving_averages.append(avg)

            if len(moving_averages) < 2:
                return 0.5

            # Calculate convergence (rate of change in moving averages)
            changes = []
            for i in range(1, len(moving_averages)):
                change = abs(moving_averages[i] - moving_averages[i-1]) / (abs(moving_averages[i-1]) + 1e-6)
                changes.append(change)

            # Smaller rate of change means better convergence
            avg_change = np.mean(changes)
            convergence = 1.0 - min(1.0, avg_change)

            return convergence

        except Exception:
            return 0.5

    def _evaluate_energy_noise_stability(self, template):
        """Evaluate energy estimation stability under noise

        Reference: Wang et al. "Noise-induced barren plateaus in variational quantum algorithms"
        Nature Communications 12, 6961 (2021)
        """
        if self.energy_estimator is None or not hasattr(self.energy_estimator, 'use_noise'):
            return 0.5

        try:
            n_trials = 10
            energies_no_noise = []
            energies_with_noise = []

            # Dummy input
            input_data = np.random.randn(2**template.n_qubits)
            input_data = input_data / (np.linalg.norm(input_data) + 1e-10)

            # Estimation without noise
            original_noise_state = self.energy_estimator.use_noise
            self.energy_estimator.use_noise = False

            for _ in range(n_trials):
                energy = self.energy_estimator.estimate_energy_unsupervised(template, input_data)
                energies_no_noise.append(energy)

            # Estimation with noise
            self.energy_estimator.use_noise = True

            for _ in range(n_trials):
                energy = self.energy_estimator.estimate_energy_unsupervised(template, input_data)
                energies_with_noise.append(energy)

            # Restore original state
            self.energy_estimator.use_noise = original_noise_state

            # Compare variances
            var_no_noise = np.var(energies_no_noise)
            var_with_noise = np.var(energies_with_noise)

            # Smaller increase in variance due to noise means more stable
            if var_no_noise > 1e-6:
                stability = 1.0 - min(1.0, (var_with_noise - var_no_noise) / var_no_noise)
            else:
                stability = 0.8  # When variance is very small

            return max(0.0, stability)

        except Exception:
            return 0.5

    def _evaluate_energy_information_quality(self, template):
        """Evaluate information theoretical quality of energy estimation

        Reference: Meyer et al. "Fisher Information in Noisy Intermediate-Scale Quantum Applications"
        Quantum 5, 539 (2021)
        """
        try:
            # Evaluate correlation between entangling capability and energy estimation accuracy
            entangling_score = self._compute_entanglement_capability_scientific(template)

            # Balance between parameter count and circuit depth
            n_params = len(template.parameter_map)
            depth = self._calculate_circuit_depth_scientific(template)

            if depth > 0:
                param_depth_ratio = n_params / depth
                # Ideal ratio is around 1-2
                balance_score = 1.0 - abs(param_depth_ratio - 1.5) / 5.0
                balance_score = max(0.0, min(1.0, balance_score))
            else:
                balance_score = 0.0

            # Measurement basis diversity
            measurement_diversity = self._evaluate_measurement_basis_diversity(template)

            # Information theoretical quality
            info_quality = (
                0.4 * entangling_score +
                0.3 * balance_score +
                0.3 * measurement_diversity
            )

            return info_quality

        except Exception:
            return 0.5

    def _compute_quantum_fisher_information_score(self, template):
        """Calculate score based on quantum Fisher information

        Reference: Liu et al. "Variational quantum eigensolver with fewer qubits"
        Phys. Rev. Research 1, 023025 (2019)
        """
        try:
            # Simple estimation of quantum Fisher information
            n_params = len(template.parameter_map)
            n_qubits = template.n_qubits

            # Number of entangling gates
            entangling_gates = ['CNOT', 'CZ', 'SWAP', 'iSWAP']
            n_entangling = sum(1 for g in template.gate_sequence if g['gate'] in entangling_gates)

            # Ratio to theoretical upper bound of Fisher information
            theoretical_max = 4 * n_qubits * n_params

            # Estimate effective Fisher information
            effective_fisher = 0.0

            # Contribution from parameterized gates
            param_gates = [g for g in template.gate_sequence if g.get('trainable', False)]
            for gate in param_gates:
                if len(gate['qubits']) == 1:
                    effective_fisher += 2.0  # Single-qubit gate
                else:
                    effective_fisher += 4.0  # Two-qubit gate

            # Amplification by entangling
            entanglement_factor = 1.0 + 0.5 * (n_entangling / max(1, len(template.gate_sequence)))
            effective_fisher *= entanglement_factor

            # Normalization
            if theoretical_max > 0:
                fisher_score = min(1.0, effective_fisher / theoretical_max)
            else:
                fisher_score = 0.0

            return fisher_score

        except Exception:
            return 0.5

    def _evaluate_measurement_basis_diversity(self, template):
        """Evaluate measurement basis diversity"""
        try:
            # Analyze last gate layer of circuit
            last_layer_gates = []
            qubit_last_gate = {}

            # Identify last gate for each qubit
            for gate in template.gate_sequence:
                for qubit in gate['qubits']:
                    qubit_last_gate[qubit] = gate

            # Evaluate measurement basis diversity
            basis_types = set()
            for qubit, gate in qubit_last_gate.items():
                if gate['gate'] in ['RX', 'H']:
                    basis_types.add('X')
                elif gate['gate'] in ['RY']:
                    basis_types.add('Y')
                else:
                    basis_types.add('Z')

            # Diversity score
            diversity = len(basis_types) / 3.0  # Maximum 3 basis types

            return diversity

        except Exception:
            return 0.33  # Default to 1 basis type

    def update_observations(self, circuit_features, objectives):
        """Update observation data (unified device version)"""
        # Unify devices
        if isinstance(circuit_features, torch.Tensor):
            circuit_features = circuit_features.to(self.device)
        if isinstance(objectives, torch.Tensor):
            objectives = objectives.to(self.device)

        self.observations_X.append(circuit_features)
        self.observations_Y.append(objectives)

        # Update GP models
        if len(self.observations_X) >= 5:
            self._update_gp_models()

    def _update_gp_models(self):
        """Update Gaussian process models (BoTorch version with built-in transforms & constant-target guard)."""
        X = torch.stack(self.observations_X).to(self.device, dtype=torch.float64)
        Y = torch.stack(self.observations_Y).to(self.device, dtype=torch.float64)

        models = []
        d = X.shape[-1]
        eps_const = 1e-10       # Threshold to detect near-constant targets
        jitter_std = 1e-6      # Tiny jitter to avoid exactly zero variance

        for i in range(self.n_objectives):
            y_i = Y[:, i:i+1]

            # --- Guard for constant/near-constant targets -----------------------
            if torch.nan_to_num(y_i.std()).item() < eps_const:
                y_i = y_i + jitter_std * torch.randn_like(y_i)

            # --- Build a GP with transforms -------------------------------------
            gp = SingleTaskGP(
                X, y_i,
                input_transform=Normalize(d=d),
                outcome_transform=Standardize(m=1),
            ).to(self.device, dtype=torch.float64)

            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            models.append(gp)

        if models:
            self.models = ModelListGP(*models)
            self.ref_point = torch.full((self.n_objectives,), -0.1,
                                        device=self.device, dtype=torch.float64)
        else:
            _logger.warning("Failed to create GP models")
            self.models = None

    def _manual_fit_gp(self, gp, mll, X, y):
        """Manually fit GP model"""
        import torch.optim as optim

        gp.train()
        gp.likelihood.train()

        optimizer = optim.LBFGS(
            gp.parameters(),
            lr=0.1,
            max_iter=20,
            line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            output = gp(X)
            loss = -mll(output, y)
            loss.backward()
            return loss

        for _ in range(10):
            loss = optimizer.step(closure)
            if loss.item() < 1e-6:
                break

        gp.eval()
        gp.likelihood.eval()

    def _encode_circuit_features_detailed_cpu(self, template):
        """Detailed circuit feature encoding (CPU version)"""
        features = []

        # Basic statistics
        features.extend([
            float(len(template.gate_sequence)),
            float(self._calculate_circuit_depth_scientific(template)),
            float(len(template.parameter_map)),
            float(template.n_qubits)
        ])

        # Gate type distribution (normalized)
        gate_types = ['RX', 'RY', 'RZ', 'H', 'S', 'T', 'CNOT', 'CZ', 'SWAP']
        gate_counts = {gt: 0 for gt in gate_types}

        for gate in template.gate_sequence:
            if gate['gate'] in gate_counts:
                gate_counts[gate['gate']] += 1

        total_gates = sum(gate_counts.values())
        for gate_type in gate_types:
            features.append(gate_counts[gate_type] / (total_gates + 1))

        # Entangling structure
        entanglement_features = self._compute_entanglement_features(template)
        features.extend(entanglement_features)

        # Parameter placement patterns
        param_features = self._compute_parameter_features(template)
        features.extend(param_features)

        # Symmetry features
        symmetry_features = self._compute_symmetry_features(template)
        features.extend(symmetry_features)

        # Return as list (will be converted to tensor later)
        return features

    def _encode_circuit_features_detailed(self, template):
        """Detailed circuit feature encoding (unified device version)"""
        # Call CPU version then convert to tensor
        features_list = self._encode_circuit_features_detailed_cpu(template)
        return torch.tensor(features_list, dtype=torch.float64).to(self.device)

    def _compute_pareto_front_indices(self, Y):
        """Calculate Pareto front indices"""
        n = Y.shape[0]
        is_dominated = torch.zeros(n, dtype=torch.bool)

        for i in range(n):
            for j in range(n):
                if i != j:
                    # Check if j dominates i (assuming minimization)
                    if torch.all(Y[j] <= Y[i]) and torch.any(Y[j] < Y[i]):
                        is_dominated[i] = True
                        break

        return ~is_dominated

    # Helper method group
    def _get_hardware_connectivity(self, n_qubits):
        """Get hardware connectivity map"""
        # Linear topology
        connectivity = {}
        for i in range(n_qubits):
            connectivity[i] = []
            if i > 0:
                connectivity[i].append(i-1)
            if i < n_qubits - 1:
                connectivity[i].append(i+1)
        return connectivity

    def _are_connected(self, q1, q2, connectivity):
        """Check if two qubits are directly connected"""
        return q2 in connectivity.get(q1, [])

    def _compute_swap_count(self, q1, q2, connectivity):
        """Calculate required SWAP gate count (shortest path)"""
        from collections import deque

        visited = {q1}
        queue = deque([(q1, 0)])

        while queue:
            current, dist = queue.popleft()
            if current == q2:
                return dist - 1  # SWAP count is distance-1

            for neighbor in connectivity.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return float('inf')  # Unreachable

    def _evaluate_parallelization_scientific(self, template):
        """Scientific evaluation of parallelization capability"""
        # Number of parallel executable gates in each time slot
        time_slots = self._compute_time_slots(template)

        if not time_slots:
            return 0.0

        # Calculate parallelism
        total_gates = len(template.gate_sequence)
        actual_depth = len(time_slots)
        max_parallelism = max(len(slot) for slot in time_slots)
        avg_parallelism = total_gates / actual_depth

        # Theoretical maximum parallelism
        theoretical_max_parallelism = template.n_qubits // 2  # 2-qubit gates

        # Parallelization efficiency
        parallelization_score = (0.5 * avg_parallelism / theoretical_max_parallelism +
                               0.5 * max_parallelism / theoretical_max_parallelism)

        return max(0.0, min(1.0, parallelization_score))

    def _compute_time_slots(self, template):
        """Split gates into time slots"""
        time_slots = []
        qubit_busy_until = {}

        for gate_info in template.gate_sequence:
            qubits = gate_info['qubits']

            # Earliest executable time
            earliest_time = 0
            for q in qubits:
                if q in qubit_busy_until:
                    earliest_time = max(earliest_time, qubit_busy_until[q] + 1)

            # Add to time slot
            while len(time_slots) <= earliest_time:
                time_slots.append([])

            time_slots[earliest_time].append(gate_info)

            # Update qubit usage time
            for q in qubits:
                qubit_busy_until[q] = earliest_time

        return time_slots

    def _supports_quasi_probability_decomposition(self, template):
        """Check quasi-probability decomposition support"""
        # Check if structure is easily decomposable to Pauli operators
        pauli_decomposable_gates = ['RX', 'RY', 'RZ', 'H', 'CNOT', 'CZ']

        decomposable_count = sum(1 for g in template.gate_sequence
                               if g['gate'] in pauli_decomposable_gates)

        return decomposable_count / len(template.gate_sequence) > 0.8

    def _compute_clifford_ratio(self, template):
        """Calculate Clifford gate ratio"""
        clifford_gates = ['H', 'S', 'CNOT', 'CZ']
        clifford_count = sum(1 for g in template.gate_sequence
                           if g['gate'] in clifford_gates)

        return clifford_count / len(template.gate_sequence) if template.gate_sequence else 0

    def _build_connectivity_graph(self, template):
        """Build circuit connectivity graph"""
        import networkx as nx

        graph = nx.Graph()
        graph.add_nodes_from(range(template.n_qubits))

        for gate_info in template.gate_sequence:
            if len(gate_info['qubits']) >= 2:
                q1, q2 = gate_info['qubits'][0], gate_info['qubits'][1]
                if q1 < template.n_qubits and q2 < template.n_qubits:
                    graph.add_edge(q1, q2)

        return graph

    def _estimate_meyer_wallach(self, connectivity_graph, n_qubits):
        """Estimate Meyer-Wallach entanglement measure"""
        import networkx as nx

        if connectivity_graph.number_of_edges() == 0:
            return 0.0

        # Algebraic connectivity of graph (Fiedler value)
        laplacian = nx.laplacian_matrix(connectivity_graph).todense()
        eigenvalues = np.linalg.eigvalsh(laplacian)
        algebraic_connectivity = eigenvalues[1] if len(eigenvalues) > 1 else 0

        # Normalization
        return min(1.0, algebraic_connectivity / n_qubits)

    def _calculate_circuit_depth_scientific(self, template):
        """Scientific calculation of circuit depth"""
        if not template.gate_sequence:
            return 0

        qubit_layers = {}
        max_depth = 0

        for gate_info in template.gate_sequence:
            qubits = gate_info['qubits']

            # Determine current layer
            current_layer = 0
            for q in qubits:
                if q < template.n_qubits and q in qubit_layers:
                    current_layer = max(current_layer, qubit_layers[q] + 1)

            # Update
            for q in qubits:
                if q < template.n_qubits:
                    qubit_layers[q] = current_layer

            max_depth = max(max_depth, current_layer + 1)

        return max_depth

    def _compute_pauli_twirling_compatibility(self, template):
        """Calculate Pauli twirling compatibility"""
        # Check existence of single-qubit gates before and after CNOT gates
        twirling_compatible = 0
        total_cnots = 0

        for i, gate_info in enumerate(template.gate_sequence):
            if gate_info['gate'] in ['CNOT', 'CZ']:
                total_cnots += 1

                # Check single-qubit gates before and after
                has_pre_gate = False
                has_post_gate = False

                if i > 0:
                    prev_gate = template.gate_sequence[i-1]
                    if len(prev_gate['qubits']) == 1:
                        has_pre_gate = True

                if i < len(template.gate_sequence) - 1:
                    next_gate = template.gate_sequence[i+1]
                    if len(next_gate['qubits']) == 1:
                        has_post_gate = True

                if has_pre_gate and has_post_gate:
                    twirling_compatible += 1

        return twirling_compatible / total_cnots if total_cnots > 0 else 0

    def _compute_circuit_symmetry(self, template):
        """Calculate circuit symmetry score"""
        # Layer structure symmetry
        layers = self._decompose_into_layers(template)

        if len(layers) < 2:
            return 0.0

        # Variance of gate counts in each layer
        layer_sizes = [len(layer) for layer in layers]
        size_variance = np.var(layer_sizes)
        size_symmetry = 1.0 / (1.0 + size_variance)

        # Gate type symmetry
        gate_type_symmetry = self._compute_gate_type_symmetry(layers)

        return 0.5 * size_symmetry + 0.5 * gate_type_symmetry

    def _decompose_into_layers(self, template):
        """Decompose circuit into layers"""
        layers = []
        current_layer = []
        used_qubits = set()

        for gate in template.gate_sequence:
            gate_qubits = set(gate['qubits'])

            if gate_qubits & used_qubits:
                if current_layer:
                    layers.append(current_layer)
                current_layer = [gate]
                used_qubits = gate_qubits
            else:
                current_layer.append(gate)
                used_qubits |= gate_qubits

        if current_layer:
            layers.append(current_layer)

        return layers

    def _compute_gate_type_symmetry(self, layers):
        """Calculate gate type symmetry between layers"""
        if len(layers) < 2:
            return 0.0

        # Gate type distribution for each layer
        layer_distributions = []

        for layer in layers:
            distribution = {}
            for gate in layer:
                gate_type = gate['gate']
                distribution[gate_type] = distribution.get(gate_type, 0) + 1
            layer_distributions.append(distribution)

        # Evaluate symmetry with KL divergence
        symmetry_scores = []

        for i in range(len(layers) - 1):
            kl_div = self._kl_divergence_gates(layer_distributions[i],
                                              layer_distributions[i+1])
            symmetry_scores.append(np.exp(-kl_div))

        return np.mean(symmetry_scores)

    def _kl_divergence_gates(self, dist1, dist2):
        """KL divergence between two gate distributions"""
        all_gates = set(dist1.keys()) | set(dist2.keys())

        total1 = sum(dist1.values())
        total2 = sum(dist2.values())

        if total1 == 0 or total2 == 0:
            return float('inf')

        kl_div = 0.0
        for gate in all_gates:
            p1 = dist1.get(gate, 0) / total1
            p2 = dist2.get(gate, 0) / total2

            if p1 > 0 and p2 > 0:
                kl_div += p1 * np.log(p1 / p2)

        return kl_div

    def _estimate_cost_function_locality(self, template):
        """Estimate cost function locality"""
        # Estimation based on number of measured qubits
        # Simplified: assume 1/3 of all qubits are measured
        measured_qubits = template.n_qubits // 3
        locality = measured_qubits / template.n_qubits

        return locality

    def _evaluate_initialization_strategy(self, template):
        """Evaluate initialization strategy"""
        # Analyze parameter placement pattern
        param_positions = []

        for i, gate in enumerate(template.gate_sequence):
            if gate.get('trainable', False):
                param_positions.append(i / len(template.gate_sequence))

        if not param_positions:
            return 0.0

        # Deviation from uniform distribution
        ideal_positions = np.linspace(0, 1, len(param_positions))
        position_deviation = np.mean(np.abs(np.array(sorted(param_positions)) - ideal_positions))

        return 1.0 - position_deviation

    def _evaluate_gradient_flow(self, template):
        """Evaluate gradient flow"""
        # Evaluate light cone spread
        light_cones = self._compute_light_cones(template)

        # Ideally grows exponentially
        if len(light_cones) < 2:
            return 0.5

        growth_rates = []
        for i in range(1, len(light_cones)):
            if light_cones[i-1] > 0:
                growth_rate = light_cones[i] / light_cones[i-1]
                growth_rates.append(growth_rate)

        avg_growth = np.mean(growth_rates) if growth_rates else 1.0

        # Ideal growth rate is 1.5-2.0
        if 1.5 <= avg_growth <= 2.0:
            return 1.0
        elif avg_growth < 1.5:
            return avg_growth / 1.5
        else:
            return 2.0 / avg_growth

    def _compute_light_cones(self, template):
        """Calculate light cone size for each layer"""
        layers = self._decompose_into_layers(template)
        light_cone_sizes = []

        affected_qubits = set()

        for layer in layers:
            for gate in layer:
                affected_qubits.update(gate['qubits'])

            light_cone_sizes.append(len(affected_qubits))

        return light_cone_sizes

    def _evaluate_entanglement_growth(self, template):
        """Evaluate entanglement growth pattern"""
        layers = self._decompose_into_layers(template)

        entanglement_per_layer = []
        for layer in layers:
            entangling_count = sum(1 for g in layer
                                 if g['gate'] in ['CNOT', 'CZ', 'SWAP'])
            entanglement_per_layer.append(entangling_count)

        if len(entanglement_per_layer) < 2:
            return 0.5

        # Gradual growth is ideal
        growth_smoothness = 0.0
        for i in range(1, len(entanglement_per_layer)):
            diff = entanglement_per_layer[i] - entanglement_per_layer[i-1]
            if diff >= 0:  # Non-decreasing
                growth_smoothness += 1.0

        return growth_smoothness / (len(entanglement_per_layer) - 1)

    def _compute_entanglement_features(self, template):
        """Calculate entanglement features"""
        features = []

        # Number of entangling gates
        entangling_gates = ['CNOT', 'CZ', 'SWAP']
        n_entangling = sum(1 for g in template.gate_sequence
                         if g['gate'] in entangling_gates)

        features.append(n_entangling / (len(template.gate_sequence) + 1))

        # Distribution of entangling gates
        if n_entangling > 0:
            positions = []
            for i, g in enumerate(template.gate_sequence):
                if g['gate'] in entangling_gates:
                    positions.append(i / len(template.gate_sequence))

            # Mean and variance of positions
            features.append(np.mean(positions))
            features.append(np.std(positions))
        else:
            features.extend([0.0, 0.0])

        return features

    def _compute_parameter_features(self, template):
        """Calculate parameter placement features"""
        features = []

        param_positions = []
        for i, g in enumerate(template.gate_sequence):
            if g.get('trainable', False):
                param_positions.append(i / len(template.gate_sequence))

        if param_positions:
            features.append(np.mean(param_positions))
            features.append(np.std(param_positions))
            features.append(len(param_positions) / len(template.gate_sequence))
        else:
            features.extend([0.0, 0.0, 0.0])

        return features

    def _compute_symmetry_features(self, template):
        """Calculate symmetry features"""
        # Simplification: similarity between first and second half gate distributions
        if len(template.gate_sequence) < 4:
            return [0.0]

        mid = len(template.gate_sequence) // 2
        first_half = template.gate_sequence[:mid]
        second_half = template.gate_sequence[mid:]

        # Gate type histograms
        hist1 = {}
        hist2 = {}

        for g in first_half:
            hist1[g['gate']] = hist1.get(g['gate'], 0) + 1

        for g in second_half:
            hist2[g['gate']] = hist2.get(g['gate'], 0) + 1

        # Cosine similarity
        all_gates = set(hist1.keys()) | set(hist2.keys())

        vec1 = [hist1.get(g, 0) for g in all_gates]
        vec2 = [hist2.get(g, 0) for g in all_gates]

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 > 0 and norm2 > 0:
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        else:
            similarity = 0.0

        return [similarity]

class GQEQuantumCircuitGeneratorWithGPT:
    """GPT-based GQE quantum circuit generator"""

    def __init__(self, n_qubits=6, noise_budget=0.01, hardware_topology='linear',
                 use_pretrained_gpt=False, use_ai_energy_prediction=True,
                 energy_prediction_mode='unsupervised'):  # Default changed to 'unsupervised'
        self.n_qubits = n_qubits
        self.noise_budget = noise_budget
        self.hardware_topology = hardware_topology
        self.use_pretrained_gpt = use_pretrained_gpt

        # Real device constraint parameters
        self.max_circuit_depth = 20
        self.preferred_gates = ['RY', 'RZ', 'CNOT', 'CZ']

        # Define gate vocabulary
        self._initialize_gate_vocabulary()

        # Initialize GPT model
        self._initialize_gpt_model()

        # Circuit evaluation history
        self.circuit_history = []
        self.energy_history = []

        # Additional: detailed history per round
        self.round_history = []
        self.gpt_generation_history = []

        # Additional: search parameters
        self.exploration_rate = 0.9  # Initial exploration rate
        self.exploration_decay = 0.85  # Exploration rate decay
        self.diversity_bonus = 0.2  # Diversity bonus

        # Initialize AI-enhanced energy estimator
        self.initialize_novelty_tracking()
        self.use_ai_energy_prediction = use_ai_energy_prediction
        self.energy_prediction_mode = energy_prediction_mode

        # Add real data cache
        self.cached_training_data = None
        self.cached_prepared_inputs = None

        if use_ai_energy_prediction:
            if energy_prediction_mode == 'unsupervised':
                # Use new Neural Error Mitigated estimator
                self.ai_energy_estimator = UnsupervisedQuantumEnergyEstimator(n_qubits, use_noise=True, shots=1000)
                _logger.info("Initialized unsupervised quantum energy estimator")
            else:
                raise ValueError(f"Unknown energy prediction mode: {energy_prediction_mode}")
        else:
            self.ai_energy_estimator = None

        self.mo_bayesian_optimizer = MultiObjectiveBayesianCircuitOptimizer(
                n_qubits=n_qubits,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                energy_estimator=self.ai_energy_estimator
            )
        _logger.info("Multi-objective Bayesian optimization enabled")

    def _calculate_circuit_depth_internal(self, gate_sequence):
        """Calculate circuit depth (internal use)"""
        if not gate_sequence:
            return 0

        qubit_layers = {}
        max_layer = 0

        for gate_info in gate_sequence:
            qubits = gate_info['qubits']
            current_layer = 0

            for q in qubits:
                if q < self.n_qubits and q in qubit_layers:
                    current_layer = max(current_layer, qubit_layers[q] + 1)

            for q in qubits:
                if q < self.n_qubits:
                    qubit_layers[q] = current_layer

            max_layer = max(max_layer, current_layer)

        return max_layer + 1

    def initialize_novelty_tracking(self):
        """Initialize novelty tracking"""
        if not hasattr(self, 'circuit_history'):
            self.circuit_history = []

        if not hasattr(self, 'novelty_history'):
            self.novelty_history = []

        if not hasattr(self, 'best_templates'):
            self.best_templates = []

    def update_circuit_history(self, template, score):
        """Update circuit history"""
        if not hasattr(self, 'circuit_history'):
            self.circuit_history = []

        circuit_data = {
            'gate_sequence': template.gate_sequence,
            'parameter_map': template.parameter_map,
            'score': score,
            'timestamp': time.time()
        }

        self.circuit_history.append(circuit_data)

        # Limit history size
        if len(self.circuit_history) > 200:
            self.circuit_history = self.circuit_history[-150:]

        # Update best templates
        if not hasattr(self, 'best_templates'):
            self.best_templates = []

        if not self.best_templates or score > getattr(self.best_templates[-1], 'best_score', -float('inf')):
            template.best_score = score
            self.best_templates.append(template)

            # Limit best templates history size
            if len(self.best_templates) > 20:
                self.best_templates = self.best_templates[-15:]

    def set_training_data(self, training_data):
        """Set training data from external source"""
        self.cached_training_data = training_data
        self.cached_prepared_inputs = self._prepare_training_data_for_energy_estimation()

    def _prepare_training_data_for_energy_estimation(self):
        """Prepare real data for energy estimation (PennyLane optimized version)"""
        if self.cached_training_data is None:
            return None

        # Create input vectors from each data point
        all_points = []

        # Collect all training points
        for data_type in ['initial_points', 'boundary_points', 'interior_points']:
            if data_type in self.cached_training_data:
                all_points.extend(self.cached_training_data[data_type])

        # Create list of input vectors
        prepared_inputs = []

        # Group data points for efficiency
        # Group spatially close points
        spatial_groups = {}
        grid_resolution = 10  # Grid resolution

        for point in all_points:
            # Calculate grid indices
            grid_x = int(point.x * grid_resolution / L)
            grid_y = int(point.y * grid_resolution / L)
            grid_z = int(point.z * grid_resolution / L)
            grid_t = int(point.t * grid_resolution / T)

            grid_key = (grid_x, grid_y, grid_z, grid_t)

            if grid_key not in spatial_groups:
                spatial_groups[grid_key] = []
            spatial_groups[grid_key].append(point)

        # Select representative points from each group
        for grid_key, group_points in spatial_groups.items():
            # Calculate center point of the group
            center_x = np.mean([p.x for p in group_points])
            center_y = np.mean([p.y for p in group_points])
            center_z = np.mean([p.z for p in group_points])
            center_t = np.mean([p.t for p in group_points])

            # Select the point closest to center as representative
            min_dist = float('inf')
            representative_point = None

            for point in group_points:
                dist = np.sqrt(
                    (point.x - center_x)**2 +
                    (point.y - center_y)**2 +
                    (point.z - center_z)**2 +
                    (point.t - center_t)**2
                )
                if dist < min_dist:
                    min_dist = dist
                    representative_point = point

            # Normalized input with coordinates and time
            normalized_input = np.array([
                representative_point.x / L,
                representative_point.y / L,
                representative_point.z / L,
                representative_point.t / T
            ])

            # Analytical solution value
            if hasattr(representative_point, 'u_true') and representative_point.u_true is not None:
                true_value = representative_point.u_true
            else:
                true_value = analytical_solution(
                    representative_point.x,
                    representative_point.y,
                    representative_point.z,
                    representative_point.t
                )

            # Calculate additional physical features
            # Distance from boundary
            boundary_distance = min(
                representative_point.x, L - representative_point.x,
                representative_point.y, L - representative_point.y,
                representative_point.z, L - representative_point.z
            ) / L

            # Distance from center
            center_distance = np.sqrt(
                (representative_point.x - L/2)**2 +
                (representative_point.y - L/2)**2 +
                (representative_point.z - L/2)**2
            ) / (L * np.sqrt(3) / 2)

            prepared_inputs.append({
                'coordinates': normalized_input,
                'point': representative_point,
                'true_value': true_value,
                'point_type': data_type,
                'boundary_distance': boundary_distance,
                'center_distance': center_distance,
                'group_size': len(group_points),  # Number of points in group
                'grid_key': grid_key
            })

        # Sort by importance (prioritize boundary and initial points)
        def point_importance(item):
            importance = 0.0
            if item['point_type'] == 'initial_points':
                importance += 1.0
            elif item['point_type'] == 'boundary_points':
                importance += 0.8
            else:  ## interior_points
                importance += 0.5

            # Closer to boundary is more important
            importance += (1 - item['boundary_distance']) * 0.2

            # Larger group size is more important (higher representativeness)
            importance += np.log1p(item['group_size']) * 0.1

            return -importance  # Negative for descending sort

        prepared_inputs.sort(key=point_importance)

        return prepared_inputs

    def _create_measurement_data_from_real_data(self, prepared_data, precise_energy, ai_predicted_energy):
        """Create measurement data for learning from real data (optimized for PennyLane format)"""
        measurement_dim = 2**self.n_qubits

        # Structure aligned with PennyLane qml.state or qml.density_matrix output format
        measurement_data = {}

        # Basic energy information
        measurement_data['energies'] = np.array([precise_energy, ai_predicted_energy])

        # Structure real data statistics
        if len(prepared_data) > 0:
            sample_size = min(20, len(prepared_data))  # Use more samples
            sample_data = prepared_data[:sample_size]

            # Coordinate data array
            coords_array = np.array([d['coordinates'] for d in sample_data])
            measurement_data['coordinates'] = coords_array

            # True values array
            true_values = np.array([d['true_value'] for d in sample_data])
            measurement_data['true_values'] = true_values

            # Statistics
            measurement_data['statistics'] = {
                'coord_mean': np.mean(coords_array, axis=0),
                'coord_std': np.std(coords_array, axis=0),
                'value_mean': np.mean(true_values),
                'value_std': np.std(true_values),
                'value_range': (np.min(true_values), np.max(true_values))
            }

            # Data point type distribution
            type_distribution = {}
            for d in sample_data:
                pt_type = d['point_type']
                type_distribution[pt_type] = type_distribution.get(pt_type, 0) + 1
            measurement_data['type_distribution'] = type_distribution

            # Physical features
            boundary_distances = [d.get('boundary_distance', 0.5) for d in sample_data]
            center_distances = [d.get('center_distance', 0.5) for d in sample_data]

            measurement_data['physical_features'] = {
                'boundary_distances': np.array(boundary_distances),
                'center_distances': np.array(center_distances),
                'mean_boundary_dist': np.mean(boundary_distances),
                'mean_center_dist': np.mean(center_distances)
            }

        # Convert to PennyLane measurement result format
        # Express as quantum state expectation values
        measurement_array = np.zeros(measurement_dim)

        # Place energy information first
        measurement_array[0] = precise_energy
        measurement_array[1] = ai_predicted_energy

        # Expand statistics into array
        idx = 2
        if 'statistics' in measurement_data:
            stats = measurement_data['statistics']
            # Coordinate mean values (4D)
            for i in range(4):
                if idx < measurement_dim:
                    measurement_array[idx] = stats['coord_mean'][i]
                    idx += 1

            # Coordinate standard deviations (4D)
            for i in range(4):
                if idx < measurement_dim:
                    measurement_array[idx] = stats['coord_std'][i]
                    idx += 1

            # Value statistics
            if idx < measurement_dim:
                measurement_array[idx] = stats['value_mean']
                idx += 1
            if idx < measurement_dim:
                measurement_array[idx] = stats['value_std']
                idx += 1
            if idx < measurement_dim:
                measurement_array[idx] = stats['value_range'][0]  # min
                idx += 1
            if idx < measurement_dim:
                measurement_array[idx] = stats['value_range'][1]  # max
                idx += 1

        # Data type distribution
        if 'type_distribution' in measurement_data and idx + 4 < measurement_dim:
            for pt_type in ['initial_points', 'boundary_points', 'interior_points']:
                measurement_array[idx] = measurement_data['type_distribution'].get(pt_type, 0)
                idx += 1

        # Physical features
        if 'physical_features' in measurement_data and idx + 2 < measurement_dim:
            phys_feat = measurement_data['physical_features']
            measurement_array[idx] = phys_feat['mean_boundary_dist']
            idx += 1
            measurement_array[idx] = phys_feat['mean_center_dist']
            idx += 1

        return measurement_array

    def _create_quantum_state_from_real_data(self, point_data, n_qubits):
        """Create quantum state from real data (improved version utilizing PennyLane features)"""
        # Build quantum state from coordinate information
        coords = point_data['coordinates']
        true_val = point_data['true_value']

        # State vector dimension
        state_dim = 2**n_qubits

        # Method 1: Physically meaningful basis state superposition
        # Quantum state construction considering heat conduction equation solution characteristics

        # Map spatial coordinates to qubits
        # Each qubit represents different spatial regions
        qubit_regions = np.linspace(0, 1, n_qubits + 1)

        # Calculate initial amplitudes
        amplitudes = np.zeros(state_dim, dtype=complex)

        for basis_idx in range(state_dim):
            # Convert basis state to binary representation
            basis_binary = format(basis_idx, f'0{n_qubits}b')

            # Calculate each qubit's contribution
            amplitude = 1.0
            phase = 0.0

            for qubit_idx, bit in enumerate(basis_binary):
                # Center of spatial region represented by this qubit
                region_center = (qubit_regions[qubit_idx] + qubit_regions[qubit_idx + 1]) / 2

                if bit == '1':
                    # Excited state: amplitude based on distance from coordinates
                    for dim_idx, coord in enumerate(coords[:3]):  # x, y, z
                        distance = abs(coord - region_center)
                        amplitude *= np.exp(-distance**2 / (2 * 0.1**2))
                        phase += coord * np.pi * (qubit_idx + 1) / n_qubits
                else:
                    # Ground state: complementary amplitude
                    for dim_idx, coord in enumerate(coords[:3]):
                        distance = abs(coord - region_center)
                        amplitude *= (1 - 0.5 * np.exp(-distance**2 / (2 * 0.2**2)))

            # Time evolution influence
            time_factor = coords[3]  # Normalized time
            amplitude *= np.exp(-time_factor * basis_idx / (2 * state_dim))

            # Reflect true value influence in phase
            phase += 2 * np.pi * true_val * basis_idx / state_dim

            # Set as complex amplitude
            amplitudes[basis_idx] = amplitude * np.exp(1j * phase)

        # Normalization (as required by PennyLane)
        norm = np.linalg.norm(amplitudes)
        if norm > 1e-10:
            amplitudes = amplitudes / norm
        else:
            # Fallback: uniform superposition
            amplitudes = np.ones(state_dim) / np.sqrt(state_dim)

        # Separate real and imaginary parts (for PennyLane MottonenStatePreparation requirements)
        state_vector = np.real(amplitudes).astype(np.float64)

        # If you want to keep phase information, use complex numbers
        # However, AmplitudeEmbedding only supports real numbers
        if np.max(np.abs(np.imag(amplitudes))) > 1e-10:
            # If phase is important, use only absolute values
            state_vector = np.abs(amplitudes).astype(np.float64)
            # Re-normalize
            norm = np.linalg.norm(state_vector)
            if norm > 1e-10:
                state_vector = state_vector / norm

        return state_vector

    def _estimate_circuit_energy_enhanced(self, template, update_learning=False):
        """AI-enhanced energy estimation (real data version)"""

        if not self.ai_energy_estimator or not self.use_ai_energy_prediction:
            return self._estimate_circuit_energy(template)

        try:
            if self.energy_prediction_mode == 'unsupervised':
                # Use cached real data
                prepared_data = self.cached_prepared_inputs

                if not prepared_data:
                    # Fallback to conventional method if no data
                    input_dim = 2**self.n_qubits
                    input_data = np.random.randn(input_dim)
                    input_data = input_data / (np.linalg.norm(input_data) + 1e-10)
                    ai_predicted_energy = self.ai_energy_estimator.estimate_energy_unsupervised(
                        template, input_data
                    )
                else:
                    # Estimate energy with multiple real data points and take average
                    energy_estimates = []

                    # Use maximum 10 data points
                    sample_size = min(10, len(prepared_data))
                    sample_indices = np.random.choice(len(prepared_data), sample_size, replace=False)

                    for idx in sample_indices:
                        point_data = prepared_data[idx]

                        # Create quantum state from real data
                        input_data = self._create_quantum_state_from_real_data(
                            point_data, self.n_qubits
                        )

                        # Energy estimation
                        energy = self.ai_energy_estimator.estimate_energy_unsupervised(
                            template, input_data
                        )

                        energy_estimates.append(energy)

                    # Average energy
                    ai_predicted_energy = np.mean(energy_estimates)

                if update_learning:
                    try:
                        precise_energy = self._estimate_circuit_energy(template)

                        # Update learning with measurement results (using real data)
                        if prepared_data:
                            # Create learning measurement data from real data
                            measurement_data = self._create_measurement_data_from_real_data(
                                prepared_data, precise_energy, ai_predicted_energy
                            )
                        else:
                            # Fallback
                            measurement_data = np.array([precise_energy, ai_predicted_energy])
                            if len(measurement_data) < 2**self.n_qubits:
                                padded_data = np.zeros(2**self.n_qubits)
                                padded_data[:len(measurement_data)] = measurement_data
                                measurement_data = padded_data

                        self.ai_energy_estimator.update_learning(template, measurement_data)

                        return float(precise_energy)
                    except Exception as e:
                        _logger.warning(f"Precise calculation failed, using AI prediction: {e}")
                        return float(ai_predicted_energy)
                else:
                    return float(ai_predicted_energy)

            elif self.energy_prediction_mode == 'ensemble':
                return self.ai_energy_estimator.predict_energy(template)

        except Exception as e:
            _logger.error(f"AI-enhanced energy estimation error: {e}")
            return self._estimate_circuit_energy(template)

    def _initialize_gate_vocabulary(self):
        """Initialize gate vocabulary"""
        self.gate_tokens = ['[PAD]', '[START]', '[END]', '[SEP]']

        # Single qubit gates
        for gate in ['RX', 'RY', 'RZ', 'H', 'S', 'T']:
            for q in range(self.n_qubits):
                self.gate_tokens.append(f'{gate}_{q}')

        # Two-qubit gates
        for gate in ['CNOT', 'CZ', 'SWAP']:
            for q1 in range(self.n_qubits):
                for q2 in range(self.n_qubits):
                    if q1 != q2:
                        self.gate_tokens.append(f'{gate}_{q1}_{q2}')

        # Parameter value tokens (discretized)
        param_values = np.linspace(-np.pi, np.pi, 16)
        for i, val in enumerate(param_values):
            self.gate_tokens.append(f'PARAM_{i}')

        # Token mapping
        self.token_to_id = {token: i for i, token in enumerate(self.gate_tokens)}
        self.id_to_token = {i: token for i, token in enumerate(self.gate_tokens)}
        self.vocab_size = len(self.gate_tokens)

        _logger.info(f"Gate vocabulary size: {self.vocab_size}")

    def _initialize_gpt_model(self):
        """Initialize GPT model"""
        self.gpt_model = QuantumCircuitGPT(
                vocab_size=self.vocab_size,
                n_embd=256,
                n_head=8,
                n_layer=6,
                block_size=128,
                dropout=0.1
            ).to(device)

        self.gpt_optimizer = torch.optim.Adam(
            self.gpt_model.parameters(),
            lr=5e-4
        )

        if self.use_pretrained_gpt:
            # Use pre-trained model (custom fine-tuned)
            try:
                model_path = 'quantum_circuit_gpt.pth'
                ketgpt_done = False
                if os.path.exists(model_path):
                    _logger.info(f"Loading pre-trained GPT model: {model_path}")
                    try:
                        # PyTorch 2.6+ compatibility
                        if hasattr(torch.serialization, 'safe_globals'):
                            with torch.serialization.safe_globals([QuantumCircuitTemplate]):
                                checkpoint = torch.load(model_path, map_location=device)
                        else:
                            checkpoint = torch.load(model_path, map_location=device, weights_only=False)

                        self.gpt_model.load_state_dict(checkpoint['model_state_dict'])
                        ketgpt_done = checkpoint.get('ketgpt_pretrained', False)
                    except Exception as e:
                        _logger.error(f"Model loading error: {e}")
                        _logger.warning("Initializing as new model")

                # Always ensure ketGPT pre-training has been done
                if not ketgpt_done:
                    _logger.info("KetGPT pre-training not found in checkpoint, running ketGPT initialization...")
                    self._initialize_ketgpt_dataset()

            except Exception as e:
                _logger.error(f"GPT model initialization error: {e}")
                self.gpt_model = None
        else:
            # New ketGPT model
            self._initialize_ketgpt_dataset()

        _logger.info(f"GPT model parameters: {sum(p.numel() for p in self.gpt_model.parameters())}")

    def _initialize_ketgpt_dataset(self):
        """Load and preprocess ketGPT dataset (h5py error countermeasure version)
            References:
            - Apak et al. "KetGPT - Dataset Augmentation of Quantum Circuits
            using Transformers" arXiv:2402.13352 (2024)
        """
        try:
            # Load PennyLane ketGPT dataset
            [ketgpt_dataset_temp] = qml.data.load("ketgpt")

            # Extract only necessary data from h5py object
            circuits_data = []
            for circuit in ketgpt_dataset_temp.circuits:
                # Convert circuit data to Python standard types
                circuits_data.append(circuit)

            _logger.info(f"KetGPT dataset loaded: {len(circuits_data)} circuits")

            # Delete reference to h5py object
            del ketgpt_dataset_temp

            # Prepare data for pre-training
            self.pretrain_data = []
            for i, circuit in enumerate(circuits_data):
                gate_sequence = self._pennylane_to_gate_sequence(circuit)
                if gate_sequence and len(gate_sequence) <= self.max_circuit_depth:
                    self.pretrain_data.append({
                        'gate_sequence': gate_sequence,
                        'energy': -1.0 - 0.01 * i,  # Dummy energy value
                        'score': 0.8 + 0.001 * i
                    })

            # Pre-train GPT model
            if self.gpt_model is not None and self.pretrain_data:
                _logger.info("Pre-training GPT model with ketGPT data...")
                self._train_gpt_on_circuits(self.pretrain_data, epochs=len(self.pretrain_data)*10)
                self.use_ketgpt_data = True
                _logger.info(f"KetGPT pre-training completed with {len(self.pretrain_data)} circuits")

        except Exception as e:
            _logger.error(f"KetGPT dataset loading error: {e}")
            self.use_ketgpt_data = False

    def _pennylane_to_gate_sequence(self, pennylane_ops):
        """Convert PennyLane operations to gate_sequence format"""
        gate_sequence = []
        param_counter = 0

        for op in pennylane_ops:
            gate_info = {
                'gate': op.name.upper(),
                'qubits': list(op.wires),
                'trainable': hasattr(op, 'parameters') and len(op.parameters) > 0
            }

            if gate_info['trainable']:
                gate_info['param_idx'] = param_counter
                param_counter += 1

            # Gate type normalization
            gate_map = {
                'PAULIX': 'X', 'PAULIY': 'Y', 'PAULIZ': 'Z',
                'HADAMARD': 'H', 'CNOT': 'CNOT', 'CZ': 'CZ',
                'RX': 'RX', 'RY': 'RY', 'RZ': 'RZ'
            }

            if gate_info['gate'] in gate_map:
                gate_info['gate'] = gate_map[gate_info['gate']]
                gate_sequence.append(gate_info)

        return gate_sequence

    def _circuit_to_tokens(self, gate_sequence):
        """Convert circuit to token sequence"""
        tokens = [self.token_to_id['[START]']]

        for gate_info in gate_sequence:
            try:
                gate_type = gate_info['gate']
                qubits = gate_info['qubits']

                # Generate gate token
                if len(qubits) == 1:
                    token_str = f'{gate_type}_{qubits[0]}'
                elif len(qubits) == 2:
                    token_str = f'{gate_type}_{qubits[0]}_{qubits[1]}'
                else:
                    continue  # Skip gates with 3+ qubits

                # Add token only if it exists
                if token_str in self.token_to_id:
                    tokens.append(self.token_to_id[token_str])

                # Parameter token (safe processing)
                if gate_info.get('trainable', False) and gate_info.get('param_idx') is not None:
                    param_idx = gate_info['param_idx']
                    # Ensure param_idx is integer
                    if isinstance(param_idx, (int, np.integer)):
                        param_token = f'PARAM_{param_idx % 16}'
                        if param_token in self.token_to_id:
                            tokens.append(self.token_to_id[param_token])

            except Exception as e:
                _logger.warning(f"Gate tokenization error: {e}, Gate: {gate_info}")
                continue

        tokens.append(self.token_to_id['[END]'])
        return tokens

    def _tokens_to_circuit(self, tokens):
        """Build circuit from token sequence"""
        gate_sequence = []
        parameter_map = {}
        param_counter = 0

        i = 0
        while i < len(tokens):
            if tokens[i] in [self.token_to_id['[PAD]'],
                        self.token_to_id['[START]'],
                        self.token_to_id['[END]']]:
                i += 1
                continue

            token_str = self.id_to_token.get(tokens[i], '')

            # Parse gate tokens
            if '_' in token_str and not token_str.startswith('PARAM'):
                parts = token_str.split('_')
                gate_type = parts[0]

                if gate_type in ['RX', 'RY', 'RZ', 'H', 'S', 'T']:
                    # Single qubit gate
                    qubit = int(parts[1])
                    trainable = gate_type in ['RX', 'RY', 'RZ']

                    gate_info = {
                        'gate': gate_type,
                        'qubits': [qubit],
                        'param_idx': param_counter if trainable else None,
                        'trainable': trainable
                    }

                    if trainable:
                        parameter_map[f'{gate_type}_gate_{len(gate_sequence)}'] = param_counter
                        param_counter += 1

                    gate_sequence.append(gate_info)

                elif gate_type in ['CNOT', 'CZ', 'SWAP']:
                    # Two-qubit gate
                    if len(parts) >= 3:
                        qubit1 = int(parts[1])
                        qubit2 = int(parts[2])

                        # Ensure control and target qubits are different
                        if qubit1 != qubit2:
                            gate_info = {
                                'gate': gate_type,
                                'qubits': [qubit1, qubit2],
                                'param_idx': None,
                                'trainable': False
                            }

                            gate_sequence.append(gate_info)
                        # else: Skip if same qubit

            i += 1

        return gate_sequence, parameter_map

    def _train_gpt_on_circuits(self, training_data, epochs=10):
        """Train GPT model on circuit data (improved version)"""
        if self.gpt_model is None:
            return

        _logger.info(f"Starting GPT model training ({len(training_data)} data, {epochs} epochs)")

        # Prepare dataset (maintain diversity)
        sequences = []
        energies = []
        scores = []

        # Normalize data
        all_energies = [data['energy'] for data in training_data]
        energy_mean = np.mean(all_energies)
        energy_std = np.std(all_energies) + 1e-6

        for data in training_data:
            tokens = self._circuit_to_tokens(data['gate_sequence'])
            sequences.append(tokens)
            # Normalize energy
            normalized_energy = (data['energy'] - energy_mean) / energy_std
            energies.append(normalized_energy)
            scores.append(data.get('score', 0.5))

        # Weighted sampling (emphasize high-score data)
        weights = np.array(scores)
        weights = weights / weights.sum()

        dataset = QuantumCircuitDataset(sequences, energies)

        # Use weighted sampler
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(dataset),
            replacement=True
        )

        dataloader = DataLoader(dataset, batch_size=16, sampler=sampler)

        self.gpt_model.train()

        # Learning rate scheduling
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.gpt_optimizer,
            T_max=epochs,
            eta_min=1e-5
        )

        best_loss = float('inf')
        patience = 0

        for epoch in range(epochs):
            total_loss = 0.0

            for batch_idx, (seq_batch, energy_batch) in enumerate(dataloader):
                seq_batch = seq_batch.to(device)
                energy_batch = energy_batch.to(device)

                # GPT forward pass
                logits, loss, energy_pred = self.gpt_model(
                    seq_batch,
                    targets=seq_batch,
                    energies=energy_batch
                )

                # Add regularization term
                l2_reg = 0.0
                for param in self.gpt_model.parameters():
                    l2_reg += torch.norm(param, 2)

                total_batch_loss = loss + 0.0001 * l2_reg

                # Backpropagation
                self.gpt_optimizer.zero_grad()
                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.gpt_model.parameters(), 1.0)
                self.gpt_optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            scheduler.step()

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience = 0
            else:
                patience += 1

            if patience > 50:
                _logger.info(f"  Early stopping: Epoch {epoch + 1}")
                break

            if (epoch + 1) % 10 == 0:
                _logger.info(f"  Epoch {epoch + 1}/{epochs}, Average loss: {avg_loss:.4f}")

    def _generate_fallback_circuit(self):
        """Generate fallback circuit (when GPT is unavailable)"""
        gate_sequence = []
        parameter_map = {}
        param_counter = 0

        # Hardware efficient ansatz
        n_layers = min(3, self.max_circuit_depth // (self.n_qubits + 1))

        for layer in range(n_layers):
            # RY rotation layer
            for q in range(self.n_qubits):
                gate_sequence.append({
                    'gate': 'RY',
                    'qubits': [q],
                    'param_idx': param_counter,
                    'trainable': True
                })
                parameter_map[f'ry_l{layer}_q{q}'] = param_counter
                param_counter += 1

            # CNOT layer
            if layer < n_layers - 1:
                for q in range(self.n_qubits - 1):
                    gate_sequence.append({
                        'gate': 'CNOT',
                        'qubits': [q, q + 1],
                        'param_idx': None,
                        'trainable': False
                    })

        return gate_sequence, parameter_map

    def _estimate_circuit_energy(self, template):
        """Improved version: Unsupervised learning circuit energy estimation (real data version)"""
        try:
            n_qubits = template.n_qubits

            # Device setting
            if self.noise_budget > 0:
                dev = qml.device('default.mixed', wires=n_qubits, shots=1024)
            else:
                dev = qml.device('default.qubit', wires=n_qubits)

            # Use unsupervised learning energy estimator
            if self.ai_energy_estimator and isinstance(self.ai_energy_estimator,
                                                      UnsupervisedQuantumEnergyEstimator):
                # Generate input from real data
                if self.cached_prepared_inputs:
                    # Randomly select real data point
                    point_data = self.cached_prepared_inputs[
                        np.random.randint(len(self.cached_prepared_inputs))
                    ]
                    input_data = self._create_quantum_state_from_real_data(
                        point_data, n_qubits
                    )
                else:
                    # Fallback: random data
                    input_dim = 2**n_qubits
                    input_data = np.random.randn(input_dim)
                    input_data = input_data / (np.linalg.norm(input_data) + 1e-10)

                # Unsupervised energy estimation
                energy = self.ai_energy_estimator.estimate_energy_unsupervised(
                    template, input_data
                )

                # Update learning with measurement results (real data based)
                if self.cached_prepared_inputs:
                    # Generate measurement results from multiple real data points
                    measurement_results = []
                    for i in range(min(10, len(self.cached_prepared_inputs))):
                        point_data = self.cached_prepared_inputs[i]
                        measurement_results.append(point_data['true_value'])

                    measurement_results = np.array(measurement_results)
                    # Pad to appropriate size
                    if len(measurement_results) < 2**n_qubits:
                        padded_results = np.zeros(2**n_qubits)
                        padded_results[:len(measurement_results)] = measurement_results
                        measurement_results = padded_results
                else:
                    measurement_results = np.random.randn(10)

                self.ai_energy_estimator.update_learning(template, measurement_results)

                return energy

            # Fallback: General variational quantum eigensolver (VQE) approach
            # Problem-independent Hamiltonian (built from real data)
            coeffs = []
            obs = []

            if self.cached_prepared_inputs:
                # Generate coefficients from real data
                sample_size = min(n_qubits, len(self.cached_prepared_inputs))
                for i in range(sample_size):
                    point_data = self.cached_prepared_inputs[i]
                    coords = point_data['coordinates']
                    true_val = point_data['true_value']

                    # Coefficients based on coordinates and true values
                    coeff = -true_val * (1.0 + 0.1 * coords[3])  # Consider time
                    coeffs.append(coeff)
                    obs.append(qml.PauliZ(i))

                # Interaction terms (from coordinate correlations)
                for i in range(min(n_qubits - 1, sample_size - 1)):
                    point1 = self.cached_prepared_inputs[i]
                    point2 = self.cached_prepared_inputs[i + 1]

                    # Spatial proximity-based interactions
                    dist = np.linalg.norm(
                        point1['coordinates'][:3] - point2['coordinates'][:3]
                    )
                    coeff = -0.5 * np.exp(-dist)
                    coeffs.append(coeff)
                    obs.append(qml.PauliZ(i) @ qml.PauliZ(i+1))
            else:
                # Fallback: random coefficients
                for i in range(n_qubits):
                    coeff = np.random.uniform(-1, 1)
                    coeffs.append(coeff)
                    obs.append(qml.PauliZ(i))

                for i in range(n_qubits - 1):
                    coeff = np.random.uniform(-0.5, 0.5)
                    coeffs.append(coeff)
                    obs.append(qml.PauliZ(i) @ qml.PauliZ(i+1))

            H = qml.Hamiltonian(coeffs, obs)

            @qml.qnode(dev)
            def energy_circuit():
                # Initial state preparation (real data based)
                if self.cached_prepared_inputs:
                    # Build initial state from real data
                    sample_idx = np.random.randint(len(self.cached_prepared_inputs))
                    point_data = self.cached_prepared_inputs[sample_idx]
                    coords = point_data['coordinates']

                    # Coordinate-based initialization
                    for i in range(n_qubits):
                        angle = np.pi * coords[i % 4]  # Cycle through x, y, z, t
                        qml.RY(angle, wires=i)
                else:
                    # Fallback: random initialization
                    for i in range(n_qubits):
                        qml.RY(np.random.uniform(0, np.pi), wires=i)

                # Execute circuit based on template
                param_values = np.random.uniform(-np.pi/4, np.pi/4,
                                               size=len(template.parameter_map))
                param_counter = 0

                for gate_info in template.gate_sequence:
                    gate_type = gate_info['gate']
                    qubits = gate_info['qubits']

                    if any(q >= n_qubits for q in qubits):
                        continue

                    if gate_type == 'H':
                        qml.Hadamard(wires=qubits[0])
                    elif gate_type == 'RY' and gate_info.get('trainable', False):
                        if param_counter < len(param_values):
                            qml.RY(param_values[param_counter], wires=qubits[0])
                            param_counter += 1
                    elif gate_type == 'RZ' and gate_info.get('trainable', False):
                        if param_counter < len(param_values):
                            qml.RZ(param_values[param_counter], wires=qubits[0])
                            param_counter += 1
                    elif gate_type == 'RX' and gate_info.get('trainable', False):
                        if param_counter < len(param_values):
                            qml.RX(param_values[param_counter], wires=qubits[0])
                            param_counter += 1
                    elif gate_type == 'CNOT' and len(qubits) >= 2:
                        if qubits[0] != qubits[1]:
                            qml.CNOT(wires=qubits[:2])
                    elif gate_type == 'CZ' and len(qubits) >= 2:
                        if qubits[0] != qubits[1]:
                            qml.CZ(wires=qubits[:2])

                return qml.expval(H)

            # Calculate energy expectation value
            energy = float(energy_circuit())

            return energy

        except Exception as e:
            _logger.error(f"Energy calculation error: {e}")
            # Fallback: estimation based on circuit complexity
            return -1.0 + 0.01 * len(template.parameter_map) + 0.005 * len(template.gate_sequence)

    def save_gpt_generation_history(self, save_path='results/'):
        """Save GPT generation history (simplified round information)"""
        os.makedirs(save_path, exist_ok=True)

        history_path = os.path.join(save_path, 'efficient_gpt_generation_history.json')

        history = {
            'generation_info': {
                'total_circuits': len(self.circuit_history),
                'optimization_method': 'efficient_single_round'
            },
            'best_circuits': [],
            'energy_progression': self.energy_history,
            'gpt_model_info': {
                'vocab_size': self.vocab_size,
                'n_embd': 256,
                'n_head': 8,
                'n_layer': 6,
                'parameters': sum(p.numel() for p in self.gpt_model.parameters()) if self.gpt_model else 0
            },
            'multi_objective_info': {
                'n_objectives': 9,
                'objective_names': [
                    'Hardware Efficiency', 'Noise Resilience', 'Expressivity',
                    'Mitigation Compatibility', 'Trainability', 'Entanglement Capability',
                    'Depth Efficiency', 'Parameter Efficiency', 'Energy Estimation Quality'
                ]
            },
            'optimization_summary': {
                'method': 'efficient_evaluation',
                'total_evaluations': len(self.circuit_history),
                'average_evaluation_time': 0.0  # Add if calculable
            }
        }

        # Save best circuit information
        if hasattr(self, 'mo_optimization_history') and self.mo_optimization_history.get('pareto_fronts'):
            final_pareto = self.mo_optimization_history['pareto_fronts'][-1]
            for i in range(min(5, len(final_pareto))):
                history['best_circuits'].append({
                    'rank': i + 1,
                    'objectives': final_pareto[i].tolist() if hasattr(final_pareto[i], 'tolist') else list(final_pareto[i]),
                    'average_score': float(final_pareto[i].mean())
                })

        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        _logger.info(f"Efficient GPT generation history saved: {history_path}")

    def _find_pareto_optimal(self, candidates):
        """Find indices of Pareto optimal solutions"""
        objectives_matrix = torch.stack([c['objectives'] for c in candidates])
        n = len(candidates)

        is_dominated = torch.zeros(n, dtype=torch.bool)

        for i in range(n):
            for j in range(n):
                if i != j:
                    # Treat as maximization problem (higher scores are better)
                    if torch.all(objectives_matrix[j] >= objectives_matrix[i]) and \
                       torch.any(objectives_matrix[j] > objectives_matrix[i]):
                        is_dominated[i] = True
                        break

        return torch.where(~is_dominated)[0].tolist()

    def _save_gpt_model(self, best_circuits):
        """Save GPT model and multi-objective optimization results"""
        model_path = 'quantum_circuit_gpt.pth'

        # Save Pareto optimal solution information
        pareto_solutions = []
        if len(best_circuits) > 10:
            for circuit in best_circuits[:10]:  # Top 10
                pareto_solutions.append({
                    'objectives': circuit['objectives'].tolist(),
                    'energy': circuit['energy'],
                    'gate_count': len(circuit['template'].gate_sequence),
                    'param_count': len(circuit['template'].parameter_map)
                })
        else:
            for circuit in best_circuits:
                pareto_solutions.append({
                    'objectives': circuit['objectives'].tolist(),
                    'energy': circuit['energy'],
                    'gate_count': len(circuit['template'].gate_sequence),
                    'param_count': len(circuit['template'].parameter_map)
                })

        save_data = {
            'model_state_dict': self.gpt_model.state_dict(),
            'optimizer_state_dict': self.gpt_optimizer.state_dict(),
            'vocab_size': self.vocab_size,
            'multi_objective': True,
            'n_objectives': 9,
            'pareto_solutions': pareto_solutions,
            'ketgpt_pretrained': getattr(self, 'use_ketgpt_data', False),
        }

        torch.save(save_data, model_path, _use_new_zipfile_serialization=True)
        _logger.info(f"Multi-objective optimization GPT model saved: {model_path}")

    def _visualize_multi_objective_details(self, save_path):
        """Detailed visualization of multi-objective optimization (including energy estimation quality)"""
        if not self.mo_optimization_history['objectives_evolution']:
            return

        # Names of 9 objective functions (added energy estimation quality)
        objective_names = [
            'Hardware\nEfficiency', 'Noise\nResilience', 'Expressivity',
            'Mitigation\nCompatibility', 'Trainability', 'Entanglement\nCapability',
            'Depth\nEfficiency', 'Parameter\nEfficiency', 'Energy\nEstimation\nQuality'
        ]

        # Plot evolution of each objective function individually
        n_objectives = 9  # Changed from 8 to 9
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))  # Changed from 2x4 to 3x3
        axes = axes.flatten()

        for obj_idx in range(n_objectives):
            ax = axes[obj_idx]

            rounds = []
            means = []
            stds = []
            mins = []
            maxs = []

            for evolution in self.mo_optimization_history['objectives_evolution']:
                if evolution['mean'].shape[0] > obj_idx:  # Compatibility check
                    rounds.append(evolution['round'])
                    means.append(evolution['mean'][obj_idx])
                    stds.append(evolution['std'][obj_idx])
                    mins.append(evolution['min'][obj_idx])
                    maxs.append(evolution['max'][obj_idx])

            # Mean and standard deviation
            ax.errorbar(rounds, means, yerr=stds, fmt='o-', linewidth=2,
                    markersize=6, capsize=5, label='Mean +/- Std')

            # Min-max range
            ax.fill_between(rounds, mins, maxs, alpha=0.2)

            ax.set_xlabel('Round')
            ax.set_ylabel('Score')
            ax.set_title(objective_names[obj_idx], fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.1)

        plt.suptitle('Evolution of Individual Objectives (Including Energy Estimation Quality)', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'mo_objectives_evolution_with_energy.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Add detailed analysis of energy estimation quality
        if self.mo_optimization_history.get('energy_quality_evolution'):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Energy estimation quality evolution
            rounds = [e['round'] for e in self.mo_optimization_history['energy_quality_evolution']]
            means = [e['mean'] for e in self.mo_optimization_history['energy_quality_evolution']]
            stds = [e['std'] for e in self.mo_optimization_history['energy_quality_evolution']]

            ax1.errorbar(rounds, means, yerr=stds, fmt='go-', linewidth=2,
                        markersize=8, capsize=5, label='Energy Estimation Quality')
            ax1.set_xlabel('Optimization Round')
            ax1.set_ylabel('Quality Score')
            ax1.set_title('Energy Estimation Quality Evolution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1.1)

            # Correlation between energy estimation quality and other objectives
            if len(self.mo_optimization_history['objectives_evolution']) > 0:
                last_evolution = self.mo_optimization_history['objectives_evolution'][-1]
                if last_evolution['mean'].shape[0] >= 9:
                    correlations = []
                    for i in range(8):  # Other 8 objective functions
                        corr = np.corrcoef([e['mean'][i] for e in self.mo_optimization_history['objectives_evolution']],
                                        [e['mean'][8] for e in self.mo_optimization_history['objectives_evolution']])[0, 1]
                        correlations.append(corr)

                    ax2.bar(range(8), correlations)
                    ax2.set_xticks(range(8))
                    ax2.set_xticklabels([name.replace('\n', ' ') for name in objective_names[:8]],
                                    rotation=45, ha='right')
                    ax2.set_ylabel('Correlation with Energy Quality')
                    ax2.set_title('Correlation Analysis')
                    ax2.grid(True, alpha=0.3, axis='y')
                    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)

            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'energy_estimation_quality_analysis.png'),
                    dpi=300, bbox_inches='tight')
            plt.close()

    def _visualize_pareto_evolution(self, save_path):
        """Visualize Pareto front evolution"""
        if len(self.mo_optimization_history['pareto_fronts']) < 2:
            return

        # Display Pareto fronts in 2D projection (main two objective functions)
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        # Display with different objective function pairs
        objective_pairs = [
            (0, 1, 'Hardware Efficiency', 'Noise Resilience'),
            (2, 4, 'Expressivity', 'Trainability'),
            (0, 2, 'Hardware Efficiency', 'Expressivity'),
            (1, 3, 'Noise Resilience', 'Mitigation Compatibility')
        ]

        n_rounds = len(self.mo_optimization_history['pareto_fronts'])
        colors = plt.cm.viridis(np.linspace(0, 1, n_rounds))

        for idx, (ax, (obj1, obj2, name1, name2)) in enumerate(zip(axes.flatten(), objective_pairs)):
            for round_idx, pareto_front in enumerate(self.mo_optimization_history['pareto_fronts']):
                if pareto_front.shape[0] > 0:
                    ax.scatter(pareto_front[:, obj1], pareto_front[:, obj2],
                            c=[colors[round_idx]], s=50, alpha=0.7,
                            label=f'Round {round_idx + 1}')

            ax.set_xlabel(name1)
            ax.set_ylabel(name2)
            ax.set_title(f'{name1} vs {name2}')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1.1)
            ax.set_ylim(0, 1.1)

            if idx == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        plt.suptitle('Pareto Front Evolution (2D Projections)', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'pareto_front_evolution.png'),
                dpi=300, bbox_inches='tight')
        plt.close()

        # 3D Pareto front (final round only)
        if len(self.mo_optimization_history['pareto_fronts'][-1]) > 0:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            final_pareto = self.mo_optimization_history['pareto_fronts'][-1]

            # Display with three main objective functions
            scatter = ax.scatter(final_pareto[:, 0], final_pareto[:, 1], final_pareto[:, 2],
                            c=final_pareto[:, 4], cmap='viridis', s=100, alpha=0.8)

            ax.set_xlabel('Hardware Efficiency')
            ax.set_ylabel('Noise Resilience')
            ax.set_zlabel('Expressivity')
            ax.set_title('Final 3D Pareto Front (colored by Trainability)', fontsize=14)

            # Color bar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Trainability', rotation=270, labelpad=20)

            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'pareto_front_3d.png'),
                    dpi=300, bbox_inches='tight')
            plt.close()

    def visualize_optimization_history(self, save_path='results/'):
        """Visualize optimization history (without round information)"""
        os.makedirs(save_path, exist_ok=True)

        if not self.circuit_generation_history:
            _logger.warning("No optimization history available")
            return

        # Display overall picture in single figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Circuit update progression
        ax = axes[0, 0]
        generations = [entry['generation'] for entry in self.circuit_generation_history]
        n_params = [len(entry['template'].parameter_map) for entry in self.circuit_generation_history]
        n_gates = [len(entry['template'].gate_sequence) for entry in self.circuit_generation_history]

        ax.plot(generations, n_params, 'b-', label='Parameters', linewidth=2)
        ax.plot(generations, n_gates, 'r--', label='Gates', linewidth=2)
        ax.set_xlabel('Circuit Update')
        ax.set_ylabel('Count')
        ax.set_title('Circuit Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Objective function evolution (if available)
        ax = axes[0, 1]
        if hasattr(self, 'mo_optimization_history') and self.mo_optimization_history.get('objectives_evolution'):
            for obj_data in self.mo_optimization_history['objectives_evolution']:
                if 'mean' in obj_data:
                    ax.plot(obj_data['mean'], marker='o')
            ax.set_xlabel('Objective Index')
            ax.set_ylabel('Mean Value')
            ax.set_title('Objective Functions Evolution')
        else:
            ax.text(0.5, 0.5, 'No objective data available',
                    ha='center', va='center', transform=ax.transAxes)
        ax.grid(True, alpha=0.3)

        # 3. Energy estimation quality
        ax = axes[1, 0]
        if hasattr(self, 'energy_estimation_history') and self.energy_estimation_history:
            energy_values = [entry['actual_loss'] for entry in self.energy_estimation_history]
            ax.plot(energy_values, 'g-', linewidth=1.5)
            ax.set_xlabel('Measurement')
            ax.set_ylabel('Loss/Energy')
            ax.set_title('Energy Estimation Progress')
            ax.set_yscale('log')
        else:
            ax.text(0.5, 0.5, 'No energy data available',
                    ha='center', va='center', transform=ax.transAxes)
        ax.grid(True, alpha=0.3)

        # 4. Circuit generation method distribution
        ax = axes[1, 1]
        methods = [entry.get('method', 'unknown') for entry in self.circuit_generation_history]
        method_counts = {}
        for method in methods:
            method_counts[method] = method_counts.get(method, 0) + 1

        if method_counts:
            ax.pie(method_counts.values(), labels=method_counts.keys(),
                autopct='%1.1f%%', startangle=90)
            ax.set_title('Circuit Generation Methods')
        else:
            ax.text(0.5, 0.5, 'No method data available',
                    ha='center', va='center', transform=ax.transAxes)

        plt.suptitle('Efficient GQE-GPT Optimization Summary', fontsize=14)
        plt.tight_layout()

        summary_path = os.path.join(save_path, 'efficient_optimization_summary.png')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()

        _logger.info(f"Efficient optimization summary saved: {summary_path}")

    def generate_detailed_report(self, save_path='results/'):
        """Generate detailed report (efficient version)"""
        os.makedirs(save_path, exist_ok=True)

        report_path = os.path.join(save_path, 'efficient_gqe_optimization_report.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Efficient GQE-GPT Quantum Circuit Optimization Report\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 1. Configuration information
            f.write("1. Configuration\n")
            f.write("-" * 40 + "\n")
            f.write(f"  - Number of Qubits: {self.n_qubits}\n")
            f.write(f"  - Optimization Method: Efficient Single-Round\n")
            f.write(f"  - Circuit Updates: {len(self.circuit_generation_history)}\n")
            f.write(f"  - Number of Objectives: 9\n")
            f.write(f"  - GPT Model: {'Enabled' if self.gpt_model is not None else 'Disabled'}\n\n")

            # 2. Efficiency statistics
            f.write("2. Efficiency Statistics\n")
            f.write("-" * 40 + "\n")
            f.write(f"  - Rounds per Generation: 0 (eliminated)\n")
            f.write(f"  - Average Candidates per Update: 20\n")
            f.write(f"  - Evaluation Strategy: Batch Processing\n")
            f.write(f"  - Cache Utilization: Enabled\n\n")

            # 3. Circuit evolution overview
            if self.circuit_generation_history:
                f.write("3. Circuit Evolution Summary\n")
                f.write("-" * 40 + "\n")

                initial_circuit = self.circuit_generation_history[0]['template']
                final_circuit = self.circuit_generation_history[-1]['template']

                f.write(f"  Initial Circuit:\n")
                f.write(f"    - Gates: {len(initial_circuit.gate_sequence)}\n")
                f.write(f"    - Parameters: {len(initial_circuit.parameter_map)}\n")

                f.write(f"  Final Circuit:\n")
                f.write(f"    - Gates: {len(final_circuit.gate_sequence)}\n")
                f.write(f"    - Parameters: {len(final_circuit.parameter_map)}\n\n")

            # 4. Performance improvement
            if hasattr(self, 'optimization_summary'):
                f.write("4. Performance Summary\n")
                f.write("-" * 40 + "\n")
                f.write(f"  - Total Evaluations: {self.optimization_summary.get('total_candidates_evaluated', 'N/A')}\n")
                f.write(f"  - Best Score: {self.optimization_summary.get('best_score', 'N/A')}\n")
                f.write(f"  - Best Energy Quality: {self.optimization_summary.get('best_energy_quality', 'N/A')}\n\n")

            # 5. Recommendations
            f.write("5. Recommendations\n")
            f.write("-" * 40 + "\n")
            f.write("  - Continue using efficient single-round generation\n")
            f.write("  - Monitor energy estimation accuracy\n")
            f.write("  - Consider increasing batch size if memory allows\n")
            f.write("  - Use dynamic circuit updates for long-running optimizations\n")

        _logger.info(f"Efficient optimization report saved: {report_path}")

        return report_path

#================================================
# Global variables and helper functions for parallel processing
#================================================
_quantum_device_pool = None
_pool_lock = threading.Lock()

def initialize_quantum_device_pool(n_devices, template, shots, noise_model=None):
    """Initialize quantum device pool"""
    global _quantum_device_pool
    with _pool_lock:
        if _quantum_device_pool is None:
            _quantum_device_pool = []
            for i in range(n_devices):
                device_params = (i, template, shots, noise_model)
                _quantum_device_pool.append(device_params)
    return _quantum_device_pool

class OptimizedQuantumDevice:
    """GQE optimized quantum device (for real hardware)"""

    def __init__(self, device_id, template, shots, noise_model=None):
        self.device_id = device_id
        self.template = template
        self.shots = shots
        self.noise_model = noise_model

        # Device configuration (real hardware optimization)
        if shots is not None:
            self.dev = qml.device('default.mixed', wires=template.n_qubits, shots=shots)
            self.diff_method = "best"
        else:
            self.dev = qml.device('lightning.qubit', wires=template.n_qubits)
            self.diff_method = "adjoint"

        self._create_optimized_circuit()

    def _apply_hardware_noise(self, wire):
        """Hardware-oriented noise model (improved based on QPINN literature)

        References:
        - Trahan et al. (2024) - Depolarizing channel noise for QPINNs
        - "Quantum Physics Informed Neural Networks" ICPP 2024 - Noise considerations
        """
        if self.noise_model is None:
            return

        # Noise parameters based on current NISQ devices
        # Reference: IBM Quantum Network typical error rates
        noise_rates = {
            'light': {'depolarizing': 0.001, 'amplitude_damping': 0.0005, 'phase_damping': 0.0},
            'realistic': {'depolarizing': 0.005, 'amplitude_damping': 0.002, 'phase_damping': 0.001},
            'heavy': {'depolarizing': 0.01, 'amplitude_damping': 0.005, 'phase_damping': 0.002}
        }

        rates = noise_rates.get(self.noise_model, noise_rates['realistic'])

        # Apply noise channels with scientifically grounded probabilities
        if rates['depolarizing'] > 0:
            qml.DepolarizingChannel(rates['depolarizing'], wires=wire)

        if rates['amplitude_damping'] > 0:
            qml.AmplitudeDamping(rates['amplitude_damping'], wires=wire)

        if rates['phase_damping'] > 0:
            qml.PhaseDamping(rates['phase_damping'], wires=wire)

    def _create_optimized_circuit(self):
        """GQE template-based optimized circuit (fixed measurement ordering)

        References:
        - Trahan et al. "Quantum Physics-Informed Neural Networks" Entropy 26(8):649 (2024)
        - TE-QPINN "Trainable embedding quantum physics informed neural networks" Sci Rep (2025)
        """
        @qml.qnode(self.dev, diff_method=self.diff_method)
        def optimized_circuit(inputs, params_array):
            # Input encoding (improved based on TE-QPINN approach)
            n_inputs = len(inputs)
            input_scaling = np.pi / 2  # Stable range for real hardware

            # Angle encoding as recommended in literature
            for i in range(min(self.template.n_qubits, n_inputs)):
                angle = inputs[i] * input_scaling
                qml.RY(angle, wires=i)

                # Apply hardware noise after state preparation
                if self.shots is not None:
                    self._apply_hardware_noise(i)

            # Initialize remaining qubits with small rotation
            for i in range(n_inputs, self.template.n_qubits):
                qml.RY(np.pi * 0.25, wires=i)

            # Template-based circuit execution
            param_idx = 0
            try:
                for gate_info in self.template.gate_sequence:
                    gate_type = gate_info['gate']
                    qubits = gate_info['qubits']
                    is_trainable = gate_info.get('trainable', False)
                    intensity = gate_info.get('intensity', 1.0)

                    # Validate qubit indices
                    if any(q >= self.template.n_qubits for q in qubits):
                        continue

                    # Apply gates based on type
                    if gate_type == 'H':
                        qml.Hadamard(wires=qubits[0])
                    elif gate_type == 'RX' and is_trainable:
                        if param_idx < len(params_array):
                            angle = params_array[param_idx] * intensity
                            qml.RX(angle, wires=qubits[0])
                            param_idx += 1
                    elif gate_type == 'RY' and is_trainable:
                        if param_idx < len(params_array):
                            angle = params_array[param_idx] * intensity
                            qml.RY(angle, wires=qubits[0])
                            param_idx += 1
                    elif gate_type == 'RZ' and is_trainable:
                        if param_idx < len(params_array):
                            angle = params_array[param_idx] * intensity
                            qml.RZ(angle, wires=qubits[0])
                            param_idx += 1
                    elif gate_type == 'CNOT' and len(qubits) >= 2:
                        if qubits[0] != qubits[1]:
                            qml.CNOT(wires=qubits[:2])
                    elif gate_type == 'CZ' and len(qubits) >= 2:
                        if qubits[0] != qubits[1]:
                            qml.CZ(wires=qubits[:2])
                    elif gate_type == 'SWAP' and len(qubits) >= 2:
                        if qubits[0] != qubits[1]:
                            qml.SWAP(wires=qubits[:2])

                    # Post-gate noise for hardware simulation
                    if self.shots is not None and is_trainable:
                        for q in qubits[:1]:
                            self._apply_hardware_noise(q)

            except Exception as e:
                _logger.warning(f"Warning during circuit execution: {e}")

            # Fixed measurement implementation based on literature
            # Return a single tensor of expectation values
            # Reference: Trahan et al. (2024) - proper measurement ordering
            measurements = []

            # Z-basis measurements (standard QPINN approach)
            for i in range(min(4, self.template.n_qubits)):
                measurements.append(qml.expval(qml.PauliZ(i)))

            # X-basis measurements for additional expressivity
            if self.template.n_qubits >= 2:
                measurements.append(qml.expval(qml.PauliX(0)))
                measurements.append(qml.expval(qml.PauliX(1)))

            # Two-qubit correlation measurements
            if self.template.n_qubits >= 2:
                measurements.append(qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)))
                if self.template.n_qubits >= 3:
                    measurements.append(qml.expval(qml.PauliZ(1) @ qml.PauliZ(2)))

            if len(measurements) == 1:
                return measurements[0]
            else:
                return tuple(measurements)

        self.circuit = optimized_circuit

    def execute(self, inputs, params):
        """Execute circuit"""
        return self.circuit(inputs, params)


# Global functions for parallel execution
def compute_final_output_parallel(device_id, inputs, params, L_val, T_val):
    """Generic final output computation for parallel processing (problem-independent version)

    References:
    - Trahan et al. "Quantum Physics-Informed Neural Networks" Entropy 26(8):649 (2024)
    - Panichi et al. "Quantum physics informed neural networks for multi-variable PDEs" arXiv:2503.12244 (2025)
    - "Trainable embedding quantum physics informed neural networks" Sci Rep (2025)

    Args:
        device_id: Device ID
        inputs: Input data (normalized coordinates)
        params: Parameter dictionary
        L_val: Spatial domain size
        T_val: Temporal domain size
    """
    try:
        # Get parameters
        output_scale = float(params['output_scale'])
        output_bias = float(params['output_bias'])
        time_decay = float(params['time_decay'])
        spatial_decay = float(params['spatial_decay'])
        amplitude = float(params['amplitude'])
        x_weight = float(params['x_weight'])
        correlation_weight = float(params['correlation_weight'])

        # Get feature weights (if available)
        spatial_feature_weights = params.get('spatial_feature_weights', None)
        temporal_feature_weights = params.get('temporal_feature_weights', None)
        temporal_frequencies = params.get('temporal_frequencies', None)

        # Get coordinates (already normalized)
        x_norm = inputs[0]
        y_norm = inputs[1]
        z_norm = inputs[2]
        t_norm = inputs[3]

        # Quantum circuit outputs (virtual values - computed separately in actual parallel processing)
        z_contribution = params.get('z_contribution', 0.0)
        x_contribution = params.get('x_contribution', 0.0)
        correlation_contribution = params.get('correlation_contribution', 0.0)

        # 1. Combine quantum circuit outputs
        raw_output = (z_contribution +
                        x_weight * x_contribution +
                        correlation_weight * correlation_contribution)

        # 2. Scaling
        scaled_output = output_scale * raw_output

        # 3. Activation function transformation (tanh recommended)
        activated_output = np.tanh(scaled_output)

        # 4. Spatial feature computation (parallel version)
        spatial_features = _compute_spatial_features_parallel(
            x_norm, y_norm, z_norm, spatial_feature_weights
        )
        spatial_modulation = 1.0 + spatial_decay * spatial_features

        # 5. Temporal feature computation (parallel version)
        temporal_features = _compute_temporal_features_parallel(
            t_norm, temporal_feature_weights, temporal_frequencies
        )
        temporal_modulation = 1.0 + time_decay * temporal_features

        # 6. Feature combination
        # Reference: Multiplicative combination preserves physical properties
        result = (amplitude * activated_output *
                spatial_modulation * temporal_modulation +
                output_bias)

        # NaN/Inf check
        if np.isnan(result) or np.isinf(result):
            result = output_bias

        if abs(result) > 1e6:
                _logger.warning(f"Large output value {result:.3f} at normalized({x_norm:.3f}, {y_norm:.3f}, {z_norm:.3f}, {t_norm:.3f})")

        return result

    except Exception as e:
        _logger.error(f"Parallel output computation error (Device {device_id}): {e}")
        return params.get('output_bias', 0.01)


def _compute_spatial_features_parallel(x_norm, y_norm, z_norm, spatial_weights):
    """Spatial feature computation for parallel processing (problem-independent)"""
    features = []

    # 1st order features
    features.extend([x_norm, y_norm, z_norm])

    # 2nd order features (interaction terms)
    features.extend([
        x_norm * y_norm,
        y_norm * z_norm,
        z_norm * x_norm
    ])

    # 3rd order features (nonlinearity)
    features.extend([
        x_norm**2 + y_norm**2 + z_norm**2,  # Distance-like feature
        x_norm * y_norm * z_norm  # 3-way interaction
    ])

    # Weighted combination
    if spatial_weights is not None:
        weighted_sum = sum(w * f for w, f in zip(spatial_weights, features))
        return np.tanh(weighted_sum)
    else:
        return np.tanh(np.mean(features))


def _compute_temporal_features_parallel(t_norm, temporal_weights, frequencies):
    """Temporal feature computation for parallel processing (problem-independent, learnable frequencies)"""
    features = []

    # Polynomial basis
    features.extend([t_norm, t_norm**2, t_norm**3])

    # Fourier basis
    if frequencies is not None:
        # Constrain frequencies to positive values
        freq_values = np.abs(frequencies)
        for freq in freq_values:
            features.append(np.sin(2 * np.pi * freq * t_norm))
            features.append(np.cos(2 * np.pi * freq * t_norm))
    else:
        # Default frequencies
        for freq in [1.0, 2.0, 4.0]:
            features.append(np.sin(2 * np.pi * freq * t_norm))
            features.append(np.cos(2 * np.pi * freq * t_norm))

    # Exponential basis
    features.append(np.exp(-t_norm))
    features.append(1.0 - np.exp(-t_norm))

    # Weighted combination
    if temporal_weights is not None and len(temporal_weights) == len(features):
        weighted_sum = sum(w * f for w, f in zip(temporal_weights, features))
        return np.tanh(weighted_sum)
    else:
        return np.tanh(np.mean(features))


def compute_distance_function(x, y, z, L_val, epsilon):
    """Compute smooth distance function to boundaries

    Based on Lu et al. "Physics-informed neural networks with hard constraints"
    Returns a smooth multiplicative factor that is 0 on the boundary and ~1 in the interior.

    Args:
        x, y, z: Spatial coordinates
        L_val: Domain size
        epsilon: Boundary layer thickness (controls smoothness)

    Returns:
        distance: Smooth distance function value
    """
    # Compute minimum distance to each boundary face
    dist_x_min = x
    dist_x_max = L_val - x
    dist_y_min = y
    dist_y_max = L_val - y
    dist_z_min = z
    dist_z_max = L_val - z

    # Minimum distance to any boundary
    min_dist = min(dist_x_min, dist_x_max, dist_y_min, dist_y_max, dist_z_min, dist_z_max)

    # Smooth transition using tanh
    # tanh(d/epsilon) gives 0 at boundary (d=0) and approaches 1 in interior
    distance = np.tanh(min_dist / epsilon)

    return distance


# Also modified parallel_forward_batch_gqe function
def parallel_forward_batch_gqe(args):
    """Parallel batch processing (GQE version with fixed measurements)"""
    device_params, batch, param_dict = args

    device_id = device_params[0]
    circuit_template = device_params[1]
    shots = device_params[2]
    noise_model = device_params[3]

    # Create device with fixed circuit
    qdevice = OptimizedQuantumDevice(device_id, circuit_template, shots, noise_model)
    results = []

    # Extend parameter dictionary
    extended_params = param_dict.copy()

    # Add feature weights if available
    if hasattr(circuit_template, 'spatial_feature_weights'):
        extended_params['spatial_feature_weights'] = circuit_template.spatial_feature_weights.numpy()

    if hasattr(circuit_template, 'temporal_feature_weights'):
        extended_params['temporal_feature_weights'] = circuit_template.temporal_feature_weights.numpy()

    if hasattr(circuit_template, 'temporal_frequencies'):
        extended_params['temporal_frequencies'] = circuit_template.temporal_frequencies.numpy()

    # Get boundary epsilon for hard constraints
    boundary_epsilon = param_dict.get('boundary_epsilon', 0.1)
    use_hard_constraints = param_dict.get('use_hard_constraints', True)

    for point in batch:
        try:
            # Coordinate normalization
            x_norm = point.x / L
            y_norm = point.y / L
            z_norm = point.z / L
            t_norm = point.t / T if T > 0 else point.t

            inputs = np.array([x_norm, y_norm, z_norm, t_norm])

            # Execute quantum circuit with fixed measurement handling
            measurements = qdevice.circuit(inputs, param_dict['circuit_params'])

            # Process measurement results safely
            if hasattr(qdevice, 'n_qubits'):
                measurements_array = _process_measurements_safe(measurements)
            else:
                # Fallback processing
                if hasattr(measurements, 'numpy'):
                    measurements_array = measurements.numpy().flatten()
                else:
                    measurements_array = np.array(measurements).flatten()

            n_measurements = len(measurements_array)

            # Compute each component
            z_contribution = _compute_z_contribution_parallel(
                measurements_array, n_measurements, point.t, param_dict
            )
            x_contribution = _compute_x_contribution_parallel(
                measurements_array, n_measurements, param_dict
            )
            correlation_contribution = _compute_correlation_contribution_parallel(
                measurements_array, n_measurements, param_dict
            )

            # Update computation parameters
            extended_params['z_contribution'] = z_contribution
            extended_params['x_contribution'] = x_contribution
            extended_params['correlation_contribution'] = correlation_contribution

            # Generic output computation
            network_output = compute_final_output_parallel(
                device_id, inputs, extended_params, L, T
            )

            # Apply hard boundary constraints
            if use_hard_constraints:
                # Compute distance function
                distance = compute_distance_function(point.x, point.y, point.z, L, boundary_epsilon)

                # Get boundary condition value (g_vec)
                g_vec = boundary_condition(point.x, point.y, point.z, point.t)

                # Apply hard constraint: u = g + distance * network_output
                constrained_output = g_vec + distance * network_output
                results.append(constrained_output)
            else:
                results.append(network_output)

        except Exception as e:
            _logger.error(f"Parallel processing error (Device {device_id}): {e}")
            # Improved fallback
            analytical_val = 0.1 * initial_condition(point.x, point.y, point.z)
            fallback_value = extended_params.get('output_bias', analytical_val)
            results.append(fallback_value)

    return results


# Also modified helper functions
def _compute_z_contribution_parallel(measurements_array, n_measurements, t, params):
    """Z-basis measurement computation for parallel processing"""
    try:
        if n_measurements >= 4:
            z_measurements = measurements_array[:4]
            # Time-dependent weights (generalized)
            base_weights = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
            time_modulation = 1.0 + 0.5 * np.sin(t * np.pi / T)
            z_weights = base_weights * time_modulation

            z_contribution = np.sum(z_measurements * z_weights)
            if np.isnan(z_contribution) or np.isinf(z_contribution):
                z_contribution = 0.0
            return z_contribution
        elif n_measurements >= 2:
            return np.mean(measurements_array[:2])
        elif n_measurements > 0:
            return measurements_array[0]
        else:
            return 0.0
    except Exception:
        return 0.0


def _compute_x_contribution_parallel(measurements_array, n_measurements, params):
    """X-basis measurement computation for parallel processing"""
    try:
        if n_measurements > 4:
            x_measurements = measurements_array[4:6]
            x_mean = np.mean(x_measurements)
            if np.isnan(x_mean) or np.isinf(x_mean):
                return 0.0
            return float(params['x_weight']) * x_mean
        return 0.0
    except Exception:
        return 0.0


def _compute_correlation_contribution_parallel(measurements_array, n_measurements, params):
    """Correlation measurement computation for parallel processing"""
    try:
        if n_measurements > 6:
            correlations = measurements_array[6:]
            corr_mean = np.mean(correlations)
            if np.isnan(corr_mean) or np.isinf(corr_mean):
                return 0.0
            return float(params['correlation_weight']) * corr_mean
        return 0.0
    except Exception:
        return 0.0

def _process_measurements_safe(raw_measurements):
    """Safe processing of measurement results (common to parallel and sequential versions)"""
    try:
        if raw_measurements is None:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        if isinstance(raw_measurements, (int, float, np.integer, np.floating)):
            return np.array([float(raw_measurements)], dtype=np.float64)

        if hasattr(raw_measurements, '__array__'):
            try:
                arr = np.asarray(raw_measurements, dtype=np.float64)

                if arr.ndim == 0:
                    return np.array([float(arr.item())], dtype=np.float64)
                elif arr.size == 0:
                    return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
                else:
                    return arr.flatten()
            except:
                pass

        if isinstance(raw_measurements, tuple):
            measurements_array = np.array([
                float(m.numpy()) if hasattr(m, 'numpy') else float(m)
                for m in raw_measurements
            ])
            return measurements_array

        if hasattr(raw_measurements, '__iter__'):
            try:
                measurements_list = []
                for item in raw_measurements:
                    if hasattr(item, 'item'):
                        measurements_list.append(float(item.item()))
                    elif isinstance(item, (int, float, np.integer, np.floating)):
                        measurements_list.append(float(item))
                    else:
                        measurements_list.append(0.0)

                if len(measurements_list) == 0:
                    return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

                return np.array(measurements_list, dtype=np.float64)

            except Exception:
                return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        try:
            val = float(raw_measurements)
            return np.array([val], dtype=np.float64)
        except:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    except Exception:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

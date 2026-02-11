"""
config.py - Global configuration module for PINNs/QPINNs benchmark.

Compares classical and quantum Physics-Informed Neural Networks
for the 3D heat equation.
"""

import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pennylane as qml
import time
from typing import Tuple, List, Callable, Union, Any, Dict, Optional
import os
os.environ['OMP_NUM_THREADS'] = str(12)
from collections import deque
import warnings
from multiprocessing import Pool, cpu_count, Manager, Process
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import functools
from dataclasses import dataclass, field
import pickle
import math
import threading
from queue import Queue
import psutil
import copy
import json
from itertools import product

# Additional imports for GPT model
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# New class for adding Bayesian optimization
try:
    from botorch.models import SingleTaskGP, ModelListGP
    from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
    from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement
    from botorch.optim import optimize_acqf
    from botorch.sampling import SobolQMCNormalSampler
    from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
    from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
    from botorch.utils.sampling import sample_simplex
    from botorch.fit import fit_gpytorch_mll
    from botorch.models.transforms.input import Normalize
    from botorch.models.transforms.outcome import Standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood
    print("BoTorch/GPyTorch available")
    BOTORCH_AVAILABLE = True
except ImportError:
    print("BoTorch/GPyTorch not available. Bayesian optimization will be disabled.")
    BOTORCH_AVAILABLE = False

# Try importing Braket plugin
try:
    import boto3
    from braket.aws import AwsDevice, AwsSession
    import braket.pennylane_plugin
    BRAKET_AVAILABLE = True
except ImportError:
    BRAKET_AVAILABLE = False
    warnings.warn(
        "Amazon Braket PennyLane plugin not installed. "
        "Install with: pip install amazon-braket-pennylane-plugin"
    )

# ================================================
# Suppress warnings and set numeric options
# ================================================
warnings.filterwarnings("ignore", category=UserWarning)

np.set_printoptions(precision=8)
try:
    qml.numpy.set_printoptions(precision=8)
except AttributeError:
    pass

torch.set_default_dtype(torch.float32)

_logger = logging.getLogger('benchmark.config')

# ================================================
# Problem parameters
# ================================================
alpha = 0.01      # Thermal diffusivity
L = 1.0           # Length of cube side
T = 1.0           # Final time
sigma_0 = 0.05    # Gaussian parameter

# Discretization parameters
nx, ny, nz = 20, 20, 20   # Spatial divisions
nt = 20                    # Time divisions

# Training parameters
pinn_epochs = 3000         # PINN epochs (increased for accuracy)
qnn_epochs = 200           # QPINN epochs (increased from 100 for SPSA)

# Parallel processing parameters
N_PARALLEL_DEVICES = min(4, cpu_count() // 2)
USE_PARALLEL_TRAINING = True


# ================================================
# Backend configuration
# ================================================
class BackendConfig:
    """Unified backend configuration for CPU/GPU/QPU"""

    @staticmethod
    def get_torch_device(backend='auto'):
        """Get PyTorch device based on backend selection"""
        if backend == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            return torch.device('cpu')
        elif backend in ('cuda', 'gpu'):
            if torch.cuda.is_available():
                return torch.device('cuda')
            _logger.warning("CUDA not available, falling back to CPU")
            return torch.device('cpu')
        elif backend == 'cpu':
            return torch.device('cpu')
        else:
            return torch.device('cpu')

    @staticmethod
    def get_quantum_device(backend='auto', n_qubits=6, shots=None, noise_model=None):
        """Get PennyLane quantum device based on backend selection"""
        import pennylane as qml

        if backend == 'qpu':
            # QPU will be handled by QuantumDeviceManager
            return None  # Caller should use QuantumDeviceManager

        # Try lightning.gpu if GPU backend requested
        if backend in ('cuda', 'gpu'):
            try:
                if shots is not None:
                    return qml.device('lightning.gpu', wires=n_qubits, shots=shots)
                else:
                    return qml.device('lightning.gpu', wires=n_qubits)
            except Exception:
                _logger.warning("lightning.gpu not available, falling back to lightning.qubit")

        # For noisy simulation, use default.mixed
        if noise_model is not None:
            if shots is not None:
                return qml.device('default.mixed', wires=n_qubits, shots=shots)
            else:
                return qml.device('default.mixed', wires=n_qubits)

        # Default: lightning.qubit (fast CPU simulator)
        if shots is not None:
            return qml.device('lightning.qubit', wires=n_qubits, shots=shots)
        else:
            return qml.device('lightning.qubit', wires=n_qubits)

    @staticmethod
    def print_config(backend='auto'):
        """Log current backend configuration"""
        import pennylane as qml
        torch_device = BackendConfig.get_torch_device(backend)
        _logger.info(f"Backend: {backend}")
        _logger.info(f"PyTorch device: {torch_device}")
        _logger.info(f"PyTorch version: {torch.__version__}")
        _logger.info(f"PennyLane version: {qml.__version__}")
        if torch.cuda.is_available():
            _logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            _logger.info("CUDA: not available")


# ================================================
# Training configuration
# ================================================
@dataclass
class TrainingConfig:
    """Training configuration replacing NSGA2_COMMON_CONFIG"""
    # PINN (RAdam) settings
    pinn_lr: float = 1e-3
    pinn_weight_decay: float = 1e-2
    pinn_epochs: int = 3000
    pinn_n_samples: int = 100000

    # QPINN (SPSA) settings
    qpinn_max_iterations: int = 200
    qpinn_spsa_c: float = 0.2
    qpinn_spsa_a: float = None  # Auto-computed
    qpinn_n_samples: int = 1500

    # Common
    progress_interval: int = 20
    random_seed: int = 42


# ================================================
# JSON config loading
# ================================================
def get_nested(config: dict, dotted_key: str, default=None):
    """Safely access nested dict values using dotted keys.

    Example: get_nested(config, 'pinn.training.lr', 1e-3)
    """
    keys = dotted_key.split('.')
    val = config
    for k in keys:
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            return default
    return val


def load_benchmark_config(path: str) -> dict:
    """Load benchmark configuration from a JSON file.

    Args:
        path: Path to the JSON config file.

    Returns:
        Configuration dictionary.
    """
    with open(path, 'r') as f:
        config = json.load(f)
    return config


def apply_config(config: dict):
    """Apply loaded JSON config to global variables.

    Overrides module-level globals (alpha, L, T, sigma_0, nx, ny, nz, nt,
    pinn_epochs, qnn_epochs, N_PARALLEL_DEVICES, USE_PARALLEL_TRAINING)
    from the config dict.
    """
    global alpha, L, T, sigma_0
    global nx, ny, nz, nt
    global pinn_epochs, qnn_epochs
    global N_PARALLEL_DEVICES, USE_PARALLEL_TRAINING

    # Physics
    alpha = get_nested(config, 'physics.alpha', alpha)
    L = get_nested(config, 'physics.L', L)
    T = get_nested(config, 'physics.T', T)
    sigma_0 = get_nested(config, 'physics.sigma_0', sigma_0)

    # Grid
    nx = get_nested(config, 'grid.nx', nx)
    ny = get_nested(config, 'grid.ny', ny)
    nz = get_nested(config, 'grid.nz', nz)
    nt = get_nested(config, 'grid.nt', nt)

    # Training epochs
    pinn_epochs = get_nested(config, 'pinn.training.epochs', pinn_epochs)
    qnn_epochs = get_nested(config, 'qpinn.training.max_iterations', qnn_epochs)

    # Parallel
    max_parallel = get_nested(config, 'parallel.n_parallel_devices', 4)
    N_PARALLEL_DEVICES = min(max_parallel, cpu_count() // 2)
    USE_PARALLEL_TRAINING = get_nested(config, 'parallel.use_parallel_training', USE_PARALLEL_TRAINING)

    # OMP threads
    omp_threads = get_nested(config, 'parallel.omp_num_threads', 12)
    os.environ['OMP_NUM_THREADS'] = str(omp_threads)


# ================================================
# Global device
# ================================================
device = BackendConfig.get_torch_device()

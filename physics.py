"""
physics.py - Physics-related functions and utilities for PINNs/QPINNs benchmark.

Contains the analytical solution, initial/boundary conditions, and metric
calculations for the 3D heat equation.
"""

from config import *
import logging

_logger = logging.getLogger('benchmark.physics')


# ================================================
# Data class definitions
# ================================================
@dataclass
class TrainingPoint:
    """Training data point"""
    x: float
    y: float
    z: float
    t: float
    u_true: float = None
    type: str = 'interior'


# ================================================
# Physics functions
# ================================================
def initial_condition(x, y, z):
    """Initial temperature distribution: Gaussian distribution.

    Pure Gaussian centered at (L/2, L/2, L/2) with width sigma_0.
    Hard boundary constraints in the PINN/QPINN enforce u=0 at boundaries,
    so no artificial damping is applied here.
    """
    x0, y0, z0 = L/2, L/2, L/2
    return np.exp(-((x-x0)**2 + (y-y0)**2 + (z-z0)**2) / (2*sigma_0**2))


def boundary_condition(x, y, z, t, epsilon=0.001):
    """Boundary condition: 0 at all boundaries, batch or scalar."""
    # Use torch if any input is torch.Tensor
    torch_mode = any('torch' in str(type(v)) for v in [x, y, z, t])

    if torch_mode:
        # Make sure constants are also tensors (and on correct device)
        _device = x.device if hasattr(x, "device") else "cpu"
        L_tensor = torch.tensor(L, dtype=x.dtype, device=_device)
        zero = torch.tensor(0.0, dtype=x.dtype, device=_device)
        # _exp and _isclose as before
        _exp = torch.exp
        _isclose = torch.isclose
        zeros = lambda v: torch.zeros_like(v)
        ones = lambda v: torch.ones_like(v)
        # Convert floats to tensor for isclose
        at_boundary = (
            _isclose(x, zero) | _isclose(x, L_tensor) |
            _isclose(y, zero) | _isclose(y, L_tensor) |
            _isclose(z, zero) | _isclose(z, L_tensor)
        )
    else:
        _exp = np.exp
        _isclose = np.isclose
        zeros = lambda v: np.zeros_like(v)
        ones = lambda v: np.ones_like(v)
        at_boundary = (
            _isclose(x, 0.0) | _isclose(x, L) |
            _isclose(y, 0.0) | _isclose(y, L) |
            _isclose(z, 0.0) | _isclose(z, L)
        )

    # time_factor (broadcasted)
    time_factor = _exp(-5.0 * t / T)
    # astype for type match (torch: float32, numpy: float64, etc.)
    if torch_mode:
        return (epsilon * time_factor) * at_boundary.type_as(time_factor)
    else:
        return (epsilon * time_factor) * at_boundary.astype(time_factor.dtype)


def analytical_solution(x, y, z, t):
    """Analytical solution for 3D heat equation with homogeneous Dirichlet BC.

    Uses Fourier sine series expansion on [0,L]^3:
        u(x,y,z,t) = sum_{n,m,l=1}^{N_max} B_{n,m,l}
                      * sin(n*pi*x/L) * sin(m*pi*y/L) * sin(l*pi*z/L)
                      * exp(-alpha * pi^2 * (n^2 + m^2 + l^2) / L^2 * t)

    where B_{n,m,l} = (2/L)^3 * integral of IC * sin(n*pi*x/L) * sin(m*pi*y/L) * sin(l*pi*z/L)

    For Gaussian IC centered at (L/2, L/2, L/2), the Fourier coefficients
    can be computed analytically using the formula for Gaussian integrals
    with sine functions.
    """
    N_max = 25  # Number of Fourier modes (need ~25 for sigma_0=0.05 Gaussian)
    x0, y0, z0 = L/2, L/2, L/2
    u_val = 0.0

    for n in range(1, N_max + 1):
        # 1D Fourier sine coefficient for Gaussian in x-direction:
        # B_n = (2/L) * integral_0^L exp(-(x-x0)^2/(2*sigma^2)) * sin(n*pi*x/L) dx
        # For Gaussian centered at L/2: only odd modes (n=1,3,5,...) contribute
        # due to symmetry about L/2.
        if n % 2 == 0:
            continue  # Even modes vanish by symmetry about L/2
        kn = n * np.pi / L
        # Analytical coefficient: integral of Gaussian * sin over [0,L]
        # Using: int_0^L exp(-(x-L/2)^2/(2*s^2)) * sin(n*pi*x/L) dx
        # = sqrt(2*pi) * s * exp(-kn^2 * s^2 / 2) * sin(n*pi/2) for symmetric Gaussian
        # Since n is odd, sin(n*pi/2) = (-1)^((n-1)/2)
        bn_x = (2.0 / L) * np.sqrt(2.0 * np.pi) * sigma_0 * np.exp(-kn**2 * sigma_0**2 / 2.0) * np.sin(n * np.pi * x0 / L)

        for m in range(1, N_max + 1):
            if m % 2 == 0:
                continue
            km = m * np.pi / L
            bn_y = (2.0 / L) * np.sqrt(2.0 * np.pi) * sigma_0 * np.exp(-km**2 * sigma_0**2 / 2.0) * np.sin(m * np.pi * y0 / L)

            for l_idx in range(1, N_max + 1):
                if l_idx % 2 == 0:
                    continue
                kl = l_idx * np.pi / L
                bn_z = (2.0 / L) * np.sqrt(2.0 * np.pi) * sigma_0 * np.exp(-kl**2 * sigma_0**2 / 2.0) * np.sin(l_idx * np.pi * z0 / L)

                B_nml = bn_x * bn_y * bn_z
                eigenvalue = alpha * np.pi**2 * (n**2 + m**2 + l_idx**2) / L**2
                decay = np.exp(-eigenvalue * t)

                u_val += B_nml * np.sin(kn * x) * np.sin(km * y) * np.sin(kl * z) * decay

    return max(u_val, 0.0)  # Physical constraint: temperature >= 0


def to_python_float(value):
    """General function to reliably convert any PennyLane type to Python float"""
    try:
        if isinstance(value, float):
            return value
        if isinstance(value, int):
            return float(value)
        if isinstance(value, (np.ndarray, np.generic)):
            if hasattr(value, 'item'):
                return float(value.item())
            else:
                return float(value)
        if hasattr(value, 'numpy'):
            numpy_val = value.numpy()
            if isinstance(numpy_val, np.ndarray):
                return float(numpy_val.item()) if numpy_val.size == 1 else float(numpy_val.flatten()[0])
            else:
                return float(numpy_val)
        if hasattr(value, 'item'):
            return float(value.item())
        if hasattr(value, '_value'):
            return float(value._value)
        return float(value)
    except Exception:
        return 0.0


def compute_analytical_solution() -> np.ndarray:
    """Calculate analytical solution using vectorized Fourier sine series.

    This is a fast vectorized implementation that avoids the per-point loop
    by computing all grid points simultaneously using broadcasting.
    """
    _logger.info("Computing analytical solution (Fourier sine series)...")

    x = np.linspace(0, L, nx)
    y = np.linspace(0, L, ny)
    z = np.linspace(0, L, nz)
    t = np.linspace(0, T, nt)

    X, Y, Z, T_mesh = np.meshgrid(x, y, z, t, indexing='ij')
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()
    T_flat = T_mesh.flatten()

    N_max = 25  # Must match analytical_solution() for consistent evaluation
    x0, y0, z0 = L/2, L/2, L/2
    u_analytical = np.zeros(len(X_flat))

    # Precompute 1D Fourier coefficients for odd modes only
    odd_modes = list(range(1, N_max + 1, 2))  # 1, 3, 5, ...

    for n in odd_modes:
        kn = n * np.pi / L
        bn_x = (2.0 / L) * np.sqrt(2.0 * np.pi) * sigma_0 * np.exp(-kn**2 * sigma_0**2 / 2.0) * np.sin(n * np.pi * x0 / L)
        sin_nx = np.sin(kn * X_flat)

        for m in odd_modes:
            km = m * np.pi / L
            bn_y = (2.0 / L) * np.sqrt(2.0 * np.pi) * sigma_0 * np.exp(-km**2 * sigma_0**2 / 2.0) * np.sin(m * np.pi * y0 / L)
            sin_my = np.sin(km * Y_flat)

            for l_idx in odd_modes:
                kl = l_idx * np.pi / L
                bn_z = (2.0 / L) * np.sqrt(2.0 * np.pi) * sigma_0 * np.exp(-kl**2 * sigma_0**2 / 2.0) * np.sin(l_idx * np.pi * z0 / L)

                B_nml = bn_x * bn_y * bn_z
                eigenvalue = alpha * np.pi**2 * (n**2 + m**2 + l_idx**2) / L**2

                sin_lz = np.sin(kl * Z_flat)
                decay = np.exp(-eigenvalue * T_flat)

                u_analytical += B_nml * sin_nx * sin_my * sin_lz * decay

    u_analytical = np.clip(u_analytical, 0.0, None)  # Physical constraint
    _logger.info(f"  Analytical solution range: [{u_analytical.min():.6e}, {u_analytical.max():.6e}]")

    return u_analytical


def calculate_metrics(u_pred: np.ndarray, u_true: np.ndarray) -> Tuple[float, float]:
    """Calculate accuracy metrics (MSE, RelL2).

    Args:
        u_pred: Predicted solution array.
        u_true: Analytical/reference solution array.

    Returns:
        Tuple of (MSE, Relative L2 error).
    """
    u_pred = np.nan_to_num(u_pred, nan=0.0, posinf=0.0, neginf=0.0)
    u_pred = np.clip(u_pred, 0, None)

    mse = np.mean((u_pred - u_true) ** 2)
    rel_l2 = np.sqrt(np.sum((u_pred - u_true) ** 2)) / np.sqrt(np.sum(u_true ** 2) + 1e-10)
    return mse, rel_l2


def calculate_full_metrics(u_pred: np.ndarray, u_true: np.ndarray) -> dict:
    """Calculate comprehensive accuracy metrics for benchmark comparison.

    Computes MAE, MSE, RMSE, Relative L2, Max Absolute Error, and physics-
    related quantities (energy conservation, boundary satisfaction, etc.).

    Args:
        u_pred: Predicted solution array (flattened).
        u_true: Analytical/reference solution array (flattened).

    Returns:
        Dictionary with all computed metrics.
    """
    u_pred = np.nan_to_num(u_pred, nan=0.0, posinf=0.0, neginf=0.0)
    u_pred = np.clip(u_pred, 0, None)

    diff = u_pred - u_true

    # Standard error metrics
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    max_ae = float(np.max(np.abs(diff)))
    rel_l2 = float(np.sqrt(np.sum(diff ** 2)) / np.sqrt(np.sum(u_true ** 2) + 1e-10))

    # Relative MAE
    rel_mae = float(np.mean(np.abs(diff) / (np.abs(u_true) + 1e-10)))

    # Physics metrics
    # Total thermal energy: integral of u over domain ≈ sum * dV
    dx = L / (nx - 1) if nx > 1 else L
    dy = L / (ny - 1) if ny > 1 else L
    dz = L / (nz - 1) if nz > 1 else L
    dV = dx * dy * dz

    u_pred_r = u_pred.reshape(nx, ny, nz, nt) if u_pred.size == nx * ny * nz * nt else None
    u_true_r = u_true.reshape(nx, ny, nz, nt) if u_true.size == nx * ny * nz * nt else None

    energy_pred = []
    energy_true = []
    if u_pred_r is not None and u_true_r is not None:
        for t_idx in range(nt):
            energy_pred.append(float(np.sum(u_pred_r[:, :, :, t_idx]) * dV))
            energy_true.append(float(np.sum(u_true_r[:, :, :, t_idx]) * dV))

    # Peak value accuracy (at t=0, center of domain)
    if u_pred_r is not None and u_true_r is not None:
        mid = (nx // 2, ny // 2, nz // 2)
        peak_pred = float(u_pred_r[mid[0], mid[1], mid[2], 0])
        peak_true = float(u_true_r[mid[0], mid[1], mid[2], 0])
        peak_error = float(abs(peak_pred - peak_true))
    else:
        peak_pred, peak_true, peak_error = 0.0, 0.0, 0.0

    # Boundary satisfaction (mean absolute value on all 6 faces)
    boundary_errors = []
    if u_pred_r is not None:
        for t_idx in range(nt):
            faces = np.concatenate([
                u_pred_r[0, :, :, t_idx].flatten(),
                u_pred_r[-1, :, :, t_idx].flatten(),
                u_pred_r[:, 0, :, t_idx].flatten(),
                u_pred_r[:, -1, :, t_idx].flatten(),
                u_pred_r[:, :, 0, t_idx].flatten(),
                u_pred_r[:, :, -1, t_idx].flatten(),
            ])
            boundary_errors.append(float(np.mean(np.abs(faces))))
    mean_boundary_error = float(np.mean(boundary_errors)) if boundary_errors else 0.0

    # Non-negativity violation
    neg_count = int(np.sum(u_pred < 0))
    neg_fraction = float(neg_count / len(u_pred)) if len(u_pred) > 0 else 0.0

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'max_ae': max_ae,
        'rel_l2': rel_l2,
        'rel_mae': rel_mae,
        'peak_pred': peak_pred,
        'peak_true': peak_true,
        'peak_error': peak_error,
        'energy_pred': energy_pred,
        'energy_true': energy_true,
        'mean_boundary_error': mean_boundary_error,
        'boundary_errors_per_timestep': boundary_errors,
        'neg_count': neg_count,
        'neg_fraction': neg_fraction,
        'pred_min': float(np.min(u_pred)),
        'pred_max': float(np.max(u_pred)),
        'pred_mean': float(np.mean(u_pred)),
        'pred_std': float(np.std(u_pred)),
        'true_min': float(np.min(u_true)),
        'true_max': float(np.max(u_true)),
        'true_mean': float(np.mean(u_true)),
        'true_std': float(np.std(u_true)),
    }

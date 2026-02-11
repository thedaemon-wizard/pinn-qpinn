"""Main entry point for PINNs vs QPINNs Benchmark Comparison

Compares classical PINNs (with RAdam optimizer + ReLoBRaLo) against quantum PINNs
(with SPSA optimizer + ReLoBRaLo) for the 3D heat equation.

Outputs:
    - PNG plots: temperature fields, loss curves, ReLoBRaLo weights, profiles, error maps
    - CSV files: training loss history, per-component losses, metrics over time
    - JSON files: comparative analysis, training configuration, detailed results
    - TXT files: benchmark summary report
    - Log files: full training log with timestamps
    - PTH files: model checkpoints

Usage:
    python main.py [--backend auto|cpu|cuda|qpu]
"""
import argparse
import logging
import traceback
from config import *
from config import load_benchmark_config, get_nested, apply_config
from physics import (
    TrainingPoint, initial_condition, boundary_condition,
    analytical_solution, compute_analytical_solution, calculate_metrics,
    calculate_full_metrics
)


# ================================================
# Logging setup
# ================================================
def setup_logging(results_dir: str) -> logging.Logger:
    """Configure logging to both file and console"""
    logger = logging.getLogger('benchmark')
    logger.setLevel(logging.DEBUG)

    # File handler (detailed)
    log_path = os.path.join(results_dir, 'benchmark.log')
    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(fh)

    # Console handler (info level)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)

    return logger


# ================================================
# Checkpoint functions
# ================================================
def save_checkpoint(checkpoint_data: dict, filepath: str, logger=None):
    """Save checkpoint data to file"""
    msg = f"Saving checkpoint to {filepath}..."
    if logger:
        logger.info(msg)
    else:
        print(msg)
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint_data, filepath)
        msg = f"Checkpoint saved successfully: {filepath}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
        return True
    except Exception as e:
        msg = f"Error saving checkpoint: {str(e)}"
        if logger:
            logger.error(msg)
        else:
            print(msg)
        return False


def load_checkpoint(filepath: str, map_device=None, logger=None) -> Optional[dict]:
    """Load checkpoint data from file"""
    if not os.path.exists(filepath):
        return None

    if map_device is None:
        map_device = device

    msg = f"Loading checkpoint from {filepath}..."
    if logger:
        logger.info(msg)
    else:
        print(msg)
    try:
        checkpoint = torch.load(filepath, map_location=map_device, weights_only=False)
        msg = "Checkpoint loaded successfully"
        if logger:
            logger.info(msg)
        else:
            print(msg)
        return checkpoint
    except Exception as e:
        msg = f"Error loading checkpoint: {str(e)}"
        if logger:
            logger.error(msg)
        else:
            print(msg)
        return None


def check_existing_checkpoints(results_dir: str) -> Tuple[bool, bool]:
    """Check if checkpoint files exist"""
    pinn_checkpoint = os.path.join(results_dir, 'pinn_radam_checkpoint.pth')
    qpinn_checkpoint = os.path.join(results_dir, 'qpinn_spsa_checkpoint.pth')
    return os.path.exists(pinn_checkpoint), os.path.exists(qpinn_checkpoint)


# ================================================
# CSV Output
# ================================================
def save_pinn_training_csv(results_dir: str, pinn_losses, training_history=None):
    """Save PINN training loss history and per-component losses to CSV"""
    # Total loss history
    csv_path = os.path.join(results_dir, 'pinn_training_losses.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Total_Loss'])
        for i, loss in enumerate(pinn_losses):
            writer.writerow([i + 1, loss])

    # Per-component loss history with ReLoBRaLo weights
    if training_history and 'loss_components' in training_history:
        csv_path2 = os.path.join(results_dir, 'pinn_loss_components.csv')
        components = training_history['loss_components']
        weights = training_history['relobralo_weights']
        lr_hist = training_history.get('lr_history', [])
        keys = list(components.keys())
        n_epochs = len(components[keys[0]])

        with open(csv_path2, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['Epoch']
            for k in keys:
                header.extend([f'Loss_{k}', f'Weight_{k}'])
            header.append('Learning_Rate')
            writer.writerow(header)

            for i in range(n_epochs):
                row = [i + 1]
                for k in keys:
                    row.append(components[k][i])
                    row.append(weights[k][i])
                row.append(lr_hist[i] if i < len(lr_hist) else '')
                writer.writerow(row)

    return csv_path


def save_qpinn_training_csv(results_dir: str, qnn_losses, training_history=None):
    """Save QPINN training loss history and ReLoBRaLo weights to CSV"""
    csv_path = os.path.join(results_dir, 'qpinn_training_losses.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Step', 'Total_Cost'])
        for i, loss in enumerate(qnn_losses):
            writer.writerow([i + 1, loss])

    # ReLoBRaLo weight history
    if training_history and 'relobralo_weights' in training_history:
        csv_path2 = os.path.join(results_dir, 'qpinn_relobralo_weights.csv')
        weights = training_history['relobralo_weights']
        keys = list(weights.keys())
        n_steps = len(weights[keys[0]]) if keys else 0

        with open(csv_path2, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['Step'] + [f'Weight_{k}' for k in keys]
            writer.writerow(header)
            for i in range(n_steps):
                row = [i + 1] + [weights[k][i] for k in keys]
                writer.writerow(row)

    return csv_path


def save_metrics_over_time_csv(results_dir: str, u_pinn, u_qnn, u_analytical):
    """Save per-timestep MSE, MAE, and relative L2 error to CSV"""
    u_pinn_r = u_pinn.reshape(nx, ny, nz, nt)
    u_qnn_r = u_qnn.reshape(nx, ny, nz, nt)
    u_analytical_r = u_analytical.reshape(nx, ny, nz, nt)
    t_vals = np.linspace(0, T, nt)

    csv_path = os.path.join(results_dir, 'metrics_over_time.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Time',
                          'PINN_MSE', 'PINN_MAE', 'PINN_RMSE', 'PINN_RelL2', 'PINN_MaxAE',
                          'QPINN_MSE', 'QPINN_MAE', 'QPINN_RMSE', 'QPINN_RelL2', 'QPINN_MaxAE',
                          'PINN_Boundary_Error', 'QPINN_Boundary_Error',
                          'PINN_Energy', 'QPINN_Energy', 'Analytical_Energy'])

        dx = L / (nx - 1) if nx > 1 else L
        dy = L / (ny - 1) if ny > 1 else L
        dz = L / (nz - 1) if nz > 1 else L
        dV = dx * dy * dz

        for t_idx in range(nt):
            u_a = u_analytical_r[:, :, :, t_idx].flatten()
            u_p = u_pinn_r[:, :, :, t_idx].flatten()
            u_q = u_qnn_r[:, :, :, t_idx].flatten()

            # Clip predictions for metrics
            u_p_c = np.clip(np.nan_to_num(u_p, nan=0.0), 0, None)
            u_q_c = np.clip(np.nan_to_num(u_q, nan=0.0), 0, None)

            mse_p = float(np.mean((u_p_c - u_a) ** 2))
            mae_p = float(np.mean(np.abs(u_p_c - u_a)))
            rmse_p = float(np.sqrt(mse_p))
            rel_p = float(np.sqrt(np.sum((u_p_c - u_a) ** 2)) / np.sqrt(np.sum(u_a ** 2) + 1e-10))
            max_ae_p = float(np.max(np.abs(u_p_c - u_a)))

            mse_q = float(np.mean((u_q_c - u_a) ** 2))
            mae_q = float(np.mean(np.abs(u_q_c - u_a)))
            rmse_q = float(np.sqrt(mse_q))
            rel_q = float(np.sqrt(np.sum((u_q_c - u_a) ** 2)) / np.sqrt(np.sum(u_a ** 2) + 1e-10))
            max_ae_q = float(np.max(np.abs(u_q_c - u_a)))

            # Boundary error
            boundary_p = np.concatenate([
                u_pinn_r[0, :, :, t_idx].flatten(), u_pinn_r[-1, :, :, t_idx].flatten(),
                u_pinn_r[:, 0, :, t_idx].flatten(), u_pinn_r[:, -1, :, t_idx].flatten(),
                u_pinn_r[:, :, 0, t_idx].flatten(), u_pinn_r[:, :, -1, t_idx].flatten()
            ])
            boundary_q = np.concatenate([
                u_qnn_r[0, :, :, t_idx].flatten(), u_qnn_r[-1, :, :, t_idx].flatten(),
                u_qnn_r[:, 0, :, t_idx].flatten(), u_qnn_r[:, -1, :, t_idx].flatten(),
                u_qnn_r[:, :, 0, t_idx].flatten(), u_qnn_r[:, :, -1, t_idx].flatten()
            ])

            # Thermal energy at this timestep
            energy_p = float(np.sum(u_pinn_r[:, :, :, t_idx]) * dV)
            energy_q = float(np.sum(u_qnn_r[:, :, :, t_idx]) * dV)
            energy_a = float(np.sum(u_analytical_r[:, :, :, t_idx]) * dV)

            writer.writerow([t_vals[t_idx],
                              mse_p, mae_p, rmse_p, rel_p, max_ae_p,
                              mse_q, mae_q, rmse_q, rel_q, max_ae_q,
                              np.mean(np.abs(boundary_p)), np.mean(np.abs(boundary_q)),
                              energy_p, energy_q, energy_a])

    return csv_path


def save_validation_csv(results_dir: str, pinn_history=None, qpinn_history=None):
    """Save validation metrics collected during training to CSV"""
    saved = []

    # PINN validation
    if pinn_history and 'validation_history' in pinn_history:
        val_hist = pinn_history['validation_history']
        if val_hist:
            csv_path = os.path.join(results_dir, 'pinn_validation_metrics.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # Get timeslice keys from first entry
                ts_keys = sorted(val_hist[0].get('per_timeslice', {}).keys())
                header = ['Epoch', 'Elapsed_Sec', 'MSE', 'RelL2', 'Min_Pred', 'Neg_Count']
                for ts in ts_keys:
                    header.extend([f'MSE_{ts}', f'RelL2_{ts}'])
                writer.writerow(header)

                for entry in val_hist:
                    row = [entry['epoch'], f"{entry['elapsed_sec']:.2f}",
                           entry['mse'], entry['rel_l2'],
                           entry.get('min_pred', ''), entry.get('neg_count', '')]
                    for ts in ts_keys:
                        ts_data = entry.get('per_timeslice', {}).get(ts, {})
                        row.extend([ts_data.get('mse', ''), ts_data.get('rel_l2', '')])
                    writer.writerow(row)
            saved.append(csv_path)

    # QPINN validation
    if qpinn_history and 'validation_history' in qpinn_history:
        val_hist = qpinn_history['validation_history']
        if val_hist:
            csv_path = os.path.join(results_dir, 'qpinn_validation_metrics.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                ts_keys = sorted(val_hist[0].get('per_timeslice', {}).keys())
                header = ['Step', 'Elapsed_Sec', 'MSE', 'RelL2']
                for ts in ts_keys:
                    header.extend([f'MSE_{ts}', f'RelL2_{ts}'])
                writer.writerow(header)

                for entry in val_hist:
                    row = [entry.get('step', ''), f"{entry['elapsed_sec']:.2f}",
                           entry['mse'], entry['rel_l2']]
                    for ts in ts_keys:
                        ts_data = entry.get('per_timeslice', {}).get(ts, {})
                        row.extend([ts_data.get('mse', ''), ts_data.get('rel_l2', '')])
                    writer.writerow(row)
            saved.append(csv_path)

    return saved


def save_performance_csv(results_dir: str, pinn_history=None):
    """Save performance metrics (GPU/CPU) collected during PINN training to CSV"""
    if not pinn_history or 'performance_history' not in pinn_history:
        return None

    perf_hist = pinn_history['performance_history']
    if not perf_hist:
        return None

    csv_path = os.path.join(results_dir, 'performance_metrics.csv')
    # Get all metric keys from first entry
    all_keys = sorted(perf_hist[0].keys())

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(all_keys)
        for entry in perf_hist:
            writer.writerow([entry.get(k, '') for k in all_keys])

    return csv_path


# ================================================
# Visualization
# ================================================
def visualize_results_comparison(results_dir: str, u_pinn: np.ndarray, u_qnn: np.ndarray,
                                  u_analytical: np.ndarray, pinn_losses=None, qnn_losses=None,
                                  pinn_history=None, qpinn_history=None,
                                  logger=None) -> List[str]:
    """Comprehensive visualization comparing RAdam PINN vs SPSA QPINN.

    Generates multiple plot files covering temperature fields, loss curves,
    ReLoBRaLo weight evolution, error distributions, and 1D profiles.
    """
    log = logger.info if logger else print
    log("Generating comparison visualizations...")

    saved_files = []
    u_pinn_reshaped = u_pinn.reshape(nx, ny, nz, nt)
    u_analytical_reshaped = u_analytical.reshape(nx, ny, nz, nt)
    u_qnn_reshaped = u_qnn.reshape(nx, ny, nz, nt)

    x = np.linspace(0, L, nx)
    y = np.linspace(0, L, ny)
    t = np.linspace(0, T, nt)

    # ---------------------------------------------------
    # 1. Temperature field comparison at multiple times
    # ---------------------------------------------------
    z_mid_idx = nz // 2
    t_indices = [0, nt // 4, nt // 2, 3 * nt // 4, nt - 1]

    fig, axes = plt.subplots(3, len(t_indices), figsize=(20, 12))
    for i, t_idx in enumerate(t_indices):
        u_pinn_2d = u_pinn_reshaped[:, :, z_mid_idx, t_idx]
        u_analytical_2d = u_analytical_reshaped[:, :, z_mid_idx, t_idx]
        u_qnn_2d = u_qnn_reshaped[:, :, z_mid_idx, t_idx]
        vmin = 0
        vmax = max(np.max(u_analytical_2d), np.max(u_pinn_2d), np.max(u_qnn_2d)) * 1.1
        if vmax <= 0:
            vmax = 1.0

        for row, (data, label) in enumerate([(u_pinn_2d, 'PINN RAdam'),
                                              (u_qnn_2d, 'QPINN SPSA'),
                                              (u_analytical_2d, 'Analytical')]):
            im = axes[row, i].imshow(data.T, origin='lower', extent=[0, L, 0, L],
                                      cmap='hot', vmin=vmin, vmax=vmax)
            axes[row, i].set_title(f'{label} (t={t[t_idx]:.2f})')
            axes[row, i].set_xlabel('x')
            axes[row, i].set_ylabel('y')
            fig.colorbar(im, ax=axes[row, i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    path = os.path.join(results_dir, 'comparison_heat_equation.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    saved_files.append(path)

    # ---------------------------------------------------
    # 2. Loss curves comparison + MSE over time
    # ---------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    if pinn_losses:
        axes[0].semilogy(pinn_losses, 'b-', linewidth=1, alpha=0.7)
        axes[0].set_title('PINN (RAdam) Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)

    if qnn_losses:
        axes[1].semilogy(qnn_losses, 'r-', linewidth=1, alpha=0.7)
        axes[1].set_title('QPINN (SPSA) Training Loss')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Cost')
        axes[1].grid(True, alpha=0.3)

    mse_pinn_t, mse_qnn_t = [], []
    for t_idx in range(nt):
        u_a_t = u_analytical_reshaped[:, :, :, t_idx].flatten()
        mse_p, _ = calculate_metrics(u_pinn_reshaped[:, :, :, t_idx].flatten(), u_a_t)
        mse_q, _ = calculate_metrics(u_qnn_reshaped[:, :, :, t_idx].flatten(), u_a_t)
        mse_pinn_t.append(mse_p)
        mse_qnn_t.append(mse_q)

    axes[2].semilogy(t, mse_pinn_t, 'b-', linewidth=2, label='PINN (RAdam)')
    axes[2].semilogy(t, mse_qnn_t, 'r--', linewidth=2, label='QPINN (SPSA)')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('MSE')
    axes[2].set_title('MSE over Time')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(results_dir, 'loss_comparison.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    saved_files.append(path)

    # ---------------------------------------------------
    # 3. 1D profile comparison
    # ---------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, t_idx in enumerate([nt // 4, nt // 2, 3 * nt // 4, nt - 1]):
        ax = axes[i // 2, i % 2]
        ax.plot(x, u_analytical_reshaped[:, ny // 2, nz // 2, t_idx],
                'g-', linewidth=3, label='Analytical', alpha=0.8)
        ax.plot(x, u_pinn_reshaped[:, ny // 2, nz // 2, t_idx],
                'b--', linewidth=2, label='PINN (RAdam)')
        ax.plot(x, u_qnn_reshaped[:, ny // 2, nz // 2, t_idx],
                'r:', linewidth=2, label='QPINN (SPSA)')
        ax.set_title(f'Temperature Profile at t={t[t_idx]:.2f}')
        ax.set_xlabel('x')
        ax.set_ylabel('Temperature')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=-0.05)

    plt.tight_layout()
    path = os.path.join(results_dir, 'profile_comparison.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    saved_files.append(path)

    # ---------------------------------------------------
    # 4. PINN per-component loss + ReLoBRaLo weights
    # ---------------------------------------------------
    if pinn_history and 'loss_components' in pinn_history:
        comps = pinn_history['loss_components']
        weights = pinn_history['relobralo_weights']
        loss_keys = list(comps.keys())

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        colors = {'initial': 'blue', 'peak': 'orange', 'boundary': 'green', 'pde': 'red', 'nonneg': 'purple'}

        for k in loss_keys:
            c = colors.get(k, 'gray')
            ax1.semilogy(comps[k], label=f'{k}', color=c, linewidth=1, alpha=0.8)
        ax1.set_ylabel('Loss (log scale)')
        ax1.set_title('PINN Per-Component Loss History')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        for k in loss_keys:
            c = colors.get(k, 'gray')
            ax2.plot(weights[k], label=f'w_{k}', color=c, linewidth=1.5)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('ReLoBRaLo Weight')
        ax2.set_title('PINN ReLoBRaLo Adaptive Loss Weights')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(results_dir, 'pinn_relobralo_evolution.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_files.append(path)

    # ---------------------------------------------------
    # 5. QPINN ReLoBRaLo weights
    # ---------------------------------------------------
    if qpinn_history and 'relobralo_weights' in qpinn_history:
        weights = qpinn_history['relobralo_weights']
        loss_keys = list(weights.keys())

        if loss_keys and len(weights[loss_keys[0]]) > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = {'initial': 'blue', 'boundary': 'green', 'interior': 'red'}

            for k in loss_keys:
                c = colors.get(k, 'gray')
                ax.plot(weights[k], label=f'w_{k}', color=c, linewidth=1.5)
            ax.set_xlabel('Step')
            ax.set_ylabel('ReLoBRaLo Weight')
            ax.set_title('QPINN ReLoBRaLo Adaptive Loss Weights')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            path = os.path.join(results_dir, 'qpinn_relobralo_evolution.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            saved_files.append(path)

    # ---------------------------------------------------
    # 6. Error distribution histograms
    # ---------------------------------------------------
    error_pinn = np.abs(u_pinn - u_analytical)
    error_qnn = np.abs(u_qnn - u_analytical)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(error_pinn[error_pinn > 0], bins=50, alpha=0.7, color='blue', label='PINN')
    ax1.hist(error_qnn[error_qnn > 0], bins=50, alpha=0.5, color='red', label='QPINN')
    ax1.set_xlabel('Absolute Error')
    ax1.set_ylabel('Count')
    ax1.set_title('Error Distribution')
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # Error heatmap at mid-time
    t_mid_idx = nt // 2
    err_pinn_2d = np.abs(u_pinn_reshaped[:, :, z_mid_idx, t_mid_idx] -
                          u_analytical_reshaped[:, :, z_mid_idx, t_mid_idx])
    im = ax2.imshow(err_pinn_2d.T, origin='lower', extent=[0, L, 0, L], cmap='viridis')
    ax2.set_title(f'PINN Absolute Error at t={t[t_mid_idx]:.2f}, z=L/2')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    path = os.path.join(results_dir, 'error_distribution.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    saved_files.append(path)

    # ---------------------------------------------------
    # 7. Learning rate schedule (PINN)
    # ---------------------------------------------------
    if pinn_history and 'lr_history' in pinn_history and pinn_history['lr_history']:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.semilogy(pinn_history['lr_history'], 'b-', linewidth=1)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('PINN Learning Rate Schedule (CosineAnnealingWarmRestarts)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(results_dir, 'pinn_learning_rate.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_files.append(path)

    log(f"Generated {len(saved_files)} visualization files")
    return saved_files


# ================================================
# JSON Output
# ================================================
def save_comparative_results(results_dir: str, pinn_results: dict, qpinn_results: dict,
                              mse_pinn=None, rel_l2_pinn=None,
                              mse_qnn=None, rel_l2_qnn=None,
                              pinn_history=None, qpinn_history=None,
                              pinn_full_metrics=None, qpinn_full_metrics=None,
                              pinn_time=0, qnn_time=0,
                              logger=None):
    """Save comprehensive comparative analysis results to JSON"""

    # System information
    system_info = {
        'pytorch_version': torch.__version__,
        'pennylane_version': qml.__version__,
        'python_version': sys.version,
        'device': str(device),
    }
    if torch.cuda.is_available():
        system_info['gpu_name'] = torch.cuda.get_device_name(0)
        system_info['gpu_memory_total_gb'] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        system_info['cuda_version'] = torch.version.cuda
    try:
        import psutil
        system_info['cpu_count'] = psutil.cpu_count(logical=True)
        system_info['ram_total_gb'] = round(psutil.virtual_memory().total / 1e9, 2)
    except ImportError:
        pass

    comparative_data = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'comparison_type': 'SPINN+PINNsFormer PINN (RAdam) vs QPINN (SPSA)',
            'system': system_info,
            'problem': {
                'equation': '3D Heat Equation: du/dt = alpha * nabla^2(u)',
                'alpha': alpha,
                'L': L,
                'T': T,
                'sigma_0': sigma_0,
                'initial_condition': 'Gaussian centered at (L/2, L/2, L/2)',
                'boundary_condition': 'Homogeneous Dirichlet (u=0 on all faces)',
                'grid': f'{nx}x{ny}x{nz}x{nt}',
                'total_grid_points': nx * ny * nz * nt
            }
        },
        'pinn_radam_results': pinn_results,
        'qpinn_spsa_results': qpinn_results,
        'metrics': {
            'pinn': pinn_full_metrics if pinn_full_metrics else {'mse': mse_pinn, 'relative_l2': rel_l2_pinn},
            'qpinn': qpinn_full_metrics if qpinn_full_metrics else {'mse': mse_qnn, 'relative_l2': rel_l2_qnn},
        },
        'timing': {
            'pinn_training_time_sec': pinn_time,
            'qpinn_training_time_sec': qnn_time,
            'time_ratio_qpinn_over_pinn': qnn_time / max(pinn_time, 1e-6),
        },
        'comparative_analysis': {
            'pinn_architecture': 'SPINN body networks + PINNsFormer encoder-decoder',
            'pinn_optimizer': 'RAdam (decoupled weight decay) + L-BFGS',
            'qpinn_optimizer': 'SPSA (Simultaneous Perturbation)',
            'adaptive_weighting': 'ReLoBRaLo (both)',
            'pinn_loss_components': ['Initial Condition', 'Peak', 'Boundary Condition', 'PDE Residual', 'Non-negativity'],
            'qpinn_loss_components': ['Initial Condition', 'Boundary Condition', 'Interior/PDE Residual'],
        }
    }

    # Add detailed training history if available
    if pinn_history:
        comparative_data['pinn_training_detail'] = {
            'best_unweighted_loss': pinn_history.get('best_unweighted_loss'),
            'final_relobralo_weights': pinn_history.get('final_weights'),
            'epochs': pinn_history.get('epochs'),
            'radam_epochs': pinn_history.get('radam_epochs'),
            'lbfgs_epochs': pinn_history.get('lbfgs_epochs'),
            'n_params': pinn_history.get('n_params'),
        }
    if qpinn_history:
        comparative_data['qpinn_training_detail'] = {
            'best_cost': qpinn_history.get('best_cost'),
            'final_relobralo_weights': qpinn_history.get('final_weights'),
            'max_iterations': qpinn_history.get('max_iterations'),
            'n_circuit_params': qpinn_history.get('n_circuit_params'),
            'n_total_params': qpinn_history.get('n_total_params'),
        }

    json_path = os.path.join(results_dir, 'comparative_analysis.json')
    with open(json_path, 'w') as f:
        json.dump(comparative_data, f, indent=2, default=str)

    log = logger.info if logger else print
    log(f"Comparative analysis saved: {json_path}")
    return json_path


# ================================================
# TXT Summary Report
# ================================================
def save_summary_report(results_dir: str, pinn_results: dict, qpinn_results: dict,
                         mse_pinn, rel_l2_pinn, mse_qnn, rel_l2_qnn,
                         pinn_time, qnn_time,
                         pinn_history=None, qpinn_history=None,
                         boundary_analysis=None,
                         pinn_full_metrics=None, qpinn_full_metrics=None):
    """Save human-readable benchmark summary report with comprehensive metrics."""
    report_path = os.path.join(results_dir, 'benchmark_summary.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 90 + "\n")
        f.write("  PINN vs QPINN Benchmark Summary Report\n")
        f.write("  3D Heat Equation: du/dt = alpha * nabla^2(u)\n")
        f.write("  Architecture: SPINN + PINNsFormer (Separable Transformer PINN)\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"PyTorch: {torch.__version__}, PennyLane: {qml.__version__}\n")
        f.write(f"Device: {device}\n")
        if torch.cuda.is_available():
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
            f.write(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
        try:
            import psutil
            f.write(f"CPU: {psutil.cpu_count(logical=True)} logical cores\n")
            f.write(f"RAM: {psutil.virtual_memory().total / 1e9:.1f} GB\n")
        except ImportError:
            pass
        f.write("\n")

        # ── Problem Configuration ──
        f.write("Problem Configuration\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Equation:     3D Heat Equation (homogeneous Dirichlet BC)\n")
        f.write(f"  Alpha:        {alpha} (thermal diffusivity)\n")
        f.write(f"  Domain:       [0, {L}]^3 x [0, {T}]\n")
        f.write(f"  Grid:         {nx} x {ny} x {nz} x {nt} = {nx*ny*nz*nt:,} points\n")
        f.write(f"  IC:           Gaussian(sigma={sigma_0}) at ({L/2}, {L/2}, {L/2})\n\n")

        # ── PINN Section ──
        f.write("PINN: SPINN + PINNsFormer (RAdam + L-BFGS + ReLoBRaLo)\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Architecture:   SPINN per-axis body nets -> PINNsFormer encoder-decoder\n")
        f.write(f"  Optimizer:      RAdam (decoupled_weight_decay) + L-BFGS\n")
        f.write(f"  Loss weighting: ReLoBRaLo (adaptive multi-objective)\n")
        f.write(f"  Training time:  {pinn_time:.2f} s\n")
        if pinn_history:
            f.write(f"  Total epochs:   {pinn_history.get('epochs', 'N/A')}")
            radam_e = pinn_history.get('radam_epochs', '')
            lbfgs_e = pinn_history.get('lbfgs_epochs', '')
            if radam_e or lbfgs_e:
                f.write(f"  (RAdam: {radam_e}, L-BFGS: {lbfgs_e})")
            f.write("\n")
            f.write(f"  Parameters:     {pinn_history.get('n_params', 'N/A'):,}\n")
            f.write(f"  Best loss:      {pinn_history.get('best_unweighted_loss', 'N/A')}\n")
            fw = pinn_history.get('final_weights', {})
            if fw:
                f.write(f"  ReLoBRaLo wts:  {', '.join(f'{k}={v:.4f}' for k, v in fw.items())}\n")
        f.write("\n")

        # PINN error metrics
        f.write("  Error Metrics:\n")
        if pinn_full_metrics:
            pm = pinn_full_metrics
            f.write(f"    MSE:            {pm['mse']:.6e}\n")
            f.write(f"    RMSE:           {pm['rmse']:.6e}\n")
            f.write(f"    MAE:            {pm['mae']:.6e}\n")
            f.write(f"    Max AE:         {pm['max_ae']:.6e}\n")
            f.write(f"    Relative L2:    {pm['rel_l2']:.6e}\n")
            f.write(f"    Relative MAE:   {pm['rel_mae']:.6e}\n")
        else:
            f.write(f"    MSE:            {mse_pinn:.6e}\n")
            f.write(f"    Relative L2:    {rel_l2_pinn:.6e}\n")
        f.write("\n")

        # PINN physics metrics
        if pinn_full_metrics:
            pm = pinn_full_metrics
            f.write("  Physics Metrics:\n")
            f.write(f"    Peak value (t=0, center):  pred={pm['peak_pred']:.6f}, "
                    f"true={pm['peak_true']:.6f}, error={pm['peak_error']:.6e}\n")
            f.write(f"    Boundary satisfaction:      mean |u_boundary| = {pm['mean_boundary_error']:.6e}\n")
            f.write(f"    Non-negativity violations:  {pm['neg_count']} ({pm['neg_fraction']*100:.2f}%)\n")
            f.write(f"    Prediction range:           [{pm['pred_min']:.6f}, {pm['pred_max']:.6f}]\n")
            if pm.get('energy_pred') and pm.get('energy_true'):
                f.write(f"    Energy (t=0):              pred={pm['energy_pred'][0]:.6f}, "
                        f"true={pm['energy_true'][0]:.6f}\n")
                if len(pm['energy_pred']) > 1:
                    f.write(f"    Energy (t=T):              pred={pm['energy_pred'][-1]:.6f}, "
                            f"true={pm['energy_true'][-1]:.6f}\n")
            f.write("\n")

        # ── QPINN Section ──
        f.write("QPINN (SPSA + ReLoBRaLo)\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Optimizer:      PennyLane SPSAOptimizer\n")
        f.write(f"  Loss weighting: ReLoBRaLo (adaptive multi-objective)\n")
        f.write(f"  Training time:  {qnn_time:.2f} s\n")
        if qpinn_history:
            f.write(f"  Max iterations: {qpinn_history.get('max_iterations', 'N/A')}\n")
            f.write(f"  Circuit params: {qpinn_history.get('n_circuit_params', 'N/A')}\n")
            f.write(f"  Total params:   {qpinn_history.get('n_total_params', 'N/A')}\n")
            f.write(f"  Best cost:      {qpinn_history.get('best_cost', 'N/A')}\n")
            fw = qpinn_history.get('final_weights', {})
            if fw:
                f.write(f"  ReLoBRaLo wts:  {', '.join(f'{k}={v:.4f}' for k, v in fw.items())}\n")
        f.write("\n")

        # QPINN error metrics
        f.write("  Error Metrics:\n")
        if qpinn_full_metrics:
            qm = qpinn_full_metrics
            f.write(f"    MSE:            {qm['mse']:.6e}\n")
            f.write(f"    RMSE:           {qm['rmse']:.6e}\n")
            f.write(f"    MAE:            {qm['mae']:.6e}\n")
            f.write(f"    Max AE:         {qm['max_ae']:.6e}\n")
            f.write(f"    Relative L2:    {qm['rel_l2']:.6e}\n")
            f.write(f"    Relative MAE:   {qm['rel_mae']:.6e}\n")
        else:
            f.write(f"    MSE:            {mse_qnn:.6e}\n")
            f.write(f"    Relative L2:    {rel_l2_qnn:.6e}\n")
        f.write("\n")

        # QPINN physics metrics
        if qpinn_full_metrics:
            qm = qpinn_full_metrics
            f.write("  Physics Metrics:\n")
            f.write(f"    Peak value (t=0, center):  pred={qm['peak_pred']:.6f}, "
                    f"true={qm['peak_true']:.6f}, error={qm['peak_error']:.6e}\n")
            f.write(f"    Boundary satisfaction:      mean |u_boundary| = {qm['mean_boundary_error']:.6e}\n")
            f.write(f"    Non-negativity violations:  {qm['neg_count']} ({qm['neg_fraction']*100:.2f}%)\n")
            f.write(f"    Prediction range:           [{qm['pred_min']:.6f}, {qm['pred_max']:.6f}]\n")
            if qm.get('energy_pred') and qm.get('energy_true'):
                f.write(f"    Energy (t=0):              pred={qm['energy_pred'][0]:.6f}, "
                        f"true={qm['energy_true'][0]:.6f}\n")
                if len(qm['energy_pred']) > 1:
                    f.write(f"    Energy (t=T):              pred={qm['energy_pred'][-1]:.6f}, "
                            f"true={qm['energy_true'][-1]:.6f}\n")
            f.write("\n")

        # ── Comparative Table ──
        f.write("=" * 90 + "\n")
        f.write("  Comparative Summary\n")
        f.write("=" * 90 + "\n\n")

        header = f"  {'Metric':<30s} {'PINN':>18s} {'QPINN':>18s} {'Better':>10s}\n"
        f.write(header)
        f.write("  " + "-" * 76 + "\n")

        def write_metric_row(name, pinn_val, qpinn_val, lower_better=True):
            if pinn_val is not None and qpinn_val is not None:
                better = 'PINN' if (pinn_val < qpinn_val) == lower_better else 'QPINN'
                f.write(f"  {name:<30s} {pinn_val:>18.6e} {qpinn_val:>18.6e} {better:>10s}\n")

        if pinn_full_metrics and qpinn_full_metrics:
            pm, qm = pinn_full_metrics, qpinn_full_metrics
            write_metric_row("MSE", pm['mse'], qm['mse'])
            write_metric_row("RMSE", pm['rmse'], qm['rmse'])
            write_metric_row("MAE", pm['mae'], qm['mae'])
            write_metric_row("Max AE", pm['max_ae'], qm['max_ae'])
            write_metric_row("Relative L2", pm['rel_l2'], qm['rel_l2'])
            write_metric_row("Relative MAE", pm['rel_mae'], qm['rel_mae'])
            write_metric_row("Peak Error", pm['peak_error'], qm['peak_error'])
            write_metric_row("Mean Boundary Error", pm['mean_boundary_error'], qm['mean_boundary_error'])
            write_metric_row("Neg. Violations", pm['neg_count'], qm['neg_count'])
            f.write(f"  {'Training Time (s)':<30s} {pinn_time:>18.2f} {qnn_time:>18.2f} "
                    f"{'PINN' if pinn_time < qnn_time else 'QPINN':>10s}\n")
            pinn_params = pinn_history.get('n_params', 0) if pinn_history else 0
            qpinn_params = qpinn_history.get('n_total_params', 0) if qpinn_history else 0
            if pinn_params and qpinn_params:
                f.write(f"  {'Parameters':<30s} {pinn_params:>18,d} {qpinn_params:>18,d}\n")
        else:
            write_metric_row("MSE", mse_pinn, mse_qnn)
            write_metric_row("Relative L2", rel_l2_pinn, rel_l2_qnn)
            f.write(f"  {'Training Time (s)':<30s} {pinn_time:>18.2f} {qnn_time:>18.2f}\n")

        f.write("\n")

        # Improvement summary
        if mse_pinn > 0:
            mse_diff = ((mse_pinn - mse_qnn) / mse_pinn) * 100
            if mse_diff > 0:
                f.write(f"  QPINN MSE improvement over PINN: {mse_diff:.2f}%\n")
            else:
                f.write(f"  PINN MSE advantage over QPINN: {-mse_diff:.2f}%\n")
        if qnn_time > 0 and pinn_time > 0:
            f.write(f"  Time ratio (QPINN/PINN): {qnn_time / pinn_time:.2f}x\n")
        f.write("\n")

        # ── Boundary Analysis ──
        if boundary_analysis:
            f.write("Boundary Condition Satisfaction (per timestep)\n")
            f.write("-" * 50 + "\n")
            f.write(f"  {'t':>6s}  {'PINN avg|u|':>14s}  {'QPINN avg|u|':>14s}\n")
            for entry in boundary_analysis:
                f.write(f"  {entry['t']:>6.2f}  {entry['pinn_avg']:>14.6e}  {entry['qpinn_avg']:>14.6e}\n")
            f.write("\n")

        # ── Energy Conservation ──
        if pinn_full_metrics and pinn_full_metrics.get('energy_pred'):
            f.write("Thermal Energy Evolution\n")
            f.write("-" * 50 + "\n")
            t_vals = np.linspace(0, T, nt)
            f.write(f"  {'t':>6s}  {'Analytical':>14s}  {'PINN':>14s}")
            if qpinn_full_metrics and qpinn_full_metrics.get('energy_pred'):
                f.write(f"  {'QPINN':>14s}")
            f.write("\n")
            for i, t_v in enumerate(t_vals):
                if i < len(pinn_full_metrics['energy_true']):
                    f.write(f"  {t_v:>6.3f}  {pinn_full_metrics['energy_true'][i]:>14.6e}"
                            f"  {pinn_full_metrics['energy_pred'][i]:>14.6e}")
                    if qpinn_full_metrics and qpinn_full_metrics.get('energy_pred') and i < len(qpinn_full_metrics['energy_pred']):
                        f.write(f"  {qpinn_full_metrics['energy_pred'][i]:>14.6e}")
                    f.write("\n")
            f.write("\n")

        # ── Output Files ──
        f.write("Output Files\n")
        f.write("-" * 50 + "\n")
        f.write("  Plots:  comparison_heat_equation.png, loss_comparison.png,\n")
        f.write("          profile_comparison.png, pinn_relobralo_evolution.png,\n")
        f.write("          qpinn_relobralo_evolution.png, error_distribution.png,\n")
        f.write("          pinn_learning_rate.png\n")
        f.write("  CSV:    pinn_training_losses.csv, pinn_loss_components.csv,\n")
        f.write("          qpinn_training_losses.csv, qpinn_relobralo_weights.csv,\n")
        f.write("          metrics_over_time.csv, performance_metrics.csv\n")
        f.write("  JSON:   comparative_analysis.json\n")
        f.write("  Log:    benchmark.log\n")
        f.write("  Model:  pinn_radam_checkpoint.pth, qpinn_spsa_checkpoint.pth\n")

    return report_path


# ================================================
# Main function
# ================================================
def main(backend='auto', config_path=None):
    """Main function for PINN/QPINN comparison benchmark

    Args:
        backend: Backend selection ('auto', 'cpu', 'cuda', 'gpu', 'qpu')
        config_path: Path to JSON config file (optional)
    """

    # Lazy imports to avoid circular dependencies
    from pinn_model import PINN, evaluate_pinn, train_pinn
    from qpinn_model import GQEQuantumPINN

    # Load JSON config if provided
    bench_config = {}
    if config_path and os.path.exists(config_path):
        bench_config = load_benchmark_config(config_path)
        apply_config(bench_config)

    # Create output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results/')
    os.makedirs(results_dir, exist_ok=True)

    # Setup logging
    logger = setup_logging(results_dir)

    logger.info("=" * 60)
    logger.info("3D Heat Equation: PINN (RAdam) vs QPINN (SPSA) Benchmark")
    logger.info("=" * 60)

    if config_path:
        logger.info(f"Config loaded from: {config_path}")

    BackendConfig.print_config(backend)
    global device
    device = BackendConfig.get_torch_device(backend)
    logger.debug(f"Device set to: {device}")

    training_config = TrainingConfig()
    logger.info(f"Training Configuration:")
    logger.info(f"  PINN: RAdam, lr={training_config.pinn_lr}, epochs={training_config.pinn_epochs}")
    logger.info(f"  QPINN: SPSA, max_iter={training_config.qpinn_max_iterations}, c={training_config.qpinn_spsa_c}")
    logger.info(f"  Adaptive weighting: ReLoBRaLo (both)")

    # Check for existing checkpoints
    pinn_checkpoint_path = os.path.join(results_dir, 'pinn_radam_checkpoint.pth')
    qpinn_checkpoint_path = os.path.join(results_dir, 'qpinn_spsa_checkpoint.pth')
    pinn_exists, qpinn_exists = check_existing_checkpoints(results_dir)

    logger.info(f"Checkpoint Status: PINN={'Found' if pinn_exists else 'Not found'}, "
                f"QPINN={'Found' if qpinn_exists else 'Not found'}")

    # Initialize variables
    pinn_model = None
    pinn_losses = []
    pinn_time = 0
    u_pinn = None
    pinn_results = {}
    pinn_history = None

    qsolver = None
    qnn_losses = []
    qnn_time = 0
    u_qnn = None
    qpinn_results = {}
    qpinn_history = None

    # ============================
    # 1. PINN (RAdam)
    # ============================
    if pinn_exists:
        logger.info("=== Loading PINN from checkpoint ===")
        pinn_checkpoint = load_checkpoint(pinn_checkpoint_path, logger=logger)
        if pinn_checkpoint:
            try:
                config = pinn_checkpoint.get('training_config', {})
                layers = config.get('layers', [5, 128, 256, 256, 128, 1])
                pinn_model = PINN(
                    layers=layers,
                    use_hard_constraints=config.get('use_hard_constraints', True),
                    boundary_epsilon=config.get('boundary_epsilon', 0.1),
                    fourier_features=config.get('use_fourier_features', True),
                    use_transformer=config.get('use_transformer', True),
                    transformer_memory_efficient=config.get('transformer_memory_efficient', True)
                ).to(device)
                pinn_model.load_state_dict(pinn_checkpoint['model_state_dict'])
                pinn_losses = pinn_checkpoint['losses']
                pinn_time = pinn_checkpoint['training_time']
                u_pinn = pinn_checkpoint['predictions']
                pinn_results = pinn_checkpoint['results']
                pinn_history = pinn_checkpoint.get('training_history')
                logger.info(f"PINN loaded: time={pinn_time:.2f}s, "
                            f"final_loss={pinn_losses[-1] if pinn_losses else 'N/A'}")
            except Exception as e:
                logger.error(f"Error loading PINN: {e}")
                pinn_exists = False

    if not pinn_exists:
        logger.info("=== Training PINN with RAdam ===")
        pinn_model, pinn_losses, pinn_time = train_pinn(config=bench_config)
        u_pinn = evaluate_pinn(pinn_model)

        pinn_history = getattr(pinn_model, 'training_history', None)

        pinn_results = {
            'method': 'PINN with RAdam + ReLoBRaLo',
            'training_time': pinn_time,
            'final_loss': pinn_losses[-1] if pinn_losses else None,
            'optimizer': 'RAdam (decoupled_weight_decay=True)',
            'adaptive_weighting': 'ReLoBRaLo',
        }

        checkpoint_data = {
            'model_state_dict': pinn_model.state_dict(),
            'training_config': pinn_model.training_config if hasattr(pinn_model, 'training_config') else {},
            'losses': pinn_losses,
            'training_time': pinn_time,
            'predictions': u_pinn,
            'results': pinn_results,
            'training_history': pinn_history,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        save_checkpoint(checkpoint_data, pinn_checkpoint_path, logger=logger)

    logger.info(f"PINN completed in {pinn_time:.2f} seconds")

    # Save PINN CSV
    save_pinn_training_csv(results_dir, pinn_losses, pinn_history)
    logger.debug("PINN training CSV saved")

    # Clear CUDA memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # ============================
    # 2. QPINN (SPSA)
    # ============================
    if qpinn_exists:
        logger.info("=== Loading QPINN from checkpoint ===")
        qpinn_checkpoint = load_checkpoint(qpinn_checkpoint_path, logger=logger)
        if qpinn_checkpoint:
            try:
                qsolver = GQEQuantumPINN(
                    n_qubits=qpinn_checkpoint.get('n_qubits', 6),
                    backend=qpinn_checkpoint.get('backend', 'default.mixed'),
                    shots=qpinn_checkpoint.get('shots', 1000),
                    noise_model=qpinn_checkpoint.get('noise_model', 'realistic'),
                    use_parallel=qpinn_checkpoint.get('use_parallel', True),
                    n_parallel_devices=qpinn_checkpoint.get('n_parallel_devices', N_PARALLEL_DEVICES),
                    use_gpt_circuit_generation=qpinn_checkpoint.get('use_gpt_circuit_generation', True)
                )
                if 'circuit_params' in qpinn_checkpoint and qpinn_checkpoint['circuit_params'] is not None:
                    qsolver.circuit_params = qpinn_checkpoint['circuit_params']
                qnn_losses = qpinn_checkpoint['losses']
                qnn_time = qpinn_checkpoint['training_time']
                u_qnn = qpinn_checkpoint['predictions']
                qpinn_results = qpinn_checkpoint['results']
                qpinn_history = qpinn_checkpoint.get('training_history')
                logger.info(f"QPINN loaded: time={qnn_time:.2f}s")
            except Exception as e:
                logger.error(f"Error loading QPINN: {e}")
                qpinn_exists = False

    if not qpinn_exists:
        logger.info("=== Training QPINN with SPSA ===")
        qpinn_circuit_cfg = get_nested(bench_config, 'qpinn.circuit', {})
        qsolver = GQEQuantumPINN(
            n_qubits=qpinn_circuit_cfg.get('n_qubits', 6),
            backend=qpinn_circuit_cfg.get('backend', 'default.mixed'),
            shots=qpinn_circuit_cfg.get('shots', 1000),
            noise_model=qpinn_circuit_cfg.get('noise_model', 'realistic'),
            use_parallel=get_nested(bench_config, 'parallel.use_parallel_training', True),
            n_parallel_devices=N_PARALLEL_DEVICES,
            use_gpt_circuit_generation=qpinn_circuit_cfg.get('use_gpt_circuit_generation', True)
        )

        qpinn_train_cfg = get_nested(bench_config, 'qpinn.training', {})
        try:
            _, qnn_losses, qnn_time = qsolver.train_spsa(
                n_samples=qpinn_train_cfg.get('n_samples', training_config.qpinn_n_samples),
                max_iterations=qpinn_train_cfg.get('max_iterations', training_config.qpinn_max_iterations),
                spsa_c=qpinn_train_cfg.get('spsa_c', training_config.qpinn_spsa_c),
                spsa_a=qpinn_train_cfg.get('spsa_a', training_config.qpinn_spsa_a),
                config=bench_config,
            )

            qpinn_history = getattr(qsolver, 'training_history_detail', None)

            u_qnn = qsolver.evaluate()
            logger.info("QPINN evaluation completed")

            qpinn_results = {
                'method': 'QPINN with SPSA + ReLoBRaLo',
                'training_time': qnn_time,
                'final_loss': qnn_losses[-1] if qnn_losses else None,
                'optimizer': 'SPSAOptimizer',
                'adaptive_weighting': 'ReLoBRaLo',
                'circuit_parameters': len(qsolver.circuit_template.parameter_map) if hasattr(qsolver, 'circuit_template') else 0,
            }

            checkpoint_data = {
                'n_qubits': qsolver.n_qubits,
                'backend': qsolver.backend,
                'shots': qsolver.shots,
                'noise_model': qsolver.noise_model,
                'use_parallel': qsolver.use_parallel,
                'n_parallel_devices': qsolver.n_parallel_devices,
                'use_gpt_circuit_generation': qsolver.use_gpt_circuit_generation,
                'circuit_params': qsolver.circuit_params if hasattr(qsolver, 'circuit_params') else None,
                'losses': qnn_losses,
                'training_time': qnn_time,
                'predictions': u_qnn,
                'results': qpinn_results,
                'training_history': qpinn_history,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            save_checkpoint(checkpoint_data, qpinn_checkpoint_path, logger=logger)

        except Exception as e:
            logger.error(f"Error during QPINN training: {e}")
            logger.debug(traceback.format_exc())
            u_qnn = np.zeros(nx * ny * nz * nt)
            qnn_losses = []
            qnn_time = 0
            qpinn_results = {'method': 'Failed', 'error': str(e)}

    # Save QPINN CSV
    save_qpinn_training_csv(results_dir, qnn_losses, qpinn_history)
    logger.debug("QPINN training CSV saved")

    # ============================
    # 3. QPINN Visualization
    # ============================
    if qsolver is not None:
        try:
            logger.info("Generating QPINN circuit visualizations...")
            qsolver.visualize_quantum_circuit(save_path=results_dir)
            qsolver.save_circuit_information(save_path=results_dir)
            qsolver.visualize_circuit_metrics(save_path=results_dir)
        except Exception as e:
            logger.warning(f"QPINN visualization error (non-fatal): {e}")

    # ============================
    # 4. Comparison & Metrics
    # ============================
    u_analytical = compute_analytical_solution()

    mse_pinn, rel_l2_pinn = calculate_metrics(u_pinn, u_analytical)
    mse_qnn, rel_l2_qnn = calculate_metrics(u_qnn, u_analytical)

    # Compute comprehensive metrics
    pinn_full_metrics = calculate_full_metrics(u_pinn, u_analytical)
    qpinn_full_metrics = calculate_full_metrics(u_qnn, u_analytical)

    logger.info("")
    logger.info("=" * 80)
    logger.info("Benchmark Results: SPINN+PINNsFormer PINN (RAdam) vs QPINN (SPSA)")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"{'Metric':<25s} {'PINN':>18s} {'QPINN':>18s}")
    logger.info("-" * 61)
    logger.info(f"{'MSE':<25s} {pinn_full_metrics['mse']:>18.6e} {qpinn_full_metrics['mse']:>18.6e}")
    logger.info(f"{'RMSE':<25s} {pinn_full_metrics['rmse']:>18.6e} {qpinn_full_metrics['rmse']:>18.6e}")
    logger.info(f"{'MAE':<25s} {pinn_full_metrics['mae']:>18.6e} {qpinn_full_metrics['mae']:>18.6e}")
    logger.info(f"{'Max AE':<25s} {pinn_full_metrics['max_ae']:>18.6e} {qpinn_full_metrics['max_ae']:>18.6e}")
    logger.info(f"{'Relative L2':<25s} {pinn_full_metrics['rel_l2']:>18.6e} {qpinn_full_metrics['rel_l2']:>18.6e}")
    logger.info(f"{'Relative MAE':<25s} {pinn_full_metrics['rel_mae']:>18.6e} {qpinn_full_metrics['rel_mae']:>18.6e}")
    logger.info(f"{'Peak Error':<25s} {pinn_full_metrics['peak_error']:>18.6e} {qpinn_full_metrics['peak_error']:>18.6e}")
    logger.info(f"{'Boundary Error (mean)':<25s} {pinn_full_metrics['mean_boundary_error']:>18.6e} {qpinn_full_metrics['mean_boundary_error']:>18.6e}")
    logger.info(f"{'Training Time (s)':<25s} {pinn_time:>18.2f} {qnn_time:>18.2f}")

    if mse_pinn > 0:
        mse_improvement = ((mse_pinn - mse_qnn) / mse_pinn) * 100
        if mse_improvement > 0:
            logger.info(f"\nQPINN MSE improvement: {mse_improvement:.2f}%")
        else:
            logger.info(f"\nPINN MSE advantage: {-mse_improvement:.2f}%")

    if qnn_time > 0 and pinn_time > 0:
        logger.info(f"Time ratio (QPINN/PINN): {qnn_time / pinn_time:.2f}x")

    # Boundary condition analysis
    logger.info("\n=== Boundary Condition Satisfaction ===")
    u_pinn_reshaped = u_pinn.reshape(nx, ny, nz, nt)
    u_qnn_reshaped = u_qnn.reshape(nx, ny, nz, nt)
    boundary_analysis = []

    for t_idx in [0, nt // 4, nt // 2, 3 * nt // 4, nt - 1]:
        boundary_pinn = np.concatenate([
            u_pinn_reshaped[0, :, :, t_idx].flatten(),
            u_pinn_reshaped[-1, :, :, t_idx].flatten(),
            u_pinn_reshaped[:, 0, :, t_idx].flatten(),
            u_pinn_reshaped[:, -1, :, t_idx].flatten(),
            u_pinn_reshaped[:, :, 0, t_idx].flatten(),
            u_pinn_reshaped[:, :, -1, t_idx].flatten()
        ])
        boundary_qnn = np.concatenate([
            u_qnn_reshaped[0, :, :, t_idx].flatten(),
            u_qnn_reshaped[-1, :, :, t_idx].flatten(),
            u_qnn_reshaped[:, 0, :, t_idx].flatten(),
            u_qnn_reshaped[:, -1, :, t_idx].flatten(),
            u_qnn_reshaped[:, :, 0, t_idx].flatten(),
            u_qnn_reshaped[:, :, -1, t_idx].flatten()
        ])
        t_val = t_idx * T / (nt - 1)
        pinn_avg = float(np.mean(np.abs(boundary_pinn)))
        qpinn_avg = float(np.mean(np.abs(boundary_qnn)))
        boundary_analysis.append({'t': t_val, 'pinn_avg': pinn_avg, 'qpinn_avg': qpinn_avg})
        logger.info(f"  t={t_val:.2f}: PINN avg={pinn_avg:.6f}, QPINN avg={qpinn_avg:.6f}")

    # Thermal energy conservation
    if pinn_full_metrics.get('energy_pred') and pinn_full_metrics.get('energy_true'):
        logger.info("\n=== Thermal Energy Conservation ===")
        t_vals_log = np.linspace(0, T, nt)
        for i in [0, nt//4, nt//2, 3*nt//4, nt-1]:
            if i < len(pinn_full_metrics['energy_true']):
                logger.info(f"  t={t_vals_log[i]:.2f}: Analytical={pinn_full_metrics['energy_true'][i]:.6e}, "
                            f"PINN={pinn_full_metrics['energy_pred'][i]:.6e}, "
                            f"QPINN={qpinn_full_metrics['energy_pred'][i]:.6e}")

    # ============================
    # 5. Generate all output files
    # ============================
    logger.info("\n=== Generating output files ===")

    # Metrics CSV
    save_metrics_over_time_csv(results_dir, u_pinn, u_qnn, u_analytical)
    logger.info("Metrics over time CSV saved")

    # Validation and performance CSVs
    val_files = save_validation_csv(results_dir, pinn_history=pinn_history, qpinn_history=qpinn_history)
    for vf in val_files:
        logger.info(f"Validation metrics CSV saved: {os.path.basename(vf)}")
    perf_file = save_performance_csv(results_dir, pinn_history=pinn_history)
    if perf_file:
        logger.info(f"Performance metrics CSV saved: {os.path.basename(perf_file)}")

    # Plots
    plot_files = visualize_results_comparison(
        results_dir, u_pinn, u_qnn, u_analytical,
        pinn_losses, qnn_losses,
        pinn_history=pinn_history,
        qpinn_history=qpinn_history,
        logger=logger
    )

    # JSON
    save_comparative_results(
        results_dir, pinn_results, qpinn_results,
        mse_pinn=mse_pinn, rel_l2_pinn=rel_l2_pinn,
        mse_qnn=mse_qnn, rel_l2_qnn=rel_l2_qnn,
        pinn_history=pinn_history, qpinn_history=qpinn_history,
        pinn_full_metrics=pinn_full_metrics, qpinn_full_metrics=qpinn_full_metrics,
        pinn_time=pinn_time, qnn_time=qnn_time,
        logger=logger
    )

    # Summary report
    report_path = save_summary_report(
        results_dir, pinn_results, qpinn_results,
        mse_pinn, rel_l2_pinn, mse_qnn, rel_l2_qnn,
        pinn_time, qnn_time,
        pinn_history=pinn_history, qpinn_history=qpinn_history,
        boundary_analysis=boundary_analysis,
        pinn_full_metrics=pinn_full_metrics, qpinn_full_metrics=qpinn_full_metrics
    )
    logger.info(f"Summary report saved: {report_path}")

    # ============================
    # 6. Final summary
    # ============================
    logger.info("\n" + "=" * 60)
    logger.info("Benchmark completed. Output files:")
    logger.info("=" * 60)
    for f_name in sorted(os.listdir(results_dir)):
        f_path = os.path.join(results_dir, f_name)
        size = os.path.getsize(f_path)
        if size > 1024 * 1024:
            size_str = f"{size / (1024 * 1024):.1f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} B"
        logger.info(f"  {f_name:45s} {size_str:>10s}")

    logger.info(f"\nResults directory: {results_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PINN vs QPINN Benchmark')
    parser.add_argument('--backend', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'gpu', 'qpu'],
                        help='Backend selection (default: auto)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON config file (default: None)')
    args = parser.parse_args()
    main(backend=args.backend, config_path=args.config)

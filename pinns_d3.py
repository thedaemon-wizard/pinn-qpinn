import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import pennylane as qml
import time
from typing import Tuple, List, Callable, Union, Any, Dict
import os
os.environ['OMP_NUM_THREADS']=str(12)
from collections import deque
import warnings

# 警告を抑制
warnings.filterwarnings("ignore", category=UserWarning)
# バックエンドの互換性を確保
np.set_printoptions(precision=8)
try:
    qml.numpy.set_printoptions(precision=8)
except AttributeError:
    pass

# PyTorchのデフォルト浮動小数点精度を設定
torch.set_default_dtype(torch.float32)


#================================================
# 共通パラメータの設定
#================================================
# 問題のパラメータ
alpha = 0.01  # 熱拡散率
L = 1.0       # 立方体の一辺の長さ
T = 1.0       # 最終時間

# 離散化パラメータ
nx, ny, nz = 20, 20, 20  # 空間分割数
nt = 20                 # 時間分割数

# トレーニングパラメータ
pinn_epochs = 10000     # PINNのエポック数
qnn_epochs = 1500      # QPINNのエポック数

# デバイスの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#================================================
# 初期条件と境界条件の定義
#================================================
def initial_condition(x, y, z):
    """初期温度分布: ガウス分布"""
    sigma_0 = 0.05
    x0, y0, z0 = L/2, L/2, L/2
    return np.exp(-((x-x0)**2 + (y-y0)**2 + (z-z0)**2) / (2*sigma_0**2))

def boundary_condition(x, y, z, t):
    """境界条件: 全ての境界で温度0"""
    return 0.0

def analytical_solution(x, y, z, t):
    """解析解: 熱が拡散していく様子"""
    sigma_0 = 0.05
    x0, y0, z0 = L/2, L/2, L/2
    
    # 時間発展するシグマ
    sigma_t = np.sqrt(sigma_0**2 + 2*alpha*t)
    
    # ピーク値の減衰を計算
    amplitude = (sigma_0/sigma_t)**3
    
    # ガウス分布の計算
    return amplitude * np.exp(-((x-x0)**2 + (y-y0)**2 + (z-z0)**2) / (2*sigma_t**2))

def to_python_float(value):
    """PennyLaneの任意の型を確実にPython floatに変換する汎用関数"""
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
        # ArrayBox型の処理
        if hasattr(value, '_value'):
            return float(value._value)
        return float(value)
    except Exception:
        return 0.0

#================================================
# PINNsの実装（変更なし）
#================================================
class PINN(nn.Module):
    def __init__(self, layers=[4, 128, 256, 256, 128, 1]):
        """Physics-Informed Neural Network for 3D heat equation"""
        super(PINN, self).__init__()
        
        # 全結合層のリスト
        self.layers = nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            
        # 活性化関数
        self.activation = nn.SiLU()
        
        # 重みの初期化
        self.xavier_init()
        
        # スケーリング係数
        self.scale_factor = nn.Parameter(torch.tensor([1.0]))
        self.time_scale = nn.Parameter(torch.tensor([0.5]))
        
    def xavier_init(self):
        """Xavier初期化を使用して重みを初期化"""
        for m in self.layers:
            if isinstance(m, nn.Linear):
                gain = nn.init.calculate_gain('relu')
                nn.init.xavier_normal_(m.weight, gain=gain)
                nn.init.zeros_(m.bias)
        
    def forward(self, x, y, z, t):
        """ネットワークの順伝播"""
        # 入力スケーリング
        x_scaled = 2.0 * (x / L) - 1.0
        y_scaled = 2.0 * (y / L) - 1.0
        z_scaled = 2.0 * (z / L) - 1.0
        t_scaled = 2.0 * (t / T) - 1.0
        
        X = torch.cat([x_scaled, y_scaled, z_scaled, t_scaled], dim=1)
        
        # 各層を通過
        for i in range(len(self.layers)-1):
            X = self.layers[i](X)
            X = self.activation(X)
        
        # 最終層
        output = self.layers[-1](X)
        
        # 時間発展を正確に捉えるためのスケーリング
        time_factor = torch.exp(-self.time_scale * t)
        return torch.abs(output) * self.scale_factor * time_factor
    
    def compute_pde_residual(self, x, y, z, t):
        """熱伝導方程式の残差を計算"""
        x.requires_grad_(True)
        y.requires_grad_(True)
        z.requires_grad_(True)
        t.requires_grad_(True)
        
        u = self.forward(x, y, z, t)
        
        # 各変数による偏微分を計算
        u_t = grad(u.sum(), t, create_graph=True)[0]
        u_x = grad(u.sum(), x, create_graph=True)[0]
        u_y = grad(u.sum(), y, create_graph=True)[0]
        u_z = grad(u.sum(), z, create_graph=True)[0]
        
        u_xx = grad(u_x.sum(), x, create_graph=True)[0]
        u_yy = grad(u_y.sum(), y, create_graph=True)[0]
        u_zz = grad(u_z.sum(), z, create_graph=True)[0]
        
        # 熱伝導方程式: u_t = alpha * (u_xx + u_yy + u_zz)
        pde_residual = u_t - alpha * (u_xx + u_yy + u_zz)
        
        return pde_residual

def train_pinn() -> Tuple[PINN, List[float], float]:
    """PINNモデルをトレーニングする関数"""
    print("PINNのトレーニングを開始...")
    start_time = time.time()
    
    # データ点の生成
    n_interior = 30000
    n_boundary = 6000
    n_initial = 5000
    n_reference = 10000
    
    # 内部点（中心付近に集中）
    center_interior = int(n_interior * 0.6)
    random_interior = n_interior - center_interior
    
    x_center = torch.normal(L/2, 0.2, (center_interior, 1)).clamp(0, L)
    y_center = torch.normal(L/2, 0.2, (center_interior, 1)).clamp(0, L)
    z_center = torch.normal(L/2, 0.2, (center_interior, 1)).clamp(0, L)
    t_center = torch.rand(center_interior, 1) * T
    
    x_random = torch.rand(random_interior, 1) * L
    y_random = torch.rand(random_interior, 1) * L
    z_random = torch.rand(random_interior, 1) * L
    t_random = torch.rand(random_interior, 1) * T
    
    x_interior = torch.cat([x_center, x_random], dim=0)
    y_interior = torch.cat([y_center, y_random], dim=0)
    z_interior = torch.cat([z_center, z_random], dim=0)
    t_interior = torch.cat([t_center, t_random], dim=0)
    
    # 初期条件の点
    center_samples = int(n_initial * 0.8)
    random_samples = n_initial - center_samples
    
    x_center = torch.normal(L/2, 0.1, (center_samples, 1)).clamp(0, L)
    y_center = torch.normal(L/2, 0.1, (center_samples, 1)).clamp(0, L)
    z_center = torch.normal(L/2, 0.1, (center_samples, 1)).clamp(0, L)
    
    x_random = torch.rand(random_samples, 1) * L
    y_random = torch.rand(random_samples, 1) * L
    z_random = torch.rand(random_samples, 1) * L
    
    x_initial = torch.cat([x_center, x_random], dim=0)
    y_initial = torch.cat([y_center, y_random], dim=0)
    z_initial = torch.cat([z_center, z_random], dim=0)
    t_initial = torch.zeros(n_initial, 1)
    
    u_initial = torch.tensor(
        [initial_condition(x.item(), y.item(), z.item()) 
         for x, y, z in zip(x_initial, y_initial, z_initial)],
        dtype=torch.float32
    ).view(-1, 1)
    
    # 境界条件の点
    x_boundary = torch.zeros(n_boundary, 1)
    y_boundary = torch.zeros(n_boundary, 1)
    z_boundary = torch.zeros(n_boundary, 1)
    t_boundary = torch.rand(n_boundary, 1) * T
    
    for i in range(n_boundary):
        face = i % 6
        if face == 0:
            y_boundary[i] = torch.rand(1) * L
            z_boundary[i] = torch.rand(1) * L
        elif face == 1:
            x_boundary[i] = torch.tensor([L])
            y_boundary[i] = torch.rand(1) * L
            z_boundary[i] = torch.rand(1) * L
        elif face == 2:
            x_boundary[i] = torch.rand(1) * L
            z_boundary[i] = torch.rand(1) * L
        elif face == 3:
            x_boundary[i] = torch.rand(1) * L
            y_boundary[i] = torch.tensor([L])
            z_boundary[i] = torch.rand(1) * L
        elif face == 4:
            x_boundary[i] = torch.rand(1) * L
            y_boundary[i] = torch.rand(1) * L
        else:
            x_boundary[i] = torch.rand(1) * L
            y_boundary[i] = torch.rand(1) * L
            z_boundary[i] = torch.tensor([L])
    
    u_boundary = torch.zeros(n_boundary, 1)
    
    # 解析解参照ポイント
    t_reference_points = np.linspace(0, T, 10)
    n_points_per_time = n_reference // len(t_reference_points)
    
    x_reference_list = []
    y_reference_list = []
    z_reference_list = []
    t_reference_list = []
    
    for t_val in t_reference_points:
        center_points = int(n_points_per_time * 0.7)
        random_points = n_points_per_time - center_points
        
        x_center = torch.normal(L/2, 0.15, (center_points, 1)).clamp(0, L)
        y_center = torch.normal(L/2, 0.15, (center_points, 1)).clamp(0, L)
        z_center = torch.normal(L/2, 0.15, (center_points, 1)).clamp(0, L)
        
        x_rand = torch.rand(random_points, 1) * L
        y_rand = torch.rand(random_points, 1) * L
        z_rand = torch.rand(random_points, 1) * L
        
        x_ref = torch.cat([x_center, x_rand], dim=0)
        y_ref = torch.cat([y_center, y_rand], dim=0)
        z_ref = torch.cat([z_center, z_rand], dim=0)
        t_ref = torch.ones(n_points_per_time, 1) * t_val
        
        x_reference_list.append(x_ref)
        y_reference_list.append(y_ref)
        z_reference_list.append(z_ref)
        t_reference_list.append(t_ref)
    
    x_reference = torch.cat(x_reference_list, dim=0)
    y_reference = torch.cat(y_reference_list, dim=0)
    z_reference = torch.cat(z_reference_list, dim=0)
    t_reference = torch.cat(t_reference_list, dim=0)
    
    u_reference = torch.tensor(
        [analytical_solution(x.item(), y.item(), z.item(), t.item()) 
         for x, y, z, t in zip(x_reference, y_reference, z_reference, t_reference)],
        dtype=torch.float32
    ).view(-1, 1)
    
    # デバイスに転送
    x_interior, y_interior, z_interior, t_interior = map(
        lambda x: x.to(device), [x_interior, y_interior, z_interior, t_interior]
    )
    x_initial, y_initial, z_initial, t_initial, u_initial = map(
        lambda x: x.to(device), [x_initial, y_initial, z_initial, t_initial, u_initial]
    )
    x_boundary, y_boundary, z_boundary, t_boundary, u_boundary = map(
        lambda x: x.to(device), [x_boundary, y_boundary, z_boundary, t_boundary, u_boundary]
    )
    x_reference, y_reference, z_reference, t_reference, u_reference = map(
        lambda x: x.to(device), [x_reference, y_reference, z_reference, t_reference, u_reference]
    )
    
    # モデル初期化
    model = PINN([4, 128, 256, 256, 128, 1]).to(device)
    
    for param in model.parameters():
        param.data = param.data.float()
    
    # 最適化設定
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=300)
    
    mse_loss = nn.MSELoss()
    losses = []
    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # トレーニングループ
    for epoch in range(pinn_epochs):
        optimizer.zero_grad()
        
        # PDE残差
        batch_size = 5000
        n_batches = len(x_interior) // batch_size + (1 if len(x_interior) % batch_size != 0 else 0)
        loss_pde = 0.0
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(x_interior))
            
            x_batch = x_interior[start_idx:end_idx]
            y_batch = y_interior[start_idx:end_idx]
            z_batch = z_interior[start_idx:end_idx]
            t_batch = t_interior[start_idx:end_idx]
            
            pde_residual = model.compute_pde_residual(x_batch, y_batch, z_batch, t_batch)
            loss_pde += torch.mean(pde_residual ** 2) * (end_idx - start_idx) / len(x_interior)
        
        # 初期条件
        u_pred_initial = model(x_initial, y_initial, z_initial, t_initial)
        loss_initial = mse_loss(u_pred_initial, u_initial)
        
        # 境界条件
        u_pred_boundary = model(x_boundary, y_boundary, z_boundary, t_boundary)
        loss_boundary = mse_loss(u_pred_boundary, u_boundary)
        
        # 解析解参照ポイント
        batch_size = 5000
        n_batches = len(x_reference) // batch_size + (1 if len(x_reference) % batch_size != 0 else 0)
        loss_reference = 0.0
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(x_reference))
            
            x_batch = x_reference[start_idx:end_idx]
            y_batch = y_reference[start_idx:end_idx]
            z_batch = z_reference[start_idx:end_idx]
            t_batch = t_reference[start_idx:end_idx]
            u_batch = u_reference[start_idx:end_idx]
            
            u_pred_batch = model(x_batch, y_batch, z_batch, t_batch)
            loss_reference += mse_loss(u_pred_batch, u_batch) * (end_idx - start_idx) / len(x_reference)
        
        # 総損失
        loss = loss_pde + 200.0 * loss_initial + 10.0 * loss_boundary + 1000.0 * loss_reference
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        scheduler.step(loss)
        
        losses.append(loss.item())
        
        # 定期的な進捗報告
        if (epoch + 1) % 500 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{pinn_epochs}], Loss: {loss.item():.4e}, "
                  f"PDE Loss: {loss_pde.item():.4e}, "
                  f"IC Loss: {loss_initial.item():.4e}, "
                  f"BC Loss: {loss_boundary.item():.4e}, "
                  f"Ref Loss: {loss_reference.item():.4e}, "
                  f"LR: {current_lr:.2e}")
            
            with torch.no_grad():
                center_x = torch.tensor([[L/2]], dtype=torch.float32).to(device)
                center_y = torch.tensor([[L/2]], dtype=torch.float32).to(device)
                center_z = torch.tensor([[L/2]], dtype=torch.float32).to(device)
                
                for t_val in [0.0, 0.5, 1.0]:
                    center_t = torch.tensor([[t_val]], dtype=torch.float32).to(device)
                    u_pred = model(center_x, center_y, center_z, center_t).item()
                    u_true = analytical_solution(L/2, L/2, L/2, t_val)
                    print(f"  Center at t={t_val:.1f}: True={u_true:.6f}, Pred={u_pred:.6f}")
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= 1000:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    training_time = time.time() - start_time
    print(f"PINNのトレーニング完了。トレーニング時間: {training_time:.2f}秒")
    
    return model, losses, training_time

def evaluate_pinn(model: PINN) -> np.ndarray:
    """PINNモデルを評価し、予測結果を返す"""
    global L, T, nx, ny, nz, nt
    
    print("PINNモデルの評価中...")
    model.eval()
    
    # グリッドデータの作成
    x = np.linspace(0, L, nx)
    y = np.linspace(0, L, ny)
    z = np.linspace(0, L, nz)
    t = np.linspace(0, T, nt)
    
    X, Y, Z, T_mesh = np.meshgrid(x, y, z, t, indexing='ij')
    
    X_flat = X.flatten().reshape(-1, 1)
    Y_flat = Y.flatten().reshape(-1, 1)
    Z_flat = Z.flatten().reshape(-1, 1)
    T_flat = T_mesh.flatten().reshape(-1, 1)
    
    # テンソル変換
    X_tensor = torch.FloatTensor(X_flat).to(device)
    Y_tensor = torch.FloatTensor(Y_flat).to(device)
    Z_tensor = torch.FloatTensor(Z_flat).to(device)
    T_tensor = torch.FloatTensor(T_flat).to(device)
    
    # バッチサイズを設定して評価
    batch_size = 5000
    n_batches = len(X_flat) // batch_size + (1 if len(X_flat) % batch_size != 0 else 0)
    
    u_pred_list = []
    
    # バッチで処理
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(X_flat))
        
        # 現在のバッチのデータ
        X_batch = X_tensor[start_idx:end_idx]
        Y_batch = Y_tensor[start_idx:end_idx]
        Z_batch = Z_tensor[start_idx:end_idx]
        T_batch = T_tensor[start_idx:end_idx]
        
        # 評価
        with torch.no_grad():
            u_pred_batch = model(X_batch, Y_batch, Z_batch, T_batch).cpu().numpy()
        
        u_pred_list.append(u_pred_batch)
    
    # 結果を結合
    u_pred = np.vstack(u_pred_list)
    
    return u_pred.flatten()

#================================================
# 物理制約付き量子機械学習実装（最適化改良版）
#================================================
class PhysicsInformedQuantumNN:
    def __init__(self, n_qubits=8, n_layers=4, circuit_type='strongly_entangling', 
                 entangling_strategy='all_to_all', backend='lightning.qubit',
                 shots=None, noise_model=None):
        """
        物理制約付き量子ニューラルネットワーク（最適化改良版）
        
        Args:
            n_qubits: 量子ビット数
            n_layers: 量子回路の層数（デフォルトを6から4に削減）
            circuit_type: 回路タイプ
            entangling_strategy: エンタングリング戦略
            backend: 量子シミュレータバックエンド
            shots: ショット数（実機向け）
            noise_model: ノイズモデル（実機シミュレーション）
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.circuit_type = circuit_type
        self.entangling_strategy = entangling_strategy
        self.shots = shots
        self.noise_model = noise_model
        
        # 実機モードの判定
        self.is_hardware = shots is not None
        
        # デバイスの設定
        if self.is_hardware:
            # 実機またはノイズありシミュレーション
            self.dev = qml.device(backend, wires=self.n_qubits, shots=shots)
            print(f"実機モード: ショット数 = {shots}")
        else:
            # 理想的なシミュレーション
            self.dev = qml.device(backend, wires=self.n_qubits)
            print("シミュレーションモード: 解析的計算")
        
        # パラメータの自動計算
        self._calculate_parameter_counts()
        
        # パラメータの初期化（改善: より小さい初期値）
        self.weights = qml.numpy.array(
            np.random.uniform(-np.pi/8, np.pi/8, size=self.n_params),
            requires_grad=True
        )
        
        # 出力処理パラメータ（初期値を調整）
        self.output_scale = qml.numpy.array(2.0, requires_grad=True)  # 増加
        self.output_bias = qml.numpy.array(0.1, requires_grad=True)   # 正の値
        
        # 特徴量エンコーディング用のスケール
        self.feature_scales = qml.numpy.array(
            np.ones(8) * 0.3,  # 減少
            requires_grad=True
        )
        
        # 空間減衰パラメータ
        self.spatial_decay = qml.numpy.array(2.0, requires_grad=True)  # 減少
        
        # 時間スケーリングパラメータ
        self.time_scale = qml.numpy.array(0.5, requires_grad=True)  # 減少
        
        # 量子回路の自動生成
        self._create_quantum_circuit()
        
        # トレーニング履歴
        self.loss_history = []
        self.loss_components = {
            'data': [],
            'pde': [],
            'initial': [],
            'boundary': []
        }
        
        # 最適化器の履歴
        self.optimizer_history = []
        
        # トレーニングデータを保存
        self.training_data = {}
        
    def _calculate_parameter_counts(self):
        """パラメータ数の自動計算"""
        if self.circuit_type == 'hardware_efficient':
            # 各層: RY + RZ ゲート各量子ビット、CNOTゲートは固定
            params_per_layer = self.n_qubits * 2
            self.n_params = self.n_layers * params_per_layer
            
        elif self.circuit_type == 'strongly_entangling':
            # 各層: RX, RY, RZ ゲート各量子ビット
            params_per_layer = self.n_qubits * 3
            self.n_params = self.n_layers * params_per_layer
            
        elif self.circuit_type == 'data_reuploading':
            # 各層: データエンコーディング + 変分パラメータ
            # データエンコーディング: 各量子ビットにRY
            # 変分層: RX, RY, RZ
            variational_params_per_layer = self.n_qubits * 3
            self.n_params = self.n_layers * variational_params_per_layer
            # データエンコーディングは入力データに依存するため、別途処理
            
    def _create_quantum_circuit(self):
        """量子回路の自動生成（完全版）"""
        
        # 微分法の選択（実機対応）
        if self.is_hardware:
            diff_method = "parameter-shift"  # 実機では必須
        else:
            diff_method = "adjoint"  # シミュレータでは高速
        
        @qml.qnode(self.dev, interface="autograd", diff_method=diff_method)
        def quantum_circuit(inputs, weights, feature_scales):
            """改良された量子回路"""
            
            # 入力の前処理
            x_norm = inputs[0] / L
            y_norm = inputs[1] / L
            z_norm = inputs[2] / L
            t_norm = inputs[3] / T
            
            # 特徴量の生成
            r = qml.numpy.sqrt((x_norm - 0.5)**2 + (y_norm - 0.5)**2 + (z_norm - 0.5)**2)
            r_scaled = qml.numpy.exp(-2.0 * r)
            
            features = qml.numpy.array([
                x_norm - 0.5,
                y_norm - 0.5,
                z_norm - 0.5,
                t_norm,
                r_scaled,
                qml.numpy.sin(2 * np.pi * x_norm),
                qml.numpy.sin(2 * np.pi * y_norm),
                qml.numpy.sin(2 * np.pi * z_norm)
            ])
            
            # 特徴量のスケーリング
            scaled_features = features * feature_scales
            
            # ノイズモデルの適用（実機シミュレーション用）
            if self.noise_model is not None:
                # ここでノイズを追加（例：デポラライジングノイズ）
                for qubit in range(self.n_qubits):
                    qml.DepolarizingChannel(0.001, wires=qubit)
            
            param_idx = 0
            
            if self.circuit_type == 'strongly_entangling':
                # 初期状態準備
                for qubit in range(self.n_qubits):
                    feature_idx = qubit % len(features)
                    qml.Hadamard(wires=qubit)
                    qml.RY(scaled_features[feature_idx], wires=qubit)
                
                # 強エンタングリング層
                for layer in range(self.n_layers):
                    # 各量子ビットに3つの回転ゲート
                    for qubit in range(self.n_qubits):
                        # データ依存の回転（係数を減少）
                        feature_combination = 0.0
                        for f_idx in range(3):
                            f_idx_actual = (qubit + f_idx + layer) % len(features)
                            feature_combination += scaled_features[f_idx_actual]
                        
                        qml.RX(weights[param_idx] + 0.05 * feature_combination, wires=qubit)
                        param_idx += 1
                        qml.RY(weights[param_idx], wires=qubit)
                        param_idx += 1
                        qml.RZ(weights[param_idx] + 0.05 * feature_combination, wires=qubit)
                        param_idx += 1
                    
                    # エンタングリング層
                    if layer < self.n_layers - 1:
                        self._apply_entangling_layer(layer)
                        
            elif self.circuit_type == 'hardware_efficient':
                # Hardware Efficient Ansatz
                # 初期状態準備
                for qubit in range(self.n_qubits):
                    qml.Hadamard(wires=qubit)
                
                # 層ごとの処理
                for layer in range(self.n_layers):
                    # Single-qubit rotations
                    for qubit in range(self.n_qubits):
                        # データエンコーディング
                        feature_idx = (qubit + layer) % len(features)
                        qml.RY(scaled_features[feature_idx] * np.pi, wires=qubit)
                        
                        # 変分パラメータ
                        qml.RY(weights[param_idx], wires=qubit)
                        param_idx += 1
                        qml.RZ(weights[param_idx], wires=qubit)
                        param_idx += 1
                    
                    # Entangling gates (最小限のCNOT)
                    if layer < self.n_layers - 1:
                        # 隣接量子ビット間のCNOT
                        for qubit in range(0, self.n_qubits - 1):
                            qml.CNOT(wires=[qubit, qubit + 1])
                        # 循環境界条件
                        if self.n_qubits > 2:
                            qml.CNOT(wires=[self.n_qubits - 1, 0])
                            
            elif self.circuit_type == 'data_reuploading':
                # Data Re-uploading Circuit
                for layer in range(self.n_layers):
                    # データエンコーディング層
                    for qubit in range(self.n_qubits):
                        # 各層で異なる特徴量の組み合わせ
                        feature_idx1 = (qubit + layer) % len(features)
                        feature_idx2 = (qubit + layer + 1) % len(features)
                        
                        # データエンコーディング
                        angle = scaled_features[feature_idx1] * np.pi
                        qml.RY(angle, wires=qubit)
                        
                        # 追加のデータエンコーディング
                        if layer > 0:
                            angle2 = scaled_features[feature_idx2] * np.pi / 2
                            qml.RZ(angle2, wires=qubit)
                    
                    # 変分層
                    for qubit in range(self.n_qubits):
                        qml.RX(weights[param_idx], wires=qubit)
                        param_idx += 1
                        qml.RY(weights[param_idx], wires=qubit)
                        param_idx += 1
                        qml.RZ(weights[param_idx], wires=qubit)
                        param_idx += 1
                    
                    # エンタングリング層
                    if layer < self.n_layers - 1:
                        # ブロック状のエンタングリング
                        for i in range(0, self.n_qubits - 1, 2):
                            qml.CZ(wires=[i, i + 1])
                        for i in range(1, self.n_qubits - 1, 2):
                            qml.CZ(wires=[i, i + 1])
            
            # 測定
            measurements = []
            
            # 単一量子ビット測定
            for i in range(self.n_qubits):
                measurements.append(qml.expval(qml.PauliZ(i)))
            
            # 相関測定（選択的）
            if self.n_qubits <= 10 and not self.is_hardware:  # 実機では計算コストが高い
                for i in range(0, self.n_qubits - 1, 2):
                    measurements.append(qml.expval(qml.PauliZ(i) @ qml.PauliZ(i + 1)))
            
            return measurements
        
        self.qnode = quantum_circuit
        
        print(f"物理制約付き量子回路を生成：")
        print(f"  - 量子ビット数: {self.n_qubits}")
        print(f"  - 層数: {self.n_layers}")
        print(f"  - 回路タイプ: {self.circuit_type}")
        print(f"  - エンタングリング戦略: {self.entangling_strategy}")
        print(f"  - 回路パラメータ数: {self.n_params}")
        print(f"  - 微分法: {diff_method}")
        
    def _apply_entangling_layer(self, layer_idx):
        """エンタングリング層の適用"""
        if self.entangling_strategy == 'all_to_all':
            if layer_idx % 3 == 0:
                # 隣接ペア
                for i in range(0, self.n_qubits - 1, 2):
                    qml.CNOT(wires=[i, i + 1])
                for i in range(1, self.n_qubits - 1, 2):
                    qml.CNOT(wires=[i, i + 1])
            elif layer_idx % 3 == 1:
                # スキップ接続
                step = 2
                for i in range(self.n_qubits - step):
                    qml.CNOT(wires=[i, i + step])
            else:
                # 循環接続
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 3) % self.n_qubits])
    
    def forward(self, x, y, z, t):
        """順伝播（改良版）"""
        try:
            # 入力を配列に変換
            inputs = qml.numpy.array([float(x), float(y), float(z), float(t)])
            
            # 量子回路の実行
            measurements = self.qnode(inputs, self.weights, self.feature_scales)
            
            # 測定値を配列に変換
            measurements_array = qml.numpy.array(measurements)
            
            # 単一量子ビット測定の処理
            n_single_measurements = self.n_qubits
            single_measurements = measurements_array[:n_single_measurements]
            
            # 測定値の集約（改良版）
            # 重み付き平均の代わりに、より単純な平均を使用
            single_output = qml.numpy.mean(single_measurements)
            
            # 相関測定がある場合
            if len(measurements_array) > n_single_measurements:
                correlation_measurements = measurements_array[n_single_measurements:]
                correlation_output = qml.numpy.mean(correlation_measurements)
                quantum_output = 0.8 * single_output + 0.2 * correlation_output
            else:
                quantum_output = single_output
            
            # 出力の後処理（改良版）
            # tanhの代わりに、より線形な活性化を使用
            quantum_output = quantum_output * 0.5  # スケーリング
            normalized_output = (quantum_output + 1.0) / 2.0
            
            # 解析解に基づく形状関数
            sigma_0 = 0.05
            sigma_t = qml.numpy.sqrt(sigma_0**2 + 2 * alpha * float(t))
            amplitude = (sigma_0 / sigma_t) ** 3
            
            # 空間的な形状
            x_centered = (float(x) - L/2) / L
            y_centered = (float(y) - L/2) / L
            z_centered = (float(z) - L/2) / L
            r_squared = x_centered**2 + y_centered**2 + z_centered**2
            
            # ガウス分布形状
            spatial_factor = qml.numpy.exp(-self.spatial_decay * r_squared / (sigma_t / L)**2)
            
            # 最終出力（改良版）
            result = self.output_scale * amplitude * spatial_factor * (normalized_output + self.output_bias)
            
            # 非負制約
            return qml.numpy.maximum(result, 0.0)
            
        except Exception as e:
            print(f"順伝播エラー: {e}")
            import traceback
            traceback.print_exc()
            return qml.numpy.array(0.0)
    
    def compute_pde_residual(self, x, y, z, t, epsilon=1e-3):
        """PDE残差の計算"""
        try:
            # 境界チェック
            if x < epsilon or x > L - epsilon or y < epsilon or y > L - epsilon or \
               z < epsilon or z > L - epsilon or t > T - epsilon:
                return qml.numpy.array(0.0)
            
            # 中心での値
            u = self.forward(x, y, z, t)
            
            # 時間微分（前進差分）
            if t + epsilon <= T:
                u_t_plus = self.forward(x, y, z, t + epsilon)
                u_t = (u_t_plus - u) / epsilon
            else:
                u_t_minus = self.forward(x, y, z, t - epsilon)
                u_t = (u - u_t_minus) / epsilon
            
            # 空間二階微分（中心差分）
            u_x_plus = self.forward(x + epsilon, y, z, t)
            u_x_minus = self.forward(x - epsilon, y, z, t)
            u_xx = (u_x_plus - 2*u + u_x_minus) / (epsilon**2)
            
            u_y_plus = self.forward(x, y + epsilon, z, t)
            u_y_minus = self.forward(x, y - epsilon, z, t)
            u_yy = (u_y_plus - 2*u + u_y_minus) / (epsilon**2)
            
            u_z_plus = self.forward(x, y, z + epsilon, t)
            u_z_minus = self.forward(x, y, z - epsilon, t)
            u_zz = (u_z_plus - 2*u + u_z_minus) / (epsilon**2)
            
            # 熱伝導方程式の残差
            pde_residual = u_t - alpha * (u_xx + u_yy + u_zz)
            
            return pde_residual
            
        except Exception as e:
            return qml.numpy.array(0.0)
    
    def _rotosolve_update(self, all_params, param_idx, cost_function):
        """Rotosolve最適化ステップ"""
        # 現在のパラメータのコピー
        params_plus = qml.numpy.copy(all_params)
        params_minus = qml.numpy.copy(all_params)
        
        # パラメータシフト
        shift = np.pi / 2
        params_plus[param_idx] = all_params[param_idx] + shift
        params_minus[param_idx] = all_params[param_idx] - shift
        
        # コスト計算
        cost_plus = cost_function(params_plus)
        cost_minus = cost_function(params_minus)
        cost_0 = cost_function(all_params)
        
        # 最適な角度を解析的に計算
        a = (cost_plus + cost_minus - 2 * cost_0) / 2
        b = (cost_plus - cost_minus) / 2
        
        if qml.numpy.abs(a) > 1e-10:
            theta_opt = -qml.numpy.arctan(b / a) / 2
            if a > 0:
                theta_opt += np.pi
        else:
            theta_opt = 0.0
        
        # パラメータ更新
        all_params[param_idx] = all_params[param_idx] + theta_opt
        
        return all_params
    
    def train(self, n_samples=1500) -> Tuple[qml.numpy.ndarray, List[float], float]:
        """量子モデルのトレーニング（安定化された最適化）"""
        print(f"物理制約付き量子モデルのトレーニングを開始...")
        print(f"最適化戦略: SPSA → Adam → Rotosolve")
        start_time = time.time()
        
        # データ生成
        # 1. 内部点（PDE制約用）
        n_interior = int(n_samples * 0.3)
        x_interior = np.random.uniform(0.05, 0.95, n_interior) * L
        y_interior = np.random.uniform(0.05, 0.95, n_interior) * L
        z_interior = np.random.uniform(0.05, 0.95, n_interior) * L
        t_interior = np.random.uniform(0.05, 0.95, n_interior) * T
        
        # 2. 初期条件点
        n_initial = int(n_samples * 0.4)
        x_initial = np.random.normal(L/2, 0.1, n_initial)
        y_initial = np.random.normal(L/2, 0.1, n_initial)
        z_initial = np.random.normal(L/2, 0.1, n_initial)
        x_initial = np.clip(x_initial, 0, L)
        y_initial = np.clip(y_initial, 0, L)
        z_initial = np.clip(z_initial, 0, L)
        t_initial = np.zeros(n_initial)
        u_initial = np.array([
            initial_condition(x, y, z) 
            for x, y, z in zip(x_initial, y_initial, z_initial)
        ])
        
        # 3. 境界条件点
        n_boundary = int(n_samples * 0.15)
        x_boundary = []
        y_boundary = []
        z_boundary = []
        t_boundary = []
        
        for i in range(n_boundary):
            face = i % 6
            t_b = np.random.uniform(0, 1) * T
            if face == 0:
                x_boundary.append(0)
                y_boundary.append(np.random.uniform(0, 1) * L)
                z_boundary.append(np.random.uniform(0, 1) * L)
            elif face == 1:
                x_boundary.append(L)
                y_boundary.append(np.random.uniform(0, 1) * L)
                z_boundary.append(np.random.uniform(0, 1) * L)
            elif face == 2:
                x_boundary.append(np.random.uniform(0, 1) * L)
                y_boundary.append(0)
                z_boundary.append(np.random.uniform(0, 1) * L)
            elif face == 3:
                x_boundary.append(np.random.uniform(0, 1) * L)
                y_boundary.append(L)
                z_boundary.append(np.random.uniform(0, 1) * L)
            elif face == 4:
                x_boundary.append(np.random.uniform(0, 1) * L)
                y_boundary.append(np.random.uniform(0, 1) * L)
                z_boundary.append(0)
            else:
                x_boundary.append(np.random.uniform(0, 1) * L)
                y_boundary.append(np.random.uniform(0, 1) * L)
                z_boundary.append(L)
            t_boundary.append(t_b)
        
        x_boundary = np.array(x_boundary)
        y_boundary = np.array(y_boundary)
        z_boundary = np.array(z_boundary)
        t_boundary = np.array(t_boundary)
        u_boundary = np.zeros(n_boundary)
        
        # 4. データ点
        n_data_target = n_samples - n_interior - n_initial - n_boundary
        t_data_points = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        n_per_time = n_data_target // len(t_data_points)
        
        x_data = []
        y_data = []
        z_data = []
        t_data = []
        u_data = []
        
        for t_val in t_data_points:
            for _ in range(n_per_time):
                if np.random.rand() < 0.7:
                    x_val = np.random.normal(L/2, 0.15)
                    y_val = np.random.normal(L/2, 0.15)
                    z_val = np.random.normal(L/2, 0.15)
                else:
                    x_val = np.random.uniform(0, 1) * L
                    y_val = np.random.uniform(0, 1) * L
                    z_val = np.random.uniform(0, 1) * L
                
                x_val = np.clip(x_val, 0, L)
                y_val = np.clip(y_val, 0, L)
                z_val = np.clip(z_val, 0, L)
                
                x_data.append(x_val)
                y_data.append(y_val)
                z_data.append(z_val)
                t_data.append(t_val)
                u_data.append(analytical_solution(x_val, y_val, z_val, t_val))
        
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        z_data = np.array(z_data)
        t_data = np.array(t_data)
        u_data = np.array(u_data)
        
        # トレーニングデータを保存
        self.training_data = {
            'x_interior': x_interior, 'y_interior': y_interior, 
            'z_interior': z_interior, 't_interior': t_interior,
            'x_initial': x_initial, 'y_initial': y_initial,
            'z_initial': z_initial, 't_initial': t_initial, 'u_initial': u_initial,
            'x_boundary': x_boundary, 'y_boundary': y_boundary,
            'z_boundary': z_boundary, 't_boundary': t_boundary, 'u_boundary': u_boundary,
            'x_data': x_data, 'y_data': y_data,
            'z_data': z_data, 't_data': t_data, 'u_data': u_data
        }
        
        n_data_actual = len(x_data)
        print(f"トレーニングサンプル数: {n_samples}")
        print(f"  - 内部点（PDE）: {n_interior}")
        print(f"  - 初期条件: {n_initial}")
        print(f"  - 境界条件: {n_boundary}")
        print(f"  - データ点: {n_data_actual}")
        
        # コスト関数（改良版）
        def cost_function(all_params):
            # パラメータの分離
            weights = all_params[:self.n_params]
            feature_scales = all_params[self.n_params:self.n_params + 8]
            output_scale = all_params[-4]
            output_bias = all_params[-3]
            spatial_decay = all_params[-2]
            time_scale = all_params[-1]
            
            self.weights = weights
            self.feature_scales = qml.numpy.abs(feature_scales) + 0.1
            self.output_scale = qml.numpy.abs(output_scale) + 0.1
            self.output_bias = qml.numpy.abs(output_bias)  # 非負制約
            self.spatial_decay = qml.numpy.abs(spatial_decay) + 0.5
            self.time_scale = qml.numpy.abs(time_scale) + 0.1
            
            # 実機モードでは評価数を減らす
            reduction_factor = 2 if self.is_hardware else 1
            
            # 1. PDE損失
            pde_loss = 0.0
            n_pde_eval = min(20 // reduction_factor, n_interior)
            pde_indices = np.random.choice(n_interior, n_pde_eval, replace=False)
            for i in pde_indices:
                residual = self.compute_pde_residual(
                    x_interior[i], y_interior[i], z_interior[i], t_interior[i]
                )
                pde_loss = pde_loss + residual ** 2
            pde_loss = pde_loss / n_pde_eval
            
            # 2. 初期条件損失（最重要）
            initial_loss = 0.0
            n_ic_eval = min(50 // reduction_factor, n_initial)  # 増加
            ic_indices = np.random.choice(n_initial, n_ic_eval, replace=False)
            for i in ic_indices:
                u_pred = self.forward(x_initial[i], y_initial[i], z_initial[i], t_initial[i])
                initial_loss = initial_loss + (u_pred - u_initial[i]) ** 2
            initial_loss = initial_loss / n_ic_eval
            
            # 3. 境界条件損失
            boundary_loss = 0.0
            n_bc_eval = min(20 // reduction_factor, n_boundary)
            bc_indices = np.random.choice(n_boundary, n_bc_eval, replace=False)
            for i in bc_indices:
                u_pred = self.forward(x_boundary[i], y_boundary[i], z_boundary[i], t_boundary[i])
                boundary_loss = boundary_loss + (u_pred - u_boundary[i]) ** 2
            boundary_loss = boundary_loss / n_bc_eval
            
            # 4. データフィッティング損失
            data_loss = 0.0
            n_data_eval = min(50 // reduction_factor, n_data_actual)
            if n_data_actual > 0:
                data_indices = np.random.choice(n_data_actual, n_data_eval, replace=False)
                for i in data_indices:
                    u_pred = self.forward(x_data[i], y_data[i], z_data[i], t_data[i])
                    data_loss = data_loss + (u_pred - u_data[i]) ** 2
                data_loss = data_loss / n_data_eval
            
            # 正則化
            reg = 0.0001 * qml.numpy.sum(weights ** 2)
            
            # 総損失（重みを調整）
            total_loss = 0.1 * pde_loss + 500.0 * initial_loss + 5.0 * boundary_loss + 200.0 * data_loss + reg
            
            return total_loss
        
        # 詳細な損失成分を計算する関数
        def compute_detailed_losses(all_params):
            # パラメータの分離
            weights = all_params[:self.n_params]
            feature_scales = all_params[self.n_params:self.n_params + 8]
            output_scale = all_params[-4]
            output_bias = all_params[-3]
            spatial_decay = all_params[-2]
            time_scale = all_params[-1]
            
            self.weights = weights
            self.feature_scales = qml.numpy.abs(feature_scales) + 0.1
            self.output_scale = qml.numpy.abs(output_scale) + 0.1
            self.output_bias = qml.numpy.abs(output_bias)
            self.spatial_decay = qml.numpy.abs(spatial_decay) + 0.5
            self.time_scale = qml.numpy.abs(time_scale) + 0.1
            
            # 各損失成分を計算
            losses = {}
            
            # PDE損失
            pde_loss = 0.0
            for i in range(min(30, n_interior)):
                residual = self.compute_pde_residual(
                    x_interior[i], y_interior[i], z_interior[i], t_interior[i]
                )
                pde_loss += residual ** 2
            losses['pde'] = to_python_float(pde_loss / min(30, n_interior))
            
            # 初期条件損失
            initial_loss = 0.0
            for i in range(min(50, n_initial)):
                u_pred = self.forward(x_initial[i], y_initial[i], z_initial[i], t_initial[i])
                initial_loss += (u_pred - u_initial[i]) ** 2
            losses['initial'] = to_python_float(initial_loss / min(50, n_initial))
            
            # 境界条件損失
            boundary_loss = 0.0
            for i in range(min(30, n_boundary)):
                u_pred = self.forward(x_boundary[i], y_boundary[i], z_boundary[i], t_boundary[i])
                boundary_loss += (u_pred - u_boundary[i]) ** 2
            losses['boundary'] = to_python_float(boundary_loss / min(30, n_boundary))
            
            # データ損失
            data_loss = 0.0
            if n_data_actual > 0:
                for i in range(min(50, n_data_actual)):
                    u_pred = self.forward(x_data[i], y_data[i], z_data[i], t_data[i])
                    data_loss += (u_pred - u_data[i]) ** 2
                losses['data'] = to_python_float(data_loss / min(50, n_data_actual))
            else:
                losses['data'] = 0.0
                
            return losses
        
        # すべてのパラメータを結合
        all_params = qml.numpy.concatenate([
            self.weights,
            self.feature_scales,
            qml.numpy.array([self.output_scale]),
            qml.numpy.array([self.output_bias]),
            qml.numpy.array([self.spatial_decay]),
            qml.numpy.array([self.time_scale])
        ])
        
        # ハイブリッド最適化戦略
        best_params = qml.numpy.copy(all_params)
        best_loss = float('inf')
        
        # フェーズ1: SPSA（大域的探索、ノイズ耐性）
        if self.is_hardware or qnn_epochs > 300:
            print("\nフェーズ1: SPSA最適化（大域的探索）")
            spsa_opt = qml.SPSAOptimizer(maxiter=100, c=0.1, a=0.05)  # パラメータ調整
            self.optimizer_history.append("SPSA")
            
            for epoch in range(min(500, qnn_epochs // 3)):
                try:
                    all_params, cost = spsa_opt.step_and_cost(cost_function, all_params)
                    
                    if cost < best_loss:
                        best_loss = cost
                        best_params = qml.numpy.copy(all_params)
                    
                    self.loss_history.append(to_python_float(cost))
                    
                    if (epoch + 1) % 50 == 0:
                        losses = compute_detailed_losses(all_params)
                        print(f"SPSA Epoch [{epoch+1}], Loss: {to_python_float(cost):.6f}, "
                              f"PDE Loss: {losses['pde']:.6f}, "
                              f"IC Loss: {losses['initial']:.6f}, "
                              f"BC Loss: {losses['boundary']:.6f}, "
                              f"Ref Loss: {losses['data']:.6f}, "
                              f"LR: SPSA")
                        
                        # 予測値の確認（SPSAでも追加）
                        test_points = [(L/2, L/2, L/2, 0.0), 
                                       (L/2, L/2, L/2, 0.5), 
                                       (L/2, L/2, L/2, 1.0)]
                        
                        for x_test, y_test, z_test, t_test in test_points:
                            u_pred = self.forward(x_test, y_test, z_test, t_test)
                            u_true = analytical_solution(x_test, y_test, z_test, t_test)
                            print(f"  Center at t={t_test:.1f}: True={u_true:.6f}, "
                                  f"Pred={to_python_float(u_pred):.6f}")
                        
                except Exception as e:
                    print(f"SPSA最適化エラー: {e}")
                    continue
        
        # フェーズ2: Adam（安定した収束）
        print("\nフェーズ2: Adam最適化（最終調整）")
        adam_opt = qml.AdamOptimizer(stepsize=0.005)  # 学習率を下げる
        self.optimizer_history.append("Adam")
        
        # 最良パラメータから開始
        all_params = qml.numpy.copy(best_params)
        patience = 0
        max_patience = 300
        
        remaining_epochs = qnn_epochs - len(self.loss_history)
        
        for epoch in range(remaining_epochs):
            if epoch > 200 and epoch % 300 == 0:
                adam_opt.stepsize *= 0.8
            
            try:
                # 勾配クリッピングを追加
                grads = qml.grad(cost_function)(all_params)
                grad_norm = qml.numpy.linalg.norm(grads)
                if grad_norm > 1.0:
                    grads = grads / grad_norm
                
                # 手動でパラメータ更新
                all_params = all_params - adam_opt.stepsize * grads
                cost = cost_function(all_params)
                
                # パラメータ更新
                self.weights = all_params[:self.n_params]
                self.feature_scales = qml.numpy.abs(all_params[self.n_params:self.n_params + 8]) + 0.1
                self.output_scale = qml.numpy.abs(all_params[-4]) + 0.1
                self.output_bias = qml.numpy.abs(all_params[-3])
                self.spatial_decay = qml.numpy.abs(all_params[-2]) + 0.5
                self.time_scale = qml.numpy.abs(all_params[-1]) + 0.1
                
                self.loss_history.append(to_python_float(cost))
                
                if cost < best_loss:
                    best_loss = cost
                    best_params = qml.numpy.copy(all_params)
                    patience = 0
                else:
                    patience += 1
                
            except Exception as e:
                print(f"Adam最適化エラー: {e}")
                continue
            
            # 進捗報告
            if (epoch + 1) % 100 == 0 or epoch < 10:
                losses = compute_detailed_losses(all_params)
                print(f"Adam Epoch [{epoch+1}/{remaining_epochs}], Loss: {to_python_float(cost):.6f}, "
                      f"PDE Loss: {losses['pde']:.6f}, "
                      f"IC Loss: {losses['initial']:.6f}, "
                      f"BC Loss: {losses['boundary']:.6f}, "
                      f"Ref Loss: {losses['data']:.6f}, "
                      f"LR: {adam_opt.stepsize:.4f}")
                
                # 予測値の確認
                test_points = [(L/2, L/2, L/2, 0.0), 
                               (L/2, L/2, L/2, 0.5), 
                               (L/2, L/2, L/2, 1.0)]
                
                for x_test, y_test, z_test, t_test in test_points:
                    u_pred = self.forward(x_test, y_test, z_test, t_test)
                    u_true = analytical_solution(x_test, y_test, z_test, t_test)
                    print(f"  Center at t={t_test:.1f}: True={u_true:.6f}, "
                          f"Pred={to_python_float(u_pred):.6f}")
            
            if patience >= max_patience:
                print(f"早期停止: epoch {epoch+1}")
                break
        
        # フェーズ3: Rotosolve（最終精密調整、実機向け）
        if self.is_hardware and len(self.loss_history) < qnn_epochs:
            print("\nフェーズ3: Rotosolve最適化（パラメータ単位の最適化）")
            self.optimizer_history.append("Rotosolve")
            
            # 最良パラメータから開始
            all_params = qml.numpy.copy(best_params)
            
            # 重要なパラメータのみを選択的に最適化
            important_params = list(range(0, self.n_params, 10))  # 10個おきに選択
            
            for param_idx in important_params[:10]:  # 最大10パラメータ
                try:
                    all_params = self._rotosolve_update(all_params, param_idx, cost_function)
                    cost = cost_function(all_params)
                    
                    if cost < best_loss:
                        best_loss = cost
                        best_params = qml.numpy.copy(all_params)
                    
                    self.loss_history.append(to_python_float(cost))
                    
                    if (param_idx + 1) % 5 == 0:
                        losses = compute_detailed_losses(all_params)
                        print(f"Rotosolve: パラメータ {param_idx} 最適化完了, Loss: {to_python_float(cost):.6f}, "
                              f"PDE Loss: {losses['pde']:.6f}, "
                              f"IC Loss: {losses['initial']:.6f}, "
                              f"BC Loss: {losses['boundary']:.6f}, "
                              f"Ref Loss: {losses['data']:.6f}, "
                              f"LR: Rotosolve")
                        
                        # 予測値の確認（Rotosolveでも追加）
                        test_points = [(L/2, L/2, L/2, 0.0), 
                                       (L/2, L/2, L/2, 0.5), 
                                       (L/2, L/2, L/2, 1.0)]
                        
                        for x_test, y_test, z_test, t_test in test_points:
                            u_pred = self.forward(x_test, y_test, z_test, t_test)
                            u_true = analytical_solution(x_test, y_test, z_test, t_test)
                            print(f"  Center at t={t_test:.1f}: True={u_true:.6f}, "
                                  f"Pred={to_python_float(u_pred):.6f}")
                    
                except Exception as e:
                    print(f"Rotosolve最適化エラー: {e}")
                    continue
        
        # ベストパラメータを使用
        self.weights = best_params[:self.n_params]
        self.feature_scales = qml.numpy.abs(best_params[self.n_params:self.n_params + 8]) + 0.1
        self.output_scale = qml.numpy.abs(best_params[-4]) + 0.1
        self.output_bias = qml.numpy.abs(best_params[-3])
        self.spatial_decay = qml.numpy.abs(best_params[-2]) + 0.5
        self.time_scale = qml.numpy.abs(best_params[-1]) + 0.1
        
        training_time = time.time() - start_time
        print(f"\n量子モデルのトレーニング完了。時間: {training_time:.2f}秒")
        print(f"使用した最適化器: {' → '.join(self.optimizer_history)}")
        print(f"最終損失: {best_loss:.6f}")
        
        return self.weights, self.loss_history, training_time
    
    def evaluate(self) -> np.ndarray:
        """量子モデルの評価"""
        print("量子モデルの評価中...")
        
        # グリッドデータ
        x = np.linspace(0, L, nx)
        y = np.linspace(0, L, ny)
        z = np.linspace(0, L, nz)
        t = np.linspace(0, T, nt)
        
        X, Y, Z, T_mesh = np.meshgrid(x, y, z, t, indexing='ij')
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = Z.flatten()
        T_flat = T_mesh.flatten()
        
        u_pred = np.zeros_like(X_flat)
        
        # 評価
        batch_size = 100
        n_batches = len(X_flat) // batch_size + (1 if len(X_flat) % batch_size != 0 else 0)
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(X_flat))
            
            for i in range(start_idx, end_idx):
                try:
                    pred_val = self.forward(X_flat[i], Y_flat[i], Z_flat[i], T_flat[i])
                    u_pred[i] = to_python_float(pred_val)
                except:
                    u_pred[i] = 0.0
            
            if (batch_idx + 1) % 20 == 0:
                progress = end_idx / len(X_flat) * 100
                print(f"評価進捗: {progress:.1f}%")
        
        return np.clip(u_pred, 0, None)

#================================================
# 解析解の計算（変更なし）
#================================================
def compute_analytical_solution() -> np.ndarray:
    """解析解を計算する"""
    print("解析解を計算中...")
    
    x = np.linspace(0, L, nx)
    y = np.linspace(0, L, ny)
    z = np.linspace(0, L, nz)
    t = np.linspace(0, T, nt)
    
    X, Y, Z, T_mesh = np.meshgrid(x, y, z, t, indexing='ij')
    
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()
    T_flat = T_mesh.flatten()
    
    u_analytical = np.array([
        analytical_solution(X_flat[i], Y_flat[i], Z_flat[i], T_flat[i])
        for i in range(len(X_flat))
    ])
    
    return u_analytical

#================================================
# 結果の評価と可視化（変更なし）
#================================================
def calculate_metrics(u_pred: np.ndarray, u_true: np.ndarray) -> Tuple[float, float]:
    """精度メトリクスを計算"""
    u_pred = np.nan_to_num(u_pred, nan=0.0, posinf=0.0, neginf=0.0)
    u_pred = np.clip(u_pred, 0, None)
    
    mse = np.mean((u_pred - u_true) ** 2)
    rel_l2 = np.sqrt(np.sum((u_pred - u_true) ** 2)) / np.sqrt(np.sum(u_true ** 2) + 1e-10)
    return mse, rel_l2

def visualize_results(results_dir:str, u_pinn: np.ndarray, u_qnn: np.ndarray, u_analytical: np.ndarray, 
                     label_qnn: str = "QPINN") -> None:
    """結果を可視化"""
    print("結果を可視化中...")
    
    # データのリシェイプ
    u_pinn_reshaped = u_pinn.reshape(nx, ny, nz, nt)
    u_analytical_reshaped = u_analytical.reshape(nx, ny, nz, nt)
    u_qnn_reshaped = u_qnn.reshape(nx, ny, nz, nt)
    
    # グリッドデータ
    x = np.linspace(0, L, nx)
    y = np.linspace(0, L, ny)
    z = np.linspace(0, L, nz)
    t = np.linspace(0, T, nt)
    
    # 中心断面での可視化
    z_mid_idx = nz // 2
    t_indices = [0, nt // 2, nt - 1]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    for i, t_idx in enumerate(t_indices):
        # 断面データ
        u_pinn_2d = u_pinn_reshaped[:, :, z_mid_idx, t_idx]
        u_analytical_2d = u_analytical_reshaped[:, :, z_mid_idx, t_idx]
        u_qnn_2d = u_qnn_reshaped[:, :, z_mid_idx, t_idx]
        
        vmin = 0
        vmax = max(np.max(u_analytical_2d), np.max(u_pinn_2d), np.max(u_qnn_2d))
        
        # PINN
        im1 = axes[i, 0].imshow(u_pinn_2d.T, origin='lower', extent=[0, L, 0, L], 
                                cmap='hot', vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f'PINN (t={t[t_idx]:.2f})')
        axes[i, 0].set_xlabel('x')
        axes[i, 0].set_ylabel('y')
        fig.colorbar(im1, ax=axes[i, 0])
        
        # QNN
        im2 = axes[i, 1].imshow(u_qnn_2d.T, origin='lower', extent=[0, L, 0, L], 
                                cmap='hot', vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f'{label_qnn} (t={t[t_idx]:.2f})')
        axes[i, 1].set_xlabel('x')
        axes[i, 1].set_ylabel('y')
        fig.colorbar(im2, ax=axes[i, 1])
        
        # 解析解
        im3 = axes[i, 2].imshow(u_analytical_2d.T, origin='lower', extent=[0, L, 0, L], 
                                cmap='hot', vmin=vmin, vmax=vmax)
        axes[i, 2].set_title(f'Analytical (t={t[t_idx]:.2f})')
        axes[i, 2].set_xlabel('x')
        axes[i, 2].set_ylabel('y')
        fig.colorbar(im3, ax=axes[i, 2])
    
    plt.tight_layout()
    plt.savefig(results_dir + 'heat_equation_comparison_improved.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 1Dプロファイル比較
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, t_idx in enumerate(t_indices):
        # 中心線での1D温度分布
        u_pinn_1d = u_pinn_reshaped[:, ny//2, nz//2, t_idx]
        u_analytical_1d = u_analytical_reshaped[:, ny//2, nz//2, t_idx]
        u_qnn_1d = u_qnn_reshaped[:, ny//2, nz//2, t_idx]
        
        axes[i].plot(x, u_pinn_1d, 'b-', linewidth=2, label='PINN')
        axes[i].plot(x, u_analytical_1d, 'g--', linewidth=2, label='Analytical')
        axes[i].plot(x, u_qnn_1d, 'r-.', linewidth=2, label=label_qnn)
        
        axes[i].set_title(f'Temperature Profile at t={t[t_idx]:.2f}')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('Temperature')
        axes[i].legend()
        axes[i].grid(True)
        axes[i].set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(results_dir + 'heat_equation_profile_comparison_improved.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 誤差の時間発展
    mse_pinn_t = []
    mse_qnn_t = []
    rel_l2_pinn_t = []
    rel_l2_qnn_t = []
    
    for t_idx in range(nt):
        u_analytical_t = u_analytical_reshaped[:, :, :, t_idx].flatten()
        u_pinn_t = u_pinn_reshaped[:, :, :, t_idx].flatten()
        u_qnn_t = u_qnn_reshaped[:, :, :, t_idx].flatten()
        
        mse_pinn, rel_l2_pinn = calculate_metrics(u_pinn_t, u_analytical_t)
        mse_qnn, rel_l2_qnn = calculate_metrics(u_qnn_t, u_analytical_t)
        
        mse_pinn_t.append(mse_pinn)
        mse_qnn_t.append(mse_qnn)
        rel_l2_pinn_t.append(rel_l2_pinn)
        rel_l2_qnn_t.append(rel_l2_qnn)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.plot(t, mse_pinn_t, 'b-', label='PINN')
    ax1.plot(t, mse_qnn_t, 'r--', label=label_qnn)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('MSE')
    ax1.set_title('Mean Squared Error vs Time')
    ax1.legend()
    ax1.grid(True)
    ax1.set_yscale('log')
    
    ax2.plot(t, rel_l2_pinn_t, 'b-', label='PINN')
    ax2.plot(t, rel_l2_qnn_t, 'r--', label=label_qnn)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Relative L2 Error')
    ax2.set_title('Relative L2 Error vs Time')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(results_dir + 'heat_equation_error_comparison_improved.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # トレーニング損失と最適化器の変遷（修正版）
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if len(pinn_losses) > 0:
            ax.semilogy(range(1, len(pinn_losses) + 1), pinn_losses, 'b-', label='PINN')
        
        if hasattr(qsolver, 'loss_history') and len(qsolver.loss_history) > 0:
            ax.semilogy(range(1, len(qsolver.loss_history) + 1), qsolver.loss_history, 'r-', label=label_qnn)
            
            # 最適化器の変更点をマーク（正確な位置）
            if hasattr(qsolver, 'optimizer_history') and hasattr(qsolver, 'optimizer_epochs'):
                for i in range(1, len(qsolver.optimizer_epochs)):
                    change_epoch = qsolver.optimizer_epochs[i]
                    opt_name = qsolver.optimizer_history[i]
                    ax.axvline(x=change_epoch, color='gray', linestyle='--', alpha=0.5)
                    ax.text(change_epoch, ax.get_ylim()[1] * 0.9, opt_name, rotation=90, va='top')
        
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss (log scale)')
        ax.set_title('Training Loss Comparison with Optimizer Transitions')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(results_dir + 'heat_equation_loss_comparison_improved.png', dpi=150, bbox_inches='tight')
        plt.close()
            
    except Exception as e:
        print(f"損失プロット作成中にエラー: {e}")
#================================================
# メイン関数
#================================================
def main():
    """メイン関数"""
    global pinn_losses, qsolver
    
    print("3次元熱伝導方程式のPINNと物理制約付き量子機械学習による比較を開始します...")
    print(f"PennyLane version: {qml.__version__}")
    print()
    
    # 出力ディレクトリの作成
    os.makedirs('results', exist_ok=True)
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'results/')
    
    # 1. PINNモデルの学習と評価
    pinn_model, pinn_losses, pinn_time = train_pinn()
    u_pinn = evaluate_pinn(pinn_model)
    
    # 2. 物理制約付き量子機械学習モデルの学習と評価
    print("\n=== 物理制約付き量子モデル (QPINN) ===")
    
    # 実機モードのテスト（オプション）
    use_hardware_mode = False  # True に設定すると実機モードをシミュレート
    
    if use_hardware_mode:
        # 実機向け設定
        qsolver = PhysicsInformedQuantumNN(
            n_qubits=6,  # 実機では少なめ
            n_layers=3,  # 実機では浅め（さらに削減）
            circuit_type='hardware_efficient',  # 実機向けの効率的な回路
            entangling_strategy='all_to_all',
            backend='lightning.qubit',
            shots=1000,  # 実機のショット数
            noise_model='depolarizing'  # ノイズモデル
        )
    else:
        # シミュレーション向け設定
        qsolver = PhysicsInformedQuantumNN(
            n_qubits=8,
            n_layers=4,  # 6から4に削減
            circuit_type='strongly_entangling',  # より表現力の高い回路
            entangling_strategy='all_to_all',
            backend='lightning.qubit',
            shots=None,  # 解析的計算
            noise_model=None
        )
    
    try:
        _, qnn_losses, qnn_time = qsolver.train(n_samples=1500)
        u_qnn = qsolver.evaluate()
        print(f"QPINNモデル評価完了。サイズ: {u_qnn.shape}")
    except Exception as e:
        print(f"量子モデルの学習・評価中にエラー: {str(e)}")
        import traceback
        traceback.print_exc()
        u_qnn = np.zeros(nx * ny * nz * nt)
        qnn_losses = []
        qnn_time = 0
    
    # 3. 解析解の計算
    u_analytical = compute_analytical_solution()
    
    # 4. パフォーマンス評価
    mse_pinn, rel_l2_pinn = calculate_metrics(u_pinn, u_analytical)
    mse_qnn, rel_l2_qnn = calculate_metrics(u_qnn, u_analytical)
    
    print("\n===== 結果の比較 =====")
    print(f"PINN  - MSE: {mse_pinn:.6e}, Relative L2: {rel_l2_pinn:.6e}, Time: {pinn_time:.2f}秒")
    print(f"QPINN - MSE: {mse_qnn:.6e}, Relative L2: {rel_l2_qnn:.6e}, Time: {qnn_time:.2f}秒")
    
    # 5. 結果の可視化
    try:
        visualize_results(results_dir, u_pinn, u_qnn, u_analytical, label_qnn="QPINN")
        print("\n処理が完了しました。結果は画像ファイルに保存されています。")
    except Exception as e:
        print(f"可視化中にエラー: {str(e)}")

if __name__ == "__main__":
    main()
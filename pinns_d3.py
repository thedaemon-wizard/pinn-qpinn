import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import pennylane as qml
import time
from typing import Tuple, List, Callable, Union, Any, Dict, Optional
import os
os.environ['OMP_NUM_THREADS']=str(12)
from collections import deque
import warnings
from scipy.optimize import minimize
from multiprocessing import Pool, cpu_count, Manager, Process
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import functools
from dataclasses import dataclass
import pickle
import threading
from queue import Queue
import psutil
import copy
import json
from itertools import product

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
pinn_epochs = 2000     # PINNのエポック数
qnn_epochs = 2000      # QPINNのエポック数（実機モードでは長めに）

# 並列処理パラメータ
N_PARALLEL_DEVICES = min(4, cpu_count() // 2)  # 並列デバイス数（利用可能なCPUコア数に応じて調整）
USE_PARALLEL_TRAINING = True  # 並列処理を使用するか

# デバイスの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#================================================
# データクラスの定義（並列処理用）
#================================================
@dataclass
class TrainingPoint:
    """トレーニングデータポイント"""
    x: float
    y: float
    z: float
    t: float
    u_true: float = None
    type: str = 'interior'  # 'interior', 'initial', 'boundary', 'data'

@dataclass
class BatchResult:
    """バッチ処理結果"""
    loss: float
    predictions: List[float]
    gradients: np.ndarray = None

@dataclass
class CircuitArchitecture:
    """量子回路アーキテクチャの定義"""
    n_qubits: int
    n_layers: int
    gate_sequence: List[Dict[str, Any]]
    entangling_pattern: str
    feature_map: str
    measurement_basis: List[str]
    score: float = 0.0
    metadata: Dict[str, Any] = None

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

# 安定した数値計算のためのヘルパー関数
def stable_exp(x, max_val=10.0):
    """安定した指数関数計算"""
    x_clipped = qml.numpy.clip(x, -max_val, max_val)
    return qml.numpy.exp(x_clipped)

def stable_tanh(x, max_val=10.0):
    """安定したtanh関数計算"""
    x_clipped = qml.numpy.clip(x, -max_val, max_val)
    return qml.numpy.tanh(x_clipped)

def stable_sigmoid(x, max_val=10.0):
    """安定したsigmoid関数計算"""
    x_clipped = qml.numpy.clip(x, -max_val, max_val)
    return 1.0 / (1.0 + stable_exp(-x_clipped))

#================================================
# Transformerベースの量子回路生成器（改善版）
#================================================
class QuantumCircuitTransformer(nn.Module):
    """量子回路アーキテクチャを生成するTransformerモデル（汎用性重視版）"""
    def __init__(self, vocab_size=50, d_model=128, nhead=4, num_layers=3, 
                 dim_feedforward=512, max_seq_length=100):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._create_positional_encoding(max_seq_length, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # ゲートタイプの定義（汎用的なゲートセット）
        self.gate_vocab = {
            'RY': 0, 'RZ': 1, 'RX': 2, 'CNOT': 3, 'CZ': 4, 
            'H': 5, 'U3': 6, 'CRY': 7, 'CRZ': 8, 'END': 9, 'PAD': 10
        }
        self.reverse_gate_vocab = {v: k for k, v in self.gate_vocab.items()}
        
    def _create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def generate_circuit_architecture(self, n_qubits, n_layers, problem_type='general'):
        """汎用的な量子回路アーキテクチャを生成"""
        gate_sequence = []
        
        # 問題タイプに基づいた生成戦略の選択
        if problem_type == 'general':
            strategy = self._general_strategy
        elif problem_type == 'optimization':
            strategy = self._optimization_strategy
        elif problem_type == 'variational':
            strategy = self._variational_strategy
        else:
            strategy = self._general_strategy
        
        gate_sequence = strategy(n_qubits, n_layers)
        
        architecture = CircuitArchitecture(
            n_qubits=n_qubits,
            n_layers=n_layers,
            gate_sequence=gate_sequence,
            entangling_pattern='transformer_generated',
            feature_map='general_encoding',
            measurement_basis=['Z'] * n_qubits,
            metadata={'generator': 'transformer', 'type': problem_type}
        )
        
        return architecture
    
    def _general_strategy(self, n_qubits, n_layers):
        """汎用的な回路生成戦略"""
        gate_sequence = []
        
        for layer in range(n_layers):
            # 単一量子ビットゲート層（多様性を持たせる）
            for q in range(n_qubits):
                if np.random.rand() < 0.7:  # 70%の確率でゲートを配置
                    if np.random.rand() < 0.6:
                        # 基本的な回転ゲート
                        gate_type = np.random.choice(['RY', 'RZ', 'RX'], p=[0.4, 0.3, 0.3])
                        gate_sequence.append({
                            'gate': gate_type,
                            'qubits': [q],
                            'params': [f'theta_{layer}_{q}_{gate_type}']
                        })
                    else:
                        # U3ゲート（より一般的な単一量子ビットゲート）
                        gate_sequence.append({
                            'gate': 'U3',
                            'qubits': [q],
                            'params': [f'theta_{layer}_{q}_U3_0', 
                                     f'theta_{layer}_{q}_U3_1',
                                     f'theta_{layer}_{q}_U3_2']
                        })
            
            # エンタングリング層
            if layer < n_layers - 1:
                entangling_pairs = self._generate_entangling_pairs(n_qubits, layer)
                for pair in entangling_pairs:
                    if np.random.rand() < 0.8:
                        gate_type = np.random.choice(['CNOT', 'CZ', 'CRY', 'CRZ'], 
                                                   p=[0.4, 0.2, 0.2, 0.2])
                        if gate_type in ['CRY', 'CRZ']:
                            gate_sequence.append({
                                'gate': gate_type,
                                'qubits': pair,
                                'params': [f'phi_{layer}_{pair[0]}_{pair[1]}']
                            })
                        else:
                            gate_sequence.append({
                                'gate': gate_type,
                                'qubits': pair
                            })
        
        return gate_sequence
    
    def _optimization_strategy(self, n_qubits, n_layers):
        """最適化問題向けの回路生成戦略"""
        gate_sequence = []
        
        # 初期Hadamard層
        for q in range(n_qubits):
            gate_sequence.append({
                'gate': 'H',
                'qubits': [q]
            })
        
        # QAOA風の層構造
        for layer in range(n_layers):
            # 問題ハミルトニアン層
            for q in range(n_qubits):
                gate_sequence.append({
                    'gate': 'RZ',
                    'qubits': [q],
                    'params': [f'gamma_{layer}_{q}']
                })
            
            # ミキサー層
            for q in range(n_qubits):
                gate_sequence.append({
                    'gate': 'RX',
                    'qubits': [q],
                    'params': [f'beta_{layer}_{q}']
                })
            
            # エンタングリング
            if layer < n_layers - 1:
                for q in range(0, n_qubits - 1, 2):
                    if q + 1 < n_qubits:
                        gate_sequence.append({
                            'gate': 'CZ',
                            'qubits': [q, q + 1]
                        })
        
        return gate_sequence
    
    def _variational_strategy(self, n_qubits, n_layers):
        """変分アルゴリズム向けの回路生成戦略"""
        gate_sequence = []
        
        for layer in range(n_layers):
            # 強い表現力を持つ層
            for q in range(n_qubits):
                # U3ゲート
                gate_sequence.append({
                    'gate': 'U3',
                    'qubits': [q],
                    'params': [f'u3_{layer}_{q}_0', f'u3_{layer}_{q}_1', f'u3_{layer}_{q}_2']
                })
            
            # フルエンタングリング
            if layer < n_layers - 1:
                for i in range(n_qubits):
                    for j in range(i + 1, n_qubits):
                        if np.random.rand() < 0.3:  # 30%の確率で接続
                            gate_sequence.append({
                                'gate': 'CRY',
                                'qubits': [i, j],
                                'params': [f'cry_{layer}_{i}_{j}']
                            })
        
        return gate_sequence
    
    def _generate_entangling_pairs(self, n_qubits, layer_idx):
        """多様なエンタングリングパターンを生成"""
        pairs = []
        
        pattern_choice = np.random.choice(['nearest', 'alternating', 'all_to_all', 'ladder'])
        
        if pattern_choice == 'nearest':
            # 最近傍エンタングリング
            for i in range(n_qubits - 1):
                if np.random.rand() < 0.7:
                    pairs.append([i, i + 1])
        
        elif pattern_choice == 'alternating':
            # 交互エンタングリング
            offset = layer_idx % 2
            for i in range(offset, n_qubits - 1, 2):
                if i + 1 < n_qubits and np.random.rand() < 0.7:
                    pairs.append([i, i + 1])
        
        elif pattern_choice == 'all_to_all':
            # 全結合（確率的に選択）
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    if np.random.rand() < 0.2:  # 20%の確率で接続
                        pairs.append([i, j])
        
        elif pattern_choice == 'ladder':
            # ラダー型
            if n_qubits >= 4:
                mid = n_qubits // 2
                for i in range(mid - 1):
                    if np.random.rand() < 0.7:
                        pairs.append([i, i + 1])
                    if np.random.rand() < 0.7:
                        pairs.append([i + mid, i + mid + 1])
                    if np.random.rand() < 0.5:
                        pairs.append([i, i + mid])
        
        return pairs

class CircuitArchitectureSearch:
    """量子回路アーキテクチャの自動探索（改善版）"""
    def __init__(self, n_qubits, problem_type='general'):
        self.n_qubits = n_qubits
        self.problem_type = problem_type
        self.transformer = QuantumCircuitTransformer()
        self.best_architecture = None
        self.best_score = -float('inf')
        
    def search(self, n_candidates=10, n_layers_range=(2, 6)):
        """アーキテクチャ探索を実行"""
        candidates = []
        
        # Transformerベースの生成（多様な戦略を使用）
        for i in range(n_candidates // 2):
            n_layers = np.random.randint(n_layers_range[0], n_layers_range[1] + 1)
            problem_types = ['general', 'optimization', 'variational']
            prob_type = np.random.choice(problem_types)
            arch = self.transformer.generate_circuit_architecture(
                self.n_qubits, n_layers, prob_type
            )
            candidates.append(arch)
        
        # 基本的なテンプレート
        for n_layers in range(n_layers_range[0], n_layers_range[1] + 1):
            arch = self._create_hardware_efficient_architecture(n_layers)
            candidates.append(arch)
            if len(candidates) >= n_candidates:
                break
        
        # 評価（汎用的なメトリクス）
        for arch in candidates:
            score = self._evaluate_architecture(arch)
            arch.score = score
            
            if score > self.best_score:
                self.best_score = score
                self.best_architecture = arch
        
        return self.best_architecture
    
    def _create_hardware_efficient_architecture(self, n_layers):
        """ハードウェア効率的なアーキテクチャを作成"""
        gate_sequence = []
        
        for layer in range(n_layers):
            # RY層
            for q in range(self.n_qubits):
                gate_sequence.append({
                    'gate': 'RY',
                    'qubits': [q],
                    'params': [f'ry_{layer}_{q}']
                })
            
            # RZ層
            for q in range(self.n_qubits):
                gate_sequence.append({
                    'gate': 'RZ',
                    'qubits': [q],
                    'params': [f'rz_{layer}_{q}']
                })
            
            # CNOT層（circular）
            if layer < n_layers - 1:
                for q in range(self.n_qubits):
                    gate_sequence.append({
                        'gate': 'CNOT',
                        'qubits': [q, (q + 1) % self.n_qubits]
                    })
        
        return CircuitArchitecture(
            n_qubits=self.n_qubits,
            n_layers=n_layers,
            gate_sequence=gate_sequence,
            entangling_pattern='circular',
            feature_map='hardware_efficient',
            measurement_basis=['Z'] * self.n_qubits,
            metadata={'type': 'hardware_efficient'}
        )
    
    def _evaluate_architecture(self, architecture):
        """アーキテクチャの汎用的な評価"""
        n_gates = len(architecture.gate_sequence)
        n_params = sum(len(g.get('params', [])) for g in architecture.gate_sequence)
        n_entangling = sum(1 for g in architecture.gate_sequence if len(g['qubits']) > 1)
        
        # ゲートの多様性
        gate_types = set(g['gate'] for g in architecture.gate_sequence)
        gate_diversity = len(gate_types)
        
        # エンタングリングの接続性
        entangling_pairs = set()
        for g in architecture.gate_sequence:
            if len(g['qubits']) > 1:
                entangling_pairs.add(tuple(sorted(g['qubits'])))
        connectivity = len(entangling_pairs) / (self.n_qubits * (self.n_qubits - 1) / 2)
        
        # 総合スコア（汎用性を重視）
        score = 0.0
        score += 0.2 * n_params  # パラメータ数
        score += 0.2 * n_entangling  # エンタングリング数
        score += 0.3 * gate_diversity  # ゲートの多様性
        score += 0.2 * connectivity * 10  # 接続性
        score -= 0.001 * n_gates  # 全体のゲート数（軽いペナルティ）
        
        # 適度な深さを評価
        if 3 <= architecture.n_layers <= 6:
            score += 2.0
        
        return score

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
# 並列処理用のグローバル変数とヘルパー関数
#================================================
_quantum_device_pool = None
_pool_lock = threading.Lock()

def initialize_quantum_device_pool(n_devices, n_qubits, architecture, shots, params_shape, noise_model=None):
    """量子デバイスプールの初期化（ノイズモデル対応）"""
    global _quantum_device_pool
    with _pool_lock:
        if _quantum_device_pool is None:
            _quantum_device_pool = []
            for i in range(n_devices):
                device_params = (i, n_qubits, architecture, shots, params_shape, noise_model)
                _quantum_device_pool.append(device_params)
    return _quantum_device_pool

class ParallelQuantumDevice:
    """並列実行用の量子デバイスラッパー（ノイズモデル対応版）"""
    def __init__(self, device_id, n_qubits, architecture, shots, noise_model=None):
        self.device_id = device_id
        self.n_qubits = n_qubits
        self.architecture = architecture
        self.shots = shots
        self.noise_model = noise_model
        
        # デバイスの設定（default.mixedを使用）
        if shots is not None:
            self.dev = qml.device('default.mixed', wires=n_qubits, shots=shots)
            self.diff_method = "parameter-shift"
        else:
            self.dev = qml.device('lightning.qubit', wires=n_qubits)
            self.diff_method = "adjoint"
        
        self._create_circuit_from_architecture()
    
    def _apply_noise_model(self, wire):
        """ノイズモデルの適用"""
        if self.noise_model is None:
            return
        
        if self.noise_model == 'depolarizing':
            # デポラライジングノイズ
            qml.DepolarizingChannel(0.01, wires=wire)
        elif self.noise_model == 'amplitude_damping':
            # 振幅減衰ノイズ
            qml.AmplitudeDamping(0.01, wires=wire)
        elif self.noise_model == 'phase_damping':
            # 位相減衰ノイズ
            qml.PhaseDamping(0.01, wires=wire)
        elif self.noise_model == 'bitflip':
            # ビットフリップノイズ
            qml.BitFlip(0.01, wires=wire)
    
    def _create_circuit_from_architecture(self):
        """アーキテクチャ定義から量子回路を作成（汎用版）"""
        @qml.qnode(self.dev, interface="autograd", diff_method=self.diff_method)
        def circuit(inputs, params_dict):
            # 汎用的な入力エンコーディング
            n_inputs = len(inputs)
            
            # 入力の正規化（問題に依存しない）
            normalized_inputs = []
            for i in range(n_inputs):
                if i < len(inputs):
                    # 入力値をクリップして安定化
                    clipped_input = qml.numpy.clip(inputs[i], 0.0, 1.0)
                    normalized_inputs.append(2.0 * clipped_input - 1.0)  # [-1, 1]範囲に正規化
            
            # データエンコーディング（複数の方式を組み合わせ）
            # 1. 角度エンコーディング
            for i in range(min(n_inputs, self.n_qubits)):
                angle = normalized_inputs[i] * np.pi
                # 角度をクリップして安定化
                angle = qml.numpy.clip(angle, -np.pi, np.pi)
                qml.RY(angle, wires=i)
                if self.noise_model:
                    self._apply_noise_model(i)
            
            # 2. 振幅エンコーディング風の初期化（残りの量子ビット）
            if self.n_qubits > n_inputs:
                for i in range(n_inputs, self.n_qubits):
                    # データの組み合わせから角度を生成
                    combined_angle = 0.0
                    for j in range(n_inputs):
                        combined_angle += normalized_inputs[j] * (j + 1) / (i + 1)
                    # 角度をクリップ
                    combined_angle = qml.numpy.clip(combined_angle * np.pi, -np.pi, np.pi)
                    qml.RY(combined_angle, wires=i)
                    if self.noise_model:
                        self._apply_noise_model(i)
            
            # アーキテクチャに基づいた回路の実行（パラメータをクリップ）
            param_idx = 0
            for gate_info in self.architecture.gate_sequence:
                gate_type = gate_info['gate']
                qubits = gate_info['qubits']
                gate_params = gate_info.get('params', [])
                
                # ゲートの適用（パラメータをクリップして安定化）
                if gate_type == 'RY' and gate_params:
                    if param_idx < len(params_dict['circuit_params']):
                        param_val = qml.numpy.clip(params_dict['circuit_params'][param_idx], -2*np.pi, 2*np.pi)
                        qml.RY(param_val, wires=qubits[0])
                        param_idx += 1
                elif gate_type == 'RZ' and gate_params:
                    if param_idx < len(params_dict['circuit_params']):
                        param_val = qml.numpy.clip(params_dict['circuit_params'][param_idx], -2*np.pi, 2*np.pi)
                        qml.RZ(param_val, wires=qubits[0])
                        param_idx += 1
                elif gate_type == 'RX' and gate_params:
                    if param_idx < len(params_dict['circuit_params']):
                        param_val = qml.numpy.clip(params_dict['circuit_params'][param_idx], -2*np.pi, 2*np.pi)
                        qml.RX(param_val, wires=qubits[0])
                        param_idx += 1
                elif gate_type == 'U3' and gate_params:
                    if param_idx + 2 < len(params_dict['circuit_params']):
                        param1 = qml.numpy.clip(params_dict['circuit_params'][param_idx], -2*np.pi, 2*np.pi)
                        param2 = qml.numpy.clip(params_dict['circuit_params'][param_idx + 1], -2*np.pi, 2*np.pi)
                        param3 = qml.numpy.clip(params_dict['circuit_params'][param_idx + 2], -2*np.pi, 2*np.pi)
                        qml.U3(param1, param2, param3, wires=qubits[0])
                        param_idx += 3
                elif gate_type == 'CNOT':
                    qml.CNOT(wires=qubits)
                elif gate_type == 'CZ':
                    qml.CZ(wires=qubits)
                elif gate_type == 'CRY' and gate_params:
                    if param_idx < len(params_dict['circuit_params']):
                        param_val = qml.numpy.clip(params_dict['circuit_params'][param_idx], -2*np.pi, 2*np.pi)
                        qml.CRY(param_val, wires=qubits)
                        param_idx += 1
                elif gate_type == 'CRZ' and gate_params:
                    if param_idx < len(params_dict['circuit_params']):
                        param_val = qml.numpy.clip(params_dict['circuit_params'][param_idx], -2*np.pi, 2*np.pi)
                        qml.CRZ(param_val, wires=qubits)
                        param_idx += 1
                elif gate_type == 'H':
                    qml.Hadamard(wires=qubits[0])
                
                # ゲート後のノイズ
                if self.noise_model:
                    for q in qubits:
                        self._apply_noise_model(q)
            
            # 測定（複数の基底で測定して情報を増やす）
            measurements = []
            
            # Z基底測定
            for i in range(self.n_qubits):
                measurements.append(qml.expval(qml.PauliZ(i)))
            
            # 追加の測定（より多くの情報を抽出）
            if self.n_qubits >= 2:
                # いくつかのXY測定を追加
                for i in range(min(2, self.n_qubits)):
                    measurements.append(qml.expval(qml.PauliX(i)))
                    measurements.append(qml.expval(qml.PauliY(i)))
            
            return measurements
        
        self.circuit = circuit
    
    def execute(self, inputs, params):
        """回路の実行"""
        return self.circuit(inputs, params)

# 並列実行用のグローバル関数（ノイズ対応版）
def parallel_forward_batch(args):
    """並列実行用のバッチ順伝播（ノイズ対応版）"""
    device_params, batch_data, param_dict = args
    device_id, n_qubits, architecture, shots, _, noise_model = device_params
    
    # デバイスの作成
    device = ParallelQuantumDevice(device_id, n_qubits, architecture, shots, noise_model)
    
    results = []
    for point in batch_data:
        try:
            inputs = qml.numpy.array([point.x, point.y, point.z, point.t])
            measurements = device.execute(inputs, param_dict)
            
            # 測定結果の処理（汎用的な処理）
            measurements_array = qml.numpy.array(measurements)
            
            # Z基底測定の平均（主要な出力）
            z_measurements = measurements_array[:device.n_qubits]
            z_output = qml.numpy.mean(z_measurements)
            
            # 追加測定がある場合の処理
            if len(measurements_array) > device.n_qubits:
                x_measurements = measurements_array[device.n_qubits:device.n_qubits+2]
                y_measurements = measurements_array[device.n_qubits+2:device.n_qubits+4]
                
                # 全体的な量子状態の指標
                quantum_state_indicator = (z_output + 
                                         0.1 * qml.numpy.mean(x_measurements) + 
                                         0.1 * qml.numpy.mean(y_measurements))
            else:
                quantum_state_indicator = z_output
            
            # 非線形変換（汎用的な活性化、安定版）
            scaled_output = quantum_state_indicator * param_dict['output_scale']
            
            # 安定した活性化関数の使用
            tanh_output = stable_tanh(scaled_output)
            sigmoid_output = stable_sigmoid(scaled_output)
            activated_output = 0.7 * tanh_output + 0.3 * sigmoid_output
            
            # バイアスと追加の変換
            result = activated_output * param_dict['amplitude_scale'] + param_dict['output_bias']
            
            # 時間依存性（汎用的な処理）
            if 't' in param_dict and param_dict['t'] > 0:
                time_factor = stable_exp(-param_dict['time_scale'] * param_dict['t'])
                result = result * time_factor
            
            # 出力の正規化（問題に依存しない）
            result = qml.numpy.abs(result)  # 非負制約
            
            results.append(to_python_float(result))
        except Exception as e:
            print(f"Device {device_id} error: {e}")
            results.append(0.1)
    
    return results

class GeneralQuantumNN:
    """汎用的な量子ニューラルネットワーク（PennyLane SPSA対応版）"""
    def __init__(self, n_qubits=8, n_layers=4, use_architecture_search=True,
                 backend='default.mixed', shots=None, noise_model=None, 
                 use_parallel=True, n_parallel_devices=None):
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.use_architecture_search = use_architecture_search
        self.shots = shots
        self.noise_model = noise_model
        self.use_parallel = use_parallel and USE_PARALLEL_TRAINING
        
        # 並列デバイス数の設定
        if n_parallel_devices is None:
            self.n_parallel_devices = N_PARALLEL_DEVICES
        else:
            self.n_parallel_devices = n_parallel_devices
        
        # 実機モードの判定
        self.is_hardware = shots is not None
        self.backend = backend
        
        # 実機モードでの設定
        if self.is_hardware:
            self.min_shots = max(5000, self.shots)  # より多くのショット数
            if self.use_parallel:
                self.shots_per_device = max(1000, self.min_shots // self.n_parallel_devices)
            print(f"実機モード: ショット数 = {self.min_shots}")
            print(f"ノイズモデル: {self.noise_model}")
            if self.use_parallel:
                print(f"並列処理: {self.n_parallel_devices} デバイス, 各 {self.shots_per_device} ショット")
        else:
            print("シミュレーションモード")
        
        # アーキテクチャ探索または基本アーキテクチャ
        if use_architecture_search:
            print("量子回路アーキテクチャの自動探索を開始...")
            self.architecture_search = CircuitArchitectureSearch(n_qubits, problem_type='general')
            self.architecture = self.architecture_search.search(n_candidates=15)
            print(f"最適アーキテクチャを発見: スコア = {self.architecture.score:.4f}")
        else:
            self.architecture = self._get_hardware_efficient_architecture()
        
        # パラメータ数の計算
        self._calculate_parameter_counts()
        
        # メインデバイスの設定
        if self.is_hardware:
            self.dev = qml.device(self.backend, wires=self.n_qubits, shots=self.min_shots)
        else:
            self.dev = qml.device('lightning.qubit', wires=self.n_qubits)
        
        # パラメータの初期化（より小さな初期値で安定化）
        self.circuit_params = qml.numpy.array(
            np.random.uniform(-0.01, 0.01, size=self.n_circuit_params),  # より小さな初期値
            requires_grad=True
        )
        
        # 出力処理パラメータ（汎用的なパラメータ、安定した初期値）
        self.output_scale = qml.numpy.array(0.5, requires_grad=True)  # より小さな初期値
        self.output_bias = qml.numpy.array(0.0, requires_grad=True)
        self.time_scale = qml.numpy.array(0.5, requires_grad=True)    # より小さな初期値
        self.amplitude_scale = qml.numpy.array(0.5, requires_grad=True)  # より小さな初期値
        
        # 量子回路の作成
        self._create_quantum_circuit()
        
        # 並列処理の初期化
        if self.use_parallel:
            self.process_pool = ProcessPoolExecutor(max_workers=self.n_parallel_devices)
            initialize_quantum_device_pool(
                self.n_parallel_devices, self.n_qubits, self.architecture,
                self.shots_per_device if self.is_hardware else None,
                self.n_circuit_params, self.noise_model
            )
            print(f"並列処理プールを初期化: {self.n_parallel_devices} ワーカー")
        
        # トレーニング履歴
        self.loss_history = []
        self.training_data = None
        
    def _get_hardware_efficient_architecture(self):
        """ハードウェア効率的なアーキテクチャ"""
        gate_sequence = []
        
        for layer in range(self.n_layers):
            # 単一量子ビットゲート層
            for q in range(self.n_qubits):
                gate_sequence.append({
                    'gate': 'U3',
                    'qubits': [q],
                    'params': [f'u3_{layer}_{q}_0', f'u3_{layer}_{q}_1', f'u3_{layer}_{q}_2']
                })
            
            # エンタングリング層
            if layer < self.n_layers - 1:
                # Circular entangling
                for q in range(self.n_qubits):
                    gate_sequence.append({
                        'gate': 'CNOT',
                        'qubits': [q, (q + 1) % self.n_qubits]
                    })
        
        return CircuitArchitecture(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            gate_sequence=gate_sequence,
            entangling_pattern='circular',
            feature_map='hardware_efficient',
            measurement_basis=['Z'] * self.n_qubits,
            metadata={'type': 'hardware_efficient'}
        )
    
    def _calculate_parameter_counts(self):
        """アーキテクチャに基づいてパラメータ数を計算"""
        self.n_circuit_params = sum(
            len(gate.get('params', [])) 
            for gate in self.architecture.gate_sequence
        )
        print(f"回路パラメータ数: {self.n_circuit_params}")
    
    def _apply_noise_to_circuit(self, wire):
        """メイン回路用のノイズ適用"""
        if self.noise_model is None or not self.is_hardware:
            return
        
        if self.noise_model == 'depolarizing':
            qml.DepolarizingChannel(0.01, wires=wire)
        elif self.noise_model == 'amplitude_damping':
            qml.AmplitudeDamping(0.01, wires=wire)
        elif self.noise_model == 'phase_damping':
            qml.PhaseDamping(0.01, wires=wire)
        elif self.noise_model == 'bitflip':
            qml.BitFlip(0.01, wires=wire)
        elif self.noise_model == 'combined':
            # 複合ノイズモデル
            qml.DepolarizingChannel(0.005, wires=wire)
            qml.AmplitudeDamping(0.005, wires=wire)
    
    def _create_quantum_circuit(self):
        """量子回路の作成（汎用版、安定化版）"""
        diff_method = "parameter-shift" if self.is_hardware else "adjoint"
        
        @qml.qnode(self.dev, interface="autograd", diff_method=diff_method)
        def quantum_circuit(inputs, circuit_params):
            # 入力の正規化（汎用的、安定化）
            normalized_inputs = []
            for i in range(len(inputs)):
                # 入力値をクリップして安定化
                clipped_input = qml.numpy.clip(inputs[i], 0.0, 1.0)
                normalized_inputs.append(2.0 * clipped_input - 1.0)
            
            # データエンコーディング
            for i in range(min(len(inputs), self.n_qubits)):
                angle = normalized_inputs[i] * np.pi
                # 角度をクリップ
                angle = qml.numpy.clip(angle, -np.pi, np.pi)
                qml.RY(angle, wires=i)
                if self.is_hardware:
                    self._apply_noise_to_circuit(i)
            
            # 追加のエンコーディング
            if self.n_qubits > len(inputs):
                for i in range(len(inputs), self.n_qubits):
                    combined_angle = 0.0
                    for j in range(len(inputs)):
                        combined_angle += normalized_inputs[j] * (j + 1) / (i + 1)
                    # 角度をクリップ
                    combined_angle = qml.numpy.clip(combined_angle * np.pi, -np.pi, np.pi)
                    qml.RY(combined_angle, wires=i)
                    if self.is_hardware:
                        self._apply_noise_to_circuit(i)
            
            # アーキテクチャに基づく回路実行（パラメータをクリップ）
            param_idx = 0
            for gate_info in self.architecture.gate_sequence:
                gate_type = gate_info['gate']
                qubits = gate_info['qubits']
                gate_params = gate_info.get('params', [])
                
                if gate_type == 'RY' and gate_params:
                    if param_idx < len(circuit_params):
                        param_val = qml.numpy.clip(circuit_params[param_idx], -2*np.pi, 2*np.pi)
                        qml.RY(param_val, wires=qubits[0])
                        param_idx += 1
                elif gate_type == 'RZ' and gate_params:
                    if param_idx < len(circuit_params):
                        param_val = qml.numpy.clip(circuit_params[param_idx], -2*np.pi, 2*np.pi)
                        qml.RZ(param_val, wires=qubits[0])
                        param_idx += 1
                elif gate_type == 'RX' and gate_params:
                    if param_idx < len(circuit_params):
                        param_val = qml.numpy.clip(circuit_params[param_idx], -2*np.pi, 2*np.pi)
                        qml.RX(param_val, wires=qubits[0])
                        param_idx += 1
                elif gate_type == 'U3' and gate_params:
                    if param_idx + 2 < len(circuit_params):
                        param1 = qml.numpy.clip(circuit_params[param_idx], -2*np.pi, 2*np.pi)
                        param2 = qml.numpy.clip(circuit_params[param_idx + 1], -2*np.pi, 2*np.pi)
                        param3 = qml.numpy.clip(circuit_params[param_idx + 2], -2*np.pi, 2*np.pi)
                        qml.U3(param1, param2, param3, wires=qubits[0])
                        param_idx += 3
                elif gate_type == 'CNOT':
                    qml.CNOT(wires=qubits)
                elif gate_type == 'CZ':
                    qml.CZ(wires=qubits)
                elif gate_type == 'CRY' and gate_params:
                    if param_idx < len(circuit_params):
                        param_val = qml.numpy.clip(circuit_params[param_idx], -2*np.pi, 2*np.pi)
                        qml.CRY(param_val, wires=qubits)
                        param_idx += 1
                elif gate_type == 'CRZ' and gate_params:
                    if param_idx < len(circuit_params):
                        param_val = qml.numpy.clip(circuit_params[param_idx], -2*np.pi, 2*np.pi)
                        qml.CRZ(param_val, wires=qubits)
                        param_idx += 1
                elif gate_type == 'H':
                    qml.Hadamard(wires=qubits[0])
                
                # ゲート後のノイズ
                if self.is_hardware:
                    for q in qubits:
                        self._apply_noise_to_circuit(q)
            
            # 測定
            measurements = []
            
            # Z基底測定
            for i in range(self.n_qubits):
                measurements.append(qml.expval(qml.PauliZ(i)))
            
            # 追加測定
            if self.n_qubits >= 2 and not self.is_hardware:
                for i in range(min(2, self.n_qubits)):
                    measurements.append(qml.expval(qml.PauliX(i)))
                    measurements.append(qml.expval(qml.PauliY(i)))
            
            return measurements
        
        self.qnode = quantum_circuit
        
        print(f"量子回路を生成:")
        print(f"  - ゲート数: {len(self.architecture.gate_sequence)}")
        print(f"  - エンタングリングパターン: {self.architecture.entangling_pattern}")
        if self.is_hardware:
            print(f"  - デバイス: {self.backend}")
            print(f"  - ノイズモデル: {self.noise_model}")
    
    def forward(self, x, y, z, t):
        """順伝播（単一入力）（汎用版、安定化版）"""
        try:
            # 入力の正規化（0-1範囲）とクリップ
            inputs = qml.numpy.array([
                np.clip(float(x) / L, 0.0, 1.0),
                np.clip(float(y) / L, 0.0, 1.0),
                np.clip(float(z) / L, 0.0, 1.0),
                np.clip(float(t) / T, 0.0, 1.0)
            ])
            
            # 量子回路の実行
            measurements = self.qnode(inputs, self.circuit_params)
            
            # 測定結果の処理
            measurements_array = qml.numpy.array(measurements)
            
            # Z基底測定の処理
            z_measurements = measurements_array[:self.n_qubits]
            z_output = qml.numpy.mean(z_measurements)
            
            # 追加測定の処理（ある場合）
            if len(measurements_array) > self.n_qubits:
                additional_measurements = measurements_array[self.n_qubits:]
                combined_output = z_output + 0.1 * qml.numpy.mean(additional_measurements)
            else:
                combined_output = z_output
            
            # 非線形変換と出力処理（安定化版）
            scaled_output = combined_output * self.output_scale
            
            # 安定した活性化関数の使用
            tanh_output = stable_tanh(scaled_output)
            sigmoid_output = stable_sigmoid(scaled_output)
            activated_output = 0.7 * tanh_output + 0.3 * sigmoid_output
            
            # 最終的な出力
            result = activated_output * self.amplitude_scale + self.output_bias
            
            # 時間依存性（安定化版）
            if float(t) > 0:
                time_factor = stable_exp(-self.time_scale * float(t) / T)
                result = result * time_factor
            
            # 非負制約
            result = qml.numpy.abs(result)
            
            return result
            
        except Exception as e:
            print(f"順伝播エラー: {e}")
            return qml.numpy.array(0.1)
    
    def forward_batch_parallel(self, batch_points):
        """バッチ処理の並列実行"""
        if not self.use_parallel or len(batch_points) < self.n_parallel_devices:
            return [self.forward(p.x, p.y, p.z, p.t) for p in batch_points]
        
        # バッチ分割
        batch_size_per_device = max(1, len(batch_points) // self.n_parallel_devices)
        batches = []
        
        for i in range(self.n_parallel_devices):
            start_idx = i * batch_size_per_device
            if i == self.n_parallel_devices - 1:
                batch = batch_points[start_idx:]
            else:
                end_idx = start_idx + batch_size_per_device
                batch = batch_points[start_idx:end_idx]
            
            if len(batch) > 0:
                batches.append(batch)
        
        # パラメータ辞書（値をクリップ）
        param_dict = {
            'circuit_params': self.circuit_params,
            'output_scale': qml.numpy.clip(self.output_scale, 0.1, 10.0),
            'output_bias': self.output_bias,
            'time_scale': qml.numpy.clip(self.time_scale, 0.1, 10.0),
            'amplitude_scale': qml.numpy.clip(self.amplitude_scale, 0.1, 10.0)
        }
        
        # デバイスプールの取得
        device_pool = _quantum_device_pool[:len(batches)]
        
        # 並列実行
        args_list = [(device_params, batch, param_dict) 
                     for device_params, batch in zip(device_pool, batches)]
        
        futures = []
        for args in args_list:
            future = self.process_pool.submit(parallel_forward_batch, args)
            futures.append(future)
        
        # 結果の収集
        all_results = []
        for i, future in enumerate(futures):
            try:
                results = future.result(timeout=60)
                all_results.extend(results)
            except Exception as e:
                print(f"並列処理エラー（バッチ {i}）: {e}")
                all_results.extend([0.1] * len(batches[i]))
        
        return all_results
    
    def compute_pde_residual(self, x, y, z, t, epsilon=1e-3):
        """PDE残差の計算（汎用版）"""
        try:
            if x < epsilon or x > L - epsilon or y < epsilon or y > L - epsilon or \
               z < epsilon or z > L - epsilon or t > T - epsilon:
                return qml.numpy.array(0.0)
            
            # 中心差分用の評価点
            points = [
                TrainingPoint(x, y, z, t),
                TrainingPoint(x + epsilon, y, z, t),
                TrainingPoint(x - epsilon, y, z, t),
                TrainingPoint(x, y + epsilon, z, t),
                TrainingPoint(x, y - epsilon, z, t),
                TrainingPoint(x, y, z + epsilon, t),
                TrainingPoint(x, y, z - epsilon, t)
            ]
            
            # 時間微分用の点
            if t + epsilon <= T:
                points.append(TrainingPoint(x, y, z, t + epsilon))
            if t - epsilon >= 0:
                points.append(TrainingPoint(x, y, z, t - epsilon))
            
            # 評価
            if self.use_parallel and len(points) >= self.n_parallel_devices:
                values = self.forward_batch_parallel(points)
            else:
                values = [self.forward(p.x, p.y, p.z, p.t) for p in points]
            
            # 数値微分
            u = values[0]
            u_x_plus = values[1]
            u_x_minus = values[2]
            u_y_plus = values[3]
            u_y_minus = values[4]
            u_z_plus = values[5]
            u_z_minus = values[6]
            
            # 時間微分
            if len(values) >= 9:
                u_t_plus = values[7]
                u_t_minus = values[8]
                u_t = (u_t_plus - u_t_minus) / (2 * epsilon)
            elif len(values) >= 8:
                u_t_plus = values[7]
                u_t = (u_t_plus - u) / epsilon
            else:
                u_t = qml.numpy.array(0.0)
            
            # 空間二階微分
            u_xx = (u_x_plus - 2*u + u_x_minus) / (epsilon**2)
            u_yy = (u_y_plus - 2*u + u_y_minus) / (epsilon**2)
            u_zz = (u_z_plus - 2*u + u_z_minus) / (epsilon**2)
            
            # PDE残差
            pde_residual = u_t - alpha * (u_xx + u_yy + u_zz)
            
            return pde_residual
            
        except Exception as e:
            return qml.numpy.array(0.0)
    
    def train(self, n_samples=2500) -> Tuple[qml.numpy.ndarray, List[float], float]:
        """PennyLane SPSAを使用したトレーニング（安定化版）"""
        print(f"量子モデルのトレーニングを開始...")
        print(f"最適化手法: {'PennyLane SPSA' if self.is_hardware else 'Adam'}")
        print(f"並列処理: {'有効' if self.use_parallel else '無効'}")
        if self.use_parallel:
            print(f"並列デバイス数: {self.n_parallel_devices}")
        
        start_time = time.time()
        
        # データ生成
        interior_points, initial_points, boundary_points, data_points = \
            self._generate_training_data(n_samples)
        
        # トレーニングデータを保存
        self.training_data = {
            'interior_points': interior_points,
            'initial_points': initial_points,
            'boundary_points': boundary_points,
            'data_points': data_points
        }
        
        print(f"トレーニングサンプル数: {n_samples}")
        print(f"  - 内部点（PDE）: {len(interior_points)}")
        print(f"  - 初期条件: {len(initial_points)}")
        print(f"  - 境界条件: {len(boundary_points)}")
        print(f"  - データ点: {len(data_points)}")
        
        # コスト関数の定義（安定化版）
        def cost_function(all_params):
            # パラメータの分離とクリップ
            self.circuit_params = qml.numpy.clip(all_params[:self.n_circuit_params], -2*np.pi, 2*np.pi)
            self.output_scale = qml.numpy.clip(qml.numpy.abs(all_params[-4]) + 0.1, 0.1, 10.0)
            self.output_bias = qml.numpy.clip(all_params[-3], -10.0, 10.0)
            self.time_scale = qml.numpy.clip(qml.numpy.abs(all_params[-2]) + 0.1, 0.1, 10.0)
            self.amplitude_scale = qml.numpy.clip(qml.numpy.abs(all_params[-1]) + 0.1, 0.1, 10.0)
            
            # 損失計算（例外処理付き）
            try:
                total_loss = self._compute_loss_parallel()
                # 損失値の妥当性チェック
                if np.isnan(total_loss) or np.isinf(total_loss):
                    print("警告: 無効な損失値が検出されました。デフォルト値を返します。")
                    return 1000.0
                return total_loss
            except Exception as e:
                print(f"損失計算エラー: {e}")
                return 1000.0
        
        # 全パラメータの結合
        all_params = qml.numpy.concatenate([
            self.circuit_params,
            qml.numpy.array([self.output_scale]),
            qml.numpy.array([self.output_bias]),
            qml.numpy.array([self.time_scale]),
            qml.numpy.array([self.amplitude_scale])
        ])
        
        all_params.requires_grad = True
        
        # ベストパラメータの追跡
        best_params = qml.numpy.copy(all_params)
        best_loss = float('inf')
        
        if self.is_hardware:
            # 実機モード: PennyLane SPSA最適化（安定化版）
            print("\nPennyLane SPSA最適化（安定化版）")
            
            # SPSA最適化器の設定（より慎重なパラメータ）
            spsa_opt = qml.SPSAOptimizer(
                maxiter=100,  # step_and_costは100ステップずつ実行
                a=0.05,     # ステップサイズパラメータ（より小さく）
                c=0.05,     # 摂動パラメータ（より小さく）
                A=50,       # 安定性パラメータ（より大きく）
                alpha=0.602,  # ステップサイズ減衰指数
                gamma=0.101   # 摂動減衰指数
            )
            
            # 初期評価
            try:
                initial_loss = to_python_float(cost_function(all_params))
                print(f"初期損失: {initial_loss:.6f}")
                self.loss_history.append(initial_loss)
            except Exception as e:
                print(f"初期評価エラー: {e}")
                initial_loss = 1000.0
                self.loss_history.append(initial_loss)
            
            # 最適化のメインループ
            patience_counter = 0
            max_patience = 300  # 早期停止のための忍耐パラメータ
            
            for epoch in range(qnn_epochs):
                try:
                    # SPSAによる1ステップの最適化
                    all_params, current_cost = spsa_opt.step_and_cost(
                        cost_function, 
                        all_params
                    )
                    
                    # パラメータのクリップ（発散防止）
                    all_params[:self.n_circuit_params] = qml.numpy.clip(
                        all_params[:self.n_circuit_params], -2*np.pi, 2*np.pi
                    )
                    all_params[-4:] = qml.numpy.clip(all_params[-4:], -10.0, 10.0)
                    
                    current_cost = to_python_float(current_cost)
                    self.loss_history.append(current_cost)
                    
                    if current_cost < best_loss:
                        best_loss = current_cost
                        best_params = qml.numpy.copy(all_params)
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    # 進捗報告
                    if (epoch + 1) % 100 == 0 or epoch == 0:
                        print(f"Epoch [{epoch+1}/{qnn_epochs}], "
                              f"Loss: {current_cost:.6f}, "
                              f"Best Loss: {best_loss:.6f}")
                        
                        # 予測値確認
                        self._print_predictions()
                    
                    # 早期停止チェック
                    if patience_counter >= max_patience:
                        print(f"早期停止（エポック {epoch+1}）: {max_patience} エポック改善なし")
                        break
                    
                except Exception as e:
                    print(f"SPSA最適化エラー（エポック {epoch+1}）: {e}")
                    # エラー時は現在の損失値を記録
                    try:
                        current_loss = to_python_float(cost_function(all_params))
                        self.loss_history.append(current_loss)
                    except:
                        self.loss_history.append(self.loss_history[-1] if self.loss_history else initial_loss)
                    continue
        
        else:
            # シミュレータモード: Adam最適化
            print("\nAdam最適化")
            
            adam_opt = qml.AdamOptimizer(stepsize=0.001)  # より小さなステップサイズ
            
            for epoch in range(min(1000, qnn_epochs)):
                try:
                    # 勾配計算とパラメータ更新
                    all_params, cost = adam_opt.step_and_cost(cost_function, all_params)
                    
                    # パラメータのクリップ
                    all_params[:self.n_circuit_params] = qml.numpy.clip(
                        all_params[:self.n_circuit_params], -2*np.pi, 2*np.pi
                    )
                    all_params[-4:] = qml.numpy.clip(all_params[-4:], -10.0, 10.0)
                    
                    self.loss_history.append(to_python_float(cost))
                    
                    if cost < best_loss:
                        best_loss = cost
                        best_params = qml.numpy.copy(all_params)
                    
                    if (epoch + 1) % 100 == 0:
                        print(f"Epoch [{epoch+1}], Loss: {to_python_float(cost):.6f}")
                        self._print_predictions()
                        
                except Exception as e:
                    print(f"Adam最適化エラー: {e}")
                    continue
        
        # 最良パラメータを設定
        self.circuit_params = best_params[:self.n_circuit_params]
        self.output_scale = qml.numpy.clip(qml.numpy.abs(best_params[-4]) + 0.1, 0.1, 10.0)
        self.output_bias = qml.numpy.clip(best_params[-3], -10.0, 10.0)
        self.time_scale = qml.numpy.clip(qml.numpy.abs(best_params[-2]) + 0.1, 0.1, 10.0)
        self.amplitude_scale = qml.numpy.clip(qml.numpy.abs(best_params[-1]) + 0.1, 0.1, 10.0)
        
        training_time = time.time() - start_time
        print(f"\n量子モデルのトレーニング完了。時間: {training_time:.2f}秒")
        print(f"最終損失: {to_python_float(best_loss):.6f}")
        
        # 最終評価
        print("\n最終評価:")
        self._print_predictions()
        
        return self.circuit_params, self.loss_history, training_time
    
    def _compute_loss_parallel(self):
        """並列処理を使用した損失計算（汎用版、安定化版）"""
        reduction_factor = 5 if self.is_hardware else 1
        
        # 最小バッチサイズ
        min_batch_size = self.n_parallel_devices if self.use_parallel else 1
        
        try:
            # 初期条件損失
            initial_loss = 0.0
            n_ic_eval = max(min_batch_size, min(200 // reduction_factor, len(self.training_data['initial_points'])))
            ic_indices = np.random.choice(len(self.training_data['initial_points']), n_ic_eval, replace=False)
            
            ic_batch = [self.training_data['initial_points'][i] for i in ic_indices]
            
            if self.use_parallel:
                predictions = self.forward_batch_parallel(ic_batch)
            else:
                predictions = [self.forward(p.x, p.y, p.z, p.t) for p in ic_batch]
            
            for i, pred in enumerate(predictions):
                true_val = ic_batch[i].u_true
                diff = to_python_float(pred) - true_val
                # 差分をクリップして爆発を防ぐ
                diff = np.clip(diff, -10.0, 10.0)
                initial_loss += diff ** 2
            initial_loss = initial_loss / len(ic_batch)
            
            # PDE損失（軽量化）
            pde_loss = 0.0
            n_pde_eval = max(min_batch_size, min(10 // reduction_factor, len(self.training_data['interior_points'])))
            pde_indices = np.random.choice(len(self.training_data['interior_points']), n_pde_eval, replace=False)
            
            for idx in pde_indices:
                point = self.training_data['interior_points'][idx]
                residual = self.compute_pde_residual(point.x, point.y, point.z, point.t)
                residual_val = to_python_float(residual)
                # 残差をクリップ
                residual_val = np.clip(residual_val, -10.0, 10.0)
                pde_loss += residual_val ** 2
            pde_loss = pde_loss / n_pde_eval
            
            # 境界条件損失
            boundary_loss = 0.0
            n_bc_eval = max(min_batch_size, min(50 // reduction_factor, len(self.training_data['boundary_points'])))
            bc_indices = np.random.choice(len(self.training_data['boundary_points']), n_bc_eval, replace=False)
            
            bc_batch = [self.training_data['boundary_points'][i] for i in bc_indices]
            
            if self.use_parallel:
                predictions = self.forward_batch_parallel(bc_batch)
            else:
                predictions = [self.forward(p.x, p.y, p.z, p.t) for p in bc_batch]
            
            for pred in predictions:
                pred_val = to_python_float(pred)
                # 予測値をクリップ
                pred_val = np.clip(pred_val, -10.0, 10.0)
                boundary_loss += pred_val ** 2
            boundary_loss = boundary_loss / len(bc_batch)
            
            # データ損失
            data_loss = 0.0
            if len(self.training_data['data_points']) > 0:
                n_data_eval = max(min_batch_size, min(150 // reduction_factor, len(self.training_data['data_points'])))
                data_indices = np.random.choice(len(self.training_data['data_points']), n_data_eval, replace=False)
                
                data_batch = [self.training_data['data_points'][i] for i in data_indices]
                
                if self.use_parallel:
                    predictions = self.forward_batch_parallel(data_batch)
                else:
                    predictions = [self.forward(p.x, p.y, p.z, p.t) for p in data_batch]
                
                for i, pred in enumerate(predictions):
                    true_val = data_batch[i].u_true
                    diff = to_python_float(pred) - true_val
                    # 差分をクリップ
                    diff = np.clip(diff, -10.0, 10.0)
                    data_loss += diff ** 2
                data_loss = data_loss / len(data_batch)
            
            # 正則化（最小限）
            reg = 0.0001 * qml.numpy.sum(qml.numpy.clip(self.circuit_params, -10.0, 10.0) ** 2)
            
            # 総損失（汎用的な重み）
            if self.is_hardware:
                # 実機モードでは初期条件とデータを重視
                total_loss = 0.01 * pde_loss + 200.0 * initial_loss + \
                            5.0 * boundary_loss + 100.0 * data_loss + to_python_float(reg)
            else:
                # シミュレーションモードでは物理的制約も考慮
                total_loss = 0.1 * pde_loss + 100.0 * initial_loss + \
                            2.0 * boundary_loss + 50.0 * data_loss + to_python_float(reg)
            
            # 最終的な損失値の妥当性チェック
            if np.isnan(total_loss) or np.isinf(total_loss):
                print("警告: 損失計算で無効な値が検出されました")
                return 1000.0
            
            return total_loss
            
        except Exception as e:
            print(f"損失計算エラー: {e}")
            return 1000.0
    
    def _generate_training_data(self, n_samples):
        """トレーニングデータの生成（汎用版）"""
        # 内部点
        n_interior = int(n_samples * 0.2)  # PDEの比率を減らす
        interior_points = []
        for _ in range(n_interior):
            x = np.random.uniform(0.05, 0.95) * L
            y = np.random.uniform(0.05, 0.95) * L
            z = np.random.uniform(0.05, 0.95) * L
            t = np.random.uniform(0.05, 0.95) * T
            interior_points.append(TrainingPoint(x, y, z, t, type='interior'))
        
        # 初期条件点（より重要）
        n_initial = int(n_samples * 0.4)
        initial_points = []
        for _ in range(n_initial):
            # 多様な分布から選択
            if np.random.rand() < 0.7:
                # 中心付近
                x = np.random.normal(L/2, 0.1)
                y = np.random.normal(L/2, 0.1)
                z = np.random.normal(L/2, 0.1)
            else:
                # ランダム
                x = np.random.uniform(0, 1) * L
                y = np.random.uniform(0, 1) * L
                z = np.random.uniform(0, 1) * L
            
            x = np.clip(x, 0, L)
            y = np.clip(y, 0, L)
            z = np.clip(z, 0, L)
            t = 0.0
            
            u_true = initial_condition(x, y, z)
            initial_points.append(TrainingPoint(x, y, z, t, u_true, type='initial'))
        
        # 境界条件点
        n_boundary = int(n_samples * 0.1)
        boundary_points = []
        
        for i in range(n_boundary):
            face = i % 6
            t_b = np.random.uniform(0, 1) * T
            
            if face == 0:
                x_b, y_b, z_b = 0, np.random.uniform(0, 1) * L, np.random.uniform(0, 1) * L
            elif face == 1:
                x_b, y_b, z_b = L, np.random.uniform(0, 1) * L, np.random.uniform(0, 1) * L
            elif face == 2:
                x_b, y_b, z_b = np.random.uniform(0, 1) * L, 0, np.random.uniform(0, 1) * L
            elif face == 3:
                x_b, y_b, z_b = np.random.uniform(0, 1) * L, L, np.random.uniform(0, 1) * L
            elif face == 4:
                x_b, y_b, z_b = np.random.uniform(0, 1) * L, np.random.uniform(0, 1) * L, 0
            else:
                x_b, y_b, z_b = np.random.uniform(0, 1) * L, np.random.uniform(0, 1) * L, L
            
            boundary_points.append(TrainingPoint(x_b, y_b, z_b, t_b, 0.0, type='boundary'))
        
        # データ点（解析解からのサンプル）
        n_data = n_samples - n_interior - n_initial - n_boundary
        data_points = []
        t_data_points = np.linspace(0, T, 10)
        
        for t_val in t_data_points:
            n_per_time = n_data // len(t_data_points)
            
            for _ in range(n_per_time):
                # 多様なサンプリング
                if np.random.rand() < 0.6:
                    # 重要領域
                    x_val = np.random.normal(L/2, 0.15)
                    y_val = np.random.normal(L/2, 0.15)
                    z_val = np.random.normal(L/2, 0.15)
                else:
                    # ランダム
                    x_val = np.random.uniform(0, 1) * L
                    y_val = np.random.uniform(0, 1) * L
                    z_val = np.random.uniform(0, 1) * L
                
                x_val = np.clip(x_val, 0, L)
                y_val = np.clip(y_val, 0, L)
                z_val = np.clip(z_val, 0, L)
                
                u_val = analytical_solution(x_val, y_val, z_val, t_val)
                data_points.append(TrainingPoint(x_val, y_val, z_val, t_val, u_val, type='data'))
        
        return interior_points, initial_points, boundary_points, data_points
    
    def _print_predictions(self):
        """予測値の表示"""
        test_points = [
            (L/2, L/2, L/2, 0.0),
            (L/2, L/2, L/2, 0.1),
            (L/2, L/2, L/2, 0.5),
            (L/2, L/2, L/2, 1.0),
            (L/4, L/4, L/4, 0.5),
            (3*L/4, 3*L/4, 3*L/4, 0.5)
        ]
        
        for x_test, y_test, z_test, t_test in test_points[:4]:  # 主要な点のみ表示
            try:
                u_pred = self.forward(x_test, y_test, z_test, t_test)
                u_true = analytical_solution(x_test, y_test, z_test, t_test)
                error = abs(to_python_float(u_pred) - u_true)
                print(f"  t={t_test:.1f}: True={u_true:.6f}, "
                      f"Pred={to_python_float(u_pred):.6f}, Error={error:.6f}")
            except Exception as e:
                print(f"  予測エラー at t={t_test:.1f}: {e}")
    
    def evaluate(self) -> np.ndarray:
        """モデルの評価（並列処理対応）"""
        print("量子モデルの評価中...")
        print(f"並列処理: {'有効' if self.use_parallel else '無効'}")
        
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
        
        # 評価ポイントの作成
        eval_points = [
            TrainingPoint(X_flat[i], Y_flat[i], Z_flat[i], T_flat[i], type='eval')
            for i in range(len(X_flat))
        ]
        
        u_pred = np.zeros_like(X_flat)
        
        # 並列処理を使用
        if self.use_parallel:
            print("並列評価を実行中...")
            
            batch_size = max(100, self.n_parallel_devices * 20)
            n_batches = len(eval_points) // batch_size + (1 if len(eval_points) % batch_size != 0 else 0)
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(eval_points))
                
                batch_points = eval_points[start_idx:end_idx]
                predictions = self.forward_batch_parallel(batch_points)
                
                for i, pred in enumerate(predictions):
                    u_pred[start_idx + i] = to_python_float(pred)
                
                if (batch_idx + 1) % 10 == 0:
                    progress = end_idx / len(eval_points) * 100
                    print(f"評価進捗: {progress:.1f}%")
        else:
            # 通常の評価
            print("逐次評価を実行中...")
            
            for i, point in enumerate(eval_points):
                u_pred[i] = to_python_float(
                    self.forward(point.x, point.y, point.z, point.t)
                )
                
                if (i + 1) % 1000 == 0:
                    progress = (i + 1) / len(eval_points) * 100
                    print(f"評価進捗: {progress:.1f}%")
        
        return np.clip(u_pred, 0, None)
    
    def __del__(self):
        """デストラクタ"""
        if hasattr(self, 'process_pool'):
            self.process_pool.shutdown(wait=True)

#================================================
# 可視化と評価関数（変更なし）
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

def calculate_metrics(u_pred: np.ndarray, u_true: np.ndarray) -> Tuple[float, float]:
    """精度メトリクスを計算"""
    u_pred = np.nan_to_num(u_pred, nan=0.0, posinf=0.0, neginf=0.0)
    u_pred = np.clip(u_pred, 0, None)
    
    mse = np.mean((u_pred - u_true) ** 2)
    rel_l2 = np.sqrt(np.sum((u_pred - u_true) ** 2)) / np.sqrt(np.sum(u_true ** 2) + 1e-10)
    return mse, rel_l2

def visualize_results(results_dir: str, u_pinn: np.ndarray, u_qnn: np.ndarray, 
                     u_analytical: np.ndarray, label_qnn: str = "QPINN",
                     qsolver=None) -> None:
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
    plt.savefig(results_dir + 'heat_equation_comparison_spsa.png', dpi=150, bbox_inches='tight')
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
    plt.savefig(results_dir + 'heat_equation_profile_comparison_spsa.png', dpi=150, bbox_inches='tight')
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
    plt.savefig(results_dir + 'heat_equation_error_comparison_spsa.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # トレーニング損失の可視化
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if len(pinn_losses) > 0:
            ax.semilogy(range(1, len(pinn_losses) + 1), pinn_losses, 'b-', label='PINN')
        
        if hasattr(qsolver, 'loss_history') and len(qsolver.loss_history) > 0:
            ax.semilogy(range(1, len(qsolver.loss_history) + 1), 
                       qsolver.loss_history, 'r-', label=label_qnn)
        
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss (log scale)')
        ax.set_title('Training Loss Comparison')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(results_dir + 'heat_equation_loss_comparison_spsa.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"プロット作成中にエラー: {e}")

def main():
    """メイン関数"""
    global pinn_losses, qsolver
    
    print("3次元熱伝導方程式のPINN/QPINN比較を開始（PennyLane SPSA版）...")
    print(f"PennyLane version: {qml.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"利用可能なCPUコア数: {cpu_count()}")
    print(f"並列デバイス数: {N_PARALLEL_DEVICES}")
    print()
    
    # 出力ディレクトリの作成
    os.makedirs('results', exist_ok=True)
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'results/')
    
    # 1. PINNモデルの学習と評価
    pinn_model, pinn_losses, pinn_time = train_pinn()
    u_pinn = evaluate_pinn(pinn_model)
    
    # 2. 汎用量子機械学習モデルの学習と評価
    print("\n=== 汎用量子モデル (General QPINN with Noise) ===")
    
    # 実機モードのテスト（default.mixed + ノイズモデル）
    use_hardware_mode = True
    
    if use_hardware_mode:
        # 実機向け設定（default.mixed + ノイズモデル）
        qsolver = GeneralQuantumNN(
            n_qubits=8,
            n_layers=4,
            use_architecture_search=True,
            backend='default.mixed',
            shots=5000,
            noise_model='combined',  # 複合ノイズモデル
            use_parallel=True,
            n_parallel_devices=N_PARALLEL_DEVICES
        )
    else:
        # シミュレーション向け設定
        qsolver = GeneralQuantumNN(
            n_qubits=10,
            n_layers=5,
            use_architecture_search=False,
            backend='lightning.qubit',
            shots=None,
            noise_model=None,
            use_parallel=False
        )
    
    try:
        _, qnn_losses, qnn_time = qsolver.train(n_samples=3000)
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
    
    if use_hardware_mode:
        label = "QPINN (Noisy, SPSA)"
    else:
        label = "QPINN (Simulation)"
    print(f"{label} - MSE: {mse_qnn:.6e}, Relative L2: {rel_l2_qnn:.6e}, Time: {qnn_time:.2f}秒")
    
    # アーキテクチャ情報の表示
    if hasattr(qsolver, 'architecture'):
        print(f"\n使用された量子回路アーキテクチャ:")
        print(f"  - タイプ: {qsolver.architecture.metadata}")
        print(f"  - ゲート数: {len(qsolver.architecture.gate_sequence)}")
        print(f"  - エンタングリングパターン: {qsolver.architecture.entangling_pattern}")
        print(f"  - スコア: {qsolver.architecture.score:.4f}")
    
    # 5. 結果の可視化
    try:
        visualize_results(results_dir, u_pinn, u_qnn, u_analytical, 
                         label_qnn=label, qsolver=qsolver)
        print("\n処理が完了しました。結果は以下のファイルに保存されています：")
        print(f"  - heat_equation_comparison_spsa.png")
        print(f"  - heat_equation_profile_comparison_spsa.png")
        print(f"  - heat_equation_error_comparison_spsa.png")
        print(f"  - heat_equation_loss_comparison_spsa.png")
    except Exception as e:
        print(f"可視化中にエラー: {str(e)}")

if __name__ == "__main__":
    main()
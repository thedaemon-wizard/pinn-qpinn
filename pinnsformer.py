"""PINNsFormer + SPINN: Separable Transformer Architecture for Physics-Informed Neural Networks

Integrates PINNsFormer (ICLR 2024) with SPINN (NeurIPS 2023 Spotlight) for
efficient and accurate solving of multi-dimensional PDEs.

Key Innovations:
- PINNsFormer: Transformer-based temporal dependency capture with Wavelet activation
- SPINN: Separable per-axis networks reducing O(N^d) → O(Nd) complexity
- Combined architecture: SPINN body networks feed into PINNsFormer encoder

References:
- Zhao, Ding & Prakash (2024): PINNsFormer: A Transformer-Based Framework for
  Physics-Informed Neural Networks, ICLR 2024
- Cho et al. (2023): Separable Physics-Informed Neural Networks, NeurIPS 2023 Spotlight
- Vaswani et al. (2017): Attention Is All You Need
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================
# Wavelet Activation Function (PINNsFormer Eq. 4)
# ================================================
class WaveletActivation(nn.Module):
    """Wavelet activation based on Real Fourier Transform.

    WaveAct(x) = w1 * sin(x) + w2 * cos(x)

    Reference: PINNsFormer paper Eq. 4 — anticipates Fourier decomposition
    through deep neural networks. Provides smooth derivatives needed for
    PDE residual computation via autograd.
    """
    def __init__(self, learnable=True):
        super(WaveletActivation, self).__init__()
        if learnable:
            self.omega1 = nn.Parameter(torch.ones(1))
            self.omega2 = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer('omega1', torch.ones(1))
            self.register_buffer('omega2', torch.ones(1))

    def forward(self, x):
        return self.omega1 * torch.sin(x) + self.omega2 * torch.cos(x)

    def reset_parameters(self):
        nn.init.ones_(self.omega1)
        nn.init.ones_(self.omega2)


# ================================================
# SPINN Body Network (per-axis MLP)
# ================================================
class SPINNBodyNetwork(nn.Module):
    """Single-axis body network for SPINN separable architecture.

    Each body network is a small MLP: R^1 → R^r, where r is the feature rank.
    Uses Wavelet activation for smooth derivatives required by PDE residuals.

    Reference: Cho et al. (2023) Section 3.1
    """
    def __init__(self, rank, hidden_dim=64, n_hidden_layers=2):
        super(SPINNBodyNetwork, self).__init__()
        self.rank = rank

        layers = [nn.Linear(1, hidden_dim)]
        for _ in range(n_hidden_layers):
            layers.append(WaveletActivation())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(WaveletActivation())
        layers.append(nn.Linear(hidden_dim, rank))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, coord):
        """Forward pass for a single coordinate axis.

        Args:
            coord: [batch_size, 1] single coordinate values

        Returns:
            [batch_size, rank] feature representation
        """
        return self.network(coord)


# ================================================
# SPINN Aggregator (outer product + summation)
# ================================================
class SPINNAggregator(nn.Module):
    """Aggregates per-axis SPINN body network outputs via outer product.

    For 3D+time heat equation: combines x, y, z, t body network outputs
    using element-wise product (equivalent to diagonal of outer product)
    followed by a learned projection. This produces a low-rank tensor
    approximation of the solution.

    Reference: Cho et al. (2023) Section 3.2
    """
    def __init__(self, rank, output_dim):
        super(SPINNAggregator, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(rank, rank),
            WaveletActivation(),
            nn.Linear(rank, output_dim)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.projection:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, features_list):
        """Aggregate features from all axes.

        Args:
            features_list: list of [batch_size, rank] tensors (one per axis)

        Returns:
            [batch_size, output_dim] aggregated output
        """
        # Element-wise product of all axis features (Hadamard product)
        aggregated = features_list[0]
        for feat in features_list[1:]:
            aggregated = aggregated * feat
        return self.projection(aggregated)


# ================================================
# Feed-Forward Network (PINNsFormer)
# ================================================
class FeedForward(nn.Module):
    """Feed-forward network with Wavelet activation for Transformer blocks.

    Reference: PINNsFormer implementation — 3-layer FFN with WaveAct.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            WaveletActivation(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_ff),
            WaveletActivation(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# ================================================
# Pseudo Sequence Generator (PINNsFormer Sec. 3.1)
# ================================================
class PseudoSequenceGenerator(nn.Module):
    """Converts point-wise inputs to pseudo sequences for Transformer.

    Given a feature vector at a single (x,y,z,t) point, generates a sequence
    of length L by applying learned temporal shifts. Each position in the
    sequence represents the solution at a different pseudo-time step.

    Reference: PINNsFormer Section 3.1, Equation 3
    """
    def __init__(self, input_dim=4, seq_length=16, d_model=128, delta_t=0.1):
        super(PseudoSequenceGenerator, self).__init__()
        self.seq_length = seq_length
        self.d_model = d_model
        self.delta_t = delta_t

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        nn.init.xavier_uniform_(self.input_projection.weight, gain=0.5)
        nn.init.zeros_(self.input_projection.bias)

        # Temporal modulators for each sequence position
        self.temporal_modulators = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(seq_length)
        ])
        for i, mod in enumerate(self.temporal_modulators):
            nn.init.eye_(mod.weight)
            mod.weight.data *= (0.9 + 0.1 * i / seq_length)
            nn.init.zeros_(mod.bias)

        # Sinusoidal position embeddings (reduced amplitude)
        self.position_embeddings = nn.Parameter(torch.zeros(1, seq_length, d_model))
        self._init_position_embeddings()

    def _init_position_embeddings(self):
        pe = torch.zeros(self.seq_length, self.d_model)
        position = torch.arange(0, self.seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() *
                            -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.position_embeddings.data = pe.unsqueeze(0) * 0.1

    def forward(self, features):
        """Generate pseudo-sequence from features.

        Args:
            features: [batch_size, input_dim]

        Returns:
            [batch_size, seq_length, d_model]
        """
        projected = self.input_projection(features)

        sequences = []
        for i in range(self.seq_length):
            modulated = self.temporal_modulators[i](projected)
            # Physics-aware decay: dominant mode for 3D heat eq
            decay = torch.exp(torch.tensor(-0.296 * i * self.delta_t,
                                           device=projected.device))
            sequences.append(modulated * decay)

        seq = torch.stack(sequences, dim=1)
        return seq + self.position_embeddings


# ================================================
# Encoder Layer (PINNsFormer)
# ================================================
class PINNsFormerEncoderLayer(nn.Module):
    """Encoder layer with self-attention + FFN + Wavelet residual.

    Follows original PINNsFormer: self-attention where Q=K=V from same source,
    residual connection with Wavelet activation, then feed-forward block.

    Reference: PINNsFormer Section 3.2
    """
    def __init__(self, d_model=128, n_heads=8, d_ff=512, dropout=0.1):
        super(PINNsFormerEncoderLayer, self).__init__()

        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.wavelet_res = WaveletActivation()
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        if hasattr(self.self_attention, 'in_proj_weight'):
            nn.init.xavier_uniform_(self.self_attention.in_proj_weight, gain=0.5)
        if hasattr(self.self_attention, 'out_proj'):
            nn.init.xavier_uniform_(self.self_attention.out_proj.weight, gain=0.5)

    def forward(self, x, mask=None):
        attn_out, _ = self.self_attention(x, x, x, attn_mask=mask)
        x = self.norm1(self.wavelet_res(x) + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


# ================================================
# Decoder Layer (PINNsFormer)
# ================================================
class PINNsFormerDecoderLayer(nn.Module):
    """Decoder layer with cross-attention from encoder output.

    Reference: PINNsFormer Section 3.3
    """
    def __init__(self, d_model=128, n_heads=8, d_ff=512, dropout=0.1):
        super(PINNsFormerDecoderLayer, self).__init__()

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.wavelet_res = WaveletActivation()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        attn_out, _ = self.cross_attention(
            query=tgt, key=memory, value=memory, attn_mask=memory_mask
        )
        tgt = self.norm1(self.wavelet_res(tgt) + self.dropout1(attn_out))
        ffn_out = self.ffn(tgt)
        tgt = self.norm2(tgt + self.dropout2(ffn_out))
        return tgt


# ================================================
# PINNsFormer Encoder
# ================================================
class PINNsFormerEncoder(nn.Module):
    """Stacked encoder layers with final Wavelet activation.

    Reference: PINNsFormer — N stacked encoder layers + final WaveAct.
    """
    def __init__(self, n_layers=4, d_model=128, n_heads=8, d_ff=512, dropout=0.1):
        super(PINNsFormerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            PINNsFormerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.final_activation = WaveletActivation()

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.final_activation(x)


# ================================================
# PINNsFormer Decoder
# ================================================
class PINNsFormerDecoder(nn.Module):
    """Stacked decoder layers with final Wavelet activation.

    Reference: PINNsFormer decoder architecture.
    """
    def __init__(self, n_layers=2, d_model=128, n_heads=8, d_ff=512, dropout=0.1):
        super(PINNsFormerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            PINNsFormerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.final_activation = WaveletActivation()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        x = tgt
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return self.final_activation(x)


# ================================================
# Output Projection
# ================================================
class OutputProjection(nn.Module):
    """Projects Transformer sequence output to scalar prediction.

    Uses learned temporal weights to aggregate sequence positions,
    then projects to output dimension through a small network with
    Wavelet activation.
    """
    def __init__(self, seq_length=16, d_model=128, output_dim=1,
                 use_hard_constraints=False, boundary_epsilon=1e-3,
                 domain_size=1.0):
        super(OutputProjection, self).__init__()

        self.seq_length = seq_length
        self.use_hard_constraints = use_hard_constraints

        # Learnable temporal weights (emphasize early time steps)
        self.temporal_weights = nn.Parameter(torch.ones(seq_length) / seq_length)

        # Output projection: d_model → output_dim
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            WaveletActivation(),
            nn.Linear(d_model // 2, output_dim)
        )
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            weights = torch.exp(-torch.arange(self.seq_length, dtype=torch.float32) / 4.0)
            self.temporal_weights.data = weights / weights.sum()
        for layer in self.output_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        """Project sequence to scalar output.

        Args:
            x: [batch_size, seq_length, d_model]

        Returns:
            [batch_size, output_dim]
        """
        weights = F.softmax(self.temporal_weights, dim=0).view(1, -1, 1)
        x_agg = (x * weights).sum(dim=1)
        return self.output_layer(x_agg)


# ================================================
# SpatioTemporalMixer (retained for backward compat)
# ================================================
class SpatioTemporalMixer(nn.Module):
    """Spatio-temporal mixing via dual attention (backward compatibility).

    This module is retained for loading existing checkpoints. The new
    architecture uses the SPINN body networks for spatial feature extraction
    and PINNsFormer encoder for temporal dependencies.
    """
    def __init__(self, d_model=128, n_heads=8, dropout=0.1):
        super(SpatioTemporalMixer, self).__init__()

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
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self._init_weights()

    def _init_weights(self):
        for module in [self.spatial_attention, self.temporal_attention]:
            if hasattr(module, 'in_proj_weight'):
                nn.init.xavier_uniform_(module.in_proj_weight, gain=0.5)
            if hasattr(module, 'out_proj') and hasattr(module.out_proj, 'weight'):
                nn.init.xavier_uniform_(module.out_proj.weight, gain=0.5)
        for layer in self.mixing_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
                nn.init.zeros_(layer.bias)

    def forward(self, seq):
        spatial_out, _ = self.spatial_attention(seq, seq, seq)
        seq = self.norm1(seq + 0.3 * spatial_out)
        temporal_out, _ = self.temporal_attention(seq, seq, seq)
        seq = self.norm2(seq + 0.3 * temporal_out)
        mixed = self.mixing_layer(seq)
        return self.norm3(seq + 0.3 * mixed)




# ================================================
# S-PFormer Decoder Block (simplified pre-norm)
# ================================================
class SPFormerDecoderBlock(nn.Module):
    """Self-attention + FFN with pre-norm and scaled residuals.

    S-PFormer (2025): decoder-only with Fourier embeddings replaces
    separate encoder, yielding simpler architecture with better results.
    """
    def __init__(self, d_model=64, n_heads=4, d_ff=256, dropout=0.1,
                 residual_scale=0.3):
        super(SPFormerDecoderBlock, self).__init__()
        self.residual_scale = residual_scale
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        if hasattr(self.self_attention, 'in_proj_weight'):
            nn.init.xavier_uniform_(
                self.self_attention.in_proj_weight, gain=0.5)
        if hasattr(self.self_attention, 'out_proj'):
            nn.init.xavier_uniform_(
                self.self_attention.out_proj.weight, gain=0.5)

    def forward(self, x, mask=None):
        normed = self.norm1(x)
        attn_out, _ = self.self_attention(
            normed, normed, normed, attn_mask=mask)
        x = x + self.residual_scale * self.dropout(attn_out)
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + self.residual_scale * self.dropout(ffn_out)
        return x


class SPFormerDecoder(nn.Module):
    """Multi-layer S-PFormer decoder with final LayerNorm.

    Simplified decoder-only Transformer: no separate encoder needed
    when using rich SPINN + Fourier features as input.
    """
    def __init__(self, n_layers=2, d_model=64, n_heads=4, d_ff=256,
                 dropout=0.1, residual_scale=0.3):
        super(SPFormerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            SPFormerDecoderBlock(
                d_model, n_heads, d_ff, dropout, residual_scale)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# ================================================
# Backward compatibility aliases
# ================================================
# These keep old checkpoint loading working
TransformerBlock = PINNsFormerEncoderLayer

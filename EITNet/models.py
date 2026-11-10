# -*- coding: utf-8 -*-
import os
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ====== 外部依赖（按你的项目结构） ======
from src.models.Autoformer import series_decomp as auto_series_decomp
from src.models.layers.revin import RevIN  # 保留官方 RevIN

from xlstm1.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig
from xlstm1.blocks.mlstm.block import mLSTMBlockConfig
from xlstm1.blocks.slstm.block import sLSTMBlockConfig

from minGRU_pytorch.minGRU import minGRU

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp as patch_series_decomp

# ====== xLSTM 基本配置（沿用你的默认超参） ======
mlstm_config = mLSTMBlockConfig()
slstm_config = sLSTMBlockConfig()
config = xLSTMBlockStackConfig(
    mlstm_block=mlstm_config, slstm_block=slstm_config, num_blocks=6,
    embedding_dim=256,
    add_post_blocks_norm=True,
    _block_map=1,
    slstm_at="all",
    context_length=512,
)

# ---------------------------------------------------------------------
# 1) 基础组件：移动平均 + 分解 / 归一化 / 卷积平滑
# ---------------------------------------------------------------------
class moving_avg(nn.Module):
    """一维滑动平均（复制边界）"""

    def __init__(self, kernel_size: int, stride: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):  # x: [B, L, C]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class series_decomp2(nn.Module):
    """简单分解：x = residual + trend"""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):  # x: [B, L, C]
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class MovingAvg(nn.Module):
    """(B, L, C) 版移动平均（带 replicate padding）"""

    def __init__(self, kernel_size: int, stride: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):  # x: [B, L, C]
        pad = (self.kernel_size - 1) // 2
        x = x.permute(0, 2, 1)
        x = F.pad(x, (pad, pad), mode='replicate')
        x = self.avg(x).permute(0, 2, 1)
        return x


class SeriesDecomp9(nn.Module):
    """多核移动平均的集成：对多种 kernel 求均值"""

    def __init__(self, kernel_sizes):
        super().__init__()
        if not isinstance(kernel_sizes, (list, tuple)):
            kernel_sizes = [kernel_sizes]
        self.moving_avg_layers = nn.ModuleList([MovingAvg(ks, stride=1) for ks in kernel_sizes])

    def forward(self, x):  # x: [B, L, C]
        means = [layer(x) for layer in self.moving_avg_layers]
        moving_mean = sum(means) / len(means)
        res = x - moving_mean
        return res, moving_mean


class BatchChannelNorm(nn.Module):
    """跨 Batch+Time 的通道归一化（带 running stats）"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, gamma_init=1.0, beta_init=0.0):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(num_features) * gamma_init)
        self.beta = nn.Parameter(torch.zeros(num_features) + beta_init)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):  # x: [B, C, L]
        if self.training:
            mean = x.mean(dim=(0, 2), keepdim=False)  # [C]
            var = x.var(dim=(0, 2), unbiased=False, keepdim=False)  # [C]
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.detach()
        else:
            mean = self.running_mean
            var = self.running_var

        x = (x - mean.view(1, -1, 1)) / torch.sqrt(var.view(1, -1, 1) + self.eps)
        return self.gamma.view(1, -1, 1) * x + self.beta.view(1, -1, 1)


class LD(nn.Module):
    """高斯核初始化的 1D conv 平滑（逐通道共享卷积）"""

    def __init__(self, kernel_size=25):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, stride=1,
                              padding=kernel_size // 2, padding_mode='replicate', bias=True)
        # 高斯权重初始化
        mid = kernel_size // 2
        sigma = 1.0
        w = torch.zeros(1, 1, kernel_size)
        for i in range(kernel_size):
            w[0, 0, i] = math.exp(-((i - mid) / (2 * sigma)) ** 2)
        self.conv.weight.data = F.softmax(w, dim=-1)
        self.conv.bias.data.zero_()

    def forward(self, x):  # x: [B, L, C]
        x = x.permute(0, 2, 1)  # [B, C, L]
        ys = [self.conv(x[:, i:i + 1, :]) for i in range(x.size(1))]
        y = torch.cat(ys, dim=1).permute(0, 2, 1)  # [B, L, C]
        return y


class MultiConvLD(nn.Module):
    """两种核串行堆叠，多次平滑"""

    def __init__(self, kernel_sizes):
        super().__init__()
        k1, k2 = kernel_sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(1, 1, kernel_size=k1, stride=1, padding=k1 // 2, padding_mode='replicate', bias=True),
            nn.Conv1d(1, 1, kernel_size=k2, stride=1, padding=k2 // 2, padding_mode='replicate', bias=True),
            nn.Conv1d(1, 1, kernel_size=k1, stride=1, padding=k1 // 2, padding_mode='replicate', bias=True),
            nn.Conv1d(1, 1, kernel_size=k2, stride=1, padding=k2 // 2, padding_mode='replicate', bias=True),
        ])
        for conv, ks in zip(self.convs, [k1, k2, k1, k2]):
            mid = ks // 2
            sigma = 1.0
            w = torch.zeros(1, 1, ks)
            for i in range(ks):
                w[0, 0, i] = math.exp(-((i - mid) / (2 * sigma)) ** 2)
            conv.weight.data = F.softmax(w, dim=-1)
            conv.bias.data.zero_()

    def forward(self, x):  # x: [B, L, C]
        x = x.permute(0, 2, 1)  # [B, C, L]
        outs = []
        for c in range(x.size(1)):
            t = x[:, c:c + 1, :]
            for conv in self.convs:
                t = conv(t)
            outs.append(t)
        y = torch.cat(outs, dim=1).permute(0, 2, 1)  # [B, L, C]
        return y


class ParallelConvLD(nn.Module):
    """多核并联求和的平滑"""

    def __init__(self, kernel_sizes=(25, 15)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(1, 1, kernel_size=ks, stride=1, padding=ks // 2, padding_mode='replicate', bias=True)
            for ks in kernel_sizes
        ])
        for conv, ks in zip(self.convs, kernel_sizes):
            mid = ks // 2
            sigma = 1.0
            w = torch.zeros(1, 1, ks)
            for i in range(ks):
                w[0, 0, i] = math.exp(-((i - mid) / (2 * sigma)) ** 2)
            conv.weight.data = F.softmax(w, dim=-1)
            conv.bias.data.zero_()

    def forward(self, x):  # x: [B, L, C]
        x = x.permute(0, 2, 1)  # [B, C, L]
        outs = []
        for c in range(x.size(1)):
            t = sum(conv(x[:, c:c + 1, :]) for conv in self.convs)
            outs.append(t)
        y = torch.cat(outs, dim=1).permute(0, 2, 1)
        return y


# ---------------------------------------------------------------------
# 2) 位置编码（时间维）
# ---------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_seq_length=4096, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_length, embedding_dim)
        for pos in range(max_seq_length):
            for i in range(0, embedding_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (2 * i / embedding_dim)))
                if i + 1 < embedding_dim:
                    pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i + 1) / embedding_dim)))
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, S, D]

    def forward(self, x):  # x: [B, S, D]
        S = x.size(1)
        x = x * math.sqrt(self.embedding_dim)
        x = x + self.pe[:, :S].to(x.device)
        return self.dropout(x)


# ---------------------------------------------------------------------
# 3) 两阶段门控：专家维注意力（Stage-I） + 时间维 Transformer（Stage-II）
# ---------------------------------------------------------------------
class ExpertInteractionAttention(nn.Module):
    """
    Stage-I: 在每个时间步，对 E 个专家的 C 维向量做注意力交互，并汇聚成 h_t。
    输入:  expert_outputs [B, C, E, S]
    输出:  h_seq [B, S, d], coarse_logits [B, S, E]
    解释性输出（可选）:
      - X_enc: [B, S, E, d]
      - pool_attn: [B, S, 1, E]
      - drop_mask: [B, E] or None
    """

    def __init__(
        self,
        num_experts: int,
        feat_dim: int,
        d_model: int = 128,
        heads: int = 4,
        dropout: float = 0.0,
        redundancy_init: float = 0.2,
    ):
        super().__init__()
        self.E = num_experts
        self.C = feat_dim
        self.d = d_model

        self.proj_in = nn.Linear(self.C, self.d, bias=False)  # C -> d
        self.expert_id = nn.Parameter(torch.randn(self.E, self.d) * 0.02)  # 专家ID嵌入

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d, nhead=heads, dim_feedforward=4 * self.d,
            batch_first=True, dropout=dropout, activation='gelu'
        )
        self.expert_encoder = nn.TransformerEncoder(enc_layer, num_layers=1)

        # Common-specific expert decomposition.
        # It keeps the shared expert component and adaptively amplifies expert-specific
        # residuals, reducing excessive similarity before fusion.
        self.redundancy_gate = nn.Parameter(torch.tensor(float(redundancy_init)))

        self.pool_q = nn.Parameter(torch.randn(1, 1, self.d) * 0.02)
        self.pool_attn = nn.MultiheadAttention(self.d, heads, batch_first=True, dropout=dropout)

        self.to_logits = nn.Linear(self.d, 1, bias=False)  # 每个专家token -> 标量logit

    @torch.no_grad()
    def _repeat_pool_q(self, BxS: int):
        return self.pool_q.expand(BxS, -1, -1)

    def _common_specific_decomposition(self, x: torch.Tensor):
        # x: [B*S, E, d]
        if x.size(1) <= 1:
            return x, x, torch.zeros_like(x), x.new_tensor(1.0)
        common = x.mean(dim=1, keepdim=True)
        specific = x - common
        # scale is initialized above 1, so the first run already suppresses
        # over-similarity slightly; training can tune it back toward 1.
        scale = 1.0 + torch.tanh(self.redundancy_gate)
        x = common + scale * specific
        return x, common, specific, scale

    def forward(
        self,
        expert_outputs: torch.Tensor,
        expert_dropout_p: float = 0.2,
        training: bool = True,
        return_intermediate: bool = False,
    ):
        # expert_outputs: [B, C, E, S]
        B, C, E, S = expert_outputs.shape
        assert E == self.E and C == self.C

        # [B, S, E, C]
        X = expert_outputs.permute(0, 3, 2, 1).contiguous()

        # 投影 + 专家ID
        X = self.proj_in(X)  # [B, S, E, d]
        X = X + self.expert_id.view(1, 1, E, self.d)

        # 专家Dropout（按 E 维）
        drop_mask = None
        if training and expert_dropout_p > 0.0:
            drop_mask = (torch.rand(B, E, device=X.device) < expert_dropout_p)  # [B, E]
            # 确保每个样本至少保留一个专家
            all_drop = drop_mask.all(dim=1)
            if all_drop.any():
                idx = torch.argmax(torch.rand(B, E, device=X.device), dim=1)
                drop_mask[torch.arange(B, device=X.device), idx] = False
            X = X.masked_fill(drop_mask.view(B, 1, E, 1), 0.0)

        # 把 (B,S) 合并成 batch，做一次 E 长度的 Encoder
        X_enc_raw = self.expert_encoder(X.view(B * S, E, self.d))  # [B*S, E, d]

        # Common-specific decomposition suppresses redundant expert representation.
        X_enc, X_common, X_specific, redundancy_scale = self._common_specific_decomposition(X_enc_raw)

        # 粗粒度 per-time logits
        coarse_logits = self.to_logits(X_enc).squeeze(-1).view(B, S, E)  # [B, S, E]
        if (drop_mask is not None) and training:
            mask = drop_mask.unsqueeze(1).expand(B, S, E)
            coarse_logits = coarse_logits.masked_fill(mask, float('-inf'))

        # learnable query 做 attention pooling
        q = self._repeat_pool_q(B * S)  # [B*S, 1, d]
        h, pool_attn = self.pool_attn(q, X_enc, X_enc, need_weights=True)  # pool_attn: [B*S, 1, E]
        h = h.view(B, S, self.d)
        pool_attn = pool_attn.view(B, S, 1, E)

        if not return_intermediate:
            return h, coarse_logits

        intermediates = {
            "X_enc": X_enc.view(B, S, E, self.d),  # [B, S, E, d]
            "X_enc_before_redundancy": X_enc_raw.view(B, S, E, self.d),
            "X_common": X_common.view(B, S, 1, self.d),
            "X_specific": X_specific.view(B, S, E, self.d),
            "redundancy_scale": redundancy_scale.detach(),
            "pool_attn": pool_attn,                # [B, S, 1, E]
            "drop_mask": drop_mask,                # [B, E] or None
        }
        return h, coarse_logits, intermediates


class TemporalGatingHead(nn.Module):
    """
    Stage-II: factorized expert-time modulation.

    The temporal context is projected into R latent modes, and these modes
    are mapped to E expert logits through a shared expert-mode matrix.
    Input: h_seq [B, S, d], optional coarse_logits [B, S, E].
    Output: weights [B, S, E] with softmax over experts.
    """

    def __init__(self, d_model: int, num_experts: int, num_layers: int = 2, heads: int = 4,
                 dropout: float = 0.0, causal: bool = True, num_modes: int = 3):
        super().__init__()
        self.num_experts = num_experts
        self.num_modes = max(1, min(int(num_modes), num_experts))
        self.pos_enc = PositionalEncoding(d_model, max_seq_length=4096, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=4 * d_model,
            batch_first=True, dropout=dropout, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.to_modes = nn.Linear(d_model, self.num_modes)
        self.mode_expert = nn.Parameter(torch.empty(self.num_modes, num_experts))
        self.mode_bias = nn.Parameter(torch.zeros(num_experts))
        # Kept only for loading old checkpoints trained with a direct d -> E head.
        self.to_logits = nn.Linear(d_model, num_experts)
        self._reset_factorized_head()
        self.causal = causal
        self.register_buffer('causal_mask', None, persistent=False)

    def _reset_factorized_head(self):
        nn.init.xavier_uniform_(self.to_modes.weight)
        nn.init.zeros_(self.to_modes.bias)
        nn.init.normal_(self.mode_expert, mean=0.0, std=0.02)
        nn.init.zeros_(self.mode_bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        legacy_w_key = prefix + "to_logits.weight"
        legacy_b_key = prefix + "to_logits.bias"
        mode_w_key = prefix + "to_modes.weight"
        mode_b_key = prefix + "to_modes.bias"
        expert_key = prefix + "mode_expert"
        bias_key = prefix + "mode_bias"

        need_factor_init = (
            legacy_w_key in state_dict and (
                mode_w_key not in state_dict or
                state_dict[mode_w_key].shape != self.to_modes.weight.shape or
                mode_b_key not in state_dict or
                state_dict[mode_b_key].shape != self.to_modes.bias.shape or
                expert_key not in state_dict or
                state_dict[expert_key].shape != self.mode_expert.shape or
                bias_key not in state_dict or
                state_dict[bias_key].shape != self.mode_bias.shape
            )
        )

        if need_factor_init:
            legacy_w = state_dict[legacy_w_key].detach()
            legacy_b = state_dict.get(legacy_b_key)
            with torch.no_grad():
                u, s, vh = torch.linalg.svd(legacy_w.float(), full_matrices=False)
                r = self.num_modes
                rank = min(r, s.numel())
                to_modes_w = torch.zeros(
                    self.to_modes.weight.shape,
                    dtype=legacy_w.dtype,
                    device=legacy_w.device,
                )
                mode_expert = torch.zeros(
                    self.mode_expert.shape,
                    dtype=legacy_w.dtype,
                    device=legacy_w.device,
                )
                root_s = torch.sqrt(s[:rank].clamp_min(1e-12))
                to_modes_w[:rank] = (root_s[:, None] * vh[:rank]).to(legacy_w.dtype)
                mode_expert[:rank] = (root_s[:, None] * u[:, :rank].T).to(legacy_w.dtype)

                state_dict[mode_w_key] = to_modes_w
                state_dict[mode_b_key] = torch.zeros(
                    self.to_modes.bias.shape,
                    dtype=legacy_w.dtype,
                    device=legacy_w.device,
                )
                state_dict[expert_key] = mode_expert
                if legacy_b is None:
                    state_dict[bias_key] = torch.zeros(
                        self.mode_bias.shape,
                        dtype=legacy_w.dtype,
                        device=legacy_w.device,
                    )
                else:
                    state_dict[bias_key] = legacy_b.detach().clone()

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )

    def _get_causal_mask(self, S, device):
        if (self.causal_mask is None) or (self.causal_mask.size(0) != S):
            self.causal_mask = torch.triu(torch.ones(S, S, device=device), diagonal=1).bool()
        return self.causal_mask

    def forward(self, h_seq: torch.Tensor, extra_logits: Optional[torch.Tensor] = None,
                return_intermediate: bool = False):
        B, S, d = h_seq.shape
        x = self.pos_enc(h_seq)
        mask = self._get_causal_mask(S, h_seq.device) if self.causal else None
        z = self.encoder(x, mask=mask)  # [B, S, d]
        mode_logits = self.to_modes(z)  # [B, S, R]
        temporal_logits = torch.matmul(mode_logits, self.mode_expert) + self.mode_bias  # [B, S, E]
        logits = temporal_logits
        if extra_logits is not None:
            logits = logits + extra_logits
        weights = F.softmax(logits, dim=-1)  # over E
        if return_intermediate:
            return weights, {
                "temporal_context": z,
                "mode_logits": mode_logits,
                "mode_weights": F.softmax(mode_logits, dim=-1),
                "temporal_logits": temporal_logits,
            }
        return weights


# ---------------------------------------------------------------------
# 4) 专家模型
# ---------------------------------------------------------------------
class Model(nn.Module):
    """PatchTST 双支（res/trend）"""

    def __init__(self, configs, max_seq_len: int = 1024, d_k: Optional[int] = None, d_v: Optional[int] = None,
                 norm: str = 'BatchNorm', attn_dropout: float = 0., act: str = "gelu",
                 key_padding_mask: bool = 'auto', padding_var: Optional[int] = None,
                 attn_mask: Optional[torch.Tensor] = None, res_attention: bool = True, pre_norm: bool = False,
                 store_attn: bool = False, pe: str = 'zeros', learn_pe: bool = True,
                 pretrain_head: bool = False, head_type: str = 'flatten', verbose: bool = False, **kwargs):
        super().__init__()

        c_in = configs.enc_in
        context_window = configs.context_points
        target_window = configs.target_points
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        head_dropout = configs.head_dropout
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = 'end'
        revin = configs.revin
        affine = 1
        subtract_last = 0
        kernel_size = 25

        self.decomp_module = patch_series_decomp(kernel_size)
        self.model_trend = PatchTST_backbone(
            c_in=c_in, context_window=context_window, target_window=target_window,
            patch_len=patch_len, stride=stride, max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
            n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
            act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, attn_mask=attn_mask,
            res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn, pe=pe, learn_pe=learn_pe,
            fc_dropout=0.0, head_dropout=head_dropout, padding_patch=padding_patch,
            pretrain_head=pretrain_head, head_type=head_type, individual=False, revin=revin,
            affine=affine, subtract_last=subtract_last, verbose=verbose, **kwargs
        )
        self.model_res = PatchTST_backbone(
            c_in=c_in, context_window=context_window, target_window=target_window,
            patch_len=patch_len, stride=stride, max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
            n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
            act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, attn_mask=attn_mask,
            res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn, pe=pe, learn_pe=learn_pe,
            fc_dropout=0.0, head_dropout=head_dropout, padding_patch=padding_patch,
            pretrain_head=pretrain_head, head_type=head_type, individual=False, revin=revin,
            affine=affine, subtract_last=subtract_last, verbose=verbose, **kwargs
        )

    def forward(self, x):  # x: [B, L_ctx, C]
        res_init, trend_init = self.decomp_module(x)  # [B, L_ctx, C]
        res = self.model_res(res_init.permute(0, 2, 1))  # [B, C, L_pred]
        trend = self.model_trend(trend_init.permute(0, 2, 1))
        y = (res + trend).permute(0, 2, 1)  # [B, L_pred, C]
        return y


class model_c(nn.Module):
    """多核分解+多次平滑 + minGRU"""

    def __init__(self, configs, enc_in):
        super().__init__()
        self.configs = configs
        self.enc_in = enc_in

        self.norm = BatchChannelNorm(enc_in, gamma_init=1.0, beta_init=0.0)
        kernel_size = [25, 25]
        self.decomposition = SeriesDecomp9(kernel_size)
        self.LD = MultiConvLD(kernel_size)

        self.Linear_Seasonal = nn.Linear(configs.context_points, configs.target_points)
        self.Linear_Trend = nn.Linear(configs.context_points, configs.target_points)
        self.Linear_Seasonal.weight = nn.Parameter(
            (1 / configs.context_points) * torch.ones([configs.target_points, configs.context_points])
        )
        self.Linear_Trend.weight = nn.Parameter(
            (1 / configs.context_points) * torch.ones([configs.target_points, configs.context_points])
        )

        self.mm = nn.Linear(configs.target_points, configs.n2)
        self.mm2 = nn.Linear(configs.n2, configs.target_points)

        self.seq_model = minGRU(dim=256)
        self.proj = nn.Linear(configs.target_points, configs.target_points)

    def forward(self, x):  # x: [B, L_ctx, C]
        seasonal_init, trend_init = self.decomposition(x)  # [B, L_ctx, C]
        trend_refined = self.LD(trend_init)  # [B, L_ctx, C]

        s = self.Linear_Seasonal(seasonal_init.permute(0, 2, 1))  # [B, C, L_pred]
        t = self.Linear_Trend(trend_refined.permute(0, 2, 1))  # [B, C, L_pred]
        y = s + t  # [B, C, L_pred]

        y = self.mm(y)  # [B, C, n2]
        y = self.norm(y)
        y = self.seq_model(y)
        y = self.mm2(y)
        y = y.permute(0, 2, 1)
        return y


class model_b(nn.Module):
    """分解 + 线性 + xLSTM 堆叠"""

    def __init__(self, configs, enc_in):
        super().__init__()
        self.configs = configs
        self.enc_in = enc_in
        self.batch_norm = nn.BatchNorm1d(self.enc_in)
        self.revin_layer = RevIN(self.enc_in)

        self.decompsition = series_decomp2(25)
        self.Linear_Seasonal = nn.Linear(configs.context_points, configs.target_points)
        self.Linear_Trend = nn.Linear(configs.context_points, configs.target_points)
        self.Linear_Seasonal.weight = nn.Parameter(
            (1 / configs.context_points) * torch.ones([configs.target_points, configs.context_points])
        )
        self.Linear_Trend.weight = nn.Parameter(
            (1 / configs.context_points) * torch.ones([configs.target_points, configs.context_points])
        )

        self.mm = nn.Linear(configs.target_points, configs.n2)
        self.xlstm_stack = xLSTMBlockStack(config)
        self.mm2 = nn.Linear(config.embedding_dim, configs.target_points)

    def forward(self, x):  # x: [B, L_ctx, C]
        seasonal_init, trend_init = self.decompsition(x)
        s = self.Linear_Seasonal(seasonal_init.permute(0, 2, 1))  # [B, C, L_pred]
        t = self.Linear_Trend(trend_init.permute(0, 2, 1))        # [B, C, L_pred]
        y = self.mm(s + t)                                        # [B, C, n2]
        y = self.batch_norm(y)
        y = self.xlstm_stack(y)                                   # [B, C, embedding_dim]
        y = self.mm2(y).permute(0, 2, 1)                          # [B, L_pred, C]
        return y


class Model1(nn.Module):
    """Decomp-Linear 模型"""

    def __init__(self, pred_len, seq_len, enc_in):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in

        self.decompsition = series_decomp2(25)
        self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Seasonal.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        self.Linear_Trend.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        self.batch_norm = nn.BatchNorm1d(self.channels)
        self.Linear = nn.Linear(self.pred_len, self.pred_len)

    def forward(self, x):  # x: [B, L_ctx, C]
        seasonal_init, trend_init = self.decompsition(x)
        s = self.Linear_Seasonal(seasonal_init.permute(0, 2, 1))
        t = self.Linear_Trend(trend_init.permute(0, 2, 1))
        y = (s + t)
        y = self.batch_norm(y)
        return y.permute(0, 2, 1)


class Model2(nn.Module):
    """Norm-Linear 模型（残差法）"""

    def __init__(self, pred_len, seq_len, enc_in):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.batch_norm = nn.BatchNorm1d(self.channels)
        self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):  # x: [B, L_ctx, C]
        seq_last = x[:, -1:, :].detach()
        y = self.Linear((x - seq_last).permute(0, 2, 1)).permute(0, 2, 1) + seq_last
        y = self.batch_norm(y.permute(0, 2, 1)).permute(0, 2, 1)
        return y


class model_d(nn.Module):
    """两路线性模型的可学习门控融合"""

    def __init__(self, configs, enc_in):
        super().__init__()
        self.seq_len = configs.context_points
        self.pred_len = configs.target_points
        self.channels = enc_in
        self.batch_norm = nn.BatchNorm1d(self.channels)
        self.model2 = Model2(self.pred_len, self.seq_len, enc_in)
        self.model1 = Model1(self.pred_len, self.seq_len, enc_in)
        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        y2 = self.model2(x)
        y1 = self.model1(x)
        g = torch.sigmoid(self.gate)
        y = g * y2 + (1 - g) * y1
        y = self.batch_norm(y.permute(0, 2, 1)).permute(0, 2, 1)
        return y


# ---------------------------------------------------------------------
# 5) 多专家与两阶段门控融合
# ---------------------------------------------------------------------
class EMTSF(nn.Module):
    """
    输入:  x [B, L_ctx, C]
    输出:  y [B, L_pred, C]
    """

    def __init__(self, configs, enc_in,
                 num_experts_a=2, num_experts_b=2, num_experts_c=2, num_experts_d=2,
                 freeze_experts=True, d_model=128, heads_expert=4, heads_time=4, time_layers=2,
                 expert_dropout_p=0.1, expert_redundancy_init: Optional[float] = None,
                 time_modes: Optional[int] = None, expert_checkpoints=None):
        super().__init__()
        self.configs = configs
        self.enc_in = enc_in
        self.freeze_experts = freeze_experts
        if expert_redundancy_init is None:
            expert_redundancy_init = getattr(configs, "expert_redundancy_init", 0.2)
        if time_modes is None:
            time_modes = getattr(configs, "time_modes", 3)

        # --- 构造专家 ---
        self.experts = nn.ModuleList()
        for _ in range(num_experts_a):
            ea = Model(configs, enc_in)
            if expert_checkpoints is not None:
                ea.load_state_dict(torch.load(expert_checkpoints['PatchTST'], map_location='cpu', weights_only=True))
            if freeze_experts:
                for p in ea.parameters():
                    p.requires_grad = False
            self.experts.append(ea)

        for _ in range(num_experts_b):
            eb = model_b(configs, enc_in)
            if expert_checkpoints is not None:
                eb.load_state_dict(torch.load(expert_checkpoints['xLSTMTime'], map_location='cpu', weights_only=True))
            if freeze_experts:
                for p in eb.parameters():
                    p.requires_grad = False
            self.experts.append(eb)

        for _ in range(num_experts_c):
            ec = model_c(configs, enc_in)
            if expert_checkpoints is not None:
                ec.load_state_dict(torch.load(expert_checkpoints['minGRUTime'], map_location='cpu', weights_only=True))
            if freeze_experts:
                for p in ec.parameters():
                    p.requires_grad = False
            self.experts.append(ec)

        for _ in range(num_experts_d):
            ed = model_d(configs, enc_in)
            if expert_checkpoints is not None:
                ed.load_state_dict(torch.load(expert_checkpoints['ELM'], map_location='cpu', weights_only=True))
            if freeze_experts:
                for p in ed.parameters():
                    p.requires_grad = False
            self.experts.append(ed)

        self.num_experts = len(self.experts)

        # --- 两阶段门控 ---
        self.expert_attn = ExpertInteractionAttention(
            num_experts=self.num_experts,
            feat_dim=self.enc_in,
            d_model=d_model,
            heads=heads_expert,
            dropout=getattr(configs, 'dropout', 0.0),
            redundancy_init=expert_redundancy_init,
        )
        self.temporal_head = TemporalGatingHead(
            d_model=d_model,
            num_experts=self.num_experts,
            num_layers=time_layers,
            heads=heads_time,
            dropout=getattr(configs, 'dropout', 0.0),
            causal=True,
            num_modes=time_modes,
        )
        self.expert_dropout_p = expert_dropout_p

    def forward(self, x, return_details: bool = False):
        # 1) All experts predict [B, L_pred, C], then stack as [B, C, E, S].
        expert_outputs = [exp(x).permute(0, 2, 1) for exp in self.experts]
        expert_outputs_stacked = torch.stack(expert_outputs, dim=2)

        # 2) Stage-I: expert representation interaction.
        if return_details:
            h_seq, coarse_logits, inter = self.expert_attn(
                expert_outputs_stacked,
                expert_dropout_p=self.expert_dropout_p,
                training=self.training,
                return_intermediate=True,
            )
        else:
            h_seq, coarse_logits = self.expert_attn(
                expert_outputs_stacked,
                expert_dropout_p=self.expert_dropout_p,
                training=self.training,
                return_intermediate=False,
            )
            inter = None

        # 3) Stage-II: factorized temporal expert modulation.
        if return_details:
            weights, temporal_inter = self.temporal_head(
                h_seq,
                extra_logits=coarse_logits,
                return_intermediate=True,
            )
        else:
            weights = self.temporal_head(h_seq, extra_logits=coarse_logits)
            temporal_inter = None

        # 4) Weighted expert fusion.
        w = weights.permute(0, 2, 1).unsqueeze(1)
        fused = (expert_outputs_stacked * w).sum(dim=2)
        y = fused.permute(0, 2, 1)

        if not return_details:
            return y

        details = {
            "expert_outputs": expert_outputs_stacked,
            "expert_weights": weights,
            "coarse_logits": coarse_logits,
            "h_seq": h_seq,
        }
        if inter is not None:
            details.update({
                "X_enc": inter["X_enc"],
                "X_enc_before_redundancy": inter["X_enc_before_redundancy"],
                "X_common": inter["X_common"],
                "X_specific": inter["X_specific"],
                "redundancy_scale": inter["redundancy_scale"],
                "pool_attn": inter["pool_attn"],
                "drop_mask": inter["drop_mask"],
            })
        if temporal_inter is not None:
            details.update({
                "temporal_context": temporal_inter["temporal_context"],
                "temporal_mode_logits": temporal_inter["mode_logits"],
                "temporal_mode_weights": temporal_inter["mode_weights"],
                "temporal_logits": temporal_inter["temporal_logits"],
            })

        return y, details

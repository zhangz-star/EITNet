"""EITNet model and configuration."""
from dataclasses import dataclass
from typing import Mapping, Optional, Union
from pathlib import Path

from models import EMTSF, config as xlstm_config


@dataclass
class EITNetConfig:
    enc_in: int
    context_points: int = 512
    target_points: int = 96
    patch_len: int = 32
    stride: int = 16
    e_layers: int = 2
    n_heads: int = 16
    d_model: int = 128
    d_ff: int = 512
    dropout: float = 0.2
    head_dropout: float = 0.0
    revin: int = 1
    n2: int = 256
    expert_redundancy_init: float = 0.2
    time_modes: int = 3
    xlstm_backend: str = 'vanilla'


class EITNet(EMTSF):
    """Four experts with expert-interaction and temporal gating.

    expert_checkpoints maps PatchTST, xLSTMTime, minGRUTime and ELM to
    state-dict paths. If omitted, the experts are initialized randomly.
    """
    def __init__(self, config: EITNetConfig,
                 expert_checkpoints: Optional[Mapping[str, Union[str, Path]]] = None,
                 freeze_experts: bool = True):
        if config.enc_in < 1 or config.target_points < 1:
            raise ValueError('enc_in and target_points must be positive')
        if config.context_points != 512 or config.n2 != 256:
            raise ValueError('The xLSTM and minGRU experts require context_points=512 and n2=256')
        if config.xlstm_backend not in ('vanilla', 'cuda'):
            raise ValueError("xlstm_backend must be 'vanilla' or 'cuda'")
        if expert_checkpoints is not None:
            required = {'PatchTST', 'xLSTMTime', 'minGRUTime', 'ELM'}
            if set(expert_checkpoints) != required:
                raise ValueError(f'expert_checkpoints must contain exactly {sorted(required)}')
        xlstm_config.slstm_block.slstm.backend = config.xlstm_backend
        super().__init__(config, config.enc_in, num_experts_a=1, num_experts_b=1,
                         num_experts_c=1, num_experts_d=1,
                         freeze_experts=freeze_experts, expert_checkpoints=expert_checkpoints)

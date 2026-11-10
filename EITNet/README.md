# EITNet

EITNet is a PyTorch model for multivariate time-series forecasting. Its four expert branches are PatchTST, xLSTMTime, minGRUTime and ELM. The model combines expert predictions using expert interaction and temporal gating.

## Installation

Use Python 3.10 or newer and install a PyTorch build suitable for your hardware. Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The optional xLSTM CUDA backend requires a compatible NVIDIA GPU, CUDA toolkit, compiler and PyTorch installation. The `vanilla` backend uses PyTorch operations.

## Model interface

```python
import torch
from eitnet import EITNet, EITNetConfig

config = EITNetConfig(enc_in=7, context_points=512, target_points=96,
                      xlstm_backend="vanilla")
model = EITNet(config)
model.eval()
x = torch.randn(2, 512, config.enc_in)
with torch.no_grad():
    prediction = model(x)
```

The model accepts a tensor with shape `[batch, 512, channels]` and returns a tensor with shape `[batch, horizon, channels]`. Without supplied weights, its parameters are randomly initialized.

Expert weights can be supplied through `expert_checkpoints` as a dictionary keyed by `PatchTST`, `xLSTMTime`, `minGRUTime` and `ELM`. The `freeze_experts` argument defaults to `True`. Checkpoints are not included. Supply weights from a trusted source and match the channel count, input length, prediction horizon and architecture configuration used to create them.

```python
expert_checkpoints = {
    "PatchTST": "path/to/model_a.pth",
    "xLSTMTime": "path/to/model_b.pth",
    "minGRUTime": "path/to/model_c.pth",
    "ELM": "path/to/model_d.pth",
}
model = EITNet(config, expert_checkpoints=expert_checkpoints)
```

Without checkpoints, the default still freezes the randomly initialized expert parameters. Set `freeze_experts=False` to update them. Freezing parameters alone does not disable expert dropout or BatchNorm running-statistic updates in training mode; use evaluation mode on the expert modules when fixed behavior is required.

## Third-party components

The source includes components derived from [PatchTST](https://github.com/yuqinie98/PatchTST), [RevIN](https://github.com/ts-kim/RevIN), [xLSTM](https://github.com/NX-AI/xlstm), and [minGRU-pytorch](https://github.com/lucidrains/minGRU-pytorch). Original attribution and copyright notices in source files are retained. Consult the upstream projects and their license terms before redistribution.

"""Small explicit LoRA implementation; base modules are restored on every exit.

Adapter files contain only A/B matrices, not the frozen checkpoint. No PEFT
dependency or changes to Whisper's inference API are required.
"""
from contextlib import contextmanager
import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.enabled = True

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        y = self.base(x)
        if self.enabled:
            y = y + ((self.dropout(x) @ self.A.T) @ self.B.T) * self.scale
        return y


class Adapter:
    def __init__(self, model, rank=8, alpha=16., dropout=0., modules=None, encoder_top_layers=0):
        modules = modules or ["q_proj", "v_proj", "out_proj", "fc1", "fc2"]
        if type(rank) is not int or rank < 1 or not math.isfinite(alpha) or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("invalid LoRA rank/alpha/dropout")
        if not modules or not set(modules) <= {"q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"}:
            raise ValueError("unsupported LoRA modules")
        nenc = len(model.model.encoder.layers)
        if type(encoder_top_layers) is not int or not 0 <= encoder_top_layers <= nenc:
            raise ValueError("encoder_top_layers exceeds encoder depth")
        self.model = model
        self.original_training = model.training
        self.original_grad = [(p, p.requires_grad) for p in model.parameters()]
        self.sites = {}
        for name, module in list(model.named_modules()):
            parts = name.split(".")
            decoder = name.startswith("model.decoder.layers.")
            encoder = (name.startswith("model.encoder.layers.") and int(parts[3]) >= nenc - encoder_top_layers)
            if (decoder or encoder) and parts[-1] in modules and isinstance(module, nn.Linear):
                parent = model.get_submodule(name.rsplit(".", 1)[0])
                self.sites[name] = (parent, parts[-1], module)
        if not self.sites:
            raise ValueError("no LoRA target modules matched")
        for p in model.parameters():
            p.requires_grad_(False)
        try:
            for name, (parent, attr, base) in self.sites.items():
                setattr(parent, attr, LoRALinear(base, rank, alpha, dropout))
        except BaseException:
            self.restore()
            raise
        self.config = {"rank": rank, "alpha": alpha, "dropout": dropout,
                       "modules": modules, "encoder_top_layers": encoder_top_layers,
                       "sites": list(self.sites)}

    def layers(self):
        return [getattr(parent, attr) for parent, attr, _ in self.sites.values()]

    def parameters(self):
        return [p for layer in self.layers() for p in (layer.A, layer.B)]

    @contextmanager
    def teacher(self):
        """Frozen, deterministic teacher, even after student updates."""
        training = self.model.training
        self.model.eval()
        for layer in self.layers():
            layer.enabled = False
        try:
            with torch.no_grad():
                yield
        finally:
            for layer in self.layers():
                layer.enabled = True
            self.model.train(training)

    def save(self, directory, metadata):
        directory.mkdir(parents=True, exist_ok=True)
        state = {f"{name}.{key}": getattr(getattr(parent, attr), key).detach().cpu()
                 for name, (parent, attr, _) in self.sites.items() for key in ("A", "B")}
        from safetensors.torch import save_file
        import json

        save_file(state, str(directory / "adapter.safetensors"))
        (directory / "adapter.json").write_text(json.dumps({**metadata, **self.config}, indent=2))

    def restore(self):
        for parent, attr, base in self.sites.values():
            setattr(parent, attr, base)
        for p, requires_grad in self.original_grad:
            p.requires_grad_(requires_grad)
        self.model.train(self.original_training)

    def load(self, directory):
        """Load only adapter tensors into an identically configured base model."""
        import json
        from safetensors.torch import load_file

        metadata = json.loads((directory / "adapter.json").read_text())
        if any(metadata.get(k) != v for k, v in self.config.items()):
            raise ValueError("adapter architecture/configuration mismatch")
        state = load_file(str(directory / "adapter.safetensors"))
        expected = {f"{name}.{key}": getattr(getattr(parent, attr), key)
                    for name, (parent, attr, _) in self.sites.items() for key in ("A", "B")}
        if state.keys() != expected.keys() or any(state[k].shape != v.shape for k, v in expected.items()):
            raise ValueError("adapter tensor names/shapes mismatch")
        if not all(torch.isfinite(value).all() for value in state.values()):
            raise ValueError("nonfinite adapter tensors")
        with torch.no_grad():
            for key, parameter in expected.items():
                parameter.copy_(state[key].to(parameter))

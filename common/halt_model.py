"""halt_model.py — Qwen2.5-3B with per-layer halt gates and adapters.

Architecture (CALM-style soft halting):
  - Base Qwen2.5-3B frozen (fp16).
  - At each exit layer L in EXIT_LAYERS:
      p_halt_L = sigmoid(W_L · h_L)          # (B, T)
      adapter_L(h_L) = h_L + up(gelu(down(h_L)))
      logits_L = LM_head(adapter_L(h_L))     # (B, T, V)
  - Soft halting weights:
      survive_L = Π_{l<L} (1 - p_halt_l)
      p_exit_L = survive_L · p_halt_L
  - Output distribution:
      out_logits = Σ_L p_exit_L · logits_L + survive_last · logits_last
  - Expected depth (for reward shaping):
      E[depth] = Σ_L L · p_exit_L + n_layers · survive_last

Only gates + adapter parameters are trainable; base and LM head frozen.

Patches applied (v2):
  - Gate bias initialized to -2.0 (sigmoid(-2) ≈ 0.12) so the model
    starts near-full-depth and training lowers depth only where safe.
    Previous value of +2.0 caused an immediate halt at the earliest
    exit layer, leaving no room to learn selectivity.
  - Optional depth normalization to [0, 1] via --normalize_depth.
    Default off; depth loss uses raw layer indices.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


EXIT_LAYERS = [6, 12, 18, 24, 30, 36]


class BottleneckAdapter(nn.Module):
    """h -> h + up(gelu(down(h))). Initialized as identity (up=0)."""
    def __init__(self, hidden, dim=64):
        super().__init__()
        self.down = nn.Linear(hidden, dim)
        self.up = nn.Linear(dim, hidden)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(F.gelu(self.down(x)))


class HaltModel(nn.Module):
    def __init__(self, base, exit_layers=EXIT_LAYERS,
                 adapter_dim=64, train_adapters=True,
                 normalize_depth=False):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        hidden = base.config.hidden_size
        self.hidden = hidden
        self.vocab = base.config.vocab_size
        self.exit_layers = list(exit_layers)
        self.n_exits = len(self.exit_layers)
        self.n_layers = len(base.model.layers)
        self.normalize_depth = normalize_depth

        self.gates = nn.ModuleList([
            nn.Linear(hidden, 1) for _ in self.exit_layers
        ])
        for g in self.gates:
            # Bias -2.0 -> sigmoid(-2) ≈ 0.12, model starts near full
            # depth. Training raises halt probability only where safe.
            nn.init.zeros_(g.weight)
            nn.init.constant_(g.bias, -2.0)

        if train_adapters:
            self.adapters = nn.ModuleList([
                BottleneckAdapter(hidden, adapter_dim)
                for _ in self.exit_layers
            ])
        else:
            self.adapters = None

        self.lm_head = base.lm_head  # frozen

    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            out = self.base(input_ids, attention_mask=attention_mask,
                            output_hidden_states=True, use_cache=False)
            hs = out.hidden_states  # tuple of (B, T, H), len n_layers+1

        B, T = input_ids.shape
        device = input_ids.device
        head_dtype = self.lm_head.weight.dtype

        survive = torch.ones(B, T, device=device, dtype=torch.float32)
        out_logits = torch.zeros(B, T, self.vocab,
                                 device=device, dtype=torch.float32)
        expected_depth = torch.zeros(B, T, device=device, dtype=torch.float32)
        exit_mass = []

        last_logits_L = None
        for i, L in enumerate(self.exit_layers):
            h = hs[L].to(torch.float32)
            if self.adapters is not None:
                h = self.adapters[i](h)
            logits_L = self.lm_head(h.to(head_dtype)).float()

            p_halt = torch.sigmoid(self.gates[i](hs[L].to(torch.float32)))
            p_halt = p_halt.squeeze(-1)  # (B, T)

            p_exit = survive * p_halt
            out_logits = out_logits + p_exit.unsqueeze(-1) * logits_L
            expected_depth = expected_depth + float(L) * p_exit
            survive = survive * (1.0 - p_halt)
            exit_mass.append(p_exit)
            last_logits_L = logits_L

        # Remainder goes to last exit layer
        out_logits = out_logits + survive.unsqueeze(-1) * last_logits_L
        expected_depth = expected_depth + float(self.exit_layers[-1]) * survive

        if self.normalize_depth:
            d_min = float(self.exit_layers[0])
            d_max = float(self.exit_layers[-1])
            expected_depth = (expected_depth - d_min) / (d_max - d_min)
            expected_depth = expected_depth.clamp(0.0, 1.0)

        return {
            "logits": out_logits,
            "expected_depth": expected_depth,
            "exit_mass": exit_mass,
        }

    def trainable_parameters(self):
        params = list(self.gates.parameters())
        if self.adapters is not None:
            params += list(self.adapters.parameters())
        return params
"""controller_model.py — Qwen2.5-3B + LoRA + latent controller.

Controller reads h_L[t], computes c_t, injects W_in·c_t into h_L[t+1].
Layer L=18 (0-indexed: 17) is the read/write point.

Patches (v3):
  - Added `c_t` pooling (mean over sequence) to the forward output 
    to support diversity regularization (H3).
  - Reset `_last_c_norms` and `_last_c_t` at start of every forward.
  - `set_lambda` for external lambda warmup scheduling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, TaskType

CONTROLLER_LAYER = 17
CONTROLLER_DIM = 64
ADAPTER_HIDDEN = 256
LORA_R = 16
LORA_ALPHA = 32
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


class LatentController(nn.Module):
    def __init__(self, hidden_size, adapter_dim=ADAPTER_HIDDEN,
                 controller_dim=CONTROLLER_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, controller_dim),
        )
        # Small random init on output, not zero. This breaks the double-zero
        # bottleneck: at step 0, c is tiny but non-zero, so W_in gets a
        # gradient from c, and the controller gets a gradient from W_in.
        nn.init.normal_(self.mlp[-1].weight, std=0.02)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h):
        return self.mlp(h)


class ControllerModel(nn.Module):
    def __init__(self, base_model, lambda_reg=0.001):
        super().__init__()
        self.lambda_reg = float(lambda_reg)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGETS,
            lora_dropout=0.05,
        )
        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()

        hidden = self.model.base_model.config.hidden_size
        self.hidden = hidden

        self.controller = LatentController(hidden)
        self.w_in = nn.Linear(CONTROLLER_DIM, hidden, bias=False)
        # Small random init, not zero. See LatentController comment.
        nn.init.normal_(self.w_in.weight, std=0.02)

        for name, p in self.model.named_parameters():
            if "lora_" not in name:
                p.requires_grad = False

        self._hook_handle = None
        self._last_c_norms = None
        self._last_c_t = None  # <-- NEW: store pooled c_t
        self._register_hook()

    def _register_hook(self):
        layer = self.model.base_model.model.model.layers[CONTROLLER_LAYER]

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
                rest = output[1:]
            else:
                h = output
                rest = None

            # Only inject when seq_len > 1 (skip KV-cache decode steps)
            if h.shape[1] > 1:
                c = self.controller(h[:, :-1, :].float())
                injection = self.w_in(c).to(h.dtype)
                h = h.clone()
                h[:, 1:, :] = h[:, 1:, :] + injection
                
                # Store both norms and the pooled direction vector
                self._last_c_norms = c.norm(dim=-1)
                self._last_c_t = c.mean(dim=1) # <-- NEW: mean pool over sequence

            if rest is not None:
                return (h,) + rest
            return h

        self._hook_handle = layer.register_forward_hook(hook_fn)

    def set_lambda(self, lam):
        self.lambda_reg = float(lam)

    def forward(self, input_ids, attention_mask=None, labels=None,
                use_cache=False):
        # Reset before every forward to avoid stale state
        self._last_c_norms = None
        self._last_c_t = None  # <-- NEW: reset

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=use_cache,
        )
        logits = outputs.logits

        loss = None
        ce_loss = None
        reg_loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            per_token = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="none",
            )
            with torch.no_grad():
                weight = 1.0 + 1.0 * per_token.detach().clamp(min=0.0,
                                                              max=10.0)
                weight_sum = weight.sum()
            ce_loss = (per_token * weight).sum() / weight_sum.clamp_min(1e-9)

            reg_loss = torch.tensor(0.0, device=logits.device)
            if self._last_c_norms is not None and self.lambda_reg > 0:
                reg_loss = self.lambda_reg * (self._last_c_norms ** 2).mean()

            loss = ce_loss + reg_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "reg_loss": reg_loss,
            "logits": logits,
            "c_norms": self._last_c_norms,
            "c_t": self._last_c_t,  # <-- NEW: return pooled c_t
        }

    def trainable_parameters(self):
        params = [p for n, p in self.model.named_parameters() if p.requires_grad]
        params += list(self.controller.parameters())
        params += list(self.w_in.parameters())
        return params

    def lora_parameters(self):
        return [p for n, p in self.model.named_parameters() if p.requires_grad]

    def controller_parameters(self):
        return list(self.controller.parameters()) + list(self.w_in.parameters())

    def remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def reinstall_hook(self):
        if self._hook_handle is None:
            self._register_hook()

    def generate(self, input_ids, **kwargs):
        """Generate with controller active (base_model.generate)."""
        return self.model.generate(input_ids, **kwargs)

    def generate_no_controller(self, input_ids, **kwargs):
        """Generate with hook removed (LoRA-only / frozen baseline)."""
        self.remove_hook()
        try:
            return self.model.generate(input_ids, **kwargs)
        finally:
            self.reinstall_hook()
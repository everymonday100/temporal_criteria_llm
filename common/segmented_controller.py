"""segmented_controller.py — Qwen + LoRA + segmented controller (v3).

Патчи относительно v2:
  - Soft magnitude cap на выходе контроллера. Без cap L2-регуляризация
    имеет только runaway или collapse (бифуркация): direction u дрейфует
    в сторону больших x, потому что L2-градиент зависит только от |c|,
    а CE-градиент от (u, |c|) совместно. Cap делает |c| ограниченным
    по построению, direction может свободно дрейфовать.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType
from collections import defaultdict, deque

CONTROLLER_LAYER = 17
CONTROLLER_DIM = 64
WINDOW = 8
LORA_R = 4
LORA_ALPHA = 8
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
MAX_NEW_DEFAULT = 290
C_MAX_DEFAULT = 3.0


class SegmentedController(nn.Module):
    """Token encoder + mean pool + head with magnitude cap."""

    def __init__(self, hidden_size=2048, window=WINDOW,
                 ctrl_dim=CONTROLLER_DIM, c_max=C_MAX_DEFAULT):
        super().__init__()
        self.window = window
        self.ctrl_dim = ctrl_dim
        self.c_max = float(c_max)

        self.token_encoder = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, ctrl_dim),
        )
        nn.init.normal_(self.head[-1].weight, std=0.02)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, h_window):
        tokens = self.token_encoder(h_window.float())
        pooled = tokens.mean(dim=0)
        c_raw = self.head(pooled)
        # Soft cap. Identity when ||c_raw|| <= c_max.
        # When above, scale down; gradient pushes direction to reduce
        # norm without collapsing to zero (because scale factor is
        # differentiable and always positive).
        norm = c_raw.norm() + 1e-9
        scale = torch.clamp(self.c_max / norm, max=1.0)
        return c_raw * scale


def _pad_window(window, target_len):
    if window.shape[0] >= target_len:
        return window
    pad = torch.zeros(
        target_len - window.shape[0], window.shape[1],
        device=window.device, dtype=window.dtype)
    return torch.cat([pad, window], dim=0)


class SegmentedModel(nn.Module):
    """Frozen Qwen + LoRA + segmented controller."""

    def __init__(self, base, n_injections=1, lambda_reg=0.001,
                 lora_r=None, c_max=C_MAX_DEFAULT):
        super().__init__()
        self.n_injections = n_injections
        self.lambda_reg = lambda_reg

        cfg_r = lora_r if lora_r is not None else LORA_R
        cfg_alpha = cfg_r * 2
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg_r, lora_alpha=cfg_alpha,
            target_modules=LORA_TARGETS,
            lora_dropout=0.05,
        )
        self.model = get_peft_model(base, lora_cfg)
        for name, p in self.model.named_parameters():
            if "lora_" not in name:
                p.requires_grad = False

        hidden = self.model.base_model.config.hidden_size
        self.hidden = hidden

        self.controller = SegmentedController(hidden, c_max=c_max)
        self.w_in = nn.Linear(CONTROLLER_DIM, hidden, bias=False)
        nn.init.normal_(self.w_in.weight, std=0.02)

        # Hook state
        self._in_training_forward = False
        self._injection_positions = []
        self._decode_positions = set()
        self._c_vectors = []
        self._inject_positions = []
        self._h_buffer = None
        self._gen_step = 0
        self._hook_handle = None
        self._diag_x = None
        # Real-time cosine diagnostic
        self.diagnose_cosine = False
        self.cosine_buffer_size = 20
        self._cosine_buffer = deque(maxlen=self.cosine_buffer_size)
        self._current_problem_id = -1
        self._register_hook()

    def set_diag_x(self, x_tensor):
        self._diag_x = x_tensor

    def _compute_decode_positions(self, max_new=MAX_NEW_DEFAULT):
        n_decode = max(0, self.n_injections - 1)
        if n_decode == 0:
            return set()
        step = max_new / (n_decode + 1)
        return set(int(round(step * (i + 1))) for i in range(n_decode))

    def _compute_training_positions(self, response_start, response_len,
                                     seq_len):
        if response_start <= 0:
            return []
        positions = [response_start - 1]
        if self.n_injections > 1 and response_len > 0:
            n_extra = self.n_injections - 1
            step = response_len / (n_extra + 1)
            for i in range(1, n_extra + 1):
                pos = response_start + int(round(step * i)) - 1
                if 0 <= pos < seq_len - 1:
                    positions.append(pos)
        return positions

    def _register_hook(self):
        layer = self.model.base_model.model.model.layers[CONTROLLER_LAYER]

        def hook_fn(module, args, output):
            if isinstance(output, tuple):
                h, rest = output[0], output[1:]
            else:
                h, rest = output, None

            seq_len = h.shape[1]
            h_new = None
            W = self.controller.window

            if self._in_training_forward:
                for pos in self._injection_positions:
                    if pos < 0 or pos >= seq_len:
                        continue
                    lo = max(0, pos - W + 1)
                    window = h[0, lo:pos + 1, :]
                    window = _pad_window(window, W)
                    c = self.controller(window)
                    if self._diag_x is not None:
                        u = c / (c.norm() + 1e-9)
                        c = self._diag_x * u
                    inj = self.w_in(c).to(h.dtype)
                    if h_new is None:
                        h_new = h.clone()
                    h_new[:, pos, :] = h_new[:, pos, :] + inj
                    self._c_vectors.append(c)
                    self._inject_positions.append(pos)

            elif seq_len > 1:
                if self.n_injections >= 1:
                    pos = seq_len - 1
                    lo = max(0, pos - W + 1)
                    window = h[0, lo:pos + 1, :]
                    window = _pad_window(window, W)
                    c = self.controller(window)
                    inj = self.w_in(c).to(h.dtype)
                    h_new = h.clone()
                    h_new[:, pos, :] = h_new[:, pos, :] + inj
                    if torch.is_grad_enabled():
                        self._c_vectors.append(c)
                    else:
                        self._c_vectors.append(c.detach())
                    self._inject_positions.append(0)
                self._gen_step = 0
                self._h_buffer = None

            elif seq_len == 1 and self.n_injections > 1:
                cur = h[0, -1:, :]
                if self._h_buffer is None:
                    self._h_buffer = cur.clone()
                else:
                    self._h_buffer = torch.cat(
                        [self._h_buffer, cur], dim=0)[-W:]
                self._gen_step += 1
                if self._gen_step in self._decode_positions:
                    buf = _pad_window(self._h_buffer, W)
                    c = self.controller(buf)
                    inj = self.w_in(c).to(h.dtype)
                    h_new = h.clone()
                    h_new[:, -1, :] = h_new[:, -1, :] + inj
                    if torch.is_grad_enabled():
                        self._c_vectors.append(c)
                    else:
                        self._c_vectors.append(c.detach())
                    self._inject_positions.append(self._gen_step)

            if h_new is not None:
                h = h_new
            if rest is not None:
                return (h,) + rest
            return h

        self._hook_handle = layer.register_forward_hook(hook_fn)

    def forward(self, input_ids, attention_mask=None, labels=None):
        seq_len = input_ids.shape[1]

        self._injection_positions = []
        if labels is not None:
            mask = (labels != -100)
            if mask.any():
                response_start = int(mask.float().argmax().item())
                response_len = int(mask.sum().item())
                self._injection_positions = self._compute_training_positions(
                    response_start, response_len, seq_len)

        self._c_vectors = []
        self._inject_positions = []
        self._h_buffer = None
        self._gen_step = 0
        self._decode_positions = self._compute_decode_positions()

        self._in_training_forward = True
        try:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False,
            )
        finally:
            self._in_training_forward = False

        logits = outputs.logits

        loss = None
        ce_loss = None
        ce_raw = None
        reg_loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ce_per_token = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="none",
            )
            ce_raw = ce_per_token.mean()
            with torch.no_grad():
                weight = 1.0 + 1.0 * ce_per_token.detach().clamp(
                    min=0.0, max=10.0)
                weight_sum = weight.sum()
            ce_loss = (ce_per_token * weight).sum() / weight_sum.clamp_min(1e-9)
            reg_loss = torch.tensor(0.0, device=logits.device)
            if self._c_vectors and self.lambda_reg > 0:
                norms = torch.stack([c.norm() for c in self._c_vectors])
                reg_loss = self.lambda_reg * norms.mean()   # L1
            loss = ce_loss + reg_loss

        c_norms_snapshot = [float(c.norm().detach().item())
                            for c in self._c_vectors]

        # Real-time cosine diagnostic: capture per-position vectors
        # for this problem before clearing the working lists.
        if self.diagnose_cosine and self._c_vectors:
            per_pos = {}
            for pos, c in zip(self._inject_positions, self._c_vectors):
                if int(pos) not in per_pos:
                    per_pos[int(pos)] = (
                        c.detach().float().cpu().numpy().copy())
            self._cosine_buffer.append({
                "pid": self._current_problem_id,
                "per_pos": per_pos,
            })

        self._c_vectors = []

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "ce_raw": ce_raw,
            "reg_loss": reg_loss,
            "logits": logits,
            "c_norms": c_norms_snapshot,
        }

    def lora_parameters(self):
        return [p for n, p in self.model.named_parameters()
                if p.requires_grad]

    def controller_parameters(self):
        return (list(self.controller.parameters()) +
                list(self.w_in.parameters()))

    def remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def reset_hooks(self):
        self.remove_hook()
        self._register_hook()

    def set_problem_id(self, pid):
        self._current_problem_id = int(pid)

    def enable_cosine_diagnostic(self, buffer_size=20):
        self.diagnose_cosine = True
        self.cosine_buffer_size = buffer_size
        self._cosine_buffer = deque(maxlen=buffer_size)

    def compute_cosine_within(self):
        """Pairwise cosine within each injection position across
        problems in the buffer. Deduplicated by problem_id (keeps
        latest entry per problem).
        """
        if not self._cosine_buffer:
            return {}
        latest_by_pid = {}
        for entry in self._cosine_buffer:
            latest_by_pid[entry["pid"]] = entry["per_pos"]
        by_pos = defaultdict(list)
        for per_pos in latest_by_pid.values():
            for pos, v in per_pos.items():
                by_pos[pos].append(v)
        out = {}
        for pos, vecs in by_pos.items():
            if len(vecs) < 2:
                continue
            cosines = []
            for i in range(len(vecs)):
                for j in range(i + 1, len(vecs)):
                    a, b = vecs[i], vecs[j]
                    na = float(np.linalg.norm(a)) + 1e-9
                    nb = float(np.linalg.norm(b)) + 1e-9
                    cosines.append(float(np.dot(a, b) / (na * nb)))
            if cosines:
                out[int(pos)] = {
                    "mean": float(np.mean(cosines)),
                    "std": float(np.std(cosines)),
                    "n_pairs": len(cosines),
                    "n_problems": len(vecs),
                }
        return out
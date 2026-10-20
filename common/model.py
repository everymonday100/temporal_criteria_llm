"""model.py — unified loader for Qwen2.5 and Mamba2 checkpoints."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


QWEN_PATH  = r"E:\OllamaModels\Qwen2.5-3B"
MAMBA_PATH = r"E:\OllamaModels\mamba2-1.3b-hf"


def load_model(kind: str = "qwen", path: str = None):
    """kind: 'qwen' | 'mamba2'. path overrides the default location."""
    if kind == "mamba2":
        return _load_mamba(path or MAMBA_PATH)
    return _load_qwen(path or QWEN_PATH)


def _load_qwen(path):
    print(f"[model] loading Qwen from {path}")
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=torch.float16,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    print(f"[model] qwen device={next(model.parameters()).device}")
    return tok, model


def _load_mamba(path):
    print(f"[model] loading Mamba2 from {path}")
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # No attn_implementation — Mamba has no attention.
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.eval()
    print(f"[model] mamba device={next(model.parameters()).device}")

    # Diagnostics — confirm which unembedding access path exists.
    print(f"[model] has get_output_embeddings: "
          f"{hasattr(model, 'get_output_embeddings')}")
    print(f"[model] has lm_head: {hasattr(model, 'lm_head')}")

    return tok, model


def unload_model(model):
    del model
    torch.cuda.empty_cache()
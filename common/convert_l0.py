"""convert_l0.py — convert cache/sm_L0.npz to sm/L0_traj.npz."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CACHE = Path(r"E:\OllamaModels\temporal_llm\data\stage1\cache")
SM    = Path(r"E:\OllamaModels\temporal_llm\data\stage1\sm")


def convert(sfx=""):
    src = CACHE / f"sm_L0{sfx}.npz"
    dst = SM / f"L0{sfx}_traj.npz"
    if not src.exists():
        print(f"[skip] {src.name} missing")
        return
    d = np.load(src, allow_pickle=True)
    n = int(d["n_prompts"])
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    n_train = int(n * 0.7)
    train_ids = sorted(perm[:n_train].tolist())
    test_ids  = sorted(perm[n_train:].tolist())
    np.savez(dst,
             iface="L0",
             lambdas=d["lambdas"],
             train_ids=np.array(train_ids),
             test_ids=np.array(test_ids),
             prompts=d["prompts"] if "prompts" in d else np.array([], dtype=object),
             seg_texts=d["texts"] if "texts" in d else np.array([], dtype=object))
    print(f"[ok] wrote {dst}")


if __name__ == "__main__":
    for sfx in ("", "_mamba2"):
        convert(sfx)
"""boundary_train_fm.py — Train and evaluate forward model on extracted hidden states."""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, r2_score
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.boundary_model import ForwardModel

OUT_DIR = Path(r"E:\OllamaModels\temporal_llm\data\stage1\boundary")
LAYERS = [12, 18, 24]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["F", "L"], default="F")
    p.add_argument("--design", choices=["R", "O"], required=True)
    p.add_argument("--control", type=str, default="none",
                   choices=["none", "C1", "C2", "C7", "C8", "C9"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_data(design, variant, control, layer):
    train_file = OUT_DIR / f"hidden_train_streams_{variant}.npz"
    test_file = OUT_DIR / f"hidden_test_streams_{variant}.npz"

    tr = np.load(train_file)
    te = np.load(test_file)

    tr_mask = tr["design"] == design
    te_mask = te["design"] == design

    h_tr = torch.tensor(tr[f"h_{layer}"][tr_mask], dtype=torch.float32)
    a_tr = torch.tensor(tr["action"][tr_mask], dtype=torch.long)
    y_tr = torch.tensor(tr["delta_y"][tr_mask], dtype=torch.float32)

    h_te = torch.tensor(te[f"h_{layer}"][te_mask], dtype=torch.float32)
    a_te = torch.tensor(te["action"][te_mask], dtype=torch.long)
    y_te = torch.tensor(te["delta_y"][te_mask], dtype=torch.float32)
    is_ext_te = torch.tensor(te["is_external"][te_mask], dtype=torch.bool)
    sids_te = te["stream_id"][te_mask]

    if control == "C2":
        perm = torch.randperm(len(a_tr))
        a_tr = a_tr[perm]
    elif control == "C7":
        a_tr = torch.randint(-2, 3, a_tr.shape, dtype=torch.long)
        a_te = torch.randint(-2, 3, a_te.shape, dtype=torch.long)
    elif control == "C8":
        sids = tr["stream_id"][tr_mask]
        for sid in np.unique(sids):
            idx = np.where(sids == sid)[0]
            if len(idx) > 1:
                a_vals = a_tr[idx].clone()
                a_vals[1:] = a_vals[:-1].clone()  # <--- добавить .clone()
                a_vals[0] = 0
                a_tr[idx] = a_vals
    elif control == "C9":
        is_ext_tr = tr["is_external"][tr_mask]
        self_idx = np.where(~is_ext_tr)[0]
        ext_idx = np.where(is_ext_tr)[0]
        n = min(len(self_idx), len(ext_idx))
        if n > 0:
            chosen = np.concatenate([self_idx[:n], ext_idx[:n]])
            h_tr = h_tr[chosen]
            a_tr = a_tr[chosen]
            y_tr = y_tr[chosen]

    return (h_tr, a_tr, y_tr), (h_te, a_te, y_te, is_ext_te, sids_te)


def train_model(h_tr, a_tr, y_tr, h_te, a_te, y_te, is_ext_te, epochs=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ForwardModel(hidden_size=h_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    ds = TensorDataset(h_tr, a_tr, y_tr)
    dl = DataLoader(ds, batch_size=128, shuffle=True)

    model.train()
    for ep in range(epochs):
        for bh, ba, by in dl:
            bh, ba, by = bh.to(device), ba.to(device), by.to(device)
            opt.zero_grad()
            pred = model(bh, ba)
            loss = loss_fn(pred, by)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred_y = model(h_te.to(device), a_te.to(device)).cpu()
        pred_tr = model(h_tr.to(device), a_tr.to(device)).cpu()

    residuals = (y_te - pred_y).abs()
    return residuals, pred_y, pred_tr


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[FM Train] Design={args.design} Variant={args.variant} "
          f"Control={args.control} Seed={args.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for layer in LAYERS:
        (h_tr, a_tr, y_tr), (h_te, a_te, y_te, is_ext_te, sids_te) = load_data(
            args.design, args.variant, args.control, layer
        )

        if args.control == "C1":
            model = ForwardModel(hidden_size=h_tr.shape[1]).to(device)
            model.eval()
            with torch.no_grad():
                pred_y = model(h_te.to(device), a_te.to(device)).cpu()
                pred_tr = model(h_tr.to(device), a_tr.to(device)).cpu()
            residuals = (y_te - pred_y).abs()
        else:
            residuals, pred_y, pred_tr = train_model(
                h_tr, a_tr, y_tr, h_te, a_te, y_te, is_ext_te
            )

        auc = roc_auc_score(is_ext_te.numpy(), residuals.numpy())
        r2_test = r2_score(y_te.numpy(), pred_y.numpy())
        r2_train = r2_score(y_tr.numpy(), pred_tr.numpy())

        results[layer] = {"auc": auc, "r2": r2_test, "r2_train": r2_train}
        print(f"  L={layer:2d}  AUC={auc:.4f}  "
              f"R2_test={r2_test:+.4f}  R2_train={r2_train:+.4f}")

        out_npz = OUT_DIR / f"preds_{args.design}_{args.variant}_{args.control}_L{layer}.npz"
        np.savez(out_npz, stream_id=sids_te,
                 residuals=residuals.numpy(), is_ext=is_ext_te.numpy())

    import json
    out_json = OUT_DIR / f"fm_results_{args.design}_{args.variant}_{args.control}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
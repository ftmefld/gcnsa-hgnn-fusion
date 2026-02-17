import os
import argparse
import numpy as np
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from loader import MoleculeDataset
from splitters import scaffold_split
import pandas as pd
from sklearn.metrics import roc_auc_score

# ----------------------- utils -----------------------
class FusionMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.ff(x)

class FusionDataset(Dataset):
    def __init__(self, Xg: np.ndarray, Xh: np.ndarray, y: np.ndarray):
        X = np.concatenate([Xg, Xh], axis=1)
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_npz(path: str) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.load(path, allow_pickle=False)
    return arr["mol_id"].astype(np.int64), arr["emb"].astype(np.float32)


def inner_join_by_id(ids1, X1, ids2, X2):
    m1 = {int(k): i for i, k in enumerate(ids1.tolist())}
    take1, take2, out = [], [], []
    for j, k in enumerate(ids2.tolist()):
        k = int(k)
        if k in m1:
            out.append(k)
            take1.append(m1[k])
            take2.append(j)
    if not out:
        raise RuntimeError("No overlapping mol_id between embeddings.")
    out = np.array(out, dtype=np.int64)
    return out, X1[np.array(take1)], X2[np.array(take2)]


def zscore_block(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-9
    return (X - mu) / sd


def masked_bce_loss(logits, y01):
    return F.binary_cross_entropy_with_logits(logits, y01)



def eval_auc(logits: np.ndarray, y_true: np.ndarray) -> float:
    rocs = []
    for t in range(y_true.shape[1]):
        mask = (y_true[:, t] * y_true[:, t]) > 0
        if mask.sum() == 0:
            continue
        yt = (y_true[mask, t] + 1.0) / 2.0
        ys = logits[mask, t]
        if (yt == 1).sum() > 0 and (yt == 0).sum() > 0:
            rocs.append(roc_auc_score(yt, ys))
    return float(np.mean(rocs)) if rocs else float("nan")

# ----------------------- labels via scaffold split -----------------------

def build_split_labels(data_root: str, dataset: str, seed: int, split: str):
    ds = MoleculeDataset(os.path.join(data_root, dataset), dataset=dataset)
    smiles_csv = os.path.join(data_root, dataset, 'raw', 'sider.txt')
    smiles = pd.read_csv(smiles_csv, header=None)[0].tolist()
    tr, va, te = scaffold_split(ds, smiles, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=seed)
    dmap = {"train": tr, "val": va, "test": te}[split]
    # iterate to collect ids, y
    ids, ys = [], []
    loader = DataLoader(dmap, batch_size=1024, shuffle=False)
    for batch in loader:
        ids.append(batch.id.view(-1).numpy())
        ys.append(batch.y.view(batch.y.size(0), -1).numpy())
    ids = np.concatenate(ids, axis=0).astype(np.int64)
    ys = np.concatenate(ys, axis=0).astype(np.float32)
    return ids, ys

# ----------------------- main -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--gcn_train', required=True)
    ap.add_argument('--gcn_val', required=True)
    ap.add_argument('--gcn_test', required=True)
    ap.add_argument('--hgnn_train', required=True)
    ap.add_argument('--hgnn_val', required=True)
    ap.add_argument('--hgnn_test', required=True)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--normalize', action='store_true')
    ap.add_argument('--device', type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device('cpu')

    # --- load embeddings ---
    id_g_tr, Xg_tr = load_npz(args.gcn_train)
    id_g_va, Xg_va = load_npz(args.gcn_val)
    id_g_te, Xg_te = load_npz(args.gcn_test)
    id_h_tr, Xh_tr = load_npz(args.hgnn_train)
    id_h_va, Xh_va = load_npz(args.hgnn_val)
    id_h_te, Xh_te = load_npz(args.hgnn_test)

    # --- align by mol_id per split ---
    ids_tr, Xg_tr, Xh_tr = inner_join_by_id(id_g_tr, Xg_tr, id_h_tr, Xh_tr)
    ids_va, Xg_va, Xh_va = inner_join_by_id(id_g_va, Xg_va, id_h_va, Xh_va)
    ids_te, Xg_te, Xh_te = inner_join_by_id(id_g_te, Xg_te, id_h_te, Xh_te)

    # --- optional block-wise normalization ---
    if args.normalize:
        Xg_tr = zscore_block(Xg_tr); Xh_tr = zscore_block(Xh_tr)
        # use train stats to transform val/test
        def apply_stats(X, mu, sd):
            return (X - mu) / (sd + 1e-9)
        g_mu, g_sd = Xg_tr.mean(0, keepdims=True), Xg_tr.std(0, keepdims=True)
        h_mu, h_sd = Xh_tr.mean(0, keepdims=True), Xh_tr.std(0, keepdims=True)
        Xg_va = apply_stats(Xg_va, g_mu, g_sd); Xh_va = apply_stats(Xh_va, h_mu, h_sd)
        Xg_te = apply_stats(Xg_te, g_mu, g_sd); Xh_te = apply_stats(Xh_te, h_mu, h_sd)

    # --- labels per split (same scaffold split/seed) ---
    ids_y_tr, y_tr = build_split_labels(args.data_dir, args.dataset, args.seed, 'train')
    ids_y_va, y_va = build_split_labels(args.data_dir, args.dataset, args.seed, 'val')
    ids_y_te, y_te = build_split_labels(args.data_dir, args.dataset, args.seed, 'test')

    # align labels to embedding ids
    def align_y(ids_emb, ids_lbl, Y):
        pos = {int(k): i for i, k in enumerate(ids_lbl.tolist())}
        take = [pos[int(k)] for k in ids_emb.tolist()]
        return Y[np.array(take)]

    y_tr = align_y(ids_tr, ids_y_tr, y_tr)
    y_va = align_y(ids_va, ids_y_va, y_va)
    y_te = align_y(ids_te, ids_y_te, y_te)

    # --- datasets & loaders ---
    dtr = FusionDataset(Xg_tr, Xh_tr, y_tr)
    dva = FusionDataset(Xg_va, Xh_va, y_va)
    dte = FusionDataset(Xg_te, Xh_te, y_te)

    Ltr = DataLoader(dtr, batch_size=args.batch_size, shuffle=True)
    Lva = DataLoader(dva, batch_size=args.batch_size, shuffle=False)
    Lte = DataLoader(dte, batch_size=args.batch_size, shuffle=False)

    # --- model ---
    in_dim = dtr.X.shape[1]
    out_dim = dtr.y.shape[1]
    model = FusionMLP(in_dim, out_dim, hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_auc = -1.0
    best_state: Dict[str, torch.Tensor] = {}
    patience, bad = 20, 0

    for epoch in range(1, args.epochs + 1):
        # train
        model.train()
        for xb, yb in Ltr:
            xb = xb.to(device); yb = yb.to(device)
            logits = model(xb)
            loss = masked_bce_loss(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()

        # eval
        model.eval()
        def collect(loader):
            all_logits, all_y = [], []
            with torch.no_grad():
                for xb, yb in loader:
                    xb = xb.to(device)
                    all_logits.append(model(xb).cpu().numpy())
                    all_y.append(yb.numpy())
            return np.concatenate(all_logits), np.concatenate(all_y)
        val_logits, val_y = collect(Lva)
        test_logits, test_y = collect(Lte)
        tr_logits, tr_y = collect(Ltr)

        val_auc = eval_auc(val_logits, val_y)
        tr_auc  = eval_auc(tr_logits, tr_y)
        te_auc  = eval_auc(test_logits, test_y)
        print(f"Epoch {epoch:03d} | train AUC {tr_auc:.4f} | val AUC {val_auc:.4f} | test AUC {te_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    if best_state:
        model.load_state_dict(best_state)
        val_logits, val_y = next(((np.concatenate([model(torch.from_numpy(dva.X[i*args.batch_size:(i+1)*args.batch_size]).to(device)).detach().cpu().numpy() for i in range((len(dva)+args.batch_size-1)//args.batch_size)]), dva.y.numpy())), (None, None))

    # Final evaluation
    model.eval()
    def collect(loader):
        all_logits, all_y = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                all_logits.append(model(xb).cpu().numpy())
                all_y.append(yb.numpy())
        return np.concatenate(all_logits), np.concatenate(all_y)
    val_logits, val_y = collect(Lva)
    test_logits, test_y = collect(Lte)
    tr_logits, tr_y = collect(Ltr)
    print("==== FINAL ====")
    print(f"Train AUC: {eval_auc(tr_logits, tr_y):.6f}")
    print(f"Val   AUC: {eval_auc(val_logits, val_y):.6f}")
    print(f"Test  AUC: {eval_auc(test_logits, test_y):.6f}")

if __name__ == '__main__':
    main()

import os
import argparse
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid


# ------------------ utils ------------------
def load_npz_emb(path: str):
    arr = np.load(path, allow_pickle=False)
    return arr["mol_id"].astype(np.int64), arr["emb"].astype(np.float32)

def load_npz_labels(path: str):
    arr = np.load(path, allow_pickle=False)
    return arr["mol_id"].astype(np.int64), arr["y"].astype(np.float32)

def inner_join_by_id(ids1, X1, ids2, X2):
    pos = {int(k): i for i, k in enumerate(ids1.tolist())}
    out_ids, t1, t2 = [], [], []
    for j, k in enumerate(ids2.tolist()):
        k = int(k)
        if k in pos:
            out_ids.append(k); t1.append(pos[k]); t2.append(j)
    if not out_ids:
        raise RuntimeError("No overlapping mol_id between sources.")
    idx1 = np.array(t1, dtype=np.int64)
    idx2 = np.array(t2, dtype=np.int64)
    return np.array(out_ids, dtype=np.int64), X1[idx1], X2[idx2]

def align_labels_safe(ids_emb: np.ndarray, ids_lab: np.ndarray, Y: np.ndarray):
    pos = {int(k): i for i, k in enumerate(ids_lab.tolist())}
    take_lab, take_emb, missing = [], [], 0
    for j, k in enumerate(ids_emb.tolist()):
        k = int(k)
        if k in pos:
            take_emb.append(j); take_lab.append(pos[k])
        else:
            missing += 1
    if len(take_emb) == 0:
        raise RuntimeError("No overlapping IDs between embeddings and labels for this split.")
    if missing > 0:
        print(f"[warn] {missing} / {len(ids_emb)} embedding IDs had no labels and were dropped for this split.")
    ids_new = ids_emb[np.array(take_emb, dtype=np.int64)]
    Y_new   = Y[np.array(take_lab, dtype=np.int64)]
    return ids_new, Y_new

def zscore_fit(X):
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True) + 1e-9
    return mu, sd

def zscore_apply(X, mu, sd):
    return (X - mu) / sd

def auc_multi(logits_np: np.ndarray, y_np: np.ndarray, labels_signed: bool) -> float:
    rocs = []
    for t in range(y_np.shape[1]):
        if labels_signed:
            mask = (y_np[:, t] * y_np[:, t]) > 0
            if mask.sum() == 0:
                continue
            yt = (y_np[mask, t] + 1.0) / 2.0
            ys = logits_np[mask, t]
        else:
            yt = y_np[:, t]; ys = logits_np[:, t]
            if (yt == 1).sum() == 0 or (yt == 0).sum() == 0:
                continue
        rocs.append(roc_auc_score(yt, ys))
    return float(np.mean(rocs)) if rocs else float("nan")


# ------------------ data/model ------------------
class FusionDataset2(Dataset):
    def __init__(self, Xg: np.ndarray, Xa: np.ndarray, Y: np.ndarray):
        X = np.concatenate([Xg, Xa], axis=1)
        self.X = torch.from_numpy(X.astype(np.float32))
        self.Y = torch.from_numpy(Y.astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


class GatedFusion2MLP(nn.Module):
    def __init__(self, dim_g: int, dim_a: int, out_dim: int,
                 hidden: int = 768, dropout: float = 0.3):
        super().__init__()
        self.dim_g, self.dim_a = dim_g, dim_a

        # split hidden into 2 parts
        h_g = hidden // 2 + (1 if hidden % 2 > 0 else 0)
        h_a = hidden - h_g

        self.gate_g = nn.Sequential(nn.LayerNorm(dim_g), nn.Linear(dim_g, dim_g), nn.Sigmoid())
        self.gate_a = nn.Sequential(nn.LayerNorm(dim_a), nn.Linear(dim_a, dim_a), nn.Sigmoid())

        self.proj_g = nn.Linear(dim_g, h_g)
        self.proj_a = nn.Linear(dim_a, h_a)

        self.mlp = nn.Sequential(
            nn.LayerNorm(h_g + h_a),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(h_g + h_a, h_g + h_a),
            nn.LayerNorm(h_g + h_a),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(h_g + h_a, out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xg = x[:, :self.dim_g]
        xa = x[:, self.dim_g:]

        xg = xg * self.gate_g(xg)
        xa = xa * self.gate_a(xa)

        hg = self.proj_g(xg)
        ha = self.proj_a(xa)

        h = torch.cat([hg, ha], dim=1)
        return self.mlp(h)


# ------------------ single run: train + eval ------------------
def train_eval(args):
    # ----- Seeding & determinism -----
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")

    # ---- load embeddings ----
    id_g_tr, Xg_tr = load_npz_emb(args.gcn_train)
    id_g_va, Xg_va = load_npz_emb(args.gcn_val)
    id_g_te, Xg_te = load_npz_emb(args.gcn_test)

    id_a_tr, Xa_tr = load_npz_emb(args.apgcn_train)
    id_a_va, Xa_va = load_npz_emb(args.apgcn_val)
    id_a_te, Xa_te = load_npz_emb(args.apgcn_test)

    # ---- join by id across both (per split) ----
    ids_tr, Xg_tr, Xa_tr = inner_join_by_id(id_g_tr, Xg_tr, id_a_tr, Xa_tr)
    ids_va, Xg_va, Xa_va = inner_join_by_id(id_g_va, Xg_va, id_a_va, Xa_va)
    ids_te, Xg_te, Xa_te = inner_join_by_id(id_g_te, Xg_te, id_a_te, Xa_te)

    # ---- labels ----
    lid_tr, y_tr = load_npz_labels(args.lab_train)
    lid_va, y_va = load_npz_labels(args.lab_val)
    lid_te, y_te = load_npz_labels(args.lab_test)

    ids_tr, y_tr = align_labels_safe(ids_tr, lid_tr, y_tr)
    ids_va, y_va = align_labels_safe(ids_va, lid_va, y_va)
    ids_te, y_te = align_labels_safe(ids_te, lid_te, y_te)

    # ---- normalization (fit on train, apply to all) ----
    if args.normalize:
        g_mu, g_sd = zscore_fit(Xg_tr); Xg_tr = zscore_apply(Xg_tr, g_mu, g_sd)
        a_mu, a_sd = zscore_fit(Xa_tr); Xa_tr = zscore_apply(Xa_tr, a_mu, a_sd)

        Xg_va = zscore_apply(Xg_va, g_mu, g_sd)
        Xa_va = zscore_apply(Xa_va, a_mu, a_sd)

        Xg_te = zscore_apply(Xg_te, g_mu, g_sd)
        Xa_te = zscore_apply(Xa_te, a_mu, a_sd)

    # ---- datasets/loaders ----
    dtr = FusionDataset2(Xg_tr, Xa_tr, y_tr)
    dva = FusionDataset2(Xg_va, Xa_va, y_va)
    dte = FusionDataset2(Xg_te, Xa_te, y_te)

    g = torch.Generator()
    g.manual_seed(args.seed)
    Ltr = DataLoader(dtr, batch_size=args.batch_size, shuffle=True, generator=g)
    Lva = DataLoader(dva, batch_size=args.batch_size, shuffle=False)
    Lte = DataLoader(dte, batch_size=args.batch_size, shuffle=False)

    # ---- model ----
    dim_g = Xg_tr.shape[1]
    dim_a = Xa_tr.shape[1]
    out_dim = dtr.Y.shape[1]

    model = GatedFusion2MLP(dim_g, dim_a, out_dim, hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # pos_weight (optional; computed from TRAIN labels only)
    pos_weight = None
    if args.use_pos_weight:
        pw = []
        for t in range(y_tr.shape[1]):
            if args.labels_signed:
                mask = (y_tr[:, t] * y_tr[:, t]) > 0
                pos = ((y_tr[mask, t] + 1.0) / 2.0).sum()
                neg = mask.sum() - pos
            else:
                mask = np.ones_like(y_tr[:, t], dtype=bool)
                pos = (y_tr[mask, t] == 1).sum()
                neg = mask.sum() - pos
            pw.append(float(neg / max(1.0, pos)))
        pos_weight = torch.tensor(pw, dtype=torch.float32, device=device)

    def loss_fn(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if args.labels_signed:
            y01 = (y + 1.0) / 2.0
            mask = (y * y) > 0
            if pos_weight is None:
                loss = F.binary_cross_entropy_with_logits(logits, y01, reduction='none')
            else:
                loss = F.binary_cross_entropy_with_logits(logits, y01, pos_weight=pos_weight, reduction='none')
            loss = loss[mask].mean() if mask.any() else torch.tensor(0.0, device=logits.device)
        else:
            if pos_weight is None:
                loss = F.binary_cross_entropy_with_logits(logits, y, reduction='mean')
            else:
                loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction='mean')
        return loss

    def collect_logits(loader: DataLoader):
        model.eval()
        outs, ys = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                outs.append(model(xb).cpu().numpy())
                ys.append(yb.numpy())
        return np.concatenate(outs, 0), np.concatenate(ys, 0)

    # ---- training with early stopping on val AUC ----
    best_auc, best_state, bad = -1.0, None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        for xb, yb in Ltr:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()

        tr_logits, tr_y = collect_logits(Ltr)
        va_logits, va_y = collect_logits(Lva)
        te_logits, te_y = collect_logits(Lte)

        tr_auc = auc_multi(tr_logits, tr_y, labels_signed=args.labels_signed)
        va_auc = auc_multi(va_logits, va_y, labels_signed=args.labels_signed)
        te_auc = auc_multi(te_logits, te_y, labels_signed=args.labels_signed)
        print(f"Epoch {ep:03d} | train AUC {tr_auc:.4f} | val AUC {va_auc:.4f} | test AUC {te_auc:.4f}")

        if va_auc > best_auc:
            best_auc = va_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
            if args.save_best:
                torch.save(best_state, args.save_best)
        else:
            bad += 1
            if bad >= args.early_patience:
                print("Early stopping.")
                break

    if best_state:
        model.load_state_dict(best_state)

    tr_logits, tr_y = collect_logits(Ltr)
    va_logits, va_y = collect_logits(Lva)
    te_logits, te_y = collect_logits(Lte)

    final_tr_auc = auc_multi(tr_logits, tr_y, args.labels_signed)
    final_va_auc = auc_multi(va_logits, va_y, args.labels_signed)
    final_te_auc = auc_multi(te_logits, te_y, args.labels_signed)

    print("==== FINAL (best on val) ====")
    print(f"Train AUC: {final_tr_auc:.6f}")
    print(f"Val   AUC: {final_va_auc:.6f}")
    print(f"Test  AUC: {final_te_auc:.6f}")

    if args.save_logits:
        np.savez(args.save_logits,
                 train_logits=tr_logits, train_y=tr_y,
                 val_logits=va_logits,   val_y=va_y,
                 test_logits=te_logits,  test_y=te_y)
        print(f"Saved logits to {args.save_logits}")

    return best_auc, final_tr_auc, final_va_auc, final_te_auc


# ------------------ main + grid search ------------------
def _parse_list(s: str, cast):
    s = (s or "").strip()
    if not s:
        return []
    return [cast(x) for x in s.split(",")]

def main():
    ap = argparse.ArgumentParser()

    # embeddings (2 sources)
    ap.add_argument('--gcn_train', required=True)
    ap.add_argument('--gcn_val',   required=True)
    ap.add_argument('--gcn_test',  required=True)

    ap.add_argument('--apgcn_train', required=True)
    ap.add_argument('--apgcn_val',   required=True)
    ap.add_argument('--apgcn_test',  required=True)

    # labels
    ap.add_argument('--lab_train', required=True)
    ap.add_argument('--lab_val',   required=True)
    ap.add_argument('--lab_test',  required=True)

    # training
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--hidden', type=int, default=768)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--early_patience', type=int, default=30)

    # seeding
    ap.add_argument('--seed', type=int, default=42, help='random seed for numpy/torch/cuda')

    # options
    ap.add_argument('--normalize', action='store_true')
    ap.add_argument('--labels_signed', action='store_true')
    ap.add_argument('--use_pos_weight', action='store_true')
    ap.add_argument('--save_best', type=str, default='fusion2_best.pt')
    ap.add_argument('--save_logits', type=str, default='')

    # grid search options
    ap.add_argument('--do_gridsearch', action='store_true')
    ap.add_argument('--grid_hidden', type=str, default='')
    ap.add_argument('--grid_dropout', type=str, default='')
    ap.add_argument('--grid_lr', type=str, default='')
    ap.add_argument('--grid_wd', type=str, default='')
    ap.add_argument('--grid_seed', type=str, default='')

    args = ap.parse_args()

    if not args.do_gridsearch:
        train_eval(args)
        return

    hidden_list  = _parse_list(args.grid_hidden, int)    or [args.hidden]
    dropout_list = _parse_list(args.grid_dropout, float) or [args.dropout]
    lr_list      = _parse_list(args.grid_lr, float)      or [args.lr]
    wd_list      = _parse_list(args.grid_wd, float)      or [args.wd]
    seed_list    = _parse_list(args.grid_seed, int)      or [args.seed]

    grid = ParameterGrid({
        "hidden":  hidden_list,
        "dropout": dropout_list,
        "lr":      lr_list,
        "wd":      wd_list,
        "seed":    seed_list,
    })

    best_cfg = None
    best_val = -1.0
    best_test = None

    print(f"\nRunning grid search over {len(grid)} configurations...\n")
    for i, hp in enumerate(grid):
        print("=" * 80)
        print(f"Config {i+1}/{len(grid)}: hidden={hp['hidden']}, dropout={hp['dropout']}, "
              f"lr={hp['lr']}, wd={hp['wd']}, seed={hp['seed']}")
        print("=" * 80)

        cfg = copy.deepcopy(args)
        cfg.hidden = hp["hidden"]
        cfg.dropout = hp["dropout"]
        cfg.lr = hp["lr"]
        cfg.wd = hp["wd"]
        cfg.seed = hp["seed"]

        # avoid overwriting files during grid search
        cfg.save_best = ""
        cfg.save_logits = ""

        best_auc, tr_auc, va_auc, te_auc = train_eval(cfg)
        print(f"[GRID] Config {i+1}: best_val_auc={best_auc:.6f}, final_val_auc={va_auc:.6f}, test_auc={te_auc:.6f}")

        # choose by final val AUC (same as your original code)
        score = va_auc
        if score > best_val:
            best_val = score
            best_test = te_auc
            best_cfg = hp

    print("\n===== GRID SEARCH SUMMARY =====")
    print(f"Best config (by val AUC): {best_cfg}")
    print(f"Best val AUC:   {best_val:.6f}")
    print(f"Test AUC @best: {best_test:.6f}")


if __name__ == "__main__":
    main()

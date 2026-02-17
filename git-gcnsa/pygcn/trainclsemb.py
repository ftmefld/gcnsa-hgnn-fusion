from __future__ import division
from __future__ import print_function
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os

from utils import accuracy, precision, recall, f1, auc, acquiretvt
from models import GCNSA
import utils_data
import torch.nn.functional as F


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ===== NEW: device helpers =====
def _resolve_device(device_arg: str) -> torch.device:
    """
    Accepts: 'auto' | 'cpu' | 'cuda' | 'cuda:0', 'cuda:1', ...
    """
    d = device_arg.strip().lower()
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device_arg)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("[Device] CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return dev


def _print_device_info(device: torch.device):
    print("──────────────── Device Info ────────────────")
    print(f"[Device] Using device: {device}")
    print(f"[Device] torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"[Device] torch.version.cuda: {torch.version.cuda}")
    print(f"[Device] cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None}")
    if device.type == "cuda":
        # lock to the chosen GPU
        idx = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_device(idx)
        print(f"[Device] CUDA device count : {torch.cuda.device_count()}")
        print(f"[Device] Current device    : {torch.cuda.current_device()}")
        print(f"[Device] Name[{idx}]       : {torch.cuda.get_device_name(idx)}")
        print(f"[Device] Capability[{idx}] : {torch.cuda.get_device_capability(idx)}")
    print("─────────────────────────────────────────────")


# ===== helper to compute the pre-fc embedding exactly like the model does =====
@torch.no_grad()
def compute_gcnsa_pre_fc_embedding(model: GCNSA, features: torch.Tensor, adj: torch.Tensor, K: int) -> torch.Tensor:
    """
    Returns the pre-fc embedding used by your current GCNSA:
        z3 = cat([naz, aaz])  # shape [N, 2*nhid]
    where:
        y1 = relu(fc0(X)); y = dropout(y1)
        x  = modifiedt(y)
        z  = cat([x, y])             # not used downstream in your shrink
        naz, _ = structurel(X)       # optimized features
        aaz = repeated spmm over adj starting from z (but you kept only 'aaz' in the concat)
    """
    model.eval()
    X = features

    # Frontend (same as forward)
    y1 = F.relu(model.fc0(X))
    y  = F.dropout(y1, model.dropout1, training=False)
    x  = model.modifiedt(y)
    z  = torch.cat([x, y], dim=1)          # 2p

    naz, _ = model.structurel(X)           # [N, nhid]

    # K-hop aggregations
    az  = z
    aaz = z
    for _ in range(K - 1):
        az  = torch.spmm(adj, az)
        aaz = torch.spmm(adj, az)

    # Your current pre-fc embedding
    z3 = torch.cat([naz, aaz], dim=1)      # [N, 2*nhid]
    z3 = F.dropout(z3, model.dropout1, training=False)
    return z3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='dataset name (e.g., Tox21)')
    parser.add_argument('--trainsplit', type=float, default=0.6, help='fraction of nodes for training')
    parser.add_argument('--tvt_index_path', default=None, help='path to tvt index .npy')
    parser.add_argument('--K', type=int, default=3, help='Feature aggregation hops')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.03, help='Learning rate')
    parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--hd', type=int, default=48, help='Hidden units')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--epsilon', type=float, default=0.9, help='Similarity threshold')
    parser.add_argument('--r', type=int, default=3, help='Minimum new neighbors')
    parser.add_argument('--patience', type=int, default=400, help='Early stopping patience')
    parser.add_argument('--emb_dir', type=str, default='emb', help='Directory to save per-split embeddings NPZ')

    # ===== NEW: device selector =====
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help="Compute device: 'auto' (default), 'cpu', 'cuda', or 'cuda:N' (e.g., 'cuda:1')"
    )

    args = parser.parse_args()

    # Resolve and report device
    device = _resolve_device(args.device)
    _print_device_info(device)

    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    # Default tvt path
    if args.tvt_index_path is None:
        split_str = str(args.trainsplit)
        base = os.path.join('tvtsplit', args.dataset, split_str)
        filename = f"{args.dataset}_tvt_index.npy"
        args.tvt_index_path = os.path.join(base, filename)

    # Load data
    if args.dataset in {'SIDER', 'ClinTox', 'Tox21', 'Toxcast'}:
        adj, features, labels = utils_data.load_data(args.dataset, args.trainsplit)
    else:
        raise ValueError("Update dataset switch if you need additional names.")

    print(f"[Data] labels shape: {labels.shape}")
    for nid in range(min(5, labels.size(0))):
        print(f"   node {nid} -> {labels[nid].cpu().numpy()}")

    # TVT split
    idx_train, idx_val, idx_test = acquiretvt(args.dataset, args.trainsplit, labels, args.tvt_index_path)

    # Move to device
    features, adj, labels = features.to(device), adj.to(device), labels.to(device)
    idx_train, idx_val, idx_test = idx_train.to(device), idx_val.to(device), idx_test.to(device)

    # Build model (TRANSDUCTIVE: n_samples = total N, and we feed full features/adj each pass)
    model = GCNSA(
        nfeat=features.shape[1], nhid=args.hd,
        nclass=labels.size(1), dropout1=args.dropout,
        epsilon=args.epsilon, n_samples=adj.size(0), r=args.r
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    approx_model_mb = trainable_params * 4 / (1024 ** 2)
    print("────────────────────────────────────────────────────────")
    print("[Model] GCNSA created")
    print(f"[Model] Total parameters     : {total_params:,}")
    print(f"[Model] Trainable parameters : {trainable_params:,}")
    print(f"[Model] ~Trainable size      : {approx_model_mb:.2f} MB (float32)")
    print("────────────────────────────────────────────────────────")

    def _to_float(x):
        return x.item() if hasattr(x, "item") else float(x)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.BCEWithLogitsLoss()

    def train(epoch):
        model.train()
        optimizer.zero_grad()
        
        # Perform a forward pass
        logits, _ = model(features, adj, args.K)  # full graph forward (transductive)
        
        # Select the training subset
        train_logits = logits[idx_train]
        train_labels = labels[idx_train]

        # --- Add checks for shapes and values of tensors ---
        print(f"Train logits shape: {train_logits.shape}")
        print(f"Train labels shape: {train_labels.shape}")
        
        # Check unique values in train labels to ensure they are binary (0 or 1)
        print(train_labels)
        
        # Check for NaNs or Infs in the tensors
        if torch.any(torch.isnan(train_logits)) or torch.any(torch.isinf(train_logits)):
            print("NaNs or Infs detected in train_logits")
        if torch.any(torch.isnan(train_labels)) or torch.any(torch.isinf(train_labels)):
            print("NaNs or Infs detected in train_labels")

        # Compute the loss
        loss_train = criterion(train_logits, train_labels)
        
        # Compute other metrics (AUC, Accuracy, etc.)
        auc_train  = auc(train_logits, train_labels)
        acc_train  = accuracy(train_logits, train_labels)
        pre_train  = precision(train_logits, train_labels)
        rec_train  = recall(train_logits, train_labels)
        f1_train   = f1(train_logits, train_labels)

        # Backpropagation
        loss_train.backward()
        optimizer.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_logits, _ = model(features, adj, args.K)
            val_logits = val_logits[idx_val]
            val_labels = labels[idx_val]

            loss_val = criterion(val_logits, val_labels)
            auc_val  = auc(val_logits, val_labels)
            acc_val  = accuracy(val_logits, val_labels)
            pre_val  = precision(val_logits, val_labels)
            rec_val  = recall(val_logits, val_labels)
            f1_val   = f1(val_logits, val_labels)

        # Print training and validation results at intervals
        if epoch % 10 == 0:
            print(f"Epoch {epoch:04d} | "
                f"Train: Loss {loss_train.item():.4f} | AUC {_to_float(auc_train):.4f} | "
                f"ACC {acc_train:.4f} | P {pre_train:.4f} | R {rec_train:.4f} | F1 {f1_train:.4f}")
            print(f"             | "
                f"Val  : Loss {loss_val.item():.4f} | AUC {_to_float(auc_val):.4f} | "
                f"ACC {acc_val:.4f} | P {pre_val:.4f} | R {rec_val:.4f} | F1 {f1_val:.4f}")

        return loss_val.item(), _to_float(auc_val)


    @torch.no_grad()
    def test_and_export(ckpt_path: str):
        # Load best checkpoint and evaluate (as before)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        output, learned_adj = model(features, adj, args.K)
        train_logits = output[idx_train]; train_labels = labels[idx_train]
        val_logits   = output[idx_val];   val_labels   = labels[idx_val]
        test_logits  = output[idx_test];  test_labels  = labels[idx_test]

        print("✅ Final learned_adj stats:")
        print("  min:", learned_adj.min().item())
        print("  max:", learned_adj.max().item())
        print("  mean:", learned_adj.mean().item())

        print(f"Train Acc: {accuracy(train_logits, train_labels):.4f} | "
              f"Val Acc: {accuracy(val_logits, val_labels):.4f} | "
              f"Test Acc: {accuracy(test_logits, test_labels):.4f}")

        print(f"Train AUC: {auc(train_logits, train_labels):.4f} | "
              f"Val AUC: {auc(val_logits, val_labels):.4f} | "
              f"Test AUC: {auc(test_logits, test_labels):.4f}")

        # ===== compute the rich pre-fc embedding on the FULL graph (transductive) =====
        Z = compute_gcnsa_pre_fc_embedding(model, features, adj, args.K)  # [N, 2*nhid]

        # Prepare save dir
        out_dir = os.path.join(args.emb_dir, args.dataset)
        os.makedirs(out_dir, exist_ok=True)

        # Helper to save a split
        def save_split(name: str, idx_tensor: torch.Tensor):
            idx_np = idx_tensor.detach().cpu().numpy().astype(np.int64)
            emb_np = Z[idx_tensor].detach().cpu().numpy()
            out_path = os.path.join(out_dir, f"avgcnsa_{name}.npz")
            np.savez(out_path, mol_id=idx_np, emb=emb_np)
            print(f"💾 Saved {name} embeddings to {out_path} | shape: {emb_np.shape}")

        save_split("train", idx_train)
        save_split("val",   idx_val)
        save_split("test",  idx_test)

    # ===== training loop with early stop on Val AUC =====
    t0 = time.time()
    best_auc = float("-inf")
    bad_counter = 0
    ckpt_path = 'best_model_auc.pth'

    for epoch in range(args.epochs):
        val_loss, val_auc = train(epoch)
        if val_auc > best_auc:
            best_auc = val_auc
            bad_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"✅ New best Val AUC: {best_auc:.4f} (epoch {epoch}) — saved {ckpt_path}")
        else:
            bad_counter += 1
            if bad_counter == args.patience:
                print(f"⏹️ Early stopping at epoch {epoch} (no AUC improvement for {args.patience} epochs)")
                break

    print(f"Optimization finished in {time.time() - t0:.2f}s")
    # evaluate + EXPORT per-split embeddings
    test_and_export(ckpt_path)


if __name__ == '__main__':
    main()

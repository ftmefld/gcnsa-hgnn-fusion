from __future__ import division
from __future__ import print_function
import time
import argparse
import numpy as np
import glob
import torch
import torch.optim as optim
import torch.nn as nn

from sample import Sampler
from utils import accuracy, precision, recall, f1, auc, acquiretvt
from models import GCNSA
import utils_data
import os

import torch, os
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)
print("torch.backends.cudnn.version():", torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None)
if torch.cuda.is_available():
    print("device count:", torch.cuda.device_count())
    print("current device:", torch.cuda.current_device())
    print("name[0]:", torch.cuda.get_device_name(0))
    print("capabilities:", torch.cuda.get_device_capability(0))


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='dataset name (e.g., cora)')
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
    args = parser.parse_args()

    # Default tvt path
    if args.tvt_index_path is None:
        split_str = str(args.trainsplit)
        base = os.path.join('tvtsplit', args.dataset, split_str)
        filename = f"{args.dataset}_tvt_index.npy"
        args.tvt_index_path = os.path.join(base, filename)

    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load data
    if args.dataset in {'SIDER', 'ClinTox', 'Tox21', 'Toxcast'}:
        adj, features, labels = utils_data.load_data(args.dataset, args.trainsplit)

    """
    if args.dataset in {'cora', 'citeseer', 'pubmed'}:
        sampler = Sampler(args.dataset, '../data/', 'full')
        labels = sampler.get_label_and_idxes(True)
        adj, features = sampler.randomedge_sampler(percent=1.0, normalization='AugNormAdj', cuda=True)
    else:
        adj, features, labels = utils_data.load_data(args.dataset, args.trainsplit)
    """
    # Debug print loaded labels
    print(f"[Data] Loaded label tensor of shape: {labels.shape}")
    print("[Data] Sample labels:")
    for nid in range(min(5, labels.size(0))):
        print(f"   node {nid} -> {labels[nid].cpu().numpy()}")

    # TVT split
    idx_train, idx_val, idx_test = acquiretvt(args.dataset, args.trainsplit, labels, args.tvt_index_path)

    # Move to device
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features, adj, labels = features.to(device), adj.to(device), labels.to(device)
    idx_train, idx_val, idx_test = idx_train.to(device), idx_val.to(device), idx_test.to(device)

    # Build model
    model = GCNSA(
        nfeat=features.shape[1], nhid=args.hd,
        nclass=labels.size(1), dropout1=args.dropout,
        epsilon=args.epsilon, n_samples=adj.size(0), r=args.r
    ).to(device)
    
    

    # ---- NEW: show parameter counts ----
    total_params, trainable_params = count_parameters(model)
    bytes_per_param = 4  # float32
    approx_model_mb = trainable_params * bytes_per_param / (1024 ** 2)
    print("────────────────────────────────────────────────────────")
    print("[Model] GCNSA created")
    print(f"[Model] Total parameters     : {total_params:,}")
    print(f"[Model] Trainable parameters : {trainable_params:,}")
    print(f"[Model] ~Trainable size      : {approx_model_mb:.2f} MB (float32)")
    print("────────────────────────────────────────────────────────")
    # -----------------------------------
    
    # --- helper to safely get a python float from tensor/number ---
    def _to_float(x):
        return x.item() if hasattr(x, "item") else float(x)


    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    #weight=torch.Tensor([1, 10]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    #criterion = nn.CrossEntropyLoss(reduction="mean")

    def train(epoch):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(features, adj, args.K)
        train_logits = logits[idx_train]
        train_labels = labels[idx_train]

        # ----- Train loss + metrics -----
        loss_train = criterion(train_logits, train_labels)
        auc_train  = auc(train_logits, train_labels)
        acc_train  = accuracy(train_logits, train_labels)
        pre_train  = precision(train_logits, train_labels)
        rec_train  = recall(train_logits, train_labels)
        f1_train   = f1(train_logits, train_labels)

        loss_train.backward()
        optimizer.step()

        # ----- Validation loss + metrics -----
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

        # ----- Logging -----
        if epoch % 10 == 0:
            print(f"Epoch {epoch:04d} | "
                  f"Train: Loss {loss_train.item():.4f} | AUC {_to_float(auc_train):.4f} | "
                  f"ACC {acc_train:.4f} | P {pre_train:.4f} | R {rec_train:.4f} | F1 {f1_train:.4f}")
            print(f"             | "
                  f"Val  : Loss {loss_val.item():.4f} | AUC {_to_float(auc_val):.4f} | "
                  f"ACC {acc_val:.4f} | P {pre_val:.4f} | R {rec_val:.4f} | F1 {f1_val:.4f}")

            # Per-label breakdown (train + val)
            num_labels = train_labels.size(1)
            for i in range(num_labels):
                # Train (per label)
                li_tr  = criterion(train_logits[:, i], train_labels[:, i])
                ai_tr  = auc(train_logits[:, i], train_labels[:, i])
                acc_tr = accuracy(train_logits[:, i], train_labels[:, i])
                pre_tr = precision(train_logits[:, i], train_labels[:, i])
                rec_tr = recall(train_logits[:, i], train_labels[:, i])
                f1_tr  = f1(train_logits[:, i], train_labels[:, i])

                # Val (per label)
                li_va  = criterion(val_logits[:, i], val_labels[:, i])
                ai_va  = auc(val_logits[:, i], val_labels[:, i])
                acc_va = accuracy(val_logits[:, i], val_labels[:, i])
                pre_va = precision(val_logits[:, i], val_labels[:, i])
                rec_va = recall(val_logits[:, i], val_labels[:, i])
                f1_va  = f1(val_logits[:, i], val_labels[:, i])

                print(f"   Label {i:2d}: "
                    f"Train — Loss {li_tr.item():.4f} | AUC {ai_tr:.4f} | "
                    f"ACC {acc_tr:.4f} | P {pre_tr:.4f} | R {rec_tr:.4f} | F1 {f1_tr:.4f}  ||  "
                    f"Val — Loss {li_va.item():.4f} | AUC {ai_va:.4f} | "
                    f"ACC {acc_va:.4f} | P {pre_va:.4f} | R {rec_va:.4f} | F1 {f1_va:.4f}")

        return loss_val.item(), _to_float(auc_val)


    def test():
        model.eval()
        with torch.no_grad():
            output, learned_adj = model(features, adj, args.K)
            train_logits = output[idx_train]
            train_labels = labels[idx_train]
            val_logits = output[idx_val]
            val_labels = labels[idx_val]
            test_logits = output[idx_test]
            test_labels = labels[idx_test]
            # Save learned adjacency as a PyTorch tensor file
            torch.save(learned_adj.cpu(), 'learned_adjacency_clintox1.pt')
            # After saving
            print("✅ Final learned_adj stats:")
            print("  min:", learned_adj.min().item())
            print("  max:", learned_adj.max().item())
            print("  mean:", learned_adj.mean().item())
            print("  is identity:", torch.allclose(learned_adj, torch.eye(learned_adj.size(0), device=learned_adj.device)))

            acc_train = accuracy(train_logits, train_labels)
            acc_val = accuracy(val_logits, val_labels)
            acc_test = accuracy(test_logits, test_labels)

            precision_train = precision(train_logits, train_labels)
            precision_val = precision(val_logits, val_labels)
            precision_test = precision(test_logits, test_labels)

            recall_train = recall(train_logits, train_labels)
            recall_val = recall(val_logits, val_labels)
            recall_test = recall(test_logits, test_labels)

            f1_train = f1(train_logits, train_labels)
            f1_val = f1(val_logits, val_labels)
            f1_test = f1(test_logits, test_labels)

            auc_train = auc(train_logits, train_labels)
            auc_val = auc(val_logits, val_labels)
            auc_test = auc(test_logits, test_labels)

            print(f"Train Acc: {acc_train:.4f} | Val Acc: {acc_val:.4f} | Test Acc: {acc_test:.4f}")
            print(f"Train Precision: {precision_train:.4f} | Val Precision: {precision_val:.4f} | Test Precision: {precision_test:.4f}")
            print(f"Train Recall: {recall_train:.4f} | Val Recall: {recall_val:.4f} | Test Recall: {recall_test:.4f}")
            print(f"Train F1: {f1_train:.4f} | Val F1: {f1_val:.4f} | Test F1: {f1_test:.4f}")
            print(f"Train AUC: {auc_train:.4f} | Val AUC: {auc_val:.4f} | Test AUC: {auc_test:.4f}")

        return auc_test.item()

    # Main loop
    t0 = time.time()
    best_auc = float("-inf")
    bad_counter = 0
    ckpt_path = 'best_model_auc.pth'

    for epoch in range(args.epochs):
        val_loss, val_auc = train(epoch)   # we now receive val AUC
        if val_auc > best_auc:             # maximize AUC
            best_auc = val_auc
            bad_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            # optional: print a small notice when we hit a new best
            print(f"✅ New best Val AUC: {best_auc:.4f} (epoch {epoch}) — model saved to {ckpt_path}")
        else:
            bad_counter += 1
            if bad_counter == args.patience:
                print(f"⏹️ Early stopping at epoch {epoch} (no AUC improvement for {args.patience} epochs)")
                break

    print(f"Optimization finished in {time.time() - t0:.2f}s")
    model.load_state_dict(torch.load(ckpt_path))
    test()

if __name__ == '__main__':
    main()
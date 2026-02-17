import os
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

from fhgnn import FHGNN
from splitters import scaffold_split
from loader import HiMolGraph, MoleculeDataset

criterion = nn.BCEWithLogitsLoss(reduction="none")


def train(model, device, loader, optimizer):
    model.train()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        is_valid = y**2 > 0
        loss_mat = criterion(pred.double(), (y + 1) / 2)
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros_like(loss_mat))
        loss = torch.sum(loss_mat) / torch.sum(is_valid)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def train_reg(args, model, device, loader, optimizer):
    model.train()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch)
        y = batch.y.view(pred.shape).to(torch.float64)
        loss = torch.sum((pred - y) ** 2) / y.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


@torch.no_grad()
def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch)
        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    # compute masked BCE on the last batch shapes (consistent with your original code)
    y = torch.tensor(y_true, dtype=torch.float64)
    p = torch.tensor(y_scores, dtype=torch.float64)
    is_valid = y**2 > 0
    loss_mat = criterion(p.double(), (y + 1) / 2)
    loss_mat = torch.where(is_valid, loss_mat, torch.zeros_like(loss_mat))
    loss = torch.sum(loss_mat) / torch.sum(is_valid)

    roc_list = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid_i = y_true[:, i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid_i, i] + 1) / 2, y_scores[is_valid_i, i]))
    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" % (1 - float(len(roc_list)) / y_true.shape[1]))
    eval_roc = sum(roc_list) / len(roc_list) if len(roc_list) > 0 else float("nan")
    return eval_roc, loss


@torch.no_grad()
def eval_reg(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch)
        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy().flatten()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy().flatten()

    mse = mean_squared_error(y_true, y_scores)
    mae = mean_absolute_error(y_true, y_scores)
    rmse = np.sqrt(mean_squared_error(y_true, y_scores))
    return mse, mae, rmse


# ===== NEW: embedding extraction via forward hook =====
@torch.no_grad()
def extract_embeddings(model, device, loader, num_tasks):
    model.eval()
    ids, embs = [], []
    for batch in loader:
        batch = batch.to(device)
        logits, emb = model(batch, return_embeddings=True)  # <-- use the flag
        ids.append(batch.id.detach().cpu())
        embs.append(emb.detach().cpu())
    ids = torch.cat(ids).numpy().astype(np.int64)
    emb = torch.cat(embs).numpy().astype(np.float32)
    return ids, emb



def main():
    parser = argparse.ArgumentParser(description='PyTorch implementation of FH-GNN')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for the prediction layer')
    parser.add_argument('--dataset', type=str, default='bbbp',
                        help='[bbbp, bace, sider, clintox,tox21, toxcast, esol,freesolv,lipophilicity]')
    parser.add_argument('--data_dir', type=str, default='./dataset/', help="the path of input CSV file")
    parser.add_argument('--save_dir', type=str, default='./model_checkpoints', help="the path to save output model")
    parser.add_argument('--emb_dir', type=str, default='./emb', help="directory to save extracted embeddings")
    parser.add_argument('--depth', type=int, default=7, help="the depth of molecule encoder")
    parser.add_argument('--seed', type=int, default=42, help="seed for splitting the dataset")
    parser.add_argument('--runseed', type=int, default=42, help="seed for minibatch selection, random initialization")
    parser.add_argument('--eval_train', type=int, default=1, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataset loading')
    args = parser.parse_args()

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    if args.dataset in ['tox21', 'bace', 'bbbp', 'sider', 'clintox', 'toxcast']:
        task_type = 'cls'
    else:
        task_type = 'reg'

    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "toxcast":
        num_tasks=617
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    elif args.dataset in ['esol', 'freesolv', 'lipophilicity']:
        num_tasks = 1
    else:
        raise ValueError("Invalid dataset name.")

    # set up dataset
    print('process data')
    dataset = MoleculeDataset(os.path.join(args.data_dir, args.dataset), dataset=args.dataset)

    print("scaffold")
    smiles_list = pd.read_csv(os.path.join(args.data_dir, args.dataset, './/processed//smiles.csv'),
                              header=None)[0].tolist()
    train_dataset, valid_dataset, test_dataset = scaffold_split(
        dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader   = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = FHGNN(
        data_name=args.dataset, atom_fdim=89, bond_fdim=98,
        hidden_size=512, depth=args.depth, device=device, out_dim=num_tasks,
    ).to(device)

    # optimizer
    optimizer = optim.Adam([{"params": model.parameters(), "lr": args.lr}])
    print(optimizer)

    os.makedirs(args.save_dir, exist_ok=True)
    model_ckpt = os.path.join(args.save_dir, args.dataset + '.pth')

    # training
    if task_type == 'cls':
        best_auc = -1.0
        for epoch in range(1, args.epochs + 1):
            print('====epoch:', epoch)
            train(model, device, train_loader, optimizer)

            print('====Evaluation')
            if args.eval_train:
                train_auc, train_loss = eval(args, model, device, train_loader)
            else:
                print('omit the training accuracy computation')
                train_auc = 0
            val_auc, val_loss = eval(args, model, device, val_loader)
            test_auc, test_loss = eval(args, model, device, test_loader)

            if best_auc < val_auc:
                best_auc = val_auc
                best_epoch = epoch
                torch.save(model.state_dict(), model_ckpt)

            print(f"train_auc: {train_auc:.6f}  val_auc: {val_auc:.6f}  test_auc: {test_auc:.6f}")

        # final eval @ best
        print("\n==== Final evaluation (best checkpoint) ====")
        print(f"Best epoch (by val AUC): {best_epoch}, best val AUC: {best_auc:.6f}")
        state = torch.load(model_ckpt, map_location=device)
        model.load_state_dict(state)
        final_train_auc, final_train_loss = eval(args, model, device, train_loader)
        final_val_auc, final_val_loss     = eval(args, model, device, val_loader)
        final_test_auc, final_test_loss   = eval(args, model, device, test_loader)
        print(f"FINAL @ epoch {best_epoch} --> "
              f"Train AUC: {final_train_auc:.6f} (loss={final_train_loss:.6f}) | "
              f"Val AUC: {final_val_auc:.6f} (loss={final_val_loss:.6f}) | "
              f"Test AUC: {final_test_auc:.6f} (loss={final_test_loss:.6f})")

    else:  # regression
        best_rmse = float("inf")
        for epoch in range(1, args.epochs + 1):
            print('====epoch:', epoch)
            train_reg(args, model, device, train_loader, optimizer)

            print('====Evaluation')
            if args.eval_train:
                train_mse, train_mae, train_rmse = eval_reg(args, model, device, train_loader)
            else:
                print('omit the training accuracy computation')
                train_mse, train_mae, train_rmse = 0, 0, 0
            val_mse, val_mae, val_rmse = eval_reg(args, model, device, val_loader)
            test_mse, test_mae, test_rmse = eval_reg(args, model, device, test_loader)

            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_epoch = epoch
                torch.save(model.state_dict(), model_ckpt)

            print(f"train_mse: {train_mse:.6f}  val_mse: {val_mse:.6f}  test_mse: {test_mse:.6f}")
            print(f"train_mae: {train_mae:.6f}  val_mae: {val_mae:.6f}  test_mae: {test_mae:.6f}")
            print(f"train_rmse: {train_rmse:.6f}  val_rmse: {val_rmse:.6f}  test_rmse: {test_rmse:.6f}")

        # final eval @ best
        print("\n==== Final evaluation (best checkpoint) ====")
        print(f"Best epoch (by val RMSE): {best_epoch}, best val RMSE: {best_rmse:.6f}")
        state = torch.load(model_ckpt, map_location=device)
        model.load_state_dict(state)
        tr_mse, tr_mae, tr_rmse = eval_reg(args, model, device, train_loader)
        va_mse, va_mae, va_rmse = eval_reg(args, model, device, val_loader)
        te_mse, te_mae, te_rmse = eval_reg(args, model, device, test_loader)
        print(f"FINAL @ epoch {best_epoch} --> "
              f"Train RMSE: {tr_rmse:.6f} | Val RMSE: {va_rmse:.6f} | Test RMSE: {te_rmse:.6f}")

    # ======= NEW: Export embeddings from the best checkpoint =======
    print("\n==== Exporting FHGNN embeddings from best checkpoint ====")
    state = torch.load(model_ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    os.makedirs(os.path.join(args.emb_dir, args.dataset), exist_ok=True)

    def save_split(name, loader):
        ids, emb = extract_embeddings(model, device, loader, num_tasks=num_tasks)
        out_path = os.path.join(args.emb_dir, args.dataset, f"fhgnn_{name}.npz")
        np.savez(out_path, mol_id=ids, emb=emb)
        print(f"Saved {name} embeddings: {emb.shape} -> {out_path}")

    save_split("train", train_loader)
    save_split("val",   val_loader)
    save_split("test",  test_loader)


if __name__ == "__main__":
    main()



"""
# BACE
python train.py --dataset bace --epochs 100 --lr 0.0001 --batch_size 128 --depth 5

# BBBP
python train.py --dataset bbbp --epochs 100 --lr 0.0001 --batch_size 64  --depth 7

# Tox21
python train.py --dataset tox21 --epochs 100 --lr 0.001  --batch_size 512 --depth 5

# Toxcast
python train.py --dataset toxcast --epochs 100 --lr 0.001  --batch_size 512 --depth 5


# SIDER
python train.py --dataset sider --epochs 100 --lr 0.0001 --batch_size 32  --depth 7

# ClinTox
python train.py --dataset clintox --epochs 100 --lr 0.0001 --batch_size 128 --depth 5

# ESOL
python train.py --dataset esol --epochs 100 --lr 0.0005 --batch_size 32  --depth 5

# FreeSolv
python train.py --dataset freesolv --epochs 100 --lr 0.0001 --batch_size 32  --depth 5

# Lipophilicity
python train.py --dataset lipophilicity --epochs 100 --lr 0.0005 --batch_size 128 --depth 5

"""
import torch
import numpy as np
from itertools import compress
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import os
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
# from torch_geometric.data import DataLoader  # old
from torch_geometric.loader import DataLoader  # new

from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

from fhgnn import FHGNN
#from splitters import scaffold_split
from loader import HiMolGraph, MoleculeDataset

def generate_scaffold(smiles, include_chirality=False):
    return MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality
    )

def scaffold_split(
    dataset, smiles_list, task_idx=None, null_value=0,
    frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=42,
    include_chirality=True, return_indices=True
):
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    # Filter by nulls (if requested) and keep original indices
    if task_idx is not None:
        y_task = np.array([data.y[task_idx].item() for data in dataset])
        non_null = y_task != null_value
    else:
        non_null = np.ones(len(dataset), dtype=bool)

    # (orig_index, smiles) pairs for eligible samples
    eligible = list(compress(enumerate(smiles_list), non_null))
    eligible_indices = [i for i, _ in eligible]

    rng = np.random.RandomState(seed)

    # Group eligible samples by scaffold
    scaffolds = defaultdict(list)
    for ind, smiles in eligible:
        scaf = generate_scaffold(smiles, include_chirality=include_chirality)
        scaffolds[scaf].append(ind)

    scaffold_sets = list(scaffolds.values())
    rng.shuffle(scaffold_sets)

    # Compute target sizes based on eligible count
    n_eligible = len(eligible_indices)
    n_valid = int(np.floor(frac_valid * n_eligible))
    n_test  = int(np.floor(frac_test  * n_eligible))

    train_idx, valid_idx, test_idx = [], [], []
    for sset in scaffold_sets:
        if len(valid_idx) + len(sset) <= n_valid:
            valid_idx.extend(sset)
        elif len(test_idx) + len(sset) <= n_test:
            test_idx.extend(sset)
        else:
            train_idx.extend(sset)

    # Safety checks (optional)
    if len(valid_idx) == 0 or len(test_idx) == 0 or len(train_idx) == 0:
        raise RuntimeError(
            f"Empty split: train={len(train_idx)}, valid={len(valid_idx)}, test={len(test_idx)}. "
            "Consider changing seed or fractions."
        )

    # Slice datasets
    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(valid_idx)]
    test_dataset  = dataset[torch.tensor(test_idx)]

    if return_indices:
        return (train_dataset, valid_dataset, test_dataset,
                train_idx, valid_idx, test_idx)
    else:
        return train_dataset, valid_dataset, test_dataset, train_idx, valid_idx, test_idx
    

dataset = MoleculeDataset(os.path.join(".//dataset", "toxcast"), dataset="toxcast")

print("scaffold")
smiles_list = pd.read_csv(os.path.join(".//dataset", "toxcast", './/processed//smiles.csv'), header=None)[0].tolist()

def save_indices_csv(train_idx, val_idx, test_idx, prefix=""):
    paths = {
        "train": f"{prefix}train_idx.txt",
        "val":   f"{prefix}val_idx.txt",
        "test":  f"{prefix}test_idx.txt",
    }
    for name, arr in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        with open(paths[name], "w") as f:
            f.write(",".join(map(str, arr)) + "\n")  # comma-separated, single line
    return paths


# example use
# train_ds, val_ds, test_ds, train_idx, val_idx, test_idx = scaffold_split(...)



train_ds, val_ds, test_ds, train_idx, val_idx, test_idx = scaffold_split(dataset, smiles_list, null_value=0, 
                                                                frac_train=0.8,frac_valid=0.1, 
                                                                frac_test=0.1,seed = 42)

save_indices_csv(train_idx, val_idx, test_idx)

import numpy as np
import torch as th
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from sklearn.metrics import precision_score, recall_score, f1_score

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

def _to_numpy(x: torch.Tensor):
    return x.detach().cpu().numpy()

def _as_1d(a):
    """Squeeze a possible (N,1) array to (N,)."""
    return a.squeeze(axis=1) if (a.ndim == 2 and a.shape[1] == 1) else a

def _sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))

def _maybe_probs(x_np: np.ndarray):
    """If not in [0,1], treat as logits and apply sigmoid."""
    x_min, x_max = np.min(x_np), np.max(x_np)
    if x_min < 0.0 or x_max > 1.0:
        return _sigmoid_np(x_np)
    return x_np

# -------------------- Precision / Recall / F1 --------------------

def precision(output: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5):
    """
    Precision for:
      - multiclass (output: N x C, labels: N)  -> macro-average
      - binary (output: N or N x 1, labels: N) -> scalar
      - multilabel (output: N x L, labels: N x L) -> macro-average over labels
    """
    y_true = _to_numpy(labels)
    y_pred_src = _to_numpy(output)

    # Multiclass single-label
    if y_pred_src.ndim == 2 and y_pred_src.shape[1] > 1 and y_true.ndim == 1:
        preds = np.argmax(y_pred_src, axis=1)
        return precision_score(y_true, preds, average='macro', zero_division=0)

    # Binary or Multilabel: threshold after sigmoid (if needed)
    probs = _maybe_probs(y_pred_src)
    preds = (probs >= threshold).astype(int)

    # Binary single-label → 1D
    if y_true.ndim == 1 and preds.ndim == 2 and preds.shape[1] == 1:
        preds = _as_1d(preds)

    # Multilabel macro average or binary scalar
    avg = 'macro' if y_true.ndim == 2 else 'binary'
    return precision_score(y_true, preds, average=avg, zero_division=0)

def recall(output: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5):
    y_true = _to_numpy(labels)
    y_pred_src = _to_numpy(output)

    if y_pred_src.ndim == 2 and y_pred_src.shape[1] > 1 and y_true.ndim == 1:
        preds = np.argmax(y_pred_src, axis=1)
        return recall_score(y_true, preds, average='macro', zero_division=0)

    probs = _maybe_probs(y_pred_src)
    preds = (probs >= threshold).astype(int)
    if y_true.ndim == 1 and preds.ndim == 2 and preds.shape[1] == 1:
        preds = _as_1d(preds)

    avg = 'macro' if y_true.ndim == 2 else 'binary'
    return recall_score(y_true, preds, average=avg, zero_division=0)

def f1(output: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5):
    y_true = _to_numpy(labels)
    y_pred_src = _to_numpy(output)

    if y_pred_src.ndim == 2 and y_pred_src.shape[1] > 1 and y_true.ndim == 1:
        preds = np.argmax(y_pred_src, axis=1)
        return f1_score(y_true, preds, average='macro', zero_division=0)

    probs = _maybe_probs(y_pred_src)
    preds = (probs >= threshold).astype(int)
    if y_true.ndim == 1 and preds.ndim == 2 and preds.shape[1] == 1:
        preds = _as_1d(preds)

    avg = 'macro' if y_true.ndim == 2 else 'binary'
    return f1_score(y_true, preds, average=avg, zero_division=0)

# -------------------- Accuracy --------------------

def accuracy(output: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5):
    """
    Accuracy:
      - Multiclass (N x C, N): standard accuracy.
      - Binary (N) or (N x 1): standard accuracy after threshold.
      - Multilabel (N x L): Hamming accuracy (#correct labels / total labels).
    """
    y_true = _to_numpy(labels)
    y_pred_src = _to_numpy(output)

    # Multiclass single-label
    if y_pred_src.ndim == 2 and y_pred_src.shape[1] > 1 and y_true.ndim == 1:
        preds = np.argmax(y_pred_src, axis=1)
        return float((preds == y_true).mean())

    # Binary or multilabel
    probs = _maybe_probs(y_pred_src)
    preds = (probs >= threshold).astype(int)

    # Binary single-label
    if y_true.ndim == 1:
        if preds.ndim == 2 and preds.shape[1] == 1:
            preds = _as_1d(preds)
        return float((preds == y_true).mean())

    # Multilabel Hamming accuracy
    return float((preds == y_true).mean())



import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score

def auc(output, labels, average='macro'):
    """
    Compute ROC AUC for binary (single-label) or multi-label classification.

    Args:
        output: torch.Tensor
            - shape [n_samples] (logits)    OR
            - shape [n_samples, 2] (logits or probs for binary) OR
            - shape [n_samples, n_labels] (logits for each label)
        labels: torch.Tensor
            - shape [n_samples] (0/1)                 OR
            - shape [n_samples, n_labels] (0/1 multi-hot)
        average: str
            - how to average in the multilabel case; one of {'macro','micro','weighted'}.

    Returns:
        float: ROC AUC score
    """
    # move to numpy
    output_np = output.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    # --- single-label binary case ---
    if labels_np.ndim == 1:
        # if you passed shape [n,2], take the "positive" column
        if output_np.ndim == 2 and output_np.shape[1] == 2:
            # assume raw logits or probs: softmax → prob of class 1
            probs = F.softmax(torch.from_numpy(output_np), dim=1)[:, 1].numpy()
        else:
            # 1D logits: sigmoid → prob
            probs = torch.sigmoid(torch.from_numpy(output_np)).numpy().reshape(-1)
        return roc_auc_score(labels_np, probs)

    # --- multi-label case (labels_np.ndim == 2) ---
    # apply independent sigmoid to each column of logits
    probs = torch.sigmoid(torch.from_numpy(output_np)).numpy()
    # use sklearn's multilabel ROC AUC (one-vs-rest) with the given averaging
    return roc_auc_score(labels_np, probs, average=average)



import numpy as np
import torch as th

def acquiretvt(dataset, trainsplit, labels, tvt_index_path):
    """
    Single hold-out split using a precomputed tvt_index file.

    Behavior change:
      - If N * trainsplit is not an integer (e.g., 1106.3, 1106.5, 1106.7),
        we round it UP using ceil -> 1107.

    Args:
        dataset: (ignored) name, kept for API compatibility
        trainsplit (float): fraction of samples for training
        labels (Tensor): full labels tensor, used for size check
        tvt_index_path (str): path to .npy file containing a permutation of all indices
    Returns:
        idx_train, idx_val, idx_test (LongTensor)
    """
    # Load the index permutation
    all_idx = np.load(tvt_index_path)
    all_idx = th.tensor(all_idx, dtype=th.long)
    N = all_idx.size(0)

    # Compute train split (ceil on non-integer counts)
    eps = 0.01 # small epsilon to prevent edge cases in rounding
    n_train = int(np.ceil(N * float(trainsplit) + eps))

    if n_train < 0 or n_train > N:
        raise ValueError(f"Invalid train split: {n_train} out of {N}")

    # Remaining examples after train
    rem = N - n_train
    
    # Split remaining data equally into val and test
    n_val = n_test = int(np.floor(rem * 0.5))  # Ensure equal sizes for val and test
    
    # Sanity checks
    if n_train + n_val + n_test != N:
        raise RuntimeError(f"Splits do not sum to N: {n_train}+{n_val}+{n_test}!={N}")

    # Slice indices
    idx_train = all_idx[:n_train]
    idx_val   = all_idx[n_train:n_train + n_val]
    idx_test  = all_idx[n_train + n_val:]

    return idx_train, idx_val, idx_test





def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.tensor(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64),
        dtype=th.long
    )
    values = th.tensor(sparse_mx.data, dtype=th.float32)
    shape = sparse_mx.shape
    return th.sparse.FloatTensor(indices, values, th.Size(shape))

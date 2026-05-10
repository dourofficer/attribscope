"""
classifier.py

A self-contained, embedding-agnostic binary classifier with train and inference utilities.
Follows HaloScope conventions (raw logit output, BCE loss, SGD + cosine LR decay)
but is fully transferable to any binary classification problem on dense embeddings.
"""

import copy
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

def seed_everything(seed: int = 100):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def downsample(X: torch.Tensor, y: torch.Tensor, weight: int = 1):
    """
    Undersample label 0 so that n(label 0) / n(label 1) = weight.
    All label 1 samples are kept.
    """
    idx_pos = (y == 1).nonzero(as_tuple=True)[0]
    idx_neg = (y == 0).nonzero(as_tuple=True)[0]

    n_neg_keep = min(len(idx_pos) * weight, len(idx_neg))
    idx_neg_sampled = idx_neg[torch.randperm(len(idx_neg))[:n_neg_keep]]

    idx_all = torch.cat([idx_pos, idx_neg_sampled])
    idx_all = idx_all[torch.randperm(len(idx_all))]  # shuffle

    return X[idx_all], y[idx_all]


def upsample(X: torch.Tensor, y: torch.Tensor, weight: int = 1):
    """
    Upsample label 1 (with replacement) so that n(label 0) / n(label 1) = weight.
    All label 0 samples are kept. If the current ratio is already <= weight,
    positives are left untouched (no downsampling of label 1).
    """
    idx_pos = (y == 1).nonzero(as_tuple=True)[0]
    idx_neg = (y == 0).nonzero(as_tuple=True)[0]

    n_pos_target = max(len(idx_neg) // weight, len(idx_pos))
    idx_pos_sampled = idx_pos[torch.randint(0, len(idx_pos), (n_pos_target,))]

    idx_all = torch.cat([idx_pos_sampled, idx_neg])
    idx_all = idx_all[torch.randperm(len(idx_all))]  # shuffle

    return X[idx_all], y[idx_all]

def key_hidden(s):
    if s == 'embed': return (-1, 0, '')
    match = re.search(r'(\d+)', s)
    num = int(match.group(1))
    return (0, num, s)

def key_grads(s):
    if s == 'embed': return ['', -1]
    parts = re.split(r'(\d+)', s)
    return [int(p) if p.isdigit() else p for p in parts]

# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class MLPClassifier(nn.Module):
    """
    Two-layer MLP binary classifier.

    Args:
        input_dim:   Dimensionality of input embeddings (d).
        hidden_dim:  Width of the hidden layer. Default 1024 (matches HaloScope).
        seed:        RNG seed for deterministic weight initialisation.

    Forward:
        x (Tensor): shape (N, input_dim)
        Returns raw logit (N, 1) — NOT a probability.
        Apply torch.sigmoid() externally if a probability is needed.
    """

    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 1024, 
        noise: float = 0.01,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.noise = noise

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # for param in self.fc1.parameters():
        #     param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, input_dim)
        Returns:
            logits: (N, 1)  — raw, unbounded scalar per sample.
        """
        if self.training and self.noise is not None:
            assert self.noise >= 0.0, "Noise must be non-negative."
            x = x + self.noise * torch.randn_like(x)
        return self.fc2(F.relu(self.fc1(x)))


class LogisticRegression(nn.Module):
    """
    Linear binary classifier (logistic regression).
    Args:
        input_dim: Dimensionality of input embeddings (d).
    """

    def __init__(self, input_dim: int, noise: float = None):
        super().__init__()
        self.input_dim = input_dim
        self.noise = noise
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, input_dim)
        Returns:
            logits: (N, 1)  — raw, unbounded scalar per sample.
        """
        if self.training and self.noise is not None:
            x = x + self.noise * torch.randn_like(x)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def quick_eval(
    clf: MLPClassifier,
    X: torch.Tensor,
    y: torch.Tensor,
    threshold: float = 0.5,
    device: str = None,
) -> dict:
    if device is None: device = X.device

    clf = clf.to(device)
    clf.eval()
    X = X.float().to(device)
    y = y.float().to(device)

    with torch.no_grad():
        logits = clf(X).view(-1)
        scores = torch.sigmoid(logits)
        preds  = (scores >= threshold).float()

    tp = ((preds == 1) & (y == 1)).sum().item()
    fp = ((preds == 1) & (y == 0)).sum().item()
    fn = ((preds == 0) & (y == 1)).sum().item()

    accuracy  = (preds == y).float().mean().item()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) \
                if (precision + recall) > 0 else 0.0
    auroc = roc_auc_score(y.cpu().numpy(), scores.cpu().numpy())

    return {
        "accuracy": accuracy, 
        "precision": precision, 
        "recall": recall,
        "f1": f1,
        "auroc": auroc
    }

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    epochs: int = 50,
    learning_rate: float = 0.05,
    weight_decay: float = 3e-4,
    momentum: float = 0.9,
    pos_weight: torch.Tensor = None,
    logging_steps: int = 10,
    val_metric: str = None,
    device: str = "cuda",
) -> tuple[MLPClassifier, dict]:

    if pos_weight is not None:
        pos_weight = pos_weight.to(device)

    # clf = MLPClassifier(input_dim, hidden_dim).to(device)
    clf = model.to(device)
    optimizer = torch.optim.SGD(
        clf.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_score: float = -1.0
    best_state: dict | None = None
    best_epoch: int = -1

    for epoch in range(1, epochs + 1):
        clf.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.float().to(device), yb.float().to(device)
            logits = clf(xb).view(-1)
            loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        if val_loader is not None:
            X_val = torch.cat([xb for xb, _ in val_loader]).to(device)
            y_val = torch.cat([yb for _, yb in val_loader]).to(device)
            val_metrics = quick_eval(clf, X_val, y_val, device=device)
            val_score = val_metrics[val_metric]
            if val_score >= best_score and epoch >= 50:
                best_score = val_score
                best_state = copy.deepcopy(clf.state_dict())
                best_epoch = epoch

        if logging_steps is not None and epoch % logging_steps == 0:
            avg_loss = epoch_loss / len(train_loader)
            X_tr = torch.cat([xb for xb, _ in train_loader]).to(device)
            y_tr = torch.cat([yb for _, yb in train_loader]).to(device)
            train_metrics = quick_eval(clf, X_tr, y_tr, device=device)
            log = (
                f"Epoch {epoch}/{epochs} - loss: {avg_loss:.4f}"
                f" | accuracy: {train_metrics['accuracy']:.4f}"
                f" | precision: {train_metrics['precision']:.4f}"
                f" | recall: {train_metrics['recall']:.4f}"
                f" | F1: {train_metrics['f1']:.4f}"
                f" | auroc: {train_metrics['auroc']:.4f}"
            )
            if val_loader is not None:
                log += f" | val {val_metric}: {val_score:.4f} (best: {best_score:.4f})"
            print(log)

    # Load best checkpoint if validation was used, else keep final weights
    if best_state is not None:
        print(f"Load from best state from epoch {best_epoch}")
        clf.load_state_dict(best_state)
        X_val = torch.cat([xb for xb, _ in val_loader]).to(device)
        y_val = torch.cat([yb for _, yb in val_loader]).to(device)
        val_metrics = quick_eval(clf, X_val, y_val, device=device)
        val_score = val_metrics[val_metric]
        print(f"Best {val_metric}: {val_score}")

    X_tr = torch.cat([xb for xb, _ in train_loader]).to(device)
    y_tr = torch.cat([yb for _, yb in train_loader]).to(device)
    final_metrics = quick_eval(clf, X_tr, y_tr, device=device)
    return clf, final_metrics

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def infer(
    clf: MLPClassifier,
    X: torch.Tensor,
    return_logits: bool = False,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Score embeddings with a trained classifier.

    Args:
        clf:            A trained MLPClassifier.
        X:              (M, d) float tensor of embeddings.
        return_logits:  If True, return raw logits (unbounded).
                        If False (default), return sigmoid probabilities in [0, 1].
        device:         Device to run inference on.

    Returns:
        scores: (M,) tensor of logits or probabilities depending on return_logits.
    """
    clf = clf.to(device)
    clf.eval()
    X = X.float().to(device)

    with torch.no_grad():
        logits = clf(X).view(-1)
        return logits if return_logits else torch.sigmoid(logits)
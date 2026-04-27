# src/utils.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, precision_recall_curve
)

def evaluate_model(model, dataloader, device, pos_label=1):
    """Evaluate model on dataloader and return metrics and curve data."""
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, pos_label]
            preds = logits.argmax(dim=1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds),
        "recall": recall_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds),
        "auc": roc_auc_score(all_labels, all_probs),
    }
    curves = {
        "roc": roc_curve(all_labels, all_probs),
        "pr": precision_recall_curve(all_labels, all_probs),
    }
    return metrics, curves

def plot_roc_curve(fpr, tpr, auc, save_path="roc_curve.png"):
    """Plot ROC curve and save to file."""
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0,1], [0,1], "--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(save_path, dpi=150)

def plot_pr_curve(recall, precision, save_path="pr_curve.png"):
    """Plot Precision-Recall curve and save to file."""
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.savefig(save_path, dpi=150)


"""
train_gnn.py - Training, Validation, and Evaluation Pipeline for Graph Convolutional Network (GCN).

Benchmarks and evaluates a 2-layer Graph Convolutional Network (GCN) on graph node classification:
- Primary benchmark: Cora citation network (Planetoid benchmark dataset).
- Optional benchmark: Zachary's Karate Club network (--dataset karate).

Provides:
- Semi-supervised node classification with train/val/test masking.
- Model checkpointing based on best validation accuracy.
- Comprehensive evaluation metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
  (implemented with pure NumPy for zero external dependency on scikit-learn).
- Diagnostic visualizations: Training/Validation dynamics, Confusion Matrix heatmap,
  and 2D latent node embedding projections.
- Sample unseen test node predictions with confidence scores.
"""

import os
import sys
import copy
import time
import random
import argparse
from typing import Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend suitable for scripts
import matplotlib.pyplot as plt

from torch_geometric.datasets import Planetoid, KarateClub
import networkx as nx

from gnn_model import GCN


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Metric Utilities (Pure NumPy - Zero Dependency on scikit-learn)
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> Dict[str, Any]:
    """
    Computes classification accuracy, weighted precision, recall, F1-score,
    and confusion matrix using pure NumPy without requiring scikit-learn.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    # Confusion Matrix: rows = true labels, columns = predicted labels
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    total = len(y_true)
    accuracy = float(np.trace(cm) / total) if total > 0 else 0.0

    support = cm.sum(axis=1)
    precisions = []
    recalls = []
    f1s = []

    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = support[c] - tp

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float((2 * prec * rec) / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    sum_support = support.sum()
    if sum_support > 0:
        weighted_precision = float(sum(p * s for p, s in zip(precisions, support)) / sum_support)
        weighted_recall = float(sum(r * s for r, s in zip(recalls, support)) / sum_support)
        weighted_f1 = float(sum(f * s for f, s in zip(f1s, support)) / sum_support)
    else:
        weighted_precision, weighted_recall, weighted_f1 = 0.0, 0.0, 0.0

    return {
        "accuracy": accuracy,
        "precision": weighted_precision,
        "recall": weighted_recall,
        "f1": weighted_f1,
        "confusion_matrix": cm,
    }


def project_to_2d_pca(embeddings: np.ndarray) -> np.ndarray:
    """Projects high-dimensional embeddings to 2D using SVD (Principal Component Analysis)."""
    centered = embeddings - np.mean(embeddings, axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ Vt[:2].T


# ---------------------------------------------------------------------------
# Visualization Routines
# ---------------------------------------------------------------------------

def plot_training_dynamics(
    train_losses, val_losses, train_accs, val_accs, save_path: str
) -> None:
    """Plots training and validation loss and accuracy curves."""
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(epochs, train_losses, label="Train Loss", color="#2E86AB", lw=2)
    ax1.plot(epochs, val_losses, label="Val Loss", color="#D64045", lw=2, linestyle="--")
    ax1.set_title("Cross-Entropy Loss Dynamics", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(frameon=True)

    # Accuracy curve
    ax2.plot(epochs, [a * 100 for a in train_accs], label="Train Accuracy", color="#4CAF50", lw=2)
    ax2.plot(epochs, [a * 100 for a in val_accs], label="Val Accuracy", color="#FF9800", lw=2, linestyle="--")
    ax2.set_title("Classification Accuracy Dynamics", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(frameon=True)

    plt.suptitle("GCN Training & Validation Metrics Over Epochs", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualization Saved] Training dynamics -> {save_path}")


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: str) -> None:
    """Plots a heatmap confusion matrix with text annotations."""
    if len(class_names) != cm.shape[0]:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Test Set Confusion Matrix",
        ylabel="True Class",
        xlabel="Predicted Class",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualization Saved] Confusion matrix  -> {save_path}")


def plot_embeddings(
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    class_names: list,
    save_path: str,
    title: str = "Learned Latent Node Representations (2D Projection)",
) -> None:
    """Plots 2D node representations colored by class."""
    fig, ax = plt.subplots(figsize=(9, 7))
    num_classes = len(class_names)
    cmap = plt.get_cmap("tab10")

    for c in range(num_classes):
        mask = labels == c
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            label=f"{class_names[c]} (Class {c})",
            color=cmap(c % 10),
            alpha=0.75,
            edgecolors="k",
            s=40,
        )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Latent Component 1", fontsize=10)
    ax.set_ylabel("Latent Component 2", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=8, frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualization Saved] Latent embeddings -> {save_path}")


def plot_karate_topology(edge_index: torch.Tensor, labels: np.ndarray, save_path: str) -> None:
    """Visualizes the Zachary's Karate Club network topology using NetworkX."""
    G = nx.Graph()
    G.add_nodes_from(range(len(labels)))
    edges = edge_index.t().tolist()
    G.add_edges_from([(u, v) for u, v in edges if u < v])

    pos = nx.spring_layout(G, seed=42, k=0.35)
    colors = ["#2E86AB" if l == 0 else "#D64045" for l in labels]
    node_sizes = [200 + 40 * G.degree[i] for i in range(len(G))]

    plt.figure(figsize=(9, 7))
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=node_sizes, alpha=0.9, edgecolors="black")
    nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color="gray", width=1.5)
    labels_dict = {i: str(i) for i in range(len(G))}
    nx.draw_networkx_labels(G, pos, labels=labels_dict, font_size=9, font_weight="bold", font_color="white")

    plt.plot([], [], marker="o", color="#2E86AB", ls="", label="Mr. Hi Faction (Class 0)", markersize=10)
    plt.plot([], [], marker="o", color="#D64045", ls="", label="Officer Faction (Class 1)", markersize=10)
    plt.title("Zachary's Karate Club - Graph Topology (34 Nodes, 78 Edges)", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="upper right", frameon=True)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Visualization Saved] Graph topology   -> {save_path}")


# ---------------------------------------------------------------------------
# Dataset Loaders
# ---------------------------------------------------------------------------

def load_cora_dataset(data_dir: str = "data/Cora") -> Tuple[Planetoid, Any]:
    """Downloads and loads the Cora citation network dataset from PyTorch Geometric."""
    dataset = Planetoid(root=data_dir, name="Cora")
    data = dataset[0]
    return dataset, data


def load_karate_club_data(train_ratio: float = 0.6, val_ratio: float = 0.2, seed: int = 42):
    """Loads Zachary's Karate Club dataset and creates stratified train/val/test masks."""
    dataset = KarateClub()
    data = dataset[0]
    num_nodes = data.num_nodes

    rng = np.random.RandomState(seed)
    indices = rng.permutation(num_nodes)

    n_train = int(num_nodes * train_ratio)
    n_val = int(num_nodes * val_ratio)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:n_train]] = True
    val_mask[indices[n_train:n_train + n_val]] = True
    test_mask[indices[n_train + n_val:]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return dataset, data


# ---------------------------------------------------------------------------
# Model Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_mask(model: torch.nn.Module, data, mask: torch.Tensor) -> Tuple[float, float]:
    """Computes cross-entropy loss and accuracy on a given mask."""
    model.eval()
    logits = model(data.x, data.edge_index)
    loss = F.cross_entropy(logits[mask], data.y[mask]).item()
    predictions = logits.argmax(dim=1)
    correct = (predictions[mask] == data.y[mask]).sum().item()
    total = int(mask.sum())
    acc = correct / total if total > 0 else 0.0
    return loss, acc


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

def train_gcn_cora(
    epochs: int = 200,
    hidden_channels: int = 64,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    dropout: float = 0.5,
    seed: int = 42,
    save_path: str = "cora_gcn_best.pt",
    output_dir: str = "results",
) -> Dict[str, Any]:
    """Full training, validation, and evaluation pipeline for GCN on the Cora dataset."""
    print("=" * 70)
    print("   GRAPH CONVOLUTIONAL NETWORK (GCN) - CORA CITATION BENCHMARK")
    print("=" * 70)

    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Dataset
    print("\n[1/5] Loading Cora dataset...")
    dataset, data = load_cora_dataset()

    cora_classes = [
        "Theory",
        "Reinforcement_Learning",
        "Genetic_Algorithms",
        "Neural_Networks",
        "Probabilistic_Methods",
        "Case_Based",
        "Rule_Learning",
    ]
    if len(cora_classes) != dataset.num_classes:
        cora_classes = [f"Class {i}" for i in range(dataset.num_classes)]

    print(f"Dataset:                  {dataset.name}")
    print(f"Number of nodes:          {data.num_nodes:,}")
    print(f"Number of edges:          {data.num_edges:,}")
    print(f"Number of node features:  {dataset.num_node_features}")
    print(f"Number of classes:        {dataset.num_classes}")
    print(f"Training nodes:           {int(data.train_mask.sum())}")
    print(f"Validation nodes:         {int(data.val_mask.sum())}")
    print(f"Test nodes:               {int(data.test_mask.sum())}")

    # 2. Hardware and Model Initialization
    print("\n[2/5] Initializing GCN Model & Optimizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    model = GCN(
        input_channels=dataset.num_node_features,
        hidden_channels=hidden_channels,
        output_channels=dataset.num_classes,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    print(f"Device:                   {device}")
    print(f"Trainable parameters:     {model.count_parameters():,}")
    print(f"Optimizer:                Adam(lr={lr}, weight_decay={weight_decay})")
    print(f"Hidden channels:          {hidden_channels}")
    print(f"Dropout probability:      {dropout}")

    # 3. Training Loop
    print(f"\n[3/5] Starting Training Loop ({epochs} Epochs)...")
    best_val_accuracy = -1.0
    best_state = None
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Track metrics
        train_loss, train_acc = evaluate_mask(model, data, data.train_mask)
        val_loss, val_acc = evaluate_mask(model, data, data.val_mask)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"Epoch: {epoch:03d} | "
                f"Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc*100:.1f}% | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Accuracy: {val_acc*100:.1f}%"
            )

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.2f} seconds.")

    # 4. Checkpoint Restoration & Test Evaluation
    print("\n[4/5] Evaluating Best Model Checkpoint...")
    if best_state is not None:
        model.load_state_dict(best_state)
        if save_path:
            torch.save(best_state, save_path)
            print(f"Saved best model checkpoint to '{save_path}'")

    model.eval()
    with torch.no_grad():
        test_logits = model(data.x, data.edge_index)
        test_preds = test_logits.argmax(dim=1)
        test_embeddings = model.get_embeddings(data.x, data.edge_index).cpu().numpy()

    y_test_true = data.y[data.test_mask].cpu().numpy()
    y_test_pred = test_preds[data.test_mask].cpu().numpy()

    metrics = compute_classification_metrics(y_test_true, y_test_pred, dataset.num_classes)
    test_acc = metrics["accuracy"]
    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]
    cm = metrics["confusion_matrix"]

    print("\n---------------- TEST RESULTS ----------------")
    print(f"Best Validation Acc : {best_val_accuracy * 100:.2f}%")
    print(f"Test Accuracy       : {test_acc * 100:.2f}%")
    print(f"Weighted Precision  : {precision:.4f}")
    print(f"Weighted Recall     : {recall:.4f}")
    print(f"Weighted F1-Score   : {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    print("----------------------------------------------")

    # 5. Sample Test Predictions Inspection
    print("\n[5/5] Sample Test-Node Predictions:")
    print("-" * 75)
    print(f"{'Node ID':<10}{'Predicted Class':<18}{'Actual Class':<15}{'Confidence':<12}{'Result'}")
    print("-" * 75)

    with torch.no_grad():
        probabilities = torch.softmax(test_logits, dim=1)
        test_indices = torch.where(data.test_mask)[0][:10]

        for node_id in test_indices.tolist():
            predicted_class = int(test_preds[node_id])
            true_class = int(data.y[node_id])
            confidence = float(probabilities[node_id].max())
            status = "CORRECT" if predicted_class == true_class else "MISMATCH"

            print(
                f"Node {node_id:<5} "
                f"Predicted={predicted_class:<8} "
                f"Actual={true_class:<7} "
                f"Confidence={confidence:.4f}   "
                f"[{status}]"
            )
    print("-" * 75)

    # 6. Generate Diagnostic Visualizations
    print("\n[6/6] Generating Diagnostic Visualizations...")
    dynamics_path = os.path.join(output_dir, "cora_training_dynamics.png")
    cm_path = os.path.join(output_dir, "cora_confusion_matrix.png")
    emb_path = os.path.join(output_dir, "cora_node_embeddings.png")

    plot_training_dynamics(train_losses, val_losses, train_accs, val_accs, dynamics_path)
    plot_confusion_matrix(cm, cora_classes, cm_path)

    # 2D Latent space visualization via PCA
    emb_2d = project_to_2d_pca(test_embeddings)
    plot_embeddings(
        emb_2d,
        data.y.cpu().numpy(),
        cora_classes,
        emb_path,
        title="Cora Citation Network - Learned GCN Latent Node Embeddings",
    )

    print(f"\nAll artifacts and figures successfully saved to: {output_dir}")

    return {
        "best_val_accuracy": best_val_accuracy,
        "test_accuracy": test_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "parameters": model.count_parameters(),
    }


def train_gcn_karate(
    epochs: int = 150,
    hidden_channels: int = 16,
    lr: float = 0.02,
    weight_decay: float = 5e-4,
    dropout: float = 0.2,
    seed: int = 42,
    save_path: str = "karate_gcn_best.pt",
    output_dir: str = "results",
) -> Dict[str, Any]:
    """Training pipeline for GCN on Zachary's Karate Club dataset with topology visualization."""
    print("=" * 70)
    print("   GRAPH CONVOLUTIONAL NETWORK (GCN) - ZACHARY'S KARATE CLUB")
    print("=" * 70)

    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    dataset, data = load_karate_club_data(train_ratio=0.6, val_ratio=0.2, seed=seed)
    class_names = ["Mr. Hi (Class 0)", "Officer (Class 1)"]

    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    print(f"Train nodes: {int(data.train_mask.sum())}, Val nodes: {int(data.val_mask.sum())}, Test nodes: {int(data.test_mask.sum())}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    model = GCN(
        input_channels=dataset.num_node_features,
        hidden_channels=hidden_channels,
        output_channels=dataset.num_classes,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Initial untrained embeddings
    init_emb = model.get_embeddings(data.x, data.edge_index).cpu().numpy()

    best_val_loss = float("inf")
    best_state = None
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        train_loss, train_acc = evaluate_mask(model, data, data.train_mask)
        val_loss, val_acc = evaluate_mask(model, data, data.val_mask)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 25 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}%")

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, os.path.join(output_dir, save_path))

    model.eval()
    with torch.no_grad():
        test_logits = model(data.x, data.edge_index)
        test_preds = test_logits.argmax(dim=1)
        final_emb = model.get_embeddings(data.x, data.edge_index).cpu().numpy()

    y_test_true = data.y[data.test_mask].cpu().numpy()
    y_test_pred = test_preds[data.test_mask].cpu().numpy()

    metrics = compute_classification_metrics(y_test_true, y_test_pred, dataset.num_classes)
    print("\n---------------- TEST RESULTS ----------------")
    print(f"Test Accuracy : {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision     : {metrics['precision']:.4f}")
    print(f"Recall        : {metrics['recall']:.4f}")
    print(f"F1-Score      : {metrics['f1']:.4f}")
    print("Confusion Matrix:\n", metrics["confusion_matrix"])
    print("----------------------------------------------")

    # Generate Karate Club Visualizations
    plot_karate_topology(data.edge_index.cpu(), data.y.cpu().numpy(), os.path.join(output_dir, "karate_graph_topology.png"))
    plot_training_dynamics(train_losses, val_losses, train_accs, val_accs, os.path.join(output_dir, "karate_training_dynamics.png"))
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, os.path.join(output_dir, "karate_confusion_matrix.png"))

    init_2d = project_to_2d_pca(init_emb)
    final_2d = project_to_2d_pca(final_emb)
    plot_embeddings(final_2d, data.y.cpu().numpy(), class_names, os.path.join(output_dir, "karate_embeddings.png"), title="Karate Club - GCN Learned Embeddings")

    return metrics


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Graph Convolutional Network (GCN) on benchmark datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        choices=["cora", "karate"],
        help="Dataset benchmark to train on: 'cora' (default) or 'karate'",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs (default: 200)")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden unit dimension (default: 64)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (default: 0.01)")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay for Adam (default: 5e-4)")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate (default: 0.5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()

    if args.dataset == "cora":
        train_gcn_cora(
            epochs=args.epochs,
            hidden_channels=args.hidden_dim,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            seed=args.seed,
            save_path="cora_gcn_best.pt",
        )
    else:
        train_gcn_karate(
            epochs=args.epochs,
            hidden_channels=args.hidden_dim if args.hidden_dim != 64 else 16,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()

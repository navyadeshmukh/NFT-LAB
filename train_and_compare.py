"""
train_and_compare.py - Unified Benchmarking Suite for RNN vs LSTM vs GRU Sentiment Analysis.

Trains Vanilla RNN, LSTM, and GRU models on the same sentiment dataset with identical
hyperparameters, evaluates their performance, logs parameter counts, training times,
and generates a comprehensive comparison report.
"""

import time
import torch
import torch.nn as nn
from typing import Dict, Any

from dataset import get_data_loaders
from rnn_model import VanillaRNNClassifier
from lstm_model import LSTMClassifier
from gru_model import GRUClassifier


def count_parameters(model: nn.Module) -> int:
    """Returns the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_training_loop(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    num_epochs: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Trains a model and records training & evaluation metrics."""
    start_time = time.time()
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, num_epochs + 1):
        # --- Training ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for seqs, labels, lengths in train_loader:
            seqs, labels, lengths = seqs.to(device), labels.to(device), lengths.to(device)
            optimizer.zero_grad()
            predictions = model(seqs, lengths)
            loss = criterion(predictions, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            preds = (torch.sigmoid(predictions) >= 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_correct / train_total

        # --- Validation ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for seqs, labels, lengths in val_loader:
                seqs, labels, lengths = seqs.to(device), labels.to(device), lengths.to(device)
                predictions = model(seqs, lengths)
                loss = criterion(predictions, labels)

                preds = (torch.sigmoid(predictions) >= 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_correct / val_total

        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(avg_train_acc)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(avg_val_acc)

        if epoch % 5 == 0 or epoch == num_epochs:
            print(
                f"  [{model.model_type}] Epoch {epoch:02d}/{num_epochs:02d} "
                f"| Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc*100:.1f}% "
                f"| Val Loss: {avg_val_loss:.4f} Acc: {avg_val_acc*100:.1f}%"
            )

    elapsed_time = time.time() - start_time

    # Measure average inference latency for 100 passes
    dummy_input = torch.randint(0, 10, (1, 15), dtype=torch.long, device=device)
    dummy_len = torch.tensor([15], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        t0 = time.time()
        for _ in range(100):
            _ = model(dummy_input, dummy_len)
        latency_ms = ((time.time() - t0) / 100) * 1000

    return {
        "final_train_acc": history["train_acc"][-1],
        "final_val_acc": history["val_acc"][-1],
        "final_val_loss": history["val_loss"][-1],
        "training_time_sec": elapsed_time,
        "latency_ms": latency_ms,
        "history": history,
    }


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=" * 68)
    print(f"  RNN vs LSTM vs GRU: Sentiment Analysis Comparative Benchmark")
    print(f"  Execution Device: {device}")
    print(f"=" * 68)

    # 1. Prepare Data & Vocabulary
    batch_size = 8
    train_loader, val_loader, vocab = get_data_loaders(batch_size=batch_size, seed=42)
    vocab_size = len(vocab)
    vocab.save("vocab.json")
    print(f"\n[Data] Vocabulary Size: {vocab_size} tokens | Batch Size: {batch_size}")
    print(f"[Data] Training Batches: {len(train_loader)} | Validation Batches: {len(val_loader)}")

    # Shared Hyperparameters
    EMBEDDING_DIM = 64
    HIDDEN_DIM = 64
    OUTPUT_DIM = 1
    NUM_EPOCHS = 25
    LR = 0.002
    WEIGHT_DECAY = 1e-4

    # 2. Instantiate Architectures
    models = {
        "Vanilla RNN": VanillaRNNClassifier(
            vocab_size=vocab_size,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
            n_layers=1,
            dropout=0.2,
            pad_idx=vocab.PAD_IDX,
        ).to(device),
        "LSTM (Bi-directional)": LSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
            n_layers=1,
            bidirectional=True,
            dropout=0.3,
            pad_idx=vocab.PAD_IDX,
        ).to(device),
        "GRU (Bi-directional)": GRUClassifier(
            vocab_size=vocab_size,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
            n_layers=1,
            bidirectional=True,
            dropout=0.3,
            pad_idx=vocab.PAD_IDX,
        ).to(device),
    }

    results = {}
    saved_files = {
        "Vanilla RNN": "rnn_model.pt",
        "LSTM (Bi-directional)": "lstm_model.pt",
        "GRU (Bi-directional)": "gru_model.pt",
    }

    criterion = nn.BCEWithLogitsLoss()

    # 3. Train and Evaluate each Architecture
    for name, model in models.items():
        print(f"\n>>> Training {name} (Params: {count_parameters(model):,}) ...")
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        metrics = run_training_loop(model, train_loader, val_loader, optimizer, criterion, NUM_EPOCHS, device)
        metrics["params"] = count_parameters(model)
        results[name] = metrics

        # Save weights
        save_path = saved_files[name]
        torch.save(model.state_dict(), save_path)
        print(f"    Saved weights to: {save_path}")

    # 4. Print Summary Comparison Table
    print("\n" + "=" * 80)
    print(f"{'Model Architecture':<24} | {'Params':<9} | {'Train Acc':<10} | {'Val Acc':<9} | {'Time (s)':<9} | {'Latency':<8}")
    print("-" * 80)
    for name, res in results.items():
        print(
            f"{name:<24} | "
            f"{res['params']:<9,d} | "
            f"{res['final_train_acc']*100:>8.1f}% | "
            f"{res['final_val_acc']*100:>7.1f}% | "
            f"{res['training_time_sec']:>8.2f}s | "
            f"{res['latency_ms']:>6.2f}ms"
        )
    print("=" * 80)

    print("\n[Complete] All models trained, evaluated, and weights saved.")
    print("Run `python predict_sentences.py` to inspect sentence-by-sentence predictions.\n")


if __name__ == "__main__":
    main()

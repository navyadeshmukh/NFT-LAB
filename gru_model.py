r"""
gru_model.py - Gated Recurrent Unit (GRU) Network for Sentiment Analysis.

Theoretical Overview:
---------------------
Introduced by Cho et al. (2014) as a streamlined alternative to LSTM.
GRU eliminates the separate Cell State (C_t) and replaces the 3 gates of LSTM
with just 2 gates:

1. Reset Gate (r_t):
   Determines how much of the past hidden state to forget when computing candidate memory:
   r_t = sigmoid(W_r * [h_{t-1}, x_t] + b_r)

2. Update Gate (z_t):
   Controls how much of the previous state is kept vs. replaced by new candidate memory
   (acts as both forget and input gate combined):
   z_t = sigmoid(W_z * [h_{t-1}, x_t] + b_z)

3. Candidate Hidden State (\tilde{h}_t):
   Information proposal incorporating current input and reset past state:
   \tilde{h}_t = tanh(W * [r_t \odot h_{t-1}, x_t] + b)

4. Hidden State Update (h_t):
   Linear interpolation between previous state and candidate state:
   h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t

Advantages:
- Faster convergence and ~25% fewer parameters compared to standard LSTM.
- Retains gating benefits to solve vanishing gradients on long sequences.
"""

import torch
import torch.nn as nn
from typing import Tuple


class GRUClassifier(nn.Module):
    """Gated Recurrent Unit (GRU) for sentiment classification."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        output_dim: int = 1,
        n_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.model_type = "GRU"
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # PyTorch GRU
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        num_directions = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * num_directions, output_dim)

    def forward(self, text: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            text: [batch_size, seq_len]
            lengths: [batch_size]
        Returns:
            logits: [batch_size]
        """
        # [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(self.embedding(text))

        # gru_out: [batch_size, seq_len, hidden_dim * num_directions]
        # hidden (h_n): [num_layers * num_directions, batch_size, hidden_dim]
        gru_out, hidden = self.gru(embedded)

        if self.bidirectional:
            last_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            if lengths is not None:
                batch_size = text.size(0)
                idx = (lengths - 1).view(-1, 1).expand(batch_size, self.hidden_dim).unsqueeze(1)
                last_hidden = gru_out.gather(1, idx).squeeze(1)
            else:
                last_hidden = hidden[-1]

        dropped = self.dropout(last_hidden)
        logits = self.fc(dropped).squeeze(1)
        return logits


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    for seqs, labels, lengths in dataloader:
        seqs, labels, lengths = seqs.to(device), labels.to(device), lengths.to(device)
        optimizer.zero_grad()
        predictions = model(seqs, lengths)
        loss = criterion(predictions, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds = (torch.sigmoid(predictions) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        epoch_loss += loss.item()

    return epoch_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for seqs, labels, lengths in dataloader:
            seqs, labels, lengths = seqs.to(device), labels.to(device), lengths.to(device)
            predictions = model(seqs, lengths)
            loss = criterion(predictions, labels)

            preds = (torch.sigmoid(predictions) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader), correct / total


def predict_sentiment(model: nn.Module, sentence: str, vocab, device: torch.device) -> Tuple[str, float]:
    """Predicts sentiment for a single sentence."""
    model.eval()
    indices = vocab.text_to_indices(sentence)
    if len(indices) == 0:
        return "Neutral / Empty", 0.5

    tensor = torch.tensor([indices], dtype=torch.long, device=device)
    length = torch.tensor([len(indices)], dtype=torch.long, device=device)

    with torch.no_grad():
        logit = model(tensor, length)
        prob = torch.sigmoid(logit).item()

    label = "Positive" if prob >= 0.5 else "Negative"
    return label, prob


if __name__ == "__main__":
    from dataset import get_data_loaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running GRU Training on device: {device}")

    train_loader, val_loader, vocab = get_data_loaders(batch_size=8)
    model = GRUClassifier(
        vocab_size=len(vocab),
        embedding_dim=64,
        hidden_dim=64,
        output_dim=1,
        n_layers=1,
        bidirectional=True,
        dropout=0.3,
        pad_idx=vocab.PAD_IDX,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)

    num_epochs = 25
    print("\n--- Training Bidirectional GRU ---")
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if epoch % 5 == 0 or epoch == num_epochs:
            print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}%")

    torch.save(model.state_dict(), "gru_model.pt")
    vocab.save("vocab.json")
    print("\nModel saved to gru_model.pt and vocab to vocab.json")

    # Interactive test sentences
    test_samples = [
        "The software runs smoothly and is a pleasure to use.",
        "Completely useless product, customer service was awful.",
        "Not bad, actually turned out better than I thought.",
    ]
    print("\n--- Testing Single Sentences ---")
    for sent in test_samples:
        lbl, conf = predict_sentiment(model, sent, vocab, device)
        print(f"Sentence: \"{sent}\"")
        print(f" -> Prediction: {lbl} (Score: {conf:.4f})\n")

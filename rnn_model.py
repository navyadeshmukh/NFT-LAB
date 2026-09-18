"""
rnn_model.py - Vanilla / Simple Recurrent Neural Network (RNN) for Sentiment Analysis.

Theoretical Overview:
---------------------
In a standard Elman RNN:
    h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)
    y_t = W_ho * h_t + b_o

At each time-step t:
- Input vector x_t (word embedding) is combined with the previous hidden state h_{t-1}.
- The recurrence allows memory to persist over time.
- For sentiment classification (many-to-one), the final hidden state h_T summarizes
  the whole sentence into a single vector, passed to a linear classification head.

Limitation:
- Subject to vanishing and exploding gradients when backpropagating through long sequences,
  making it difficult to capture long-range dependencies.
"""

import torch
import torch.nn as nn
from typing import Tuple


class VanillaRNNClassifier(nn.Module):
    """Vanilla (Elman) RNN for sequence classification."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        output_dim: int = 1,
        n_layers: int = 1,
        dropout: float = 0.2,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.model_type = "RNN"
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Embedding layer maps word token IDs into continuous dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # Standard RNN layer
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            nonlinearity="tanh",
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            text: Tensor of token IDs with shape [batch_size, seq_len]
            lengths: Tensor of actual lengths for each sentence [batch_size]
        Returns:
            logits: Output logits of shape [batch_size]
        """
        # embedded shape: [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(self.embedding(text))

        # rnn_out: [batch_size, seq_len, hidden_dim]
        # hidden: [n_layers, batch_size, hidden_dim]
        rnn_out, hidden = self.rnn(embedded)

        if lengths is not None:
            # Extract hidden state at each sequence's actual last non-padded token
            batch_size = text.size(0)
            idx = (lengths - 1).view(-1, 1).expand(batch_size, self.hidden_dim).unsqueeze(1)
            last_hidden = rnn_out.gather(1, idx).squeeze(1)
        else:
            # Fallback to top-layer final hidden state
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
        
        # Gradient clipping to mitigate exploding gradients
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
    print(f"Running Vanilla RNN Training on device: {device}")

    train_loader, val_loader, vocab = get_data_loaders(batch_size=8)
    model = VanillaRNNClassifier(
        vocab_size=len(vocab),
        embedding_dim=64,
        hidden_dim=64,
        output_dim=1,
        n_layers=1,
        dropout=0.2,
        pad_idx=vocab.PAD_IDX,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    num_epochs = 25
    print("\n--- Training Vanilla RNN ---")
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if epoch % 5 == 0 or epoch == num_epochs:
            print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}%")

    # Save model
    torch.save(model.state_dict(), "rnn_model.pt")
    vocab.save("vocab.json")
    print("\nModel saved to rnn_model.pt and vocab to vocab.json")

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

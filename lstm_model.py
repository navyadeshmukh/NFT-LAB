r"""
lstm_model.py - Long Short-Term Memory (LSTM) Network for Sentiment Analysis.

Theoretical Overview:
---------------------
Proposed by Hochreiter & Schmidhuber (1997) to overcome the vanishing gradient
problem in vanilla RNNs. LSTM introduces an internal memory "Cell State" (C_t)
and three multiplicative gates:

1. Forget Gate (f_t):
   Decides what information to discard from the previous cell state C_{t-1}.
   f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)

2. Input Gate (i_t) & Candidate Cell State (\tilde{C}_t):
   Decides which new information to store in the cell state.
   i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)
   \tilde{C}_t = tanh(W_c * [h_{t-1}, x_t] + b_c)

3. Cell State Update (C_t):
   Combines forgotten old memory and weighted candidate memory via additive update:
   C_t = f_t * C_{t-1} + i_t * \tilde{C}_t
   (Note: The additive update creates a "gradient highway" that prevents vanishing gradients)

4. Output Gate (o_t) & Hidden State (h_t):
   Decides what parts of the cell state make it into the visible hidden state h_t:
   o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)
   h_t = o_t * tanh(C_t)
"""

import torch
import torch.nn as nn
from typing import Tuple


class LSTMClassifier(nn.Module):
    """LSTM sequence classifier with optional bidirectionality and dropout."""

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
        self.model_type = "LSTM"
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # PyTorch LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        num_directions = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        # Linear layer mapping concatenated forward + backward hidden states to output
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

        # lstm_out: [batch_size, seq_len, hidden_dim * num_directions]
        # hidden (h_n): [num_layers * num_directions, batch_size, hidden_dim]
        # cell (c_n):   [num_layers * num_directions, batch_size, hidden_dim]
        lstm_out, (hidden, cell) = self.lstm(embedded)

        if self.bidirectional:
            # Concatenate the final forward hidden state and backward hidden state
            # hidden[-2] is the last forward layer, hidden[-1] is the last backward layer
            last_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            if lengths is not None:
                batch_size = text.size(0)
                idx = (lengths - 1).view(-1, 1).expand(batch_size, self.hidden_dim).unsqueeze(1)
                last_hidden = lstm_out.gather(1, idx).squeeze(1)
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
    print(f"Running LSTM Training on device: {device}")

    train_loader, val_loader, vocab = get_data_loaders(batch_size=8)
    model = LSTMClassifier(
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
    print("\n--- Training Bidirectional LSTM ---")
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if epoch % 5 == 0 or epoch == num_epochs:
            print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}%")

    torch.save(model.state_dict(), "lstm_model.pt")
    vocab.save("vocab.json")
    print("\nModel saved to lstm_model.pt and vocab to vocab.json")

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

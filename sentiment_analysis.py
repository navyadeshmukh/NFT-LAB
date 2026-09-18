"""
sentiment_analysis.py - Main entrypoint for the Sentiment Analysis suite.

Provides a unified interface to:
1. Run comparative training and evaluation (RNN vs LSTM vs GRU).
2. Run multi-sentence comparative inference.
3. Classify arbitrary interactive user input.
"""

import sys
import os
import torch

from dataset import Vocabulary
from rnn_model import VanillaRNNClassifier
from lstm_model import LSTMClassifier
from gru_model import GRUClassifier


def load_all_models(vocab_path: str = "vocab.json", device: torch.device = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(vocab_path):
        print(f"Error: {vocab_path} not found. Running training first...")
        import train_and_compare
        train_and_compare.main()

    vocab = Vocabulary.load(vocab_path)
    vocab_size = len(vocab)

    rnn = VanillaRNNClassifier(vocab_size=vocab_size, embedding_dim=64, hidden_dim=64, pad_idx=vocab.PAD_IDX).to(device)
    lstm = LSTMClassifier(vocab_size=vocab_size, embedding_dim=64, hidden_dim=64, bidirectional=True, pad_idx=vocab.PAD_IDX).to(device)
    gru = GRUClassifier(vocab_size=vocab_size, embedding_dim=64, hidden_dim=64, bidirectional=True, pad_idx=vocab.PAD_IDX).to(device)

    if os.path.exists("rnn_model.pt"):
        rnn.load_state_dict(torch.load("rnn_model.pt", map_location=device))
    if os.path.exists("lstm_model.pt"):
        lstm.load_state_dict(torch.load("lstm_model.pt", map_location=device))
    if os.path.exists("gru_model.pt"):
        gru.load_state_dict(torch.load("gru_model.pt", map_location=device))

    rnn.eval()
    lstm.eval()
    gru.eval()

    return {"RNN": rnn, "LSTM": lstm, "GRU": gru}, vocab, device


def predict_single(models, vocab, sentence: str, device):
    indices = vocab.text_to_indices(sentence)
    if len(indices) == 0:
        return {name: ("Neutral", 0.5) for name in models}

    tensor = torch.tensor([indices], dtype=torch.long, device=device)
    length = torch.tensor([len(indices)], dtype=torch.long, device=device)

    results = {}
    with torch.no_grad():
        for name, model in models.items():
            logit = model(tensor, length)
            prob = torch.sigmoid(logit).item()
            label = "Positive" if prob >= 0.5 else "Negative"
            results[name] = (label, prob)
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        import train_and_compare
        train_and_compare.main()
    elif len(sys.argv) > 1 and sys.argv[1] == "--predict":
        import predict_sentences
        predict_sentences.main()
    else:
        print("Starting comparative training...")
        import train_and_compare
        train_and_compare.main()
        print("\nExecuting multi-sentence prediction analysis...")
        import predict_sentences
        predict_sentences.main()

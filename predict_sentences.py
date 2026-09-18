"""
predict_sentences.py - Multi-Sentence Sentiment Evaluation across RNN, LSTM, and GRU.

Loads trained models (Vanilla RNN, LSTM, GRU) and evaluates multiple sentences separately.
Includes positive, negative, negation, nuanced, and mixed-sentiment sentences to highlight
the difference in performance and contextual handling across the three architectures.
"""

import sys
import os
import torch
from typing import List, Dict, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dataset import Vocabulary
from rnn_model import VanillaRNNClassifier
from lstm_model import LSTMClassifier
from gru_model import GRUClassifier


# ----------------------------------------------------------------------
# Diverse Multi-Sentence Evaluation Benchmark
# ----------------------------------------------------------------------
EVALUATION_SENTENCES: List[Dict[str, str]] = [
    # Category 1: Clear Positive Sentences
    {
        "sentence": "The user interface is slick, intuitive, and remarkably fast.",
        "category": "Clear Positive",
        "expected": "Positive",
    },
    {
        "sentence": "I absolutely loved this masterpiece, brilliantly directed and acted!",
        "category": "Strong Positive",
        "expected": "Positive",
    },
    {
        "sentence": "Customer support was polite, prompt, and solved the issue immediately.",
        "category": "Clear Positive",
        "expected": "Positive",
    },
    {
        "sentence": "Outstanding battery life and great value for the price.",
        "category": "Clear Positive",
        "expected": "Positive",
    },

    # Category 2: Clear Negative Sentences
    {
        "sentence": "The app crashes constantly and corrupts my saved files.",
        "category": "Clear Negative",
        "expected": "Negative",
    },
    {
        "sentence": "Horrible build quality, cheap flimsy plastic that cracked instantly.",
        "category": "Clear Negative",
        "expected": "Negative",
    },
    {
        "sentence": "Terrible customer service, rude representatives and no refund given.",
        "category": "Strong Negative",
        "expected": "Negative",
    },
    {
        "sentence": "Total waste of money, do not make the mistake of buying this.",
        "category": "Strong Negative",
        "expected": "Negative",
    },

    # Category 3: Negations & Nuance (Crucial test for recurrent gating)
    {
        "sentence": "The product was not good at all, completely failed expectations.",
        "category": "Negated Negative",
        "expected": "Negative",
    },
    {
        "sentence": "Not bad at all, actually quite pleasant and well made.",
        "category": "Negated Positive",
        "expected": "Positive",
    },
    {
        "sentence": "Never disappointed by this company, always delivers top quality.",
        "category": "Negated Positive",
        "expected": "Positive",
    },
    {
        "sentence": "I did not enjoy the show, it dragged on endlessly.",
        "category": "Negated Negative",
        "expected": "Negative",
    },

    # Category 4: Mixed Sentiment / Contrast (Requires long-term contextual state)
    {
        "sentence": "The acting was decent, but the terrible script ruined everything.",
        "category": "Mixed (Ending Negative)",
        "expected": "Negative",
    },
    {
        "sentence": "Despite the slow opening, the movie ended on a magnificent high note.",
        "category": "Mixed (Ending Positive)",
        "expected": "Positive",
    },
    {
        "sentence": "The camera hardware looks nice, but the software is disastrously buggy.",
        "category": "Mixed (Ending Negative)",
        "expected": "Negative",
    },
    {
        "sentence": "I was skeptical at first, but the results were genuinely impressive.",
        "category": "Mixed (Ending Positive)",
        "expected": "Positive",
    },
]


def load_trained_models(vocab_path: str = "vocab.json", device: torch.device = None):
    """Loads vocabulary and the 3 trained model checkpoints."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file '{vocab_path}' not found. Please run train_and_compare.py first.")

    vocab = Vocabulary.load(vocab_path)
    vocab_size = len(vocab)

    # Instantiate model architectures with matching dimensions
    rnn = VanillaRNNClassifier(vocab_size=vocab_size, embedding_dim=64, hidden_dim=64, pad_idx=vocab.PAD_IDX).to(device)
    lstm = LSTMClassifier(vocab_size=vocab_size, embedding_dim=64, hidden_dim=64, bidirectional=True, pad_idx=vocab.PAD_IDX).to(device)
    gru = GRUClassifier(vocab_size=vocab_size, embedding_dim=64, hidden_dim=64, bidirectional=True, pad_idx=vocab.PAD_IDX).to(device)

    # Load weights
    for name, model, path in [("RNN", rnn, "rnn_model.pt"), ("LSTM", lstm, "lstm_model.pt"), ("GRU", gru, "gru_model.pt")]:
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
        else:
            raise FileNotFoundError(f"Weights file '{path}' not found. Please run train_and_compare.py first.")

    return {"RNN": rnn, "LSTM": lstm, "GRU": gru}, vocab, device


def predict_sentence_all_models(
    sentence: str,
    models: Dict[str, torch.nn.Module],
    vocab: Vocabulary,
    device: torch.device,
) -> Dict[str, Tuple[str, float]]:
    """Evaluates a single sentence across all models and returns labels and probabilities."""
    indices = vocab.text_to_indices(sentence)
    if len(indices) == 0:
        return {name: ("Neutral", 0.5) for name in models}

    tensor = torch.tensor([indices], dtype=torch.long, device=device)
    length = torch.tensor([len(indices)], dtype=torch.long, device=device)

    results = {}
    with torch.no_grad():
        for name, model in models.items():
            model.eval()
            logit = model(tensor, length)
            prob = torch.sigmoid(logit).item()
            label = "Positive" if prob >= 0.5 else "Negative"
            results[name] = (label, prob)

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 80)
    print("      MULTI-SENTENCE SENTIMENT ANALYSIS INFERENCE SUITE")
    print("      Comparing Vanilla RNN vs. LSTM vs. GRU on Diverse Test Sentences")
    print("=" * 80)

    try:
        models, vocab, device = load_trained_models("vocab.json", device)
    except FileNotFoundError as e:
        print(f"\n[Warning] {e}")
        print("Training models now...")
        import train_and_compare
        train_and_compare.main()
        models, vocab, device = load_trained_models("vocab.json", device)

    # If user provided custom sentences via CLI arguments
    custom_sentences = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    if custom_sentences:
        print(f"\nEvaluating {len(custom_sentences)} Custom Input Sentence(s):\n")
        for i, sent in enumerate(custom_sentences, 1):
            res = predict_sentence_all_models(sent, models, vocab, device)
            print(f"[{i}] \"{sent}\"")
            for m_name, (lbl, score) in res.items():
                print(f"    {m_name:<5} -> {lbl:<8} (Confidence / P(Pos): {score:.4f})")
            print()
        return

    # Run on the benchmark multi-sentence list
    print(f"\nRunning evaluation on {len(EVALUATION_SENTENCES)} diverse sentences:\n")

    summary_records = []

    for idx, item in enumerate(EVALUATION_SENTENCES, 1):
        sent = item["sentence"]
        cat = item["category"]
        expected = item["expected"]
        predictions = predict_sentence_all_models(sent, models, vocab, device)

        print(f"--------------------------------------------------------------------------------")
        print(f"Sentence [{idx:02d}] ({cat}) | Expected: {expected}")
        print(f"Text: \"{sent}\"")
        tokens = vocab.tokenize(sent)
        print(f"Tokens: {tokens}")
        print(f"Predictions:")
        
        row = {"idx": idx, "text": sent[:30] + "..." if len(sent) > 30 else sent, "expected": expected}
        for model_name in ["RNN", "LSTM", "GRU"]:
            lbl, score = predictions[model_name]
            match_mark = "[PASS]" if lbl == expected else "[FAIL]"
            print(f"  * {model_name:<5} -> {lbl:<8} (P(Pos): {score*100:5.1f}%) {match_mark}")
            row[f"{model_name}_pred"] = f"{lbl} ({score*100:4.1f}%)"
            row[f"{model_name}_ok"] = (lbl == expected)

        summary_records.append(row)
        print()

    # Final Comparative Summary Table
    print("=" * 88)
    print(f"{'#':<3} | {'Expected':<8} | {'Vanilla RNN':<18} | {'LSTM (Bi)':<18} | {'GRU (Bi)':<18}")
    print("=" * 88)
    rnn_correct, lstm_correct, gru_correct = 0, 0, 0
    for r in summary_records:
        rnn_m = "[PASS]" if r["RNN_ok"] else "[FAIL]"
        lstm_m = "[PASS]" if r["LSTM_ok"] else "[FAIL]"
        gru_m = "[PASS]" if r["GRU_ok"] else "[FAIL]"
        if r["RNN_ok"]: rnn_correct += 1
        if r["LSTM_ok"]: lstm_correct += 1
        if r["GRU_ok"]: gru_correct += 1

        print(
            f"{r['idx']:<3} | "
            f"{r['expected']:<8} | "
            f"{r['RNN_pred']:<11} {rnn_m:<6} | "
            f"{r['LSTM_pred']:<11} {lstm_m:<6} | "
            f"{r['GRU_pred']:<11} {gru_m}"
        )
    print("-" * 88)
    total = len(summary_records)
    print(
        f"{'Accuracy':<14} | "
        f"{rnn_correct}/{total} ({rnn_correct/total*100:5.1f}%)         | "
        f"{lstm_correct}/{total} ({lstm_correct/total*100:5.1f}%)         | "
        f"{gru_correct}/{total} ({gru_correct/total*100:5.1f}%)"
    )
    print("=" * 88)


if __name__ == "__main__":
    main()

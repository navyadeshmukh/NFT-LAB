"""
dataset.py - Text Preprocessing, Vocabulary Building, and PyTorch Dataset for Sentiment Analysis.
"""

import re
import json
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------
# 1. Comprehensive Sentiment Dataset
# Labeled: 1 = Positive, 0 = Negative
# ---------------------------------------------------------
RAW_DATA: List[Tuple[str, int]] = [
    # --- Positive Sentences ---
    ("I absolutely love this movie, it was a breathtaking masterpiece!", 1),
    ("The user experience is seamless, responsive, and wonderfully designed.", 1),
    ("Customer support went above and beyond to solve my issue quickly.", 1),
    ("Fantastic performance by the entire cast, highly recommended!", 1),
    ("This phone has outstanding battery life and a stunning camera.", 1),
    ("I am extremely happy with this purchase, worth every single penny.", 1),
    ("The food at this restaurant was delicious and authentic.", 1),
    ("An inspiring and deeply moving story that touched my heart.", 1),
    ("Super easy setup, clear instructions, and works flawlessly.", 1),
    ("Brilliant innovation, this tool has doubled our team productivity.", 1),
    ("The sound quality of these headphones is crisp and immersive.", 1),
    ("Very impressed with the fast delivery and premium packaging.", 1),
    ("One of the best books I have read this year, beautifully written.", 1),
    ("Exceptional service and friendly staff made our stay memorable.", 1),
    ("The graphics and storyline in this game are genuinely amazing.", 1),
    ("Simple, elegant, and perfectly meets all our business needs.", 1),
    ("The software update fixed all bugs and improved battery longevity.", 1),
    ("A delightful surprise, exceeded all my initial expectations.", 1),
    ("Top notch engineering and robust build quality.", 1),
    ("Great value for money, reliable and sturdy product.", 1),
    ("The interface is intuitive and a joy to navigate every day.", 1),
    ("I truly adore the attention to detail in this design.", 1),
    ("Superb acting and thrilling plot twists throughout the show.", 1),
    ("I would definitely recommend this to all my friends and family.", 1),
    ("Remarkable performance and seamless integration with existing tools.", 1),
    ("Five stars! The craftsmanship and attention to detail are superb.", 1),
    ("Wonderful experience from start to finish.", 1),
    ("The coffee is rich, aromatic, and brewed to absolute perfection.", 1),
    ("Incredibly smooth workflow and stellar collaboration features.", 1),
    ("A triumph of creative storytelling and emotional depth.", 1),

    # --- Negative Sentences ---
    ("This product is a total waste of money, stopped working on day two.", 0),
    ("Terrible customer service, rude representatives and no resolution.", 0),
    ("The movie was boring, predictable, and completely uninspired.", 0),
    ("Horrible build quality, cheap plastic that cracked immediately.", 0),
    ("App crashes constantly every time I try to save my work.", 0),
    ("Extremely disappointing experience, would never buy again.", 0),
    ("The battery drains in less than two hours, completely unusable.", 0),
    ("Overpriced, mediocre food with painfully slow service.", 0),
    ("The delivery was delayed by two weeks and arrived badly damaged.", 0),
    ("Frustrating interface, confusing navigation, and filled with bugs.", 0),
    ("I deeply regret this purchase, do not make the same mistake.", 0),
    ("The storyline was nonsensical and the dialogue was cringe-worthy.", 0),
    ("Subpar performance, lags continuously under minimal load.", 0),
    ("Zero customer assistance, tickets are ignored for weeks.", 0),
    ("The camera produces blurry, noisy images even in daylight.", 0),
    ("Unreliable hardware that failed right after the return window.", 0),
    ("Worst dining experience of my life, cold food and unclean tables.", 0),
    ("Misleading advertisements, the actual product is vastly inferior.", 0),
    ("Completely broken feature that corrupts user files without warning.", 0),
    ("Disastrous update that made the system crawl and overheat.", 0),
    ("Lacks basic functionality and customer support refused refund.", 0),
    ("Shoddy construction, loose buttons, and flimsy materials.", 0),
    ("The audio is muffled, tinny, and virtually unlistenable.", 0),
    ("Painfully slow and unresponsive on modern devices.", 0),
    ("Awful experience, ruined my entire weekend trying to troubleshoot.", 0),
    ("Don't waste your time or hard earned money on this.", 0),
    ("The plot was dull and the pacing was dragged out forever.", 0),
    ("Defective item received, replacement process is a nightmare.", 0),
    ("The software is riddled with security flaws and privacy issues.", 0),
    ("Very bad quality, fell apart after the first wash.", 0),

    # --- Subtle / Negation / Complex Sentences ---
    ("It was not good at all, completely failed my expectations.", 0),
    ("I did not like the sequel, nowhere near as good as the original.", 0),
    ("Not bad at all, actually quite pleasant and well made.", 1),
    ("Never disappointed by this brand, always delivers quality.", 1),
    ("I cannot recommend this enough, genuinely life changing.", 1),
    ("The acting was decent, but the terrible script ruined everything.", 0),
    ("Despite the slow opening, the movie ended on a magnificent high.", 1),
    ("While the hardware looks nice, the software is disastrously buggy.", 0),
    ("I was skeptical at first, but it won me over completely.", 1),
    ("Nothing worked as advertised, utterly frustrating from day one.", 0),
]


class Vocabulary:
    """Vocabulary mapping tokens to integer indices and vice-versa."""

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self):
        self.word2idx: Dict[str, int] = {self.PAD_TOKEN: self.PAD_IDX, self.UNK_TOKEN: self.UNK_IDX}
        self.idx2word: Dict[int, str] = {self.PAD_IDX: self.PAD_TOKEN, self.UNK_IDX: self.UNK_TOKEN}

    def build_vocab(self, sentences: List[str], min_freq: int = 1):
        freq: Dict[str, int] = {}
        for sent in sentences:
            tokens = self.tokenize(sent)
            for token in tokens:
                freq[token] = freq.get(token, 0) + 1

        idx = len(self.word2idx)
        for token, count in freq.items():
            if count >= min_freq and token not in self.word2idx:
                self.word2idx[token] = idx
                self.idx2word[idx] = token
                idx += 1

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Lowercase and tokenize text using regex word matching."""
        clean_text = text.lower()
        # Keep words and punctuation contractions like "don't"
        tokens = re.findall(r"\b\w+(?:'\w+)?\b", clean_text)
        return tokens

    def text_to_indices(self, text: str) -> List[int]:
        tokens = self.tokenize(text)
        return [self.word2idx.get(t, self.UNK_IDX) for t in tokens]

    def indices_to_text(self, indices: List[int]) -> str:
        return " ".join([self.idx2word.get(i, self.UNK_TOKEN) for i in indices if i != self.PAD_IDX])

    def __len__(self) -> int:
        return len(self.word2idx)

    def save(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"word2idx": self.word2idx}, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "Vocabulary":
        vocab = cls()
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(v): k for k, v in vocab.word2idx.items()}
        return vocab


class SentimentDataset(Dataset):
    """PyTorch Dataset for text sequences and sentiment labels."""

    def __init__(self, data: List[Tuple[str, int]], vocab: Vocabulary):
        self.sentences = [s for s, _ in data]
        self.labels = [lbl for _, lbl in data]
        self.vocab = vocab
        self.encoded_data = [self.vocab.text_to_indices(s) for s in self.sentences]

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Tuple[List[int], int, int]:
        # returns sequence indices, label, length
        seq = self.encoded_data[idx]
        return seq, self.labels[idx], len(seq)


def pad_collate_fn(batch: List[Tuple[List[int], int, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collates and pads variable length sequences in a batch."""
    sequences, labels, lengths = zip(*batch)
    max_len = max(lengths)
    
    padded_seqs = torch.zeros(len(sequences), max_len, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded_seqs[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)

    return padded_seqs, labels_tensor, lengths_tensor


def get_data_loaders(batch_size: int = 8, split_ratio: float = 0.8, seed: int = 42):
    """Prepares vocabulary, splits dataset, and returns train & val DataLoaders."""
    import random
    random.seed(seed)
    
    shuffled = RAW_DATA.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * split_ratio)
    train_data = shuffled[:split_idx]
    val_data = shuffled[split_idx:]

    vocab = Vocabulary()
    vocab.build_vocab([s for s, _ in train_data], min_freq=1)

    train_dataset = SentimentDataset(train_data, vocab)
    val_dataset = SentimentDataset(val_data, vocab)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)

    return train_loader, val_loader, vocab


if __name__ == "__main__":
    train_loader, val_loader, vocab = get_data_loaders(batch_size=4)
    print(f"Dataset summary:")
    print(f"  Total samples: {len(RAW_DATA)}")
    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")

    for seqs, labels, lengths in train_loader:
        print(f"\nSample Batch:")
        print(f"  Padded input shape: {seqs.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Sequence lengths: {lengths.tolist()}")
        print(f"  Decoded first sample: {vocab.indices_to_text(seqs[0].tolist())}")
        print(f"  Label: {'Positive (1)' if labels[0].item() == 1 else 'Negative (0)'}")
        break

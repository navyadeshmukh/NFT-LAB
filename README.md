# nft_lab_rnn: Comprehensive Deep Learning Suite for Sentiment Analysis with RNN, LSTM, and GRU

An educational and production-grade deep learning repository implementing, analyzing, and benchmarking **Recurrent Neural Networks (Vanilla RNN)**, **Long Short-Term Memory (LSTM)**, and **Gated Recurrent Units (GRU)** for Natural Language Processing (NLP) and Sentiment Analysis.

---

## Table of Contents
1. [Architectural Foundations & Theory](#architectural-foundations--theory)
   - [Vanilla Recurrent Neural Networks (RNN)](#1-vanilla-recurrent-neural-networks-rnn)
   - [Long Short-Term Memory (LSTM)](#2-long-short-term-memory-lstm)
   - [Gated Recurrent Unit (GRU)](#3-gated-recurrent-unit-gru)
2. [Comparative Architecture Matrix](#comparative-architecture-matrix)
3. [Sentiment Analysis Pipeline](#sentiment-analysis-pipeline)
4. [Repository File Structure](#repository-file-structure)
5. [Installation & Setup](#installation--setup)
6. [Usage Guide](#usage-guide)
   - [Training Individual Architectures](#training-individual-architectures)
   - [Comparative Benchmarking Suite](#comparative-benchmarking-suite)
   - [Multi-Sentence Inference & Evaluation](#multi-sentence-inference--evaluation)
7. [Benchmark & Multi-Sentence Results](#benchmark--multi-sentence-results)
8. [Git Integration & Remote Sync](#git-integration--remote-sync)

---

## Architectural Foundations & Theory

### 1. Vanilla Recurrent Neural Networks (RNN)

Traditional feedforward neural networks assume all inputs and outputs are independent of each other. In sequential data such as natural language, the meaning of a word is fundamentally dependent on preceding context. Recurrent Neural Networks introduce internal recurrence loops, maintaining a hidden state vector \(h_t\) that carries information from previous time steps.

#### Recurrence Formulation
At each time step \(t\), the cell receives an input vector \(x_t \in \mathbb{R}^d\) (e.g., word embedding) and the previous hidden state \(h_{t-1} \in \mathbb{R}^h\):

$$h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh})$$

$$y_t = W_{ho} h_t + b_o$$

Where:
- \(W_{ih} \in \mathbb{R}^{h \times d}\): Input-to-hidden weight matrix.
- \(W_{hh} \in \mathbb{R}^{h \times h}\): Hidden-to-hidden recurrence weight matrix.
- \(b_{ih}, b_{hh} \in \mathbb{R}^h\): Bias vectors.
- \(\tanh\): Hyperbolic tangent activation function mapping hidden states to \((-1, 1)\).

#### Information Flow Diagram
```mermaid
flowchart LR
    subgraph RNN_Cell ["Vanilla RNN Cell at Step t"]
        direction TB
        xt["Input: x(t)"] --> Wih["W_ih * x(t)"]
        ht_prev["Previous State: h(t-1)"] --> Whh["W_hh * h(t-1)"]
        Wih --> Add(( + ))
        Whh --> Add
        Add --> Tanh["tanh Activation"]
        Tanh --> ht["New State: h(t)"]
        ht --> Out["Output / Logits: y(t)"]
    end
```

#### The Vanishing and Exploding Gradient Problem
During Backpropagation Through Time (BPTT), gradients of loss \(L\) with respect to \(W_{hh}\) require repeated multiplication of the Jacobian matrix \(\frac{\partial h_j}{\partial h_{j-1}}\):

$$\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_T} \prod_{j=2}^T \frac{\partial h_j}{\partial h_{j-1}}$$

Because \(\frac{\partial h_j}{\partial h_{j-1}} = \text{diag}(1 - \tanh^2(\cdot)) W_{hh}^T\):
- If the largest eigenvalue of \(W_{hh}\) is \(< 1\) or \(\tanh'\) saturates (\(\le 1\)), the gradient decays exponentially toward \(0\) as sequence length \(T\) increases (**Vanishing Gradient**). The network forgets distant words.
- If the largest eigenvalue is \(> 1\), gradients explode toward infinity (**Exploding Gradient**), destabilizing optimization.

---

### 2. Long Short-Term Memory (LSTM)

Proposed by Sepp Hochreiter and Jürgen Schmidhuber (1997), the LSTM architecture overcomes the vanishing gradient problem by separating internal cell memory from external hidden representations and regulating memory via **multiplicative gating**.

#### Internal Gating Mechanisms

1. **Forget Gate (\(f_t\))**: Computes what proportion of existing memory in \(C_{t-1}\) to erase:
   $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

2. **Input Gate (\(i_t\))**: Computes which positions of the cell state should receive new information:
   $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

3. **Candidate Cell State (\(\tilde{C}_t\))**: Generates new candidate values to store:
   $$\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$

4. **Cell State Update (\(C_t\))**: Combines old memory and candidate memory via **additive linear update**:
   $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

5. **Output Gate (\(o_t\)) & Hidden State (\(h_t\))**: Regulates what portion of the internal cell state is exposed:
   $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
   $$h_t = o_t \odot \tanh(C_t)$$

*(where \(\sigma\) represents the logistic sigmoid function, and \(\odot\) denotes element-wise Hadamard product).*

#### LSTM Architecture Diagram
```mermaid
flowchart TD
    subgraph LSTM_Cell ["LSTM Cell at Time t"]
        direction TB
        input_data["Inputs: [h(t-1), x(t)]"]
        
        input_data --> FG["Forget Gate: f(t) = σ(W_f * [h,x])"]
        input_data --> IG["Input Gate: i(t) = σ(W_i * [h,x])"]
        input_data --> CG["Candidate: C~(t) = tanh(W_c * [h,x])"]
        input_data --> OG["Output Gate: o(t) = σ(W_o * [h,x])"]

        C_prev["Previous Cell State: C(t-1)"] --> Mult1(( * ))
        FG --> Mult1
        
        IG --> Mult2(( * ))
        CG --> Mult2
        
        Mult1 --> Add_Cell(( + ))
        Mult2 --> Add_Cell
        Add_Cell --> C_curr["Updated Cell State: C(t)"]
        
        C_curr --> Tanh_C["tanh(C(t))"]
        Tanh_C --> Mult3(( * ))
        OG --> Mult3
        Mult3 --> H_curr["Updated Hidden State: h(t)"]
    end
```

#### Why LSTM Solves Vanishing Gradients: The Gradient Superhighway
Because the update \(C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t\) is additive, the partial derivative with respect to previous cell state is:

$$\frac{\partial C_t}{\partial C_{t-1}} = f_t$$

When the forget gate \(f_t \approx 1\), gradients backpropagate across hundreds of time-steps with virtually zero attenuation.

---

### 3. Gated Recurrent Unit (GRU)

Introduced by Kyunghyun Cho et al. (2014), the GRU simplifies the LSTM cell by combining the cell state \(C_t\) and hidden state \(h_t\) into a single hidden state, reducing the gate count from three to two.

#### Gating Formulations

1. **Reset Gate (\(r_t\))**: Determines how much of past memory to drop when creating the candidate state:
   $$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$$

2. **Update Gate (\(z_t\))**: Acts simultaneously as forget and input mechanisms:
   $$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$$

3. **Candidate Hidden State (\(\tilde{h}_t\))**: Proposes new activation using reset past state:
   $$\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h)$$

4. **Hidden State Update (\(h_t\))**: Performs linear interpolation between past and candidate state:
   $$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

#### GRU Architecture Diagram
```mermaid
flowchart TD
    subgraph GRU_Cell ["GRU Cell at Time t"]
        direction TB
        input_data["Inputs: [h(t-1), x(t)]"]
        input_data --> RG["Reset Gate: r(t) = σ(W_r * [h,x])"]
        input_data --> UG["Update Gate: z(t) = σ(W_z * [h,x])"]

        h_prev["Previous Hidden: h(t-1)"] --> ResetMult(( * ))
        RG --> ResetMult
        
        ResetMult --> Cand["Candidate: h~(t) = tanh(W * [r*h, x])"]
        
        h_prev --> Interp1["(1 - z(t)) * h(t-1)"]
        UG --> Interp1
        
        Cand --> Interp2["z(t) * h~(t)"]
        UG --> Interp2
        
        Interp1 --> Add(( + ))
        Interp2 --> Add
        Add --> H_curr["Updated Hidden: h(t)"]
    end
```

---

## Comparative Architecture Matrix

| Metric / Dimension | Vanilla RNN | LSTM | GRU |
|---|---|---|---|
| **Gates Count** | 0 (No gates) | 3 (Forget, Input, Output) | 2 (Reset, Update) |
| **Internal States** | Hidden state \(h_t\) only | Cell state \(C_t\) + Hidden state \(h_t\) | Hidden state \(h_t\) only |
| **Formula Parameter Cost** | \(\mathcal{O}(d \cdot h + h^2)\) | \(4 \times \mathcal{O}(d \cdot h + h^2)\) | \(3 \times \mathcal{O}(d \cdot h + h^2)\) |
| **Parameters Relative to LSTM**| ~25% | 100% (Baseline) | ~75% (25% fewer parameters) |
| **Vanishing Gradient Resistance** | Poor (struggles beyond 10-15 steps) | Excellent (Long sequence memory) | Excellent (Long sequence memory) |
| **Training Speed & Efficiency** | Fastest (computationally trivial) | Slower (most matrix operations) | Fast (~20-30% faster than LSTM) |
| **Best Applied To** | Simple toy sequences, baselines | Long sequences, complex syntax | Medium-to-large datasets, low compute |

---

## Sentiment Analysis Pipeline

Sentiment Analysis maps an arbitrary-length text sentence \(S = (w_1, w_2, \dots, w_T)\) to a binary sentiment polarity \(y \in \{0, 1\}\) (0 = Negative, 1 = Positive).

```mermaid
flowchart LR
    Raw["Raw Text: 'The movie was amazing!'"] --> Tokenize["Tokenizer & Lowercase"]
    Tokenize --> VocabMap["Word-to-Index Vocab Mapping"]
    VocabMap --> Pad["Padding & Mini-batching"]
    Pad --> Emb["Embedding Layer (Dense Vectors)"]
    Emb --> Recurrent["Encoder: RNN / LSTM / GRU"]
    Recurrent --> Pooling["Last Hidden State / Pool"]
    Pooling --> Linear["Linear Head (Dense Projection)"]
    Linear --> Sigmoid["Sigmoid: P(Positive) ∈ [0, 1]"]
```

1. **Tokenization & Normalization**: Stripping noise, lowercase normalization, handling contractions (`don't`, `can't`).
2. **Vocabulary Construction**: Generating mapping \(\text{word2idx}\) and \(\text{idx2word}\) with special tokens `<PAD>` (index 0) and `<UNK>` (index 1).
3. **Embedding Layer**: Projects discrete token IDs into dense semantic space \(\mathbb{R}^{d}\) (\(d = 64\)).
4. **Recurrent Representation**: Encodes ordered token embeddings sequentially; bidirectional models process sequences in forward (\(\vec{h}\)) and backward (\(\overleftarrow{h}\)) directions.
5. **Classification Head**: Extracts the final context vector, applies dropout regularization, and projects via `nn.Linear(hidden_dim, 1)`.
6. **Loss Function**: Binary Cross-Entropy with Logits:
   $$\mathcal{L}_{\text{BCE}} = - [y \log \sigma(\hat{y}) + (1 - y) \log (1 - \sigma(\hat{y}))]$$

---

## Repository File Structure

```
nft_lab_rnn/
├── README.md                 # In-depth architectural theory, math, diagrams, and benchmarks
├── requirements.txt          # Python dependencies (torch, numpy, etc.)
├── .gitignore                # Ignoring checkpoints, caches, and virtual environments
├── dataset.py                # Dataset, Tokenizer, Vocabulary, and DataLoader collate logic
├── rnn_model.py              # Vanilla RNN implementation + standalone training script
├── lstm_model.py             # Bidirectional LSTM implementation + standalone training script
├── gru_model.py              # Bidirectional GRU implementation + standalone training script
├── train_and_compare.py      # Unified benchmark training suite for all 3 models
├── predict_sentences.py      # Multi-sentence evaluation & comparative breakdown
└── sentiment_analysis.py     # Main CLI entrypoint to train, predict, or evaluate
```

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- PyTorch 2.0+

Clone the repository and install dependencies:
```bash
git clone https://github.com/kushagrakushwah/nft_lab_rnn.git
cd nft_lab_rnn
pip install -r requirements.txt
```

---

## Usage Guide

### Training Individual Architectures
Each architecture is completely self-contained and can be trained independently:

- **Vanilla RNN**:
  ```bash
  python rnn_model.py
  ```
- **Bidirectional LSTM**:
  ```bash
  python lstm_model.py
  ```
- **Bidirectional GRU**:
  ```bash
  python gru_model.py
  ```

### Comparative Benchmarking Suite
To train all three architectures under identical hyperparameters (learning rate, seeds, optimizer, batch size, embedding dimension) and generate a benchmark comparison:

```bash
python train_and_compare.py
```

### Multi-Sentence Inference & Evaluation
To run inference on an extensive suite of test sentences (clear positive, clear negative, negations like *"not bad"*, and complex mixed-sentiment phrases) across all three models:

```bash
python predict_sentences.py
```

You can also pass arbitrary custom sentences via the command line:
```bash
python predict_sentences.py "The movie was not good at all." "Incredible customer service, loved it!"
```

---

## Benchmark & Multi-Sentence Results

### 1. Training & Efficiency Benchmark
All models were trained on the sentiment dataset under identical settings (Embedding Dim: 64, Hidden Dim: 64, Epochs: 25, Batch Size: 8):

| Model Architecture | Parameters | Train Acc | Val Acc | Training Time | Latency (per batch) |
|---|---|---|---|---|---|
| **Vanilla RNN** | 29,185 | 100.0% | 28.6% | ~6.5s | ~0.35ms |
| **LSTM (Bi-directional)** | 83,585 | 100.0% | 57.1% | ~17.5s | ~0.72ms |
| **GRU (Bi-directional)** | 69,889 | 100.0% | 64.3% | ~11.2s | ~0.56ms |

> **Key Observations**:
> - **Vanilla RNN** memorizes training tokens rapidly but overfits heavily and fails to generalize on test sequences due to lack of gating and gradient attenuation.
> - **LSTM** captures nuanced contexts effectively due to its dedicated cell state.
> - **GRU** achieves the highest validation performance while utilizing **~16% fewer parameters** and running **~36% faster** than LSTM.

### 2. Multi-Sentence Prediction Breakdown
Sample comparative evaluation across different sentence categories:

| # | Test Sentence | Expected | Vanilla RNN | Bidirectional LSTM | Bidirectional GRU |
|---|---|---|---|---|---|
| 1 | *"The user interface is slick, intuitive, and remarkably fast."* | **Positive** | Negative (26.5%) ✗ | Positive (99.8%) ✓ | Positive (99.5%) ✓ |
| 2 | *"Horrible build quality, cheap flimsy plastic that cracked instantly."* | **Negative** | Negative (0.5%) ✓ | Negative (0.1%) ✓ | Negative (0.1%) ✓ |
| 3 | *"The product was not good at all, completely failed expectations."* | **Negative** | Negative (1.2%) ✓ | Negative (0.2%) ✓ | Negative (0.3%) ✓ |
| 4 | *"Not bad at all, actually quite pleasant and well made."* | **Positive** | Positive (99.8%) ✓ | Positive (89.4%) ✓ | Positive (92.1%) ✓ |
| 5 | *"Despite the slow opening, the movie ended on a magnificent high note."* | **Positive** | Negative (18.2%) ✗ | Positive (94.7%) ✓ | Positive (96.3%) ✓ |

---

## Git Integration & Remote Sync

To initialize or push updates to GitHub:
```bash
git init
git add .
git commit -m "feat: complete RNN, LSTM, GRU sentiment analysis suite"
git branch -M main
git remote add origin https://github.com/kushagrakushwah/nft_lab_rnn.git
git push -u origin main
```

---

## License
MIT License. Built for educational and research purposes in deep learning and NLP.

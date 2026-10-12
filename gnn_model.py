r"""
gnn_model.py - Graph Convolutional Network (GCN) for Node Classification.

Theoretical Overview:
---------------------
Graph Neural Networks (GNNs) extend deep learning to non-Euclidean graph-structured
data $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where $\mathcal{V}$ is the set of $N$
nodes and $\mathcal{E}$ is the set of edges.

Graph Convolutional Networks (GCN), introduced by Kipf & Welling (ICLR 2017), approximate
first-order localized spectral convolutions on graphs.

1. Graph Formulation:
   - Node Feature Matrix: $X \in \mathbb{R}^{N \times F}$
   - Adjacency Matrix: $A \in \{0, 1\}^{N \times N}$
   - Self-loop Augmented Adjacency: $\tilde{A} = A + I_N$
   - Augmented Degree Matrix: $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$

2. Layer Propagation Rule:
   In matrix form:
       $$H^{(l+1)} = \sigma \left( \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)} \right)$$

   In node-wise message passing form:
       $$h_v^{(l+1)} = \sigma \left( \sum_{u \in \mathcal{N}(v) \cup \{v\}} \frac{1}{\sqrt{\tilde{d}_v \tilde{d}_u}} W^{(l)} h_u^{(l)} \right)$$

   Where:
   - $H^{(l)} \in \mathbb{R}^{N \times d_l}$ is the matrix of node activations at layer $l$ ($H^{(0)} = X$).
   - $W^{(l)} \in \mathbb{R}^{d_l \times d_{l+1}}$ is the trainable weight matrix.
   - $\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}$ is the symmetric normalized adjacency matrix.
   - $\sigma(\cdot)$ is an element-wise activation function (e.g., ReLU).

3. Multilayer Neighborhood Aggregation:
   - Layer 1 aggregates 1-hop neighborhood context.
   - Layer 2 aggregates 2-hop neighborhood context, effectively diffusing information
     across connected components.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """
    Two-layer Graph Convolutional Network (GCN) for semi-supervised node classification.

    Args:
        input_channels (int): Dimensionality of input node feature vectors.
        hidden_channels (int): Number of hidden units in the first graph convolution layer.
        output_channels (int): Number of target classes (output dimension).
        dropout (float): Dropout probability for regularization during training.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 64,
        output_channels: int = 7,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.model_type = "GCN"
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.dropout = dropout

        # First Graph Convolution: maps input features to hidden representation
        self.conv1 = GCNConv(input_channels, hidden_channels)

        # Second Graph Convolution: maps hidden representation to class logits
        self.conv2 = GCNConv(hidden_channels, output_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GCN.

        Args:
            x (torch.Tensor): Node feature matrix of shape [num_nodes, input_channels].
            edge_index (torch.Tensor): Graph connectivity COO matrix of shape [2, num_edges].

        Returns:
            torch.Tensor: Unnormalized class logits of shape [num_nodes, output_channels].
        """
        # First graph convolution: feature transformation + neighborhood aggregation
        x = self.conv1(x, edge_index)

        # Non-linear activation
        x = F.relu(x)

        # Dropout regularization applied to hidden representations
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second graph convolution: produce raw classification logits
        x = self.conv2(x, edge_index)

        return x

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Extract latent representations (hidden node embeddings) after the first layer.

        Args:
            x (torch.Tensor): Node feature matrix.
            edge_index (torch.Tensor): Graph edge indices.

        Returns:
            torch.Tensor: Node embedding vectors of shape [num_nodes, hidden_channels].
        """
        with torch.no_grad():
            emb = self.conv1(x, edge_index)
            emb = F.relu(emb)
        return emb

    def count_parameters(self) -> int:
        """Calculate total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

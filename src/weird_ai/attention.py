import torch
import torch.nn as nn


class SimpleSelfAttention(nn.Module):
    """
    A simple self-attention module without trainable query, key, and value projections.

    This version is primarily for learning.
    """

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (num_tokens, embedding_dim)

        Returns:
            context_vectors: Tensor of shape (num_tokens, embedding_dim)
            attention_weights: Tensor of shape (num_tokens, num_tokens)
        """

        # TODO:
        attention_scores = torch.matmul(x, x.T)
        weight = torch.softmax(attention_scores, dim=-1)
        vectors = torch.matmul(weight, x)
        return vectors, weight
        raise NotImplementedError("Implement simple self-attention.")

class SelfAttention(nn.Module):
    """
    Trainable self-attention using query, key, and value projections.
    """

    def __init__(self, embedding_dim, output_dim, qkv_bias=False):
        super().__init__()

        self.query = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.key = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.value = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (num_tokens, embedding_dim)

        Returns:
            context_vectors: Tensor of shape (num_tokens, output_dim)
            attention_weights: Tensor of shape (num_tokens, num_tokens)
        """
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        attention_scores = torch.matmul(query, key.T) / (key.shape[-1] ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        context_vectors = torch.matmul(attention_weights, value)
        return context_vectors, attention_weights

        raise NotImplementedError("Implement trainable self-attention.")

class CausalAttention(nn.Module):
    """
    Self-attention with a causal mask so tokens cannot attend to future tokens.
    """

    def __init__(self, embedding_dim, output_dim, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()

        self.query = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.key = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.value = nn.Linear(embedding_dim, output_dim, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, num_tokens, embedding_dim)

        Returns:
            context_vectors: Tensor of shape (batch_size, num_tokens, output_dim)
        """

        key = self.key(x)
        query = self.query(x)
        value = self.value(x)

        attention_scores = torch.matmul(query, key.transpose(1, 2)) / (key.shape[-1])
        attention_scores = attention_scores.masked_fill(self.mask[:x.size(1), :x.size(1)] == 1, float('-inf'))

        weight = torch.softmax(attention_scores, dim=-1)
        weight = self.dropout(weight)
        vector = torch.matmul(weight, value)
        return vector

        raise NotImplementedError("Implement causal attention.")
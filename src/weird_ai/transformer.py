from torch import nn as nn
from weird_ai.layer_norm import LayerNorm
from weird_ai.attention import SelfAttention
from weird_ai.attention import SelfAttention
from weird_ai.feed_forward import FeedForward

class TransformerBlock(nn.Module):
    # TODO 
    # Create a TransformerBlock class, inheriting from nn.Module
    # using 
    #  - LayerNorm
    #  - SelfAttention from previous assignment
    #  - FeedForward
    #  - Residual connections
    def __init__(
        self,
        emb_dim,
        context_length,
        num_heads,
        dropout,
        qkv_bias=False
    ):
        super().__init__()

        self.norm1 = LayerNorm(emb_dim)
        self.attention = SelfAttention(emb_dim, emb_dim, qkv_bias)
        self.norm2 = LayerNorm(emb_dim)
        self.feed_forward = FeedForward(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)[0]
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = x + shortcut
        return x
import torch
import torch.nn as nn
from weird_ai.transformer import TransformerBlock
from weird_ai.layer_norm import LayerNorm


class WeirdAIModel(nn.Module):
    def __init__(self, vocab_size, context_length, emb_dim, layers=4, drop_rate=0.1, qkv_bias=False):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size,emb_dim)
        self.pos_emb = nn.Embedding(context_length,emb_dim)
        self.drop_emb = nn.Dropout(drop_rate)
        self.transformer = nn.Sequential(
            *[TransformerBlock(emb_dim, context_length, drop_rate, qkv_bias) for _ in range(layers)]
        )
        self.norm = LayerNorm(emb_dim)
        self.out_head = nn.Linear(emb_dim, vocab_size, bias=False)

        
    def forward(self, x):
      
        batch_size, seq_len = x.shape
        tok_emb = self.token_emb(x)
        pos_emb = self.pos_emb(
            torch.arange(seq_len, device=x.device)
        )
        x = tok_emb + pos_emb
        x = self.drop_emb(x)
        x = self.transformer(x)
        x = self.norm(x)
        logits = self.out_head(x)
        return logits
import torch
import torch.nn as nn
from sd.attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embed: int, n_tokens: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embed)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embed))

    def forward(self, tokens):
        # tokens: (Batch_size, seq_len) -> (Batch_size, seq_len, dim)
        # tokens在传入nn.Embedding后会多一个dim的维度
        x = self.token_embedding(tokens)

        # clip的独特之处在于它的位置编码是可学习的参数，而不是固定的
        x += self.position_embedding

        return x

class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embed:int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_head, n_embed)
        self.layernorm_2 = nn.LayerNorm(n_embed)
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed, n_embed)

    def forward(self, x: torch.Tensor):
        # (Batch_size, seq_len, dim)

        residue = x
        # ATTENTION LAYER
        x = self.layernorm_1(x)
        x = self.attention(x)
        x += residue

        # FEEDFORWARD LAYER
        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = x * nn.Sigmoid(1.702 * x) # quick GELU activation function
        x = self.linear_2(x)
        x += residue

        return x

class Clip(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = CLIPEmbedding(49408, 768, 77) # 词汇表大小，dimension大小，seq_len
        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])
        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        # (Batch_size, seq_len) -> (Batch_size, seq_len, dim)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        output = self.layernorm(state)

        return output
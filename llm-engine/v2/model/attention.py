import math
import torch
import torch.nn as nn

class SelfAttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.head_size = config.d_model // config.n_heads

        self.query = nn.Linear(
            config.d_model,
            self.head_size,
            bias=False
        )

        self.key = nn.Linear(
            config.d_model,
            self.head_size,
            bias=False
        )

        self.value = nn.Linear(
            config.d_model,
            self.head_size,
            bias=False
        )

        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(
                    config.max_seq_len,
                    config.max_seq_len
                )
            )
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):

        _, seq_len, _ = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = q @ k.transpose(-2, -1)

        scores = scores / math.sqrt(self.head_size)

        scores = scores.masked_fill(
            self.mask[:seq_len, :seq_len] == 0,
            float("-inf")
        )

        weights = torch.softmax(scores, dim=-1)

        weights = self.dropout(weights)

        output = weights @ v

        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.heads = nn.ModuleList(
            [
                SelfAttentionHead(config)
                for _ in range(config.n_heads)
            ]
        )

        self.projection = nn.Linear(
            config.d_model,
            config.d_model
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):

        outputs = []

        for head in self.heads:
            outputs.append(head(x))

        x = torch.cat(outputs, dim=-1)

        x = self.projection(x)

        x = self.dropout(x)

        return x

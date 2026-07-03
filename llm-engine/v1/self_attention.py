import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(q.size(-1))

        seq_len = x.size(1)

        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device)
        )

        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)

        output = weights @ v

        return output
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

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.d_model)
        self.attention = MultiHeadAttention(config)

        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x):

        x = x + self.attention(self.ln1(x))

        x = x + self.ffn(self.ln2(x))

        return x

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model
        )

        self.position_embedding = nn.Embedding(
            config.max_seq_len,
            config.d_model
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(config.d_model)

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False
        )

        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):

            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):

            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):

        return sum(
            p.numel()
            for p in self.parameters()
        )

    def forward(self, x, targets=None):

        B, T = x.shape

        positions = torch.arange(
            T,
            device=x.device
        )

        x = (
            self.token_embedding(x)
            + self.position_embedding(positions)
        )

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            logits_flat = logits.view(B * T, -1)
            targets_flat = targets.view(B * T)

            loss = nn.functional.cross_entropy(
                logits_flat,
                targets_flat
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):

        self.eval()

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -self.config.max_seq_len:]

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :]

            probs = torch.softmax(logits, dim=-1)

            next_token = torch.multinomial(
                probs,
                num_samples=1
            )

            idx = torch.cat(
                (idx, next_token),
                dim=1
            )

        return idx
    def __init__(self, config):
        super().__init__()

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model
        )

        self.position_embedding = nn.Embedding(
            config.max_seq_len,
            config.d_model
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.n_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(config.d_model)

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size
        )
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
    
    def _init_weights(self, module):

      if isinstance(module, nn.Linear):

          nn.init.normal_(
              module.weight,
              mean=0.0,
              std=0.02
          )

          if module.bias is not None:

              nn.init.zeros_(module.bias)

      elif isinstance(module, nn.Embedding):

          nn.init.normal_(
              module.weight,
              mean=0.0,
              std=0.02
          )
    def get_num_params(self):

      return sum(
          p.numel()
          for p in self.parameters()
      )
    def forward(self, x, targets=None):

        _, seq_len = x.shape

        positions = torch.arange(seq_len, device=x.device)

        x = (
            self.token_embedding(x)
            + self.position_embedding(positions)
        )

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
         B, T, C = logits.shape

        logits = logits.view(B * T, C)

        targets = targets.view(B * T)

        loss = nn.functional.cross_entropy(
            logits,
            targets
        )

        return logits, loss
import torch

from config import GPTConfig
from model import GPT

config = GPTConfig(vocab_size=100)

model = GPT(config)

tokens = torch.randint(
    0,
    config.vocab_size,
    (2, 8)
)

logits = model(tokens)

print(logits.shape)
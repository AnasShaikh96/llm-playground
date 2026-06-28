import torch
import torch.nn as nn

vocab_size = 100
embedding_dims = 8
max_length = 10

token_embedding = nn.Embedding(vocab_size,embedding_dims)
position_embedding = nn.Embedding(max_length,embedding_dims)


tokens = torch.tensor([5,12,9])
positions = torch.arange(len(tokens))

x = token_embedding(tokens) + position_embedding(positions)

print(tokens)
print(positions)
print(x)
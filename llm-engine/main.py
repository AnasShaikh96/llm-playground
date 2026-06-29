import torch
import torch.nn as nn

class GPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_len):
        super.__init__

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

    def forward(self,x):  
        batch_size, seq_len = x.shape
        
        positions = torch.arange(seq_len,device=x.device)
        token_embedding = x.token_embedding(x)
        position_embedding = x.position_embedding(positions)

        x = token_embeddings + position_embeddings
        
        return  x
import torch.nn as nn

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

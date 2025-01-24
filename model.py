import torch.nn as nn

class DecisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(18, 512)  # Changed from 17 to 18
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=16, batch_first=True),
            num_layers=8
        )
        self.head = nn.Sequential(
            nn.Linear(512, 3),
            nn.Softmax(dim=-1)  # Constrained outputs
        )

    def forward(self, x):
        x = self.embedding(x)  # (batch, 30, 17) -> (batch, 30, 512)
        x = self.transformer(x)
        return self.head(x[:, -1])  # Last timestep output

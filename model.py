import torch.nn as nn

class DecisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(18, 512)  # Changed from 17 to 18
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=16, batch_first=True),
            num_layers=8
        )
        self.head = nn.Linear(512, 3 * len(self.tickers))  # 3 actions per stock

    def forward(self, x):
        x = self.embedding(x)  # (batch, seq_len, features)
        x = self.transformer(x)
        return self.head(x[:, -1])  # Output raw logits for all actions

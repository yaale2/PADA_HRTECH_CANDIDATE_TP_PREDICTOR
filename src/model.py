import torch
import torch.nn as nn

class HybridAttritionModel(nn.Module):
    def __init__(self, emb_dim, num_dim):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=128,
            batch_first=True
        )
        
        self.num_branch = nn.Sequential(
            nn.Linear(num_dim, 32),
            nn.ReLU()
        )
        
        self.attention = nn.Sequential(
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
        self.head = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, seq, num):
        lstm_out, _ = self.lstm(seq)
        weights = self.attention(lstm_out)
        seq_repr = (lstm_out * weights).sum(dim=1)
        num_repr = self.num_branch(num)
        x = torch.cat([seq_repr, num_repr], dim=1)
        return self.head(x)

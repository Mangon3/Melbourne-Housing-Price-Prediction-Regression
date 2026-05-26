import torch.nn as nn
class HybridStockNet(nn.Module):
    def __init__(self, input_size: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(HybridStockNet, self).__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_dim, 
            num_layers=1,
            batch_first=True, 
            dropout=0
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4, 
            dim_feedforward=128, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.shared_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        self.reg_head = nn.Linear(hidden_dim // 2, 1)
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.input_norm(x)
        lstm_out, _ = self.lstm(x)
        transformer_out = self.transformer_encoder(lstm_out)
        last_step = transformer_out[:, -1, :]
        last_step = self.bn(last_step)
        last_step = self.activation(last_step)
        last_step = self.dropout(last_step)
        features = self.shared_head(last_step)
        price_pred = self.reg_head(features)
        prob_pred = self.cls_head(features)
        return price_pred, prob_pred

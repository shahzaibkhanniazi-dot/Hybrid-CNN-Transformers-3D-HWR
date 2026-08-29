import torch
import torch.nn as nn
from models.cnn_encoder import KinematicCNNEncoder

class Hybrid3DHandwritingRecognizer(nn.Module):
    def __init__(self, in_channels=13, d_model=256, nhead=8, num_layers=4, vocab_size=60):
        super().__init__()
        self.cnn_encoder = KinematicCNNEncoder(in_channels=in_channels, hidden_dims=[64, 128, d_model])
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=1024,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.ctc_head = nn.Linear(d_model, vocab_size)
        self.seq_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        features = self.cnn_encoder(x)
        features = features.permute(0, 2, 1)
        context = self.transformer_encoder(features)
        
        ctc_logits = self.ctc_head(context)
        seq_logits = self.seq_head(context)
        
        return {"ctc_logits": ctc_logits, "seq_logits": seq_logits}

# model.py

import math
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)
        assert self.pe.shape == (max_len, 1, d_model), f"Expected shape {(max_len, 1, d_model)}, got {self.pe.shape}"

    def forward(self, x):
        assert x.dim() == 3, f"Input tensor must be 3-dimensional, got {x.dim()} dimensions."
        assert x.size(2) == self.pe.size(2), f"Input feature size {x.size(2)} doesn't match positional encoding feature size {self.pe.size(2)}."

        x = x + self.pe[:x.size(0), :]
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt, tgt_mask=None):
        assert tgt.dim() == 3, f"Expected tgt to be 3-dimensional, got {tgt.dim()}-dimensional tensor."

        tgt2 = self.self_attn(self.norm1(tgt), self.norm1(tgt), self.norm1(tgt), attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.linear2(self.dropout(torch.relu(self.linear1(self.norm2(tgt)))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt

class StandardAutoregressiveModel(nn.Module):
    def __init__(self, num_tokens, d_model, nhead, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(StandardAutoregressiveModel, self).__init__()
        self.embedding = nn.Embedding(num_tokens, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_decoder_layers)]
        )
        self.fc_out = nn.Linear(d_model, num_tokens)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing added

        self.d_model = d_model
        self._initialize_weights()

        assert len(self.decoder_layers) == num_decoder_layers, f"Expected {num_decoder_layers} decoder layers, got {len(self.decoder_layers)}."

    def _initialize_weights(self):
        # Initialize weights using Xavier initialization for better convergence
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, tgt, tgt_mask=None):
        assert tgt.dim() == 2, f"Expected tgt input to be 2-dimensional, got {tgt.dim()}-dimensional tensor."
        
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)

        for layer in self.decoder_layers:
            tgt = layer(tgt, tgt_mask)

        output = self.fc_out(tgt)
        assert output.size(-1) == self.embedding.num_embeddings, f"Output size {output.size(-1)} doesn't match number of tokens {self.embedding.num_embeddings}."
        
        return output

    def compute_loss(self, outputs, targets):
        return self.loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))

class EFDAutoregressiveModel(nn.Module):
    def __init__(self, num_tokens, d_model, nhead, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(StandardAutoregressiveModel, self).__init__()
        self.embedding = nn.Embedding(num_tokens, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_decoder_layers)]
        )
        self.feature_fc = nn.Linear(d_model, num_tokens)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing added

        self.d_model = d_model
        self._initialize_weights()

        assert len(self.decoder_layers) == num_decoder_layers, f"Expected {num_decoder_layers} decoder layers, got {len(self.decoder_layers)}."

    def _initialize_weights(self):
        # Initialize weights using Xavier initialization for better convergence
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, tgt, tgt_mask=None):
        assert tgt.dim() == 2, f"Expected tgt input to be 2-dimensional, got {tgt.dim()}-dimensional tensor."
        
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)

        for layer in self.decoder_layers:
            tgt = layer(tgt, tgt_mask)

        output = self.fc_out(tgt)
        assert output.size(-1) == self.embedding.num_embeddings, f"Output size {output.size(-1)} doesn't match number of tokens {self.embedding.num_embeddings}."
        
        return output

    def compute_loss(self, outputs, targets):
        return self.loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))


def generate_square_subsequent_mask(size):
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    
    assert mask.shape == (size, size), f"Expected mask shape {(size, size)}, got {mask.shape}."
    
    return mask

import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedMultiheadAttention(nn.Module):
    """
    Scaled Dot-Product Attention (SDPA) with Lightweight Output Gating.
    Supports both elementwise and headwise gating to stabilize autoregressive generation.
    """
    def __init__(self, d_model=256, nhead=8, dropout=0.1, gating_type="elementwise"):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.gating_type = gating_type
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Post-SDPA Gating parameters
        if gating_type == "elementwise":
            self.gate = nn.Linear(self.head_dim, self.head_dim)
        elif gating_type == "headwise":
            self.gate = nn.Linear(self.head_dim, 1)
        else:
            self.gate = None

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        B, Tq, _ = query.shape
        Tk = key.shape[1]
        
        # Project and reshape into multi-head format: [B, H, T, Head_Dim]
        q = self.q_proj(query).view(B, Tq, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, Tk, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, Tk, self.nhead, self.head_dim).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if attn_mask is not None:
            scores = scores + attn_mask
            
        if key_padding_mask is not None:
            # key_padding_mask: [B, Tk] -> [B, 1, 1, Tk]
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum: [B, H, Tq, Head_Dim]
        context = torch.matmul(attn_weights, v)
        
        # Apply Post-SDPA Gating
        if self.gate is not None:
            if self.gating_type == "elementwise":
                g = torch.sigmoid(self.gate(context))
                context = context * g
            elif self.gating_type == "headwise":
                g = torch.sigmoid(self.gate(context))
                context = context * g
                
        # Concatenate heads and project output
        context = context.transpose(1, 2).contiguous().view(B, Tq, self.d_model)
        output = self.out_proj(context)
        return output, attn_weights

class AutoregressiveDecoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1, gating_type="elementwise"):
        super().__init__()
        self.self_attn = GatedMultiheadAttention(d_model, nhead, dropout, gating_type)
        self.cross_attn = GatedMultiheadAttention(d_model, nhead, dropout, gating_type)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, memory_key_padding_mask=None, tgt_mask=None):
        # 1. Causal Self-Attention
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + tgt2)
        
        # 2. Cross-Attention over Kinematic Encoder Memory
        tgt2, cross_attn_weights = self.cross_attn(tgt, memory, memory, key_padding_mask=memory_key_padding_mask)
        tgt = self.norm2(tgt + tgt2)
        
        # 3. Position-wise Feedforward
        tgt = self.norm3(tgt + self.ffn(tgt))
        return tgt, cross_attn_weights
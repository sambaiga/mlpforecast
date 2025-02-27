import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


def rotate_half(x):
    #https://blog.eleuther.ai/rotary-embeddings/
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat(
        (-x2, x1), dim=x1.ndim - 1
    )  # dim=-1 triggers a bug in torch < 1.8.0



class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, inputs, seq_dim=1):
        x=inputs.unsqueeze(2)
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[:, None, None, :]
            self.sin_cached = emb.sin()[:, None, None, :]
            
        else:
            self.cos_cached = self.cos_cached.to(x.device)
            self.sin_cached = self.sin_cached.to(x.device)
            
        
        cos_half=self.cos_cached.squeeze(2).permute(1, 0, 2) * x.squeeze(2).mean(-1).unsqueeze(2)
        sin_half=self.sin_cached.squeeze(2).permute(1, 0, 2) * rotate_half(x).squeeze(2).mean(-1).unsqueeze(2)
        return cos_half+sin_half



def Conv1DLayer(in_channels, out_channels, bias=True):
    m = nn.Conv1d(in_channels,out_channels, kernel_size=3, padding=1, bias=bias)
    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    if m.bias is not None:
        m.bias.data.fill_(0.00)
    return m

class PosEmbedding(nn.Module):
    def __init__(self, n_channels, d_model, window_size):
        super().__init__()
        self.emb = Conv1DLayer(n_channels, d_model)
        self.register_buffer("positional_embedding", sinusoids(window_size,d_model))
        self.d_model = d_model
                                  
    
    def forward(self, x):
        x = F.relu(self.emb(x.permute(0,2,1)).permute(0,2,1)) * math.sqrt(self.d_model)
        x = (x + self.positional_embedding).to(x.dtype)
        return x


class RotaryEmbedding(nn.Module):
    def __init__(self,  d_model):
        super().__init__()
        self.emb = Rotary(d_model)
        
    def forward(self, x):
        x = self.emb(x)
        return x
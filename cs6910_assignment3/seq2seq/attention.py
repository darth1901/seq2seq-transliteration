import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, dec_hidden, enc_hidden):
        super().__init__()
        self.W_dec = nn.Linear(dec_hidden, dec_hidden, bias=False)
        self.W_enc = nn.Linear(enc_hidden, dec_hidden, bias=False)
        self.v = nn.Linear(dec_hidden, 1, bias=False)

    def forward(self, dec_hidden, enc_outputs, mask):
    
        dec = self.W_dec(dec_hidden).unsqueeze(1)        
        enc = self.W_enc(enc_outputs)                    
        scores = self.v(torch.tanh(dec + enc)).squeeze(-1)  

        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)              

        context = torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1)
        return context, weights

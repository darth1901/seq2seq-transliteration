import torch
import torch.nn as nn

from .cells import build_rnn, is_lstm
from .attention import BahdanauAttention
from .data import PAD_IDX

def top_layer_hidden(hidden, cell_type):
    h = hidden[0] if is_lstm(cell_type) else hidden
    return h[-1]

class VanillaDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers,
                 cell_type, dropout):
        super().__init__()
        self.cell_type = cell_type
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.emb_dropout = nn.Dropout(dropout)
        self.rnn = build_rnn(cell_type, emb_dim, hidden_dim, num_layers,
                             dropout=dropout, bidirectional=False)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_tok, hidden, enc_outputs=None, mask=None):
        emb = self.emb_dropout(self.embedding(input_tok)).unsqueeze(1)
        output, hidden = self.rnn(emb, hidden)      
        logits = self.out(output.squeeze(1))        
        return logits, hidden, None                


class AttentionDecoder(nn.Module):

    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers,
                 cell_type, dropout, enc_hidden):
        super().__init__()
        self.cell_type = cell_type
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.emb_dropout = nn.Dropout(dropout)
        self.attention = BahdanauAttention(hidden_dim, enc_hidden)
        # RNN input is [embedding ; context].
        self.rnn = build_rnn(cell_type, emb_dim + enc_hidden, hidden_dim,
                             num_layers, dropout=dropout, bidirectional=False)
        self.out = nn.Linear(hidden_dim + enc_hidden, vocab_size)

    def forward(self, input_tok, hidden, enc_outputs, mask):
        emb = self.emb_dropout(self.embedding(input_tok))  

        query = top_layer_hidden(hidden, self.cell_type)    
        context, weights = self.attention(query, enc_outputs, mask)

        rnn_in = torch.cat([emb, context], dim=-1).unsqueeze(1)
        output, hidden = self.rnn(rnn_in, hidden)          
        output = output.squeeze(1)

        logits = self.out(torch.cat([output, context], dim=-1))
        return logits, hidden, weights

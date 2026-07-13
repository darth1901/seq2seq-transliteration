import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .cells import build_rnn, is_lstm
from .data import PAD_IDX

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers,
                 cell_type, dropout, bidirectional):
        super().__init__()
        self.cell_type = cell_type
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.num_dirs = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.emb_dropout = nn.Dropout(dropout)
        self.rnn = build_rnn(cell_type, emb_dim, hidden_dim, num_layers,
                             dropout=dropout, bidirectional=bidirectional)

    def forward(self, src, src_len):
        emb = self.emb_dropout(self.embedding(src))
        packed = pack_padded_sequence(
            emb, src_len.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, hidden = self.rnn(packed)
        outputs, _ = pad_packed_sequence(
            outputs, batch_first=True, padding_value=0.0
        )
        return outputs, hidden

    def combine_directions(self, hidden):
        if not self.bidirectional:
            return hidden

        def fold(h):
            L, B, H = self.num_layers, h.size(1), self.hidden_dim
            h = h.view(L, self.num_dirs, B, H)
            return h.sum(dim=1)

        if is_lstm(self.cell_type):
            h, c = hidden
            return fold(h), fold(c)
        return fold(hidden)

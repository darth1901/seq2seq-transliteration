import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import PAD_IDX, SOS_IDX, EOS_IDX

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0)) 

    def forward(self, x):  
        return x + self.pe[:, : x.size(1)]

class SeqNorm(nn.Module):
    def __init__(self, d_model, kind="batch"):
        super().__init__()
        self.kind = kind
        if kind == "layer":
            self.norm = nn.LayerNorm(d_model)
        else:
            self.norm = nn.BatchNorm1d(d_model)

    def forward(self, x):
        if self.kind == "layer":
            return self.norm(x)
        return self.norm(x.transpose(1, 2)).transpose(1, 2)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout, activation):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = getattr(F, activation)

    def forward(self, x):
        return self.lin2(self.dropout(self.act(self.lin1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, norm_kind, activation):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads,
                                               dropout=dropout, batch_first=True)
        self.ff = FeedForward(d_model, d_ff, dropout, activation)
        self.norm1 = SeqNorm(d_model, norm_kind)
        self.norm2 = SeqNorm(d_model, norm_kind)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask):
        attn, _ = self.self_attn(x, x, x,
                                 key_padding_mask=src_key_padding_mask)
        x = self.norm1(x + self.dropout(attn))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, norm_kind, activation):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads,
                                               dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads,
                                                dropout=dropout, batch_first=True)
        self.ff = FeedForward(d_model, d_ff, dropout, activation)
        self.norm1 = SeqNorm(d_model, norm_kind)
        self.norm2 = SeqNorm(d_model, norm_kind)
        self.norm3 = SeqNorm(d_model, norm_kind)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask, tgt_key_padding_mask,
                memory_key_padding_mask):
        sa, sa_w = self.self_attn(x, x, x, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)
        x = self.norm1(x + self.dropout(sa))
        ca, ca_w = self.cross_attn(x, memory, memory,
                                   key_padding_mask=memory_key_padding_mask)
        x = self.norm2(x + self.dropout(ca))
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x, ca_w


class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, cfg):
        super().__init__()
        d = cfg["d_model"]
        self.d_model = d
        self.src_emb = nn.Embedding(src_vocab_size, d, padding_idx=PAD_IDX)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d, padding_idx=PAD_IDX)
        self.pos = PositionalEncoding(d)
        self.dropout = nn.Dropout(cfg["dropout"])

        self.enc_layers = nn.ModuleList([
            EncoderLayer(d, cfg["n_heads"], cfg["d_ff"], cfg["dropout"],
                         cfg["norm_kind"], cfg["activation"])
            for _ in range(cfg["enc_layers"])
        ])
        self.dec_layers = nn.ModuleList([
            DecoderLayer(d, cfg["n_heads"], cfg["d_ff"], cfg["dropout"],
                         cfg["norm_kind"], cfg["activation"])
            for _ in range(cfg["dec_layers"])
        ])
        self.out = nn.Linear(d, tgt_vocab_size)

    def _embed_src(self, src):
        return self.dropout(self.pos(self.src_emb(src) * math.sqrt(self.d_model)))

    def _embed_tgt(self, tgt):
        return self.dropout(self.pos(self.tgt_emb(tgt) * math.sqrt(self.d_model)))

    def encode(self, src):
        pad = src == PAD_IDX
        x = self._embed_src(src)
        for layer in self.enc_layers:
            x = layer(x, src_key_padding_mask=pad)
        return x, pad

    @staticmethod
    def _causal_mask(size, device):
        return torch.triu(torch.ones(size, size, dtype=torch.bool,
                                     device=device), diagonal=1)

    def decode(self, tgt_in, memory, memory_pad):
        tgt_pad = tgt_in == PAD_IDX
        tgt_mask = self._causal_mask(tgt_in.size(1), tgt_in.device)
        x = self._embed_tgt(tgt_in)
        cross_w = None
        for layer in self.dec_layers:
            x, cross_w = layer(x, memory, tgt_mask, tgt_pad, memory_pad)
        return self.out(x), cross_w

    def forward(self, src, tgt):
        memory, memory_pad = self.encode(src)
        logits, _ = self.decode(tgt[:, :-1], memory, memory_pad)
        return logits

    @torch.no_grad()
    def generate(self, src, max_len=40, collect_attn=False):
        self.eval()
        device = src.device
        batch = src.size(0)
        memory, memory_pad = self.encode(src)

        ys = torch.full((batch, 1), SOS_IDX, dtype=torch.long, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        last_cross = None
        for _ in range(max_len):
            logits, cross_w = self.decode(ys, memory, memory_pad)
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            ys = torch.cat([ys, nxt], dim=1)
            finished |= nxt.squeeze(1) == EOS_IDX
            last_cross = cross_w
            if finished.all():
                break

        preds = []
        for i in range(batch):
            seq = ys[i, 1:].tolist()
            if EOS_IDX in seq:
                seq = seq[: seq.index(EOS_IDX)]
            preds.append(seq)
        return (preds, last_cross) if collect_attn else (preds, None)

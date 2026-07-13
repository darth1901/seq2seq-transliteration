import random
import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import VanillaDecoder, AttentionDecoder
from .cells import is_lstm
from .data import PAD_IDX, SOS_IDX, EOS_IDX

class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, cfg):
        super().__init__()
        self.cfg = cfg
        self.use_attention = cfg["use_attention"]
        self.cell_type = cfg["cell_type"]
        self.dec_layers = cfg["dec_layers"]

        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            emb_dim=cfg["emb_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["enc_layers"],
            cell_type=cfg["cell_type"],
            dropout=cfg["dropout"],
            bidirectional=cfg["bidirectional"],
        )

        enc_out_dim = cfg["hidden_dim"] * (2 if cfg["bidirectional"] else 1)

        if self.use_attention:
            self.decoder = AttentionDecoder(
                vocab_size=tgt_vocab_size,
                emb_dim=cfg["emb_dim"],
                hidden_dim=cfg["hidden_dim"],
                num_layers=cfg["dec_layers"],
                cell_type=cfg["cell_type"],
                dropout=cfg["dropout"],
                enc_hidden=enc_out_dim,
            )
        else:
            self.decoder = VanillaDecoder(
                vocab_size=tgt_vocab_size,
                emb_dim=cfg["emb_dim"],
                hidden_dim=cfg["hidden_dim"],
                num_layers=cfg["dec_layers"],
                cell_type=cfg["cell_type"],
                dropout=cfg["dropout"],
            )

    #state bridge
    def _bridge(self, hidden):
        hidden = self.encoder.combine_directions(hidden)

        def adapt(h):
            top = h[-1:].contiguous()            
            return top.repeat(self.dec_layers, 1, 1)

        if is_lstm(self.cell_type):
            h, c = hidden
            return adapt(h), adapt(c)
        return adapt(hidden)

    def _src_mask(self, src):
        return src != PAD_IDX

    #training 
    def forward(self, src, src_len, tgt, teacher_forcing=0.5):
        batch, tgt_len = tgt.shape
        enc_outputs, enc_hidden = self.encoder(src, src_len)
        hidden = self._bridge(enc_hidden)
        mask = self._src_mask(src)

        input_tok = tgt[:, 0] 
        logits_seq = []
        for t in range(1, tgt_len):
            logits, hidden, _ = self.decoder(input_tok, hidden, enc_outputs, mask)
            logits_seq.append(logits)
            teacher = random.random() < teacher_forcing
            input_tok = tgt[:, t] if teacher else logits.argmax(-1)
        return torch.stack(logits_seq, dim=1)

    #inference
    @torch.no_grad()
    def generate(self, src, src_len, max_len=40, collect_attn=False):
        self.eval()
        batch = src.size(0)
        device = src.device

        enc_outputs, enc_hidden = self.encoder(src, src_len)
        hidden = self._bridge(enc_hidden)
        mask = self._src_mask(src)

        input_tok = torch.full((batch,), SOS_IDX, dtype=torch.long, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        preds = [[] for _ in range(batch)]
        attn_rows = [[] for _ in range(batch)]

        for _ in range(max_len):
            logits, hidden, weights = self.decoder(input_tok, hidden, enc_outputs, mask)
            nxt = logits.argmax(-1)
            for i in range(batch):
                if not finished[i]:
                    tok = nxt[i].item()
                    if tok == EOS_IDX:
                        finished[i] = True
                    else:
                        preds[i].append(tok)
                        if collect_attn and weights is not None:
                            attn_rows[i].append(weights[i].detach().cpu())
            input_tok = nxt
            if finished.all():
                break

        attn = None
        if collect_attn:
            attn = [torch.stack(rows) if rows else torch.empty(0)
                    for rows in attn_rows]
        return preds, attn

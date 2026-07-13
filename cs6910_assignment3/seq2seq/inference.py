import torch
import torch.nn.functional as F

from .cells import is_lstm
from .data import SOS_IDX, EOS_IDX

def _slice_hidden(hidden, cell_type):
    return hidden 

@torch.no_grad()
def beam_search_decode(model, src, src_len, beam_size=3, max_len=40,
                       length_penalty=0.0):
    model.eval()
    device = src.device

    enc_outputs, enc_hidden = model.encoder(src, src_len)
    hidden0 = model._bridge(enc_hidden)
    mask = model._src_mask(src)

    start_tok = torch.full((1,), SOS_IDX, dtype=torch.long, device=device)
    beams = [(0.0, [], hidden0, False)]

    for _ in range(max_len):
        if all(b[3] for b in beams):
            break
        candidates = []
        for logprob, toks, hidden, finished in beams:
            if finished:
                candidates.append((logprob, toks, hidden, True))
                continue
            last = start_tok if not toks else torch.tensor(
                [toks[-1]], dtype=torch.long, device=device)
            logits, new_hidden, _ = model.decoder(last, hidden, enc_outputs, mask)
            logp = F.log_softmax(logits.squeeze(0), dim=-1)  
            top_lp, top_ix = logp.topk(beam_size)
            for lp, ix in zip(top_lp.tolist(), top_ix.tolist()):
                if ix == EOS_IDX:
                    candidates.append((logprob + lp, toks, new_hidden, True))
                else:
                    candidates.append((logprob + lp, toks + [ix], new_hidden, False))

        def score(c):
            length = max(len(c[1]), 1)
            return c[0] / (length ** length_penalty) if length_penalty else c[0]

        candidates.sort(key=score, reverse=True)
        beams = candidates[:beam_size]

    best = max(beams, key=lambda b: b[0] / max(len(b[1]), 1) ** length_penalty
               if length_penalty else b[0])
    return best[1]

@torch.no_grad()
def beam_search_batch(model, src, src_len, beam_size=3, max_len=40,
                      length_penalty=0.0):
    outputs = []
    for i in range(src.size(0)):
        s = src[i : i + 1]
        sl = src_len[i : i + 1]
        s = s[:, : sl.item()]
        outputs.append(beam_search_decode(model, s, sl, beam_size,
                                          max_len, length_penalty))
    return outputs

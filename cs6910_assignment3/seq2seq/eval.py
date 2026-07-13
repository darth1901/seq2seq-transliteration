
import os
import csv
import torch

from .data import PAD_IDX, EOS_IDX
from .inference import beam_search_batch

@torch.no_grad()
def evaluate(model, loader, src_vocab, tgt_vocab, device,
             beam_size=1, max_len=40, collect_predictions=False):
    
    model.eval()
    correct = 0
    total = 0
    predictions = []

    for src, src_len, tgt in loader:
        src, src_len, tgt = src.to(device), src_len.to(device), tgt.to(device)

        if beam_size > 1:
            pred_ids = beam_search_batch(model, src, src_len,
                                         beam_size=beam_size, max_len=max_len)
        else:
            pred_ids, _ = model.generate(src, src_len, max_len=max_len)

        for i in range(src.size(0)):
            src_str = src_vocab.decode(src[i].tolist())
            gold_str = tgt_vocab.decode(tgt[i].tolist())
            pred_str = tgt_vocab.decode(pred_ids[i], strip_special=False)
            total += 1
            if pred_str == gold_str:
                correct += 1
            if collect_predictions:
                predictions.append((src_str, gold_str, pred_str))

    acc = correct / max(total, 1)
    return acc, predictions

@torch.no_grad()
def evaluate_transformer(model, loader, src_vocab, tgt_vocab, device,
                         max_len=40, collect_predictions=False):
    model.eval()
    correct = 0
    total = 0
    predictions = []

    for src, src_len, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        pred_ids, _ = model.generate(src, max_len=max_len)
        for i in range(src.size(0)):
            src_str = src_vocab.decode(src[i].tolist())
            gold_str = tgt_vocab.decode(tgt[i].tolist())
            pred_str = tgt_vocab.decode(pred_ids[i], strip_special=False)
            total += 1
            if pred_str == gold_str:
                correct += 1
            if collect_predictions:
                predictions.append((src_str, gold_str, pred_str))

    return correct / max(total, 1), predictions

def dump_predictions(predictions, out_dir, filename="predictions.csv"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "reference", "prediction", "correct"])
        for src, gold, pred in predictions:
            w.writerow([src, gold, pred, int(pred == gold)])
    return path

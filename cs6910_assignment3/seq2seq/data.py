
import os
import csv
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3
SPECIALS = [PAD, SOS, EOS, UNK]

def _is_ascii(s):
    return all(ord(c) < 128 for c in s)

def _read_pairs(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for a, b in csv.reader(f):
            a, b = a.strip(), b.strip()
            if not a or not b:
                continue
            if _is_ascii(a) and not _is_ascii(b):
                rows.append((a, b))
            elif _is_ascii(b) and not _is_ascii(a):
                rows.append((b, a))
            else:
                rows.append((a, b))
    return rows

class CharVocab:

    def __init__(self, tokens):
        self.itos = list(SPECIALS) + [t for t in tokens if t not in SPECIALS]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    @classmethod
    def build(cls, words):
        counter = Counter()
        for w in words:
            counter.update(list(w))
        chars = sorted(counter.keys())
        return cls(chars)

    def encode(self, word, add_sos_eos=False):
        ids = [self.stoi.get(c, UNK_IDX) for c in word]
        if add_sos_eos:
            ids = [SOS_IDX] + ids + [EOS_IDX]
        return ids

    def decode(self, ids, strip_special=True):
        chars = []
        for i in ids:
            tok = self.itos[i] if 0 <= i < len(self.itos) else UNK
            if strip_special and tok in SPECIALS:
                if tok == EOS:
                    break
                continue
            chars.append(tok)
        return "".join(chars)

class TransliterationDataset(Dataset):

    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        latin, native = self.pairs[idx]
        src = torch.tensor(self.src_vocab.encode(latin), dtype=torch.long)
        tgt = torch.tensor(self.tgt_vocab.encode(native, add_sos_eos=True),
                           dtype=torch.long)
        return src, tgt

def collate_batch(batch):

    srcs, tgts = zip(*batch)
    src_len = torch.tensor([len(s) for s in srcs], dtype=torch.long)

    max_s = max(len(s) for s in srcs)
    max_t = max(len(t) for t in tgts)

    src_pad = torch.full((len(batch), max_s), PAD_IDX, dtype=torch.long)
    tgt_pad = torch.full((len(batch), max_t), PAD_IDX, dtype=torch.long)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_pad[i, : len(s)] = s
        tgt_pad[i, : len(t)] = t
    return src_pad, src_len, tgt_pad

def load_language(root, lang):

    def p(split):
        return os.path.join(root, lang, f"{lang}_{split}.csv")

    train_pairs = _read_pairs(p("train"))
    valid_pairs = _read_pairs(p("valid"))
    test_pairs = _read_pairs(p("test"))

    src_vocab = CharVocab.build(latin for latin, _ in train_pairs)
    tgt_vocab = CharVocab.build(native for _, native in train_pairs)

    ds = {
        "train": TransliterationDataset(train_pairs, src_vocab, tgt_vocab),
        "valid": TransliterationDataset(valid_pairs, src_vocab, tgt_vocab),
        "test": TransliterationDataset(test_pairs, src_vocab, tgt_vocab),
    }
    return ds, src_vocab, tgt_vocab

def make_loaders(ds, batch_size, num_workers=2):
    return {
        split: DataLoader(
            d,
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=collate_batch,
            num_workers=num_workers,
            pin_memory=True,
        )
        for split, d in ds.items()
    }

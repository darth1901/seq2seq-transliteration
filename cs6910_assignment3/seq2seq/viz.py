import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
import torch
from .data import SOS_IDX
import glob
from matplotlib.font_manager import FontProperties, fontManager

_FONT_PATTERNS = [
    "/usr/share/fonts/**/NotoSansDevanagari*.ttf",
    "/usr/share/fonts/**/*Devanagari*.ttf",
    "/usr/share/fonts/**/lohit_deva*.ttf",
]

def _devanagari_font():
    for pattern in _FONT_PATTERNS:
        hits = sorted(glob.glob(pattern, recursive=True),
                      key=lambda p: ("Sans" not in p, "UI" in p))
        if hits:
            fontManager.addfont(hits[0])
            return FontProperties(fname=hits[0])
    return None

def _attn_for_example(model, src, src_len, tgt_vocab, device, max_len=40):
    src = src.to(device)
    src_len = src_len.to(device)
    preds, attn = model.generate(src, src_len, max_len=max_len, collect_attn=True)
    pred_ids = preds[0]
    pred_chars = [tgt_vocab.itos[i] for i in pred_ids]
    A = attn[0]
    A = A[: len(pred_chars)]
    return pred_chars, A

def plot_attention_grid(model, examples, src_vocab, tgt_vocab, device,
                        rows=3, cols=3, max_len=40, save_path=None):
    fp = _devanagari_font()
    n = min(len(examples), rows * cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for k in range(rows * cols):
        ax = axes[k]
        if k >= n:
            ax.axis("off")
            continue
        src, src_len = examples[k]
        src_chars = [src_vocab.itos[i] for i in src[0].tolist()
                     if i != 0][: src_len.item()]
        pred_chars, A = _attn_for_example(model, src, src_len, tgt_vocab,
                                          device, max_len)
        A = A[:, : len(src_chars)].numpy()

        ax.imshow(A, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(src_chars)))
        ax.set_xticklabels(src_chars, fontsize=8)
        ax.set_yticks(range(len(pred_chars)))
        ax.set_yticklabels(pred_chars, fontsize=9, fontproperties=fp)
        ax.set_xlabel("source (Latin)")
        ax.set_ylabel("prediction")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

def plot_connectivity(model, src, src_len, src_vocab, tgt_vocab, device,
                      max_len=40, save_path=None):
    fp = _devanagari_font()
    src_chars = [src_vocab.itos[i] for i in src[0].tolist()
                 if i != 0][: src_len.item()]
    pred_chars, A = _attn_for_example(model, src, src_len, tgt_vocab,
                                      device, max_len)
    A = A[:, : len(src_chars)]

    fig, ax = plt.subplots(figsize=(max(6, len(src_chars)), 5))
    for j, c in enumerate(src_chars):
        ax.text(j, 1.0, c, ha="center", va="center", fontsize=12)
    for i, c in enumerate(pred_chars):
        ax.text(i, 0.0, c, ha="center", va="center", fontsize=12, fontproperties=fp)

    for i in range(len(pred_chars)):
        j = int(A[i].argmax().item())
        w = float(A[i, j].item())
        ax.plot([j, i], [1.0, 0.0], color="crimson", alpha=max(w, 0.05),
                linewidth=1 + 3 * w)

    ax.set_ylim(-0.3, 1.3)
    ax.set_xlim(-1, max(len(src_chars), len(pred_chars)))
    ax.axis("off")
    ax.set_title("connectivity: output char -> most-attended input char")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

def collect_examples(dataset, indices):
    out = []
    for idx in indices:
        src, _ = dataset[idx]
        out.append((src.unsqueeze(0), torch.tensor([len(src)])))
    return out

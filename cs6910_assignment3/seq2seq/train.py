import argparse
import os
import torch
import torch.nn as nn
import wandb

from .data import load_language, make_loaders, PAD_IDX
from .model import Seq2Seq
from .transformer import TransformerSeq2Seq
from .eval import evaluate, evaluate_transformer, dump_predictions

#config
def build_config(args):
    if args.model_type == "transformer":
        return {
            "model_type": "transformer",
            "d_model": args.emb_dim,               
            "d_ff": args.d_ff,                      
            "n_heads": args.n_heads,
            "enc_layers": args.enc_layers,          
            "dec_layers": args.dec_layers,          
            "dropout": args.dropout,
            "norm_kind": args.norm_kind,           
            "activation": args.activation,
        }
    return {
        "model_type": args.model_type,
        "use_attention": args.model_type == "attention",
        "emb_dim": args.emb_dim,
        "hidden_dim": args.hidden_dim,
        "enc_layers": args.enc_layers,
        "dec_layers": args.dec_layers,
        "cell_type": args.cell_type,
        "dropout": args.dropout,
        "bidirectional": args.bidirectional,
    }


def build_model(cfg, src_vocab_size, tgt_vocab_size):
    if cfg["model_type"] == "transformer":
        return TransformerSeq2Seq(src_vocab_size, tgt_vocab_size, cfg)
    return Seq2Seq(src_vocab_size, tgt_vocab_size, cfg)


#one epoch
def run_epoch(model, loader, criterion, optimizer, device, cfg,
              teacher_forcing, train=True):
    model.train(train)
    total_loss, total_tokens = 0.0, 0

    for src, src_len, tgt in loader:
        src, src_len, tgt = src.to(device), src_len.to(device), tgt.to(device)

        if cfg["model_type"] == "transformer":
            logits = model(src, tgt)
        else:
            logits = model(src, src_len, tgt, teacher_forcing=teacher_forcing)

        gold = tgt[:, 1:]
        loss = criterion(logits.reshape(-1, logits.size(-1)),
                         gold.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        n_tok = (gold != PAD_IDX).sum().item()
        total_loss += loss.item() * n_tok
        total_tokens += n_tok

    return total_loss / max(total_tokens, 1)


def val_accuracy(model, loader, src_vocab, tgt_vocab, device, cfg,
                 beam_size=1):
    if cfg["model_type"] == "transformer":
        acc, _ = evaluate_transformer(model, loader, src_vocab, tgt_vocab, device)
    else:
        acc, _ = evaluate(model, loader, src_vocab, tgt_vocab, device,
                          beam_size=beam_size)
    return acc


#driver 
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds, src_vocab, tgt_vocab = load_language(args.data_root, args.lang)
    loaders = make_loaders(ds, args.batch_size, num_workers=args.num_workers)

    cfg = build_config(args)
    lr, batch_size, epochs = args.lr, args.batch_size, args.epochs
    teacher_forcing, beam_size = args.teacher_forcing, args.beam_size

    use_wandb = args.wandb
    if use_wandb:
        wandb.init(project=args.wandb_project, config={**cfg,
                   "lr": lr, "batch_size": batch_size,
                   "epochs": epochs, "lang": args.lang,
                   "teacher_forcing": teacher_forcing, "beam_size": beam_size})
        wc = dict(wandb.config)
        for k in cfg:
            if k in wc:
                cfg[k] = wc[k]
        lr = wc.get("lr", lr)
        batch_size = wc.get("batch_size", batch_size)
        epochs = wc.get("epochs", epochs)
        teacher_forcing = wc.get("teacher_forcing", teacher_forcing)
        beam_size = wc.get("beam_size", beam_size)
        run_name = "-".join(f"{k}_{cfg.get(k)}" for k in
                            ("cell_type", "emb_dim", "hidden_dim",
                             "enc_layers", "dec_layers") if k in cfg)
        wandb.run.name = run_name or wandb.run.name
        loaders = make_loaders(ds, batch_size, num_workers=args.num_workers)

    model = build_model(cfg, len(src_vocab), len(tgt_vocab)).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model={cfg['model_type']}  params={n_params:,}  device={device}")

    best_val = 0.0
    for epoch in range(1, epochs + 1):
        tr_loss = run_epoch(model, loaders["train"], criterion, optimizer,
                            device, cfg, teacher_forcing, train=True)
        with torch.no_grad():
            va_loss = run_epoch(model, loaders["valid"], criterion, optimizer,
                                device, cfg, 1.0, train=False)
        va_acc = val_accuracy(model, loaders["valid"], src_vocab, tgt_vocab,
                              device, cfg, beam_size=beam_size)

        print(f"epoch {epoch:2d}  train_loss {tr_loss:.4f}  "
              f"val_loss {va_loss:.4f}  val_acc {va_acc:.4f}")
        if use_wandb:
            wandb.log({"epoch": epoch, "train_loss": tr_loss,
                       "val_loss": va_loss, "val_acc": va_acc})

        if va_acc > best_val:
            best_val = va_acc
            if args.save_path:
                torch.save({"model_state": model.state_dict(), "cfg": cfg,
                            "src_vocab": src_vocab.itos,
                            "tgt_vocab": tgt_vocab.itos}, args.save_path)

    if use_wandb:
        wandb.log({"best_val_acc": best_val})

    if args.eval_test:
        if cfg["model_type"] == "transformer":
            test_acc, preds = evaluate_transformer(
                model, loaders["test"], src_vocab, tgt_vocab, device,
                collect_predictions=True)
        else:
            test_acc, preds = evaluate(
                model, loaders["test"], src_vocab, tgt_vocab, device,
                beam_size=beam_size, collect_predictions=True)
        print(f"TEST exact-match accuracy: {test_acc:.4f}")
        if use_wandb:
            wandb.log({"test_acc": test_acc})
        if args.pred_dir:
            path = dump_predictions(preds, args.pred_dir)
            print(f"wrote predictions to {path}")

    return model, best_val


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="aksharantar_sampled")
    p.add_argument("--lang", default="hin")
    p.add_argument("--model_type", default="attention",
                   choices=["vanilla", "attention", "transformer"])

    #shared
    p.add_argument("--emb_dim", type=int, default=64) 
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--enc_layers", type=int, default=1)
    p.add_argument("--dec_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.2)

    #rnn-only
    p.add_argument("--cell_type", default="LSTM", choices=["RNN", "GRU", "LSTM"])
    p.add_argument("--bidirectional", action="store_true")

    #transformer-only
    p.add_argument("--d_ff", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--norm_kind", default="batch", choices=["batch", "layer"])
    p.add_argument("--activation", default="relu")

    #optimisation
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--teacher_forcing", type=float, default=0.5)
    p.add_argument("--beam_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)

    #logging
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="cs6910-assignment3")
    p.add_argument("--save_path", default="")
    p.add_argument("--eval_test", action="store_true")
    p.add_argument("--pred_dir", default="")
    return p


if __name__ == "__main__":
    train(get_parser().parse_args())

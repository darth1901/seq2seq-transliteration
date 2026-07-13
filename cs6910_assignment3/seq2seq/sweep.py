import argparse
import wandb
from . import train as train_mod

def rnn_sweep(model_type):
    return {
        "method": "bayes",
        "metric": {"name": "val_acc", "goal": "maximize"},
        "early_terminate": {"type": "hyperband", "min_iter": 3, "eta": 2},
        "parameters": {
            "model_type": {"value": model_type},
            "emb_dim": {"values": [16, 32, 64, 256]},
            "hidden_dim": {"values": [64, 128, 256]},
            "enc_layers": {"values": [1, 2, 3]},
            "dec_layers": {"values": [1, 2, 3]},
            "cell_type": {"values": ["RNN", "GRU", "LSTM"]},
            "bidirectional": {"values": [True, False]},
            "dropout": {"values": [0.2, 0.3]},
            "beam_size": {"values": [1, 3, 5]},
            "lr": {"values": [1e-3, 5e-4]},
            "batch_size": {"values": [64, 128]},
            "teacher_forcing": {"values": [0.5, 1.0]},
        },
    }

def transformer_sweep():
    return {
        "method": "bayes",
        "metric": {"name": "val_acc", "goal": "maximize"},
        "early_terminate": {"type": "hyperband", "min_iter": 3, "eta": 2},
        "parameters": {
            "model_type": {"value": "transformer"},
            "emb_dim": {"value": 64},          
            "d_ff": {"value": 256},            
            "enc_layers": {"value": 1},        
            "dec_layers": {"value": 1},        
            "norm_kind": {"values": ["batch", "layer"]},
            "n_heads": {"values": [2, 4, 8]},
            "activation": {"values": ["relu", "gelu"]},
            "dropout": {"values": [0.1, 0.2, 0.3]},
            "lr": {"values": [1e-3, 5e-4, 1e-4]},
            "batch_size": {"values": [64, 128]},
        },
    }

def make_agent(base_args):
    def _run():
        args = train_mod.get_parser().parse_args(base_args)
        args.wandb = True
        train_mod.train(args)
    return _run

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", default="attention",
                   choices=["vanilla", "attention", "transformer"])
    p.add_argument("--count", type=int, default=40)
    p.add_argument("--lang", default="hin")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--data_root", default="aksharantar_sampled")
    p.add_argument("--wandb_project", default="cs6910-assignment3")
    args = p.parse_args()

    if args.model_type == "transformer":
        cfg = transformer_sweep()
    else:
        cfg = rnn_sweep(args.model_type)

    sweep_id = wandb.sweep(cfg, project=args.wandb_project)

    base = ["--wandb", "--eval_test",
            "--lang", args.lang,
            "--epochs", str(args.epochs),
            "--data_root", args.data_root,
            "--wandb_project", args.wandb_project,
            "--model_type", args.model_type]
    wandb.agent(sweep_id, function=make_agent(base), count=args.count)


if __name__ == "__main__":
    main()

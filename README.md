# CS6910 Assignment 3: Sequence-to-Sequence Transliteration

RNN, attention, and Transformer models for Latin to Devanagari transliteration on the Aksharantar dataset (AI4Bharat). Character-level sequence-to-sequence, evaluated by exact-match on the test set.

## Results

| Model | Test exact-match accuracy |
|---|---|
| Vanilla seq2seq (LSTM) | 0.3813 |
| Seq2seq + Bahdanau attention (GRU) | **0.3982** |
| Transformer (built from `nn.MultiheadAttention`) | 0.2239 |

Decomposed by source word length however, attention gains **+10.8 points** over vanilla on words of 12+ characters (0.438 vs 0.330) while gaining only +1.4 on words of 5 characters or fewer. The transformer, constrained by the assignment spec to d_model 64 and one layer per stack, collapses on long words (0.028 at length 16). See the [wandb report]() for the full analysis.

## Layout

```
cs6910_assignment3/
  seq2seq/
    data.py          Aksharantar loading, char vocab, dataset, padded collate
    cells.py         RNN / GRU / LSTM factory (the swappable-cell requirement)
    encoder.py       configurable depth, bidirectionality, dropout
    decoder.py       vanilla and attention decoders (stepwise)
    attention.py     Bahdanau additive attention
    model.py         Seq2Seq wrapper: state bridge, teacher forcing, greedy generation
    transformer.py   Transformer built from internals (Q6)
    inference.py     beam search
    eval.py          exact-match accuracy and prediction dumping
    train.py         single-run CLI (all three model types)
    sweep.py         wandb Bayesian sweep with Hyperband early stopping
    viz.py           attention heatmaps (Q5d) and connectivity (Q7)
  predictions_vanilla/        test-set predictions, vanilla model
  predictions_attention/      test-set predictions, attention model
  predictions_transformers/   test-set predictions, Transformer
  requirements.txt
```

Only `train.py` and `sweep.py` are executed directly. The rest are library modules.

## Setup

```bash
pip install -r requirements.txt
```

Download and unzip the Aksharantar sampled dataset so the layout is `aksharantar_sampled/<lang>/<lang>_{train,valid,test}.csv`, then point `--data_root` at the parent folder.

**For Devanagari rendering in the attention heatmaps**, a font covering the script must be installed:

```bash
apt-get install -y fonts-noto-core      
```

Install it **before** matplotlib is first imported. Matplotlib caches its font registry at import time, so a font installed afterwards is often invisible to name-based lookup. `viz.py` loads the font by file path to work around this, but the font must exist on disk.

## Training

**Vanilla seq2seq** (best config):

```bash
python -m seq2seq.train --model_type vanilla --data_root aksharantar_sampled --lang hin \
  --cell_type LSTM --emb_dim 256 --hidden_dim 256 --enc_layers 3 --dec_layers 2 \
  --dropout 0.3 --lr 1e-3 --batch_size 128 --epochs 15 \
  --beam_size 3 --eval_test --save_path vanilla_best.pt --pred_dir predictions_vanilla
```

**Attention seq2seq** (best config):

```bash
python -m seq2seq.train --model_type attention --data_root aksharantar_sampled --lang hin \
  --cell_type GRU --emb_dim 256 --hidden_dim 256 --enc_layers 1 --dec_layers 1 \
  --bidirectional --dropout 0.2 --lr 5e-4 --batch_size 128 --epochs 15 \
  --beam_size 3 --eval_test --save_path attention_best.pt --pred_dir predictions_attention
```

**Transformer** (best config). The spec fixes d_model 64, d_ff 256, and one layer per stack. 

```bash
python -m seq2seq.train --model_type transformer --data_root aksharantar_sampled --lang hin \
  --emb_dim 64 --d_ff 256 --n_heads 4 --enc_layers 1 --dec_layers 1 \
  --norm_kind layer --activation gelu --dropout 0.1 \
  --lr 1e-3 --batch_size 64 --epochs 40 \
  --eval_test --save_path transformer_best.pt --pred_dir predictions_transformers
```

Add `--wandb` to log to Weights and Biases. Run `python -m seq2seq.train --help` for all flags.

## Evaluation

`--eval_test` evaluates the trained model on the held-out test set and prints exact-match accuracy. `--pred_dir <folder>` writes a `predictions.csv` with columns `source, reference, prediction, correct` to that folder. The test set is used only for this final evaluation, never during the sweep or model selection.

## Hyperparameter sweeps

```bash
python -m seq2seq.sweep --model_type attention --lang hin --count 20 --epochs 15 \
  --data_root aksharantar_sampled --wandb_project cs6910-assignment3
```

Bayesian search with Hyperband early termination. The search space is defined in `seq2seq/sweep.py`.

**Beam width is deliberately excluded from the search space.** It is a decode-time parameter: it changes how the output space is searched for an already-trained model, not what the model learns, and it lifts nearly every configuration by a similar margin. Including it multiplied run time by up to 10x (beam 5 runs took 21 to 29 minutes against 2 to 7 minutes for greedy) without changing which architecture ranked best. Sweeps run greedily; beam search is reintroduced on the single winning configuration at test time.

## Visualisation

```python
import torch
from seq2seq.data import load_language
from seq2seq.model import Seq2Seq
from seq2seq import viz

ck = torch.load("attention_best.pt", map_location="cpu", weights_only=False)
ds, sv, tv = load_language("aksharantar_sampled", "hin")
model = Seq2Seq(len(sv), len(tv), ck["cfg"])
model.load_state_dict(ck["model_state"])
model.eval()

ex = viz.collect_examples(ds["test"], list(range(9)))
viz.plot_attention_grid(model, ex, sv, tv, torch.device("cpu"), save_path="attention_grid.png")

src, sl = ex[0]
viz.plot_connectivity(model, src, sl, sv, tv, torch.device("cpu"), save_path="connectivity.png")
```

## Dependencies

PyTorch, wandb, matplotlib (and pandas for `analysis.py`). No TensorFlow, Keras, or JAX.

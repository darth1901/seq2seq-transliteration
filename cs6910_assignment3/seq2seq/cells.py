import torch.nn as nn

_CELLS = {
    "RNN": nn.RNN,
    "GRU": nn.GRU,
    "LSTM": nn.LSTM,
}

def is_lstm(cell_type):
    return cell_type.upper() == "LSTM"

def build_rnn(cell_type, input_size, hidden_size, num_layers,
              dropout=0.0, bidirectional=False):
    cell_type = cell_type.upper()
    if cell_type not in _CELLS:
        raise ValueError(f"unknown cell type {cell_type!r}")
    layer_dropout = dropout if num_layers > 1 else 0.0
    return _CELLS[cell_type](
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=layer_dropout,
        bidirectional=bidirectional,
        batch_first=True,
    )

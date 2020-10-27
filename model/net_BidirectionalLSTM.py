# -*- coding: UTF-8 -*-


from torch.nn import Module, LSTM, Linear

class Net_BidirectionalLSTM(Module):
    #Linear回归输出层
    def __init__(self, config):
        super(Net_BidirectionalLSTM, self).__init__()
        self.lstm = LSTM(input_size=config.input_size, hidden_size=config.hidden_size,
                         num_layers=config.lstm_layers, batch_first=True, dropout=config.dropout_rate,bidirectional=True)
        self.linear = Linear(in_features=config.hidden_size * 2, out_features=config.output_size)

    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        linear_out = self.linear(lstm_out)
        return linear_out, hidden






# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from config import n_letters, n_categories, HIDDEN_SIZE, DEVICE

class RNN(nn.Module):
    def __init__(self, input_size=n_letters, hidden_size=HIDDEN_SIZE, output_size=n_categories):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=False)
        self.fc = nn.Linear(hidden_size, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, hidden):
        x = x.unsqueeze(1)          # (seq_len, 57) -> (seq_len, 1, 57)
        out, hn = self.rnn(x, hidden)
        out = self.fc(out[-1])      # 取最后一个时间步
        return self.logsoftmax(out), hn

    def init_hidden(self):
        return torch.zeros(1, 1, HIDDEN_SIZE, device=DEVICE)


class LSTM(nn.Module):
    def __init__(self, input_size=n_letters, hidden_size=HIDDEN_SIZE, output_size=n_categories):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=False)
        self.fc = nn.Linear(hidden_size, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, hidden, c):
        x = x.unsqueeze(1)
        out, (hn, cn) = self.lstm(x, (hidden, c))
        out = self.fc(out[-1])
        return self.logsoftmax(out), hn, cn

    def init_hidden(self):
        h = torch.zeros(1, 1, HIDDEN_SIZE, device=DEVICE)
        c = torch.zeros(1, 1, HIDDEN_SIZE, device=DEVICE)
        return h, c


class GRU(nn.Module):
    def __init__(self, input_size=n_letters, hidden_size=HIDDEN_SIZE, output_size=n_categories):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=False)
        self.fc = nn.Linear(hidden_size, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, hidden):
        x = x.unsqueeze(1)
        out, hn = self.gru(x, hidden)
        out = self.fc(out[-1])
        return self.logsoftmax(out), hn

    def init_hidden(self):
        return torch.zeros(1, 1, HIDDEN_SIZE, device=DEVICE)













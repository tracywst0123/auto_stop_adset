import torch
from torch import nn
import torch.nn.functional as F


class MV_LSTM(torch.nn.Module):
    def __init__(self, n_features, seq_length):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 20  # number of hidden states
        self.n_layers = 1  # number of LSTM layers (stacked)

        self.conv2d = nn.Sequential(
            nn.BatchNorm2d(self.seq_len*self.n_features),
            nn.Conv2d()
        )

        self.l_lstm = torch.nn.LSTM(input_size=n_features,
                                    hidden_size=self.n_hidden,
                                    num_layers=self.n_layers,
                                    batch_first=True)
        # (batch_size,seq_len, num_directions * hidden_size)

        self.hidden1label = nn.Sequential(
            nn.Linear(self.n_hidden*self.seq_len, self.n_hidden*self.seq_len),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(self.n_hidden*self.seq_len, 1),
            nn.Sigmoid()
        )
        # self.l_linear = torch.nn.Linear(self.n_hidden * self.seq_len, 1)

    def init_hidden(self, batch_size):
        hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        self.hidden = (hidden_state, cell_state)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        lstm_out, self.hidden = self.l_lstm(x, self.hidden)
        x = lstm_out.contiguous().view(batch_size, -1)
        x = self.hidden1label(x)
        # x = self.l_linear(x)
        return x


class Conv2D2Layer(torch.nn.Module):
    def __init__(self, n_features, seq_length):
        super(Conv2D2Layer, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 20  # number of hidden states
        self.n_layers = 1  # number of LSTM layers (stacked)

        self.l_lstm = torch.nn.LSTM(input_size=n_features,
                                    hidden_size=self.n_hidden,
                                    num_layers=self.n_layers,
                                    batch_first=True)
        # (batch_size,seq_len, num_directions * hidden_size)

        self.hidden1label = nn.Sequential(
            nn.Linear(self.n_hidden*self.seq_len, self.n_hidden*self.seq_len),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(self.n_hidden*self.seq_len, 1),
            nn.Sigmoid()
        )
        # self.l_linear = torch.nn.Linear(self.n_hidden * self.seq_len, 1)

    def init_hidden(self, batch_size):
        hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        self.hidden = (hidden_state, cell_state)

    def forward(self, x):
        batch_size, seq_len, n_features = x.size()
        lstm_out, self.hidden = self.l_lstm(x, self.hidden)
        x = lstm_out.contiguous().view(batch_size, -1)
        x = self.hidden1label(x)
        # x = self.l_linear(x)
        return x
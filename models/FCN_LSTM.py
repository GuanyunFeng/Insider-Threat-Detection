# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LSTM_FCN(nn.Module):

    def __init__(self, input_channels, input_dim, hidden_dim, n_class):
        super().__init__()
        self.input_channels = input_channels
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_class = n_class
        self.relu    = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(self.input_channels, self.hidden_dim, kernel_size=7)
        self.bn1     = nn.BatchNorm1d(self.hidden_dim)
        self.conv2 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=5)
        self.bn2     = nn.BatchNorm1d(self.hidden_dim)
        self.conv3 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1)
        self.bn3     = nn.BatchNorm1d(self.hidden_dim)
        self.conv4 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1)
        self.bn4     = nn.BatchNorm1d(self.hidden_dim)
        self.conv5 = nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1)
        self.bn5     = nn.BatchNorm1d(self.hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, 2, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim*2, self.n_class)

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        h_0 = torch.randn(2, batch_size, self.hidden_dim).to('cuda')
        c_0 = torch.randn(2, batch_size, self.hidden_dim).to('cuda')
        y = x.view(batch_size, seq_len, self.input_dim)
        # output(batch_size, seq_len, num_directions * hidden_size)
        y, _ = self.lstm(y, (h_0, c_0))
        y = y[:, -1, :].view(batch_size, -1)

        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x1)))
        x3 = self.relu(self.bn3(self.conv3(x2)))
        x4 = self.relu(self.bn4(self.conv4(x3)))
        x5 = self.relu(self.bn5(self.conv5(x4)))
        #x6 = torch.cat([x5,x3], dim=2)
        x6 = self.pool(x5+x4)
        x6 = x6.view(batch_size, self.hidden_dim)
        
        out = torch.cat([x6, y], dim=1)
        pred = self.linear(out)

        return pred

from __future__ import annotations
import torch.nn as nn
import torch.nn.functional as F


class MultiPerceptron(nn.Module):

    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(28 * 28 * 1, 512)
        self.hidden_layer = nn.Linear(512, 512)
        self.output_layer = nn.Linear(512, 10)
        self.dropout1 = nn.Dropout2d()
        self.dropout2 = nn.Dropout2d()

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.dropout1(x)
        x = F.relu(self.hidden_layer(x))
        x = self.dropout2(x)
        return F.relu(self.output_layer(x))

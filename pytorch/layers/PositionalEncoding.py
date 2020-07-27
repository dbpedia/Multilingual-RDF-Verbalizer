import torch
import torch.nn as nn
from torch.autograd import Variable
import math

class PositionalEncoding(nn.Module):

    def __init__(self, hid_dim, max_length=100):

        # Compute the positional encodings once in log space.
        self.pe = torch.zeros(max_length, hid_dim)
        self.position = torch.arange(0, max_length).unsqueeze(1)

        self.div_term = torch.exp(torch.arange(0, hid_dim, 2) *
                             -(math.log(10000.0) / hid_dim))
        self.pe[:, 0::2] = torch.sin(self.position * self.div_term)
        self.pe[:, 1::2] = torch.cos(self.position * self.div_term)

        self.pe = self.pe.unsqueeze(0)

        #self.register_buffer('pe', pe)

    def forward(self, x):

        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return x

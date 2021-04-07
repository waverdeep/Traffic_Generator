from typing import Any

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim=2):
        self.linear01 = nn.Linear(latent_dim, 15)
        self.prelu01 = nn.PReLU()
        self.linear02 = nn.Linear(15, output_dim)

    def forward(self, x):
        out = self.prelu01(self.linear01(x))
        return self.linear02(out)


class Discriminator(nn.Module):
    def __init__(self, input_dim=2):
        self.linear01 = nn.Linear(input_dim, 25)
        self.prelu01 = nn.PReLU()
        self.linear02 = nn.Linear(25, 1)
        self.sigmoid01 = nn.Sigmoid()

    def forward(self, x):
        out = self.prelu01(self.linear01(x))
        return self.sigmoid01(self.linear02(out))

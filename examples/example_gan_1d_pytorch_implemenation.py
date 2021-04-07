import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim=2):
        super(Generator, self).__init__()
        self.generator = nn.Seqential(
            nn.Linear(latent_dim, 15),
            nn.Relu(),
            nn.Linear(15, output_dim)
        )

    def forward(self, x):
        return self.generator(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim=2):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, 25),
            nn.Relu(),
            nn.Linear(25, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminator(x)


def generate_latent_points(latent_dim, n):
    # generate points in the latent space
    x_input = np.random.randn(latent_dim * n)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n, latent_dim)
    return x_input



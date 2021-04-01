import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, z_dim, output_dim=2): # latent_dim
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 15),
            nn.LeakyReLU(0.1),
            nn.Linear(15, output_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)


class Discriminator(nn.Module):
    def __init__(self, in_features=2):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 25),
            nn.LeakyReLU(0.1),
            nn.Linear(25, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)






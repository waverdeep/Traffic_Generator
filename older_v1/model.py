import torch
import torch.nn as nn
import torch.nn.functional as F

# FC 모델 2계층 너무 성능이 좋지 않음.
# 그리고 음수가 나오면 안되는데 음수가 나오는 경향이 생김
# 음수를 나오지 못하게 할 무언가가 필요함 ( 후처리? )
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


class Generator_V2(nn.Module):
    def __init__(self, z_dim, output_dim=2): # latent_dim
        super(Generator_V2, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)


class Discriminator_V2(nn.Module):
    def __init__(self, in_features=2):
        super(Discriminator_V2, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)





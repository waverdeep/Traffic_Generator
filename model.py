from typing import Any

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim=2, choice_activation_func='PReLU'):
        super(Generator, self).__init__()
        self.choice_activation_func = choice_activation_func
        self.linear01 = nn.Linear(latent_dim, 15)
        self.prelu01 = nn.PReLU()
        self.relu01 = nn.ReLU()
        self.linear02 = nn.Linear(15, output_dim)

    def forward(self, x):
        if self.choice_activation_func == 'PReLU':
            out = self.prelu01(self.linear01(x))
        elif self.choice_activation_func == 'ReLU':
            out = self.relu01(self.linear01(x))
        return self.linear02(out)


class Discriminator(nn.Module):
    def __init__(self, input_dim=2, choice_activation_func='PReLU'):
        super(Discriminator, self).__init__()
        self.choice_activation_func = choice_activation_func
        self.linear01 = nn.Linear(input_dim, 25)
        self.prelu01 = nn.PReLU()
        self.relu01 = nn.ReLU()
        self.linear02 = nn.Linear(25, 1)
        self.sigmoid01 = nn.Sigmoid()

    def forward(self, x):
        if self.choice_activation_func == 'PReLU':
            out = self.prelu01(self.linear01(x))
            return self.sigmoid01(self.linear02(out))
        elif self.choice_activation_func == 'ReLU':
            out = self.relu01(self.linear01(x))
            return self.sigmoid01(self.linear02(out))



class Generator_1D_CNN(nn.Module):
    def __init__(self, latent_dim, output_dim=2, choice_activation_func='PReLU'):
        super(Generator_1D_CNN, self).__init__()
        self.choice_activation_func = choice_activation_func
        self.linear01 = nn.Linear(latent_dim, 25)
        self.prelu01 = nn.PReLU()
        self.conv1d01 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=3)
        self.conv1d02 = nn.Conv1d(in_channels=5, out_channels=9, kernel_size=5)
        self.conv1d03 = nn.Conv1d(in_channels=9, out_channels=15, kernel_size=7)
        self.conv1d04 = nn.Conv1d(in_channels=15, out_channels=20, kernel_size=5)
        self.conv1d05 = nn.Conv1d(in_channels=20, out_channels=25, kernel_size=3)
        self.linear02 = nn.Linear(321045, 15)
        self.relu01 = nn.ReLU()
        self.linear03 = nn.Linear(15, output_dim)

    def forward(self, x):
        if self.choice_activation_func == 'PReLU':
            print(x.shape)
            out = self.prelu01(self.linear01(x))
            print(out.shape)
            out = out.unsqueeze(1)
            print(out.shape)
            out = self.conv1d05(self.conv1d04(self.conv1d03(self.conv1d02(self.conv1d01(out)))))
            print(out.shape)
            out = out.squeeze(-1)
            print(out.shape)
            if self.choice_activation_func == 'PReLU':
                out = self.prelu01(self.linear02(out))
            return self.linear03(out)

        elif self.choice_activation_func == 'ReLU':
            out = self.relu01(self.linear01(x))
            out = self.conv1d03(self.conv1d02(out))
            out = torch.flatten(out)
            if self.choice_activation_func == 'PReLU':
                out = self.prelu01(self.linear02(out))
            return self.linear03(out)


class Discriminator_1D_CNN(nn.Module):
    def __init__(self, input_dim=2, choice_activation_func='PReLU'):
        super(Discriminator_1D_CNN, self).__init__()
        self.choice_activation_func = choice_activation_func
        self.linear01 = nn.Linear(input_dim, 25)
        self.prelu01 = nn.PReLU()
        self.relu01 = nn.ReLU()
        self.linear02 = nn.Linear(25, 1)
        self.sigmoid01 = nn.Sigmoid()

    def forward(self, x):
        if self.choice_activation_func == 'PReLU':
            out = self.prelu01(self.linear01(x))
        elif self.choice_activation_func == 'ReLU':
            out = self.relu01(self.linear01(x))
        return self.sigmoid01(self.linear02(out))
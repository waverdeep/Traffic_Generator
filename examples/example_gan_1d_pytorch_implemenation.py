import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import hstack
from numpy import zeros
from numpy import ones
import torch.nn.functional as F
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim=2):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 15),
            nn.ELU(),
            nn.Linear(15, output_dim)
        )

    def forward(self, x):
        return self.generator(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim=2):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, 25),
            nn.ELU(),
            nn.Linear(25, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminator(x)


def generate_real_samples(n): # n : batch
    # generate inputs in [-0.5, 0.5]
    X1 = np.random.rand(n) - 0.5
    # generate outputs X^2
    X2 = X1 * X1
    # stack arrays
    X1 = X1.reshape(n, 1)
    X2 = X2.reshape(n, 1)
    X = hstack((X1, X2))
    # generate class labels
    y = ones((n, 1))
    X = torch.tensor(X, dtype=torch.float)
    return X, y


def generate_latent_points(latent_dim, n): # n : batch
    # generate points in the latent space
    x_input = np.random.randn(latent_dim * n)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n, latent_dim)
    x_input = torch.tensor(x_input, dtype=torch.float)
    return x_input


def generate_fake_samples(generator, latent_dim, n):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n)
    # predict outputs
    X = generator(x_input)
    # create class labels
    y = zeros((n, 1))
    return X, y

# evaluate the discriminator and plot real and fake points
def summarize_performance(epoch, generator, discriminator, latent_dim, n=100):
    # prepare real samples
    with torch.no_grad():
        x_real, y_real = generate_real_samples(n)
        # evaluate discriminator on real examples
        _ = discriminator(x_real)
        # prepare fake examples
        x_fake, y_fake = generate_fake_samples(generator, latent_dim, n)
        # evaluate discriminator on fake examples
        # scatter plot real and fake data points
        plt.scatter(x_real[:, 0], x_real[:, 1], color='red')
        plt.scatter(x_fake[:, 0], x_fake[:, 1], color='blue')
        plt.show()

# paramters
latent_dim = 5

### training
generator = Generator(latent_dim=latent_dim)
discriminator = Discriminator(input_dim=2)

criterion = nn.BCELoss()

generator_optimizer = optim.Adam(generator.parameters())
discriminator_optimizer = optim.Adam(discriminator.parameters())

half_batch = int(1024 / 2)
for i in range(100000):

    discriminator.zero_grad()
    x_real, y_real = generate_real_samples(half_batch)
    x_fake, y_fake = generate_fake_samples(generator, latent_dim, half_batch)
    x_real_decision = discriminator(x_real)
    d_real_error = criterion(x_real_decision, torch.tensor(y_real, dtype=torch.float))
    d_real_error.backward()
    x_fake_decision = discriminator(x_fake)
    d_fake_error = criterion(x_fake_decision, torch.tensor(y_fake, dtype=torch.float))
    d_fake_error.backward()
    discriminator_optimizer.step()

    generator.zero_grad()
    x_gan = generate_latent_points(latent_dim=latent_dim, n=1024)
    y_gan = ones((1024, 1))
    x_gan = generator(x_gan)
    x_gan_decision = discriminator(x_gan)
    x_gan_error = criterion(x_gan_decision, torch.tensor(y_gan, dtype=torch.float))
    x_gan_error.backward()
    generator_optimizer.step()

    print(
        f"Epoch [{i}]"
    )
    if i %1000 == 0:
        summarize_performance(i,generator,discriminator,latent_dim,100)


import torch
import TrafficDataLoader
from torch.utils.data import DataLoader
import model
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from numpy import zeros
from numpy import ones


def setup_gpu():
    return "cuda" if torch.cuda.is_available() else "cpu"


def is_gpu():
    return torch.cuda.is_available()


def load_amazon_dataset(input_dir, batch_size):
    amazon_dataset = TrafficDataLoader.AmazonPrimeDataset(input_dir=input_dir)
    amazon_dataloader = DataLoader(amazon_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    return amazon_dataloader


def setup_model(latent_dim, input_dim, gpu=True):
    generator = model.Generator(latent_dim=latent_dim, output_dim=input_dim)
    discriminator = model.Discriminator(input_dim=input_dim)
    if gpu:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
    return generator, discriminator


def setup_optimizer(generator_model, discriminator_model):
    generator_optimizer = optim.Adam(generator_model.parameters())
    discriminator_optimizer = optim.Adam(discriminator_model.parameters())
    return generator_optimizer, discriminator_optimizer


def setup_criterion(gpu=True):
    criterion = nn.BCELoss()
    if gpu:
        criterion = criterion.cuda()
    return criterion


def denormalize(x):
    return 0.5 * (x * 200000 - x * 0 + 200000 + 0)


def generate_latent_points(latent_dim, n): # n : batch
    # generate points in the latent space
    # x_input = torch.randn(batch_size, latent_dim).cuda()
    # print(x_input.shape)
    # reshape into a batch of inputs for the network
    x_input = np.random.randn(latent_dim * n)
    x_input = x_input.reshape(n, latent_dim)
    # x_input = denormalize(x_input)
    x_input = torch.tensor(x_input, dtype=torch.float).cuda()
    return x_input


def generate_fake_samples(generator, latent_dim, n):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n)
    # predict outputs
    X = generator(x_input)
    # create class labels
    y = np.zeros((n, 1))
    y = torch.tensor(y, dtype=torch.float).cuda()
    return X, y


def make_real_y(n):
    y = ones((n, 1))
    return torch.tensor(y, dtype=torch.float)


def train(n_epochs, n_batch_size, loader, generator_model, discriminator_model, latent_dim):
    step = 0
    for epoch in range(n_epochs):
        print('start epoch [{}] ->'.format(epoch))
        for batch_idx, (real, y) in enumerate(loader):
            for line in real:
                real_x = line.unsqueeze(1)
                real_y = make_real_y(len(real_x))

                real_x = real_x.cuda()
                real_y = real_y.cuda()

                fake_x, fake_y = generate_fake_samples(generator_model, latent_dim, len(real_x))

                discriminator_model.zero_grad()
                real_x_decision = discriminator_model(real_x)
                d_real_error = criterion(real_x_decision, real_y)
                d_real_error.backward()

                fake_x_decision = discriminator_model(fake_x)
                d_fake_error = criterion(fake_x_decision, fake_y)
                d_fake_error.backward()
                discriminator_optimizer.step()

                generator_model.zero_grad()
                x_gan = generate_latent_points(latent_dim, 1259)
                y_gan = np.ones((1259, 1))
                y_gan = torch.tensor(y_gan, dtype=torch.float).cuda()
                x_gan = generator_model(x_gan)
                x_gan_decision = discriminator_model(x_gan)
                x_gan_error = criterion(x_gan_decision, y_gan)
                x_gan_error.backward()
                generator_optimizer.step()
            # real_x = real.cuda()
            # print("real : {}".format(real_x.shape))
            # real_y = y.cuda()
            # fake_x, fake_y = generate_fake_samples(generator_model, latent_dim, n_batch_size)
            # print("fake : {}".format(fake_x.shape))
            # discriminator_model.zero_grad()
            # real_x_decision = discriminator_model(real_x)
            # d_real_error = criterion(real_x_decision, real_y)
            # d_real_error.backward()
            #
            # fake_x_decision = discriminator_model(fake_x)
            # d_fake_error = criterion(fake_x_decision, fake_y)
            # d_fake_error.backward()
            # discriminator_optimizer.step()
            #
            # generator_model.zero_grad()
            # x_gan = generate_latent_points(latent_dim, n_batch_size)
            # y_gan = np.ones((n_batch_size, 1))
            # y_gan = torch.tensor(y_gan, dtype=torch.float).cuda()
            # x_gan = generator_model(x_gan)
            # x_gan_decision = discriminator_model(x_gan)
            # x_gan_error = criterion(x_gan_decision, y_gan)
            # x_gan_error.backward()
            # generator_optimizer.step()

            if batch_idx == 0:
                print(
                    f"Epoch [{epoch}/{n_epochs}] Batch {batch_idx}/{len(loader)} \
                                  Loss D: {d_real_error:.6f}, loss G: {x_gan_error:.6f}"
                )

                with torch.no_grad():
                    x_real = real_x
                    y_real = real_y
                    # evaluate discriminator on real examples
                    _ = discriminator_model(x_real)
                    # prepare fake examples
                    x_fake, y_fake = generate_fake_samples(generator_model, latent_dim, 1259)
                    # evaluate discriminator on fake examples
                    # scatter plot real and fake data points
                    x_fake = x_fake.cpu()
                    y_fake = y_fake.cpu()
                    x_real = x_real.cpu()
                    y_real = y_real.cpu()
                    plt.plot(x_real[:, 0], color='red', alpha=0.4)
                    plt.plot(x_fake[:, 0], color='blue', alpha=0.4)
                    plt.savefig('dataset/reformat_amazon/result_V2/fake/fake_{}.png'.format(str(step).zfill(4)))
                    plt.show()

                    plt.figure(figsize=(24, 12))
                    plt.title('fake_fixed')
                    plt.plot(x_fake[:, 0])
                    plt.savefig('dataset/reformat_amazon/result_V2/fake_only/fake_only_{}.png'.format(str(step).zfill(4)))
                    plt.show()

                step = step+1

        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'generator': generator_model,
                'discriminator': discriminator_model,
                'generator_state_dict': generator_model.state_dict(),
                'discriminator_state_dict': discriminator_model.state_dict(),
            }, "dataset/reformat_amazon/result_V2/checkpoints/model_checkpoint_{}.pt".format(epoch))


if __name__ == '__main__':
    input_dir = 'dataset/reformat_amazon/static'
    batch_size = 6000
    latent_dim = 5
    input_dim = 1
    n_epochs = 1000
    device = setup_gpu()
    print("device check : {}".format(device))
    amazon_dataloader = load_amazon_dataset(input_dir, batch_size)
    generator_model, discriminator_model = setup_model(latent_dim, input_dim, device)
    generator_optimizer, discriminator_optimizer = setup_optimizer(generator_model, discriminator_model)
    criterion = setup_criterion(device)
    train(n_epochs, batch_size, amazon_dataloader, generator_model, discriminator_model, latent_dim)



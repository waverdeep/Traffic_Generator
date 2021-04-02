import torch
import model
import torch.optim as optim
import torch.nn as nn
import TrafficDataLoader
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

### setup hyperparameters
lr = 3e-4
z_dim = 1000 # fix it
input_dim = 1259 # fix it
batch_size = 2048
num_epochs = 1000
step = 0

def setup_gpu():
    return "cuda" if torch.cuda.is_available() else "cpu"


device = setup_gpu()

### model load
discriminator = model.Discriminator(input_dim).to(device=device)
generator = model.Generator(z_dim, input_dim).to(device=device)
optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = optim.Adam(generator.parameters(), lr=lr)
criterion = nn.BCELoss()

fixed_noise = torch.randn((batch_size, z_dim)).to(device)

### dataset
amazon_dataset = TrafficDataLoader.AmazonPrimeDataset('dataset/reformat_amazon/static')
loader = DataLoader(amazon_dataset, batch_size=batch_size, shuffle=True)

print("start training")
for epoch in range(num_epochs):
    print("start epoch : {}".format(epoch))
    for batch_idx, real in enumerate(loader):

        real = real.to(device)
        batch_size = real.shape[0]
        ### train Discriminator
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = generator(noise)

        discriminator_real = discriminator(real)
        lossD_real = criterion(discriminator_real, torch.ones_like(discriminator_real))
        discriminator_fake = discriminator(fake)
        lossD_fake = criterion(discriminator_fake, torch.ones_like(discriminator_fake))
        lossD = (lossD_real + lossD_fake)/2
        discriminator.zero_grad()
        lossD.backward(retain_graph=True)
        optimizer_discriminator.step()
        ### train generator
        output = discriminator(fake)
        lossG = criterion(output, torch.ones_like(output))
        generator.zero_grad()
        lossG.backward()
        optimizer_generator.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                              Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )
            with torch.no_grad():
                fake = generator(fixed_noise).cpu()
                data = real.cpu()

                plt.figure(figsize=(24, 12))
                plt.plot(fake[0])
                plt.savefig('dataset/reformat_amazon/result/fake/fake_{}.png'.format(step))
                plt.show()
                plt.figure(figsize=(24, 12))
                plt.plot(data[0])
                plt.savefig('dataset/reformat_amazon/result/real/real_{}.png'.format(step))
                plt.show()
                step += 1

    if epoch % 10 == 0:
        torch.save({
            'epoch': epoch,
            'generator': generator,
            'discriminator': discriminator,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
        }, "model_checkpoint_{}.pt".format(epoch))


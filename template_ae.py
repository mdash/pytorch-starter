from utils_tabular import FastTensorDataLoader
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.distributions
import torch.utils
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import CyclicLR


torch.manual_seed(0)
plt.rcParams['figure.dpi'] = 200

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Encoder(nn.Module):
    def __init__(self, input_dims, hidden_dims, latent_dims):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_dims, hidden_dims)
        self.linear2 = nn.Linear(hidden_dims, latent_dims)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)


class Decoder(nn.Module):
    def __init__(self, input_dims, hidden_dims, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, hidden_dims)
        self.linear2 = nn.Linear(hidden_dims, input_dims)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        # return z.reshape((-1, 1, 28, 28)) ##for image
        return z


class Autoencoder(nn.Module):
    def __init__(self, input_dims, hidden_dims, latent_dims):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dims, hidden_dims, latent_dims)
        self.decoder = Decoder(input_dims, hidden_dims, latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def plot_losses(list_loss, list_lr):
    """
    Plot loss and learning rate over training epochs
    """
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    x = [x+1 for x in range(len(list_lr))]
    ax1.plot(x, list_loss, 'o-', color="blue")
    ax2.plot(x, list_lr, 'o-', color="red")

    ax1.set_xlabel('X data')
    ax1.set_ylabel('Loss', color='blue')
    ax2.set_ylabel('LR', color='red')

    plt.show()


def train(autoencoder, data, epochs=20, batch_size=1024,
          learning_rate=0.1, display_plots=True):

    opt = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)

    # learning rate scheduling
    # https://www.kaggle.com/isbhargav/guide-to-pytorch-learning-rate-scheduling
    lr_sched = CyclicLR(opt, base_lr=learning_rate//10,
                        max_lr=learning_rate*2,
                        step_size_up=epochs/5, mode="triangular2")

    list_loss = []
    list_lr = []
    for epoch in range(epochs):
        mean_sse = 0
        for x, y in data:
            x = x.to(device)
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum()
            loss.backward()
            opt.step()
            mean_sse += loss
        print(f'Epoch: {epoch}; Mean SSE: {mean_sse/batch_size:0.4f}')
        # print(f'Learning Rate: {opt.param_groups[0]["lr"]}')
        list_loss.append(mean_sse/batch_size)
        list_lr.append(opt.param_groups[0]["lr"])
        lr_sched.step()

    # plot loss and learning rate over epochs
    if display_plots:
        plot_losses(list_loss, list_lr)
    return autoencoder


def plot_latent(autoencoder, data, num_batches=100):
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break


if __name__ == "main":

    # training part

    # loading image data
    data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data',
                                   transform=torchvision.transforms.ToTensor(),
                                   download=True),
        batch_size=128,
        shuffle=True)

    # Tabular data loading and batches here
    data = FastTensorDataLoader(train_x, train_y,
                                batch_size=1024, shuffle=False)

    latent_dims = 2
    input_dims = 1024
    autoencoder = Autoencoder(input_dims, hidden_dims, latent_dims).to(device)

    autoencoder = train(autoencoder, data, batch_size=1024, learning_rate=0.1)

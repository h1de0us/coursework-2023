import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims, batch_size):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, 2 * latent_dim)  # Output of encoder is mean and variance of the latent distribution
        )
        # self.mean_layer = nn.Linear(hidden_dims, latent_dim)
        # self.logvar_layer = nn.Linear(hidden_dims, latent_dim)

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, input_dim),
            nn.Sigmoid()  # Output of decoder is a probability distribution over the input space
        )
        # self.output_layer = nn.Linear(input_dim, input_dim)

    def encode(self, x):
        # print(f'before encoder: {x.shape}')  # (batch_size, seq_len, input_dim)
        h = self.encoder(x)
        # print(f'after encoder: {h.shape}')
        # mean = self.mean_layer(h)
        # logvar = self.logvar_layer(h)
        mean, logvar = torch.chunk(h, 2, dim=2)  # (batch_size, seq_len, latent_dim)
        # print(f'mean: {mean.shape}, logvar: {logvar.shape}')
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        h = self.decoder(z)
        # print(f'h after decoding: {h.shape}')
        # x_hat = self.output_layer(h)
        # return x_hat
        return h  # Output of decoder is a probability distribution over the input space

    def forward(self, x):
        print(f'x in forward: {x.shape}')
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        print(f'z after reparameterization: {z.shape}')
        x_hat = self.decode(z)
        print(f'after decoding: {x_hat.shape}')
        return x_hat, mean, logvar

    def loss_function(self, x, x_hat, mean, logvar):
        BCE = nn.functional.binary_cross_entropy_with_logits(x_hat, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return BCE + KLD

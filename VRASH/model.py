import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder_rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.encoder_mean = nn.Linear(hidden_dim, latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder_rnn = nn.LSTM(latent_dim + input_dim, hidden_dim, batch_first=True)
        self.decoder_fc = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        x = x.float()
        _, hidden = self.encoder_rnn(x)
        hidden = hidden[0]
        mean = self.encoder_mean(hidden)
        logvar = self.encoder_logvar(hidden)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def decode(self, z, x, prev_outputs=None):
        x = x.unsqueeze(0)
        print(f'x: {x.shape}')
        print(f'z: {z.shape}')
        # z = z.repeat(1, 1, x.shape[2] // z.shape[2])
        # print(f'new z: {z.shape}')
        if len(prev_outputs):
            # prev_outputs = torch.cat(prev_outputs, dim=1)
            # print(f'prev: {prev_outputs.shape}')
            x = torch.cat(prev_outputs, dim=-1)
            print(f'new x: {x.shape}')
        inputs = torch.cat([z, x], dim=-1)
        inputs = inputs.float()
        print(f'inputs: {inputs.shape}')
        output, _ = self.decoder_rnn(inputs)
        output = self.decoder_fc(output)
        return output

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        prev_outputs = []
        output = None
        for t in range(x.shape[2]):
            output = self.decode(z, x[:, t], prev_outputs)
            prev_outputs.append(output)
        return output, mean, logvar

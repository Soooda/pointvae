import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, z):
        z = torch.tanh(self.linear1(z))
        y = torch.sigmoid(self.linear2(z))
        return y


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_var = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):
        x = self.linear1(x)
        h = torch.tanh(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_dim, output_size):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_size=input_size, hidden_size=hidden_size, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, hidden_size=hidden_size, output_size=output_size)

    def reparameterize(self, mu, log_var):
        std = log_var.mul(0.5).exp_()
        esp = torch.randn(mu.size()).to(mu.get_device())
        z = mu + std * esp
        return z

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var
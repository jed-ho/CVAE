import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class Stochastic(nn.Module):
    """
    Performs the reparameterization trick
    """
    def reparametrize(self, mu, log_var):
        epsilon = torch.randn_like(mu)
        std = torch.exp(0.5 * log_var)
        z = mu + std * epsilon
        return z

class GaussianSample(Stochastic):
    """
    Samples from the Gaussian latent space
    """
    def __init__(self, input_dim, output_dim):
        super(GaussianSample, self).__init__()
        self.mu = nn.Linear(input_dim, output_dim)
        self.log_var = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        mu = self.mu(x)
        log_var = self.log_var(x)
        z_sample = self.reparametrize(mu, log_var)
        return z_sample, mu, log_var

class Encoder(nn.Module):
    """
    Encoder model with fully connected architecture
    """
    def __init__(self, latent_dim, num_param, window_size):
        super(Encoder, self).__init__()
        self.num_param = num_param
        self.window_size = window_size
        self.input_dim = num_param * window_size
        hidden_dims = [512, 256, 128]  # Adjust these sizes as needed

        # Define fully connected layers
        self.fc_layers = nn.ModuleList()
        last_dim = self.input_dim
        for h_dim in hidden_dims:
            self.fc_layers.append(nn.Linear(last_dim, h_dim))
            last_dim = h_dim

        # Sampling layer
        self.sample = GaussianSample(last_dim, latent_dim)
        
    def forward(self, x):
        # Flatten the input
        x = x.reshape(x.size(0), -1)
        # Pass through fully connected layers with ReLU activation
        for layer in self.fc_layers:
            x = F.relu(layer(x))
        # Reparameterization trick
        z_sample, mu, log_var = self.sample(x)
        return z_sample, mu, log_var

class Decoder(nn.Module):
    """
    Decoder model with fully connected architecture
    """
    def __init__(self, latent_dim, num_param, window_size, scale_flag=0):
        super(Decoder, self).__init__()
        self.output_dim = num_param * window_size
        self.num_param = num_param
        self.window_size = window_size
        hidden_dims = [128, 256, 512]  # Should mirror the encoder's hidden dimensions in reverse

        # Define fully connected layers
        self.fc_layers = nn.ModuleList()
        last_dim = latent_dim  # Input is only z now
        for h_dim in hidden_dims:
            self.fc_layers.append(nn.Linear(last_dim, h_dim))
            last_dim = h_dim

        # Output layer
        self.output_layer = nn.Linear(last_dim, self.output_dim)
        # Output activation function
        if scale_flag == 1:
            self.output_activation = nn.Identity()
        elif scale_flag == 0:
            self.output_activation = nn.Sigmoid()
                    
    def forward(self, z):
        x = z  # Input is only z now
        # Pass through fully connected layers with ReLU activation
        for layer in self.fc_layers:
            x = F.relu(layer(x))
        # Final output layer with activation
        x = self.output_activation(self.output_layer(x))
        # Reshape back to original input shape
        x = x.reshape(-1, self.num_param, self.window_size)
        return x

class VAE(nn.Module):
    """
    Unsupervised VAE model
    """
    def __init__(self, latent_dim, num_param, window_size, scale_flag=0):
        super(VAE, self).__init__()
        self.z_dim = latent_dim
        self.p_dim = num_param
        self.t_dim = window_size
        self.scale_flag = scale_flag

        # Instantiate the modified Encoder and Decoder
        self.encoder = Encoder(self.z_dim, self.p_dim, self.t_dim)
        self.decoder = Decoder(self.z_dim, self.p_dim, self.t_dim, self.scale_flag)
        self.kl_div = 0
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            
    def kld_(self, q_param): 
        (mu, log_var) = q_param
        # Compute KL divergence
        kl = 0.5 * torch.sum(-1 - log_var + torch.pow(mu, 2) + torch.exp(log_var), dim=-1)
        return kl
        
    def forward(self, x):
        # Encode input and get latent variables
        z, z_mu, z_log_var = self.encoder(x)
        # Compute KL divergence
        self.kl_div = self.kld_((z_mu, z_log_var))
        # Decode the latent variable back to input space
        x_rec = self.decoder(z)
        return x_rec

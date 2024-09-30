import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

cuda = torch.cuda.is_available()

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

        # Latent representation layer
        self.latent_layer = nn.Linear(last_dim, latent_dim)
        
    def forward(self, x):
        # Flatten the input
        x = x.reshape(x.size(0), -1)
        # Pass through fully connected layers with ReLU activation
        for layer in self.fc_layers:
            x = F.relu(layer(x))
        # Get the latent representation
        z = self.latent_layer(x)
        return z

class Decoder(nn.Module):
    """
    Decoder model with fully connected architecture
    """
    def __init__(self, latent_dim, num_param, window_size, scale_flag):
        super(Decoder, self).__init__()
        self.output_dim = num_param * window_size
        self.num_param = num_param
        self.window_size = window_size
        hidden_dims = [128, 256, 512]  # Should mirror the encoder's hidden dimensions in reverse

        # Define fully connected layers
        self.fc_layers = nn.ModuleList()
        last_dim = latent_dim
        for h_dim in hidden_dims:
            self.fc_layers.append(nn.Linear(last_dim, h_dim))
            last_dim = h_dim

        # Output layer
        self.output_layer = nn.Linear(last_dim, self.output_dim)
        # Output activation function
        self.scale_flag = scale_flag
        if scale_flag == 1:
            self.output_activation = nn.Identity()
        elif scale_flag == 0:
            self.output_activation = nn.Sigmoid()
                
    def forward(self, z):
        x = z
        # Pass through fully connected layers with ReLU activation
        for layer in self.fc_layers:
            x = F.relu(layer(x))
        # Final output layer with activation
        x = self.output_activation(self.output_layer(x))
        # Reshape back to original input shape
        x = x.reshape(-1, self.num_param, self.window_size)
        return x

class Autoencoder(nn.Module):
    """
    Autoencoder model with fully connected architecture
    """
    def __init__(self, latent_dim, num_param, window_size, scale_flag):
        super(Autoencoder, self).__init__()
        self.z_dim = latent_dim
        self.p_dim = num_param
        self.t_dim = window_size
        self.scale_flag = scale_flag

        # Instantiate the Encoder and Decoder
        self.encoder = Encoder(self.z_dim, self.p_dim, self.t_dim)
        self.decoder = Decoder(self.z_dim, self.p_dim, self.t_dim, self.scale_flag)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
        
    def forward(self, x):
        # Encode input to get latent representation
        z = self.encoder(x)
        # Decode the latent representation back to input space
        x_rec = self.decoder(z)
        return x_rec

    def encode(self, x):
        # Get the latent representation of input x
        z = self.encoder(x)
        return z

    def decode(self, z):
        # Reconstruct input from latent representation z
        x_rec = self.decoder(z)
        return x_rec

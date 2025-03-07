import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Positional Encoding for Transformer modules
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: [batch, seq_len, d_model] or [seq_len, batch, d_model]
        if (
            x.dim() == 3 and x.size(0) == 1
        ):  # if batch dimension is first and length is 1, do nothing
            return x
        if x.shape[1] <= self.pe.size(1):
            return x + self.pe[:, : x.size(1)]
        else:
            raise ValueError("Input sequence length exceeds maximum length.")


# Transformer-based Encoder
class TransformerEncoderAE(nn.Module):
    def __init__(
        self, num_param, window_size, d_model=64, nhead=4, num_layers=3, num_classes=2
    ):
        """
        Args:
            num_param: number of features per time step.
            window_size: sequence length.
            d_model: transformer embedding dimension.
            nhead: number of attention heads.
            num_layers: number of transformer encoder layers.
            num_classes: number of classification labels.
        """
        super().__init__()
        self.num_param = num_param
        self.window_size = window_size
        self.d_model = d_model

        # Embed each time step (each "token") from num_param to d_model dimensions
        self.embedding = nn.Linear(num_param, d_model)
        self.pos_encoder = PositionalEncoding(d_model, window_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Classification head: use mean pooling over the sequence dimension
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # Input x: [batch, num_param, window_size]
        # We interpret the time dimension as the sequence length, so transpose to [batch, window_size, num_param]
        x = x.permute(0, 2, 1)
        x = self.embedding(x)  # [batch, window_size, d_model]
        x = self.pos_encoder(x)  # add positional encoding

        # Transformer expects input shape [seq_len, batch, d_model]
        x = x.transpose(0, 1)  # [window_size, batch, d_model]
        encoded = self.transformer_encoder(x)
        encoded = encoded.transpose(0, 1)  # back to [batch, window_size, d_model]

        # Pool the encoder outputs over time (here, using mean pooling) for classification
        pooled = encoded.mean(dim=1)  # [batch, d_model]
        class_logits = self.classifier(pooled)  # [batch, num_classes]

        return encoded, class_logits


# Transformer-based Decoder
class TransformerDecoderAE(nn.Module):
    def __init__(self, num_param, window_size, d_model=64, nhead=4, num_layers=3):
        """
        Args:
            num_param: number of features per time step.
            window_size: sequence length.
            d_model: transformer embedding dimension.
            nhead: number of attention heads.
            num_layers: number of transformer decoder layers.
        """
        super().__init__()
        self.num_param = num_param
        self.window_size = window_size
        self.d_model = d_model

        # A set of learned target queries that will serve as the starting point for decoding.
        # This has shape [window_size, d_model] and is repeated for each batch.
        self.target_queries = nn.Parameter(torch.randn(window_size, d_model))
        self.pos_decoder = PositionalEncoding(d_model, window_size)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # Project the decoder output back to the original feature dimension
        self.output_layer = nn.Linear(d_model, num_param)

    def forward(self, memory):
        # memory: output from encoder, shape [batch, window_size, d_model]
        batch_size = memory.size(0)
        # Prepare target sequence for the decoder:
        # Expand the learned queries to shape [window_size, batch, d_model]
        tgt = self.target_queries.unsqueeze(1).expand(-1, batch_size, -1)
        tgt = self.pos_decoder(tgt.transpose(0, 1)).transpose(0, 1)

        # Transformer decoder expects memory in shape [seq_len, batch, d_model]
        memory = memory.transpose(0, 1)
        decoded = self.transformer_decoder(tgt, memory)  # [window_size, batch, d_model]
        decoded = decoded.transpose(0, 1)  # [batch, window_size, d_model]

        # Project to original feature dimension for each time step
        output = self.output_layer(decoded)  # [batch, window_size, num_param]
        # Transpose back to [batch, num_param, window_size] to match original input shape
        output = output.permute(0, 2, 1)
        return output


# The overall supervised Transformer Autoencoder
class TransformerAutoencoder(nn.Module):
    def __init__(
        self,
        num_param,
        window_size,
        d_model=64,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        num_classes=2,
    ):
        super().__init__()
        self.encoder = TransformerEncoderAE(
            num_param, window_size, d_model, nhead, num_encoder_layers, num_classes
        )
        self.decoder = TransformerDecoderAE(
            num_param, window_size, d_model, nhead, num_decoder_layers
        )
        self.num_classes = num_classes

    def forward(self, x, y_onehot=None):
        """
        Args:
            x: input tensor of shape [batch, num_param, window_size]
            y_onehot: (optional) one-hot labels (not used in encoding/decoding; classification is computed directly)
        Returns:
            x_rec: reconstructed input
            class_logits: classification output from encoder
        """
        encoded, class_logits = self.encoder(x)
        x_rec = self.decoder(encoded)
        return x_rec, class_logits

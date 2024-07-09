from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from mlpforecast.net.embending import PosEmbedding, RotaryEmbedding

ACTIVATIONS = [
    "Identity",
    "ReLU",
    "Softplus",
    "Tanh",
    "Sigmoid",
    "SiLU",
    "GELU",
    "ELU",
    "SELU",
    "LeakyReLU",
    "PReLU",
]


def create_linear(in_channels, out_channels, bn=False):
    """
    Creates a linear layer with optional batch normalization.

    Parameters:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bn (bool, optional): If True, adds batch normalization. Defaults to False.

    Returns:
        nn.Module: Linear layer with optional batch normalization.
    """
    # Create a linear layer
    m = nn.Linear(in_channels, out_channels)

    # Initialize the weights using Kaiming normal initialization with a ReLU nonlinearity
    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    # Initialize the bias to zero if present
    if m.bias is not None:
        torch.nn.init.constant_(m.bias, 0)

    # Add batch normalization if requested
    if bn:
        # Create a batch normalization layer
        bn_layer = nn.BatchNorm1d(out_channels)

        # Combine the linear layer and batch normalization into a sequential module
        m = nn.Sequential(m, bn_layer)

    return m


def FeedForward(dim, expansion_factor=2, dropout=0.0, activation=nn.GELU(), bn=True):
    """
    Creates a feedforward block composed of linear layers, activation function, and dropout.

    Parameters:
        dim (int): Dimensionality of the input.
        expansion_factor (int, optional): Expansion factor for the intermediate hidden layer. Defaults to 2.
        dropout (float, optional): Dropout probability. Defaults to 0.0 (no dropout).
        activation (torch.nn.Module, optional): Activation function. Defaults to GELU().
        bn (bool, optional): If True, adds batch normalization. Defaults to True.

    Returns:
        nn.Sequential: Feedforward block.
    """
    # Create a sequential block with linear layer, activation, and dropout
    block = nn.Sequential(
        create_linear(dim, dim * expansion_factor, bn),
        activation,
        nn.Dropout(dropout),
        create_linear(dim * expansion_factor, dim, bn),
        nn.Dropout(dropout),
    )

    return block


class MLPBlock(nn.Module):
    def __init__(
        self,
        in_size=1,
        latent_dim=32,
        features_start=16,
        expansion_factor=1,
        residual=False,
        num_layers=4,
        context_size=96,
        activation=nn.ReLU(),
        bn=True,
    ):
        """
        Multi-Layer Perceptron (MLP) block with configurable layers and options.

        Parameters:
            in_size (int, optional): Size of the input. Defaults to 1.
            latent_dim (int, optional): Dimensionality of the latent space. Defaults to 32.
            features_start (int, optional): Number of features in the initial layer. Defaults to 16.
            num_layers (int, optional): Number of layers in the MLP. Defaults to 4.
            context_size (int, optional): Size of the context. Defaults to 96.
            activation (torch.nn.Module, optional): Activation function. Defaults to ReLU().
            bn (bool, optional): If True, adds batch normalization. Defaults to True.
        """
        super().__init__()

        # Calculate the size of the input after flattening
        self.in_size = in_size * context_size
        self.context_size = context_size
        self.residual = residual
        if residual:
            expansion_factor = 1

        # Initialize a list to store the layers of the MLP
        layers = [
            nn.Sequential(
                create_linear(self.in_size, features_start, bn=False),
                activation,
            )
        ]
        feats = features_start

        # Create the specified number of layers in the MLP
        for i in range(num_layers - 1):
            layers.append(
                nn.Sequential(
                    create_linear(feats, feats * expansion_factor, bn=bn), activation
                )
            )
            feats = feats * expansion_factor

        # Add the final layer with latent_dim and activation, without batch normalization
        layers.append(
            nn.Sequential(create_linear(feats, latent_dim, bn=False), activation)
        )

        # Create a ModuleList to store the layers
        self.mlp_network = nn.ModuleList(layers)

    def forward(self, x):
        """
        Forward pass of the MLP block.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the MLP block.
        """
        # Flatten the input along dimensions 1 and 2
        if x.ndim == 3:
            x = x.flatten(1, 2)

        # Pass the input through each layer in the MLP
        x = self.mlp_network[0](x)
        for i in range(1, len(self.mlp_network) - 1):
            if self.residual:
                x += self.mlp_network[i](x)
            else:
                x = self.mlp_network[i](x)

        x = self.mlp_network[-1](x)
        return x


class PastFutureEncoder(nn.Module):
    def __init__(
        self,
        emb_size=28,
        embed_type=None,
        latent_size=64,
        depth=2,
        residual=False,
        expansion_factor=2,
        context_size=96,
        activation=nn.ReLU(),
        dropout=0.25,
        n_channels=1,
    ):
        """
        Encoder module for processing past sequences.

        Parameters:
            emb_size (int, optional): Dimensionality of the embedding space. Defaults to 28.
            embed_type (String, optional): Type of embedding to use. Defaults to None. Either -> 'PosEmb', 'RotaryEmb', 'CombinedEmb'
            latent_size: (int, optional): Dimensionality of the latent space. Defaults to 64.
            depth (int, optional): Number of layers in the MLP. Defaults to 2.
            input_size (int, optional): Size of the input window. Defaults to 96.
            activation (torch.nn.Module, optional): Activation function. Defaults to ReLU().
            dropout (float, optional): Dropout probability. Defaults to 0.25.
            n_channels (int, optional): Number of input channels. Defaults to 1.
        """
        super().__init__()

        self.encoder = MLPBlock(
            in_size=emb_size if embed_type is not None else n_channels,
            latent_dim=latent_size,
            features_start=latent_size,
            expansion_factor=expansion_factor,
            residual=residual,
            num_layers=depth,
            context_size=context_size,
            activation=activation,
        )

        # Normalize the input using LayerNorm
        self.norm = nn.LayerNorm(n_channels)

        # Apply dropout to the input
        self.dropout = nn.Dropout(dropout)

        # Store hyperparameters
        self.embed_type = embed_type

        # Embedding based on the specified type
        if embed_type == "PosEmb":
            self.emb = PosEmbedding(n_channels, emb_size, window_size=context_size)
        elif embed_type == "RotaryEmb":
            self.emb = RotaryEmbedding(emb_size)
        elif embed_type == "CombinedEmb":
            self.pos_emb = self.emb = PosEmbedding(
                n_channels, emb_size, window_size=context_size
            )
            self.rotary_emb = RotaryEmbedding(emb_size)

    def forward(self, x):
        """
        Forward pass of the PastEncoder module.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after processing through the encoder.
        """
        # Normalize the input
        x = self.norm(x)

        # Apply embedding based on the specified type

        if self.embed_type == "CombinedEmb":
            x = self.pos_emb(x) + self.rotary_emb(x)
            # Apply dropout to the embedded input
            x = self.dropout(x)

        elif self.embed_type in ["PosEmb", "RotaryEmb"]:
            x = self.emb(x)
            # Apply dropout to the embedded input
            x = self.dropout(x)

        # Pass the input through the encoder
        x = self.encoder(x)

        return x
    
class MLPForecastNetwork(nn.Module):
    def __init__(
        self,
        n_target_series: int,
        n_unknown_features: int,
        n_known_calender_features: int,
        n_known_continuous_features: int,
        embedding_size: int = 28,
        embedding_type: str = None,
        combination_type: str = "additional",
        expansion_factor: int = 2,
        residual: bool = False,
        hidden_size: int = 256,
        num_layers: int = 2,
        forecast_horizon: int = 48,
        input_window_size: int = 96,
        activation_function: str = "SiLU",
        out_activation_function: str = "Identity",
        dropout_rate: float = 0.25,
        alpha=0.1,
        num_attention_heads: int = 4,
    ):
        """
        Multilayer Perceptron (MLP) Forecast Network for time series forecasting.

        Parameters:
            target_series (List): List of target variables.
            unknown_features (List, optional): List of unknown time-varying features. Defaults to [].
            known_categorical_features (List, optional): List of known categorical time-varying features. Defaults to [].
            known_continuous_features (List, optional): List of known time-varying features. Defaults to [].
            embedding_size (int, optional): Dimensionality of the embedding space. Defaults to 28.
            embedding_type (String, optional): Type of embedding to use. Defaults to None. Either -> 'PosEmb', 'RotaryEmb', 'CombinedEmb'
            hidden_size: (int, optional): Dimensionality of the latent space. Defaults to 64.
            num_layers (int, optional): Number of layers in the MLP. Defaults to 2.
            forecast_horizon (int, optional): Number of future time steps to forecast. Defaults to 48.
            activation_function (torch.nn.Module, optional): Activation function. Defaults to ReLU().
            dropout_rate (float, optional): Dropout probability. Defaults to 0.25.
            num_attention_heads (int, optional): Number of heads in the multi-head attention. Defaults to 4.
            leaky_relu_alpha (float, optional): Alpha parameter for the loss. Defaults to 0.01.
            input_window_size (int, optional): Size of the input window. Defaults to 96.
            combination_type (String, optional): Type of combination to use. Defaults to 'attn-comb'. Either -> 'attn-comb', 'weighted-comb', 'addition-comb'
        """
        super().__init__()

        # Assertion to ensure activation_function is valid
        assert (
            activation_function in ACTIVATIONS
        ), f"Invalid activation_function. Please select from: {ACTIVATIONS}"
        assert (
            activation_function in ACTIVATIONS
        ), f"Invalid activation_function. Please select from: {ACTIVATIONS}"

        assert embedding_type in [
            None,
            "PosEmb",
            "RotaryEmb",
            "CombinedEmb",
        ], "Invalid embedding type, choose from -> None, 'PosEmb', 'RotaryEmb', 'CombinedEmb'"
        # Calculate the number of output targets, unknown features, and covariates
        self.n_out = n_target_series
        self.n_unknown = n_unknown_features + self.n_out
        self.n_covariates = n_known_calender_features + n_known_continuous_features
        self.n_channels = self.n_unknown + self.n_covariates

        self.out_activation = getattr(nn, out_activation_function)()
        self.activation = getattr(nn, activation_function)()

        # Initialize PastEncoder for processing past sequences
        self.encoder = PastFutureEncoder(
            emb_size=embedding_size,
            embed_type=embedding_type,
            latent_size=hidden_size,
            depth=num_layers,
            residual=residual,
            expansion_factor=expansion_factor,
            context_size=input_window_size,
            activation=self.activation,
            dropout=dropout_rate,
            n_channels=self.n_channels,
        )

        # Initialize FutureEncoder for processing future sequences
        if self.n_covariates > 0:
            self.horizon = PastFutureEncoder(
                emb_size=embedding_size,
                embed_type=embedding_type,
                latent_size=hidden_size,
                depth=num_layers,
                residual=residual,
                expansion_factor=expansion_factor,
                context_size=forecast_horizon,
                activation=self.activation,
                dropout=dropout_rate,
                n_channels=self.n_covariates,
            )

        # Hyperparameters and components for decoding
        self.window_size = input_window_size
        self.combination_type = combination_type
        self.alpha = alpha

        assert combination_type in [
            "attn-comb",
            "weighted-comb",
            "additional",
        ], "Invalid embedding type, choose from -> 'attn-comb', 'weighted-comb', 'additional'"

        if combination_type == "attn-comb":
            self.attention = nn.MultiheadAttention(
                hidden_size, num_attention_heads, dropout=dropout_rate
            )

        if combination_type == "weighted-comb":
            self.gate = nn.Linear(2 * hidden_size, hidden_size)

        self.comb_type = combination_type

        self.decoder = nn.Sequential(
            FeedForward(
                hidden_size,
                expansion_factor=1,
                dropout=dropout_rate,
                activation=self.activation,
                bn=True,
            )
        )

        self.mu = nn.Linear(hidden_size, self.n_out * forecast_horizon)

    def forecast(self, x):
        """
        Generates forecasts for the input sequences.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            dict: Dictionary containing the forecast predictions.
        """
        with torch.no_grad():
            pred = self(x)

        return {"pred": pred}

    def forward(self, x):
        """
        Forward pass of the MLPForecastNetwork.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after processing through the network.
        """

        # Process past sequences with the encoder
        f = self.encoder(x[:, : self.window_size, :])

        # Process future sequences with the horizon encoder
        if self.n_covariates > 0:
            h = self.horizon(x[:, self.window_size :, self.n_unknown :])
            if self.comb_type == "attn-comb":
                ph_hf = self.attention(h.unsqueeze(0), f.unsqueeze(0), f.unsqueeze(0))[
                    0
                ].squeeze(0)
            elif self.comb_type == "weighted-comb":
                # Compute the gate mechanism
                gate = self.gate(torch.cat((h, f), -1)).sigmoid()
                # Combine past and future information using the gate mechanism
                ph_hf = (1 - gate) * f + gate * h
            else:
                ph_hf = h + f
        else:
            ph_hf = f

        # Decode the combined information
        z = self.decoder(ph_hf)
        # Compute the final output
        loc = self.out_activation(self.mu(z).reshape(z.size(0), -1, self.n_out))

        return loc

    def step(self, batch, metric_fn):
        """
        Training step for the MLPForecastNetwork.

        Parameters:
            batch (tuple): Tuple containing input and target tensors.
            metric_fn (callable): Metric function to evaluate.

        Returns:
            tuple: Tuple containing the loss and computed metric.
        """
        x, y = batch

        # Forward pass to obtain predictions
        y_pred = self(x)

        # Calculate the loss
        loss = self.alpha * F.mse_loss(y_pred, y, reduction="none") + (
            1 - self.alpha
        ) * F.l1_loss(y_pred, y, reduction="none")
        loss = loss.sum(1).mean()

        # Compute the specified metric
        metric=metric_fn(y_pred, y)

        return loss, metric

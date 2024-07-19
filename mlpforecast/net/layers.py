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

    Args:
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

    Args:
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
    """
    Multi-Layer Perceptron (MLP) block with configurable layers and options.

    Attributes:
        mlp_network (nn.ModuleList): List of layers in the MLP network.
        in_size (int): Size of the input after flattening.
        context_size (int): Size of the context.
        residual (bool): If True, adds residual connections.
    """

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
    """
    Encoder module for the PastFutureNetwork.

    Attributes:
        encoder (MLPBlock): MLP block for the encoder.
        norm (nn.LayerNorm): Layer normalization.
        dropout (nn.Dropout): Dropout layer.
        embedding (nn.Module): Embedding layer.
        embedding_type (str): Type of embedding to use.
        rotary_embedding (RotaryEmbedding): Rotary positional embedding.
        pos_embedding (PosEmbedding): Positional embedding.
    """

    def __init__(
        self,
        embedding_size: int = 28,
        embedding_type: str = None,
        latent_size: int = 64,
        num_layers: int = 2,
        residual: bool = False,
        expansion_factor: int = 2,
        context_size: int = 96,
        activation: nn.Module = nn.ReLU(),
        dropout_rate: float = 0.25,
        n_channels: int = 1,
    ):
        """
        Initializes the PastFutureEncoder module.

        Args:
            embedding_size (int, optional): Dimensionality of the embedding space. Defaults to 28.
            embedding_type (str, optional): Type of embedding to use. Defaults to None.
            latent_size (int, optional): Dimensionality of the latent space. Defaults to 64.
            num_layers (int, optional): Number of layers in the encoder. Defaults to 2.
            residual (bool, optional): Whether to use residual connections in the encoder. Defaults to False.
            expansion_factor (int, optional): Expansion factor for the encoder. Defaults to 2.
            context_size (int, optional): Size of the context. Defaults to 96.
            activation (nn.Module, optional): Activation function. Defaults to nn.ReLU().
            dropout_rate (float, optional): Dropout probability. Defaults to 0.25.
            n_channels (int, optional): Number of channels in the input. Defaults
        """
        super().__init__()

        self.encoder = MLPBlock(
            in_size=embedding_size if embedding_type is not None else n_channels,
            latent_dim=latent_size,
            features_start=latent_size,
            expansion_factor=expansion_factor,
            residual=residual,
            num_layers=num_layers,
            context_size=context_size,
            activation=activation,
        )

        # Normalize the input using LayerNorm
        self.norm = nn.LayerNorm(n_channels)

        # Apply dropout to the input
        self.dropout = nn.Dropout(dropout_rate)

        # Store hyperparameters
        self.embedding_type = embedding_type

        # Embedding based on the specified type
        if embedding_type == "PosEmb":
            self.embedding = PosEmbedding(
                n_channels, embedding_size, window_size=context_size
            )
        elif embedding_type == "RotaryEmb":
            self.embedding = RotaryEmbedding(embedding_size)
        elif embedding_type == "CombinedEmb":
            self.pos_embedding = PosEmbedding(
                n_channels, embedding_size, window_size=context_size
            )
            self.rotary_embedding = RotaryEmbedding(embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PastFutureEncoder module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after processing through the encoder.
        """
        # Normalize the input
        x = self.norm(x)

        # Apply embedding based on the specified type
        if self.embedding_type == "CombinedEmb":
            x = self.pos_embedding(x) + self.rotary_embedding(x)
            # Apply dropout to the embedded input
            x = self.dropout(x)
        elif self.embedding_type in ["PosEmb", "RotaryEmb"]:
            x = self.embedding(x)
            # Apply dropout to the embedded input
            x = self.dropout(x)

        # Pass the input through the encoder
        x = self.encoder(x)

        return x


class MLPForecastNetwork(nn.Module):
    """
    Multilayer Perceptron (MLP) Forecast Network for time series forecasting.

    Attributes:
        n_out (int): Number of target series.
        n_unknown (int): Number of unknown time-varying features.
        n_covariates (int): Number of known time-varying features.
        n_channels (int): Number of channels in the input.
        input_window_size (int): Size of the input window.
        forecast_horizon (int): Number of future time steps to forecast.
        out_activation (torch.nn.Module): Output activation function.
        activation (torch.nn.Module): Activation function.
        encoder (PastFutureEncoder): Encoder module.
        horizon (PastFutureEncoder): Horizon encoder module.
        combination_type (str): Type of combination to use.
        alpha (float): Alpha parameter for the loss.
        attention (nn.MultiheadAttention): Multi-head attention module.
        gate (nn.Linear): Linear layer for weighted combination.
        decoder (nn.Sequential): Decoder module.
        mu (nn.Linear): Linear layer for output.
    """

    def __init__(
        self,
        n_target_series: int,
        n_unknown_features: int,
        n_known_calendar_features: int,
        n_known_continuous_features: int,
        embedding_size: int = 28,
        embedding_type: str = None,
        combination_type: str = "attn-comb",
        expansion_factor: int = 2,
        residual: bool = False,
        hidden_size: int = 256,
        num_layers: int = 2,
        forecast_horizon: int = 48,
        input_window_size: int = 96,
        activation_function: str = "SiLU",
        out_activation_function: str = "Identity",
        dropout_rate: float = 0.25,
        alpha: float = 0.1,
        num_attention_heads: int = 4,
    ):
        """
        Multilayer Perceptron (MLP) Forecast Network for time series forecasting.

        Args:
            n_target_series (int): Number of target series.
            n_unknown_features (int): Number of unknown time-varying features.
            n_known_calendar_features (int): Number of known categorical time-varying features.
            n_known_continuous_features (int): Number of known continuous time-varying features.
            embedding_size (int, optional): Dimensionality of the embedding space. Defaults to 28.
            embedding_type (str, optional): Type of embedding to use. Defaults to None. Options: 'PosEmb', 'RotaryEmb', 'CombinedEmb'.
            combination_type (str, optional): Type of combination to use.Defaults to 'attn-comb'. Options: 'attn-comb', 'weighted-comb', 'addition-comb'.
            expansion_factor (int, optional): Expansion factor for the encoder. Defaults to 2.
            residual (bool, optional): Whether to use residual connections in the encoder. Defaults to False.
            hidden_size (int, optional): Dimensionality of the hidden layers. Defaults to 256.
            num_layers (int, optional): Number of layers in the MLP. Defaults to 2.
            forecast_horizon (int, optional): Number of future time steps to forecast. Defaults to 48.
            input_window_size (int, optional): Size of the input window. Defaults to 96.
            activation_function (str, optional): Activation function. Defaults to 'SiLU'.
            out_activation_function (str, optional): Output activation function. Defaults to 'Identity'.
            dropout_rate (float, optional): Dropout probability. Defaults to 0.25.
            alpha (float, optional): Alpha parameter for the loss. Defaults to 0.1.
            num_attention_heads (int, optional): Number of heads in the multi-head attention. Defaults to 4.
        """
        super().__init__()

        # Ensure valid activation and embedding types
        assert (
            activation_function in ACTIVATIONS
        ), f"Invalid activation_function. Please select from: {ACTIVATIONS}"
        assert (
            out_activation_function in ACTIVATIONS
        ), f"Invalid out_activation_function. Please select from: {ACTIVATIONS}"
        assert (
            embedding_type
            in [
                None,
                "PosEmb",
                "RotaryEmb",
                "CombinedEmb",
            ]
        ), "Invalid embedding type, choose from: None, 'PosEmb', 'RotaryEmb', 'CombinedEmb'"

        self.n_out = n_target_series
        self.n_unknown = n_unknown_features + self.n_out
        self.n_covariates = n_known_calendar_features + n_known_continuous_features
        self.n_channels = self.n_unknown + self.n_covariates
        self.input_window_size = input_window_size
        self.forecast_horizon = forecast_horizon
        self.out_activation = getattr(nn, out_activation_function)()
        self.activation = getattr(nn, activation_function)()

        self.encoder = PastFutureEncoder(
            embedding_size=embedding_size,
            embedding_type=embedding_type,
            latent_size=hidden_size,
            num_layers=num_layers,
            residual=residual,
            expansion_factor=expansion_factor,
            context_size=input_window_size,
            activation=self.activation,
            dropout_rate=dropout_rate,
            n_channels=self.n_channels,
        )

        if self.n_covariates > 0:
            self.horizon = PastFutureEncoder(
                embedding_size=embedding_size,
                embedding_type=embedding_type,
                latent_size=hidden_size,
                num_layers=num_layers,
                residual=residual,
                expansion_factor=expansion_factor,
                context_size=forecast_horizon,
                activation=self.activation,
                dropout_rate=dropout_rate,
                n_channels=self.n_covariates,
            )

        self.combination_type = combination_type
        self.alpha = alpha

        assert (
            combination_type
            in [
                "attn-comb",
                "weighted-comb",
                "addition-comb",
            ]
        ), "Invalid combination type, choose from: 'attn-comb', 'weighted-comb', 'addition-comb'"

        if combination_type == "attn-comb":
            self.attention = nn.MultiheadAttention(
                hidden_size, num_attention_heads, dropout=dropout_rate
            )

        if combination_type == "weighted-comb":
            self.gate = nn.Linear(2 * hidden_size, hidden_size)

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


    def forecast(self, x: torch.Tensor) -> dict:
        """
        Generates forecasts for the input sequences.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            dict: Dictionary containing the forecast predictions.
        """
        with torch.no_grad():
            pred = self(x)

        return {"pred": pred}


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLPForecastNetwork.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after processing through the network.
        """
        f = self.encoder(x[:, : self.input_window_size, :])

        if self.n_covariates > 0:
            h = self.horizon(x[:, self.input_window_size :, self.n_unknown :])
            if self.combination_type == "attn-comb":
                ph_hf = self.attention(h.unsqueeze(0), f.unsqueeze(0), f.unsqueeze(0))[
                    0
                ].squeeze(0)
            elif self.combination_type == "weighted-comb":
                gate = self.gate(torch.cat((h, f), -1)).sigmoid()
                ph_hf = (1 - gate) * f + gate * h
            else:
                ph_hf = h + f
        else:
            ph_hf = f

        z = self.decoder(ph_hf)
        loc = self.out_activation(
            self.mu(z).reshape(z.size(0), self.forecast_horizon, self.n_out)
        )

        return loc


    def step(self, batch: tuple, metric_fn: callable) -> tuple:
        """
        Training step for the MLPForecastNetwork.

        Args:
            batch (tuple): Tuple containing input and target tensors.
            metric_fn (callable): Metric function to evaluate.

        Returns:
            tuple: Tuple containing the loss and computed metric.
        """
        x, y = batch

        y_pred = self(x)

        loss = (
            self.alpha * F.mse_loss(y_pred, y, reduction="none").sum(dim=(1, 2)).mean()
            + (1 - self.alpha)
            * F.l1_loss(y_pred, y, reduction="none").sum(dim=(1, 2)).mean()
        )

        metric = metric_fn(y_pred, y)

        return loss, metric

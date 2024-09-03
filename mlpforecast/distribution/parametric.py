import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from mlpforecast.net.layers import FeedForward 

class ParametricDistribution(nn.Module):
    """
    ParametricDistribution is a neural network model for forecasting multivariate 
    time series using parametric distributions (e.g., Laplace distribution) with a 
    specific mean and scale.

    Args:
        num_outputs (int): Number of output dimensions.
        hidden_size (int): Number of units in the hidden layers.
        forecast_horizon (int): Number of time steps to forecast.
        dropout_rate (float): Dropout rate for regularization.
        activation_function (nn.Module): Activation function for the hidden layers.
        output_activation_function (nn.Module): Activation function for the output layer.
    """

    def __init__(self, num_outputs=1, hidden_size=256, forecast_horizon=48, dropout_rate=0.25,
                 activation_function=nn.SiLU(), output_activation_function=nn.Identity()):
        super().__init__()
        
        self.decoder = nn.Sequential(
            FeedForward(hidden_size, expansion_factor=1, dropout=dropout_rate,
                        activation=activation_function, bn=False)
        )
        self.mean_layer = nn.Sequential(
            nn.Linear(hidden_size, forecast_horizon * num_outputs),
            output_activation_function
        )
        
        self.num_outputs = num_outputs
        self.forecast_horizon = forecast_horizon
        self.log_scale_layer = nn.Linear(hidden_size, forecast_horizon * num_outputs)

    def forward(self, z):
        pass

    def step(self, z, y, metric_fn):
        pass

    def forecast(self, z):
        pass

    def samples(self, n_samples):
        pass


class LaplaceDistribution(ParametricDistribution):
    """
    LaplaceForecasting is a neural network model for forecasting multivariate 
    time series using Laplace distributions with a specific mean and scale.

    Args:
        num_outputs (int): Number of output dimensions.
        hidden_size (int): Number of units in the hidden layers.
        forecast_horizon (int): Number of time steps to forecast.
        dropout_rate (float): Dropout rate for regularization.
        activation_function (nn.Module): Activation function for the hidden layers.
        output_activation_function (nn.Module): Activation function for the output layer.
    """

    def __init__(self, num_outputs=1, hidden_size=256, forecast_horizon=48, dropout_rate=0.25,
                 activation_function=nn.SiLU(), output_activation_function=nn.Identity()):
        super().__init__(num_outputs=num_outputs,
                         hidden_size=hidden_size, 
                         forecast_horizon=forecast_horizon, 
                         dropout_rate=dropout_rate,
                         activation_function=activation_function, 
                         output_activation_function=output_activation_function)
        
        self.log_scale_layer = nn.Linear(hidden_size, forecast_horizon * num_outputs)

    def forward(self, z):
        """
        Forward pass of the neural network.

        Args:
            z (torch.Tensor): Input tensor.

        Returns:
            tuple: A tuple containing the predicted means and scales.
        """
        batch_size = z.size(0)
        z = self.decoder(z)
        means = self.mean_layer(z).reshape(batch_size, self.forecast_horizon, self.num_outputs)
        log_scale = self.log_scale_layer(z).reshape(batch_size, self.forecast_horizon, self.num_outputs)

        scale = log_scale.mul(0.5).exp()
        return means, scale
    
    def get_dist(self, loc, scale):
        """
        Constructs a Laplace distribution.

        Args:
            loc (torch.Tensor): Mean tensor of the distribution.
            scale (torch.Tensor): Scale tensor of the distribution.

        Returns:
            Laplace: A Laplace distribution instance.
        """
        return dist.Laplace(loc=loc, scale=scale)
    
    def step(self, z, y, metric_fn, beta=0.5):
        """
        Performs a training step.

        Args:
            z (torch.Tensor): Input tensor.
            y (torch.Tensor): Target tensor.
            metric_fn (callable): Function to compute the evaluation metric.
            beta (float): Weight for combining MSE and L1 losses.

        Returns:
            tuple: A tuple containing the loss and the metric.
        """
        loc, scale = self(z)
        pred_dist = self.get_dist(loc, scale)
        negative_log_likelihood = -pred_dist.log_prob(y).sum(1).mean()

        mse_loss = F.mse_loss(loc, y, reduction="none").sum(dim=(1, 2)).mean()
        l1_loss = F.l1_loss(loc, y, reduction="none").sum(dim=(1, 2)).mean()
        combined_loss = beta * mse_loss + (1 - beta) * l1_loss

        loss = negative_log_likelihood + combined_loss
        metric = metric_fn(loc, y)
        return loss, metric
    
    def forecast(self, x):
        """
        Generates forecasts without gradient calculations.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            dict: A dictionary containing the mean, scale, and the distribution instance.
        """
        with torch.no_grad():
            loc, scale = self(x)
            forecast_dist = self.get_dist(loc, scale)
        
        return {
            "loc": loc, 
            "scale": scale, 
            "pdist": forecast_dist
        }
    
    def sample(self, x, num_samples=500):
        with torch.no_grad():
            loc, scale = self(x)
            forecast_dist = self.get_dist(loc, scale)
            return forecast_dist.sample((num_samples, ))


class MCDMultivariateNormal(ParametricDistribution):
    """
    MultivariateNormalForecasting is a neural network model for forecasting multivariate 
    time series using normal distributions with a specific mean and covariance structure.

    Args:
        num_outputs (int): Number of output dimensions.
        hidden_size (int): Number of units in the hidden layers.
        forecast_horizon (int): Number of time steps to forecast.
        dropout_rate (float): Dropout rate for regularization.
        activation_function (nn.Module): Activation function for the hidden layers.
        output_activation_function (nn.Module): Activation function for the output layer.
    """

    def __init__(self, num_outputs=1, hidden_size=256, forecast_horizon=48, dropout_rate=0.25,
                 activation_function=nn.SiLU(), output_activation_function=nn.Identity()):
        super().__init__(num_outputs=num_outputs,
                         hidden_size=hidden_size, 
                         forecast_horizon=forecast_horizon, 
                         dropout_rate=dropout_rate,
                         activation_function=activation_function, 
                         output_activation_function=output_activation_function)
        
        self.num_eta_params = num_outputs + (num_outputs * (num_outputs - 1)) // 2
        self.log_variance_layer = nn.Linear(hidden_size, forecast_horizon * self.num_eta_params)

    def _construct_diagonal_matrix(self, eta_d):
        """
        Constructs the diagonal matrix D^2 using exp(eta_d) for batched input.

        Args:
            eta_d (torch.Tensor): Input tensor containing the diagonal elements.

        Returns:
            tuple: A tuple containing the diagonal matrix D^2 and its inverse.
        """
        diag_matrix = torch.diag_embed(eta_d.mul(0.5).exp_())
        diag_matrix_inv = torch.diag_embed(1.0 / torch.diagonal(diag_matrix, dim1=-2, dim2=-1))
        return diag_matrix, diag_matrix_inv
    
    def _construct_upper_triangular_matrix(self, eta_T):
        """
        Constructs the upper triangular matrix T for batched input.

        Args:
            eta_T (torch.Tensor): Input tensor containing the upper triangular elements.

        Returns:
            tuple: A tuple containing the upper triangular matrix T and its transpose.
        """
        batch_size, time_steps, num_params = eta_T.shape
        upper_indices = torch.triu_indices(self.num_outputs, self.num_outputs, offset=1)
        base_matrix = torch.eye(self.num_outputs, device=eta_T.device).unsqueeze(0).unsqueeze(0)
        upper_triangular_matrix = base_matrix.clone().expand(batch_size, time_steps, -1, -1)
        
        upper_triangular_matrix = upper_triangular_matrix.clone()
        
        upper_triangular_matrix[:, :, upper_indices[0], upper_indices[1]] = eta_T
        upper_triangular_matrix_transpose = upper_triangular_matrix.transpose(-2, -1)
        return upper_triangular_matrix, upper_triangular_matrix_transpose
    
    def forward(self, z):
        """
        Forward pass of the neural network.

        Args:
            z (torch.Tensor): Input tensor.

        Returns:
            tuple: A tuple containing the predicted means and the covariance matrices.
        """
        batch_size = z.size(0)
        z = self.decoder(z)
        means = self.mean_layer(z).reshape(batch_size, self.forecast_horizon, self.num_outputs)
        log_variance = self.log_variance_layer(z).reshape(batch_size, self.forecast_horizon, self.num_eta_params)

        eta_d = log_variance[:, :, :self.num_outputs]
        eta_T = log_variance[:, :, self.num_outputs:]

        diag_matrix, diag_matrix_inv = self._construct_diagonal_matrix(eta_d=eta_d)
        upper_triangular_matrix, upper_triangular_matrix_transpose = self._construct_upper_triangular_matrix(eta_T=eta_T)

        precision_matrix = torch.matmul(upper_triangular_matrix_transpose, torch.matmul(diag_matrix_inv, upper_triangular_matrix))

        return means, precision_matrix, diag_matrix, upper_triangular_matrix
    
    def get_dist(self, means, precision_matrix):
        """
        Constructs a MultivariateNormal distribution.

        Args:
            means (torch.Tensor): Mean tensor of the distribution.
            precision_matrix (torch.Tensor): Precision matrix of the distribution.

        Returns:
            MultivariateNormal: A MultivariateNormal distribution instance.
        """
        return dist.MultivariateNormal(loc=means, precision_matrix=precision_matrix)
    
    def step(self, z, y, metric_fn, beta=0.5):
        """
        Performs a training step.

        Args:
            z (torch.Tensor): Input tensor.
            y (torch.Tensor): Target tensor.
            metric_fn (callable): Function to compute the evaluation metric.
            beta (float): Weight for combining MSE and L1 losses.

        Returns:
            tuple: A tuple containing the loss and the metric.
        """
        means, precision_matrix = self(z)[:2]
        pred_dist = self.get_dist(means, precision_matrix)
        negative_log_likelihood = -pred_dist.log_prob(y).sum(1).mean()

        mse_loss = F.mse_loss(means, y, reduction="none").sum(dim=(1, 2)).mean()
        l1_loss = F.l1_loss(means, y, reduction="none").sum(dim=(1, 2)).mean()
        median_loss = beta * mse_loss + (1 - beta) * l1_loss

        loss = negative_log_likelihood + median_loss
        metric = metric_fn(means, y)
        return loss, metric
    
    def forecast(self, x):
        """
        Generates forecasts without gradient calculations.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            dict: A dictionary containing the mean, precision matrix, diagonal matrix, 
                  upper triangular matrix, and the distribution instance.
        """
        with torch.no_grad():
            means, precision_matrix, diag_matrix, upper_triangular_matrix = self(x)
            forecast_dist = self.get_dist(means, precision_matrix)
        
        return {
            "loc": means, 
            "Sigma_inv": precision_matrix, 
            "D": diag_matrix, 
            "T": upper_triangular_matrix, 
            "pdist": forecast_dist
        }
    
    def sample(self, x, num_samples=500):
        with torch.no_grad():
            loc, scale = self(x)[:2]
            forecast_dist = self.get_dist(loc, scale)
            return forecast_dist.sample((num_samples, ))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantileLoss(torch.nn.Module):
    """
    Quantile Loss implementation for quantile regression.

    This module computes the quantile loss between predicted quantiles and target values.
    The loss is defined as the asymmetric loss based on specified quantiles.

    Args:
        quantiles (list of float): A list of quantiles for which to calculate the loss.
            Default is [0.05, 0.1, 0.5, 0.9, 0.95].

    Attributes:
        quantiles (list of float): The quantiles used in loss calculation.
    """

    def __init__(self, quantiles=[0.05, 0.1, 0.5, 0.9, 0.95]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, inputs, targets):
        """
        Compute the quantile loss.

        Args:
            inputs (torch.Tensor): Predicted quantiles of shape (B, N) where B is the batch size
                and N is the number of quantiles.
            targets (torch.Tensor): Ground truth target values of shape (B, 1) or (B, N).

        Returns:
            torch.Tensor: The mean quantile loss across the batch.
        """
        # Expand targets to match the shape of inputs
        targets = targets.unsqueeze(1).expand_as(inputs)

        # Convert quantiles to a tensor and move to the same device as targets
        quantiles = torch.tensor(self.quantiles).float().to(targets.device)

        # Calculate the error between targets and inputs
        error = (targets - inputs).permute(0, 2, 1)

        # Compute the quantile loss
        loss = torch.max(quantiles * error, (quantiles - 1) * error)

        return loss.mean()

class QuantileHuberLoss(nn.Module):
    """
    Computes the quantile Huber loss for quantile regression.

    This loss combines the properties of Huber loss with quantile estimation.

    Args:
        kappa (float): The threshold for Huber loss. Defaults to 1.0.
        eps (float): A small value to avoid division by zero. Defaults to 1e-6.
    """

    def __init__(self, kappa=1.0, eps=1e-6):
        super().__init__()
        self.kappa = kappa
        self.eps = eps

    def forward(self, td_errors, taus):
        """
        Calculate the quantile Huber loss.

        Args:
            td_errors (torch.Tensor): The temporal difference errors.
            taus (torch.Tensor): The quantiles corresponding to the errors.

        Returns:
            torch.Tensor: The computed quantile Huber loss.
        """
        # Avoid NaN when kappa is close to zero
        kappa = self.kappa + self.eps
        
        # Calculate element-wise Huber loss
        element_wise_huber_loss = self.calculate_huber_loss(td_errors, kappa)
        
        # Calculate element-wise quantile Huber loss
        quantile_mask = torch.abs(taus - (td_errors.detach() < 0).float())
        element_wise_quantile_huber_loss = quantile_mask * element_wise_huber_loss / kappa

        return element_wise_quantile_huber_loss

    @staticmethod
    def calculate_huber_loss(td_errors, kappa):
        """
        Calculate the Huber loss.

        Args:
            td_errors (torch.Tensor): The temporal difference errors.
            kappa (float): The threshold for Huber loss.

        Returns:
            torch.Tensor: The computed Huber loss.
        """
        return torch.where(
            td_errors.abs() <= kappa,
            0.5 * td_errors.pow(2),
            kappa * (td_errors.abs() - 0.5 * kappa)
        )
  

class QuantileProposalLoss(nn.Module):
    """
    Computes the quantile proposal loss for quantile regression.

    Args:
        None
    """

    def __init__(self):
        super().__init__()

    def forward(self, quantile, quantile_hats, taus):
        """
        Calculate the quantile proposal loss.

        Args:
            quantile (torch.Tensor): The target quantiles.
            quantile_hats (torch.Tensor): The predicted quantiles.
            taus (torch.Tensor): The quantile levels.

        Returns:
            torch.Tensor: The computed quantile proposal loss.
        """
        # Ensure that gradients are not tracked for these tensors
        assert not taus.requires_grad
        assert not quantile_hats.requires_grad
        assert not quantile.requires_grad

        value_1 = quantile - quantile_hats[:, :-1]
        signs_1 = quantile > torch.cat([quantile_hats[:, :1, :], quantile[:, :-1, :]], dim=1)

        value_2 = quantile - quantile_hats[:, 1:]
        signs_2 = quantile < torch.cat([quantile[:, 1:], quantile_hats[:, -1:]], dim=1)

        gradient_tau = (
            torch.where(signs_1, value_1, -value_1) +
            torch.where(signs_2, value_2, -value_2)
        ).view(*value_1.size())

        tau_loss = torch.mul(gradient_tau.detach(), taus[:, 1:-1, :]).sum(1).mean()
        return tau_loss


class SmoothPinballLoss(nn.Module):
    """
    Computes the smooth pinball loss for quantile regression.

    This implementation is based on the work by Hatalis et al. (2019).

    Args:
        alpha (float): Smoothing parameter for the loss. Defaults to 1e-2.
        kappa (float): Penalty parameter for smooth crossover. Defaults to 1e3.
        margin (float): Margin for the smooth penalty. Defaults to 1e-2.
    """

    def __init__(self, alpha=1e-2, kappa=1e3, margin=1e-2):
        super().__init__()
        self.alpha = alpha
        self.kappa = kappa
        self.margin = margin

    def forward(self, y, q, tau):
        """
        Calculate the smooth pinball loss.

        Args:
            y (torch.Tensor): Ground truth values.
            q (torch.Tensor): Predicted quantiles.
            tau (torch.Tensor): Quantile levels.

        Returns:
            torch.Tensor: The computed smooth pinball loss.
        """
        error = y - q
        q_loss = (tau * error + self.alpha * F.softplus(-error / self.alpha)).sum(0).mean()

        # Calculate smooth crossover penalty
        diff = q[1:, :, :] - q[:-1, :, :]
        penalty = self.kappa * torch.square(F.relu(self.margin - diff)).mean()

        loss = penalty + q_loss
        return loss

import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_reduction(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    """
    Apply the specified reduction method to the given loss tensor.

    Args:
        loss (torch.Tensor): The computed loss tensor.
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

    Returns
    -------
        torch.Tensor: The reduced loss tensor. If reduction is 'none', the original loss tensor is returned.
                      If reduction is 'mean' or 'sum', a scalar value is returned.

    Raises
    ------
        ValueError: If an invalid reduction type is specified.
    """
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction type: {reduction}. Choose from 'none', 'mean', or 'sum'.")


def quantile_penalty_loss(
    inputs: torch.Tensor,
    kappa: float = 0.0,
    margin: float = 1e-6,
    reduction: str = "none",
):
    diff = inputs[:, 1:, :, :] - inputs[:, :-1, :, :]
    loss = kappa * torch.square(F.relu(margin - diff)).sum(1)
    return loss_reduction(loss, reduction)


def pin_ball_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    quantiles: torch.Tensor,
    kappa: float = 0.0,
    reduction: str = "none",
):
    """
    Compute the pinball loss for given inputs, targets, and quantiles.

    The pinball loss is used in quantile regression and measures the accuracy of predicted quantiles.
    It is asymmetric and depends on the quantile level, penalizing overestimation and underestimation differently.

    Args:
        inputs (torch.Tensor): Predicted values of shape (N, Q, T, C), \
            where N is the batch size, Q is the number of quantiles,\
                  T is the number of targets, and C is the number of features.
        targets (torch.Tensor): True target values of shape (N, Q, T, C).
        quantiles (torch.Tensor): Quantile levels of shape (Q,).
        reduction (str, optional): Specifies \
            the reduction to apply to the output: 'none' | 'mean' | 'sum'. Default: 'none'.

    Returns
    -------
        torch.Tensor: The computed pinball loss. \
            If reduction is 'none', the shape is (N, Q, T, C).
                      If reduction is 'mean' or 'sum', a scalar value is returned.
    """
    # Ensure inputs and targets have the same size
    assert targets.size() == inputs.size(), "Targets and inputs must have the same size"
    # Ensure inputs and quantiles have compatible dimensions
    assert inputs.ndim == quantiles.ndim, "The number of quantiles must match between inputs and quantiles"

    # Calculate the error term
    error = targets - inputs

    # Compute the pinball loss
    if kappa > 0:
        loss = (quantiles * error + kappa * F.softplus(-error / kappa)).sum(1)

    else:
        loss = torch.max(quantiles * error, (quantiles - 1) * error).sum(1)

    return loss_reduction(loss, reduction)


def quantile_huber_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    quantiles: torch.Tensor,
    kappa: float = 0.25,
    eps: float = 1e-8,
    reduction: str = "none",
) -> torch.Tensor:
    r"""
    Compute the quantile Huber loss.

    This function calculates a Huber loss adjusted for quantiles, which is useful for quantile regression.
    The Huber loss is a combination of L1 and L2 loss, being less sensitive to outliers than L2.
    \\[
    L(y, \\hat{y}) = \begin{cases}
    \frac{1}{2}(y - \\hat{y})^2 & \text{if } |y - \\hat{y}| < \\kappa \\
    \\kappa \\cdot (|y - \\hat{y}| - \frac{1}{2}\\kappa) & \text{if } |y - \\hat{y}| \\geq \\kappa
    \\end{cases}
    \\]

    where the Huber loss is defined as:
        loss = 0.5 * (y - pred)² if |y - pred| < kappa
        loss = kappa * (|y - pred| - 0.5 * kappa) if |y - pred| ≥ kappa

    Args:
        inputs (torch.Tensor): Predicted values of shape (N, Q, T, C),\
        where N is the batch size, Q is the number of quantiles, \
            T is the number of targets, and C is the number of features.
        targets (torch.Tensor): True target values of shape (N, Q, T, C).
        quantiles (torch.Tensor): Quantile levels of shape (Q,).
        kappa (float, optional): Threshold at which to switch from L2 loss to L1 \
            loss. Default: 1.0.
        eps (float, optional): A small value to prevent division by zero. \
            Default: 1e-6.
        reduction (str, optional): Specifies the reduction to apply to the output:\
              'none' | 'mean' | 'sum'. Default: 'none'.

    Returns
    -------
        torch.Tensor: The computed quantile Huber loss. \
            If reduction is 'none', the shape is (N, Q, T, C).
            If reduction is 'mean' or 'sum', a scalar value is returned.
    """
    # Ensure inputs and targets have the same size
    assert targets.size() == inputs.size(), "Targets and inputs must have the same size"

    # Calculate the error term
    errors = targets - inputs

    # Compute the Huber loss

    huber_loss = torch.where(errors.abs() <= kappa, 0.5 * errors.pow(2), kappa * (errors.abs() - 0.5 * kappa))

    # huber_loss = F.huber_loss(inputs, targets, reduction="none", delta=kappa)

    # Calculate element-wise quantile Huber loss
    quantile_mask = torch.abs(quantiles - (errors.detach() < 0).float())
    loss = (quantile_mask * (huber_loss / (kappa + eps))).sum(1)

    return loss_reduction(loss, reduction)


def quantile_proposal_loss(
    quantile: torch.Tensor,
    quantile_hats: torch.Tensor,
    taus: torch.Tensor,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Compute the quantile proposal loss.

    This function calculates the loss for quantile proposals, ensuring that the predicted quantiles
    do not cross each other and adhere to the specified quantile levels.

    Args:
        quantile (torch.Tensor): True quantile values of shape (N, Q, T, C).
        quantile_hats (torch.Tensor): Predicted quantile values of shape (N, Q, T, C).
        taus (torch.Tensor): Quantile levels of shape (N, Q, T).

    Returns
    -------
        torch.Tensor: The computed quantile proposal loss as a scalar.
    """
    # Ensure that gradients are not required for the input tensors
    assert not taus.requires_grad, "Quantile levels (taus) should not require gradients"
    assert not quantile_hats.requires_grad, "Predicted quantile values (quantile_hats) should not require gradients"
    assert not quantile.requires_grad, "True quantile values (quantile) should not require gradients"

    # Calculate the value differences
    value_1 = quantile - quantile_hats[:, :-1]
    value_2 = quantile - quantile_hats[:, 1:]

    # Determine the signs for the loss computation
    signs_1 = quantile > torch.cat([quantile_hats[:, :1, :], quantile[:, :-1, :]], dim=1)
    signs_2 = quantile < torch.cat([quantile[:, 1:, :], quantile_hats[:, -1:, :]], dim=1)

    # Compute the gradient of tau
    gradient_tau = (torch.where(signs_1, value_1, -value_1) + torch.where(signs_2, value_2, -value_2)).view(
        *value_1.size()
    )

    # Compute the tau loss
    loss = torch.mul(gradient_tau.detach(), taus[:, 1:-1, :]).sum(1)

    return loss_reduction(loss, reduction)


class QuantileLoss(torch.nn.Module):
    """
    Quantile Loss implementation for quantile regression.

    This module computes the quantile loss between predicted quantiles and target values.
    The loss is defined as the asymmetric loss based on specified quantiles.

    Args:
        quantiles (list of float): A list of quantiles for which to calculate the loss.
            Default is [0.05, 0.1, 0.5, 0.9, 0.95].

    Attributes
    ----------
        quantiles (list of float): The quantiles used in loss calculation.
    """

    def __init__(self, kappa: float = 0.0, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        self.kappa = kappa

    def forward(self, inputs, quantiles, targets):
        """
        Compute the quantile loss.

        Args:
            inputs (torch.Tensor): Predicted quantiles of shape (B, N) where B is the batch size
                and N is the number of quantiles.
            targets (torch.Tensor): Ground truth target values of shape (B, 1) or (B, N).

        Returns
        -------
            torch.Tensor: The mean quantile loss across the batch.
        """
        return pin_ball_loss(inputs, targets, quantiles, kappa=self.kappa, reduction=self.reduction)


class QuantileHuberLoss(nn.Module):
    """
    Compute the quantile Huber loss.

    This class implements a quantile Huber loss function for use in quantile regression.
    The Huber loss is less sensitive to outliers compared to the L2 loss, making it useful
    for robust regression tasks.

    Args:
        kappa (float, optional): Threshold at which to switch from L2 loss to L1 loss. Default: 1.0.
        eps (float, optional): A small value to prevent division by zero. Default: 1e-6.
        reduction (str, optional): Specifies the reduction to apply to the output:\
              'none' | 'mean' | 'sum'. Default: 'mean'.

    Methods
    -------
        forward(inputs: torch.Tensor, quantiles: torch.Tensor, targets: torch.Tensor) -> torch.Tensor
            Computes the quantile Huber loss between the predicted and target values.
    """

    def __init__(self, kappa=1.0, eps=1e-8, reduction="mean"):
        super().__init__()
        self.kappa = kappa
        self.eps = eps
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, quantiles: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the quantile Huber loss.

        Args:
            inputs (torch.Tensor): Predicted values of shape (N, Q, T, C), \
                where N is the batch size, Q is the number of quantiles, \
                    T is the number of targets, and C is the number of features.
            quantiles (torch.Tensor): Quantile levels of shape (Q,).
            targets (torch.Tensor): True target values of shape (N, Q, T, C).

        Returns
        -------
            torch.Tensor: The computed quantile Huber loss. If reduction is 'none', the shape is (N, Q, T, C).
                          If reduction is 'mean' or 'sum', a scalar value is returned.
        """
        return quantile_huber_loss(
            inputs,
            targets,
            quantiles,
            kappa=self.kappa,
            reduction=self.reduction,
        )


class QuantileProposalLoss(nn.Module):
    """
    Compute the quantile proposal loss.

    This class implements a quantile proposal loss function to ensure that the predicted quantiles
    do not cross each other and adhere to the specified quantile levels.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:\
              'none' | 'mean' | 'sum'. Default: 'none'.

    Methods
    -------
        forward(quantile: torch.Tensor, quantile_hats: torch.Tensor, taus: torch.Tensor) -> torch.Tensor
            Computes the quantile proposal loss between the predicted and target values.
    """

    def __init__(self, reduction: str = "none"):
        super().__init__()
        self.reduction = reduction

    def forward(self, quantile: torch.Tensor, quantile_hats: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the quantile proposal loss.

        Args:
            quantile (torch.Tensor): True quantile values of shape (N, Q, T, C).
            quantile_hats (torch.Tensor): Predicted quantile values of shape (N, Q, T, C).
            taus (torch.Tensor): Quantile levels of shape (N, Q, T).

        Returns
        -------
            torch.Tensor: The computed quantile proposal loss as a scalar.
        """
        return quantile_proposal_loss(quantile, quantile_hats, taus, reduction=self.reduction)

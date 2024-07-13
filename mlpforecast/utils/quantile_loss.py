import numpy as np
import torch
from torch import Tensor


def huber_l12_loss(
    y: Tensor,
    pred: Tensor,
    kappa: float = 0.1,
    reduction: str = "none",
    dim=1
) -> Tensor:
    """Compute the Huber  loss.

    The Huber loss is defined as:
        loss = 0.5 * (y - pred)² if |y - pred| < kappa
        loss = kappa * (|y - pred| - 0.5 * kappa) if |y - pred| ≥ kappa

    Args:
        y (Tensor): The target values.
        pred (Tensor): The predicted values.
        kappa (float, optional): The threshold for switching between L1 and L2 loss.
            Default is 0.1.
        reduction (str, optional): Specifies the reduction to apply to the output.
            Can be 'none', 'mean', or 'sum'. Default is 'none'.

    Returns:
        Tensor: The computed Huber quantile loss.
    """
    
    # Ensure the inputs have the same shape
    assert y.size() == pred.size(), "The shape of y and pred must be the same."

    # Calculate the absolute errors
    errors = y - pred

    # Calculate Huber loss based on the threshold kappa
    loss = torch.where(
        errors.abs() <= kappa,
        0.5 * errors.pow(2),
        kappa * (errors.abs() - 0.5 * kappa)
    )

    # Apply the specified reduction method
    if reduction == "mean":
        return loss.mean(dim=dim)
    elif reduction == "sum":
        return loss.sum(dim=dim)
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction type: {reduction}. Use 'none', 'mean', or 'sum'.")


def quantile_huber_loss(
    y: Tensor,
    pred: Tensor,
    taus: Tensor,
    kappa: float = 0.1,
    reduction: str = "none",
    dim: int = 1,
    eps: float = 1e-9
) -> Tensor:
    """
    Computes the quantile Huber loss.

    The quantile Huber loss is defined as:
    
    \[
    L(y, \hat{y}) = \begin{cases}
    \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| < \kappa \\
    \kappa \cdot (|y - \hat{y}| - \frac{1}{2}\kappa) & \text{if } |y - \hat{y}| \geq \kappa
    \end{cases}
    \]

    The quantile adjustment is applied by scaling the Huber loss using an indicator function based on the difference between the predicted and true values.

    Args:
        y (Tensor): The target values.
        pred (Tensor): The predicted values.
        taus (Tensor): The quantile levels for the loss calculation.
        kappa (float, optional): The threshold for switching between L1 and L2 loss. Default is 0.1.
        reduction (str, optional): Specifies the reduction to apply to the output. Can be 'none', 'mean', or 'sum'. Default is 'none'.
        dim (int, optional): The dimension to reduce. Default is 1.
        eps (float, optional): A small constant added to kappa to prevent division by zero. Default is 1e-9.

    Returns:
        Tensor: The computed quantile Huber loss.
    """
    
    # Adjust kappa to avoid division by zero
    kappa += eps

    # Calculate element-wise Huber loss
    loss = huber_l12_loss(y, pred, kappa)

    # Create a mask indicating where the errors are negative
    mask = (taus - (y - pred).detach() < 0).float()

    # Scale the loss by the mask and normalize by kappa
    loss = mask * loss / kappa

    # Apply the specified reduction method
    if reduction == "mean":
        return loss.mean(dim=dim)
    elif reduction == "sum":
        return loss.sum(dim=dim)
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction type: {reduction}. Use 'none', 'mean', or 'sum'.")
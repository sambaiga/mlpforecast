import numpy as np


def plot_prediction(ax, true, mu, date=None, true_max=None):
    """
    Plots the true values and predicted values on the provided axes.

    Args:
        ax (matplotlib.axes.Axes): The axes on which to plot.
        true (array-like): The true values to be plotted.
        mu (array-like): The predicted values to be plotted.
        date (array-like, optional): The date or time values for the x-axis. If None, uses an array of indices.
        true_max (float, optional): The maximum value of the true values for scaling. If None, it is computed from `true`.

    Returns:
        tuple: A tuple containing the modified axes, the list of line objects, and the list of labels.
    """
    # Set default date range if not provided
    date = np.arange(len(true)) if date is None else date

    # Plot true values
    true_line, = ax.plot(date, true, ".", mec="#ff7f0e", mfc="None", label='True')

    # Plot predicted values
    pred_line, = ax.plot(date, mu, c="#1f77b4", alpha=0.8, label='Pred')

    # Set y-axis label
    ax.set_ylabel('Power (W)')

    # Auto-scale the axes tightly
    ax.autoscale(tight=True)

    # Determine the maximum value for true values if not provided
    if true_max is None:
        true_max = np.max(true)

    # Return axes, line objects, and labels
    leg = ax.legend([true_line, pred_line], ['True', 'Pred'], \
                    loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        
        
       
    return ax
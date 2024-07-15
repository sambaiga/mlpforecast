import altair as alt
import numpy as np
import pandas as pd
import seaborn as sns

alt.themes.enable("opaque")
alt.data_transformers.disable_max_rows()
colors = [
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#f69a48",
    "#00c0bf",
    "#fdcd49",
    "#8da798",
    "#a19368",
    "#525252",
    "#a6761d",
    "#7035b7",
    "#cf166e",
]


def plot_prediction(ax, true, mu, date=None, true_max=None):
    """
    Plots the true values and predicted values on the provided axes.

    Args:
        ax (matplotlib.axes.Axes): The axes on which to plot.
        true (array-like): The true values to be plotted.
        mu (array-like): The predicted values to be plotted.
        date (array-like, optional): \
            The date or time values for the x-axis. If None, uses an array of indices.
        true_max (float, optional): \
            The maximum value of the true values for scaling. If None, it is computed from `true`.

    Returns:
        tuple: A tuple containing the modified axes, the list of line objects, and the list of labels.
    """
    # Set default date range if not provided
    date = np.arange(len(true)) if date is None else date

    # Plot true values
    (true_line,) = ax.plot(date, true, ".", mec="#ff7f0e", mfc="None", label="True")

    # Plot predicted values
    (pred_line,) = ax.plot(date, mu, c="#1f77b4", alpha=0.8, label="Pred")

    # Set y-axis label
    ax.set_ylabel("Power (W)")

    # Auto-scale the axes tightly
    ax.autoscale(tight=True)

    # Determine the maximum value for true values if not provided
    if true_max is None:
        true_max = np.max(true)

    # Return axes, line objects, and labels
    ax.legend(
        [true_line, pred_line],
        ["True", "Pred"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
    )

    return ax


def plot_correlation(ax, corr_df, cmap=sns.diverging_palette(240, 10, as_cmap=True)):
    """
    Plots a heatmap of the correlation matrix.

    Parameters:
        ax (matplotlib.axes.Axes): The axes on which to plot the heatmap.
        corr_df (pandas.DataFrame): \
            DataFrame containing the correlation data with three columns: \
                two for the pairs of items and one for the correlation values.
        cmap (matplotlib.colors.Colormap, optional): \
            Colormap to use for the heatmap. Default is a diverging palette from Seaborn.

    Returns:
        matplotlib.axes.Axes: The Axes object with the heatmap.
    """
    columns = list(corr_df.columns)
    corr = corr_df.pivot(index=columns[1], columns=columns[0], values=columns[-1])
    ax = sns.heatmap(corr.T, linewidths=0.5, cmap=cmap, center=0, annot=True, fmt=".1g")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, horizontalalignment="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment="right")
    ax.set_title("")
    ax.set_ylabel("")
    ax.set_xlabel("")
    return ax


def scatter_plot(
    data, variables, targets, hue_col=None, n_sample=1000, random_state=111
):
    """
    Creates a scatter plot matrix using Altair.

    Parameters:
        data (pandas.DataFrame): The data to plot.
        variables (list of str): List of column names to be used as variables for the x-axis.
        targets (list of str): List of column names to be used as targets for the y-axis.
        hue_col (str, optional): Column name for the color encoding. Default is None.
        n_sample (int, optional): Number of samples to draw from the data for plotting. Default is 1000.
        random_state (int, optional): Seed for random sampling. Default is 111.

    Returns:
        alt.Chart: The Altair chart object with the scatter plot matrix.
    """
    data = (
        data.sample(n=n_sample, random_state=random_state)
        if n_sample is not None
        else data
    )
    chart = alt.Chart(data)
    if hue_col is not None:
        chart = chart.mark_point(filled=True, opacity=0.7).encode(
            alt.X(
                alt.repeat("column"), type="quantitative", scale=alt.Scale(zero=False)
            ),
            alt.Y(alt.repeat("row"), type="quantitative", scale=alt.Scale(zero=False)),
            color=f"{hue_col}:N",
        )
    else:
        chart = chart.mark_point(filled=True, opacity=0.7).encode(
            alt.X(
                alt.repeat("column"), type="quantitative", scale=alt.Scale(zero=False)
            ),
            alt.Y(alt.repeat("row"), type="quantitative", scale=alt.Scale(zero=False)),
        )

    chart = (
        chart.properties(width=150, height=150)
        .repeat(row=[targets], column=variables)
        .configure_axis(grid=False, labelFontSize=12, titleFontSize=12)
        .configure_view(strokeOpacity=0)
    )
    return chart


def visualise_timeseries_altair(data, y_col, figure_path=None, y_label="Power (kW)"):
    """
    Visualizes time series data using Altair.

    Parameters:
    data (pandas.DataFrame): The data to plot, with a datetime index and the columns to be plotted.
    y_col (list of str): List of column names to plot on the y-axis.
    figure_path (str, optional): Path to save the figure. If None, the figure is not saved. Default is None.
    y_label (str, optional): Label for the y-axis. Default is 'Power (kW)'.
    colors (list of str, optional): List of colors for the lines. Default is ['blue', 'red', 'green', 'purple'].

    Returns:
    alt.Chart: The Altair chart object with the time series plot.
    """
    chart = (
        alt.Chart(data.reset_index())
        .mark_point(filled=True, opacity=0.7)
        .encode(
            x=alt.X(
                "timestamp:T", scale=alt.Scale(zero=False), axis=alt.Axis(title="Date")
            ),
            y=alt.X(f"{y_col[0]}:Q", scale=alt.Scale(zero=False), title=y_label),
            color=alt.value(colors[0]),
        )
    )
    if len(y_col) > 1:
        for i in range(1, len(y_col)):
            chart += (
                alt.Chart(data.reset_index())
                .mark_point(filled=True, opacity=0.7)
                .encode(
                    x=alt.X("timestamp:T", axis=alt.Axis(title="Date")),
                    y=alt.X(f"{y_col[i]}:Q", title=y_label),
                    color=alt.value(colors[i]),
                )
            )
    chart = (
        chart.configure_axis(grid=False, labelFontSize=12, titleFontSize=12)
        .configure_view(strokeOpacity=0)
        .properties(width=900, height=100)
    )

    return chart


def plot_kde_(ax, data, x_col, hue_col, label):
    """
    Plot a Kernel Density Estimate (KDE) and histogram on the given axes.

    Parameters:
    ax (matplotlib.axes.Axes): The axes on which to plot.
    data (DataFrame): The data to plot.
    x_col (str): The column in `data` to plot on the x-axis.
    hue_col (str): The column in `data` to use for color grouping.
    label (str): The label for the x-axis.

    Returns:
    matplotlib.axes.Axes: The axes with the plot.
    """
    sns.histplot(data, x=x_col, ax=ax, hue=hue_col, palette="tab20", kde=True)
    ax.autoscale()
    ax.set_xlabel(label)
    return ax


def plot_cdf_(ax, data, x_col, hue_col, label):
    """
    Plot a Cumulative Distribution Function (CDF) on the given axes.

    Parameters:
    ax (matplotlib.axes.Axes): The axes on which to plot.
    data (DataFrame): The data to plot.
    x_col (str): The column in `data` to plot on the x-axis.
    hue_col (str): The column in `data` to use for color grouping.
    label (str): The label for the x-axis.

    Returns:
    matplotlib.axes.Axes: The axes with the plot.
    """
    sns.kdeplot(
        data,
        x=x_col,
        ax=ax,
        hue=hue_col,
        cumulative=True,
        common_norm=False,
        common_grid=False,
        palette="tab20",
    )
    ax.autoscale()
    ax.set_xlabel(label)
    ax.set_ylim(0, 1)
    return ax


def plot_distribution(ax, df, index_col="HOUR", val_col="WindSpeed", hue_col=None):
    """
    Plot the distribution of a specified variable with mean and standard deviation bands.

    Parameters:
    ax (matplotlib.axes.Axes): The axes on which to plot.
    df (pd.DataFrame): The data frame containing the data.
    index_col (str): The column in `df` to use as the index for pivoting.
    val_col (str): The column in `df` containing the values to plot.
    hue_col (str, optional): The column in `df` to use for color grouping.

    Returns:
    matplotlib.axes.Axes: The axes with the plot.

    This function creates a pivot table to \
        calculate the mean and standard deviation of `val_col`
    grouped by `index_col` and optionally by `hue_col`. \
          It then plots the mean values with bands
    representing one standard deviation above and below the mean.
    """
    # Calculate mean values for the pivot table
    mean_values = pd.pivot_table(
        df, index=index_col, values=val_col, columns=hue_col, aggfunc=np.mean
    )
    # Calculate standard deviation values for the pivot table
    std_dev = pd.pivot_table(
        df, index=index_col, values=val_col, columns=hue_col, aggfunc=np.std
    )

    # Plot the mean values
    mean_values.plot(ax=ax)

    # Fill the area between mean ± standard deviation
    ax.fill_between(
        np.arange(len(mean_values)),
        (mean_values - std_dev).min(axis=1),
        (mean_values + std_dev).max(axis=1),
        color="lightsteelblue",
        alpha=0.5,
    )

    # Set plot labels and title
    ax.set_xlabel(index_col)
    ax.set_ylabel(val_col)
    ax.set_title(f"Distribution of {val_col} by {index_col}")

    return ax

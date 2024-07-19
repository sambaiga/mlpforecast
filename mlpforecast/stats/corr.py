import numpy as np
import pandas as pd
import ppscore as pps
from scipy.stats import norm, rankdata

from mlpforecast.plot.visual_functions import plot_correlation, scatter_plot


class CorrelationAnalyzer:
    """
    A class to calculate and visualize the correlation between variables in a data frame.
    """

    @staticmethod
    def corr(
        data,
        variable_col,
        target_col,
        method="scatter",
        ties="auto",
        hue_col=None,
        n_sample=None,
    ):
        """
        Calculate the correlation between the target column and other variables in the data frame.

        Args:
            data (pandas.DataFrame): The data frame containing the data.
            variable_col (list of str): List of column names to be used as independent variables.
            target_col (str): The name of the dependent variable column.
            method (str, optional): The method to use for calculating the correlation:
                - 'scatter' (default): Scatter plot.
                - 'pearson': Pearson correlation.
                - 'kendall': Kendall rank correlation.
                - 'spearman': Spearman rank correlation.
                - 'ppscore': Predictive Power Score (PPS).
                - 'xicor': Xi correlation.
            ties (str or bool, optional): How to handle ties in Xi correlation calculation:
                - 'auto' (default): Decide based on the uniqueness of y values.
                - True: Assume ties are present.
                - False: Assume no ties are present.
            hue_col (str, optional): The column in `data` to use for color grouping.
            n_sample (int, optional): The number of samples to use for the scatter

        Returns:
            pandas.DataFrame: DataFrame containing the correlation between the target column and each variable.

        Raises:
            ValueError: If the method is not supported.
        """
        if method == "scatter":
            return scatter_plot(
                data, variable_col, target_col, hue_col, n_sample=n_sample
            )
        elif method in ["pearson", "kendall", "spearman"]:
            return CorrelationAnalyzer._get_correlation(data, variable_col, target_col)
        elif method == "ppscore":
            return CorrelationAnalyzer._get_ppscore(data, variable_col, target_col)
        elif method == "xicor":
            return CorrelationAnalyzer._get_xicor_score(
                data, variable_col, target_col, ties
            )
        else:
            raise ValueError(
                f"Unsupported method: {method}. Choose from 'pearson', 'kendall', 'spearman', 'ppscore', or 'xicor'."
            )

    @staticmethod
    def plot(ax, corr_df):
        """
        Plot the correlation data using a heatmap.

        Args:
            ax (matplotlib.axes.Axes): The axes on which to plot the heatmap.
            corr_df (pandas.DataFrame): DataFrame containing the correlation data with three columns: \
                    two for the pairs of items and one for the correlation values.
        """
        return plot_correlation(ax, corr_df)

    @staticmethod
    def _get_correlation(data, variable_col, target_col):
        """
        Calculate the Pearson correlation between the target column and other variables.

        Args:
            data (pandas.DataFrame): The data frame containing the data.
            variable_col (list of str): List of column names to be used as independent variables.
            target_col (str): The name of the dependent variable column.

        Returns:
            pandas.DataFrame: DataFrame containing the Pearson correlation between the target column and each variable.
        """
        non_zero_varlist = [target_col] + variable_col
        correlations = (
            data[non_zero_varlist]
            .corr(method="pearson")
            .unstack()
            .sort_values(ascending=False)
        )
        correlations = pd.DataFrame(correlations).reset_index()
        correlations.columns = ["col1", "col2", "correlation"]
        _corr = correlations.query(f"col1 == '{target_col}' & col2 != '{target_col}'")
        return _corr

    @staticmethod
    def _get_ppscore(data, variable_col, target_col):
        """
        Calculate the Predictive Power Score (PPS) between the target column and other variables.

        Args:
            data (pandas.DataFrame): The data frame containing the data.
            variable_col (list of str): List of column names to be used as independent variables.
            target_col (str): The name of the dependent variable column.

        Returns:
            pandas.DataFrame: DataFrame containing the PPS between the target column and each variable.
        """
        _pscore = pps.predictors(data[variable_col + [target_col]], y=target_col)
        _pscore = _pscore[["y", "x", "ppscore"]]
        _pscore.columns = ["col1", "col2", "ppscore"]
        return _pscore

    @staticmethod
    def _get_xicor_score(data, variable_col, target_col, ties="auto"):
        """
        Calculate the Xi correlation for multiple variable-target pairs and return a sorted DataFrame.

        Args:
            data (pandas.DataFrame): The data frame containing the data.
            variable_col (list of str): List of column names to be used as independent variables.
            target_col (str): The name of the dependent variable column.

        Returns:
            pandas.DataFrame: DataFrame containing the Xi correlation between the target column and each variable.
        """
        scores = [
            CorrelationAnalyzer._xicordf(data, x, target_col, ties)
            for x in variable_col
        ]
        scores.sort(key=lambda item: item["xicor"], reverse=True)

        df_columns = ["x", "y", "xicor", "p-value"]
        data_dict = {
            column: [score[column] for score in scores] for column in df_columns
        }
        scores_df = pd.DataFrame.from_dict(data_dict)
        scores_df = scores_df[["y", "x", "xicor"]]
        scores_df.columns = ["col1", "col2", "xicor"]
        return scores_df

    @staticmethod
    def _xicordf(data, x_col, y_col, ties="auto"):
        """
        Calculate the Xi correlation for specified columns in a DataFrame.

        Args:
            data (pandas.DataFrame): The data frame containing the data.
            x_col (str): The name of the independent variable column.
            y_col (str): The name of the dependent variable column.
            ties (str or bool, optional): How to handle ties in Xi correlation calculation:
                - 'auto' (default): Decide based on the uniqueness of y values.
                - True: Assume ties are present.
                - False: Assume no ties are present.

        Returns:
            dict: Dictionary containing the x, y, Xi correlation, and p-value.
        """
        x = data[x_col].values
        y = data[y_col].values
        xicor, p_value = CorrelationAnalyzer._get_xicor(x, y, ties=ties)
        return {"x": x_col, "y": y_col, "xicor": xicor, "p-value": p_value}

    @staticmethod
    def _get_xicor(x, y, ties="auto"):
        """
        Calculate the Xi correlation coefficient and p-value between two arrays.

        Args:
            x (array-like): The first array.
            y (array-like): The second array.
            ties (str or bool, optional): How to handle ties in Xi correlation calculation:
                - 'auto' (default): Decide based on the uniqueness of y values.
                - True: Assume ties are present.
                - False: Assume no ties are present.

        Returns:
            tuple: Tuple containing the Xi correlation coefficient and p-value.

        Raises:
            IndexError: If the lengths of x and y do not match.
            ValueError: If the ties argument is not a boolean or
        """
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        n = len(y)

        if len(x) != n:
            raise IndexError(f"x, y length mismatch: {len(x)}, {len(y)}")

        if ties == "auto":
            ties = len(np.unique(y)) < n
        elif not isinstance(ties, bool):
            raise ValueError(
                f"Expected ties to be either 'auto' or boolean, got {ties} ({type(ties)}) instead"
            )

        y = y[np.argsort(x)]
        r = rankdata(y, method="ordinal")
        nominator = np.sum(np.abs(np.diff(r)))

        if ties:
            ldata = rankdata(y, method="max")
            denominator = 2 * np.sum(ldata * (n - ldata))
            nominator *= n
        else:
            denominator = np.power(n, 2) - 1
            nominator *= 3

        statistic = 1 - nominator / denominator
        p_value = norm.sf(statistic, scale=2 / 5 / np.sqrt(n))

        return statistic, p_value

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from mlpforecast.plot.visual_functions import scatter_plot, plot_correlation
import ppscore as pps


class CorrelationAnalyzer:
    """
    Class to calculate different types of correlations: Pearson, PPS, or Xi.

    Methods
    -------
    calculate(data, variable_col, target_col, method='pearson', ties='auto')
        Calculates the specified correlation measure between the target column and other variables.
    """

    @staticmethod
    def corr(data, variable_col, target_col, method='scatter', ties='auto', hue_col=None, n_sample=None):
        """
        Calculate the specified correlation measure between the target column and other variables.

        Parameters
        ----------
        data : pandas.DataFrame
            The data frame containing the data.
        variable_col : list of str
            List of column names to be used as independent variables.
        target_col : str
            The name of the dependent variable column.
        method : str, optional
            The correlation measure to use: 'pearson', 'ppscore', or 'xicor' (default is 'pearson').
        ties : {'auto', bool}, optional
            How to handle ties in Xi correlation calculation:
            - 'auto' (default): Decide based on the uniqueness of y values.
            - True: Assume ties are present.
            - False: Assume no ties are present.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the correlation measure between the target column and each variable.
        
        Raises
        ------
        ValueError
            If an unsupported method is provided.
        """
        if method=='scatter':
            return scatter_plot(data, variable_col, target_col, hue_col, n_sample=n_sample)
        elif method in [ 'pearson', 'kendall', 'spearman']:
            return CorrelationAnalyzer._get_correlation(data, variable_col, target_col)
        elif method == 'ppscore':
            return CorrelationAnalyzer._get_ppscore(data, variable_col, target_col)
        elif method == 'xicor':
            return CorrelationAnalyzer._get_xicor_score(data, variable_col, target_col, ties)
        else:
            raise ValueError(f"Unsupported method: {method}. Choose from 'pearson', 'kendall', 'spearman', 'ppscore', or 'xicor'.")


    @staticmethod
    def plot(ax, corr_df):
        return plot_correlation(ax, corr_df)
    
    @staticmethod
    def _get_correlation(data, variable_col, target_col):
        """
        Calculate the Pearson correlation between the target column and other variables.

        Parameters
        ----------
        data : pandas.DataFrame
            The data frame containing the data.
        variable_col : list of str
            List of column names to be used as independent variables.
        target_col : str
            The name of the dependent variable column.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the Pearson correlation between the target column and each variable.
        """
        non_zero_varlist = [target_col] + variable_col
        correlations = data[non_zero_varlist].corr(method='pearson').unstack().sort_values(ascending=False)
        correlations = pd.DataFrame(correlations).reset_index()
        correlations.columns = ['col1', 'col2', 'correlation']
        _corr = correlations.query(f"col1 == '{target_col}' & col2 != '{target_col}'")
        return _corr

    @staticmethod
    def _get_ppscore(data, variable_col, target_col):
        """
        Calculate the Predictive Power Score (PPS) between the target column and other variables.

        Parameters
        ----------
        data : pandas.DataFrame
            The data frame containing the data.
        variable_col : list of str
            List of column names to be used as independent variables.
        target_col : str
            The name of the dependent variable column.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the PPS between the target column and each variable.
        """
        _pscore = pps.predictors(data[variable_col + [target_col]], y=target_col)
        _pscore = _pscore[['y', 'x', 'ppscore']]
        _pscore.columns = ['col1', 'col2', 'ppscore']
        return _pscore

    @staticmethod
    def _get_xicor_score(data, variable_col, target_col, ties='auto'):
        """
        Calculate the Xi correlation for multiple variable-target pairs and return a sorted DataFrame.

        Parameters
        ----------
        data : pandas.DataFrame
            The data frame containing the data.
        variable_col : list of str
            List of column names to be used as independent variables.
        target_col : str
            The name of the dependent variable column.
        ties : {'auto', bool}, optional
            How to handle ties in Xi correlation calculation:
            - 'auto' (default): Decide based on the uniqueness of y values.
            - True: Assume ties are present.
            - False: Assume no ties are present.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing Xi correlations and p-values, sorted by the Xi correlation.
        """
        scores = [CorrelationAnalyzer._xicordf(data, x, target_col, ties) for x in variable_col]
        scores.sort(key=lambda item: item['xicor'], reverse=True)

        df_columns = ["x", "y", "xicor", "p-value"]
        data_dict = {column: [score[column] for score in scores] for column in df_columns}
        scores_df = pd.DataFrame.from_dict(data_dict)
        scores_df =scores_df[['y', 'x', 'xicor']]
        scores_df.columns=['col1', 'col2', 'xicor']
        return scores_df

    @staticmethod
    def _xicordf(data, x_col, y_col, ties='auto'):
        """
        Calculate the Xi correlation for specified columns in a DataFrame.

        Parameters
        ----------
        data : pandas.DataFrame
            The data frame containing the data.
        x_col : str
            The name of the independent variable column.
        y_col : str
            The name of the dependent variable column.
        ties : {'auto', bool}, optional
            How to handle ties in the data:
            - 'auto' (default): Decide based on the uniqueness of y values.
            - True: Assume ties are present.
            - False: Assume no ties are present.

        Returns
        -------
        dict
            A dictionary containing the Xi correlation statistic and p-value.
        """
        x = data[x_col].values
        y = data[y_col].values
        xicor, p_value = CorrelationAnalyzer._get_xicor(x, y, ties=ties)
        return {'x': x_col, 'y': y_col, 'xicor': xicor, 'p-value': p_value}

    @staticmethod
    def _get_xicor(x, y, ties="auto"):
        """
        Calculate the Xi correlation coefficient and p-value between two arrays.

        Parameters
        ----------
        x : array_like
            The independent variable array.
        y : array_like
            The dependent variable array.
        ties : {'auto', bool}, optional
            How to handle ties in the data:
            - 'auto' (default): Decide based on the uniqueness of y values.
            - True: Assume ties are present.
            - False: Assume no ties are present.

        Returns
        -------
        statistic : float
            The Xi correlation coefficient.
        p_value : float
            The p-value for the Xi correlation.

        Raises
        ------
        IndexError
            If the lengths of x and y do not match.
        ValueError
            If the ties parameter is not 'auto' or a boolean.
        """
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        n = len(y)

        if len(x) != n:
            raise IndexError(f"x, y length mismatch: {len(x)}, {len(y)}")

        if ties == "auto":
            ties = len(np.unique(y)) < n
        elif not isinstance(ties, bool):
            raise ValueError(f"Expected ties to be either 'auto' or boolean, got {ties} ({type(ties)}) instead")

        y = y[np.argsort(x)]
        r = rankdata(y, method="ordinal")
        nominator = np.sum(np.abs(np.diff(r)))

        if ties:
            l = rankdata(y, method="max")
            denominator = 2 * np.sum(l * (n - l))
            nominator *= n
        else:
            denominator = np.power(n, 2) - 1
            nominator *= 3

        statistic = 1 - nominator / denominator
        p_value = norm.sf(statistic, scale=2 / 5 / np.sqrt(n))

        return statistic, p_value
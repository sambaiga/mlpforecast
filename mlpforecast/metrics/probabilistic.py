import numpy as np
import scipy

def laplace_nll(true, loc, scale):
    """
    Calculate the Negative Log-Likelihood (NLL) for a Laplace distribution.

    The NLL is a measure of the goodness of fit of a statistical model. For the Laplace distribution,
    it is based on the log-probability density function.

    Parameters
    ----------
    true : np.ndarray
        Array of true values.
    loc : float
        The location parameter (mean) of the Laplace distribution.
    scale : float
        The scale parameter (diversity) of the Laplace distribution.

    Returns
    -------
    np.ndarray
        The negative log-likelihood of the Laplace distribution for each true value.

    Notes
    -----
    The Laplace distribution log-probability density function is defined as:
    
    \[
    \log f(x \mid \mu, b) = -\log(2b) - \frac{|x - \mu|}{b}
    \]

    where \( \mu \) is the location parameter and \( b \) is the scale parameter.

    Example
    -------
    >>> true = np.array([1.0, 2.0, 3.0])
    >>> loc = 2.0
    >>> scale = 1.0
    >>> calculate_laplace_nll(true, loc, scale)
    array([-1.69314718, -1.69314718, -2.69314718])
    """
    return scipy.stats.laplace.logpdf(true, loc=loc, scale=scale)

def pinball_loss(y, q, tau):
    """
    Calculate the pinball loss for a given set of true values, predicted quantiles, and quantile levels.

    The pinball loss, also known as quantile loss, is used to evaluate the accuracy of quantile predictions.
    It is asymmetric and penalizes overestimation and underestimation differently based on the quantile level.

    Parameters
    ----------
    y : np.ndarray
        Array of true values.
    q : np.ndarray
        Array of predicted quantiles.
    tau : float
        The quantile level (0 < tau < 1).

    Returns
    -------
    np.ndarray
        The pinball loss for each prediction.

    Notes
    -----
    The pinball loss is defined as:
    
    \[
    \text{Pinball Loss} = 
    \begin{cases} 
    \tau (y - q) & \text{if } y \geq q \\
    (1 - \tau) (q - y) & \text{if } y < q 
    \end{cases}
    \]

    Example
    -------
    >>> y = np.array([3.0, 5.0, 7.0])
    >>> q = np.array([2.5, 5.5, 6.5])
    >>> tau = 0.5
    >>> pinball_loss(y, q, tau)
    array([ 0.25,  0.25,  0.25])
    """
    return (y - q) * tau * (y >= q) + (q - y) * (1 - tau) * (y < q)

def pinball_score(true, q_values, taus):
    """
    Calculate the aggregated pinball score for a set of true values, predicted quantiles, and quantile levels.

    The pinball score is the mean of the pinball losses for multiple quantile predictions, providing an overall measure
    of the accuracy of the quantile forecasts.

    Parameters
    ----------
    true : np.ndarray
        Array of true values.
    q_values : list of np.ndarray
        List of arrays containing predicted quantiles.
    taus : list of float
        List of quantile levels corresponding to each array in `q_values`.

    Returns
    -------
    float
        The aggregated pinball score.

    Example
    -------
    >>> true = np.array([3.0, 5.0, 7.0])
    >>> q_values = [np.array([2.5, 5.5, 6.5]), np.array([2.0, 5.0, 7.5])]
    >>> taus = [0.5, 0.9]
    >>> get_pinball_score(true, q_values, taus)
    0.20833333333333334
    """
    scores = np.array([pinball_loss(true, q, tau).mean() for q, tau in zip(q_values, taus)])
    return scores.sum() / len(scores)



def continuous_ranked_probability_score(true, samples):
    """
    Calculate the Continuous Ranked Probability Score (CRPS).

    The CRPS is a proper scoring rule that is used to assess the quality of probabilistic predictions.
    It generalizes the Mean Absolute Error (MAE) to probabilistic forecasts.

    Parameters
    ----------
    true : np.ndarray
        Array of true values. Shape (n_samples,).
    samples : np.ndarray
        Array of predicted samples. Shape (n_samples,).

    Returns
    -------
    float
        The CRPS value.

    Notes
    -----
    The CRPS is defined as:
    
    \[
    \text{CRPS}(F, y) = \int_{-\infty}^{\infty} \left( F(z) - \mathbf{1}\{z \geq y\} \right)^2 dz
    \]

    where \( F \) is the cumulative distribution function of the forecast and \( y \) is the observed value.

    Example
    -------
    >>> true = np.array([1.0, 2.0, 3.0])
    >>> samples = np.array([[1.2, 2.1, 2.9], [1.0, 2.0, 3.0], [0.8, 1.9, 3.1]])
    >>> continuous_ranked_probability_score(true, samples)
    0.05555555555555555

    References
    ----------
    """
    num_samples = samples.shape[0]
    absolute_error = np.mean(np.abs(samples - true), axis=0)

    samples_sorted = np.sort(samples, axis=0)
    differences = samples_sorted[1:] - samples_sorted[:-1]
    weights = np.arange(1, num_samples) * np.arange(num_samples - 1, 0, -1)
    weights = np.expand_dims(weights, -1)

    per_observation_crps = absolute_error - np.sum(differences * weights, axis=0) / num_samples**2
    return np.mean(per_observation_crps)



def continuous_ranked_probability_score_laplace(true, loc, scale):
    """
    Calculate the Continuous Ranked Probability Score (CRPS) for a Laplace distribution.

    The CRPS is a proper scoring rule that is used to assess the quality of probabilistic predictions.
    This function specifically calculates the CRPS for a Laplace distribution.

    Parameters
    ----------
    true : np.ndarray
        Array of true values.
    loc : float
        The location parameter (mean) of the Laplace distribution.
    scale : float
        The scale parameter (diversity) of the Laplace distribution.

    Returns
    -------
    np.ndarray
        The CRPS value for each observation.

    Notes
    -----
    The CRPS for a Laplace distribution is defined as:
    
    \[
    \text{CRPS}(F, y) = \sigma \left[ \frac{s}{2} \left(2\Phi(s) - 1\right) + \frac{2\phi(s) - 1}{\sqrt{\pi}} \right]
    \]

    where \( s = \frac{y - \mu}{\sigma} \), \(\Phi\) is the CDF, and \(\phi\) is the PDF of the Laplace distribution.

    Example
    -------
    >>> true = np.array([1.0, 2.0, 3.0])
    >>> loc = 2.0
    >>> scale = 1.0
    >>> continuous_ranked_probability_score_laplace(true, loc, scale)
    array([0.53260109, 0.34134475, 0.53260109])

    References
    ----------
    Zamo, M., & Naveau, P. (2017). Estimation of the Continuous Ranked Probability Score with Limited Information 
    and Applications to Ensemble Weather Forecasts. Monthly Weather Review, 145(3), 1023-1037.
    """
    s = (true - loc) / scale
    pdf = scipy.stats.laplace.pdf(s)
    cdf = scipy.stats.laplace.cdf(s)
    crps = scale * (s * (2 * cdf - 1) + 2 * pdf - 1. / np.sqrt(np.pi))
    return crps

def continous_ranked_pscore_pwm(true, samples):
    """
    Calculate the Continuous Ranked Probability Score (CRPS).

    The CRPS is a proper scoring rule that is used to assess the quality of probabilistic predictions.
    It generalizes the Mean Absolute Error (MAE) to probabilistic forecasts.

    Parameters
    ----------
    true : np.ndarray
        Array of true values. Shape (n_samples,).
    samples : np.ndarray
        Array of predicted values. Shape (n_samples,).

    Returns
    -------
    float
        The CRPS value.

    Notes
    -----
    The CRPS is defined as:
    
    \[
    \text{CRPS}(F, y) = \int_{-\infty}^{\infty} \left( F(z) - \mathbf{1}\{z \geq y\} \right)^2 dz
    \]

    where \( F \) is the cumulative distribution function of the forecast and \( y \) is the observed value.

    Example
    -------
    >>> true = np.array([1.0, 2.0, 3.0])
    >>> samples = np.array([[1.2, 2.1, 2.9], [1.0, 2.0, 3.0], [0.8, 1.9, 3.1]])
    >>> calculate_crps(true, samples)
    0.05555555555555555

    References
    ----------
    Zamo, M., & Naveau, P. (2017). Estimation of the Continuous Ranked Probability Score with Limited Information and Applications to Ensemble Weather Forecasts.
    Monthly Weather Review, 145(3), 1023-1037.
    """
    num_samples = samples.shape[0]
    absolute_error = np.mean(np.abs(samples - true), axis=0)

    if num_samples == 1:
        return np.mean(absolute_error)

    samples_sorted = np.sort(samples, axis=0)
    differences = samples_sorted[1:] - samples_sorted[:-1]
    weights = np.arange(1, num_samples) * np.arange(num_samples - 1, 0, -1)
    weights = np.expand_dims(weights, -1)

    per_observation_crps = absolute_error - np.sum(differences * weights, axis=0) / num_samples**2
    return np.mean(per_observation_crps)



def winkler_score(pred: np.array, 
                            true: np.array,  
                            lower: np.array,
                            upper: np.array,
                            alpha: float = None) -> np.array:
    """
    Calculate the Winkler score used to evaluate both the coverage and interval width.
    
    The Winkler score is calculated as follows:
    $$
    WinklerScore = Upper - Lower + \frac{2}{\alpha} \times (Lower - True) \times (True < Lower) + \frac{2}{\alpha} \times (True - Upper) \times (True > Upper)
    $$

    This formula accounts for the coverage and interval width of the prediction intervals, as proposed by Winkler in his 1972 paper:
    R. L. Winkler, “A Decision-Theoretic Approach to Interval Estimation,”
    Journal of the American Statistical Association, vol. 67, no. 337, pp.187–191, 1972
    
    Args:
        pred (np.array): Predicted values.
        true (np.array): True values.
        lower (np.array): Lower bounds of the prediction intervals.
        upper (np.array): Upper bounds of the prediction intervals.
        alpha (float, optional): Significance level for the prediction intervals. Defaults to None.
    
    Returns:
        np.array: Winkler score for each data point.

    Example usage:
        # Assuming alpha is not None and is provided
        predicted_values = np.array([10, 15, 20, 25, 30])
        true_values = np.array([11, 14, 19, 26, 29])
        lower_bounds = np.array([9, 13, 18, 24, 28])
        upper_bounds = np.array([12, 17, 22, 27, 31])
        alpha_value = 0.05

        winkler_scores = calculate_winkler_score(predicted_values, true_values, lower_bounds, upper_bounds, alpha_value)
        print(f"Winkler Scores: {winkler_scores}")
    """
    if alpha is None:
        raise ValueError("Alpha cannot be None for the Winkler score calculation.")
    
    winkler_score = (
        upper
        - lower
        + 2.0 / alpha * (lower - true) * (true < lower)
        + 2.0 / alpha * (true - upper) * (true > upper)
    )
    
    return np.median(winkler_score)

def prediction_intervals_coverage(true: np.array, lower: np.array, upper: np.array) -> float:
    """
    Calculate the coverage of prediction intervals.

    The coverage is defined as:
    $$
    Coverage = \frac{1}{n} \sum_{i=1}^{n} \mathbb{1}(lower_i \leq true_i \leq upper_i)
    $$

    where \( \mathbb{1} \) is the indicator function that is 1 if the condition is true and 0 otherwise,
    \( n \) is the number of observations,
    \( lower \) and \( upper \) are the lower and upper bounds of the prediction intervals, respectively,
    and \( true \) is the true value.

    Args:
        true (np.array): True values.
        lower (np.array): Lower bounds of the prediction intervals.
        upper (np.array): Upper bounds of the prediction intervals.

    Returns:
        float: Coverage proportion (percentage).

    Example usage:
        true_values = np.array([10, 15, 20, 25, 30])
        lower_bounds = np.array([8, 14, 18, 22, 28])
        upper_bounds = np.array([12, 16, 22, 28, 32])

        coverage_percentage = calculate_coverage(true_values, lower_bounds, upper_bounds)
        print(f"Coverage: {coverage_percentage:.2f}%")
    """
    coverage = ((true >= lower) & (true <= upper)).mean()
    return coverage 

def average_coverage_error(coverage: float, alpha: float) -> float:
    """
    Calculate the Average Coverage Error (ACE).

    The ACE is defined as:
    $$
    ACE = Coverage - (1 - \alpha)
    $$

    where \( Coverage \) is the coverage proportion (percentage) of true values within the prediction intervals,
    and \( \alpha \) is the significance level for the prediction intervals.

    Args:
        coverage (float): Coverage proportion (percentage).
        alpha (float): Significance level for the prediction intervals.

    Returns:
        float: Average Coverage Error.

    Example:
        coverage_percentage = 95.0  # Example coverage (e.g., 95%)
        alpha_value = 0.05  # Example alpha (e.g., 0.05)

        ace = calculate_average_coverage_error(coverage_percentage, alpha_value)
    """
    ace = coverage - (1 - alpha)
    return ace

def median_prediction_width(lower: np.array, upper: np.array) -> float:
    """
    Calculate the Normalized Median Prediction Interval (NMPI).

    The NMPI is defined as:
    $$
    NMPI = \text{median}(\left| Upper - Lower \right|)
    $$

    where \( Upper \) and \( Lower \) are the upper and lower bounds of the prediction intervals, respectively.

    Args:
        lower (np.array): Lower bounds of the prediction intervals.
        upper (np.array): Upper bounds of the prediction intervals.

    Returns:
        float: NMPI value.

    Example usage:
        lower_bounds = np.array([8, 14, 18, 22, 28])
        upper_bounds = np.array([12, 16, 22, 28, 32])

        nmpi_value = calculate_nmpi(lower_bounds, upper_bounds)
    """
    nmpi = np.median(np.abs(upper - lower))
    return nmpi

def calculate_gamma_nmpi_score(nmpi: float, true_nmpic: float,k: float=1, smoothing: float=1e-6) -> float:
    """
    Calculate the Gamma NMPI Score.

    The Gamma NMPI Score is defined as:
    $$
    \gamma_{NMPI} = 1 - \text{expit}\left( k \times (\text{NMPI}_{diff} - \text{smoothing})^2 \right)
    $$

    where \( \text{expit}(x) \) is the sigmoid function \( \frac{1}{1 + e^{-x}} \),
    \( \text{NMPI}_{diff} \) is the absolute difference between the calculated NMPI and the true NMPI,
    \( k \) is a scaling factor, and
    \( \text{smoothing} \) is a smoothing parameter.

    Args:
        nmpi (float): Calculated NMPI value.
        true_nmpic (float): True NMPI value.
        k (float): Scaling factor.
        smoothing (float): Smoothing parameter.

    Returns:
        float: Gamma NMPI Score.

    Example usage:
        nmpi_value = 4.00  # Example NMPI value
        true_nmpic_value = 3.50  # Example true NMPI value
        k_value = 1.0  # Example scaling factor
        smoothing_value = 0.5  # Example smoothing parameter

        gamma_nmpi_score_value = calculate_gamma_nmpi_score(nmpi_value, true_nmpic_value, k_value, smoothing_value)
        print(f"Gamma NMPI Score: {gamma_nmpi_score_value:.2f}")
    """
    nmpic_diff = np.abs(nmpi - true_nmpic)
    gamma_nmpi_score = 1 - scipy.special.expit(k * (nmpic_diff - smoothing)**2)
    return gamma_nmpi_score


def get_standardized_gamma_nmpi(nmpi, true_nmpic, k:float=1, smoothing=1e-10):
    """
    Calculates the standardized gamma NMPI score.

    The standardized gamma NMPI score quantifies the relative position of the NMPI score between the minimum and maximum NMPI scores
    obtained from the true NMPI score. It represents how well the NMPI score performs compared to the range of possible scores.

    Args:
        nmpi (float, optional): The NMPI score to be standardized. Default is 0.1.
        true_nmpic (float, optional): The true NMPI score (ground truth). Default is 1.

    Returns:
        float: The standardized gamma NMPI score.

    """
    # Calculate the NMPI score for the given nmpi and true_nmpic
    score = calculate_gamma_nmpi_score(nmpi=nmpi, true_nmpic=true_nmpic, k=k, smoothing=smoothing)

    # Calculate the maximum NMPI score for true_nmpic
    max_score = calculate_gamma_nmpi_score(true_nmpic, true_nmpic,k=k, smoothing=smoothing)

    # Calculate the minimum NMPI scores for nmpi=0.0 and nmpi=1.0 (based on true_nmpic)
    min_score_left = calculate_gamma_nmpi_score(nmpi=0.0, true_nmpic=true_nmpic, k=k, smoothing=smoothing)
    min_score_right = calculate_gamma_nmpi_score(nmpi=1.0, true_nmpic=true_nmpic,k=k, smoothing=smoothing)

    # Calculate the standardized gamma NMPI score based on the relative position of true_nmpic between min and max scores
    gamma_nmpi_score = np.where(
        true_nmpic <= nmpi,
        (score - min_score_right) / (max_score - min_score_right),
        (score - min_score_left) / (max_score - min_score_left),
    )
    return gamma_nmpi_score

def calculate_gamma_coverage_score(picp: float, alpha: float) -> float:
    """
    Calculate the Gamma Coverage Score.

    The Gamma Coverage Score is defined as:
    \[
    \gamma_{\text{coverage}} = \frac{\exp(-\text{diff}) - \exp(-(1 - \alpha))}{1 - \exp(-(1 - \alpha))}
    \]

    where \(\text{diff}\) is calculated as:
    \[
    \text{diff} = 
    \begin{cases} 
    1 - \alpha - \text{PICP} & \text{if } \text{PICP} \leq 1 - \alpha \\
    0 & \text{otherwise}
    \end{cases}
    \]

    Args:
        picp (float): Proportion of intervals containing the true value (PICP).
        alpha (float): Significance level for the prediction intervals.

    Returns:
        float: Gamma Coverage Score.

    Raises:
        ValueError: If PICP is not between 0 and 1.

    Example usage:
        picp_value = 0.95  # Example PICP (e.g., 95%)
        alpha_value = 0.05  # Example alpha (e.g., 0.05)

        gamma_coverage_score_value = gamma_coverage_score(picp_value, alpha_value)
        print(f"Gamma Coverage Score: {gamma_coverage_score_value:.2f}")
    """
    if not 0 <= picp <= 1:
        raise ValueError("PICP must be between 0 and 1.")
    
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")

    diff = np.where(picp <= 1 - alpha, 1 - alpha - picp, 0.0)
    score = np.exp(-diff)
    min_score = np.exp(-(1 - alpha))
    normalized_score = (score - min_score) / (1 - min_score)
    return normalized_score


def cwe_score(nmpi: float, picp: float, 
                         error: float, 
                         true_nmpic: float, 
                         alpha: float = 0.90,
                           beta: int = 1, 
                           k:float=1.0, 
                           smoothing:float=1e-6) -> dict:
    """
    Calculate the Combined Weighting Index (CWI) score for probabilistic forecasts.

    The CWI score is calculated using the formula:
    $$ CWI = \frac{(1 + \beta^2) \cdot \gamma_{pic} \cdot \gamma_{nmpic}}{\beta^2 \cdot \gamma_{nmpic} + \gamma_{pic}} $$

    If an error is provided, the CWI score is adjusted as follows:
    $$ CWI_{adjusted} = CWI \cdot (1 - \text{error}) $$

    Additionally, if the PIC score is less than or equal to 0.5, the CWI score is further adjusted:
    $$ CWI_{final} = CWI_{adjusted} \cdot \text{picp} $$

    Parameters:
        nmpi (float): The NMPI metric value of the forecast.
        picp (float): The PIC score of the forecast.
        error (float): The forecast error, which is the difference between the forecast value and the true value.
        true_nmpic (float): The true NMPI score (ground truth).
        alpha (float, optional): The significance level for PIC. Default is 0.90.
        beta (int, optional): A parameter to adjust the emphasis on calibration. Default is 1.
        k (float, optional): A scaling factor for NMPI standardization. Default is 1.0.
        smoothing (float, optional): A small value added to avoid division by zero. Default is 1e-6.

    Returns:
        dict: A dictionary containing the CWI score and other relevant metrics.
    """
     # Calculate gamma values
    gamma_pic = calculate_gamma_coverage_score(picp=picp, alpha=alpha)
    gamma_nmpic = get_standardized_gamma_nmpi(nmpi=nmpi, 
                                              true_nmpic=true_nmpic, 
                                              k=k, smoothing=smoothing)
    condition1 = nmpi <= true_nmpic
    condition2 = picp >= alpha
    gamma_nmpic = np.where(condition1 & condition2, gamma_pic, gamma_nmpic)

    # Calculate CWI score
    num = (1 + beta ** 2) * (gamma_pic) * (gamma_nmpic)
    denom = (beta ** 2 * gamma_nmpic) + gamma_pic
    cwe = np.true_divide(num, denom)

    # Apply error correction if error is provided
    if error is not None:
        cwe *= (1 - error)

    # Apply additional condition based on picp value
    if picp <= 0.5:
        cwe *= picp
    return cwe



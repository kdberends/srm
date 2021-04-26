""" Additional statistical functions not provided by scipy or numpy """


# =============================================================================
# Imports
# =============================================================================


import numpy as np
from sklearn.neighbors import KernelDensity
from scipy import stats

# not using this anymore. Doesn't play well with some dependencies 
#from statsmodels.distributions.empirical_distribution import ECDF

# =============================================================================
# Functions
# =============================================================================

# BELOW FROM STATMODELS
# --------------------------
class StepFunction(object):
    """
    A basic step function.

    Values at the ends are handled in the simplest way possible:
    everything to the left of x[0] is set to ival; everything
    to the right of x[-1] is set to y[-1].

    Parameters
    ----------
    x : array-like
    y : array-like
    ival : float
        ival is the value given to the values to the left of x[0]. Default
        is 0.
    sorted : bool
        Default is False.
    side : {'left', 'right'}, optional
        Default is 'left'. Defines the shape of the intervals constituting the
        steps. 'right' correspond to [a, b) intervals and 'left' to (a, b].

    Examples
    --------
    >>> import numpy as np
    >>> from statsmodels.distributions.empirical_distribution import StepFunction
    >>>
    >>> x = np.arange(20)
    >>> y = np.arange(20)
    >>> f = StepFunction(x, y)
    >>>
    >>> print(f(3.2))
    3.0
    >>> print(f([[3.2,4.5],[24,-3.1]]))
    [[  3.   4.]
     [ 19.   0.]]
    >>> f2 = StepFunction(x, y, side='right')
    >>>
    >>> print(f(3.0))
    2.0
    >>> print(f2(3.0))
    3.0
    """

    def __init__(self, x, y, ival=0., sorted=False, side='left'):

        if side.lower() not in ['right', 'left']:
            msg = "side can take the values 'right' or 'left'"
            raise ValueError(msg)
        self.side = side

        _x = np.asarray(x)
        _y = np.asarray(y)

        if _x.shape != _y.shape:
            msg = "x and y do not have the same shape"
            raise ValueError(msg)
        if len(_x.shape) != 1:
            msg = 'x and y must be 1-dimensional'
            raise ValueError(msg)

        self.x = np.r_[-np.inf, _x]
        self.y = np.r_[ival, _y]

        if not sorted:
            asort = np.argsort(self.x)
            self.x = np.take(self.x, asort, 0)
            self.y = np.take(self.y, asort, 0)
        self.n = self.x.shape[0]

    def __call__(self, time):

        tind = np.searchsorted(self.x, time, self.side) - 1
        return self.y[tind]


class ECDF(StepFunction):
    """
    Return the Empirical CDF of an array as a step function.

    Parameters
    ----------
    x : array-like
        Observations
    side : {'left', 'right'}, optional
        Default is 'right'. Defines the shape of the intervals constituting the
        steps. 'right' correspond to [a, b) intervals and 'left' to (a, b].

    Returns
    -------
    Empirical CDF as a step function.

    Examples
    --------
    >>> import numpy as np
    >>> from statsmodels.distributions.empirical_distribution import ECDF
    >>>
    >>> ecdf = ECDF([3, 3, 1, 4])
    >>>
    >>> ecdf([3, 55, 0.5, 1.5])
    array([ 0.75,  1.  ,  0.  ,  0.25])
    """
    def __init__(self, x, side='right'):
        step = True
        if step: #TODO: make this an arg and have a linear interpolation option?
            x = np.array(x, copy=True)
            x.sort()
            nobs = len(x)
            y = np.linspace(1./nobs,1,nobs)
            super(ECDF, self).__init__(x, y, side=side, sorted=True)
        else:
            return interp1d(x,y,drop_errors=False,fill_values=ival)

# --------------------------

def invboxcox(y, ld=1):
    """
    Back-transform from box-cox transformed variable
    """
    y = np.array(y)
    if ld == 0:
        return(np.exp(y))
    else:
        return(np.exp(np.log(ld * y + 1) / ld))

def invgamma(x, a, b):
    """
    Returns probability of value x given inverse gamma distribution with 
    parameters a, b

    Arguments

        q:  float, quantile
        a:  float, shape parameter (gamma dist. parameter) 
        b:  float, scale parameter (gamma dist. parameter)

    Returns

        probability of x given a, b
    """
    return stats.gamma.pdf(1 / x, a, scale=(1 / b)) / x ** 2

def design_matrix(x, number_of_parameters=4):
    """
    Returns design matrix for x. The first column are ones, consistent with 
    the intercept term. 

    Arguments

        x                   : list, observations/predictor
        number_of_parameters: int, polynomial class. 2 = linear, 3 = cubic

    Returns

        Design matrix (nd array)
    """
    DM = np.ones([len(x), number_of_parameters])
    for i in range(number_of_parameters - 1):
        DM[:, i + 1] = x ** (i + 1)
    return DM

def get_dummy_distributions(mu, sigma, n):
    p = np.random.normal(mu, sigma, n)
    x, y = get_empirical_cdf(p)
    return p, x, y

# Empirical/experimental distributions
# --------------------------------------
def get_empirical_pdf(p, method=1):
    if method == 1: 
        bins = max([int(len(p) / 50), 10])
        counts, bins = np.histogram(p, bins=bins, density=True)
        return counts, bins[1:]
    elif method == 2:
        x = np.linspace(np.min(p), np.max(p), 100)[:, np.newaxis]
        p = p[:, np.newaxis]
        log_dens = KernelDensity(kernel='gaussian', bandwidth=0.001).fit(p).score_samples(x)

        return np.exp(log_dens), x

def empirical_cdf_to_pdf(p, v):
    """
    p = probabilities
    v = values
    """
    pdf_p = np.diff(p) / np.diff(v)
    pdf_v = v[1:]

    return pdf_p, pdf_v

def get_empirical_cdf(sample, n=100, method=1, ignore_nan=True):
    """
    Returns an experimental/empirical cdf from data. 

    Arguments:

        p : list

    Returns:

        (x, y) : lists of values (x) and cumulative probability (y)

    """

    sample = np.array(sample)
    if ignore_nan:
        sample = sample[~np.isnan(sample)]

    if method == 0:
        n = len(sample)
        val = np.sort(sample)
        p = np.array(range(n)) / float(n)
    else:
        ecdf = ECDF(sample)

        val = np.linspace(min(sample), max(sample), n + 1)
        p = ecdf(val)
    return p, val

def empirical_ppf(qs, p, val=None, single_value=False):
    """
    Constructs empirical cdf, then draws quantile by linear interpolation
    qs : array of quantiles (e.g. [2.5, 50, 97.5])
    p : array of random inputs

    return
    """
    if val is None: 
        p, val = get_empirical_cdf(p)

    if not single_value:
        output = list()
        for q in qs:
            output.append(np.interp(q / 100., p, val))
    else:
        output = np.interp(qs / 100., p, val)
    return output

def inverse_empirical_cdf(p, val, q):
    """
    Arguments:

        p  : cumulative probability vector
        val: values belonging to p
        q  : quantile for which to retrieve the probability
    """
    return np.interp(q, p, val)  

def ishigami(x1, x2, x3, param=[1, 7, 0.1]):
    """
    Ishigami function (often used, ref. saltelli?)
    """
    return param[0]*np.sin(x1) + param[1] * np.sin(x2) ** 2 + param[2]*x3**4*np.sin(x1)

def get_predictive_uncertainty(prior, n=1500):
    sigx = np.linspace(0, 6 * np.sqrt(prior.beta / (prior.alpha + 1)), n)
    sigy = 2 * sigx * invgamma(sigx ** 2, prior.alpha, prior.beta)
    sigy[np.isnan(sigy)] = 0
    sigy = sigy / np.sum(sigy)
    return sigx, sigy

def get_posterior_interval(p, X, V, alp, bet):
    """
    Credible interval
    """
    # return np.sqrt(bet/ alp * np.sum(X.T * V.dot(X.T), axis=0).T ) * stats.t.ppf(p, 2 * alp)
    Int = np.sqrt(bet / alp * (np.sum(X.T * V.dot(X.T), axis=0).T)) * stats.t.ppf(p, 2 * alp)
    
    return Int

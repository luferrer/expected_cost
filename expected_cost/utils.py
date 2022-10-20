import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.special import expit, logit, logsumexp 
import os

def make_hist(targets, scores, classi=0, nbins=100):
    """ Plot the histogram for the scores output by the system for
    one of the classes.
    
    Parameters 
    ----------

    targets : 1d array-like of size N
        Ground truth (correct) target values for a set of N samples. 
        Should take values between 0 and C-1, where C is the number
        of possible class targets.

    scores : array-like of size NxC
        Scores can be posteriors, log-posteriors, log-likelihoods
        or log-likelihood ratios (see method bayes_decisions for 
        a detailed explanation)

    classi : index to select from the scores array. Default = 0


    Returns
    -------
    centers :  the bin centers
    hists :  a lot of bin height for each class

    """

    if classi!=0 or scores.ndim!=1:
        scores = scores[:,classi]

    # Create the bins for the histogram using all the data
    h, e = np.histogram(scores, nbins)

    # Get the bin centers for plotting
    centers = (e[1:]+e[:-1])/2
    
    # Now get the (normalized heights) for the samples from each class
    hists = []
    for c in np.unique(targets):
        idx = targets == c
        hc, _ = np.histogram(scores[idx], e, density=True)
        hists.append(hc)

    return centers, hists
    


def compute_R_matrix_from_counts_for_binary_classif(K01, K10, N0, N1):
    """ Compute the error rates given the number of missclassifications, K01 and K10, 
    and the total number of samples for each class, N0, N1.
    K01: number of samples of class 0 labelled as class 1
    K10: number of samples of class 1 labelled as class 0
    N0: number of samples of class 0
    N1: number of samples of class 1
    """

    cm = np.array([[N0-K01, K01],[K10, N1-K10]])
    R = cm/cm.sum(axis=1, keepdims=True)
    return R
    

def bayes_thr_for_llrs(priors, costs):
    """ This method computes the bayes threshold on the LLRs when the cost matrix has 
    the following form:
                         0  c01
                        c10  0
    """

    cmatrix = costs.get_matrix()
    if np.any(np.array(cmatrix.shape) != 2) or cmatrix[0,0] != 0 or cmatrix[1,1] != 0:
        raise ValueError("This method is only valid for cost matrices of the form: [[0, c01], [c10, 0]]")

    return np.log(priors[0]/priors[1]*cmatrix[0,1]/cmatrix[1,0])


def llrs_to_logpost(llrs, priors):
    """ Compute the log posterior from the log-likelihood-ratios (llrs):

    logpost_class1 = log 1/(1+ e^(-logodds))
                   = - log(1 + e^(-logodds))

    where [P0, P1] = priors are the class priors and

    logodds = -llr-log(P1/P0)

    """

    logodds = llrs + np.log(priors[1]/priors[0])

    logpost_class1 = - logsumexp(np.c_[np.zeros_like(logodds), -logodds], axis=1)
    logpost_class0 = - logsumexp(np.c_[np.zeros_like(logodds), +logodds], axis=1)

    return np.c_[logpost_class0, logpost_class1]


def llks_to_logpost(llks, priors):
    """ Compute the log posterior from the log potentially-scaled likelihoods.
    The scale (a factor independent of the class, usually p(x)), does not matter 
    in this computation because it dissapears when we normalize the posteriors. 
    """

    log_posteriors_unnormed = llks + np.log(priors)
    return log_posteriors_unnormed - logsumexp(log_posteriors_unnormed, axis=1, keepdims=True)


def logpost_to_log_scaled_lks(logpost, priors):
    """ Compute the log scaled likeliihoods from the log posteriors, 
    given the priors:

    log_scaled_lk = log(p(x|c) / p(x))
                  = log(P(c|x) / P(c))
                  = logpost - log(priors)

    """

    return logpost - np.log(priors)


def logpost_to_llrs(logpost, priors):
    """ Compute the log-likeliihood ratio (for binary classification) from the 
    log posteriors, given the priors:

    llr = log(p(x|c=1) / p(x|c=0))
        = log(P(c=1|x) / P(c=0|x) * P(c=0) / P(c=1))
        = logpost_1 - logpost_0 - log(priors[1]) + log(priors[0])

    """

    return logpost[:,1] - logpost[:,0] - np.log(priors[1]) + np.log(priors[0])


def mkdir_p(dir):

    if not os.path.isdir(dir):
        os.makedirs(dir)


#########################################################################################
# Definition of a few standard metrics computed from the confusion matrix

def Fscore(K10, K01, N0, N1):
    K11 = N1-K10
    Recall    = K11/N1
    Precision = K11/(K11+K01) if K11+K01>0 else 0
    Fscore    = 2 * Precision*Recall/(Recall+Precision) if K11>0 else 0
    return Fscore
    

def MCCoeff(K10, K01, N0, N1):
    K11 = N1-K10
    K00 = N0-K01
    num = K00 * K11 - K01 * K10
    den = np.sqrt(N0 * N1 * (K01+K11) * (K10 + K00))
    return num/den if den>0 else (np.inf if num>0 else -np.inf)


def LRplus(K10, K01, N0, N1):
    R10 = K10 / N1
    R01 = K01 / N0
    return (1-R10)/R01 if R01>0 else np.inf


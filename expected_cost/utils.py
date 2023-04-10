import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit, logit, logsumexp 


def plot_hists(targets, scores, outfile):

    num_targets = scores.shape[1]
    fig, axs = plt.subplots(num_targets, figsize=(4,2*num_targets))
    for classi in np.arange(num_targets):
        ax = axs[classi]
        c, hs = make_hist(targets, scores, classi=classi)
        for j, h in enumerate(hs):
            ax.plot(c, h, label="samples of class %d"%j)
        ax.legend(bbox_to_anchor=(1, 1))
        ax.set_title("Scores for class %d"%classi)

    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


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
    # Return R matrix
    return cm/cm.sum(axis=1, keepdims=True)
    

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


def get_binary_data_priors(targets):
    '''
    Returns P0 the priors for class 0 and P1 the priors for class 1
    '''
    N0 = sum(targets==0)
    N1 = sum(targets==1)
    K = N0 + N1
    P0 = N0/K
    P1 = N1/K
    return P0, P1


def get_counts_from_binary_data(targets, decisions):
    '''
    Requires already computed hard decisions from a threshold NOT raw scores
    '''
    N0 = sum(targets==0)
    N1 = sum(targets==1)
    K00 = sum(np.logical_and(targets==0, decisions==0))
    K01 = sum(np.logical_and(targets==0, decisions==1))
    K11 = sum(np.logical_and(targets==1, decisions==1))
    K10 = sum(np.logical_and(targets==1, decisions==0))
    return N0, N1, K00, K11, K01, K10

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit, logit, logsumexp 
import os
from sklearn.utils import resample

def plot_hists(targets, scores, outfile=None, nbins=100, group_by='score', style='-', label_prefix='', axs=None):

    if scores.ndim == 1:
        scores = scores[:,np.newaxis]

    num_scores = scores.shape[1] 

    if group_by == 'score':
        num_plots = num_scores
    elif group_by == 'target':
        num_plots = len(np.unique(targets))
    elif group_by == 'all':
        num_plots = 1
    else:
        raise Exception("group_by %s not implemented"%group_by)

    if axs is None:
        _, axs = plt.subplots(num_plots, figsize=(7,2.5*num_plots))
        axs = np.atleast_1d(axs)

    for classi in np.arange(num_scores):
        c, hs = make_hist(targets, scores, classi=classi, nbins=nbins)
        for tclassi, h in hs.items():
            if group_by == 'score':
                ax = axs[classi]
                label = "samples of class %d"%tclassi
            elif group_by == 'target':
                ax = axs[tclassi]
                label = "scores from col %d"%classi
            else:
                ax = axs[0]
                label = "samples of class %d, scores from col %d"%(tclassi, classi)
            
            ax.plot(c, h, label=label_prefix+label, linestyle=style)

    for i, ax in enumerate(axs):
        ax.legend(bbox_to_anchor=(1, 1))
        if group_by == 'score':
            ax.set_title("Scores from col %d"%i)
        elif group_by == 'target':
            ax.set_title("Samples of class %d"%i)

    plt.tight_layout()
    if outfile is not None:
        plt.savefig(outfile)
        plt.close()

    return axs


def make_hist(targets, scores, classi=0, nbins=100):
    """ Plot the histogram for the scores output by the system for
    one of the classes.
    
    Parameters 
    ----------

    targets : 1d array-like of size N
        Ground truth (correct) target values for a set of N samples. 
        Should take values between 0 and C-1, where C is the number
        of possible class targets.

    scores : array-like of size NxK
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
    hists = {}
    for c in np.unique(targets):
        idx = targets == c
        hc, _ = np.histogram(scores[idx], e, density=True)
        hists[c]= hc

    return centers, hists
    

def compute_R_matrix_from_counts_for_binary_classif(N01, N10, N0, N1):
    """ Compute the error rates given the number of missclassifications, N01 and N10, 
    and the total number of samples for each class, N0, N1.
    N01: number of samples of class 0 labelled as class 1
    N10: number of samples of class 1 labelled as class 0
    N0: number of samples of class 0
    N1: number of samples of class 1
    """

    cm = np.array([[N0-N01, N01],[N10, N1-N10]])
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
    in this computation because it disappears when we normalize the posteriors. 
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


def create_bootstrap_set(samples, targets, conditions=None, stratify=None):

    assert samples.shape[0] == targets.shape[0]
    indices = np.arange(targets.shape[0])

    if conditions is not None:
        assert len(samples) == len(conditions)
        unique_conditions = np.unique(conditions)
        bt_conditions = resample(unique_conditions, replace=True, n_samples=len(unique_conditions))
        sel_indices = np.concatenate([indices[np.where(conditions == s)[0]] for s in bt_conditions])
    else:
        sel_indices = resample(indices, replace=True, n_samples=len(samples), stratify=stratify)
        conditions = np.arange(len(samples))

    return samples[sel_indices], targets[sel_indices], conditions[sel_indices]



def get_binary_data_priors(targets):
    '''
    Returns P0 the priors for class 0 and P1 the priors for class 1
    '''
    N0 = sum(targets==0)
    N1 = sum(targets==1)
    N = N0 + N1
    P0 = N0/N
    P1 = N1/N
    return P0, P1


#########################################################################################
# Definition of a few standard metrics computed from the confusion matrix

def Fscore(N10, N01, N0, N1, betasq=1):
    N11 = N1-N10
    Recall    = N11/N1
    Precision = N11/(N11+N01) if N11+N01>0 else 0
    Fscore    = (betasq+1) * Precision*Recall/(betasq*Recall+Precision) if N11>0 else 0
    return Fscore
    

def MCCoeff(N10, N01, N0, N1):
    N11 = N1-N10
    N00 = N0-N01
    num = N00 * N11 - N01 * N10
    den = np.sqrt(N0 * N1 * (N01+N11) * (N10 + N00))
    return num/den if den>0 else (np.inf if num>0 else -np.inf)


def LRplus(N10, N01, N0, N1):
    R10 = N10 / N1
    R01 = N01 / N0
    return (1-R10)/R01 if R01>0 else np.inf


def effective_priors(costs, priors):
    """ Compute the effective priors: the equivalent priors that would lead to the same cost function
    if the costs are were all set to 1 
    """

    return costs*priors/np.sum(costs*priors)


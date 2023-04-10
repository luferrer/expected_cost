""" Utilities for loading scores or creating simulated data.
"""

import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.special import logsumexp 
from expected_cost import utils
from IPython import embed

try:
    import torch    
    from expected_cost.calibration import affine_calibration_with_crossval
    has_psr = True
    #print("Found psr calibration library = %s"%has_psr)
except:
    has_psr = False


def get_llks_for_multi_classif_task(dataset, priors=None, K=100000, sim_params=None):
    # sourcery skip: raise-specific-error
    """ Load or create multi-class log (potentially scaled) likelihoods (llks).
    The method outputs both the raw (potentially miscalibrated) scores and 
    calibrated scores.
         
    Two datasets are implemented here:
    * cifar10: pre-generated scores for cifar10 data
    * gaussian_sim: simulated data with Gaussian distributions. In this case,
      the priors and K need to be set to determine the class priors and the 
      total number of samples required.
      
    The cifar10 example can be used as template to add a loader for your own set 
    of scores. If your scores are posteriors instead of llks, you can either: 
    * ignore this fact and have this method output the posteriors (in this case,  
      remember to call the expected cost methods with score_type="log-posteriors")
    * convert them to estimated llks by using logpost_to_log_scaled_lks where the 
      priors can be estimated as those used in training.

    """

    if 'cifar10' in dataset:

        # Load pre-generated scores for cifar10 data
        print("\n**** Loading in CIFAR10 data ****\n")
        preacts = np.load("data/resnet-50_cifar10/predictions.npy")
        targets = np.load("data/resnet-50_cifar10/targets.npy")

        # Compute log-softmax to get log posteriors. 
        raw_logpost = preacts - logsumexp(preacts, axis=-1, keepdims=True)

        if has_psr:
            # Calibrate the scores with cross-validation using an affine transform
            # trained with log-loss (cross-entropy)
            print("Calibrating data")
            cal_logpost = affine_calibration_with_crossval(raw_logpost, targets)
        else:
            # If calibration code is not available, load a pre-generated file
            cal_logpost = np.load("data/resnet-50_cifar10/predictions_cal_10classes.npy")

        # For this dataset, the posteriors coincide with the scaled likelihoods
        # because the priors are uniform. If that was not the case, we would use
        # utils.logpost_to_log_scaled_lks(logpost, priors), where the priors are 
        # those used in training.
        raw_llks = raw_logpost
        cal_llks = cal_logpost

    elif dataset == 'gaussian_sim':

        if sim_params is None:
            sim_params = {}


        feat_std    = sim_params.get('feat_std', 1.0)
        score_shift = sim_params.get('score_shift', 0)
        score_scale = sim_params.get('score_scale', 1.0)
        counts = np.array(np.array(priors)*K, dtype=int)
        C = len(counts)

        # Create data (features) using a Gaussian distribution for each class. 
        # Make the features unidimensional for simplicity, with same std and
        # evenly distributed means.
        print("\n**** Creating simulated data with Gaussian class distributions for %d classes ****\n"%C)
        np.random.seed(0)

        # Put the mean at 0, 1, ..., C-1. 
        means = np.arange(0, C)

        # Use the same diagonal covariance matrix for all classes.
        # The value of std will determine the difficulty of the problem.
        stds   = np.ones(C) * feat_std

        # Draw values from these distributions which we take to be
        # (unidimensional) input features
        feats, targets = draw_data_for_gaussian_model(means, stds, counts)

        # Now get the likelihoods for the features sampled above given the model.
        # These are well-calibrated by definition, since we know the generating
        # distribution.
        cal_llks = get_llks_for_gaussian_model(feats, means, stds)

        # Now generate misscalibrated llks with the provided shift and scale 
        raw_llks = score_scale * cal_llks + score_shift

    else:
        raise Exception(f"Unrecognized dataset name: {dataset}")


    return targets, raw_llks, cal_llks


def print_score_stats(scores, targets):

    for c in np.unique(targets):
        scores_c = scores[targets==c]
        print("Class %d :  mean  %5.2f    std   %5.2f"%(c, np.mean(scores_c, axis=0), np.std(scores_c, axis=0)))


def draw_data_for_gaussian_model(means, stds, counts):
    """ 
    Draw data for C classes each with unidimensional Gaussian distribution.
    The means, stds and counts arguments are lists of size C
    containing the mean, std and number of samples for each class. """

    scores = []
    targets = []
    for i, (mean, std, count) in enumerate(zip(means, stds, counts)):
        scores.append(multivariate_normal.rvs(mean, std, count))
        targets.append(np.ones(count)*i)

    scores = np.concatenate(scores)
    return scores, np.array(np.concatenate(targets), dtype=int)


def get_llks_for_gaussian_model(data, means, stds):
    """ Assuming that the input data were created with 
    draw_data_for_gaussian_model, get the log likelihoods
    for each class for the input scores.
    """

    llks = []
    for mean, std in zip(means, stds):
        llks.append(np.atleast_2d(np.log(multivariate_normal(mean, std).pdf(data))))

    return np.concatenate(llks).T


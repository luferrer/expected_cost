""" Utilities for loading scores or creating simulated data.
"""

import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.special import logsumexp 
from expected_cost import utils
import os

try:
    import torch
    from expected_cost.calibration import calibration_with_crossval, calibration_train_on_test
    from psrcal.calibration import HistogramBinningCal
    has_psr = True
#    print("Found psr calibration library = %s"%has_psr)
except:
    has_psr = False


def get_llks_for_multi_classif_task(dataset, priors=None, N=100000, sim_params=None):
    """ Load or create multi-class log (potentially scaled) likelihoods (llks).
    The method outputs both the raw (potentially miscalibrated) scores and 
    calibrated scores.
         
    Two datasets are implemented here:
    * cifar10: pre-generated scores for cifar10 data
    * gaussian_sim: simulated data with Gaussian distributions. In this case,
      the priors and N need to be set to determine the class priors and the 
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
        dir = os.path.dirname(__file__)+"/data/resnet-50_cifar10/"
        print(dir)
        preacts = np.load(dir+"predictions.npy")
        targets = np.load(dir+"targets.npy")

        # Compute log-softmax to get log posteriors. 
        raw_logpost = preacts - logsumexp(preacts, axis=-1, keepdims=True)

        if has_psr:
            # Calibrate the scores with cross-validation using an affine transform
            # trained with log-loss (cross-entropy)
            print("Calibrating data")
            cal_logpost = calibration_with_crossval(raw_logpost, targets)
        else:
            # If calibration code is not available, load a pre-generated file
            cal_logpost = np.load(dir+"predictions_cal_10classes.npy")

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
        counts = np.array(np.array(priors)*N, dtype=int)
        K = len(counts)

        # Create data (features) using a Gaussian distribution for each class. 
        # Make the features unidimensional for simplicity, with same std and
        # evenly distributed means.
        #print("\n**** Creating simulated data with Gaussian class distributions for %d classes ****\n"%K)
        np.random.seed(0)

        # Put the mean at 0, 1, ..., K-1. 
        means = np.arange(0, K)

        # Use the same diagonal covariance matrix for all classes.
        # The value of std will determine the difficulty of the problem.
        stds   = np.ones(K) * feat_std

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
    Draw data for K classes each with unidimensional Gaussian distribution.
    The means, stds and counts arguments are lists of size K
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


def create_scores_for_expts(num_classes, P0=0.9, P0m=0.9, feat_std=0.15, N=100000, score_scale_mc2=5, 
                            sim_name='gaussian_sim', calibrate=False, simple_names=False, nbins=15):

    """
    Generate a bunch of different posteriors for a K class problem (K can be changed to whatever you
    like). First, generate likelihoods with Gaussian class distributions and then compute:

    Datap: log-posteriors obtained from the llks applying the true data priors

    Mismp: log-posteriors obtained from the llks applying the mismatched data priors to simulate a
    system that was trained with the wrong priors

    Two llk versions are used (cal, mc1): miscalibrated and calibrated ones, resulting in two
    versions of each of the above posteriors. Finally, another miscalibrated version of the
    posteriors (mc2) is created by scaling the log-posteriors directly.

    If calibrate is True, for each of the 6 posteriors (Datap/Mismp-cal/mc1/mc2), calibrated
    versions, using an affine transformation, temp scaling, and histogram binning are also created.
    In each case, we train them either with cross-validation or by training on the test data.

    When newnames is set to True, use the simpler names in the latest papers rather than the names
    in the original one. 
    """ 


    K = num_classes

    # Prior vector with given above and all other priors being equal to (1-p0)/(K-1)
    data_priors = np.array([P0] + [(1 - P0) / (K - 1)] * (K - 1))

    # Mismatched priors where the p0 is used for the last class instead of the first
    mism_priors = np.array([(1 - P0m) / (K - 1)] * (K - 1) + [P0m])

    score_dict = {'cal': {}, 'mc1': {}, 'mc2': {}}

    # Parameters used for miscalibrating the true likelihoods to create the raw ones
    # The shift is set to 0 for all classes except the first one, and the scale is
    # set to 0.5
    shift_for_raw_llks = np.zeros(K)
    shift_for_raw_llks[0] = 0.5
    score_scale1 = 0.5
    sim_params = {
        'feat_std': feat_std,
        'score_scale': score_scale1,
        'score_shift': shift_for_raw_llks
        }

    # Generate the calibrated and the miscalibrated llks
    targets, score_dict['mc1']['llks'], score_dict['cal']['llks'] = get_llks_for_multi_classif_task(sim_name,
                                                                                                    priors=data_priors,
                                                                                                    sim_params=sim_params,
                                                                                                    N=N)
    
    for rc in ['cal', 'mc1', 'mc2']:

        if rc != 'mc2':
            # Take the cal or mc1 llks and compute two sets of posteriors with 
            # different priors
            llks = score_dict[rc]['llks']
            score_dict[rc]['Datap'] = utils.llks_to_logpost(llks, data_priors)
            score_dict[rc]['Mismp'] = utils.llks_to_logpost(llks, mism_priors)
        else:
            # For mc2, miscalibrate the cal posteriors by scaling them and renormalizing
            for pr in ['Datap', 'Mismp']:
                score_dict[rc][pr] = score_scale_mc2 * score_dict['cal'][pr]
                score_dict[rc][pr] -= logsumexp(score_dict[rc][pr], axis=1, keepdims=True)


        if calibrate:
            for pr in ['Datap', 'Mismp']:
                # Finally, create two sets of calibrated outputs for each set of posteriors.
                # Note that the output of this calibration method are posteriors for the priors
                # in the provided data. If you want to train calibration with a different set
                # of priors you can provide those priors through the "priors" argument.
                # If you want to get log-scaled-likelihood you just need to subtract log(priors)
                # from this method's output scores, where the priors are either those in the data
                # or the externally provided priors. 
                score_dict[rc][f'{pr}-affcal']   = calibration_with_crossval(score_dict[rc][pr], targets)
                score_dict[rc][f'{pr}-temcal']   = calibration_with_crossval(score_dict[rc][pr], targets, calparams={'bias':False})

                # Then repeat those three calibration procedures but training on test
                score_dict[rc][f'{pr}-affcaltt'] = calibration_train_on_test(score_dict[rc][pr], targets)
                score_dict[rc][f'{pr}-temcaltt'] = calibration_train_on_test(score_dict[rc][pr], targets, calparams={'bias':False})

                # For the binary case, add one calibrated version using histogram binning 
                if K == 2:
                    score_dict[rc][f'{pr}-hiscal']   = calibration_with_crossval(score_dict[rc][pr], targets, calmethod=HistogramBinningCal, calparams={'M':nbins})
                    score_dict[rc][f'{pr}-hiscaltt'] = calibration_train_on_test(score_dict[rc][pr], targets, calmethod=HistogramBinningCal, calparams={'M':nbins})


    if simple_names:
        # The stuff below creates a dict with those scores with the system names used in the latest
        # papers rather than the original ones. It discards mc1 and creates a dict with a single level.
        score_dict2 = {}
        for calmethod in ['', '-affcal', '-temcal', '-hiscal']:
            
            traintypes = ['xv', 'tt'] if calmethod != '' else ['']

            for traintype in traintypes:
                traintypet = traintype if traintype != 'xv' else ''
                caltypeo = calmethod+traintype
                caltypei = calmethod+traintypet
                if 'Datap'+caltypei in score_dict['cal']:
                    score_dict2['cal'+caltypeo]  = score_dict['cal']['Datap'+caltypei]
                    score_dict2['mcp'+caltypeo]  = score_dict['cal']['Mismp'+caltypei]
                    score_dict2['mcs'+caltypeo]  = score_dict['mc2']['Datap'+caltypei]
                    score_dict2['mcps'+caltypeo] = score_dict['mc2']['Mismp'+caltypei]        

        score_dict = score_dict2

    return score_dict, targets



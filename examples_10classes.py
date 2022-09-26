import numpy as np
from scipy.special import logsumexp 
import avcost
try:
    import psrcal
    import torch    
    from calibration import affine_calibration_with_crossval
    has_psr = True
except:
    has_psr = False

has_psr = False

adjusted_cost = True

# Load some scores from a resnet and targets for cifar10 data
print("\n**** Scoring in CIFAR10 data (10 classes) *****\n")
preacts = np.load("data/resnet-50_cifar10/predictions.npy")
targets = np.load("data/resnet-50_cifar10/targets.npy")

num_targets = np.max(targets)+1

# Compute log-softmax to get log posteriors. 
logpost = preacts - logsumexp(preacts, axis=-1, keepdims=True)

if has_psr:
    # Calibrate the scores with cross-validation using an affine transform
    # trained with log-loss (cross-entropy)
    logpostcal = affine_calibration_with_crossval(logpost, targets)
else:
    # If calibration code is not available, load a pre-generated file
    logpostcal = np.load("data/resnet-50_cifar10/predictions_cal_10classes.npy")


# Print the priors from the data and the priors that are
# derived from the posteriors from the model. 
priors_from_data = np.bincount(targets)/len(targets)
priors_from_post = np.mean(np.exp(logpost), axis=0)
print("Priors from data = %s"%" ".join(['{:.1f}'.format(i) for i in priors_from_data]))
print("Priors from post = %s"%" ".join(['{:.1f}'.format(i) for i in priors_from_post]))
print("Priors are consistent between the data and the posteriors. "+
    "Yet, as we will see, these posteriors are still not great for making Bayes decisions for all possible cost matrices because they are not well-calibrated.\n")

# Make maximum a posteriori decisions. 
map_decisions = np.argmax(logpost, axis=-1)
cost_01 = avcost.cost_matrix.zero_one_costs(num_targets)
ac_01 = avcost.average_cost(targets, map_decisions, cost_01, adjusted=adjusted_cost)

# Now make Bayes decisions based on the posteriors
ac_01_bayes,           _ , _ = avcost.average_cost_for_bayes_decisions(targets, logpost, cost_01, adjusted=adjusted_cost)
ac_01_bayes_after_cal, _ , _ = avcost.average_cost_for_bayes_decisions(targets, logpostcal, cost_01, adjusted=adjusted_cost)

print("*** Average cost for 0-1 cost matrix")
print("    Using MAP decisions:                        %.4f"%ac_01)
print("    Using Bayes decisions:                      %.4f"%ac_01_bayes)
print("    Using Bayes decisions on calibrated scores: %.4f"%ac_01_bayes_after_cal)
print("\nThese costs are the same because Bayes decisions for 0-1 cost coincide with MAP decisions and because the scores are well-calibrated for this specific cost function.\n")

# Try a cost with an abstention decision and evaluate the MAP decisions.
ab_cost = 0.2
cost_ab = avcost.cost_matrix.zero_one_costs(num_targets, abstention_cost=ab_cost)
ac_ab = avcost.average_cost(targets, map_decisions, cost_ab, adjusted=adjusted_cost)

# Now make Bayes decisions for that cost function
ac_ab_bayes, bayes_decisions_ab, _ = avcost.average_cost_for_bayes_decisions(targets, logpost, cost_ab, adjusted=adjusted_cost)
ac_ab_bayes_after_cal,       _ , _ = avcost.average_cost_for_bayes_decisions(targets, logpostcal, cost_ab, adjusted=adjusted_cost)

print("*** Average cost for cost matrix with abstention cost of %.1f"%ab_cost)
print("    Using MAP decisions:                        %.4f"%ac_ab)
print("    Using Bayes decisions:                      %.4f"%ac_ab_bayes)
print("    Using Bayes decisions on calibrated scores: %.4f"%ac_ab_bayes_after_cal)
print("\nThe first two costs are not the same because optimal (Bayes) decisions include %d abstentions on the hardest samples. "%len(np.where(bayes_decisions_ab!= map_decisions)[0]))
print("The fact that the last two costs are not the same means that the original scores were, in fact, not that well-calibrated since they were suboptimal for this cost function.\n")

# Non let's assume we expect that, on the actual data the system will be used for the priors
# are not uniform but rather one of the classes is much more frequent than the others:
nu_priors = np.array([0.91, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])

# Let's find the average cost for the 0-1 cost matrix when making Bayes
# decisions with the posteriors given by the system. The Bayes decisions will
# not change since the posteriors do not change, but the cost will. The
# third value returned by the method instead computes decisions on a new
# set of posteriors with priors matched to those provided as input.
ac_01_nup_bayes, _, ac_01_nup_bayes_with_matched_priors = avcost.average_cost_for_bayes_decisions(targets, logpost, cost_01, nu_priors, adjusted=adjusted_cost, silent=True)

print("*** Average cost for 0-1 cost matrix and non-uniform priors %s"%" ".join(['{:.2f}'.format(i) for i in nu_priors]))
print("    Using Bayes decisions with the original posteriors from the model:         %.4f"%ac_01_nup_bayes)
print("    Using Bayes decisions with the posteriors obtained with matched priors:    %.4f"%ac_01_nup_bayes_with_matched_priors)
print("\nThe second average cost is much lower because the original priors are quite bad for making decisions when priors are non-uniform, since the prior implicit in them is uniform.\n")






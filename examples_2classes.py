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

adjusted_cost = True

# Load some scores from a resnet and targets for cifar10 data
print("\n**** Scoring in CIFAR10 data (first two classes) *****\n")
preacts = np.load("data/resnet-50_cifar10/predictions.npy")
targets = np.load("data/resnet-50_cifar10/targets.npy")

# Keep only two of the scores and the samples from those two classes
# to fake a 2-class problem.
sel = targets <= 1
targets = targets[sel]
preacts = preacts[sel][:,:2]

num_targets = np.max(targets)+1

# Compute log-softmax to get log posteriors. 
logpost = preacts - logsumexp(preacts, axis=-1, keepdims=True)

if has_psr:
    # Calibrate the scores with cross-validation using an affine transform
    # trained with log-loss (cross-entropy)
    logpostcal = affine_calibration_with_crossval(logpost, targets)
else:
    # If calibration code is not available, load a pre-generated file
    logpostcal = np.load("data/resnet-50_cifar10/predictions_cal_first2classes.npy")


# Now, compute a family of cost matrices varying the cost for one of the classes
# in the log domain and leaving the other one fixed at 1.
# For each of these matrices compute the cost for maximum-a-posterior
# decisions and the cost for Bayes decisions.
# Also, compute the cost we would get by finding the  decision threshold empirically to 
# optimize the specific cost function. 
map_decisions = np.argmax(logpost, axis=-1)

print("*** Average cost for cost matrix = [[0 1] [alpha 0]]\n")

print("Alpha      MAP      Bayes   Bayes_after_cal  Optimal")

for logalpha in np.arange(-4, 4, 0.5):

    alpha = np.exp(logalpha)
    cost = avcost.cost_matrix([[0, 1],[alpha, 0]])  

    ac_map                   = avcost.average_cost(targets, map_decisions, cost, adjusted=adjusted_cost)
    ac_bayes,           _, _ = avcost.average_cost_for_bayes_decisions(targets, logpost, cost, adjusted=adjusted_cost)
    ac_min                   = avcost.average_cost_for_optimal_decisions(targets, logpost, cost, adjusted=adjusted_cost)
    ac_bayes_after_cal, _, _ = avcost.average_cost_for_bayes_decisions(targets, logpostcal, cost, adjusted=adjusted_cost)

    print("%6.3f   %6.3f    %6.3f    %6.3f     %6.3f"%(alpha, ac_map, ac_bayes, ac_bayes_after_cal, ac_min, ))

print("\nThe difference between MAP and Bayes shows the suboptimality of MAP decisions when the costs are not equal. "+
    "On the other hand, the difference between the Bayes and the Bayes_after_cal shows how well or badly calibrated the system scores are. "+
    "Larger differences indicate a more severe calibration problem at that specific operating point. "+
    "In this case, calibration is fine around the equal-cost point, but degrades significantly as the costs become more imbalanced."+
    "Finally, a difference between Bayes_after_cal and Optimal indicates that either the calibration did not work perfectly or "+
    "that the optimal cost is, in fact, too optimistic since the threshold is selected on the evaluation data itself.\n")


print("*** Average cost for cost matrix with an abstention option = [[0 1 alpha] [1 0 alpha]].\n")
print("Alpha      MAP      Bayes    Num_abstentions_with_Bayes   Bayes_after_cal  Num_abstentions_with_Bayes_after_cal")

for logalpha in np.arange(-4, 0.5, 0.5):

    alpha = np.exp(logalpha)
    cost = avcost.cost_matrix([[0, 1, alpha],[1, 0, alpha]])  

    ac_map                                     = avcost.average_cost(targets, map_decisions, cost, adjusted=adjusted_cost)
    ac_bayes,           decisions,           _ = avcost.average_cost_for_bayes_decisions(targets, logpost, cost, adjusted=adjusted_cost)
    ac_bayes_after_cal, decisions_after_cal, _ = avcost.average_cost_for_bayes_decisions(targets, logpostcal, cost, adjusted=adjusted_cost)
    print("%6.3f   %6.3f    %6.3f        %6d                  %6.3f         %6d"%(alpha, ac_map, ac_bayes, np.sum(decisions==2), ac_bayes_after_cal, np.sum(decisions_after_cal==2)))


print("\nThe lower the cost of abstention, the more samples get this label and the worse the MAP decisions are (which do not have the abstention option). "+
    "Further, the adjusted Bayes cost increases as alpha gets lower because the naïve system used for normalization (which would always choose to abstain) "+
    "is harder to beat (i.e., it has a lower cost). We can see that the cost is much lower after calibration (more abstentions are made), showing again that the "+
    "original scores were not well-calibrated across all possible operating points. Note that, in this case, optimal decisions cannot be made by sweeping a "+
    "threshold as for the square cost function above because there are three possible decisions.\n")



